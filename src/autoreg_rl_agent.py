from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .env import EvalResult, LLMUAVEnv, SimState


class AutoregPolicyNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        layer_feat_dim: int,
        num_uavs: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        state_layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(max(1, depth - 1)):
            state_layers.append(nn.Linear(in_dim, hidden_dim))
            state_layers.append(nn.LayerNorm(hidden_dim))
            state_layers.append(nn.SiLU())
            if dropout > 0:
                state_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.state_encoder = nn.Sequential(*state_layers)

        step_in = hidden_dim + layer_feat_dim + num_uavs + num_uavs + 2
        step_layers: list[nn.Module] = []
        in_dim = step_in
        for _ in range(depth):
            step_layers.append(nn.Linear(in_dim, hidden_dim))
            step_layers.append(nn.LayerNorm(hidden_dim))
            step_layers.append(nn.SiLU())
            if dropout > 0:
                step_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        step_layers.append(nn.Linear(in_dim, num_uavs))
        self.step_net = nn.Sequential(*step_layers)
        self.num_uavs = num_uavs

    def encode_state(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state_vec)

    def step_logits(
        self,
        state_h: torch.Tensor,
        layer_feat: torch.Tensor,
        prev_onehot: torch.Tensor,
        mem_ratio: torch.Tensor,
        layer_frac: torch.Tensor,
        remaining_frac: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state_h, layer_feat, prev_onehot, mem_ratio, layer_frac, remaining_frac], dim=-1)
        return self.step_net(x)


class AutoregReplayBuffer:
    def __init__(self, capacity: int):
        self.states: deque[np.ndarray] = deque(maxlen=capacity)
        self.mem_caps: deque[np.ndarray] = deque(maxlen=capacity)
        self.actions: deque[np.ndarray] = deque(maxlen=capacity)
        self.weights: deque[float] = deque(maxlen=capacity)

    def add(self, state_vec: np.ndarray, mem_caps: np.ndarray, action: np.ndarray, weight: float = 1.0) -> None:
        self.states.append(state_vec.astype(np.float32, copy=True))
        self.mem_caps.append(mem_caps.astype(np.float32, copy=True))
        self.actions.append(action.astype(np.int64, copy=True))
        self.weights.append(float(weight))

    def __len__(self) -> int:
        return len(self.states)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self.states), size=batch_size)
        states = np.stack([self.states[int(i)] for i in idx])
        caps = np.stack([self.mem_caps[int(i)] for i in idx])
        actions = np.stack([self.actions[int(i)] for i in idx])
        weights = np.asarray([self.weights[int(i)] for i in idx], dtype=np.float32)
        return states, caps, actions, weights


@dataclass
class AutoregSampleBatch:
    actions: np.ndarray
    log_probs: torch.Tensor
    entropy: torch.Tensor


@dataclass
class AutoregPolicyEval:
    action: np.ndarray
    result: EvalResult
    feasible_candidates: int


class AutoregRLAgent:
    def __init__(self, env: LLMUAVEnv, cfg: dict, device: str, rng: np.random.Generator):
        self.env = env
        self.cfg = cfg
        self.rl_cfg = cfg.get("ar_rl", {})
        self.rng = rng
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.layer_features_np = self._build_layer_features().astype(np.float32)
        self.layer_features = torch.from_numpy(self.layer_features_np).to(self.device)
        self.policy = AutoregPolicyNet(
            state_dim=env.state_dim,
            layer_feat_dim=self.layer_features_np.shape[1],
            num_uavs=env.num_uavs,
            hidden_dim=int(self.rl_cfg.get("hidden_dim", 512)),
            depth=int(self.rl_cfg.get("depth", 3)),
            dropout=float(self.rl_cfg.get("dropout", 0.03)),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=float(self.rl_cfg.get("learning_rate", 4e-4)),
            weight_decay=float(self.rl_cfg.get("weight_decay", 1e-4)),
        )
        self.replay = AutoregReplayBuffer(int(self.rl_cfg.get("replay_capacity", 100000)))

    def _build_layer_features(self) -> np.ndarray:
        p = self.env.profile
        mem = p.mem_bytes / max(float(np.mean(p.mem_bytes)), 1.0)
        cycles = p.compute_cycles / max(float(np.mean(p.compute_cycles)), 1.0)
        act_prev = np.zeros(p.num_layers, dtype=np.float64)
        act_next = np.zeros(p.num_layers, dtype=np.float64)
        imp_prev = np.zeros(p.num_layers, dtype=np.float64)
        act_prev[1:] = p.activation_bytes / max(float(np.mean(p.activation_bytes)), 1.0)
        act_next[:-1] = p.activation_bytes / max(float(np.mean(p.activation_bytes)), 1.0)
        imp_prev[1:] = p.importance / max(float(np.mean(p.importance)), 1e-12)
        pos = np.arange(p.num_layers, dtype=np.float64) / max(p.num_layers - 1, 1)
        return np.stack([mem, cycles, act_prev, act_next, imp_prev, pos], axis=1)

    def mem_caps_norm(self, state: SimState) -> np.ndarray:
        return (state.resources.mem_bytes / max(float(np.sum(self.env.profile.mem_bytes)), 1.0)).astype(np.float32)

    def _masked_dist(self, logits: torch.Tensor, mem_used: torch.Tensor, caps: torch.Tensor, layer: int, temperature: float):
        layer_mem = float(self.env.profile.mem_bytes[layer] / max(float(np.sum(self.env.profile.mem_bytes)), 1.0))
        allowed = mem_used + layer_mem <= caps + 1e-8
        no_allowed = ~torch.any(allowed, dim=-1, keepdim=True)
        allowed = torch.where(no_allowed, torch.ones_like(allowed), allowed)
        masked_logits = logits.masked_fill(~allowed, -1.0e9)
        return torch.distributions.Categorical(logits=masked_logits / max(temperature, 1e-6))

    def sample_batch(
        self,
        state_vecs_np: np.ndarray,
        mem_caps_np: np.ndarray,
        candidates: int,
        temperature: float,
    ) -> AutoregSampleBatch:
        self.policy.train()
        bsz = state_vecs_np.shape[0]
        n = self.env.num_uavs
        states = torch.from_numpy(state_vecs_np.astype(np.float32, copy=False)).to(self.device)
        caps = torch.from_numpy(mem_caps_np.astype(np.float32, copy=False)).to(self.device)
        state_h = self.policy.encode_state(states)
        state_h = state_h.unsqueeze(1).expand(-1, candidates, -1).reshape(bsz * candidates, -1)
        caps = caps.unsqueeze(1).expand(-1, candidates, -1).reshape(bsz * candidates, n)
        mem_used = torch.zeros(bsz * candidates, n, device=self.device)
        prev_onehot = torch.zeros(bsz * candidates, n, device=self.device)
        log_probs = torch.zeros(bsz * candidates, device=self.device)
        entropies = torch.zeros(bsz * candidates, device=self.device)
        actions = []

        for layer in range(self.env.num_layers):
            layer_feat = self.layer_features[layer].unsqueeze(0).expand(bsz * candidates, -1)
            layer_frac = torch.full((bsz * candidates, 1), layer / max(self.env.num_layers - 1, 1), device=self.device)
            remaining_frac = 1.0 - mem_used.sum(dim=-1, keepdim=True)
            logits = self.policy.step_logits(state_h, layer_feat, prev_onehot, mem_used, layer_frac, remaining_frac)
            dist = self._masked_dist(logits, mem_used, caps, layer, temperature)
            sampled = dist.sample()
            log_probs = log_probs + dist.log_prob(sampled)
            entropies = entropies + dist.entropy()
            actions.append(sampled)
            prev_onehot = F.one_hot(sampled, num_classes=n).float()
            layer_mem = float(self.env.profile.mem_bytes[layer] / max(float(np.sum(self.env.profile.mem_bytes)), 1.0))
            mem_used = mem_used + prev_onehot * layer_mem

        actions_t = torch.stack(actions, dim=1).view(bsz, candidates, self.env.num_layers)
        return AutoregSampleBatch(
            actions=actions_t.detach().cpu().numpy().astype(np.int64),
            log_probs=log_probs.view(bsz, candidates),
            entropy=entropies.view(bsz, candidates),
        )

    @torch.no_grad()
    def _policy_candidates_fast(
        self,
        state_vec_np: np.ndarray,
        mem_caps_np: np.ndarray,
        temperatures_np: np.ndarray,
        greedy_count: int,
    ) -> np.ndarray:
        self.policy.eval()
        n = self.env.num_uavs
        candidates = int(temperatures_np.shape[0])
        if candidates <= 0:
            return np.empty((0, self.env.num_layers), dtype=np.int64)

        states = torch.from_numpy(state_vec_np.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        caps = torch.from_numpy(mem_caps_np.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        temps = torch.from_numpy(temperatures_np.astype(np.float32, copy=False)).to(self.device).clamp_min(1e-6)
        greedy_mask = torch.zeros(candidates, dtype=torch.bool, device=self.device)
        if greedy_count > 0:
            greedy_mask[: min(greedy_count, candidates)] = True

        with torch.inference_mode():
            state_h = self.policy.encode_state(states).expand(candidates, -1)
            caps = caps.expand(candidates, -1)
            mem_used = torch.zeros(candidates, n, device=self.device)
            prev_onehot = torch.zeros(candidates, n, device=self.device)
            actions = []

            for layer in range(self.env.num_layers):
                layer_feat = self.layer_features[layer].unsqueeze(0).expand(candidates, -1)
                layer_frac = torch.full((candidates, 1), layer / max(self.env.num_layers - 1, 1), device=self.device)
                remaining_frac = 1.0 - mem_used.sum(dim=-1, keepdim=True)
                logits = self.policy.step_logits(state_h, layer_feat, prev_onehot, mem_used, layer_frac, remaining_frac)

                layer_mem = float(self.env.profile.mem_bytes[layer] / max(float(np.sum(self.env.profile.mem_bytes)), 1.0))
                allowed = mem_used + layer_mem <= caps + 1e-8
                no_allowed = ~torch.any(allowed, dim=-1, keepdim=True)
                allowed = torch.where(no_allowed, torch.ones_like(allowed), allowed)
                masked_logits = logits.masked_fill(~allowed, -1.0e9)

                greedy_choice = torch.argmax(masked_logits, dim=-1)
                scaled_logits = masked_logits / temps[:, None]
                probs = torch.softmax(scaled_logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                sampled = torch.where(greedy_mask, greedy_choice, sampled)

                actions.append(sampled)
                prev_onehot = F.one_hot(sampled, num_classes=n).float()
                mem_used = mem_used + prev_onehot * layer_mem

            actions_t = torch.stack(actions, dim=1)
            return actions_t.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def _policy_candidates_beam_fast(
        self,
        state_vec_np: np.ndarray,
        mem_caps_np: np.ndarray,
        beam_width: int,
        temperature: float,
    ) -> np.ndarray:
        self.policy.eval()
        n = self.env.num_uavs
        beam_cap = max(1, int(beam_width))
        state = torch.from_numpy(state_vec_np.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        caps = torch.from_numpy(mem_caps_np.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        temp = max(float(temperature), 1e-6)

        with torch.inference_mode():
            state_h_base = self.policy.encode_state(state).squeeze(0)
            state_h = state_h_base.unsqueeze(0)
            caps = caps.expand(1, -1)
            mem_used = torch.zeros(1, n, device=self.device)
            prev_onehot = torch.zeros(1, n, device=self.device)
            seqs = torch.empty((1, 0), dtype=torch.long, device=self.device)
            scores = torch.zeros(1, device=self.device)

            for layer in range(self.env.num_layers):
                beam_count = int(scores.shape[0])
                state_h = state_h_base.unsqueeze(0).expand(beam_count, -1)
                layer_feat = self.layer_features[layer].unsqueeze(0).expand(beam_count, -1)
                layer_frac = torch.full((beam_count, 1), layer / max(self.env.num_layers - 1, 1), device=self.device)
                remaining_frac = 1.0 - mem_used.sum(dim=-1, keepdim=True)
                logits = self.policy.step_logits(state_h, layer_feat, prev_onehot, mem_used, layer_frac, remaining_frac)

                layer_mem = float(self.env.profile.mem_bytes[layer] / max(float(np.sum(self.env.profile.mem_bytes)), 1.0))
                allowed = mem_used + layer_mem <= caps.expand(beam_count, -1) + 1e-8
                no_allowed = ~torch.any(allowed, dim=-1, keepdim=True)
                allowed = torch.where(no_allowed, torch.ones_like(allowed), allowed)
                masked_logits = logits.masked_fill(~allowed, -1.0e9)
                log_probs = torch.log_softmax(masked_logits / temp, dim=-1)

                expanded_scores = scores[:, None] + log_probs
                flat_scores = expanded_scores.reshape(-1)
                topk = min(beam_cap, int(flat_scores.shape[0]))
                top_scores, top_idx = torch.topk(flat_scores, k=topk)
                parent_idx = torch.div(top_idx, n, rounding_mode="floor")
                action_idx = top_idx % n

                if seqs.shape[1] == 0:
                    seqs = action_idx.unsqueeze(-1)
                else:
                    seqs = torch.cat([seqs[parent_idx], action_idx.unsqueeze(-1)], dim=1)
                scores = top_scores
                parent_mem = mem_used[parent_idx]
                parent_prev = prev_onehot[parent_idx]
                chosen_onehot = F.one_hot(action_idx, num_classes=n).float()
                mem_used = parent_mem + chosen_onehot * layer_mem
                prev_onehot = chosen_onehot

            return seqs.cpu().numpy().astype(np.int64)

    def _eval_temperature_schedule(self, candidates: int, default_temperature: float, greedy_count: int) -> np.ndarray:
        sample_count = max(0, candidates - greedy_count)
        temperatures = [1e-6] * min(greedy_count, candidates)
        if sample_count <= 0:
            return np.asarray(temperatures, dtype=np.float32)

        temps = self.rl_cfg.get("eval_temperatures", None)
        if temps:
            temp_values = [float(x) for x in temps]
            counts = [sample_count // len(temp_values)] * len(temp_values)
            for i in range(sample_count % len(temp_values)):
                counts[i] += 1
            for temp_value, count in zip(temp_values, counts):
                temperatures.extend([temp_value] * count)
        else:
            temperatures.extend([float(default_temperature)] * sample_count)
        return np.asarray(temperatures[:candidates], dtype=np.float32)

    def project_candidate_action(self, action: np.ndarray, state: SimState, max_passes: int = 2) -> np.ndarray:
        mode = str(self.rl_cfg.get("projection_mode", "action")).lower()
        if mode in {"block_fast", "blocks_fast", "project_blocks_fast"}:
            return self.env.project_blocks_fast(
                action,
                state,
                max_blocks=int(self.rl_cfg.get("max_blocks", self.env.num_uavs)),
            )
        if mode in {"block", "blocks", "project_blocks"}:
            return self.env.project_blocks(
                action,
                state,
                max_blocks=int(self.rl_cfg.get("max_blocks", self.env.num_uavs)),
            )
        return self.env.project_action(action, state, max_passes=max_passes)

    @torch.no_grad()
    def policy_candidates(
        self,
        state: SimState,
        candidates: int,
        temperature: float,
        greedy_count: int = 1,
    ) -> np.ndarray:
        self.policy.eval()
        state_vec = self.env.state_vector(state)
        caps = self.mem_caps_norm(state)
        mode = str(self.rl_cfg.get("candidate_mode", "sample")).lower()
        overgenerate = max(1, int(self.rl_cfg.get("candidate_overgenerate", 1)))
        raw_candidates = max(candidates, candidates * overgenerate)
        if mode == "beam":
            actions = self._policy_candidates_beam_fast(
                state_vec,
                caps,
                beam_width=raw_candidates,
                temperature=float(self.rl_cfg.get("beam_temperature", temperature)),
            )
        elif mode in {"beam_mix", "mixed_beam"}:
            temp_values = self.rl_cfg.get("eval_temperatures", None) or [float(temperature)]
            per_temp = max(1, raw_candidates // len(temp_values))
            parts = []
            for temp_value in temp_values:
                parts.append(
                    self._policy_candidates_beam_fast(
                        state_vec,
                        caps,
                        beam_width=per_temp,
                        temperature=float(temp_value),
                    )
                )
            actions = np.concatenate(parts, axis=0) if parts else np.empty((0, self.env.num_layers), dtype=np.int64)
            if actions.shape[0] < raw_candidates:
                pad = self._policy_candidates_fast(
                    state_vec,
                    caps,
                    self._eval_temperature_schedule(raw_candidates - actions.shape[0], temperature, 0),
                    greedy_count=0,
                )
                actions = np.concatenate([actions, pad], axis=0)
            if actions.shape[0] > raw_candidates:
                actions = actions[:raw_candidates]
        elif mode in {"beam_sample", "sample_beam"}:
            beam_count = int(self.rl_cfg.get("candidate_beam_count", max(1, int(round(0.75 * candidates)))))
            beam_count = min(max(1, beam_count), candidates)
            sample_count = max(0, candidates - beam_count)
            beam_actions = self._policy_candidates_beam_fast(
                state_vec,
                caps,
                beam_width=beam_count,
                temperature=float(self.rl_cfg.get("beam_temperature", temperature)),
            )
            if sample_count > 0:
                sample_raw = max(sample_count, sample_count * overgenerate)
                sample_actions = self._policy_candidates_fast(
                    state_vec,
                    caps,
                    self._eval_temperature_schedule(sample_raw, temperature, 0),
                    greedy_count=0,
                )
                actions = np.concatenate([beam_actions, sample_actions], axis=0)
            else:
                actions = beam_actions
        else:
            temperatures = self._eval_temperature_schedule(raw_candidates, temperature, greedy_count)
            actions = self._policy_candidates_fast(state_vec, caps, temperatures, greedy_count=greedy_count)
        repaired_all = [self.project_candidate_action(action, state, max_passes=2) for action in actions]
        unique: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        min_distance = int(self.rl_cfg.get("candidate_min_hamming", 0))
        for action in repaired_all:
            key = tuple(int(x) for x in action.tolist())
            if key in seen:
                continue
            if min_distance > 0 and any(int(np.sum(action != prev)) < min_distance for prev in unique):
                continue
            seen.add(key)
            unique.append(action)
            if len(unique) >= candidates:
                break
        if len(unique) < candidates and min_distance > 0:
            for action in repaired_all:
                key = tuple(int(x) for x in action.tolist())
                if key in seen:
                    continue
                seen.add(key)
                unique.append(action)
                if len(unique) >= candidates:
                    break
        if not unique:
            unique = repaired_all[:1]
        while len(unique) < candidates:
            unique.append(unique[-1].copy())
        return np.stack(unique[:candidates])

    def select_policy_candidate(self, state: SimState, candidates: int, temperature: float = 0.5) -> AutoregPolicyEval:
        actions = self.policy_candidates(
            state,
            candidates=candidates,
            temperature=temperature,
            greedy_count=int(self.rl_cfg.get("greedy_candidates", 1)),
        )
        refine_steps = int(self.rl_cfg.get("policy_refine_steps", 0))
        if refine_steps > 0:
            actions = self.policy_refine_candidates(state, actions, refine_steps)
        evals = self.env.evaluate_many(state, actions)
        feasible_idx = [i for i, ev in enumerate(evals) if ev.feasible]
        if feasible_idx:
            best_i = max(feasible_idx, key=lambda i: evals[i].reward)
        else:
            best_i = max(range(len(evals)), key=lambda i: evals[i].reward)
        return AutoregPolicyEval(actions[best_i], evals[best_i], len(feasible_idx))

    def policy_refine_candidates(self, state: SimState, actions: np.ndarray, steps: int) -> np.ndarray:
        refined: list[np.ndarray] = [np.asarray(action, dtype=np.int64).copy() for action in actions]
        evals = self.env.evaluate_many(state, np.asarray(refined, dtype=np.int64))
        best_i = max(range(len(evals)), key=lambda i: evals[i].reward)
        current = refined[best_i].copy()
        current_ev = evals[best_i]
        temp0 = float(self.rl_cfg.get("policy_refine_temp_start", 0.02))
        temp1 = float(self.rl_cfg.get("policy_refine_temp_end", 0.001))
        max_extra = int(self.rl_cfg.get("policy_refine_extra_candidates", steps))

        mem_used = self.env.memory_used(current)
        for step in range(max(0, steps)):
            frac = step / max(steps - 1, 1)
            temp = temp0 * ((temp1 / max(temp0, 1e-9)) ** frac)
            trial = current.copy()
            layer = int(self.rng.integers(0, self.env.num_layers))
            src = int(trial[layer])
            order = self.rng.permutation(self.env.num_uavs)
            moved = False
            for dst in order:
                dst = int(dst)
                if dst == src:
                    continue
                if mem_used[dst] + self.env.profile.mem_bytes[layer] > state.resources.mem_bytes[dst] + 1e-9:
                    continue
                trial[layer] = dst
                moved = True
                break
            if not moved:
                continue
            trial = self.project_candidate_action(trial, state, max_passes=1)
            ev = self.env.evaluate(state, trial)
            delta = ev.reward - current_ev.reward
            accept = delta >= 0.0 or self.rng.random() < float(np.exp(delta / max(temp, 1e-9)))
            if accept:
                current = trial
                current_ev = ev
                mem_used = self.env.memory_used(current)
            refined.append(trial)
            if len(refined) >= len(actions) + max_extra:
                break
        return np.asarray(refined, dtype=np.int64)

    def shaped_score(self, ev: EvalResult) -> float:
        if ev.feasible:
            return float(ev.reward)
        penalty = float(self.rl_cfg.get("infeasible_penalty", 2.0))
        cost_weight = float(self.rl_cfg.get("infeasible_cost_weight", 0.02))
        mem_excess = max(0.0, ev.max_mem_ratio - 1.0)
        energy_excess = max(0.0, ev.max_energy_ratio - 1.0)
        return -1.0 - penalty * (mem_excess + energy_excess) - cost_weight * min(float(ev.cost), 100.0)

    def supervised_loss(
        self,
        states_np: np.ndarray,
        caps_np: np.ndarray,
        actions_np: np.ndarray,
        weights_np: np.ndarray | None = None,
    ) -> torch.Tensor:
        bsz = states_np.shape[0]
        n = self.env.num_uavs
        states = torch.from_numpy(states_np.astype(np.float32, copy=False)).to(self.device)
        caps = torch.from_numpy(caps_np.astype(np.float32, copy=False)).to(self.device)
        targets = torch.from_numpy(actions_np.astype(np.int64, copy=False)).to(self.device)
        if weights_np is None:
            weights = torch.ones(bsz, device=self.device)
        else:
            weights = torch.from_numpy(weights_np.astype(np.float32, copy=False)).to(self.device)
        weights = weights / torch.clamp(weights.mean(), min=1e-6)
        state_h = self.policy.encode_state(states)
        mem_used = torch.zeros(bsz, n, device=self.device)
        prev_onehot = torch.zeros(bsz, n, device=self.device)
        losses = []
        for layer in range(self.env.num_layers):
            layer_feat = self.layer_features[layer].unsqueeze(0).expand(bsz, -1)
            layer_frac = torch.full((bsz, 1), layer / max(self.env.num_layers - 1, 1), device=self.device)
            remaining_frac = 1.0 - mem_used.sum(dim=-1, keepdim=True)
            logits = self.policy.step_logits(state_h, layer_feat, prev_onehot, mem_used, layer_frac, remaining_frac)
            dist = self._masked_dist(logits, mem_used, caps, layer, temperature=1.0)
            losses.append((-dist.log_prob(targets[:, layer]) * weights).mean())
            current = F.one_hot(targets[:, layer], num_classes=n).float()
            layer_mem = float(self.env.profile.mem_bytes[layer] / max(float(np.sum(self.env.profile.mem_bytes)), 1.0))
            mem_used = mem_used + current * layer_mem
            prev_onehot = current
        return torch.stack(losses).mean()

    def replay_update(self, batch_size: int, updates: int) -> dict[str, float] | None:
        if len(self.replay) < batch_size or updates <= 0:
            return None
        losses = []
        self.policy.train()
        for _ in range(updates):
            states_np, caps_np, actions_np, weights_np = self.replay.sample(batch_size, self.rng)
            loss = self.supervised_loss(states_np, caps_np, actions_np, weights_np)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.rl_cfg.get("grad_clip", 1.0)))
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return {"replay_loss": float(np.mean(losses))}
