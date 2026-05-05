from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .autoreg_rl_agent import AutoregRLAgent
from .baselines import evaluate_full_benchmark, evaluate_full_benchmark_timed
from .config import ensure_dir, load_config
from .env import LLMUAVEnv
from .llm_profile import build_qwen3_0p6b_profile, build_qwen3_0p6b_real_profile, ppl_hat_from_residuals_torch


@dataclass(frozen=True)
class HardStateSpec:
    seed: int
    state_id: int
    margin: float
    weight: float


@dataclass(frozen=True)
class HardStateItem:
    state: object
    seed: int
    state_id: int
    margin: float
    weight: float


def _torch_attempts_and_residual(env: LLMUAVEnv, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    retransmissions = env.cfg["wireless"]["retransmissions"]
    if isinstance(retransmissions, str) and retransmissions.lower() in {"inf", "infinity"}:
        return 1.0 / torch.clamp(1.0 - p, min=1e-6), torch.zeros_like(p)

    r_int = int(retransmissions)
    if r_int < 0:
        return 1.0 / torch.clamp(1.0 - p, min=1e-6), torch.zeros_like(p)
    attempts = (1.0 - p.pow(r_int + 1)) / torch.clamp(1.0 - p, min=1e-8)
    residual = p.pow(r_int + 1)
    return attempts, residual


def evaluate_training_batch_torch(
    agent: AutoregRLAgent,
    states: list,
    actions_np: np.ndarray,
) -> dict[str, torch.Tensor]:
    """GPU batched version of the environment score used during RL updates."""

    env = agent.env
    device = agent.device
    actions = torch.from_numpy(actions_np.astype(np.int64, copy=False)).to(device)
    bsz, candidates, layers = actions.shape
    flat_actions = actions.reshape(bsz * candidates, layers)
    flat_count = flat_actions.shape[0]
    n = env.num_uavs

    compute_hz = torch.as_tensor(np.stack([s.resources.compute_hz for s in states]), dtype=torch.float32, device=device)
    mem_caps = torch.as_tensor(np.stack([s.resources.mem_bytes for s in states]), dtype=torch.float32, device=device)
    energy_caps = torch.as_tensor(np.stack([s.resources.energy_j for s in states]), dtype=torch.float32, device=device)
    hover_w = torch.as_tensor(np.stack([s.resources.hover_power_w for s in states]), dtype=torch.float32, device=device)
    snr = torch.as_tensor(np.stack([s.channel.snr for s in states]), dtype=torch.float32, device=device)
    pdp = torch.as_tensor(np.stack([s.channel.pdp for s in states]), dtype=torch.float32, device=device)

    compute_hz = compute_hz[:, None, :].expand(bsz, candidates, n).reshape(flat_count, n)
    mem_caps = mem_caps[:, None, :].expand(bsz, candidates, n).reshape(flat_count, n)
    energy_caps = energy_caps[:, None, :].expand(bsz, candidates, n).reshape(flat_count, n)
    hover_w = hover_w[:, None, :].expand(bsz, candidates, n).reshape(flat_count, n)
    snr = snr[:, None, :, :].expand(bsz, candidates, n, n).reshape(flat_count, n, n)
    pdp = pdp[:, None, :, :].expand(bsz, candidates, n, n).reshape(flat_count, n, n)

    mem_bytes = torch.as_tensor(env.profile.mem_bytes, dtype=torch.float32, device=device)
    compute_cycles = torch.as_tensor(env.profile.compute_cycles, dtype=torch.float32, device=device)
    activation_bytes = torch.as_tensor(env.profile.activation_bytes, dtype=torch.float32, device=device)
    importance = torch.as_tensor(env.profile.importance, dtype=torch.float32, device=device)

    mem_by_uav = torch.zeros(flat_count, n, dtype=torch.float32, device=device)
    mem_by_uav.scatter_add_(1, flat_actions, mem_bytes[None, :].expand(flat_count, layers))
    memory_ok = torch.all(mem_by_uav <= mem_caps + 1e-6, dim=1)
    max_mem_ratio = torch.max(mem_by_uav / torch.clamp(mem_caps, min=1.0), dim=1).values

    selected_compute_hz = torch.gather(compute_hz, 1, flat_actions)
    compute_latency = compute_cycles[None, :] / torch.clamp(selected_compute_hz, min=1.0)
    total_compute_latency = torch.sum(compute_latency, dim=1)

    uav_cfg = env.cfg["uav"]
    compute_power = float(uav_cfg["compute_power_base_w"]) + float(uav_cfg["compute_power_per_ghz_w"]) * (compute_hz / 1e9)
    selected_power = torch.gather(compute_power, 1, flat_actions)
    compute_energy_per_layer = compute_latency * selected_power
    energy_by_uav = torch.zeros(flat_count, n, dtype=torch.float32, device=device)
    energy_by_uav.scatter_add_(1, flat_actions, compute_energy_per_layer)

    src = flat_actions[:, :-1]
    dst = flat_actions[:, 1:]
    transition_mask = src.ne(dst)
    row = torch.arange(flat_count, device=device)[:, None]
    p = pdp[row, src, dst].clamp(0.0, 0.999999)
    link_snr = torch.clamp(snr[row, src, dst], min=0.0)
    spectral_eff = torch.clamp(torch.log2(1.0 + link_snr), min=1e-12)
    attempts, residual = _torch_attempts_and_residual(env, p)

    coeff = activation_bytes[None, :] * 8.0 * attempts / spectral_eff
    sqrt_coeff = torch.where(transition_mask, torch.sqrt(torch.clamp(coeff, min=1e-12)), torch.zeros_like(coeff))
    sqrt_total = torch.sum(sqrt_coeff, dim=1, keepdim=True)
    bandwidth = torch.where(
        sqrt_total > 1e-12,
        float(env.cfg["wireless"]["bandwidth_hz"]) * sqrt_coeff / torch.clamp(sqrt_total, min=1e-12),
        torch.zeros_like(sqrt_coeff),
    )
    rate = torch.clamp(bandwidth * spectral_eff, min=1e-12)
    comm_latency = torch.where(
        transition_mask,
        activation_bytes[None, :] * 8.0 * attempts / rate,
        torch.zeros_like(rate),
    )
    total_comm_latency = torch.sum(comm_latency, dim=1)
    residual_by_layer = torch.where(transition_mask, residual, torch.zeros_like(residual))
    damage = torch.sum(importance[None, :] * residual_by_layer, dim=1)

    tx_energy = float(uav_cfg["tx_power_w"]) * comm_latency
    energy_by_uav.scatter_add_(1, src, tx_energy)
    latency = total_compute_latency + total_comm_latency
    energy_by_uav = energy_by_uav + hover_w * latency[:, None]
    total_energy = torch.sum(energy_by_uav, dim=1)
    energy_ok = torch.all(energy_by_uav <= energy_caps + 1e-6, dim=1)
    max_energy_ratio = torch.max(energy_by_uav / torch.clamp(energy_caps, min=1.0), dim=1).values

    reward_cfg = env.cfg["reward"]
    ppl_hat = ppl_hat_from_residuals_torch(env.profile, residual_by_layer)
    ppl_norm = (ppl_hat - float(env.profile.ppl_ref)) / max(float(env.profile.ppl_ref), 1e-9)
    latency_norm = latency / float(reward_cfg["latency_ref_s"])
    cost = float(reward_cfg["alpha"]) * ppl_norm + float(reward_cfg["beta"]) * latency_norm
    feasible = memory_ok & energy_ok
    reward = torch.where(
        feasible,
        -cost,
        torch.full_like(cost, float(reward_cfg["infeasible_reward"])),
    )

    penalty = float(agent.rl_cfg.get("infeasible_penalty", 2.0))
    cost_weight = float(agent.rl_cfg.get("infeasible_cost_weight", 0.02))
    mem_excess = torch.clamp(max_mem_ratio - 1.0, min=0.0)
    energy_excess = torch.clamp(max_energy_ratio - 1.0, min=0.0)
    shaped = torch.where(
        feasible,
        reward,
        -1.0 - penalty * (mem_excess + energy_excess) - cost_weight * torch.clamp(cost, max=100.0),
    )

    shape = (bsz, candidates)
    return {
        "shaped": shaped.reshape(shape),
        "reward": reward.reshape(shape),
        "cost": cost.reshape(shape),
        "feasible": feasible.float().reshape(shape),
        "latency_s": latency.reshape(shape),
        "ppl_hat": ppl_hat.reshape(shape),
        "total_energy_j": total_energy.reshape(shape),
        "max_mem_ratio": max_mem_ratio.reshape(shape),
        "max_energy_ratio": max_energy_ratio.reshape(shape),
    }


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _teacher_weighted_items(
    agent: AutoregRLAgent,
    env: LLMUAVEnv,
    state,
    methods: dict,
    top_k: int,
    tau: float,
    min_margin: float,
) -> list[tuple[np.ndarray, float, float]]:
    seen: set[tuple[int, ...]] = set()
    candidates: list[tuple[np.ndarray, float]] = []
    for action, ev in methods.values():
        projected = agent.project_candidate_action(action, state, max_passes=2)
        projected_ev = env.evaluate(state, projected)
        if not projected_ev.feasible:
            continue
        key = tuple(int(x) for x in projected.tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append((projected, float(projected_ev.reward)))

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[1], reverse=True)
    best_reward = candidates[0][1]
    candidates = [item for item in candidates if item[1] >= best_reward - max(0.0, min_margin)]
    candidates = candidates[: max(1, top_k)]
    rewards = np.asarray([reward for _action, reward in candidates], dtype=np.float64)
    scaled = (rewards - float(np.max(rewards))) / max(float(tau), 1e-6)
    weights = np.exp(np.clip(scaled, -60.0, 0.0))
    weights = weights / max(float(np.mean(weights)), 1e-9)
    return [(action, reward, float(weight)) for (action, reward), weight in zip(candidates, weights)]


def save_teacher_cache(path: Path, states: list[np.ndarray], caps: list[np.ndarray], actions: list[np.ndarray], weights: list[float], rewards: list[float]) -> None:
    if not actions:
        return
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        states=np.stack(states).astype(np.float32),
        caps=np.stack(caps).astype(np.float32),
        actions=np.stack(actions).astype(np.int64),
        weights=np.asarray(weights, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
    )


def load_teacher_cache(agent: AutoregRLAgent, path: Path, limit: int | None = None) -> int:
    data = np.load(path)
    states = data["states"]
    caps = data["caps"]
    actions = data["actions"]
    weights = data["weights"] if "weights" in data.files else np.ones(actions.shape[0], dtype=np.float32)
    count = actions.shape[0] if limit is None else min(int(limit), actions.shape[0])
    for idx in range(count):
        agent.replay.add(states[idx], caps[idx], actions[idx], weight=float(weights[idx]))
    return count


def load_hard_state_specs(
    rows_path: Path,
    *,
    method: str,
    baseline: str,
    margin_threshold: float,
    max_states: int | None,
    weight_scale: float,
    min_weight: float,
    max_weight: float,
) -> list[HardStateSpec]:
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    by_state: dict[tuple[int, int], dict[str, float]] = {}
    with rows_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["seed"]), int(row["state_id"]))
            by_state.setdefault(key, {})[str(row["method"])] = float(row["reward"])

    specs: list[HardStateSpec] = []
    for (seed, state_id), rewards in by_state.items():
        if method not in rewards:
            continue
        if baseline == "best_nonrl":
            other_rewards = [value for name, value in rewards.items() if name != method]
            if not other_rewards:
                continue
            baseline_reward = max(other_rewards)
        else:
            if baseline not in rewards:
                continue
            baseline_reward = rewards[baseline]
        margin = rewards[method] - baseline_reward
        if margin < margin_threshold:
            loss = max(0.0, -margin)
            weight = min_weight + weight_scale * loss
            weight = float(np.clip(weight, min_weight, max_weight))
            specs.append(HardStateSpec(seed=seed, state_id=state_id, margin=float(margin), weight=weight))

    specs.sort(key=lambda item: item.margin)
    if max_states is not None and max_states > 0:
        specs = specs[: int(max_states)]
    return specs


def materialize_hard_states(
    specs: list[HardStateSpec],
    cfg: dict,
    real_dir: Path | None,
    beam_width: int,
    anneal_steps: int,
) -> list[HardStateItem]:
    if not specs:
        return []

    by_seed: dict[int, list[HardStateSpec]] = {}
    for spec in specs:
        by_seed.setdefault(spec.seed, []).append(spec)

    hard_states: list[HardStateItem] = []
    for seed, seed_specs in sorted(by_seed.items()):
        seed_specs = sorted(seed_specs, key=lambda item: item.state_id)
        rng = np.random.default_rng(seed)
        if real_dir is not None:
            profile = build_qwen3_0p6b_real_profile(cfg["profile"], real_dir, rng)
        else:
            profile = build_qwen3_0p6b_profile(cfg["profile"], rng)
        env = LLMUAVEnv(cfg, profile, rng)
        next_index = 0
        spec_index = 0
        max_state_id = seed_specs[-1].state_id
        for state_id in range(max_state_id + 1):
            state = env.sample_state()
            # The benchmark consumes the same RNG inside the heuristic suite
            # before moving to the next state. Replaying that consumption keeps
            # seed/state_id hard cases aligned with benchmark_rows.csv.
            evaluate_full_benchmark_timed(
                env,
                state,
                rng,
                beam_width=beam_width,
                anneal_steps=anneal_steps,
            )
            while spec_index < len(seed_specs) and seed_specs[spec_index].state_id == state_id:
                spec = seed_specs[spec_index]
                hard_states.append(
                    HardStateItem(
                        state=state,
                        seed=seed,
                        state_id=state_id,
                        margin=spec.margin,
                        weight=spec.weight,
                    )
                )
                spec_index += 1
            next_index = state_id + 1
        if spec_index != len(seed_specs):
            raise RuntimeError(f"Failed to materialize all hard states for seed={seed}; stopped at state_id={next_index}")
    return hard_states


def sample_training_states(
    env: LLMUAVEnv,
    rng: np.random.Generator,
    batch_states: int,
    hard_states: list[HardStateItem],
    hard_fraction: float,
) -> tuple[list, np.ndarray]:
    hard_count = 0
    if hard_states and hard_fraction > 0.0:
        hard_count = int(round(batch_states * hard_fraction))
        hard_count = min(max(1, hard_count), batch_states)
    random_count = batch_states - hard_count
    states = [env.sample_state() for _ in range(random_count)]
    weights = [1.0] * random_count
    if hard_count > 0:
        probs = np.asarray([item.weight for item in hard_states], dtype=np.float64)
        probs = probs / max(float(np.sum(probs)), 1e-12)
        picked = rng.choice(len(hard_states), size=hard_count, replace=True, p=probs)
        for idx in picked:
            item = hard_states[int(idx)]
            states.append(item.state)
            weights.append(item.weight)
    return states, np.asarray(weights, dtype=np.float32)


def evaluate_agent(agent: AutoregRLAgent, env: LLMUAVEnv, states: list, candidates: int, temperature: float) -> dict[str, float]:
    rewards = []
    costs = []
    latencies = []
    ppls = []
    feasible = []
    feasible_candidates = []
    for state in states:
        selected = agent.select_policy_candidate(state, candidates=candidates, temperature=temperature)
        ev = selected.result
        rewards.append(ev.reward)
        costs.append(ev.cost)
        latencies.append(ev.latency_s)
        ppls.append(ev.ppl_hat)
        feasible.append(float(ev.feasible))
        feasible_candidates.append(selected.feasible_candidates)
    return {
        "reward": float(np.mean(rewards)),
        "cost": float(np.mean(costs)),
        "latency_s": float(np.mean(latencies)),
        "ppl_hat": float(np.mean(ppls)),
        "feasible_rate": float(np.mean(feasible)),
        "feasible_candidates": float(np.mean(feasible_candidates)),
    }


def collect_teacher(
    agent: AutoregRLAgent,
    env: LLMUAVEnv,
    rng: np.random.Generator,
    states: int,
    beam_width: int,
    anneal_steps: int,
    top_k: int = 4,
    tau: float = 0.02,
    cache_path: Path | None = None,
) -> list[float]:
    rewards = []
    cache_states: list[np.ndarray] = []
    cache_caps: list[np.ndarray] = []
    cache_actions: list[np.ndarray] = []
    cache_weights: list[float] = []
    cache_rewards: list[float] = []
    for idx in range(states):
        state = env.sample_state()
        methods = evaluate_full_benchmark(env, state, rng, beam_width=beam_width, anneal_steps=anneal_steps)
        state_vec = env.state_vector(state)
        caps = agent.mem_caps_norm(state)
        weighted_items = _teacher_weighted_items(
            agent,
            env,
            state,
            methods,
            top_k=top_k,
            tau=tau,
            min_margin=float(agent.rl_cfg.get("teacher_min_margin", 0.05)),
        )
        if weighted_items:
            rewards.append(weighted_items[0][1])
        for action, reward, weight in weighted_items:
            agent.replay.add(state_vec, caps, action, weight=weight)
            cache_states.append(state_vec)
            cache_caps.append(caps)
            cache_actions.append(action)
            cache_weights.append(weight)
            cache_rewards.append(reward)
        if (idx + 1) % 500 == 0:
            print(f"teacher_collected={idx + 1}/{states} reward_mean={np.mean(rewards):.4f}", flush=True)
            if cache_path is not None:
                save_teacher_cache(cache_path, cache_states, cache_caps, cache_actions, cache_weights, cache_rewards)
    if cache_path is not None:
        save_teacher_cache(cache_path, cache_states, cache_caps, cache_actions, cache_weights, cache_rewards)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_calibrated.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch-states", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--validation-states", type=int, default=None)
    parser.add_argument("--eval-candidates", type=int, default=None)
    parser.add_argument("--teacher-states", type=int, default=None)
    parser.add_argument("--teacher-updates", type=int, default=None)
    parser.add_argument("--num-uavs", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--init-policy", default=None)
    parser.add_argument("--train-projection-passes", type=int, default=None)
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--load-teacher-cache", default=None)
    parser.add_argument("--teacher-cache-limit", type=int, default=None)
    parser.add_argument("--teacher-top-k", type=int, default=None)
    parser.add_argument("--teacher-weight-tau", type=float, default=None)
    parser.add_argument("--teacher-min-margin", type=float, default=None)
    parser.add_argument("--projection-mode", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--candidate-mode", default=None)
    parser.add_argument("--beam-temperature", type=float, default=None)
    parser.add_argument("--candidate-overgenerate", type=int, default=None)
    parser.add_argument("--candidate-beam-count", type=int, default=None)
    parser.add_argument("--candidate-min-hamming", type=int, default=None)
    parser.add_argument("--hard-benchmark-rows", default=None)
    parser.add_argument("--hard-baseline", default="best_nonrl")
    parser.add_argument("--hard-margin-threshold", type=float, default=None)
    parser.add_argument("--hard-max-states", type=int, default=None)
    parser.add_argument("--hard-fraction", type=float, default=None)
    parser.add_argument("--hard-weight-scale", type=float, default=None)
    parser.add_argument("--hard-min-weight", type=float, default=None)
    parser.add_argument("--hard-max-weight", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["uav"]["num_uavs"] = args.num_uavs
    if args.num_layers is not None:
        cfg["profile"]["num_layers"] = args.num_layers
    real_dir = Path(args.real_dir) if args.real_dir else None
    if real_dir is not None:
        cfg.setdefault("profile", {})["real_profile_dir"] = real_dir.as_posix()
    ar_cfg = cfg.setdefault("ar_rl", {})
    if args.episodes is not None:
        ar_cfg["episodes"] = args.episodes
    if args.batch_states is not None:
        ar_cfg["batch_states"] = args.batch_states
    if args.candidates is not None:
        ar_cfg["candidates"] = args.candidates
    if args.validation_states is not None:
        ar_cfg["validation_states"] = args.validation_states
    if args.eval_candidates is not None:
        ar_cfg["eval_candidates"] = args.eval_candidates
    if args.teacher_states is not None:
        ar_cfg["teacher_states"] = args.teacher_states
    if args.teacher_updates is not None:
        ar_cfg["teacher_updates"] = args.teacher_updates
    if args.device is not None:
        ar_cfg["device"] = args.device
    if args.train_projection_passes is not None:
        ar_cfg["train_projection_passes"] = args.train_projection_passes
    if args.teacher_top_k is not None:
        ar_cfg["teacher_top_k"] = args.teacher_top_k
    if args.teacher_weight_tau is not None:
        ar_cfg["teacher_weight_tau"] = args.teacher_weight_tau
    if args.teacher_min_margin is not None:
        ar_cfg["teacher_min_margin"] = args.teacher_min_margin
    if args.projection_mode is not None:
        ar_cfg["projection_mode"] = args.projection_mode
    if args.max_blocks is not None:
        ar_cfg["max_blocks"] = args.max_blocks
    if args.candidate_mode is not None:
        ar_cfg["candidate_mode"] = args.candidate_mode
    if args.beam_temperature is not None:
        ar_cfg["beam_temperature"] = args.beam_temperature
    if args.candidate_overgenerate is not None:
        ar_cfg["candidate_overgenerate"] = args.candidate_overgenerate
    if args.candidate_beam_count is not None:
        ar_cfg["candidate_beam_count"] = args.candidate_beam_count
    if args.candidate_min_hamming is not None:
        ar_cfg["candidate_min_hamming"] = args.candidate_min_hamming
    if args.hard_benchmark_rows is not None:
        ar_cfg["hard_benchmark_rows"] = args.hard_benchmark_rows
    if args.hard_baseline is not None:
        ar_cfg["hard_baseline"] = args.hard_baseline
    if args.hard_margin_threshold is not None:
        ar_cfg["hard_margin_threshold"] = args.hard_margin_threshold
    if args.hard_max_states is not None:
        ar_cfg["hard_max_states"] = args.hard_max_states
    if args.hard_fraction is not None:
        ar_cfg["hard_fraction"] = args.hard_fraction
    if args.hard_weight_scale is not None:
        ar_cfg["hard_weight_scale"] = args.hard_weight_scale
    if args.hard_min_weight is not None:
        ar_cfg["hard_min_weight"] = args.hard_min_weight
    if args.hard_max_weight is not None:
        ar_cfg["hard_max_weight"] = args.hard_max_weight

    seed = int(cfg["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(args.out or ar_cfg.get("result_dir", "results/autoreg_rl"))
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if real_dir is not None:
        profile = build_qwen3_0p6b_real_profile(cfg["profile"], real_dir, rng)
    else:
        profile = build_qwen3_0p6b_profile(cfg["profile"], rng)
    env = LLMUAVEnv(cfg, profile, rng)
    agent = AutoregRLAgent(env, cfg, str(ar_cfg.get("device", "cuda")), rng)
    if args.init_policy:
        state_dict = torch.load(Path(args.init_policy), map_location=agent.device)
        agent.policy.load_state_dict(state_dict)
        print(f"loaded_init_policy={args.init_policy}", flush=True)
    if args.load_teacher_cache:
        loaded = load_teacher_cache(agent, Path(args.load_teacher_cache), args.teacher_cache_limit)
        print(f"loaded_teacher_cache={args.load_teacher_cache} samples={loaded}", flush=True)

    episodes = int(ar_cfg.get("episodes", 1000))
    batch_states = int(ar_cfg.get("batch_states", 8))
    candidates = int(ar_cfg.get("candidates", 64))
    elite_fraction = float(ar_cfg.get("elite_fraction", 0.08))
    elite_count = max(1, int(round(candidates * elite_fraction)))
    batch_size = int(ar_cfg.get("batch_size", 512))
    replay_updates = int(ar_cfg.get("replay_updates_per_step", 1))
    eval_interval = int(ar_cfg.get("eval_interval", 50))
    eval_candidates = int(ar_cfg.get("eval_candidates", candidates))
    validation_states = [env.sample_state() for _ in range(int(ar_cfg.get("validation_states", 96)))]
    train_projection_passes = int(ar_cfg.get("train_projection_passes", 0))
    temp_start = float(ar_cfg.get("temperature_start", 1.2))
    temp_end = float(ar_cfg.get("temperature_end", 0.35))
    entropy_start = float(ar_cfg.get("entropy_coef_start", 0.02))
    entropy_end = float(ar_cfg.get("entropy_coef_end", 0.002))
    advantage_scale = float(ar_cfg.get("advantage_scale", 4.0))
    reward_clip = float(ar_cfg.get("reward_clip", 8.0))
    hard_states: list[HardStateItem] = []
    hard_specs: list[HardStateSpec] = []
    hard_rows_path = ar_cfg.get("hard_benchmark_rows")
    if hard_rows_path:
        hard_specs = load_hard_state_specs(
            Path(str(hard_rows_path)),
            method=str(ar_cfg.get("hard_method", "autoreg_rl_pure")),
            baseline=str(ar_cfg.get("hard_baseline", "best_nonrl")),
            margin_threshold=float(ar_cfg.get("hard_margin_threshold", -1e-9)),
            max_states=ar_cfg.get("hard_max_states"),
            weight_scale=float(ar_cfg.get("hard_weight_scale", 24.0)),
            min_weight=float(ar_cfg.get("hard_min_weight", 1.0)),
            max_weight=float(ar_cfg.get("hard_max_weight", 16.0)),
        )
        hard_states = materialize_hard_states(
            hard_specs,
            cfg,
            real_dir,
            beam_width=int(ar_cfg.get("hard_beam_width", ar_cfg.get("teacher_beam_width", 32))),
            anneal_steps=int(ar_cfg.get("hard_anneal_steps", ar_cfg.get("teacher_anneal_steps", 128))),
        )
        hard_report_rows = [
            {
                "seed": item.seed,
                "state_id": item.state_id,
                "margin": item.margin,
                "weight": item.weight,
            }
            for item in hard_states
        ]
        write_csv(out_dir / "hard_states_used.csv", hard_report_rows)
        if hard_states:
            print(
                f"loaded_hard_states={len(hard_states)} "
                f"margin_mean={np.mean([item.margin for item in hard_states]):.6f} "
                f"margin_min={np.min([item.margin for item in hard_states]):.6f}",
                flush=True,
            )
    best_eval_reward = -float("inf")
    started = time.time()

    teacher_states = int(ar_cfg.get("teacher_states", 0))
    teacher_updates = int(ar_cfg.get("teacher_updates", 0))
    teacher_rewards: list[float] = []
    if teacher_states > 0 and teacher_updates > 0:
        print(f"teacher_pretrain states={teacher_states} updates={teacher_updates}", flush=True)
        teacher_rewards = collect_teacher(
            agent,
            env,
            rng,
            states=teacher_states,
            beam_width=int(ar_cfg.get("teacher_beam_width", 32)),
            anneal_steps=int(ar_cfg.get("teacher_anneal_steps", 128)),
            top_k=int(ar_cfg.get("teacher_top_k", 4)),
            tau=float(ar_cfg.get("teacher_weight_tau", 0.02)),
            cache_path=Path(args.teacher_cache) if args.teacher_cache else None,
        )
        teacher_batch_size = max(
            1,
            min(
                int(ar_cfg.get("teacher_batch_size", batch_size)),
                batch_size,
                len(agent.replay),
            ),
        )
        for update in range(1, teacher_updates + 1):
            stats = agent.replay_update(batch_size=teacher_batch_size, updates=1)
            if update % int(ar_cfg.get("teacher_eval_interval", 200)) == 0 or update == teacher_updates:
                ev = evaluate_agent(
                    agent,
                    env,
                    validation_states,
                    candidates=eval_candidates,
                    temperature=float(ar_cfg.get("eval_temperature", 0.35)),
                )
                if ev["reward"] > best_eval_reward:
                    best_eval_reward = ev["reward"]
                    torch.save(agent.policy.state_dict(), out_dir / "autoreg_policy_best.pt")
                print(
                    f"teacher_update={update} eval_reward={ev['reward']:.4f} "
                    f"eval_feas={ev['feasible_rate']:.3f} batch={teacher_batch_size} "
                    f"loss={(stats or {}).get('replay_loss', float('nan')):.4f} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    print(
        f"autoreg_rl device={agent.device} episodes={episodes} batch_states={batch_states} candidates={candidates}",
        flush=True,
    )
    rows: list[dict] = []
    eval_rows: list[dict] = []
    for episode in range(1, episodes + 1):
        frac = (episode - 1) / max(episodes - 1, 1)
        temperature = temp_start * ((temp_end / temp_start) ** frac)
        entropy_coef = entropy_start * ((entropy_end / max(entropy_start, 1e-9)) ** frac)
        states, state_weights_np = sample_training_states(
            env,
            rng,
            batch_states,
            hard_states,
            hard_fraction=float(ar_cfg.get("hard_fraction", 0.0)),
        )
        state_vecs = np.stack([env.state_vector(state) for state in states])
        caps = np.stack([agent.mem_caps_norm(state) for state in states])
        batch = agent.sample_batch(state_vecs, caps, candidates=candidates, temperature=temperature)

        if train_projection_passes > 0:
            train_actions = np.empty_like(batch.actions)
            for sid, state in enumerate(states):
                train_actions[sid] = np.stack(
                    [agent.project_candidate_action(action, state, max_passes=train_projection_passes) for action in batch.actions[sid]]
                )
        else:
            train_actions = batch.actions

        metrics = evaluate_training_batch_torch(agent, states, train_actions)
        rewards_t = metrics["shaped"]
        feasible_t = metrics["feasible"]
        ranked_t = torch.argsort(rewards_t, dim=1, descending=True)
        best_idx_t = ranked_t[:, 0]
        row_idx_t = torch.arange(batch_states, device=agent.device)

        reward_matrix = rewards_t.detach().cpu().numpy()
        feasible_matrix = feasible_t.detach().cpu().numpy()
        best_rewards_t = metrics["reward"][row_idx_t, best_idx_t]
        best_costs_t = metrics["cost"][row_idx_t, best_idx_t]
        best_latencies_t = metrics["latency_s"][row_idx_t, best_idx_t]
        best_ppls_t = metrics["ppl_hat"][row_idx_t, best_idx_t]
        best_feasible_t = metrics["feasible"][row_idx_t, best_idx_t]

        ranked_np = ranked_t[:, :elite_count].detach().cpu().numpy()
        for sid in range(batch_states):
            for elite_i in ranked_np[sid]:
                agent.replay.add(state_vecs[sid], caps[sid], train_actions[sid, int(elite_i)], weight=float(state_weights_np[sid]))

        baseline = rewards_t.mean(dim=1, keepdim=True)
        std = rewards_t.std(dim=1, keepdim=True).clamp_min(1e-3)
        advantages = (advantage_scale * (rewards_t - baseline) / std).clamp(-reward_clip, reward_clip)
        state_weights_t = torch.from_numpy(state_weights_np).to(agent.device).view(batch_states, 1)
        state_weights_t = state_weights_t / torch.clamp(state_weights_t.mean(), min=1e-6)
        policy_loss = -((batch.log_probs * advantages.detach()) * state_weights_t).mean()
        entropy_loss = -entropy_coef * (batch.entropy * state_weights_t).mean()
        loss = policy_loss + entropy_loss
        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), float(ar_cfg.get("grad_clip", 1.0)))
        agent.optimizer.step()
        replay_stats = agent.replay_update(batch_size=batch_size, updates=replay_updates)

        rows.append(
            {
                "episode": episode,
                "reward": float(best_rewards_t.mean().detach().cpu()),
                "cost": float(best_costs_t.mean().detach().cpu()),
                "feasible": float(best_feasible_t.mean().detach().cpu()),
                "latency_s": float(best_latencies_t.mean().detach().cpu()),
                "ppl_hat": float(best_ppls_t.mean().detach().cpu()),
                "candidate_reward": float(np.mean(reward_matrix)),
                "candidate_feasible": float(np.mean(feasible_matrix)),
                "temperature": temperature,
                "entropy_coef": entropy_coef,
                "hard_batch_fraction": float(np.mean(state_weights_np > 1.0)),
                "state_weight_mean": float(np.mean(state_weights_np)),
                "state_weight_max": float(np.max(state_weights_np)),
                "policy_loss": float(policy_loss.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "replay_loss": replay_stats["replay_loss"] if replay_stats else np.nan,
                "buffer": len(agent.replay),
            }
        )

        if episode == 1 or episode % eval_interval == 0:
            ev = evaluate_agent(
                agent,
                env,
                validation_states,
                candidates=eval_candidates,
                temperature=float(ar_cfg.get("eval_temperature", 0.35)),
            )
            ev["episode"] = episode
            eval_rows.append(ev)
            if ev["reward"] > best_eval_reward:
                best_eval_reward = ev["reward"]
                torch.save(agent.policy.state_dict(), out_dir / "autoreg_policy_best.pt")
            print(
                f"episode={episode:04d} train_reward={float(best_rewards_t.mean().detach().cpu()):.4f} "
                f"eval_reward={ev['reward']:.4f} eval_feas={ev['feasible_rate']:.3f} "
                f"eval_latency={ev['latency_s']:.3f}s cand_feas={np.mean(feasible_matrix):.3f} "
                f"loss={float(loss.detach().cpu()):.4f} buffer={len(agent.replay)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    write_csv(out_dir / "train_log.csv", rows)
    write_csv(out_dir / "eval_log.csv", eval_rows)
    torch.save(agent.policy.state_dict(), out_dir / "autoreg_policy.pt")
    if not (out_dir / "autoreg_policy_best.pt").exists():
        torch.save(agent.policy.state_dict(), out_dir / "autoreg_policy_best.pt")
    summary = {
        "episodes": episodes,
        "runtime_s": time.time() - started,
        "device": str(agent.device),
        "teacher_reward_mean": float(np.mean(teacher_rewards)) if teacher_rewards else None,
        "hard_states": len(hard_states),
        "hard_margin_mean": float(np.mean([item.margin for item in hard_states])) if hard_states else None,
        "hard_margin_min": float(np.min([item.margin for item in hard_states])) if hard_states else None,
        "hard_fraction": float(ar_cfg.get("hard_fraction", 0.0)),
        "final_eval": eval_rows[-1] if eval_rows else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
