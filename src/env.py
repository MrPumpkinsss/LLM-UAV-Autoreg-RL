from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import ChannelState, sample_channel
from .llm_profile import LLMProfile


@dataclass(frozen=True)
class UAVResources:
    compute_hz: np.ndarray
    mem_bytes: np.ndarray
    energy_j: np.ndarray
    hover_power_w: np.ndarray


@dataclass(frozen=True)
class SimState:
    channel: ChannelState
    resources: UAVResources
    previous_action: np.ndarray


@dataclass
class EvalResult:
    reward: float
    cost: float
    feasible: bool
    latency_s: float
    ppl_hat: float
    damage: float
    total_energy_j: float
    max_mem_ratio: float
    max_energy_ratio: float
    memory_ok: bool
    energy_ok: bool
    bandwidth_ok: bool


class LLMUAVEnv:
    def __init__(self, cfg: dict, profile: LLMProfile, rng: np.random.Generator):
        self.cfg = cfg
        self.profile = profile
        self.rng = rng
        self.num_uavs = int(cfg["uav"]["num_uavs"])
        self.num_layers = profile.num_layers
        self.state_dim = self.num_uavs * self.num_uavs * 2 + self.num_uavs * 3 + self.num_layers
        self.action_dim = self.num_layers
        self._last_action = np.arange(self.num_layers, dtype=np.int64) % self.num_uavs
        self._prefix_mem_bytes = np.concatenate([[0.0], np.cumsum(self.profile.mem_bytes)])

    def sample_resources(self) -> UAVResources:
        uav = self.cfg["uav"]
        n = self.num_uavs
        compute_ghz = self.rng.uniform(float(uav["compute_ghz_min"]), float(uav["compute_ghz_max"]), size=n)
        mem_mb = self.rng.uniform(float(uav["mem_mb_min"]), float(uav["mem_mb_max"]), size=n)
        energy_j = self.rng.uniform(float(uav["energy_j_min"]), float(uav["energy_j_max"]), size=n)
        hover_w = self.rng.uniform(float(uav["hover_power_w_min"]), float(uav["hover_power_w_max"]), size=n)
        return UAVResources(
            compute_hz=(compute_ghz * 1e9).astype(np.float64),
            mem_bytes=(mem_mb * 1024.0 * 1024.0).astype(np.float64),
            energy_j=energy_j.astype(np.float64),
            hover_power_w=hover_w.astype(np.float64),
        )

    def sample_state(self) -> SimState:
        channel = sample_channel({**self.cfg["wireless"], **self.cfg["uav"]}, self.num_uavs, self.rng)
        resources = self.sample_resources()
        state = SimState(channel=channel, resources=resources, previous_action=self._last_action.copy())
        return state

    def state_vector(self, state: SimState) -> np.ndarray:
        snr_norm = np.log1p(np.nan_to_num(state.channel.snr, posinf=1e6)) / np.log1p(1e6)
        pdp = state.channel.pdp
        compute_norm = state.resources.compute_hz / 1e10
        mem_norm = state.resources.mem_bytes / (512.0 * 1024.0 * 1024.0)
        energy_norm = state.resources.energy_j / 2000.0
        prev_norm = state.previous_action.astype(np.float64) / max(self.num_uavs - 1, 1)
        vec = np.concatenate(
            [
                snr_norm.reshape(-1),
                pdp.reshape(-1),
                compute_norm,
                mem_norm,
                energy_norm,
                prev_norm,
            ]
        )
        return vec.astype(np.float32)

    def repair_action(self, action: np.ndarray, state: SimState, repair_energy: bool = False) -> np.ndarray:
        repaired = np.asarray(action, dtype=np.int64).copy()
        repaired = np.clip(repaired, 0, self.num_uavs - 1)

        mem_used = self.memory_used(repaired)
        caps = state.resources.mem_bytes
        # Move layers out of overfull UAVs, preferring moves with low local cost.
        for _ in range(self.num_layers * self.num_uavs):
            over = np.where(mem_used > caps)[0]
            if over.size == 0:
                break
            src = int(over[np.argmax(mem_used[over] / caps[over])])
            layer_ids = np.where(repaired == src)[0]
            if layer_ids.size == 0:
                break
            # Large layers relieve memory pressure fastest.
            layer_ids = layer_ids[np.argsort(self.profile.mem_bytes[layer_ids])[::-1]]
            moved = False
            for layer in layer_ids:
                layer_mem = self.profile.mem_bytes[layer]
                dst_order = np.argsort(mem_used / caps)
                for dst in dst_order:
                    dst = int(dst)
                    if dst == src:
                        continue
                    if mem_used[dst] + layer_mem <= caps[dst]:
                        repaired[layer] = dst
                        mem_used[src] -= layer_mem
                        mem_used[dst] += layer_mem
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
        if repair_energy:
            repaired = self._repair_energy(repaired, state)
        return repaired

    def project_action(self, action: np.ndarray, state: SimState, max_passes: int = 3) -> np.ndarray:
        """Projection used by pure policies: no heuristic seeds, only local repair."""

        repaired = self.repair_action(action, state, repair_energy=False)
        current = self.evaluate(state, repaired)
        if current.feasible:
            return repaired

        for _ in range(max_passes):
            best_action = repaired
            best_eval = current
            layer_order = np.argsort(-self.profile.compute_cycles)
            mem_used = self.memory_used(repaired)
            for layer in layer_order:
                src = int(repaired[layer])
                for dst in range(self.num_uavs):
                    if dst == src:
                        continue
                    if mem_used[dst] + self.profile.mem_bytes[layer] > state.resources.mem_bytes[dst] + 1e-9:
                        continue
                    trial = repaired.copy()
                    trial[layer] = dst
                    ev = self.evaluate(state, trial)
                    better = (
                        ev.feasible and not best_eval.feasible
                        or ev.max_energy_ratio < best_eval.max_energy_ratio - 1e-8
                        or (
                            abs(ev.max_energy_ratio - best_eval.max_energy_ratio) <= 1e-8
                            and ev.max_mem_ratio < best_eval.max_mem_ratio - 1e-8
                        )
                        or (
                            abs(ev.max_energy_ratio - best_eval.max_energy_ratio) <= 1e-8
                            and abs(ev.max_mem_ratio - best_eval.max_mem_ratio) <= 1e-8
                            and ev.cost < best_eval.cost
                        )
                    )
                    if better:
                        best_action = trial
                        best_eval = ev
            if np.array_equal(best_action, repaired):
                break
            repaired = best_action
            current = best_eval
            if current.feasible:
                break
        return repaired

    def project_blocks(self, action: np.ndarray, state: SimState, max_blocks: int | None = None) -> np.ndarray:
        """Convert a raw layer assignment into feasible-ish contiguous layer blocks."""

        raw = np.asarray(action, dtype=np.int64)
        if max_blocks is None:
            max_blocks = self.num_uavs
        blocks: list[tuple[int, int, int]] = []
        start = 0
        current = int(raw[0])
        for layer in range(1, self.num_layers):
            nxt = int(raw[layer])
            if nxt != current and len(blocks) < max_blocks - 1:
                blocks.append((start, layer, current))
                start = layer
                current = nxt
        blocks.append((start, self.num_layers, current))

        projected = np.empty(self.num_layers, dtype=np.int64)
        mem_used = np.zeros(self.num_uavs, dtype=np.float64)
        for start, end, preferred in blocks:
            block_mem = float(np.sum(self.profile.mem_bytes[start:end]))
            order = [preferred] + [int(i) for i in np.argsort(mem_used / np.maximum(state.resources.mem_bytes, 1.0)) if int(i) != preferred]
            chosen = order[-1]
            for uav in order:
                if mem_used[uav] + block_mem <= state.resources.mem_bytes[uav] + 1e-9:
                    chosen = int(uav)
                    break
            projected[start:end] = chosen
            mem_used[chosen] += block_mem

        return self.project_action(projected, state, max_passes=1)

    def project_blocks_fast(self, action: np.ndarray, state: SimState, max_blocks: int | None = None) -> np.ndarray:
        """Fast contiguous block projection for inference candidate generation."""

        raw = np.asarray(action, dtype=np.int64)
        if max_blocks is None:
            max_blocks = self.num_uavs
        blocks: list[tuple[int, int, int]] = []
        start = 0
        current = int(raw[0])
        for layer in range(1, self.num_layers):
            nxt = int(raw[layer])
            if nxt != current and len(blocks) < max_blocks - 1:
                blocks.append((start, layer, current))
                start = layer
                current = nxt
        blocks.append((start, self.num_layers, current))

        projected = np.empty(self.num_layers, dtype=np.int64)
        mem_used = np.zeros(self.num_uavs, dtype=np.float64)
        caps = state.resources.mem_bytes
        for start, end, preferred in blocks:
            block_mem = float(self._prefix_mem_bytes[end] - self._prefix_mem_bytes[start])
            order = [preferred] + [
                int(i)
                for i in np.argsort(mem_used / np.maximum(caps, 1.0))
                if int(i) != preferred
            ]
            chosen = order[-1]
            for uav in order:
                if mem_used[uav] + block_mem <= caps[uav] + 1e-9:
                    chosen = int(uav)
                    break
            projected[start:end] = chosen
            mem_used[chosen] += block_mem

        return projected

    def _repair_energy(self, action: np.ndarray, state: SimState) -> np.ndarray:
        repaired = action.copy()
        max_iters = int(self.cfg["reward"].get("repair_energy_iters", 0))
        if max_iters <= 0:
            return repaired

        current = self.evaluate(state, repaired)
        if current.energy_ok:
            return repaired

        for _ in range(max_iters):
            mem_used = self.memory_used(repaired)
            best_action = None
            best_eval = current
            energy_by_layer_proxy = self.profile.compute_cycles / state.resources.compute_hz[repaired]
            layer_order = np.argsort(-energy_by_layer_proxy)

            for layer in layer_order:
                src = int(repaired[layer])
                for dst in range(self.num_uavs):
                    if dst == src:
                        continue
                    layer_mem = self.profile.mem_bytes[layer]
                    if mem_used[dst] + layer_mem > state.resources.mem_bytes[dst] + 1e-9:
                        continue
                    trial = repaired.copy()
                    trial[layer] = dst
                    ev = self.evaluate(state, trial)
                    better = (
                        ev.feasible and not best_eval.feasible
                        or ev.max_energy_ratio < best_eval.max_energy_ratio - 1e-6
                        or (
                            abs(ev.max_energy_ratio - best_eval.max_energy_ratio) <= 1e-6
                            and ev.cost < best_eval.cost
                        )
                    )
                    if better:
                        best_action = trial
                        best_eval = ev

            if best_action is None or best_eval.max_energy_ratio >= current.max_energy_ratio - 1e-6:
                break
            repaired = best_action
            current = best_eval
            if current.energy_ok:
                break
        return repaired

    def memory_used(self, action: np.ndarray) -> np.ndarray:
        return np.bincount(action, weights=self.profile.mem_bytes, minlength=self.num_uavs).astype(np.float64)

    def _attempts_and_residual(self, p: float) -> tuple[float, float]:
        r = self.cfg["wireless"]["retransmissions"]
        if isinstance(r, str) and r.lower() in {"inf", "infinity"}:
            return 1.0 / max(1.0 - p, 1e-6), 0.0
        r_int = int(r)
        if r_int < 0:
            return 1.0 / max(1.0 - p, 1e-6), 0.0
        attempts = (1.0 - p ** (r_int + 1)) / max(1.0 - p, 1e-8)
        residual = p ** (r_int + 1)
        return attempts, residual

    def _allocate_bandwidths(self, state: SimState, transitions: list[tuple[int, int, int, float]]) -> np.ndarray:
        if not transitions:
            return np.zeros(0, dtype=np.float64)

        coeffs = []
        eps = 1e-12
        for layer, src, dst, p in transitions:
            snr = max(float(state.channel.snr[src, dst]), 0.0)
            spectral_eff = max(float(np.log2(1.0 + snr)), eps)
            attempts, _ = self._attempts_and_residual(p)
            coeff = float(self.profile.activation_bytes[layer]) * 8.0 * attempts / spectral_eff
            coeffs.append(coeff)

        coeffs_arr = np.asarray(coeffs, dtype=np.float64)
        sqrt_coeffs = np.sqrt(np.maximum(coeffs_arr, eps))
        total = float(np.sum(sqrt_coeffs))
        if total <= eps:
            return np.full(len(transitions), float(self.cfg["wireless"]["bandwidth_hz"]) / max(len(transitions), 1), dtype=np.float64)
        return float(self.cfg["wireless"]["bandwidth_hz"]) * sqrt_coeffs / total

    def evaluate(self, state: SimState, action: np.ndarray) -> EvalResult:
        action = np.asarray(action, dtype=np.int64)
        wireless = self.cfg["wireless"]
        reward_cfg = self.cfg["reward"]
        uav_cfg = self.cfg["uav"]

        mem_used = self.memory_used(action)
        memory_ok = bool(np.all(mem_used <= state.resources.mem_bytes + 1e-9))
        max_mem_ratio = float(np.max(mem_used / np.maximum(state.resources.mem_bytes, 1.0)))

        compute_latency = self.profile.compute_cycles / state.resources.compute_hz[action]
        total_compute_latency = float(np.sum(compute_latency))

        compute_ghz = state.resources.compute_hz / 1e9
        compute_power = float(uav_cfg["compute_power_base_w"]) + float(uav_cfg["compute_power_per_ghz_w"]) * compute_ghz
        compute_energy_per_layer = compute_latency * compute_power[action]
        energy_by_uav = np.bincount(action, weights=compute_energy_per_layer, minlength=self.num_uavs).astype(np.float64)

        transitions = []
        weights = []
        for layer in range(self.num_layers - 1):
            src = int(action[layer])
            dst = int(action[layer + 1])
            if src != dst:
                p = float(state.channel.pdp[src, dst])
                weights.append(float(self.profile.activation_bytes[layer]) * (1.0 + p))
                transitions.append((layer, src, dst, p))

        total_comm_latency = 0.0
        damage = 0.0
        if transitions:
            bandwidths = self._allocate_bandwidths(state, transitions)
            for idx, (layer, src, dst, p) in enumerate(transitions):
                snr = max(float(state.channel.snr[src, dst]), 0.0)
                spectral_eff = max(float(np.log2(1.0 + snr)), 1e-12)
                rate = float(bandwidths[idx]) * spectral_eff
                base_latency = float(self.profile.activation_bytes[layer]) * 8.0 / rate
                attempts, residual = self._attempts_and_residual(p)
                comm_latency = base_latency * attempts
                total_comm_latency += comm_latency
                energy_by_uav[src] += float(uav_cfg["tx_power_w"]) * comm_latency
                damage += float(self.profile.importance[layer]) * residual

        latency = total_compute_latency + total_comm_latency
        energy_by_uav += state.resources.hover_power_w * latency
        total_energy = float(np.sum(energy_by_uav))
        energy_ok = bool(np.all(energy_by_uav <= state.resources.energy_j + 1e-9))
        max_energy_ratio = float(np.max(energy_by_uav / np.maximum(state.resources.energy_j, 1.0)))
        bandwidth_ok = True

        ppl_hat = float(self.profile.ppl_ref * np.exp(self.profile.ppl_gamma * damage))
        ppl_norm = (ppl_hat - self.profile.ppl_ref) / max(self.profile.ppl_ref, 1e-9)
        latency_norm = latency / float(reward_cfg["latency_ref_s"])
        cost = float(reward_cfg["alpha"]) * ppl_norm + float(reward_cfg["beta"]) * latency_norm
        feasible = memory_ok and energy_ok and bandwidth_ok
        reward = -cost if feasible else float(reward_cfg["infeasible_reward"])

        return EvalResult(
            reward=float(reward),
            cost=float(cost),
            feasible=feasible,
            latency_s=float(latency),
            ppl_hat=ppl_hat,
            damage=float(damage),
            total_energy_j=total_energy,
            max_mem_ratio=max_mem_ratio,
            max_energy_ratio=max_energy_ratio,
            memory_ok=memory_ok,
            energy_ok=energy_ok,
            bandwidth_ok=bandwidth_ok,
        )

    def evaluate_many(self, state: SimState, actions: np.ndarray) -> list[EvalResult]:
        return [self.evaluate(state, action) for action in actions]

    def set_last_action(self, action: np.ndarray) -> None:
        self._last_action = np.asarray(action, dtype=np.int64).copy()
