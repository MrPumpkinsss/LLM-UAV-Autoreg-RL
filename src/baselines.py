from __future__ import annotations

import itertools
import math

import numpy as np

from .env import EvalResult, LLMUAVEnv, SimState


def random_feasible(env: LLMUAVEnv, state: SimState, rng: np.random.Generator, tries: int = 256) -> np.ndarray:
    best = None
    best_eval = None
    for _ in range(tries):
        action = rng.integers(0, env.num_uavs, size=env.num_layers, dtype=np.int64)
        action = env.repair_action(action, state)
        ev = env.evaluate(state, action)
        if ev.feasible:
            return action
        if best is None or ev.max_mem_ratio + ev.max_energy_ratio < best_eval.max_mem_ratio + best_eval.max_energy_ratio:
            best = action
            best_eval = ev
    return best if best is not None else np.arange(env.num_layers, dtype=np.int64) % env.num_uavs


def latency_greedy(env: LLMUAVEnv, state: SimState) -> np.ndarray:
    action = np.full(env.num_layers, -1, dtype=np.int64)
    mem_used = np.zeros(env.num_uavs, dtype=np.float64)
    compute_order = np.argsort(-state.resources.compute_hz)
    for layer in range(env.num_layers):
        placed = False
        for uav in compute_order:
            uav = int(uav)
            if mem_used[uav] + env.profile.mem_bytes[layer] <= state.resources.mem_bytes[uav]:
                action[layer] = uav
                mem_used[uav] += env.profile.mem_bytes[layer]
                placed = True
                break
        if not placed:
            action[layer] = int(np.argmin(mem_used / np.maximum(state.resources.mem_bytes, 1.0)))
            mem_used[action[layer]] += env.profile.mem_bytes[layer]
    return env.repair_action(action, state)


def block_balanced(env: LLMUAVEnv, state: SimState) -> np.ndarray:
    order = np.argsort(-state.resources.compute_hz)
    capacities = state.resources.mem_bytes[order]
    total_cap = np.sum(capacities)
    target_layers = np.maximum(1, np.round(env.num_layers * capacities / max(total_cap, 1.0)).astype(int))
    while np.sum(target_layers) > env.num_layers:
        target_layers[np.argmax(target_layers)] -= 1
    while np.sum(target_layers) < env.num_layers:
        target_layers[np.argmax(capacities)] += 1
    action = np.empty(env.num_layers, dtype=np.int64)
    start = 0
    for idx, count in enumerate(target_layers):
        end = min(env.num_layers, start + int(count))
        action[start:end] = int(order[idx])
        start = end
    if start < env.num_layers:
        action[start:] = int(order[-1])
    return env.repair_action(action, state)


def pdp_aware_greedy(env: LLMUAVEnv, state: SimState) -> np.ndarray:
    action = np.full(env.num_layers, -1, dtype=np.int64)
    mem_used = np.zeros(env.num_uavs, dtype=np.float64)
    for layer in range(env.num_layers):
        scores = []
        for uav in range(env.num_uavs):
            if mem_used[uav] + env.profile.mem_bytes[layer] > state.resources.mem_bytes[uav]:
                scores.append(np.inf)
                continue
            compute = env.profile.compute_cycles[layer] / state.resources.compute_hz[uav]
            link_cost = 0.0
            if layer > 0 and action[layer - 1] != uav:
                prev = int(action[layer - 1])
                p = float(state.channel.pdp[prev, uav])
                snr = max(float(state.channel.snr[prev, uav]), 0.0)
                rate = float(env.cfg["wireless"]["bandwidth_hz"]) * np.log2(1.0 + snr)
                rate = max(rate, float(env.cfg["wireless"]["min_rate_bps"]))
                attempts, residual = env._attempts_and_residual(p)
                link_cost = env.profile.activation_bytes[layer - 1] * 8.0 / rate * attempts
                link_cost += env.cfg["reward"]["latency_ref_s"] * env.profile.importance[layer - 1] * residual
            scores.append(compute + link_cost)
        if np.all(~np.isfinite(scores)):
            uav = int(np.argmin(mem_used / np.maximum(state.resources.mem_bytes, 1.0)))
        else:
            uav = int(np.argmin(scores))
        action[layer] = uav
        mem_used[uav] += env.profile.mem_bytes[layer]
    return env.repair_action(action, state)


def evaluate_baselines(env: LLMUAVEnv, state: SimState, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, EvalResult]]:
    actions = {
        "random": random_feasible(env, state, rng),
        "latency_greedy": latency_greedy(env, state),
        "block_balanced": block_balanced(env, state),
        "pdp_aware_greedy": pdp_aware_greedy(env, state),
    }
    return {name: (action, env.evaluate(state, action)) for name, action in actions.items()}


def local_search(
    env: LLMUAVEnv,
    state: SimState,
    rng: np.random.Generator,
    initial_actions: list[np.ndarray] | None = None,
    max_passes: int = 2,
    random_seed_tries: int = 32,
) -> tuple[np.ndarray, EvalResult]:
    """Strong one-layer-move local search under hard feasibility constraints."""

    seeds: list[np.ndarray] = []
    if initial_actions:
        seeds.extend(initial_actions)
    seeds.extend(
        [
            pdp_aware_greedy(env, state),
            latency_greedy(env, state),
            block_balanced(env, state),
            random_feasible(env, state, rng, tries=random_seed_tries),
        ]
    )

    best_action: np.ndarray | None = None
    best_eval: EvalResult | None = None
    for seed in seeds:
        action = env.repair_action(seed, state)
        current = env.evaluate(state, action)
        if current.feasible and (best_eval is None or current.reward > best_eval.reward):
            best_action = action.copy()
            best_eval = current
        elif best_eval is None:
            best_action = action.copy()
            best_eval = current

    assert best_action is not None and best_eval is not None

    current_action = best_action.copy()
    current_eval = best_eval
    for _ in range(max_passes):
        improved = False
        layer_order = rng.permutation(env.num_layers)
        for layer in layer_order:
            src = int(current_action[layer])
            move_order = rng.permutation(env.num_uavs)
            for dst in move_order:
                dst = int(dst)
                if dst == src:
                    continue
                trial = current_action.copy()
                trial[layer] = dst
                trial = env.repair_action(trial, state)
                ev = env.evaluate(state, trial)
                if not ev.feasible:
                    continue
                if ev.reward > current_eval.reward + 1e-9:
                    current_action = trial
                    current_eval = ev
                    improved = True
        if not improved:
            break

    return current_action, current_eval


def evaluate_strong_baselines(env: LLMUAVEnv, state: SimState, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, EvalResult]]:
    base = evaluate_baselines(env, state, rng)
    passes = int(env.cfg.get("heuristics", {}).get("local_search_passes", 2))
    tries = int(env.cfg.get("heuristics", {}).get("local_search_seed_tries", 64))
    ls_action, ls_eval = local_search(
        env,
        state,
        rng,
        initial_actions=[a for a, _ in base.values()],
        max_passes=passes,
        random_seed_tries=tries,
    )
    base["local_search"] = (ls_action, ls_eval)
    return base


def beam_search(env: LLMUAVEnv, state: SimState, beam_width: int = 24) -> tuple[np.ndarray, EvalResult]:
    """Layer-wise beam search with exact environment scoring at completed actions."""

    beams: list[tuple[np.ndarray, float]] = [(np.empty(0, dtype=np.int64), 0.0)]
    mem_caps = state.resources.mem_bytes
    for layer in range(env.num_layers):
        next_beams: list[tuple[np.ndarray, float]] = []
        for prefix, _score in beams:
            mem_used = np.bincount(prefix, weights=env.profile.mem_bytes[: len(prefix)], minlength=env.num_uavs)
            for uav in range(env.num_uavs):
                if mem_used[uav] + env.profile.mem_bytes[layer] > mem_caps[uav] + 1e-9:
                    continue
                new_prefix = np.append(prefix, uav).astype(np.int64)
                compute = float(np.sum(env.profile.compute_cycles[: layer + 1] / state.resources.compute_hz[new_prefix]))
                link_cost = 0.0
                for l in range(layer):
                    src = int(new_prefix[l])
                    dst = int(new_prefix[l + 1])
                    if src == dst:
                        continue
                    p = float(state.channel.pdp[src, dst])
                    attempts, residual = env._attempts_and_residual(p)
                    snr = max(float(state.channel.snr[src, dst]), 0.0)
                    spectral_eff = max(float(np.log2(1.0 + snr)), 1e-12)
                    link_cost += float(env.profile.activation_bytes[l]) * 8.0 * attempts / (
                        float(env.cfg["wireless"]["bandwidth_hz"]) * spectral_eff
                    )
                    link_cost += float(env.cfg["reward"]["latency_ref_s"]) * float(env.profile.importance[l]) * residual
                next_beams.append((new_prefix, -(compute + link_cost)))
        if not next_beams:
            fallback = block_balanced(env, state)
            return fallback, env.evaluate(state, fallback)
        next_beams.sort(key=lambda x: x[1], reverse=True)
        beams = next_beams[:beam_width]

    best_action = None
    best_eval = None
    for action, _ in beams:
        action = env.repair_action(action, state)
        ev = env.evaluate(state, action)
        if ev.feasible and (best_eval is None or ev.reward > best_eval.reward):
            best_action = action
            best_eval = ev
        elif best_eval is None:
            best_action = action
            best_eval = ev
    assert best_action is not None and best_eval is not None
    return best_action, best_eval


def _action_blocks(action: np.ndarray) -> list[tuple[int, int, int]]:
    blocks: list[tuple[int, int, int]] = []
    start = 0
    current = int(action[0])
    for layer in range(1, len(action)):
        nxt = int(action[layer])
        if nxt != current:
            blocks.append((start, layer, current))
            start = layer
            current = nxt
    blocks.append((start, len(action), current))
    return blocks


def _materialize_blocks(blocks: list[tuple[int, int, int]], num_layers: int) -> np.ndarray:
    action = np.empty(num_layers, dtype=np.int64)
    for start, end, uav in blocks:
        action[start:end] = int(uav)
    return action


def _cutset_from_action(action: np.ndarray) -> tuple[int, ...]:
    return tuple(end for _start, end, _uav in _action_blocks(action)[:-1])


def _valid_cutset(cuts: tuple[int, ...], num_layers: int, max_blocks: int) -> tuple[int, ...] | None:
    cleaned = tuple(sorted({int(c) for c in cuts if 0 < int(c) < num_layers}))
    if len(cleaned) > max_blocks - 1:
        return None
    if any(a >= b for a, b in zip(cleaned, cleaned[1:])):
        return None
    return cleaned


def _block_proxy_score(
    env: LLMUAVEnv,
    state: SimState,
    start: int,
    end: int,
    uav: int,
    prev_uav: int | None,
) -> float:
    reward_cfg = env.cfg["reward"]
    compute = float(np.sum(env.profile.compute_cycles[start:end] / state.resources.compute_hz[uav]))
    score = float(reward_cfg["beta"]) * compute / float(reward_cfg["latency_ref_s"])
    if prev_uav is not None and prev_uav != uav and start > 0:
        src = int(prev_uav)
        dst = int(uav)
        p = float(state.channel.pdp[src, dst])
        attempts, residual = env._attempts_and_residual(p)
        snr = max(float(state.channel.snr[src, dst]), 0.0)
        spectral_eff = max(float(np.log2(1.0 + snr)), 1e-12)
        comm = float(env.profile.activation_bytes[start - 1]) * 8.0 * attempts / (
            float(env.cfg["wireless"]["bandwidth_hz"]) * spectral_eff
        )
        damage = float(env.profile.importance[start - 1]) * residual
        score += float(reward_cfg["beta"]) * comm / float(reward_cfg["latency_ref_s"])
        score += float(reward_cfg["alpha"]) * float(env.profile.ppl_gamma) * damage
    return score


def block_beam_strong(
    env: LLMUAVEnv,
    state: SimState,
    beam_width: int = 256,
    max_blocks: int | None = None,
    initial_actions: list[np.ndarray] | None = None,
    exact_evals: int | None = None,
) -> tuple[np.ndarray, EvalResult]:
    """Strong structured search over block cuts and block-to-UAV assignments."""

    if max_blocks is None:
        max_blocks = env.num_uavs
    max_blocks = max(1, int(max_blocks))
    beam_width = max(1, int(beam_width))
    exact_evals = int(exact_evals or max(512, beam_width * 16))

    seeds = list(initial_actions or [])
    seeds.extend([block_balanced(env, state), pdp_aware_greedy(env, state), latency_greedy(env, state)])
    cutsets: list[tuple[int, ...]] = []
    seen_cuts: set[tuple[int, ...]] = set()

    def add_cutset(cuts: tuple[int, ...]) -> None:
        valid = _valid_cutset(cuts, env.num_layers, max_blocks)
        if valid is None or valid in seen_cuts:
            return
        seen_cuts.add(valid)
        cutsets.append(valid)

    for action in seeds:
        cuts = _cutset_from_action(action)
        add_cutset(cuts)
        for idx in range(len(cuts)):
            for delta in [-4, -3, -2, -1, 1, 2, 3, 4]:
                shifted = list(cuts)
                shifted[idx] += delta
                add_cutset(tuple(shifted))

    for block_count in range(2, max_blocks + 1):
        uniform = tuple(int(round(x)) for x in np.linspace(0, env.num_layers, block_count + 1)[1:-1])
        add_cutset(uniform)
        prefix_mem = env._prefix_mem_bytes
        total_mem = float(prefix_mem[-1])
        mem_cuts = []
        for q in range(1, block_count):
            target = total_mem * q / block_count
            mem_cuts.append(int(np.searchsorted(prefix_mem, target, side="left")))
        add_cutset(tuple(mem_cuts))

    best_action = None
    best_eval = None
    for seed in seeds:
        ev = env.evaluate(state, seed)
        if ev.feasible and (best_eval is None or ev.reward > best_eval.reward):
            best_action = seed.copy()
            best_eval = ev
        elif best_eval is None:
            best_action = seed.copy()
            best_eval = ev

    scored_actions: list[tuple[float, np.ndarray]] = []
    prefix_mem = env._prefix_mem_bytes
    reward_cfg = env.cfg["reward"]
    latency_ref = float(reward_cfg["latency_ref_s"])
    alpha = float(reward_cfg["alpha"])
    beta = float(reward_cfg["beta"])

    for cuts in cutsets:
        points = (0,) + cuts + (env.num_layers,)
        block_count = len(points) - 1
        block_mem = np.asarray([prefix_mem[points[i + 1]] - prefix_mem[points[i]] for i in range(block_count)])
        compute_cost = np.empty((block_count, env.num_uavs), dtype=np.float64)
        for i in range(block_count):
            start, end = points[i], points[i + 1]
            cycles = float(np.sum(env.profile.compute_cycles[start:end]))
            compute_cost[i] = beta * (cycles / state.resources.compute_hz) / latency_ref

        for assignment in itertools.product(range(env.num_uavs), repeat=block_count):
            mem_used = np.zeros(env.num_uavs, dtype=np.float64)
            feasible_mem = True
            for idx, uav in enumerate(assignment):
                mem_used[uav] += block_mem[idx]
                if mem_used[uav] > state.resources.mem_bytes[uav] + 1e-9:
                    feasible_mem = False
                    break
            if not feasible_mem:
                continue

            proxy = 0.0
            for idx, uav in enumerate(assignment):
                proxy += float(compute_cost[idx, uav])
                if idx == 0:
                    continue
                prev_uav = int(assignment[idx - 1])
                if prev_uav == int(uav):
                    continue
                boundary = points[idx]
                p = float(state.channel.pdp[prev_uav, uav])
                attempts, residual = env._attempts_and_residual(p)
                snr = max(float(state.channel.snr[prev_uav, uav]), 0.0)
                spectral_eff = max(float(np.log2(1.0 + snr)), 1e-12)
                comm = float(env.profile.activation_bytes[boundary - 1]) * 8.0 * attempts / (
                    float(env.cfg["wireless"]["bandwidth_hz"]) * spectral_eff
                )
                damage = float(env.profile.importance[boundary - 1]) * residual
                proxy += beta * comm / latency_ref
                proxy += alpha * float(env.profile.ppl_gamma) * damage

            action = np.empty(env.num_layers, dtype=np.int64)
            for idx, uav in enumerate(assignment):
                action[points[idx] : points[idx + 1]] = int(uav)
            scored_actions.append((proxy, action))

    if not scored_actions:
        assert best_action is not None and best_eval is not None
        return best_action, best_eval

    scored_actions.sort(key=lambda item: item[0])
    seen: set[tuple[int, ...]] = set()
    evaluated = 0
    for _score, action in scored_actions:
        key = tuple(int(x) for x in action.tolist())
        if key in seen:
            continue
        seen.add(key)
        ev = env.evaluate(state, action)
        evaluated += 1
        if ev.feasible and (best_eval is None or ev.reward > best_eval.reward):
            best_action = action.copy()
            best_eval = ev
        elif best_eval is None:
            best_action = action.copy()
            best_eval = ev
        if evaluated >= exact_evals:
            break

    assert best_action is not None and best_eval is not None
    return best_action, best_eval


def block_lns_strong(
    env: LLMUAVEnv,
    state: SimState,
    rng: np.random.Generator,
    initial_actions: list[np.ndarray] | None = None,
    steps: int = 64,
    max_blocks: int | None = None,
) -> tuple[np.ndarray, EvalResult]:
    """Block-level large-neighborhood search around strong structured seeds."""

    if max_blocks is None:
        max_blocks = env.num_uavs
    seeds = list(initial_actions or [])
    seeds.extend([block_balanced(env, state), pdp_aware_greedy(env, state), latency_greedy(env, state)])
    best_action = None
    best_eval = None
    for seed in seeds:
        action = env.project_blocks_fast(seed, state, max_blocks=max_blocks)
        ev = env.evaluate(state, action)
        if ev.feasible and (best_eval is None or ev.reward > best_eval.reward):
            best_action = action.copy()
            best_eval = ev
        elif best_eval is None:
            best_action = action.copy()
            best_eval = ev

    assert best_action is not None and best_eval is not None
    current_action = best_action.copy()
    current_eval = best_eval

    for step in range(max(0, int(steps))):
        blocks = _action_blocks(current_action)
        candidates: list[np.ndarray] = []

        # Destroy/repair by reassigning one or two existing blocks.
        block_indices = list(range(len(blocks)))
        rng.shuffle(block_indices)
        for idx in block_indices[: max(1, min(len(blocks), 3))]:
            start, end, old_uav = blocks[idx]
            for uav in range(env.num_uavs):
                if uav == old_uav:
                    continue
                trial_blocks = list(blocks)
                trial_blocks[idx] = (start, end, int(uav))
                candidates.append(_materialize_blocks(trial_blocks, env.num_layers))

        if len(blocks) >= 2:
            idx = int(rng.integers(0, len(blocks) - 1))
            first = blocks[idx]
            second = blocks[idx + 1]
            for uav in range(env.num_uavs):
                trial_blocks = list(blocks)
                trial_blocks[idx : idx + 2] = [(first[0], second[1], int(uav))]
                candidates.append(_materialize_blocks(trial_blocks, env.num_layers))

            boundary = first[1]
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_boundary = boundary + delta
                if new_boundary <= first[0] or new_boundary >= second[1]:
                    continue
                trial_blocks = list(blocks)
                trial_blocks[idx] = (first[0], new_boundary, first[2])
                trial_blocks[idx + 1] = (new_boundary, second[1], second[2])
                candidates.append(_materialize_blocks(trial_blocks, env.num_layers))

        # Random partial reset preserving contiguity.
        if len(blocks) > 1:
            lo = int(rng.integers(0, len(blocks)))
            hi = int(rng.integers(lo + 1, len(blocks) + 1))
            start = blocks[lo][0]
            end = blocks[hi - 1][1]
            for uav in rng.permutation(env.num_uavs)[: min(env.num_uavs, 3)]:
                trial = current_action.copy()
                trial[start:end] = int(uav)
                candidates.append(trial)

        best_trial_action = current_action
        best_trial_eval = current_eval
        seen: set[tuple[int, ...]] = set()
        for candidate in candidates:
            candidate = env.project_blocks_fast(candidate, state, max_blocks=max_blocks)
            key = tuple(int(x) for x in candidate.tolist())
            if key in seen:
                continue
            seen.add(key)
            ev = env.evaluate(state, candidate)
            if ev.feasible and ev.reward > best_trial_eval.reward + 1e-9:
                best_trial_action = candidate
                best_trial_eval = ev

        if best_trial_eval.reward > current_eval.reward + 1e-9:
            current_action = best_trial_action.copy()
            current_eval = best_trial_eval
            if current_eval.reward > best_eval.reward + 1e-9:
                best_action = current_action.copy()
                best_eval = current_eval
        elif step % 8 == 7 and candidates:
            candidate = env.project_blocks_fast(candidates[int(rng.integers(0, len(candidates)))], state, max_blocks=max_blocks)
            ev = env.evaluate(state, candidate)
            if ev.feasible:
                current_action = candidate
                current_eval = ev

    return best_action, best_eval


def simulated_annealing(
    env: LLMUAVEnv,
    state: SimState,
    rng: np.random.Generator,
    initial: np.ndarray | None = None,
    steps: int = 256,
    start_temp: float = 0.03,
    end_temp: float = 0.002,
) -> tuple[np.ndarray, EvalResult]:
    if initial is None:
        initial = pdp_aware_greedy(env, state)
    current = env.repair_action(initial, state)
    current_eval = env.evaluate(state, current)
    best = current.copy()
    best_eval = current_eval

    for step in range(max(1, steps)):
        frac = step / max(steps - 1, 1)
        temp = start_temp * ((end_temp / start_temp) ** frac)
        trial = current.copy()
        layer = int(rng.integers(0, env.num_layers))
        trial[layer] = int(rng.integers(0, env.num_uavs))
        trial = env.repair_action(trial, state)
        ev = env.evaluate(state, trial)
        delta = ev.reward - current_eval.reward
        accept = delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9))
        if accept:
            current = trial
            current_eval = ev
        if ev.feasible and ev.reward > best_eval.reward:
            best = trial
            best_eval = ev

    return best, best_eval


def evaluate_full_benchmark(
    env: LLMUAVEnv,
    state: SimState,
    rng: np.random.Generator,
    beam_width: int = 24,
    anneal_steps: int = 256,
) -> dict[str, tuple[np.ndarray, EvalResult]]:
    base = evaluate_strong_baselines(env, state, rng)
    beam_action, beam_eval = beam_search(env, state, beam_width=beam_width)
    anneal_action, anneal_eval = simulated_annealing(
        env,
        state,
        rng,
        initial=base["local_search"][0],
        steps=anneal_steps,
    )
    base["beam_search"] = (beam_action, beam_eval)
    base["simulated_annealing"] = (anneal_action, anneal_eval)
    heur_cfg = env.cfg.get("heuristics", {})
    max_blocks = int(heur_cfg.get("max_blocks", env.num_uavs))
    block_beam_action, block_beam_eval = block_beam_strong(
        env,
        state,
        beam_width=int(heur_cfg.get("block_beam_width", max(beam_width, 256))),
        max_blocks=max_blocks,
    )
    base["block_beam_strong"] = (block_beam_action, block_beam_eval)
    block_lns_action, block_lns_eval = block_lns_strong(
        env,
        state,
        rng,
        initial_actions=[a for a, _ in base.values()],
        steps=int(heur_cfg.get("block_lns_steps", 64)),
        max_blocks=max_blocks,
    )
    base["block_lns_strong"] = (block_lns_action, block_lns_eval)
    # Local search from the new strong candidates usually dominates raw beam/SA.
    refined_action, refined_eval = local_search(
        env,
        state,
        rng,
        initial_actions=[a for a, _ in base.values()],
        max_passes=int(env.cfg.get("heuristics", {}).get("local_search_passes", 2)),
        random_seed_tries=int(env.cfg.get("heuristics", {}).get("local_search_seed_tries", 64)),
    )
    base["hybrid_heuristic"] = (refined_action, refined_eval)
    return base


def exhaustive_best(env: LLMUAVEnv, state: SimState, max_states: int = 250000) -> tuple[np.ndarray, EvalResult] | None:
    total = env.num_uavs ** env.num_layers
    if total > max_states:
        return None
    best_action = None
    best_eval = None
    for combo in itertools.product(range(env.num_uavs), repeat=env.num_layers):
        action = np.asarray(combo, dtype=np.int64)
        ev = env.evaluate(state, action)
        if ev.feasible and (best_eval is None or ev.reward > best_eval.reward):
            best_action = action.copy()
            best_eval = ev
    if best_action is None:
        return None
    return best_action, best_eval
