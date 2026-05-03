from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .autoreg_rl_agent import AutoregRLAgent
from .baselines import evaluate_full_benchmark
from .config import ensure_dir, load_config
from .env import LLMUAVEnv
from .llm_profile import build_qwen3_0p6b_profile


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
) -> list[float]:
    rewards = []
    for idx in range(states):
        state = env.sample_state()
        methods = evaluate_full_benchmark(env, state, rng, beam_width=beam_width, anneal_steps=anneal_steps)
        action, ev = max((item for item in methods.values()), key=lambda item: item[1].reward)
        action = env.project_action(action, state, max_passes=2)
        agent.replay.add(env.state_vector(state), agent.mem_caps_norm(state), action)
        rewards.append(ev.reward)
        if (idx + 1) % 500 == 0:
            print(f"teacher_collected={idx + 1}/{states} reward_mean={np.mean(rewards):.4f}", flush=True)
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["uav"]["num_uavs"] = args.num_uavs
    if args.num_layers is not None:
        cfg["profile"]["num_layers"] = args.num_layers
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

    seed = int(cfg["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(args.out or ar_cfg.get("result_dir", "results/autoreg_rl"))
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    profile = build_qwen3_0p6b_profile(cfg["profile"], rng)
    env = LLMUAVEnv(cfg, profile, rng)
    agent = AutoregRLAgent(env, cfg, str(ar_cfg.get("device", "cuda")), rng)

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
    temp_start = float(ar_cfg.get("temperature_start", 1.2))
    temp_end = float(ar_cfg.get("temperature_end", 0.35))
    entropy_start = float(ar_cfg.get("entropy_coef_start", 0.02))
    entropy_end = float(ar_cfg.get("entropy_coef_end", 0.002))
    advantage_scale = float(ar_cfg.get("advantage_scale", 4.0))
    reward_clip = float(ar_cfg.get("reward_clip", 8.0))
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
        )
        for update in range(1, teacher_updates + 1):
            stats = agent.replay_update(batch_size=batch_size, updates=1)
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
                    f"eval_feas={ev['feasible_rate']:.3f} loss={(stats or {}).get('replay_loss', float('nan')):.4f} "
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
        states = [env.sample_state() for _ in range(batch_states)]
        state_vecs = np.stack([env.state_vector(state) for state in states])
        caps = np.stack([agent.mem_caps_norm(state) for state in states])
        batch = agent.sample_batch(state_vecs, caps, candidates=candidates, temperature=temperature)

        reward_matrix = np.empty((batch_states, candidates), dtype=np.float64)
        feasible_matrix = np.empty((batch_states, candidates), dtype=np.float64)
        best_rewards = []
        best_costs = []
        best_latencies = []
        best_ppls = []
        best_feasible = []

        for sid, state in enumerate(states):
            actions = np.stack([env.project_action(action, state, max_passes=2) for action in batch.actions[sid]])
            evals = env.evaluate_many(state, actions)
            shaped = np.asarray([agent.shaped_score(ev) for ev in evals], dtype=np.float64)
            reward_matrix[sid] = shaped
            feasible_matrix[sid] = np.asarray([float(ev.feasible) for ev in evals], dtype=np.float64)
            ranked = np.argsort(shaped)[::-1]
            best_ev = evals[int(ranked[0])]
            best_rewards.append(best_ev.reward)
            best_costs.append(best_ev.cost)
            best_latencies.append(best_ev.latency_s)
            best_ppls.append(best_ev.ppl_hat)
            best_feasible.append(float(best_ev.feasible))
            for elite_i in ranked[:elite_count]:
                agent.replay.add(state_vecs[sid], caps[sid], actions[int(elite_i)])

        rewards_t = torch.from_numpy(reward_matrix.astype(np.float32)).to(agent.device)
        baseline = rewards_t.mean(dim=1, keepdim=True)
        std = rewards_t.std(dim=1, keepdim=True).clamp_min(1e-3)
        advantages = (advantage_scale * (rewards_t - baseline) / std).clamp(-reward_clip, reward_clip)
        policy_loss = -(batch.log_probs * advantages.detach()).mean()
        entropy_loss = -entropy_coef * batch.entropy.mean()
        loss = policy_loss + entropy_loss
        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), float(ar_cfg.get("grad_clip", 1.0)))
        agent.optimizer.step()
        replay_stats = agent.replay_update(batch_size=batch_size, updates=replay_updates)

        rows.append(
            {
                "episode": episode,
                "reward": float(np.mean(best_rewards)),
                "cost": float(np.mean(best_costs)),
                "feasible": float(np.mean(best_feasible)),
                "latency_s": float(np.mean(best_latencies)),
                "ppl_hat": float(np.mean(best_ppls)),
                "candidate_reward": float(np.mean(reward_matrix)),
                "candidate_feasible": float(np.mean(feasible_matrix)),
                "temperature": temperature,
                "entropy_coef": entropy_coef,
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
                f"episode={episode:04d} train_reward={np.mean(best_rewards):.4f} "
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
        "final_eval": eval_rows[-1] if eval_rows else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
