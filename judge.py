"""
Evaluation script for NTU DRL HW4 Q3 (DMC humanoid-walk).

Usage:
    python judge.py [--student-path PATH] [--output PATH] [--num-episodes N]

Loads the student's agent from student_agent.py (class `Agent`) and evaluates
it over NUM_EPISODES episodes of `humanoid-walk`, writing a results JSON
compatible with the leaderboard's `update_score.py`:

    {"score": mean-std, "mean_return": ..., "std_return": ..., "num_episodes": ...}
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

from dmc import make_dmc_env


# ── Config ──────────────────────────────────────────────────────────────
TASK              = "humanoid-walk"
NUM_EPISODES      = 100
ENV_SEED_BASE     = 1000   # per-episode env seeds: 1000..(1000+N-1), same for every student
POLICY_SEED_BASE  = 0      # per-episode policy RNG seeds: 0..(N-1)
MAX_TOTAL_SIZE_MB = 51     # submission size limit (source + weights)


# ── Submission size guard ───────────────────────────────────────────────
def check_submission_size(student_path: str) -> None:
    root = Path(student_path)
    total = 0
    files = []
    for f in root.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            total += size
            files.append((f.relative_to(root), size))
    files.sort(key=lambda x: x[1], reverse=True)

    print("\nStudent submission files:")
    for name, size in files:
        print(f"  {name}: {size / 1024 / 1024:.2f} MB")
    print(f"  Total: {total / 1024 / 1024:.2f} MB (limit: {MAX_TOTAL_SIZE_MB} MB)\n")

    if total > MAX_TOTAL_SIZE_MB * 1024 * 1024:
        raise RuntimeError(
            f"Submission too large: {total / 1024 / 1024:.2f} MB "
            f"(limit: {MAX_TOTAL_SIZE_MB} MB)"
        )


# ── Student agent loader ────────────────────────────────────────────────
def load_student_agent(student_path: str):
    agent_file = Path(student_path) / "student_agent.py"
    if not agent_file.exists():
        raise FileNotFoundError(f"student_agent.py not found at {agent_file}")

    check_submission_size(student_path)
    sys.path.insert(0, str(Path(student_path).resolve()))

    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Agent"):
        raise ImportError("student_agent.py must define an `Agent` class (Q3 spec).")

    try:
        agent = module.Agent()
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate student Agent(): {exc}") from exc

    if not callable(getattr(agent, "act", None)):
        raise TypeError("Agent must implement act(observation) -> np.ndarray")

    return agent


# ── Anti-cheat RNG seeding ──────────────────────────────────────────────
def seed_policy_rngs(env, agent, seed: int) -> None:
    """Seed every RNG that could affect the policy's actions.

    Called *after* env.reset() + optional agent.reset() so students cannot
    override these inside their own reset(). Covers stdlib random, numpy,
    torch (CPU + CUDA), env.action_space, and the agent's own action_space
    (for random-baseline style `action_space.sample()`).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    actor_space = getattr(agent, "action_space", None)
    if actor_space is not None and actor_space is not env.action_space:
        try:
            actor_space.seed(seed)
        except Exception:
            pass


# ── Episode loop ────────────────────────────────────────────────────────
def run_episode(agent, env_seed: int, policy_seed: int) -> float:
    env = make_dmc_env(TASK, env_seed, flatten=True, use_pixels=False)
    obs, _ = env.reset()
    if hasattr(agent, "reset") and callable(agent.reset):
        try:
            agent.reset()
        except TypeError:
            pass  # signature mismatch — skip
    seed_policy_rngs(env, agent, policy_seed)

    total = 0.0
    done = False
    while not done:
        action = agent.act(obs)
        obs, r, term, trunc, _ = env.step(action)
        total += float(r)
        done = bool(term) or bool(trunc)
    env.close()
    return total


def run_eval(agent, num_episodes: int = NUM_EPISODES) -> dict:
    returns = []
    print(f"Evaluating {TASK} over {num_episodes} episodes...")
    for i in range(num_episodes):
        env_seed    = ENV_SEED_BASE    + i
        policy_seed = POLICY_SEED_BASE + i
        ret = run_episode(agent, env_seed=env_seed, policy_seed=policy_seed)
        returns.append(ret)
        print(f"  ep={i:3d}  env_seed={env_seed}  policy_seed={policy_seed}  return={ret:.2f}")

    mean_return = float(np.mean(returns))
    std_return  = float(np.std(returns))
    score       = mean_return - std_return

    print(f"\n{'='*60}")
    print(f"  mean(return) = {mean_return:.4f}")
    print(f"  std(return)  = {std_return:.4f}")
    print(f"  score        = mean − std = {score:.4f}")
    print(f"{'='*60}\n")

    return {
        "score":        round(score,       4),
        "mean_return":  round(mean_return, 4),
        "std_return":   round(std_return,  4),
        "num_episodes": num_episodes,
    }


def save_results(results: dict, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a student HW4 Q3 agent on humanoid-walk.")
    parser.add_argument("--student-path", default=".", help="Directory containing student_agent.py")
    parser.add_argument("--output",       default="results.json", help="Path to write results JSON")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES, help="Override episode count (for debugging)")
    args = parser.parse_args()

    agent = load_student_agent(args.student_path)
    results = run_eval(agent, num_episodes=args.num_episodes)
    save_results(results, args.output)
