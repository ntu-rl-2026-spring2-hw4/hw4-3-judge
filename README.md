# HW4 Q3 Judge Workflow

Self-hosted evaluation pipeline for NTU DRL HW4 Q3 (DMC **humanoid-run**).
Runs on GitHub-hosted `ubuntu-latest` — no GPU required.

## Structure
```
hw4-q3-judge-workflow/
├── judge.py                          # evaluation driver
├── dmc.py                            # DMC gymnasium wrapper (copied from student template)
├── requirements.txt                  # pinned RL/sim stack
└── .github/workflows/evaluate.yml    # reusable workflow_call
```

## How it's used

A student repo's `.github/workflows/main.yml` calls this workflow via `uses:`:

```yaml
name: Evaluate HW4 Q3
on:
  push:        { branches: [main] }
  workflow_dispatch:

jobs:
  evaluate:
    uses: ntu-rl-2026-spring2-hw4/hw4-3-judge/.github/workflows/evaluate.yml@main
    secrets:
      LEADERBOARD_TOKEN: ${{ secrets.LEADERBOARD_TOKEN }}
```

The workflow:

1. Checks out the student repo and this judge repo side-by-side.
2. Parses `Q3/meta.xml` for the student ID (rejects the placeholder `r00000000`).
3. Installs MuJoCo system libs (`libosmesa6`, `libgl1`, `libglfw3`) and
   `MUJOCO_GL=osmesa` for headless rendering.
4. Installs the judge's pinned Python deps, then the student's own
   `requirements.txt` (so the student can add torch, etc. without changing
   the sim stack).
5. Runs `judge.py`, which evaluates `Agent()` over 100 episodes of
   `humanoid-run` and writes `results.json`:
   ```json
   { "score": 453.2, "mean_return": 480.0, "std_return": 26.8, "num_episodes": 100 }
   ```
   `score = mean(returns) − std(returns)`.
6. POSTs `repository_dispatch(event_type=submit_score)` to the leaderboard repo.

## Anti-cheat

For every episode:

- The env is instantiated with a fixed seed from `ENV_SEED_BASE + i`
  (same sequence for every student).
- `seed_policy_rngs(env, agent, POLICY_SEED_BASE + i)` seeds stdlib
  `random`, `numpy`, `torch` (CPU+CUDA), `env.action_space`, and the
  agent's own `action_space` — *after* `env.reset()` and optional
  `agent.reset()`, so students cannot override.

## Local smoke test

```bash
pip install -r requirements.txt torch
MUJOCO_GL=osmesa python judge.py \
  --student-path ../DRL-Assignment-4/Q3 \
  --output /tmp/r.json \
  --num-episodes 2
```
