# Learning Macroeconomic Policy via RL — Implementation Spec

> Living build plan for the current repo. Milestone 1 is largely complete; Milestone 2 is the next implementation target.

---

## 0. Current Repo Layout

```
.
├── README.md
├── spec.pdf
├── spec.md
├── main.tex
├── run_economy.py                  # pluggable simulation runner
├── sretegies/
│   ├── linear_stretegy.py          # linear Taylor-like baseline policy
│   └── none_stretegy.py            # no-intervention baseline policy
├── src/
│   ├── env/                        # milestone 2 target location
│   ├── models/
│   │   └── economy.py              # state-transition dynamics
│   ├── policies/                   # reserved for later policy modules
│   ├── rl/                         # reserved for SAC training code
│   └── utils/
│       ├── config.py               # economy, reward, and baseline-policy parameters
│       └── plotting.py             # reusable plotting helpers
├── tests/
│   ├── test_economy.py
│   └── test_run_economy.py
├── outputs/
│   └── figures/
└── viz/
    └── manim_scenes.py             # optional later visualization
```

---

## 1. Milestone 1 Snapshot

### 1.1 What is already built

- `src/models/economy.py` implements the current macro transition system with inflation, unemployment, growth, interest rate, debt, expected inflation, and tax dynamics.
- The simulator uses a three-part action interface:
  - `Δr` for the policy rate
  - `ΔG` for government spending
  - `Δτ` for the tax take
- `run_economy.py` provides a reusable rollout entry point that can load different policy makers through `module:function` import strings.
- Two baseline policies now exist:
  - `sretegies/linear_stretegy.py`
  - `sretegies/none_stretegy.py`
- `src/utils/plotting.py` contains the standard Matplotlib output pipeline.
- Unit tests cover the economy equations, reward behavior, episode termination/truncation, and the pluggable runner contract.

### 1.2 Current strengths

- The economy core is stable enough to run repeated rollouts.
- Policy generation is already decoupled from simulation, which is exactly what we want before RL.
- The dependency stack already includes `gymnasium`, `torch`, and `stable_baselines3`.
- We can compare human-written baselines before training any RL agent.

### 1.3 Known gaps carried into milestone 2

- There is no `Gymnasium` environment wrapper yet.
- Action bounds are not fully centralized for RL:
  - `delta_tau_bounds` lives in `EconomyConfig`
  - `delta_r_bounds` and `delta_G_bounds` currently live in `TaylorPolicyConfig`
- The current policy-facing observation does not expose government spending `G_t`, even though the transition equations depend on it. That means the current rollout interface is convenient, but not fully Markov for RL.

---

## 2. Milestone 2 Goal

Build a `Gymnasium` environment around the existing simulator so an RL agent can interact with the economy through the standard API without duplicating transition logic.

### 2.1 Target files

- `src/env/macro_env.py`
- `src/env/__init__.py`
- `tests/test_macro_env.py`

### 2.2 Design principle

`MacroEnv` should wrap `src/models/economy.py`, not reimplement it.

The economy model should remain the single source of truth for:

- state transitions
- reward calculation
- termination and truncation logic
- shock sampling

---

## 3. Recommended Environment Design

### 3.1 Observation space

Recommended: use an 8-dimensional continuous `Box` so the RL state is as close to Markov as possible.

| Dim | Symbol | Meaning | Typical Range |
|-----|--------|---------|---------------|
| 0 | `pi` | Inflation | `[-5, 15]` |
| 1 | `u` | Unemployment | `[0, 20]` |
| 2 | `g` | GDP growth | `[-10, 15]` |
| 3 | `r` | Policy rate | `[0, 10]` |
| 4 | `d` | Debt/GDP | `[0, 200]` |
| 5 | `E_pi` | Expected inflation | `[-5, 15]` |
| 6 | `tau` | Tax take | `[0, 50]` |
| 7 | `G` | Government spending level | `[0, 100]` |

Why add `G`?

- The next growth step depends on lagged government spending.
- If `G` stays hidden, the agent is acting in a partially observed environment.
- For milestone 2, a Markov state is the cleaner default.

### 3.2 Action space

Use a 3-dimensional continuous `Box`:

| Dim | Symbol | Meaning | Recommended Default Bounds |
|-----|--------|---------|-----------------------------|
| 0 | `delta_r` | Interest-rate change | `[-0.8, 0.8]` |
| 1 | `delta_G` | Government spending change | `[-2.0, 2.0]` |
| 2 | `delta_tau` | Tax change | `[-2.0, 2.0]` |

Recommendation:

- move all generic action bounds into `EconomyConfig` or a new shared action config
- do not leave RL action limits inside `TaylorPolicyConfig`, because those bounds are not Taylor-specific

### 3.3 Reset / step API

`MacroEnv` should expose:

- `reset(seed=None, options=None) -> obs, info`
- `step(action) -> obs, reward, terminated, truncated, info`
- `observation_space`
- `action_space`
- `render()` as optional placeholder

### 3.4 Info dictionary

Keep the env `info` dictionary rich enough for debugging and later evaluation:

- `eps_d`
- `eps_s`
- `G_t`
- `tau`
- `delta_r`
- `delta_G`
- `delta_tau`
- `step`
- `terminated`
- `truncated`

---

## 4. Milestone 2 Test Plan

Create `tests/test_macro_env.py` with at least these checks:

1. `reset()` returns an observation with the expected shape and dtype.
2. `step()` returns `obs, reward, terminated, truncated, info` in Gymnasium format.
3. Horizon handling sets `truncated=True` without forcing `terminated=True`.
4. Debt-limit breach sets `terminated=True`.
5. Deterministic stepping works when shocks are fixed or seeded.
6. The environment passes `gymnasium.utils.env_checker.check_env`.
7. Action clipping and observation bounds behave as expected.

---

## 5. After Milestone 2

### 5.1 Milestone 3: SAC training

Expected files:

- `src/rl/sac_trainer.py`
- `sretegies/train_sac.py`

Use `stable_baselines3.SAC` with a custom network config and callbacks for macro metrics.

### 5.2 Milestone 4: Evaluation

Expected files:

- `sretegies/evaluate.py`

Compare at least:

- learned SAC policy
- linear Taylor-like baseline
- no-intervention baseline
- optional random baseline

### 5.3 Milestone 5: Ablations

Expected files:

- `sretegies/run_ablations.py`

Good first ablations:

- no expectations channel
- weaker or no lag structure
- reward-weight sensitivity

---

## 6. Readiness Assessment

### 6.1 What is ready

- The core simulator exists and is test-covered.
- The project already has a reusable rollout entry point.
- Baseline policies are modular and easy to benchmark.
- The Python dependencies needed for Gymnasium and SAC are already installed.

### 6.2 What is not ready yet

- `MacroEnv` itself does not exist.
- The observation design for RL has not been finalized in code.
- Generic action bounds are not yet centralized outside the Taylor policy config.
- There is no environment test suite or env-checker integration yet.

### 6.3 Verdict

Yes, we are ready to start milestone 2.

More precisely:

- we are ready to build the environment wrapper now
- we are not yet ready to start RL training until that wrapper and its tests exist

---

## 7. Immediate Milestone 2 To-Do List

1. Add shared RL action bounds to config.
2. Decide whether the RL observation includes `G_t` explicitly. Recommended: yes.
3. Implement `src/env/macro_env.py` as a thin wrapper over `Economy`.
4. Add `tests/test_macro_env.py`.
5. Run Gymnasium's env checker.
6. Smoke-test the env with random actions before touching SAC.

---

*Last updated: 2026-04-27*
