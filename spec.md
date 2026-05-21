# Learning Macroeconomic Policy via RL — Implementation Spec

> Living build plan for the current repo. Milestones 1 and 2 are implemented; Milestone 3 is now the active target.

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
│   ├── env/                        # MacroEnv registration + wrappers
│   ├── models/
│   │   └── economy.py              # state-transition dynamics
│   ├── policies/                   # reserved for later policy modules
│   ├── rl/                         # SAC training code
│   └── utils/
│       ├── config.py               # economy, reward, and baseline-policy parameters
│       └── plotting.py             # reusable plotting helpers
├── train_sac.py                    # SAC training CLI
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

### 1.3 What milestone 2 delivered

- `MacroEnv` now wraps the simulator through the Gymnasium API.
- The RL observation vector is finalized as:
  - `pi, u, g, r, d, E_pi, tau, G`
- Generic action bounds now live in `EconomyConfig`.
- The environment is registered and tested.
- A normalized-action wrapper exists for SAC training.

---

## 2. Milestone 3 Goal

Train a usable SAC baseline on top of the implemented environment, with saved models, periodic evaluation, checkpoints, and macro-aware training diagnostics.

### 2.1 Target files

- `src/rl/sac_trainer.py`
- `src/rl/callbacks.py`
- `train_sac.py`
- `tests/test_sac_trainer.py`

### 2.2 Design principle

The SAC trainer should build on the environment and simulator that already exist. Training code should not duplicate environment logic.

The RL layer should remain responsible for:

- model construction
- training/evaluation callbacks
- checkpointing and metadata saving
- learning-specific wrappers like normalized actions

---

## 3. Milestone 3 Training Design

### 3.1 Environment choice

Use the already-implemented environment:

- `MacroEnv-v0` for native action bounds
- `MacroEnvNormalized-v0` or the equivalent wrapper path for SAC training

Recommended default for SAC:

- train with normalized actions in `[-1, 1]^3`
- keep the observation vector unchanged

### 3.2 Model / network

Default SAC network:

```python
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
    activation_fn=torch.nn.ReLU,
)
```

### 3.3 Training features

The training entry point should support:

- final model saving
- best-model saving from periodic evaluation
- periodic checkpoints
- JSON metadata about the run
- episode-level macro metrics

### 3.4 Metrics to log

At minimum:

- episode mean inflation
- episode std inflation
- episode mean unemployment
- episode std unemployment
- episode mean growth
- episode mean debt
- episode max debt
- episode return

---

## 4. Milestone 3 Test Plan

Create or maintain trainer-focused tests with at least these checks:

1. the normalized training env builds correctly
2. the SAC model object can be constructed
3. callback wiring builds correctly
4. metadata saving works
5. a tiny CLI smoke run can save a model

---

## 5. After Milestone 3

### 5.1 Milestone 4: Evaluation

Expected files:

- `sretegies/evaluate.py`

Compare at least:

- learned SAC policy
- linear Taylor-like baseline
- no-intervention baseline
- optional random baseline

### 5.2 Milestone 5: Ablations

Expected files:

- `sretegies/run_ablations.py`

Good first ablations:

- no expectations channel
- weaker or no lag structure
- reward-weight sensitivity

---

## 6. Current Assessment

### 6.1 What is ready

- The core simulator exists and is test-covered.
- The project already has a reusable rollout entry point.
- Baseline policies are modular and easy to benchmark.
- The Python dependencies needed for Gymnasium and SAC are already installed.

### 6.2 What is not ready yet

- the SAC scaffold still needs longer real runs and tuning
- there is no evaluation/benchmark script yet
- there is no trained-model action-maker module yet for rollout reuse

### 6.3 Verdict

Yes, we are actively in milestone 3.

More precisely:

- we are ready to run meaningful SAC experiments now
- we are not yet at milestone 4 until evaluation scripts and model comparisons exist

---

## 7. Immediate Milestone 3 To-Do List

1. Run longer SAC training jobs.
2. Inspect saved checkpoints and episode metrics.
3. Add a trained-model strategy module for rollout/evaluation reuse.
4. Build the milestone 4 evaluation script.

---

*Last updated: 2026-04-27*
