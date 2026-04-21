# Learning Macroeconomic Policy via RL — Implementation Spec

> Derived from project proposal `spec.pdf`. This document breaks the build into actionable milestones.

---

## 0. Environment & Repo Layout

```
.
├── spec.pdf                  # original proposal
├── spec.md                   # this file
├── requirements.txt          # frozen Python deps
├── .venv/                    # local virtual env
├── src/
│   ├── env/
│   │   └── macro_env.py      # Gymnasium environment
│   ├── models/
│   │   └── economy.py        # state-transition dynamics (eq 4-8)
│   ├── policies/
│   │   ├── random_policy.py
│   │   └── taylor_rule.py    # baseline rule-based policy
│   ├── rl/
│   │   └── sac_trainer.py    # SAC training wrapper
│   └── utils/
│       └── config.py         # hyper-parameters & reward weights
├── experiments/
│   ├── train_sac.py          # CLI entry-point for training
│   ├── evaluate.py           # benchmark vs baselines
│   └── run_ablations.py      # ablation studies (sec 8)
├── viz/
│   └── manim_scenes.py       # Manim animations for dynamics
├── notebooks/
│   └── analysis.ipynb        # exploration & plotting
└── outputs/
    ├── models/               # saved checkpoints
    ├── logs/                 # tensorboard / csv logs
    └── figures/              # plots & rendered videos
```

---

## 1. Build the Gymnasium Environment (`MacroEnv`)

**File:** `src/env/macro_env.py`  
**Depends on:** `src/models/economy.py`, `src/utils/config.py`

### 1.1 State Space (`observation_space`)
Continuous Box with 6 dims (eq 2):

| Dim | Symbol | Description | Typical Range |
|-----|--------|-------------|---------------|
| 0 | $\pi_t$ | Inflation rate | [-5, 15] % |
| 1 | $u_t$ | Unemployment rate | [0, 20] % |
| 2 | $g_t$ | GDP growth rate | [-10, 15] % |
| 3 | $r_t$ | Nominal interest rate | [0, 20] % |
| 4 | $d_t$ | Debt-to-GDP ratio | [0, 200] % |
| 5 | $E_t\pi_{t+1}$ | Expected inflation | [-5, 15] % |

### 1.2 Action Space (`action_space`)
Continuous Box with 2 dims (eq 3):

| Dim | Symbol | Description | Range |
|-----|--------|-------------|-------|
| 0 | $\Delta r_t$ | Interest-rate change | [-2.5, +2.5] pp |
| 1 | $\Delta G_t$ | Fiscal impulse (gov spending change) | [-5, +5] % GDP |

### 1.3 Transition Dynamics (eq 4–8)
Implement in `src/models/economy.py` as a `step(state, action, shock)` function:

1. **Aggregate Demand with Lag** (eq 4)  
   $g_t = \alpha_1 G_{t-1} - \alpha_2 r_{t-1} + \epsilon^d_t$

2. **Expectations-Augmented Phillips Curve** (eq 5)  
   $\pi_t = E_t\pi_{t+1} + \beta(g_t - g^*) + \epsilon^s_t$

3. **Adaptive Expectations** (eq 6)  
   $E_t\pi_{t+1} = \rho\pi_t + (1-\rho)E_{t-1}\pi_t$

4. **Okun’s Law** (eq 7)  
   $u_t = u_{t-1} - \gamma g_t$

5. **Debt Dynamics** (eq 8)  
   $d_t = d_{t-1} + G_t - T_t$  
   (simplify: $T_t = \tau \cdot Y_t$ or constant)

6. **Interest rate & spending accumulation**  
   $r_t = r_{t-1} + \Delta r_t$  
   $G_t = G_{t-1} + \Delta G_t$

### 1.4 Reward Function (eq 9)
Quadratic welfare objective:

$$r_t = -w_1 \pi_t^2 - w_2 u_t^2 + w_3 g_t - w_4 d_t$$

Default weights (configurable in `src/utils/config.py`):
- $w_1 = 1.0$ (inflation aversion)
- $w_2 = 1.0$ (unemployment aversion)
- $w_3 = 0.5$ (growth preference)
- $w_4 = 0.1$ (debt penalty)

### 1.5 Shock Engine
At each step sample:
- $\epsilon^d_t \sim \mathcal{N}(0, \sigma_d)$ — demand shock
- $\epsilon^s_t \sim \mathcal{N}(0, \sigma_s)$ — supply shock

Support **structural break** mode (sec 7.2): mid-episode parameter regime switch.

### 1.6 Episode Logic
- **Reset:** sample initial state near steady-state or from historical distribution.
- **Horizon:** fixed $T = 200$ steps (≈ 50 years quarterly) or early termination if $d_t > d_{\max}$.
- **Truncation:** return `truncated=True` at $t = T$.

### 1.7 Gymnasium API Compliance
Must inherit `gymnasium.Env` and implement:
- `__init__(self, config=None)`
- `reset(self, seed=None, options=None)` → `obs, info`
- `step(self, action)` → `obs, reward, terminated, truncated, info`
- `render(self)` (optional, placeholder for now)

---

## 2. Baseline Policies

**Files:** `src/policies/random_policy.py`, `src/policies/taylor_rule.py`

### 2.1 Random Policy
Sample $\Delta r_t, \Delta G_t$ uniformly from action space. Used as sanity-check lower bound.

### 2.2 Taylor Rule
Classic monetary reaction function, extended with a simple fiscal rule:

- **Monetary:** $\Delta r_t = \phi_\pi (\pi_t - \pi^*) + \phi_u (u_t - u^*)$
- **Fiscal:** $\Delta G_t = -\psi_g (g_t - g^*) - \psi_d (d_t - d^*)$

Parameters from literature (configurable):
- $\phi_\pi = 1.5$, $\phi_u = 0.5$
- $\psi_g = 0.3$, $\psi_d = 0.1$
- Targets: $\pi^* = 2\%$, $u^* = 5\%$, $g^* = 2.5\%$, $d^* = 60\%$

---

## 3. SAC Training Loop

**File:** `src/rl/sac_trainer.py`  
**Entry point:** `experiments/train_sac.py`

### 3.1 Architecture Choice
Use `stable-baselines3.SAC` rather than hand-rolled SAC for reliability, but wrap it so we can:
- Inject custom network architectures (2–3 hidden-layer MLP as per sec 5.1–5.2)
- Log custom metrics (inflation variance, unemployment variance, mean growth, debt trajectory)
- Save checkpoints every $N$ episodes

### 3.2 Custom Network Config
```python
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
    activation_fn=torch.nn.ReLU,
)
```

### 3.3 Training Hyper-parameters
| Param | Value | Notes |
|-------|-------|-------|
| Learning rate | 3e-4 | standard SAC |
| Buffer size | 1e6 | replay buffer |
| Batch size | 256 | |
| $\gamma$ | 0.99 | discount factor |
| $\tau$ | 0.005 | soft-update coefficient |
| Entropy target | auto | SAC default |
| Total timesteps | 2e6 | ~10k episodes |
| Eval frequency | 5e3 | episodes |

### 3.4 Logging
- TensorBoard logs to `outputs/logs/sac/`
- Custom callback recording per-episode:
  - `episode/mean_inflation`, `episode/std_inflation`
  - `episode/mean_unemployment`, `episode/std_unemployment`
  - `episode/mean_growth`
  - `episode/mean_debt`, `episode/max_debt`
  - `episode/return`

---

## 4. Evaluation & Benchmarking

**File:** `experiments/evaluate.py`

### 4.1 Metrics (sec 7.3)
For each policy, run 100 episodes and compute:
1. $\text{Var}(\pi)$ — inflation variance
2. $\text{Var}(u)$ — unemployment variance
3. $\bar{g}$ — mean GDP growth
4. Debt trajectory statistics (mean, max, terminal)
5. Cumulative welfare: $\sum_t r_t$

### 4.2 Shock Scenarios (sec 7.2)
Evaluate under:
- **Baseline shocks:** $\sigma_d = 1.0$, $\sigma_s = 0.5$
- **Demand shock:** 3× demand volatility for 20 steps
- **Supply shock:** 3× supply volatility for 20 steps
- **Structural break:** mid-episode switch to high-debt regime ($\alpha$’s change)

### 4.3 Output
Generate comparison table (Markdown / CSV) and save to `outputs/figures/benchmark.csv`.

---

## 5. Ablation Studies

**File:** `experiments/run_ablations.py`

Re-train SAC with modified environments (sec 8):

| Study | Modification | Purpose |
|-------|--------------|---------|
| A1 — No Expectations | Fix $E_t\pi_{t+1} = \pi^*$ (static) | Test role of expectations |
| A2 — No Lag Structure | Use contemporaneous $G_t, r_t$ in AD | Test role of lags |
| A3 — Reward Sensitivity | Vary $(w_1, w_2, w_3, w_4)$ | Test policy indifference curves |

Train each variant with same hyper-parameters, then evaluate with `experiments/evaluate.py`.

---

## 6. Visualization with Manim

**File:** `viz/manim_scenes.py`

### 6.1 Scene: Economy Dynamics
Animate a single evaluation episode:
- Time-series plot of $\pi_t, u_t, g_t, r_t, d_t$ scrolling left-to-right
- Action bars showing $\Delta r_t, \Delta G_t$ at each step
- Shock indicators (vertical flash when $|\epsilon| > 2\sigma$)

### 6.2 Scene: Policy Comparison
Side-by-side trajectories of SAC vs Taylor rule under identical shock seed:
- Left panel: Taylor rule
- Right panel: SAC policy
- Highlight divergence moments

### 6.3 Scene: Phase Portrait
Plot $(\pi_t, u_t)$ phase space with trajectory tail, showing how SAC drives the economy toward target steady-state vs baselines.

### 6.4 Render Script
Provide `viz/render_all.py` CLI that renders selected scenes to `outputs/figures/manim/`.

---

## 7. Notebook & Analysis

**File:** `notebooks/analysis.ipynb`

Jupyter notebook for:
- Loading trained models & evaluation rollouts
- Plotting learning curves (reward, entropy, Q-values)
- Policy response functions (heatmaps of $\Delta r$ vs $(\pi, u)$)
- Sensitivity analysis plots

---

## 8. Milestone Checklist

| # | Milestone | Key Deliverable | Est. Effort |
|---|-----------|-----------------|-------------|
| 1 | **Economic model** | `src/models/economy.py` + unit tests | 1–2 hrs |
| 2 | **Gymnasium env** | `src/env/macro_env.py` registered & tested | 1–2 hrs |
| 3 | **Baselines** | `src/policies/` + evaluate script | 1 hr |
| 4 | **SAC training** | `experiments/train_sac.py` running | 1–2 hrs |
| 5 | **Evaluation suite** | `experiments/evaluate.py` + benchmark table | 1 hr |
| 6 | **Ablations** | `experiments/run_ablations.py` + results | 1 hr |
| 7 | **Manim viz** | `viz/manim_scenes.py` + rendered videos | 2–3 hrs |
| 8 | **Notebook** | `notebooks/analysis.ipynb` with plots | 1 hr |

---

## 9. Open Questions / Decisions

1. **Tax rule simplification:** Should $T_t$ be constant, proportional to $Y_t$, or follow a separate fiscal rule?
2. **Observation normalization:** Normalize to zero-mean / unit-variance before feeding to NN? (Recommended: yes, via `gymnasium.wrappers.NormalizeObservation` or manual running stats.)
3. **Multi-agent extension:** Future work mentions multi-agent economies — keep interfaces modular so a second agent (fiscal authority separate from monetary) can be added later.
4. **Real-world calibration:** For the demo, use stylized parameters. If later calibrated to US/EU data, move parameters to a YAML config.

---

*Last updated: 2026-04-21*
