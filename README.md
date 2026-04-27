# 🏛️ Teaching an AI to Run a Country's Economy

> *Can a neural network learn to be a better central banker than a human?*

This project builds a **miniature simulated economy** on your computer and eventually trains an AI (using Reinforcement Learning) to manage it. Right now, we have built the economic "engine" — the math that makes inflation, unemployment, growth, and debt interact with each other quarter by quarter.

---

## 📖 The Big Idea (No Jargon)

Every few months, real-world policymakers face tough questions:

- Should the central bank **raise interest rates** to fight inflation, even if it might cause unemployment?
- Should the government **spend more** to boost growth, even if it means borrowing more?
- Should the government **raise taxes** to pay down debt, even if it might slow the economy?

Today, many countries follow simple **rules of thumb**. For example, the famous *Taylor Rule* says: "if inflation is 1% above target, raise interest rates by 1.5%." It's a good starting point, but it's rigid.

**This project asks:** What if we let an AI *learn* its own rule by trial and error? It could try millions of combinations, discover subtle patterns, and maybe react to crises in ways humans haven't thought of.

Before we can train the AI, though, we need a **practice world** — a simulated economy where it can experiment safely. That's what Milestone 1 is.

Milestone 1 is now largely in place. The next step is Milestone 2: wrapping this simulator as a proper `Gymnasium` environment for RL.

---

## 🗂️ What This Repo Contains (So Far)

| Folder / File | What it does |
|---------------|-------------|
| `src/models/economy.py` | The **economic engine**. Contains the equations that move the economy forward one quarter at a time. |
| `src/utils/config.py` | The **settings menu**. All the numbers that control the economy (how sensitive inflation is to growth, how fast unemployment responds, etc.). |
| `run_economy.py` | The **driver script**. Loads an action maker, runs the simulation for a chosen number of quarters, and saves charts plus JSON history. |
| `sretegies/linear_stretegy.py` | The current **hand-written baseline policy**. It implements a simple linear Taylor-like rule that `run_economy.py` can load. |
| `sretegies/none_stretegy.py` | A **no-intervention baseline**. It always returns zero policy changes, so the economy evolves without active government or central-bank intervention. |
| `src/utils/plotting.py` | The shared **plotting module** for simulation histories. |
| `tests/` | The current **assertion layer**. It checks the core economy equations and the pluggable simulation runner. |
| `viz/manim_scenes.py` | An **animation template** (optional). Can turn a simulation run into a video using Manim. |
| `outputs/figures/` | Where the charts and JSON data get saved when you run the simulator. |

---

## 🚀 Quick Start: Run the Simulator

Make sure you have activated the Python virtual environment (it should already be set up):

```bash
# 1. Go to the project folder
cd "Learning Macroeconomic Policy"

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Run the simulator
python run_economy.py

# 4. Optional: run the no-intervention baseline
python run_economy.py --action-maker sretegies.none_stretegy:build_action_maker
```

After a few seconds, check the `outputs/figures/` folder. You should see:

- `trajectory_baseline.png` — a 6-panel chart showing how the economy evolved
- `actions_baseline.png` — a chart showing what the policy rule did each quarter
- `history_baseline.json` — the raw recorded trajectory for later analysis

---

## 📊 How to Read the Figures

### `trajectory_baseline.png` — The Economy Over Time

This is a 6-panel chart. Each panel tracks one economic variable across 200 quarters (50 years). The x-axis is always **Quarter** (0 to 200). Let's go through each panel:

#### 1. Inflation π (%)
- **What it means:** How fast prices are rising. If π = 2%, something that cost $100 last year costs $102 now.
- **What to look for:** Most central banks aim for ~2%. In our simulation, you can see inflation swinging above and below target. When it spikes, the policy rule tries to cool it down by raising interest rates.

#### 2. Unemployment u (%)
- **What it means:** The percentage of people who want to work but can't find a job.
- **What to look for:** Unemployment and inflation tend to move in *opposite* directions — this is the famous **Phillips Curve** trade-off. When growth is hot, unemployment falls, but inflation rises. The policymaker has to balance the two.

#### 3. GDP Growth g (%)
- **What it means:** How fast the total output of the economy is growing. g = 2.5% means the economy produced 2.5% more stuff this year than last year.
- **What to look for:** Growth wiggles around because of random **shocks** (unexpected events) and because policy tools affect it with a **lag** — a rate cut today might take several quarters to fully stimulate spending.

#### 4. Interest Rate r (%)
- **What it means:** The nominal interest rate set by the central bank. Higher rates make borrowing expensive, which cools down spending and inflation. Lower rates encourage borrowing and stimulate the economy.
- **What to look for:** Notice how rates sometimes hit **0%** and stay there for a while. This is the **Zero Lower Bound** (ZLB) — central banks can't easily cut below 0% with normal tools. It's a major challenge in real-world recessions.

#### 5. Debt/GDP d (%)
- **What it means:** How much the government owes, expressed as a percentage of the economy's total yearly output. d = 60% means the government debt equals 60% of one year's GDP.
- **What to look for:** Debt goes up when the government spends more than it collects in taxes (a **budget deficit**). Now that the tax rate is adjustable, you can see both spending *and* tax policy affecting the debt trajectory. Raise taxes or cut spending → debt falls. Cut taxes or spend more → debt rises.

#### 6. Expected Inflation Eπ (%)
- **What it means:** What households and businesses *think* inflation will be next year. This matters because expectations become self-fulfilling: if workers expect 5% inflation, they demand 5% wage raises, and firms raise prices by 5%.
- **What to look for:** In our model, people form expectations **adaptively** — they look at recent inflation and slowly update their beliefs. You can see Eπ trailing behind actual inflation.

### `actions_baseline.png` — What the Policymaker Did

This chart shows the **policy actions** taken each quarter:

- **Δr (monetary)** — How much the central bank changed the interest rate this quarter. Positive = raised rates (tightening). Negative = cut rates (stimulating).
- **ΔG (fiscal)** — How much the government changed its spending this quarter. Positive = spent more. Negative = austerity.
- **Δτ (tax)** — How much the government changed the tax rate this quarter. Positive = raised taxes (reduces deficit). Negative = tax cuts (stimulating).

These actions are calculated by a simple **Taylor-like rule** in `sretegies/linear_stretegy.py`. It's not the AI yet — it's a hand-written formula. The AI will replace this later.

#### Economic Health Over Time (bottom panel)
- **What it means:** The **reward** score each quarter. Think of it as a "report card" for the economy. It penalizes deviations of inflation and unemployment from their targets, rewards growth above potential, and lightly penalizes high debt. **Zero = perfectly on target.** Positive = doing better than target. Negative = missing targets.
- **What to look for:** A healthy economy hovers near zero. During a crisis, the reward drops sharply into negative territory. Notice how the supply shock scenario drives the reward much lower than baseline — that's stagflation at work.

### `trajectory_demand_shock.png`

This is the same economy, but between quarters 50 and 70 we injected a **demand shock** — a sudden, unusually large burst of consumer spending (or collapse of it).

Run it with:

```bash
python run_economy.py --shock-scenario demand_shock --run-name demand_shock
```

- **What is a demand shock?** Imagine everyone suddenly decides to buy new cars and houses, or the opposite — everyone panics and stops spending. This disturbs the normal flow of the economy.
- **What to look for:** GDP growth becomes much more volatile during the shock period. Inflation and unemployment swing more wildly. The interest rate shoots up or down as the rule tries to compensate.

### `trajectory_supply_shock.png`

This is the same economy, but between quarters 50 and 70 we injected a **supply shock** — a sudden change in production costs.

Run it with:

```bash
python run_economy.py --shock-scenario supply_shock --run-name supply_shock
```

- **What is a supply shock?** Imagine oil prices suddenly triple because of a war, or a pandemic shuts down factories. It becomes more expensive to produce things, so prices rise even if demand hasn't changed.
- **What to look for:** Supply shocks are nasty because they cause **stagflation** — inflation goes UP while growth goes DOWN. The policymaker faces a terrible trade-off: raise rates to fight inflation (making unemployment worse), or cut rates to save jobs (making inflation worse).

---

## 🧠 Key Concepts Explained

| Term | Simple Explanation |
|------|-------------------|
| **Simulation** | A computer program that pretends to be a real economy. Like a video game where you are the central bank. |
| **Policy Action** | A decision made by the government or central bank. In our model: changing interest rates (monetary policy), changing government spending (fiscal policy), or changing the tax rate (fiscal policy). |
| **Shock** | A random unexpected event that disturbs the economy. Like a pandemic, an oil crisis, or a financial panic. |
| **Baseline** | The "normal" version of the simulation, with only small random noise. It's the reference point we compare everything else to. |
| **Lag** | The delay between a policy action and its effect. If you cut interest rates today, businesses might not borrow and invest for several months. |
| **Adaptive Expectations** | The idea that people form expectations about the future based on what they have recently observed. If inflation was high last year, they expect it to be high next year too. |
| **Zero Lower Bound (ZLB)** | Interest rates can't easily go below 0%. When they hit 0%, central banks lose their main tool for stimulating the economy. |
| **Stagflation** | A bad situation where inflation is high AND growth is low / unemployment is high. Very hard to fix with normal policy tools. |

---

## 🐍 The Python Scripts Explained

### `src/models/economy.py`

This is the **heart** of the project. It defines a class called `Economy` that keeps track of all the variables (inflation, unemployment, etc.) and moves them forward one quarter at a time.

The key method is `step(action)`:
1. It takes a policy action (change in interest rate, change in spending, change in tax rate).
2. It samples random shocks.
3. It applies the economic equations to compute next quarter's values.
4. It returns the new state, a "reward" score, and whether the episode ended.

Think of it like a turn-based game: every quarter, the AI (or demo rule) makes a move, and the economy updates in response.

### `src/utils/config.py`

This file stores all the "knobs and dials" of the economy in two dataclasses:

- `EconomyConfig` — parameters like "how much does a rate cut boost demand?" (`alpha_2`), "how steep is the Phillips Curve?" (`beta`), etc.
- `RewardConfig` — weights for the welfare function. We penalize deviations of inflation and unemployment from their targets, reward growth above potential, and lightly penalize debt above its sustainable level.

If you want to experiment, try changing these numbers and re-running the simulator.

### `run_economy.py`

This is the script you actually run. It does three things:

1. **Loads an action maker** — by default it imports the linear Taylor-style policy from `sretegies/linear_stretegy.py`, but later an RL policy can expose the same builder interface.
2. **Runs the simulation** — loops for the requested number of quarters, applying the chosen action maker each step.
3. **Saves outputs** — writes the raw JSON history and, unless disabled, calls the plotting module to create the standard figures in `outputs/figures/`.

The important design change is that the runner no longer cares whether the action comes from a rule, an RL agent, or something else. You swap policies with `--action-maker`.

### `sretegies/linear_stretegy.py`

This file holds the current human-written baseline policy only. It takes the current observed state and turns it into three actions:

- a rate move `Δr`
- a spending move `ΔG`
- a tax move `Δτ`

Because it is separate from the simulation loop, we can later replace it with an RL action maker without rewriting the rest of the pipeline.

### `sretegies/none_stretegy.py`

This is the simplest baseline in the project. It always returns:

- `Δr = 0`
- `ΔG = 0`
- `Δτ = 0`

That gives us a useful benchmark for the question:

"How much stabilization is really coming from policy, rather than from the economy drifting on its own?"

---

## 🔮 Where This Is Going

Milestone 1 was about building the **world**. The next milestones will bring in the **AI**:

| Milestone | What happens |
|-----------|-------------|
| **Milestone 2** | Wrap the economy in a **Gymnasium environment** — a standard interface that RL libraries can connect to. The agent will see inflation, unemployment, growth, debt, etc. and choose three actions: how much to move interest rates, spending, and taxes. |
| **Milestone 3** | Train a **Soft Actor-Critic (SAC)** agent to learn its own policy rule through trial and error. |
| **Milestone 4** | Compare the AI against the Taylor Rule and a random policy. Can the AI discover better ways to combine monetary and fiscal tools? |
| **Milestone 5** | Run **ablation studies** — e.g., "what if people had perfect foresight instead of adaptive expectations?" |
| **Milestone 6** | Create **Manim animations** to visualize how the AI thinks and reacts to crises. |

---

## ✅ Milestone 2 Readiness

Short answer: **yes, we are ready to start Milestone 2, but Milestone 2 is not built yet.**

What is already ready:

- the economy core exists and is test-covered
- the action interface already supports the full 3-part policy move `Δr`, `ΔG`, `Δτ`
- the rollout runner is modular, so RL can later replace a hand-written strategy cleanly
- `gymnasium`, `torch`, and `stable-baselines3` are already in the environment

What still needs to be done for Milestone 2:

- build `src/env/macro_env.py`
- decide the final RL observation vector
- centralize generic action bounds for RL instead of keeping some of them inside the Taylor policy config
- add environment tests and run Gymnasium's env checker

One important modeling note:

The current policy-maker interface does not expose government spending `G_t`, even though the simulator uses lagged government spending in the transition equations. For RL, we should probably expose `G_t` in the environment observation so the agent gets a proper Markov state.

That means the project is in a good place to begin Milestone 2, and the next clean move is to build the environment wrapper around the simulator we already have.

---

## 📝 Requirements

The project uses a Python virtual environment. Key packages:

- `numpy`, `scipy`, `pandas` — math and data handling
- `matplotlib`, `seaborn` — static charts
- `manim` — animations (optional)
- `torch`, `stable-baselines3`, `gymnasium` — for later RL training

All dependencies are listed in `requirements.txt`.

---

## 🤝 Tips for Playing Around

- **Change the policy rule:** Open `sretegies/linear_stretegy.py` and tweak the linear coefficients. What happens if the central bank is more aggressive? More passive? What if the government raises taxes faster when debt is high?
- **Run a no-intervention baseline:** `python run_economy.py --action-maker sretegies.none_stretegy:build_action_maker`
- **Change the economy:** Open `src/utils/config.py` and tweak `alpha_1`, `beta`, `gamma`, etc. What makes the economy more stable? More volatile?
- **Change the action source:** Run `python run_economy.py --action-maker your_module:build_action_maker` once you create an RL policy builder with the same interface.
- **Change the shocks:** In `run_economy.py`, modify when the demand/supply shocks happen or how strong they are.

The best way to learn is to break things and see what happens!

---

*Built for a demo project on Learning Macroeconomic Policy via Reinforcement Learning.*
