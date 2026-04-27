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

---

## 🗂️ What This Repo Contains (So Far)

| Folder / File | What it does |
|---------------|-------------|
| `src/models/economy.py` | The **economic engine**. Contains the equations that move the economy forward one quarter at a time. |
| `src/utils/config.py` | The **settings menu**. All the numbers that control the economy (how sensitive inflation is to growth, how fast unemployment responds, etc.). |
| `experiments/demo_economy.py` | The **driver script**. Sets up a simple human-written policy rule, runs the simulation for 200 quarters (50 years), and draws charts. |
| `viz/manim_scenes.py` | An **animation template** (optional). Can turn a simulation run into a video using Manim. |
| `outputs/figures/` | Where the charts and JSON data get saved when you run the demo. |

---

## 🚀 Quick Start: Run the Demo

Make sure you have activated the Python virtual environment (it should already be set up):

```bash
# 1. Go to the project folder
cd "Learning Macroeconomic Policy"

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Run the demo
python experiments/demo_economy.py
```

After a few seconds, check the `outputs/figures/` folder. You should see:

- `trajectory(Baseline).png` — a 6-panel chart showing how the economy evolved
- `actions(Baseline).png` — a chart showing what the policy rule did each quarter
- `trajectory(Demand_Shock).png` — the same economy, but with a sudden demand crisis
- `trajectory(Supply_Shock).png` — the same economy, but with a supply-side crisis

---

## 📊 How to Read the Figures

### `trajectory(Baseline).png` — The Economy Over Time

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

### `actions(Baseline).png` — What the Policymaker Did

This chart shows the **policy actions** taken each quarter:

- **Δr (monetary)** — How much the central bank changed the interest rate this quarter. Positive = raised rates (tightening). Negative = cut rates (stimulating).
- **ΔG (fiscal)** — How much the government changed its spending this quarter. Positive = spent more. Negative = austerity.
- **Δτ (tax)** — How much the government changed the tax rate this quarter. Positive = raised taxes (reduces deficit). Negative = tax cuts (stimulating).

These actions are calculated by a simple **Taylor-like rule** in the demo script. It's not the AI yet — it's a hand-written formula. The AI will replace this later.

#### Economic Health Over Time (bottom panel)
- **What it means:** The **reward** score each quarter. Think of it as a "report card" for the economy. It penalizes deviations of inflation and unemployment from their targets, rewards growth above potential, and lightly penalizes high debt. **Zero = perfectly on target.** Positive = doing better than target. Negative = missing targets.
- **What to look for:** A healthy economy hovers near zero. During a crisis, the reward drops sharply into negative territory. Notice how the supply shock scenario drives the reward much lower than baseline — that's stagflation at work.

### `trajectory(Demand_Shock).png`

This is the same economy, but between quarters 50 and 70 we injected a **demand shock** — a sudden, unusually large burst of consumer spending (or collapse of it).

- **What is a demand shock?** Imagine everyone suddenly decides to buy new cars and houses, or the opposite — everyone panics and stops spending. This disturbs the normal flow of the economy.
- **What to look for:** GDP growth becomes much more volatile during the shock period. Inflation and unemployment swing more wildly. The interest rate shoots up or down as the rule tries to compensate.

### `trajectory(Supply_Shock).png`

This is the same economy, but between quarters 50 and 70 we injected a **supply shock** — a sudden change in production costs.

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

If you want to experiment, try changing these numbers and re-running the demo!

### `experiments/demo_economy.py`

This is the script you actually run. It does three things:

1. **Defines a policy rule** — a simple formula that looks at the current economy and decides what to do. It's basically a stripped-down Taylor Rule with two fiscal components: government spending and the tax rate.
2. **Runs the simulation** — loops for 200 quarters, applying the rule each step.
3. **Draws the charts** — uses Matplotlib to create the figures you see in `outputs/figures/`.

It also exports the data as JSON so Manim can animate it later.

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

## 📝 Requirements

The project uses a Python virtual environment. Key packages:

- `numpy`, `scipy`, `pandas` — math and data handling
- `matplotlib`, `seaborn` — static charts
- `manim` — animations (optional)
- `torch`, `stable-baselines3`, `gymnasium` — for later RL training

All dependencies are listed in `requirements.txt`.

---

## 🤝 Tips for Playing Around

- **Change the policy rule:** Open `experiments/demo_economy.py` and tweak the numbers in `taylor_policy()`. What happens if the central bank is more aggressive? More passive? What if the government raises taxes faster when debt is high?
- **Change the economy:** Open `src/utils/config.py` and tweak `alpha_1`, `beta`, `gamma`, etc. What makes the economy more stable? More volatile?
- **Change the shocks:** In `demo_economy.py`, modify when the demand/supply shocks happen or how strong they are.

The best way to learn is to break things and see what happens!

---

*Built for a demo project on Learning Macroeconomic Policy via Reinforcement Learning.*
