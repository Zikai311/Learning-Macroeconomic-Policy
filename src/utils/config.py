"""
Configuration for the macroeconomic simulation.
All parameters are stylized for the demo and can be overridden
via a dict passed to the Economy or Environment constructor.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EconomyConfig:
    """Parameters for the economic dynamics (equations 4-8)."""

    # --- Aggregate Demand (eq 4) ---
    alpha_1: float = 0.15     # How strongly last quarter's government spending lifts current growth.
    alpha_2: float = 0.15     # How strongly last quarter's interest rate suppresses current growth.

    # --- Phillips Curve (eq 5) ---
    beta: float = 0.20        # How sensitive inflation is to growth running above/below potential.
    g_star: float = 2.5       # Potential or trend growth rate; the "normal speed" of the economy (%).

    # --- Adaptive Expectations (eq 6) ---
    rho: float = 0.5          # Weight on current inflation when agents update expected future inflation.

    # --- Okun's Law (eq 7) ---
    gamma: float = 0.10       # How much unemployment moves when growth deviates from potential.

    # --- Debt Dynamics (eq 8) ---
    tau: float = 20.0         # Initial tax revenue / tax take (% of GDP); matches G_0 so debt is roughly stable at reset.

    # --- Shock standard deviations ---
    sigma_d: float = 0.12     # Standard deviation of demand shocks: unexpected consumption / investment swings.
    sigma_s: float = 0.10     # Standard deviation of supply shocks: unexpected cost / productivity disturbances.

    # --- Targets ---
    pi_star: float = 2.0      # Inflation target used in policy and reward (%).
    u_star: float = 5.0       # Natural / target unemployment rate used in policy and reward (%).
    d_star: float = 60.0      # Debt level viewed as fiscally sustainable in the reward (% of GDP).

    # --- Steady-state initialisation ---
    pi_0: float = 2.0         # Initial inflation around the policy target (%).
    u_0: float = 5.0          # Initial unemployment around its long-run benchmark (%).
    g_0: float = 2.5          # Initial GDP growth around potential (%).
    r_0: float = 3.0          # Initial nominal policy rate; roughly real rate + inflation target (%).
    d_0: float = 60.0         # Initial debt ratio (% of GDP).
    E_pi_0: float = 2.0       # Initial expected inflation (%).
    G_0: float = 20.0         # Initial government spending level (% of GDP).

    # --- Bounds (for clipping / visualisation) ---
    pi_bounds: Tuple[float, float] = (-5.0, 15.0)      # Inflation clipping range to avoid unrealistic explosions (%).
    u_bounds: Tuple[float, float] = (0.0, 20.0)        # Unemployment clipping range (% of labor force).
    g_bounds: Tuple[float, float] = (-10.0, 15.0)      # Growth clipping range (%).
    r_bounds: Tuple[float, float] = (0.0, 10.0)        # Policy-rate bounds; also encodes a zero lower bound (%).
    d_bounds: Tuple[float, float] = (0.0, 200.0)       # Debt ratio clipping range (% of GDP).
    E_pi_bounds: Tuple[float, float] = (-5.0, 15.0)    # Expected inflation clipping range (%).
    G_bounds: Tuple[float, float] = (0.0, 100.0)       # Government spending level bounds (% of GDP).
    tau_bounds: Tuple[float, float] = (0.0, 50.0)      # Total tax take bounds (% of GDP).
    delta_r_bounds: Tuple[float, float] = (-0.8, 0.8)  # Maximum quarter-to-quarter policy-rate adjustment (percentage points).
    delta_G_bounds: Tuple[float, float] = (-2.0, 2.0)  # Maximum quarter-to-quarter spending adjustment (% of GDP).
    delta_tau_bounds: Tuple[float, float] = (-2.0, 2.0)  # Maximum quarter-to-quarter tax adjustment (percentage points).

    # --- Episode settings ---
    dt: float = 0.25          # Time per step in years: 0.25 means one quarter.
    max_steps: int = 200      # Episode horizon in quarters; 200 = 50 years.
    debt_limit: float = 150.0 # Early-stop threshold if debt becomes dangerously high (% of GDP).

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RewardConfig:
    """Weights for the quadratic welfare objective (eq 9)."""

    w1: float = 1.0   # Penalty weight on inflation missing its target.
    w2: float = 1.0   # Penalty weight on unemployment missing its target.
    w3: float = 1.0   # Reward weight on growth running above/below potential.
    w4: float = 0.05  # Penalty weight on debt being above its sustainable benchmark.

    def compute(
        self,
        pi: float,
        u: float,
        g: float,
        d: float,
        *,
        pi_star: float,
        u_star: float,
        g_star: float,
        d_star: float,
    ) -> float:
        """Quadratic welfare (deviations from target):
        r_t = -w1*(pi-pi*)^2 - w2*(u-u*)^2 + w3*(g-g*) - w4*(d-d*)
        """
        return -(self.w1 * (pi - pi_star) ** 2 +
                 self.w2 * (u - u_star) ** 2) + \
                self.w3 * (g - g_star) - \
                self.w4 * (d - d_star)


@dataclass
class TaylorPolicyConfig:
    """Coefficients for the hand-written baseline policy used in the demo."""

    # Firmer inflation targeting without returning to the earlier hair-trigger rule.
    phi_pi: float = 0.8       # How aggressively the central bank raises/cuts rates when inflation misses target.
    phi_u: float = 0.06       # How strongly the central bank reacts to unemployment gaps.

    # More visible counter-cyclical fiscal response with debt stabilization.
    psi_g: float = 1.0        # How strongly spending responds against the business cycle (higher in slumps, lower in booms).
    psi_d: float = 0.08       # How strongly spending is restrained when debt is high.
    psi_tau: float = 0.10     # How strongly taxes rise when debt is above target and fall when debt is below target.
