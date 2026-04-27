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
    alpha_1: float = 0.15     # fiscal multiplier on lagged spending
    alpha_2: float = 0.10     # interest-rate sensitivity of demand

    # --- Phillips Curve (eq 5) ---
    beta: float = 0.25        # slope of Phillips curve
    g_star: float = 2.5       # potential growth rate (%)

    # --- Adaptive Expectations (eq 6) ---
    rho: float = 0.7          # weight on current inflation vs past expectation

    # --- Okun's Law (eq 7) ---
    gamma: float = 0.15       # Okun coefficient

    # --- Debt Dynamics (eq 8) ---
    tau: float = 20.0         # initial tax revenue as % of GDP; matches G_0 by default

    # --- Shock standard deviations ---
    sigma_d: float = 1.0      # demand shock std
    sigma_s: float = 0.5      # supply shock std

    # --- Targets ---
    pi_star: float = 2.0      # inflation target (%)
    u_star: float = 5.0       # natural unemployment rate (%)
    d_star: float = 60.0      # sustainable debt-to-GDP (%)

    # --- Steady-state initialisation ---
    pi_0: float = 2.0
    u_0: float = 5.0
    g_0: float = 2.5
    r_0: float = 3.0          # nominal rate ≈ real rate + inflation
    d_0: float = 60.0
    E_pi_0: float = 2.0
    G_0: float = 20.0         # gov spending % of GDP (initial level)

    # --- Bounds (for clipping / visualisation) ---
    pi_bounds: Tuple[float, float] = (-5.0, 15.0)
    u_bounds: Tuple[float, float] = (0.0, 20.0)
    g_bounds: Tuple[float, float] = (-10.0, 15.0)
    r_bounds: Tuple[float, float] = (0.0, 10.0)
    d_bounds: Tuple[float, float] = (0.0, 200.0)
    E_pi_bounds: Tuple[float, float] = (-5.0, 15.0)
    tau_bounds: Tuple[float, float] = (0.0, 50.0)      # tax rate % of GDP
    delta_tau_bounds: Tuple[float, float] = (-2.0, 2.0)  # max quarterly tax change (pp)

    # --- Episode settings ---
    dt: float = 0.25          # one step = one quarter
    max_steps: int = 200      # 50 years
    debt_limit: float = 150.0 # early termination if exceeded

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RewardConfig:
    """Weights for the quadratic welfare objective (eq 9)."""

    w1: float = 1.0   # inflation aversion
    w2: float = 1.0   # unemployment aversion
    w3: float = 1.0   # growth preference
    w4: float = 0.05  # debt penalty

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

    # Softer monetary response reduces over-reaction and clipping.
    phi_pi: float = 0.6
    phi_u: float = 0.1

    # Mild counter-cyclical fiscal response with debt stabilization.
    psi_g: float = 0.12
    psi_d: float = 0.02
    psi_tau: float = 0.02

    # Action bounds for the baseline controller.
    delta_r_bounds: Tuple[float, float] = (-2.5, 2.5)
    delta_G_bounds: Tuple[float, float] = (-5.0, 5.0)
    delta_tau_bounds: Tuple[float, float] = (-1.5, 1.5)
