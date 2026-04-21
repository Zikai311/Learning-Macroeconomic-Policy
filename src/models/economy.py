"""
Core macroeconomic state-transition dynamics.

Implements equations (4)-(8) from the proposal:
    g_t  = alpha_1 * G_{t-1} - alpha_2 * r_{t-1} + eps_d        (Aggregate Demand)
    pi_t = E_t pi_{t+1} + beta*(g_t - g_star) + eps_s           (Phillips Curve)
    E_t pi_{t+1} = rho*pi_t + (1-rho)*E_{t-1} pi_t              (Adaptive Expectations)
    u_t  = u_{t-1} - gamma * g_t                                 (Okun's Law)
    d_t  = d_{t-1} + G_t - T_t                                   (Debt Dynamics)

The class is stateful: call step(action) repeatedly.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

from src.utils.config import EconomyConfig, RewardConfig


class Economy:
    """Deterministic / stochastic macro-economy simulator."""

    def __init__(
        self,
        config: Optional[EconomyConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = config or EconomyConfig()
        self.rwd = reward_config or RewardConfig()
        self.rng = np.random.default_rng(seed)

        # --- internal state ---
        # We keep both the *observed* state s_t and the *lagged* variables
        # needed for the next transition (G_{t-1}, r_{t-1}, E_{t-1} pi_t).
        self.pi: float = 0.0
        self.u: float = 0.0
        self.g: float = 0.0
        self.r: float = 0.0
        self.d: float = 0.0
        self.E_pi: float = 0.0

        self.G_lag: float = 0.0   # G_{t-1}
        self.r_lag: float = 0.0   # r_{t-1}
        self.E_pi_lag: float = 0.0

        self.step_count: int = 0
        self.history: list = []   # stores dicts for post-hoc analysis / viz

        self.reset()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> Dict[str, float]:
        """Reset economy to initial steady-state (or near it with small noise)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        c = self.cfg
        noise = lambda scale: float(self.rng.normal(0.0, scale))

        self.pi       = c.pi_0 + noise(0.2)
        self.u        = c.u_0  + noise(0.2)
        self.g        = c.g_0  + noise(0.2)
        self.r        = c.r_0  + noise(0.1)
        self.d        = c.d_0  + noise(1.0)
        self.E_pi     = c.E_pi_0 + noise(0.2)

        self.G_lag    = c.G_0
        self.r_lag    = c.r_0
        self.E_pi_lag = c.E_pi_0

        self.step_count = 0
        self.history.clear()

        obs = self._obs_dict()
        self._record(obs, action=None, shocks=None, reward=0.0)
        return obs

    def step(
        self,
        action: Tuple[float, float],
        demand_shock: Optional[float] = None,
        supply_shock: Optional[float] = None,
    ) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        """
        Advance one time-step.

        Parameters
        ----------
        action : (delta_r, delta_G)
            delta_r  – change in nominal interest rate (pp)
            delta_G  – change in gov spending (% of GDP)
        demand_shock, supply_shock :
            Optional manual overrides.  If None, sampled from N(0, sigma).

        Returns
        -------
        obs   : dict of current observed state
        reward: scalar welfare
        done  : True if debt limit exceeded
        info  : dict with raw shocks and intermediate variables
        """
        c = self.cfg
        delta_r, delta_G = action

        # --- 1. update accumulated policy instruments -------------------
        self.r = np.clip(self.r_lag + delta_r, *c.r_bounds)
        G_t = np.clip(self.G_lag + delta_G, 0.0, 100.0)   # keep sensible

        # --- 2. sample shocks ------------------------------------------
        eps_d = demand_shock if demand_shock is not None else float(self.rng.normal(0.0, c.sigma_d))
        eps_s = supply_shock if supply_shock is not None else float(self.rng.normal(0.0, c.sigma_s))

        # --- 3. transition equations -----------------------------------
        # Eq 4: Aggregate Demand with lag
        self.g = c.alpha_1 * self.G_lag - c.alpha_2 * self.r_lag + eps_d
        self.g = np.clip(self.g, *c.g_bounds)

        # Eq 5: Phillips Curve (expectations-augmented)
        self.pi = self.E_pi_lag + c.beta * (self.g - c.g_star) + eps_s
        self.pi = np.clip(self.pi, *c.pi_bounds)

        # Eq 6: Adaptive Expectations
        self.E_pi = c.rho * self.pi + (1.0 - c.rho) * self.E_pi_lag
        self.E_pi = np.clip(self.E_pi, *c.E_pi_bounds)

        # Eq 7: Okun's Law  (use previous-period unemployment)
        # When g_t > 0 (boom), unemployment falls: u_t = u_{t-1} - gamma * (g_t - g_star)
        u_prev = self.u
        self.u = u_prev - c.gamma * (self.g - c.g_star)
        self.u = np.clip(self.u, *c.u_bounds)

        # Eq 8: Debt Dynamics  (simplified: T_t = tau, constant % of GDP)
        # d_t = d_{t-1} + G_t - tau
        self.d = self.d + G_t - c.tau
        self.d = np.clip(self.d, *c.d_bounds)

        # --- 4. update lags --------------------------------------------
        self.r_lag = self.r
        self.G_lag = G_t
        self.E_pi_lag = self.E_pi

        # --- 5. reward -------------------------------------------------
        reward = self.rwd.compute(self.pi, self.u, self.g, self.d)

        # --- 6. termination --------------------------------------------
        self.step_count += 1
        done = self.d > c.debt_limit
        truncated = self.step_count >= c.max_steps

        info = {
            "eps_d": eps_d,
            "eps_s": eps_s,
            "G_t": G_t,
            "delta_r": delta_r,
            "delta_G": delta_G,
            "step": self.step_count,
        }

        obs = self._obs_dict()
        self._record(obs, action=action, shocks=(eps_d, eps_s), reward=reward)
        return obs, reward, done or truncated, info

    def get_history(self) -> list:
        """Return list of recorded time-steps for plotting / animation."""
        return self.history

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _obs_dict(self) -> Dict[str, float]:
        return {
            "pi": self.pi,
            "u": self.u,
            "g": self.g,
            "r": self.r,
            "d": self.d,
            "E_pi": self.E_pi,
        }

    def _record(
        self,
        obs: Dict[str, float],
        action: Optional[Tuple[float, float]],
        shocks: Optional[Tuple[float, float]],
        reward: float,
    ) -> None:
        entry = {
            "step": self.step_count,
            **obs,
            "reward": reward,
        }
        if action is not None:
            entry["delta_r"] = action[0]
            entry["delta_G"] = action[1]
        if shocks is not None:
            entry["eps_d"] = shocks[0]
            entry["eps_s"] = shocks[1]
        self.history.append(entry)
