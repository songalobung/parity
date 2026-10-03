"""Monte Carlo FX path simulator with calm and jump regimes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from parity.devaluation import DevaluationModel
from parity.loaders import load_priors
from parity.magnitude import MagnitudeModel


@dataclass
class SimulationResult:
    """Encapsulates simulated FX trajectories and distribution summaries.

    Attributes
    ----------
    paths : np.ndarray
        Array of shape (n_paths, n_months + 1) containing exchange rate levels.
    spot_rate : float
        Starting exchange rate (LCU per USD) at period 0.
    n_months : int
        Number of forward simulation months.
    n_paths : int
        Number of simulated paths.
    seed : Optional[int]
        Random seed used for generation.
    jump_counts : np.ndarray
        Array of shape (n_paths,) containing number of devaluation jumps per path.
    """

    paths: np.ndarray
    spot_rate: float
    n_months: int
    n_paths: int
    seed: Optional[int]
    jump_counts: np.ndarray

    @property
    def terminal_rates(self) -> np.ndarray:
        """Terminal exchange rate at month T across all paths."""
        return self.paths[:, -1]

    @property
    def annualized_depreciation(self) -> np.ndarray:
        """Annualized currency depreciation rate per path: (S_T / S_0)^(12 / T) - 1."""
        if self.n_months == 0:
            return np.zeros(self.n_paths, dtype=float)
        years = self.n_months / 12.0
        return (self.terminal_rates / self.spot_rate) ** (1.0 / years) - 1.0

    def summary(self) -> Dict[str, Any]:
        """Compute key summary statistics of the simulation distributions."""
        term = self.terminal_rates
        ann_deprec = self.annualized_depreciation

        return {
            "spot_rate": self.spot_rate,
            "n_paths": self.n_paths,
            "n_months": self.n_months,
            "terminal_mean": float(np.mean(term)),
            "terminal_median": float(np.median(term)),
            "terminal_p10": float(np.percentile(term, 10)),
            "terminal_p90": float(np.percentile(term, 90)),
            "depreciation_mean_annual": float(np.mean(ann_deprec)),
            "depreciation_median_annual": float(np.median(ann_deprec)),
            "depreciation_p10_annual": float(np.percentile(ann_deprec, 10)),
            "depreciation_p90_annual": float(np.percentile(ann_deprec, 90)),
            "p_at_least_one_devaluation": float(np.mean(self.jump_counts > 0)),
            "mean_jump_count": float(np.mean(self.jump_counts)),
        }


def simulate_fx_paths(
    spot_rate: float,
    n_months: int = 60,
    n_paths: int = 10000,
    devaluation_model: Optional[DevaluationModel] = None,
    magnitude_model: Optional[MagnitudeModel] = None,
    features_obs: Optional[Dict[str, float]] = None,
    inflation_differential_annual: float = 0.08,
    regime: str = "devaluation",
    calm_volatility_annual: Optional[float] = None,
    ppp_passthrough_drift: Optional[float] = None,
    post_jump_retracement: Optional[float] = None,
    adjustment_months: Optional[int] = None,
    forced_jump_month: Optional[int] = None,
    forced_jump_size: Optional[float] = None,
    forced_annual_depreciation: Optional[float] = None,
    seed: Optional[int] = 42,
    priors_path: Optional[Union[str, Path]] = None,
) -> SimulationResult:
    """Simulate Monte Carlo foreign exchange rate paths with calm and jump regimes.

    Parameters
    ----------
    spot_rate : float
        Current spot exchange rate (LCU per USD).
    n_months : int, default 60
        Forecast horizon in months.
    n_paths : int, default 10000
        Number of Monte Carlo paths.
    devaluation_model : Optional[DevaluationModel], default None
        Fitted or prior devaluation hazard model.
    magnitude_model : Optional[MagnitudeModel], default None
        Jump size model.
    features_obs : Optional[Dict[str, float]], default None
        Observation dictionary of current macro features.
    inflation_differential_annual : float, default 0.08
        Annual domestic minus US inflation rate in decimal (e.g. 0.08 = 8%).
    regime : str, default 'devaluation'
        Currency regime ('peg_break', 'float', 'managed_crawl_change', 'devaluation').
    calm_volatility_annual : Optional[float], default None
        Annualized volatility during normal crawl (loaded from priors if None).
    ppp_passthrough_drift : Optional[float], default None
        Fraction of inflation differential driving normal crawl (from priors if None).
    post_jump_retracement : Optional[float], default None
        Fraction of jump retraced post-jump (from priors if None).
    adjustment_months : Optional[int], default None
        Post-jump adjustment duration in months (from priors if None).
    forced_jump_month : Optional[int], default None
        Month index for forced discrete devaluation (used by scenarios).
    forced_jump_size : Optional[float], default None
        Magnitude of forced devaluation jump.
    forced_annual_depreciation : Optional[float], default None
        Fixed annual crawl rate overriding PPP drift.
    seed : Optional[int], default 42
        RNG seed for exact reproducibility.
    priors_path : Optional[Union[str, Path]], default None
        Path to priors.yaml.

    Returns
    -------
    SimulationResult
        Simulated paths matrix and distribution summaries.
    """
    if spot_rate <= 0:
        raise ValueError(f"spot_rate must be positive; got {spot_rate}")
    if n_months <= 0:
        raise ValueError(f"n_months must be positive; got {n_months}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive; got {n_paths}")

    # Load priors for defaults
    priors = load_priors(priors_path)
    sim_priors = priors.get("fx_path_simulation", {})

    if calm_volatility_annual is None:
        calm_volatility_annual = float(sim_priors.get("calm_volatility_annual", 0.06))
    if ppp_passthrough_drift is None:
        ppp_passthrough_drift = float(sim_priors.get("ppp_passthrough_drift", 0.85))
    if post_jump_retracement is None:
        post_jump_retracement = float(sim_priors.get("post_jump_retracement", 0.15))
    if adjustment_months is None:
        adjustment_months = int(sim_priors.get("adjustment_months", 6))

    # Initialize models if not provided
    if devaluation_model is None:
        devaluation_model = DevaluationModel(horizon=12, priors_path=priors_path)
    if magnitude_model is None:
        magnitude_model = MagnitudeModel(priors_path=priors_path)

    features = features_obs or {}
    parallel_prem = float(features.get("parallel_premium", 0.0))
    reserves_cov = float(features.get("reserves_cover", 3.5))

    # Calculate monthly hazard from 12-month probability
    # P_12m = 1 - (1 - lambda_m)^12  =>  lambda_m = 1 - (1 - P_12m)^(1/12)
    p_12m = float(devaluation_model.predict_proba(features))
    p_12m = np.clip(p_12m, 0.001, 0.999)
    monthly_hazard = float(1.0 - (1.0 - p_12m) ** (1.0 / 12.0))

    # Monthly drift and volatility for calm regime
    if forced_annual_depreciation is not None:
        monthly_drift = forced_annual_depreciation / 12.0
    else:
        monthly_drift = (ppp_passthrough_drift * inflation_differential_annual) / 12.0

    monthly_vol = calm_volatility_annual / np.sqrt(12.0)

    # Setup random number generator
    rng = np.random.default_rng(seed)

    # Initialize paths array: shape (n_paths, n_months + 1)
    paths = np.empty((n_paths, n_months + 1), dtype=float)
    paths[:, 0] = spot_rate

    jump_counts = np.zeros(n_paths, dtype=int)
    # Track remaining adjustment months for each path
    adjusting_months_left = np.zeros(n_paths, dtype=int)
    monthly_retrace_rate = post_jump_retracement / max(1, adjustment_months)

    # Run simulation forward month by month
    for m in range(1, n_months + 1):
        prev_rates = paths[:, m - 1]

        # 1. Normal calm diffusion shock
        z = rng.normal(0.0, 1.0, size=n_paths)
        calm_multipliers = np.exp((monthly_drift - 0.5 * (monthly_vol ** 2)) + monthly_vol * z)

        # 2. Check for devaluation jumps
        if forced_jump_month is not None and m == forced_jump_month:
            # Deterministic forced jump on all paths
            jump_size = forced_jump_size if forced_jump_size is not None else 0.30
            jump_occurred = np.ones(n_paths, dtype=bool)
            jump_draws = np.full(n_paths, jump_size, dtype=float)
        else:
            jump_occurred = rng.uniform(0.0, 1.0, size=n_paths) < monthly_hazard
            n_jumps = int(np.sum(jump_occurred))
            if n_jumps > 0:
                jump_draws = np.zeros(n_paths, dtype=float)
                jump_draws[jump_occurred] = magnitude_model.sample(
                    n_samples=n_jumps,
                    parallel_premium=parallel_prem,
                    reserves_cover=reserves_cov,
                    regime=regime,
                    rng=rng,
                )
            else:
                jump_draws = np.zeros(n_paths, dtype=float)

        # Update jump tracking
        jump_counts[jump_occurred] += 1
        adjusting_months_left[jump_occurred] = adjustment_months

        # 3. Compute rates for period m
        new_rates = prev_rates * calm_multipliers

        # Apply discrete jump on jumping paths
        new_rates[jump_occurred] *= (1.0 + jump_draws[jump_occurred])

        # Apply post-jump retracement dampener on paths currently adjusting (if not currently jumping)
        active_adjust = (adjusting_months_left > 0) & (~jump_occurred)
        if np.any(active_adjust):
            # Retracement dampens exchange rate level slightly
            new_rates[active_adjust] *= (1.0 - monthly_retrace_rate)
            adjusting_months_left[active_adjust] -= 1

        paths[:, m] = new_rates

    return SimulationResult(
        paths=paths,
        spot_rate=spot_rate,
        n_months=n_months,
        n_paths=n_paths,
        seed=seed,
        jump_counts=jump_counts,
    )
