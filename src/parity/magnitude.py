"""Devaluation magnitude model estimating discrete jump size distributions."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from parity.loaders import load_priors


class MagnitudeModel:
    """Estimates the discrete jump size distribution given a devaluation event.

    The model anchors expected devaluation magnitude on the parallel market premium,
    macro reserve cover, and exchange rate regime (peg, managed crawl, or float).
    It parameterizes jump sizes via a Lognormal (or Gamma) distribution.

    Parameters
    ----------
    priors_path : Optional[Union[str, Path]], default None
        Path to priors.yaml. If None, default config is loaded.
    distribution : str, default 'lognormal'
        Distribution family for devaluation jump size ('lognormal' or 'gamma').
    """

    def __init__(
        self,
        priors_path: Optional[Union[str, Path]] = None,
        distribution: str = "lognormal",
    ):
        if distribution not in ("lognormal", "gamma"):
            raise ValueError(f"distribution must be 'lognormal' or 'gamma'; got '{distribution}'")
        self.distribution = distribution
        self.priors_path = priors_path
        self._load_priors()
        self.is_fitted: bool = False
        self.fitted_params_: Optional[Dict[str, float]] = None

    def _load_priors(self) -> None:
        """Load prior parameters from priors.yaml."""
        priors = load_priors(self.priors_path)
        mag_priors = priors.get("magnitude_model", {})

        self.premium_passthrough_mean: float = float(
            mag_priors.get("premium_passthrough_mean", 0.70)
        )
        self.peg_break_base_jump: float = float(
            mag_priors.get("peg_break_base_jump", 0.35)
        )
        self.managed_crawl_base_jump: float = float(
            mag_priors.get("managed_crawl_base_jump", 0.20)
        )
        self.float_overshoot_base_jump: float = float(
            mag_priors.get("float_overshoot_base_jump", 0.40)
        )
        self.shape_param_lognormal_sigma: float = float(
            mag_priors.get("shape_param_lognormal_sigma", 0.30)
        )

    def _get_base_jump(self, regime: str) -> float:
        """Return base jump parameter by exchange rate regime."""
        clean_regime = regime.strip().lower()
        if clean_regime in ("peg_break", "peg"):
            return self.peg_break_base_jump
        elif clean_regime in ("float", "float_and_overshoot"):
            return self.float_overshoot_base_jump
        elif clean_regime in ("managed_crawl_change", "managed", "crawl"):
            return self.managed_crawl_base_jump
        else:
            # Default general devaluation baseline
            return 0.25

    def get_expected_jump(
        self,
        parallel_premium: float = 0.0,
        reserves_cover: float = 3.5,
        regime: str = "devaluation",
    ) -> float:
        """Compute expected (mean) jump magnitude given macro state.

        Parameters
        ----------
        parallel_premium : float, default 0.0
            Parallel market premium spread in decimal (e.g. 0.20 for 20%).
        reserves_cover : float, default 3.5
            Gross reserves in months of import cover.
        regime : str, default 'devaluation'
            Regime identifier: 'peg_break', 'float', 'managed_crawl_change', or 'devaluation'.

        Returns
        -------
        float
            Expected mean jump percentage (e.g. 0.30 = 30% devaluation).
        """
        base_jump = self._get_base_jump(regime)
        passthrough_jump = float(self.premium_passthrough_mean * max(0.0, parallel_premium))

        # Anchor jump: closes fraction of parallel gap or regime base
        anchor = max(base_jump, passthrough_jump)

        # Reserves buffer factor: reserves < 3 months amplify jump size; > 4.5 months buffer it
        if reserves_cover < 3.0:
            reserves_factor = 1.0 + (3.0 - max(0.0, reserves_cover)) * 0.08
        elif reserves_cover > 4.5:
            reserves_factor = max(0.80, 1.0 - (reserves_cover - 4.5) * 0.04)
        else:
            reserves_factor = 1.0

        expected_jump = anchor * reserves_factor
        return float(np.clip(expected_jump, 0.05, 1.50))

    def estimate_magnitude(
        self,
        parallel_premium: float = 0.0,
        reserves_cover: float = 3.5,
        regime: str = "devaluation",
    ) -> Dict[str, float]:
        """Estimate mean, median, P10, and P90 percentiles of devaluation jump size.

        Parameters
        ----------
        parallel_premium : float, default 0.0
            Parallel market premium spread.
        reserves_cover : float, default 3.5
            Reserves in months of imports.
        regime : str, default 'devaluation'
            Regime identifier.

        Returns
        -------
        Dict[str, float]
            Dictionary containing mean, median, p10, p90, and dispersion parameters.
        """
        mean_jump = self.get_expected_jump(
            parallel_premium=parallel_premium,
            reserves_cover=reserves_cover,
            regime=regime,
        )
        sigma = self.shape_param_lognormal_sigma

        if self.distribution == "lognormal":
            # For lognormal: E[X] = exp(mu + sigma^2 / 2) -> mu = ln(E[X]) - sigma^2 / 2
            mu_log = np.log(mean_jump) - (sigma ** 2) / 2.0
            dist = stats.lognorm(s=sigma, scale=np.exp(mu_log))

            median_jump = float(np.exp(mu_log))
            p10_jump = float(dist.ppf(0.10))
            p90_jump = float(dist.ppf(0.90))
        else:
            # Gamma distribution parameterized by shape k and scale theta
            # mean = k * theta, var = k * theta^2
            variance = (mean_jump * sigma) ** 2
            k = (mean_jump ** 2) / variance
            theta = variance / mean_jump
            dist = stats.gamma(a=k, scale=theta)

            median_jump = float(dist.median())
            p10_jump = float(dist.ppf(0.10))
            p90_jump = float(dist.ppf(0.90))

        return {
            "mean": mean_jump,
            "median": median_jump,
            "p10": p10_jump,
            "p90": p90_jump,
            "dispersion_sigma": sigma,
        }

    def sample(
        self,
        n_samples: int,
        parallel_premium: float = 0.0,
        reserves_cover: float = 3.5,
        regime: str = "devaluation",
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample discrete jump magnitudes from the calibrated distribution.

        Parameters
        ----------
        n_samples : int
            Number of draws.
        parallel_premium : float, default 0.0
            Parallel market premium.
        reserves_cover : float, default 3.5
            Reserves cover.
        regime : str, default 'devaluation'
            Regime.
        rng : Optional[np.random.Generator], default None
            RNG generator for reproducibility.

        Returns
        -------
        np.ndarray
            Array of positive jump magnitudes (e.g. 0.32 = +32%).
        """
        if rng is None:
            rng = np.random.default_rng()

        mean_jump = self.get_expected_jump(
            parallel_premium=parallel_premium,
            reserves_cover=reserves_cover,
            regime=regime,
        )
        sigma = self.shape_param_lognormal_sigma

        if self.distribution == "lognormal":
            mu_log = np.log(mean_jump) - (sigma ** 2) / 2.0
            draws = rng.lognormal(mean=mu_log, sigma=sigma, size=n_samples)
        else:
            variance = (mean_jump * sigma) ** 2
            k = (mean_jump ** 2) / variance
            theta = variance / mean_jump
            draws = rng.gamma(shape=k, scale=theta, size=n_samples)

        return np.clip(draws, 0.02, 3.0)

    def fit(self, events_df: pd.DataFrame) -> "MagnitudeModel":
        """Fit empirical jump distribution from historical events if available.

        Parameters
        ----------
        events_df : pd.DataFrame
            DataFrame containing 'size_pct' from fx_events.csv.

        Returns
        -------
        MagnitudeModel
            Self instance.
        """
        if "size_pct" not in events_df.columns:
            warnings.warn("No 'size_pct' column in events_df. Retaining documented priors.", UserWarning)
            return self

        valid_sizes = events_df["size_pct"].dropna().to_numpy(dtype=float)
        valid_sizes = valid_sizes[valid_sizes > 0.0]

        if len(valid_sizes) < 3:
            warnings.warn(
                f"Only {len(valid_sizes)} historical jump size observations. Retaining documented priors.",
                UserWarning,
            )
            return self

        mean_empirical = float(np.mean(valid_sizes))
        log_sizes = np.log(valid_sizes)
        sigma_empirical = float(np.std(log_sizes, ddof=1)) if len(valid_sizes) > 1 else self.shape_param_lognormal_sigma

        self.peg_break_base_jump = mean_empirical
        self.shape_param_lognormal_sigma = max(0.15, min(0.60, sigma_empirical))
        self.is_fitted = True
        self.fitted_params_ = {
            "mean_jump": mean_empirical,
            "sigma": self.shape_param_lognormal_sigma,
            "n_events": len(valid_sizes),
        }
        return self
