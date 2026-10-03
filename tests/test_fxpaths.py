"""Tests for Monte Carlo FX path simulator: shape, regimes, reproducibility, and scenarios."""

import numpy as np
import pytest

from parity.devaluation import DevaluationModel
from parity.fxpaths import SimulationResult, simulate_fx_paths
from parity.magnitude import MagnitudeModel


class TestFXPathSimulation:
    def test_simulation_shape_and_initial_condition(self):
        res = simulate_fx_paths(spot_rate=100.0, n_months=36, n_paths=500, seed=42)
        assert isinstance(res, SimulationResult)
        assert res.paths.shape == (500, 37)
        # All paths must start at spot rate
        assert np.all(res.paths[:, 0] == 100.0)
        assert len(res.terminal_rates) == 500
        assert len(res.annualized_depreciation) == 500

    def test_fixed_seed_reproducibility(self):
        """Success criterion: Simulations are strictly reproducible with a fixed seed."""
        res1 = simulate_fx_paths(spot_rate=1450.0, n_months=24, n_paths=1000, seed=12345)
        res2 = simulate_fx_paths(spot_rate=1450.0, n_months=24, n_paths=1000, seed=12345)
        res_diff_seed = simulate_fx_paths(spot_rate=1450.0, n_months=24, n_paths=1000, seed=99999)

        # Exact bitwise identity with identical seed
        assert np.array_equal(res1.paths, res2.paths)
        assert np.array_equal(res1.jump_counts, res2.jump_counts)

        # Different results with different seed
        assert not np.array_equal(res1.paths, res_diff_seed.paths)

    def test_summary_statistics_structure(self):
        res = simulate_fx_paths(spot_rate=130.0, n_months=48, n_paths=1000, seed=42)
        summ = res.summary()

        assert summ["spot_rate"] == 130.0
        assert summ["n_paths"] == 1000
        assert summ["n_months"] == 48
        assert summ["terminal_p10"] < summ["terminal_median"] < summ["terminal_p90"]
        assert summ["depreciation_p10_annual"] < summ["depreciation_median_annual"] < summ["depreciation_p90_annual"]
        assert 0.0 <= summ["p_at_least_one_devaluation"] <= 1.0

    def test_forced_step_devaluation(self):
        """Verify scenario switchboard feature: forced step devaluation at month 12."""
        spot = 100.0
        jump_size = 0.30
        res = simulate_fx_paths(
            spot_rate=spot,
            n_months=24,
            n_paths=500,
            forced_jump_month=12,
            forced_jump_size=jump_size,
            seed=42,
        )
        # Every path must experience at least one jump
        assert np.all(res.jump_counts >= 1)

        # Compare month 11 and month 12 rates: average jump should be >= 30%
        jump_ratios = res.paths[:, 12] / res.paths[:, 11] - 1.0
        assert np.mean(jump_ratios) >= 0.28

    def test_hazard_sensitivity(self):
        """Higher devaluation risk features must produce more jumps and higher terminal rates."""
        low_risk_features = {
            "parallel_premium": 0.0,
            "reserves_cover": 6.5,
            "reserves_change_3m": 0.05,
            "ca_pct_gdp": -1.0,
            "rer_misalignment": -0.05,
            "rate_differential": 8.0,
            "imf_program": 1,
            "imf_review_off_track": 0,
        }
        high_risk_features = {
            "parallel_premium": 0.35,
            "reserves_cover": 1.8,
            "reserves_change_3m": -0.10,
            "ca_pct_gdp": -6.0,
            "rer_misalignment": 0.25,
            "rate_differential": 2.0,
            "imf_program": 1,
            "imf_review_off_track": 1,
        }

        res_low = simulate_fx_paths(spot_rate=100.0, n_months=24, n_paths=1000, features_obs=low_risk_features, seed=42)
        res_high = simulate_fx_paths(spot_rate=100.0, n_months=24, n_paths=1000, features_obs=high_risk_features, seed=42)

        assert res_high.summary()["p_at_least_one_devaluation"] > res_low.summary()["p_at_least_one_devaluation"]
        assert res_high.summary()["terminal_mean"] > res_low.summary()["terminal_mean"]

    def test_forced_annual_crawl(self):
        """Verify slow_crawl scenario: forced 8% steady annual crawl with zero hazard."""
        # Devaluation model with zero probability
        class ZeroDevalModel:
            def predict_proba(self, *args, **kwargs):
                return 0.0001

        res = simulate_fx_paths(
            spot_rate=100.0,
            n_months=36,
            n_paths=500,
            devaluation_model=ZeroDevalModel(),  # type: ignore[arg-type]
            forced_annual_depreciation=0.08,
            calm_volatility_annual=0.001,  # negligible volatility
            seed=42,
        )
        ann_deprec = res.annualized_depreciation
        assert np.isclose(np.mean(ann_deprec), 0.08, atol=0.005)
