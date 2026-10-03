"""Tests for devaluation magnitude model: priors, anchoring, regimes, and sampling."""

import numpy as np
import pandas as pd
import pytest

from parity.loaders import load_priors
from parity.magnitude import MagnitudeModel


class TestMagnitudeModelPriorsAndRegimes:
    def test_priors_loading(self):
        model = MagnitudeModel()
        priors = load_priors()
        assert model.premium_passthrough_mean == float(
            priors["magnitude_model"]["premium_passthrough_mean"]
        )
        assert model.peg_break_base_jump == float(
            priors["magnitude_model"]["peg_break_base_jump"]
        )

    def test_regime_base_jumps(self):
        model = MagnitudeModel()
        # With zero parallel premium and neutral reserves (3.5)
        peg_jump = model.get_expected_jump(parallel_premium=0.0, reserves_cover=3.5, regime="peg_break")
        crawl_jump = model.get_expected_jump(parallel_premium=0.0, reserves_cover=3.5, regime="managed_crawl_change")
        float_jump = model.get_expected_jump(parallel_premium=0.0, reserves_cover=3.5, regime="float")

        assert np.isclose(peg_jump, 0.35)
        assert np.isclose(crawl_jump, 0.20)
        assert np.isclose(float_jump, 0.40)
        assert float_jump > peg_jump > crawl_jump

    def test_parallel_premium_anchoring(self):
        """Higher parallel premium must raise the expected jump magnitude."""
        model = MagnitudeModel()
        # When parallel premium is 80%, passthrough is 70% * 80% = 56%
        jump_high_prem = model.get_expected_jump(parallel_premium=0.80, reserves_cover=3.5, regime="managed_crawl_change")
        jump_zero_prem = model.get_expected_jump(parallel_premium=0.0, reserves_cover=3.5, regime="managed_crawl_change")

        assert jump_high_prem > jump_zero_prem
        assert np.isclose(jump_high_prem, 0.80 * 0.70)

    def test_reserves_cover_buffering(self):
        """Depleted reserves amplify jump size; ample reserves cushion jump size."""
        model = MagnitudeModel()
        depleted = model.get_expected_jump(parallel_premium=0.0, reserves_cover=1.5, regime="peg_break")
        normal = model.get_expected_jump(parallel_premium=0.0, reserves_cover=3.5, regime="peg_break")
        ample = model.get_expected_jump(parallel_premium=0.0, reserves_cover=6.0, regime="peg_break")

        assert depleted > normal > ample


class TestMagnitudeEstimationAndSampling:
    def test_estimate_magnitude_percentiles(self):
        model = MagnitudeModel(distribution="lognormal")
        est = model.estimate_magnitude(parallel_premium=0.25, reserves_cover=3.0, regime="devaluation")

        assert "mean" in est
        assert "median" in est
        assert "p10" in est
        assert "p90" in est
        assert est["p10"] < est["median"] < est["p90"]
        assert est["p10"] > 0

    def test_gamma_distribution_option(self):
        model = MagnitudeModel(distribution="gamma")
        est = model.estimate_magnitude(parallel_premium=0.20, reserves_cover=3.5)
        assert est["p10"] < est["median"] < est["p90"]

    def test_sample_reproducibility_and_bounds(self):
        model = MagnitudeModel()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        draws1 = model.sample(n_samples=500, parallel_premium=0.20, rng=rng1)
        draws2 = model.sample(n_samples=500, parallel_premium=0.20, rng=rng2)

        assert np.array_equal(draws1, draws2)
        assert (draws1 > 0.0).all()
        assert np.mean(draws1) > 0.15

    def test_fit_from_events(self):
        model = MagnitudeModel()
        events_df = pd.DataFrame({
            "country": ["Nigeria", "Egypt", "Ghana", "Kenya"],
            "size_pct": [0.36, 0.38, 0.45, 0.25],
        })
        model.fit(events_df)
        assert model.is_fitted
        assert model.fitted_params_ is not None
        assert np.isclose(model.peg_break_base_jump, np.mean([0.36, 0.38, 0.45, 0.25]))
