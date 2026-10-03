"""Tests for devaluation probability model: priors, prediction, fitting, monotonicity, and evaluation."""

import numpy as np
import pandas as pd
import pytest

from parity.devaluation import DevaluationModel, sigmoid
from parity.features import build_feature_matrix, generate_synthetic_panel
from parity.loaders import load_priors


class TestDevaluationModelPriors:
    def test_init_with_priors_12m(self):
        model = DevaluationModel(horizon=12)
        assert not model.is_fitted
        assert model.horizon == 12
        priors = load_priors()
        assert model.intercept == float(priors["devaluation_probability"]["intercept_12m"])
        assert model.coefficients["parallel_premium"] == float(
            priors["devaluation_probability"]["coef_parallel_premium"]
        )

    def test_init_with_priors_24m(self):
        model = DevaluationModel(horizon=24)
        assert model.horizon == 24
        priors = load_priors()
        assert model.intercept == float(priors["devaluation_probability"]["intercept_24m"])
        # 24m baseline probability is higher than 12m baseline probability
        assert model.intercept > DevaluationModel(horizon=12).intercept

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon must be 12 or 24"):
            DevaluationModel(horizon=36)


class TestDevaluationPredictionsAndMonotonicity:
    @pytest.fixture
    def base_features(self):
        return {
            "parallel_premium": 0.05,
            "reserves_cover": 4.0,
            "reserves_change_3m": 0.0,
            "ca_pct_gdp": -3.0,
            "rer_misalignment": 0.0,
            "rate_differential": 8.0,
            "imf_program": 1,
            "imf_review_off_track": 0,
        }

    def test_predict_proba_dict_and_dataframe(self, base_features):
        model = DevaluationModel(horizon=12)
        p_scalar = model.predict_proba(base_features)
        assert isinstance(p_scalar, float)
        assert 0.0 < p_scalar < 1.0

        df = pd.DataFrame([base_features, base_features])
        p_array = model.predict_proba(df)
        assert isinstance(p_array, np.ndarray)
        assert len(p_array) == 2
        assert np.isclose(p_array[0], p_scalar)

    def test_monotonicity_parallel_premium(self, base_features):
        """Monotonicity: higher parallel premium MUST raise devaluation probability."""
        model = DevaluationModel(horizon=12)
        low_prem = dict(base_features, parallel_premium=0.0)
        high_prem = dict(base_features, parallel_premium=0.30)
        p_low = model.predict_proba(low_prem)
        p_high = model.predict_proba(high_prem)
        assert p_high > p_low, f"Expected p_high ({p_high}) > p_low ({p_low})"

    def test_monotonicity_reserves_cover(self, base_features):
        """Monotonicity: lower reserves cover MUST raise devaluation probability."""
        model = DevaluationModel(horizon=12)
        ample_reserves = dict(base_features, reserves_cover=6.0)
        depleted_reserves = dict(base_features, reserves_cover=1.5)
        p_ample = model.predict_proba(ample_reserves)
        p_depleted = model.predict_proba(depleted_reserves)
        assert p_depleted > p_ample, f"Expected depleted ({p_depleted}) > ample ({p_ample})"

    def test_monotonicity_rer_misalignment(self, base_features):
        """Monotonicity: higher real exchange rate overvaluation raises probability."""
        model = DevaluationModel(horizon=12)
        fair = dict(base_features, rer_misalignment=0.0)
        overvalued = dict(base_features, rer_misalignment=0.25)
        assert model.predict_proba(overvalued) > model.predict_proba(fair)

    def test_monotonicity_imf_off_track(self, base_features):
        """Monotonicity: derailed IMF review raises devaluation probability."""
        model = DevaluationModel(horizon=12)
        on_track = dict(base_features, imf_program=1, imf_review_off_track=0)
        off_track = dict(base_features, imf_program=1, imf_review_off_track=1)
        assert model.predict_proba(off_track) > model.predict_proba(on_track)


class TestDevaluationModelFitting:
    @pytest.fixture
    def synthetic_dataset(self):
        panel_df, known_coefs = generate_synthetic_panel(
            n_countries=5, n_months=84, seed=42
        )
        enriched = build_feature_matrix(panel_df)
        return enriched, known_coefs

    def test_fit_recovers_expected_coefficient_signs(self, synthetic_dataset):
        """Success criterion: fit() recovers known coefficient directions on synthetic data."""
        df, known_coefs = synthetic_dataset
        model = DevaluationModel(horizon=12)
        model.fit(df, C=1.0, cv_splits=3)

        assert model.is_fitted
        assert model.fitted_coefficients is not None

        # Key expected signs
        # 1. Parallel premium is strongly positive
        assert model.fitted_coefficients["parallel_premium"] > 0
        # 2. Reserves cover is negative
        assert model.fitted_coefficients["reserves_cover"] < 0
        # 3. RER misalignment is positive
        assert model.fitted_coefficients["rer_misalignment"] > 0

    def test_evaluate_produces_brier_and_auc(self, synthetic_dataset):
        df, _ = synthetic_dataset
        model = DevaluationModel(horizon=12)
        model.fit(df, C=1.0)
        metrics = model.evaluate(df)

        assert "brier_score" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert metrics["roc_auc"] > 0.50, f"Expected AUC > 0.50, got {metrics['roc_auc']}"
        assert len(metrics["calibration_prob_true"]) > 0

    def test_sigmoid_numerical_stability(self):
        assert sigmoid(0.0) == 0.5
        assert np.isclose(sigmoid(100.0), 1.0)
        assert np.isclose(sigmoid(-100.0), 0.0)
