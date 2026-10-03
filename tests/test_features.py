"""Tests for feature engineering: parallel premium, RER misalignment, macro indicators, and targets."""

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from parity.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_feature_matrix,
    compute_devaluation_events,
    compute_forward_targets,
    compute_imf_review_off_track,
    compute_parallel_premium,
    compute_rate_differential,
    compute_rer_misalignment,
    generate_synthetic_panel,
)
from parity.schemas import FXPanelRow


class TestParallelPremium:
    def test_parallel_premium_standard(self):
        official = pd.Series([100.0, 100.0, 100.0])
        parallel = pd.Series([115.0, 130.0, 100.0])
        premium = compute_parallel_premium(official, parallel)
        assert np.isclose(premium.iloc[0], 0.15)
        assert np.isclose(premium.iloc[1], 0.30)
        assert np.isclose(premium.iloc[2], 0.00)

    def test_parallel_premium_missing_warns_and_falls_back(self):
        official = pd.Series([100.0, 110.0])
        parallel = pd.Series([np.nan, np.nan])
        with pytest.warns(UserWarning, match="parallel_rate is entirely missing"):
            premium = compute_parallel_premium(official, parallel, fallback=0.0)
        assert (premium == 0.0).all()

    def test_parallel_premium_partial_missing_warns(self):
        official = pd.Series([100.0, 100.0])
        parallel = pd.Series([120.0, np.nan])
        with pytest.warns(UserWarning, match="parallel_rate is missing in 1 rows"):
            premium = compute_parallel_premium(official, parallel, fallback=0.0)
        assert np.isclose(premium.iloc[0], 0.20)
        assert np.isclose(premium.iloc[1], 0.00)

    def test_parallel_premium_clipped_at_zero(self):
        official = pd.Series([100.0])
        parallel = pd.Series([95.0])  # Parallel below official
        premium = compute_parallel_premium(official, parallel)
        assert premium.iloc[0] == 0.0


class TestRERMisalignment:
    def test_rer_misalignment_calculation(self):
        # 3 periods: base date, period 1 (high local inflation, fixed official rate),
        # period 2 (official devalues)
        dates = [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 3, 31)]
        df = pd.DataFrame({
            "country": ["Kenya", "Kenya", "Kenya"],
            "date": dates,
            "official_rate": [100.0, 100.0, 120.0],
            "inflation_local_pct": [12.0, 12.0, 12.0],  # 1% per month
            "inflation_us_pct": [0.0, 0.0, 0.0],
        })
        misalign = compute_rer_misalignment(df)
        assert np.isclose(misalign.iloc[0], 0.0)
        # Period 1: local inf diff = 12 / 1200 = 0.01; nominal deprec = 0; misalign = 0.01 (overvalued)
        assert misalign.iloc[1] > 0.0
        # Period 2: nominal deprec = 20/100 = 0.20; cumulative inf diff = 0.02; misalign = -0.18 (undervalued)
        assert misalign.iloc[2] < 0.0


class TestMacroIndicators:
    def test_rate_differential(self):
        policy = pd.Series([14.5, 12.0])
        us = pd.Series([4.5, 5.0])
        diff = compute_rate_differential(policy, us)
        assert np.isclose(diff.iloc[0], 10.0)
        assert np.isclose(diff.iloc[1], 7.0)

    def test_imf_review_off_track(self):
        prog = pd.Series([1, 1, 0, 1])
        review = pd.Series([0, 1, 0, None])
        off_track = compute_imf_review_off_track(prog, review)
        # Only active program with review=0 should be 1
        assert list(off_track) == [1, 0, 0, 0]


class TestDevaluationEventsAndTargets:
    def test_devaluation_event_detection(self):
        dates = pd.date_range("2025-01-31", periods=6, freq="ME").date
        # Month 3: jumps from 100 to 120 (+20% > 15% threshold)
        rates = [100.0, 101.0, 102.0, 123.0, 124.0, 125.0]
        df = pd.DataFrame({
            "country": ["Nigeria"] * 6,
            "date": dates,
            "official_rate": rates,
        })
        events = compute_devaluation_events(df, monthly_threshold=0.15, three_month_threshold=0.25)
        assert events.iloc[3] == 1
        assert events.iloc[0] == 0
        assert events.iloc[1] == 0

    def test_external_events_overlay(self):
        dates = [dt.date(2025, 1, 31), dt.date(2025, 2, 28)]
        panel_df = pd.DataFrame({
            "country": ["Egypt", "Egypt"],
            "date": dates,
            "official_rate": [30.0, 30.1],  # Calm rates
        })
        events_df = pd.DataFrame({
            "country": ["Egypt"],
            "date": [dt.date(2025, 2, 28)],
            "event_type": ["float"],
            "size_pct": [0.4],
        })
        events = compute_devaluation_events(panel_df, events_df=events_df)
        assert events.iloc[0] == 0
        assert events.iloc[1] == 1

    def test_forward_targets(self):
        dates = pd.date_range("2020-01-31", periods=36, freq="ME").date
        panel_df = pd.DataFrame({
            "country": ["Kenya"] * 36,
            "date": dates,
        })
        # Event occurs at month 15
        event_series = pd.Series(0, index=panel_df.index)
        event_series.iloc[15] = 1

        targets = compute_forward_targets(panel_df, event_series, horizons=(12, 24))
        # Month 10: month 15 is 5 months forward, within 12m horizon -> 1.0
        assert targets.loc[10, "target_devaluation_12m"] == 1.0
        # Month 1: month 15 is 14 months forward, outside 12m horizon -> 0.0
        assert targets.loc[1, "target_devaluation_12m"] == 0.0
        # Month 1: month 15 is within 24m horizon -> 1.0
        assert targets.loc[1, "target_devaluation_24m"] == 1.0
        # End of series without event: right-censored -> NaN
        assert pd.isna(targets.loc[30, "target_devaluation_12m"])


class TestFeatureMatrixAndSyntheticPanel:
    def test_build_feature_matrix_structure(self):
        synthetic_df, _ = generate_synthetic_panel(n_countries=2, n_months=36, seed=42)
        enriched = build_feature_matrix(synthetic_df)

        for col in FEATURE_COLUMNS:
            assert col in enriched.columns, f"Missing feature column: {col}"
            assert not enriched[col].isna().any(), f"NaNs found in feature: {col}"

        for col in TARGET_COLUMNS:
            assert col in enriched.columns, f"Missing target column: {col}"

        assert "is_devaluation_event" in enriched.columns

    def test_synthetic_panel_schema_validity(self):
        synthetic_df, known_coefs = generate_synthetic_panel(n_countries=2, n_months=24, seed=123)
        assert len(synthetic_df) == 48
        assert "parallel_premium" in known_coefs

        # Validate that each row meets FXPanelRow schema
        for _, row in synthetic_df.head(10).iterrows():
            row_dict = row.to_dict()
            FXPanelRow(**row_dict)
