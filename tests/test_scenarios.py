"""Tests for scenario switchboard: loading YAML scenarios, executing overrides, and comparison matrix."""

import datetime as dt
import pandas as pd
import pytest

from parity.deal import DealAnalysisResult
from parity.scenarios import load_scenarios, run_scenario, run_scenario_switchboard
from parity.schemas import DealConfig, RepatriationRateMode


@pytest.fixture
def sample_deal_df():
    return pd.DataFrame({
        "period": [0, 12, 24, 36],
        "equity_invested_lcu": [100.0, 0.0, 0.0, 0.0],
        "distribution_lcu": [0.0, 5.0, 5.0, 0.0],
        "exit_proceeds_lcu": [0.0, 0.0, 0.0, 140.0],
    })


@pytest.fixture
def sample_deal_config():
    return DealConfig(
        country="Nigeria",
        close_date=dt.date(2026, 1, 31),
        hurdle_usd_irr=0.15,
        repatriation_rate=RepatriationRateMode.OFFICIAL,
        blocked_funds_haircut_pct=0.0,
        withholding_tax_pct=0.0,
    )


class TestScenariosLoadingAndExecution:
    def test_load_all_scenarios_from_yaml(self):
        scenarios = load_scenarios()
        expected = [
            "base",
            "imf_program_lost",
            "premium_widens",
            "step_devaluation",
            "slow_crawl",
            "float_and_overshoot",
        ]
        for name in expected:
            assert name in scenarios, f"Missing scenario '{name}' in config/scenarios.yaml"

    def test_run_scenario_step_devaluation_impact(self, sample_deal_df, sample_deal_config):
        base_features = {
            "parallel_premium": 0.05,
            "reserves_cover": 4.5,
            "reserves_change_3m": 0.0,
            "ca_pct_gdp": -3.0,
            "rer_misalignment": 0.0,
            "rate_differential": 8.0,
            "imf_program": 1,
            "imf_review_off_track": 0,
        }
        res_base = run_scenario(
            scenario_name="base",
            deal_df=sample_deal_df,
            deal_config=sample_deal_config,
            base_features=base_features,
            spot_rate=100.0,
            n_paths=500,
            seed=42,
        )
        res_step = run_scenario(
            scenario_name="step_devaluation",
            deal_df=sample_deal_df,
            deal_config=sample_deal_config,
            base_features=base_features,
            spot_rate=100.0,
            n_paths=500,
            seed=42,
        )

        assert isinstance(res_base, DealAnalysisResult)
        assert isinstance(res_step, DealAnalysisResult)
        # Step devaluation (-30% at month 12) strictly reduces USD return
        assert res_step.usd_irr_mean < res_base.usd_irr_mean
        assert res_step.usd_moic_mean < res_base.usd_moic_mean
        assert res_step.p_below_hurdle >= res_base.p_below_hurdle

    def test_run_scenario_switchboard_table(self, sample_deal_df, sample_deal_config):
        selected = ["base", "imf_program_lost", "premium_widens", "slow_crawl"]
        table_df, results_map = run_scenario_switchboard(
            deal_df=sample_deal_df,
            deal_config=sample_deal_config,
            spot_rate=100.0,
            selected_scenarios=selected,
            n_paths=300,
            seed=42,
        )

        assert len(table_df) == len(selected)
        assert set(results_map.keys()) == set(selected)

        required_cols = [
            "scenario",
            "description",
            "headline_lcu_irr",
            "usd_irr_mean",
            "usd_irr_median",
            "usd_moic_mean",
            "p_below_hurdle",
            "expected_fx_drag",
            "breakeven_lcu_irr_50",
            "breakeven_lcu_irr_75",
        ]
        for col in required_cols:
            assert col in table_df.columns, f"Missing column: {col}"

        # All headline LCU IRRs must be identical (deal cash flows did not change)
        assert table_df["headline_lcu_irr"].nunique() == 1
