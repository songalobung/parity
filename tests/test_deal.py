"""Tests for deal engine: cash flow conversions, zero-depreciation identity, frictions, and breakevens."""

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from parity.deal import (
    DealAnalysisResult,
    analyze_deal,
    calculate_irr_single,
    calculate_moic_single,
    convert_lcu_flows_to_usd,
)
from parity.fxpaths import SimulationResult, simulate_fx_paths
from parity.schemas import DealConfig, RepatriationRateMode


@pytest.fixture
def standard_deal_df():
    # 3-year deal: invested 100m, distributions 10m at month 12 and 24, exit 150m at month 36
    return pd.DataFrame({
        "period": [0, 12, 24, 36],
        "equity_invested_lcu": [100.0, 0.0, 0.0, 0.0],
        "distribution_lcu": [0.0, 10.0, 10.0, 0.0],
        "exit_proceeds_lcu": [0.0, 0.0, 0.0, 150.0],
    })


@pytest.fixture
def baseline_deal_config():
    return DealConfig(
        country="Kenya",
        close_date=dt.date(2026, 1, 31),
        hurdle_usd_irr=0.15,
        repatriation_rate=RepatriationRateMode.OFFICIAL,
        repatriation_blend_weight=0.5,
        blocked_funds_haircut_pct=0.0,
        withholding_tax_pct=0.0,
    )


class TestZeroDepreciationIdentity:
    def test_zero_depreciation_reproduces_lcu_irr_exactly(self, standard_deal_df, baseline_deal_config):
        """CRITICAL CHARTER SUCCESS CRITERION:

        Zero-depreciation path reproduces the local-currency IRR exactly.
        """
        spot = 125.0
        n_paths = 50
        n_months = 36

        # Construct flat simulation result where S_t = spot for all paths and months
        flat_paths = np.full((n_paths, n_months + 1), spot, dtype=float)
        sim_result = SimulationResult(
            paths=flat_paths,
            spot_rate=spot,
            n_months=n_months,
            n_paths=n_paths,
            seed=42,
            jump_counts=np.zeros(n_paths, dtype=int),
        )

        res = analyze_deal(
            deal_df=standard_deal_df,
            deal_config=baseline_deal_config,
            sim_result=sim_result,
        )

        # Exact equivalence between LCU and USD returns on flat path
        assert np.isclose(res.usd_irr_mean, res.headline_lcu_irr, atol=1e-5)
        assert np.isclose(res.usd_irr_median, res.headline_lcu_irr, atol=1e-5)
        assert np.isclose(res.usd_moic_mean, res.headline_lcu_moic, atol=1e-5)
        assert np.isclose(res.expected_fx_drag, 0.0, atol=1e-5)


class TestFrictionsImpact:
    def test_withholding_tax_and_haircut_reduce_usd_return(self, standard_deal_df, baseline_deal_config):
        sim = simulate_fx_paths(spot_rate=100.0, n_months=36, n_paths=200, seed=42)

        # Baseline friction-free result
        res_frictionless = analyze_deal(standard_deal_df, baseline_deal_config, sim_result=sim)

        # Friction deal: 10% WHT and 5% blocked funds haircut
        friction_config = DealConfig(
            country="Kenya",
            close_date=dt.date(2026, 1, 31),
            hurdle_usd_irr=0.15,
            repatriation_rate=RepatriationRateMode.OFFICIAL,
            blocked_funds_haircut_pct=0.05,
            withholding_tax_pct=0.10,
        )
        res_frictions = analyze_deal(standard_deal_df, friction_config, sim_result=sim)

        assert res_frictions.usd_irr_mean < res_frictionless.usd_irr_mean
        assert res_frictions.usd_moic_mean < res_frictionless.usd_moic_mean

    def test_parallel_repatriation_spread_reduces_usd_return(self, standard_deal_df):
        sim = simulate_fx_paths(spot_rate=100.0, n_months=36, n_paths=200, seed=42)

        cfg_official = DealConfig(
            country="Nigeria",
            close_date=dt.date(2026, 1, 31),
            hurdle_usd_irr=0.15,
            repatriation_rate=RepatriationRateMode.OFFICIAL,
        )
        cfg_parallel = DealConfig(
            country="Nigeria",
            close_date=dt.date(2026, 1, 31),
            hurdle_usd_irr=0.15,
            repatriation_rate=RepatriationRateMode.PARALLEL,
        )

        res_off = analyze_deal(standard_deal_df, cfg_official, sim_result=sim, parallel_premium=0.25)
        res_par = analyze_deal(standard_deal_df, cfg_parallel, sim_result=sim, parallel_premium=0.25)

        assert res_par.usd_irr_mean < res_off.usd_irr_mean
        assert res_par.usd_moic_mean < res_off.usd_moic_mean


class TestEdgeCases:
    def test_negative_irr_deal(self, baseline_deal_config):
        """Edge case: capital loss deal with negative IRR."""
        loss_deal = pd.DataFrame({
            "period": [0, 24],
            "equity_invested_lcu": [100.0, 0.0],
            "distribution_lcu": [0.0, 0.0],
            "exit_proceeds_lcu": [0.0, 60.0],  # 40% loss
        })
        sim = simulate_fx_paths(spot_rate=100.0, n_months=24, n_paths=100, seed=42)
        res = analyze_deal(loss_deal, baseline_deal_config, sim_result=sim)

        assert res.headline_lcu_irr < 0.0
        assert res.headline_lcu_moic == 0.60
        assert res.usd_irr_mean < 0.0
        assert res.p_below_hurdle == 1.0

    def test_total_loss_deal(self, baseline_deal_config):
        """Edge case: complete wipeout with 0 exit proceeds."""
        wipeout_deal = pd.DataFrame({
            "period": [0, 12],
            "equity_invested_lcu": [100.0, 0.0],
            "distribution_lcu": [0.0, 0.0],
            "exit_proceeds_lcu": [0.0, 0.0],
        })
        cfs = np.array([-100.0, 0.0])
        irr = calculate_irr_single(cfs)
        moic = calculate_moic_single(cfs)
        assert irr == -1.0
        assert moic == 0.0


class TestBreakevenLocalIRR:
    def test_breakeven_confidence_hierarchy(self, standard_deal_df, baseline_deal_config):
        """75% confidence breakeven must be strictly greater than 50% confidence breakeven."""
        sim = simulate_fx_paths(spot_rate=100.0, n_months=36, n_paths=500, seed=42)
        res = analyze_deal(standard_deal_df, baseline_deal_config, sim_result=sim)

        assert res.breakeven_lcu_irr_75 > res.breakeven_lcu_irr_50
        assert res.breakeven_lcu_irr_50 > baseline_deal_config.hurdle_usd_irr
