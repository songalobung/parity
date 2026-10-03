"""Tests for hedging module: quoted forwards, CIP proxy fallback, and breakeven depreciation."""

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from parity.deal import analyze_deal
from parity.fxpaths import simulate_fx_paths
from parity.hedging import (
    HedgeAnalysisResult,
    analyze_hedging,
    compute_breakeven_annual_depreciation,
    get_forward_rates_for_deal,
)
from parity.schemas import DealConfig, RepatriationRateMode


@pytest.fixture
def sample_deal_df():
    return pd.DataFrame({
        "period": [0, 12, 24, 36],
        "equity_invested_lcu": [100.0, 0.0, 0.0, 0.0],
        "distribution_lcu": [0.0, 10.0, 10.0, 0.0],
        "exit_proceeds_lcu": [0.0, 0.0, 0.0, 150.0],
    })


@pytest.fixture
def sample_deal_config():
    return DealConfig(
        country="Kenya",
        close_date=dt.date(2026, 1, 31),
        hurdle_usd_irr=0.15,
        repatriation_rate=RepatriationRateMode.OFFICIAL,
        blocked_funds_haircut_pct=0.0,
        withholding_tax_pct=0.0,
    )


class TestForwardRatesAndCIPFallback:
    def test_quoted_forwards_used_when_available(self):
        forwards_df = pd.DataFrame({
            "country": ["Kenya", "Kenya", "Kenya"],
            "date": [dt.date(2026, 1, 31)] * 3,
            "tenor_months": [12, 24, 36],
            "forward_rate": [135.0, 145.0, 158.0],
        })
        rates, is_proxy, reason = get_forward_rates_for_deal(
            periods=[0, 12, 24, 36],
            spot_rate=125.0,
            forwards_df=forwards_df,
            country="Kenya",
        )
        assert not is_proxy
        assert reason is None
        assert rates[0] == 125.0
        assert rates[12] == 135.0
        assert rates[24] == 145.0
        assert rates[36] == 158.0

    def test_cip_proxy_fallback_when_no_forwards(self):
        """CHARTER SUCCESS CRITERION: CIP proxy fallback when forward quotes missing, flagged as proxy."""
        rates, is_proxy, reason = get_forward_rates_for_deal(
            periods=[0, 12, 24],
            spot_rate=100.0,
            forwards_df=None,  # No forwards provided
            country="Nigeria",
            local_rate_pct=20.0,
            us_rate_pct=5.0,
        )
        assert is_proxy is True
        assert reason is not None
        assert "Covered Interest Parity" in reason
        assert rates[0] == 100.0
        # F_12 = 100 * (1.20 / 1.05) ~ 114.28
        expected_12m = 100.0 * (1.20 / 1.05)
        assert np.isclose(rates[12], expected_12m, atol=1e-2)
        assert rates[24] > rates[12]


class TestHedgeAnalysis:
    def test_analyze_hedging_output_structure(self, sample_deal_df, sample_deal_config):
        spot = 125.0
        sim = simulate_fx_paths(spot_rate=spot, n_months=36, n_paths=300, seed=42)
        unhedged_res = analyze_deal(sample_deal_df, sample_deal_config, sim_result=sim)

        hedge_res = analyze_hedging(
            deal_df=sample_deal_df,
            deal_config=sample_deal_config,
            unhedged_result=unhedged_res,
            spot_rate=spot,
            local_rate_pct=14.0,
            us_rate_pct=4.5,
        )

        assert isinstance(hedge_res, HedgeAnalysisResult)
        assert hedge_res.hedged_usd_irr is not None
        assert hedge_res.hedged_usd_moic > 0.0
        assert hedge_res.is_cip_proxy is True
        assert hedge_res.breakeven_annual_depreciation > 0.0
        assert 0.0 <= hedge_res.p_hedging_beats_unhedged <= 1.0

        d = hedge_res.to_dict()
        assert "hedged_usd_irr" in d
        assert "annual_hedge_cost" in d
        assert "breakeven_annual_depreciation" in d

    def test_breakeven_depreciation_accuracy(self, sample_deal_df, sample_deal_config):
        """Verify that converting at breakeven depreciation matches hedged IRR."""
        spot = 100.0
        sim = simulate_fx_paths(spot_rate=spot, n_months=36, n_paths=200, seed=42)
        unhedged_res = analyze_deal(sample_deal_df, sample_deal_config, sim_result=sim)

        hedge_res = analyze_hedging(
            deal_df=sample_deal_df,
            deal_config=sample_deal_config,
            unhedged_result=unhedged_res,
            spot_rate=spot,
            local_rate_pct=15.0,
            us_rate_pct=5.0,
        )

        breakeven_deprec = hedge_res.breakeven_annual_depreciation
        # Construct path with exactly breakeven annual depreciation
        t = np.arange(37, dtype=float)
        be_rates = spot * ((1.0 + breakeven_deprec) ** (t / 12.0))

        # Build USD cash flows along this breakeven path
        cfs_be = np.zeros(37)
        for _, row in sample_deal_df.iterrows():
            p = int(row["period"])
            outflow = float(row["equity_invested_lcu"]) / spot
            inflow = float(row["distribution_lcu"] + row["exit_proceeds_lcu"]) / be_rates[p]
            cfs_be[p] += inflow - outflow

        from parity.deal import calculate_irr_single
        irr_at_be = calculate_irr_single(cfs_be)

        # Must closely match hedged USD IRR
        assert np.isclose(irr_at_be, hedge_res.hedged_usd_irr, atol=1e-3)
