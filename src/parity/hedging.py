"""Hedging comparison: hedged vs unhedged USD returns, CIP proxy fallback, and breakeven devaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize

from parity.deal import DealAnalysisResult, calculate_irr_single, calculate_moic_single
from parity.schemas import DealConfig, RepatriationRateMode


@dataclass
class HedgeAnalysisResult:
    """Encapsulates hedged versus unhedged deal return comparison.

    Attributes
    ----------
    hedged_usd_irr : float
        Deterministic annualized USD IRR achieved under the forward hedge contract.
    hedged_usd_moic : float
        Hedged USD Multiple on Invested Capital.
    unhedged_usd_irr_mean : float
        Expected (mean) unhedged USD IRR across simulated paths.
    unhedged_usd_moic_mean : float
        Expected unhedged USD MOIC across simulated paths.
    is_cip_proxy : bool
        True if forward rates were synthesized via Covered Interest Parity proxy;
        False if sourced from quoted forwards in forwards.csv.
    proxy_reason : Optional[str]
        Explanation if CIP proxy was utilized.
    annual_hedge_cost : float
        Annualized return cost of hedging versus unhedged expected USD IRR
        (unhedged_usd_irr_mean - hedged_usd_irr).
    hedge_cost_vs_lcu : float
        Return drag of hedging versus headline local-currency IRR
        (headline_lcu_irr - hedged_usd_irr).
    breakeven_annual_depreciation : float
        Annual currency depreciation rate at which the unhedged return equals the hedged return.
        If actual annual depreciation exceeds this rate, hedging was profitable.
    p_hedging_beats_unhedged : float
        Fraction of simulated paths where hedged USD IRR > simulated unhedged USD IRR.
    forward_rates : Dict[int, float]
        Mapping of period month index to forward exchange rate applied.
    """

    hedged_usd_irr: float
    hedged_usd_moic: float
    unhedged_usd_irr_mean: float
    unhedged_usd_moic_mean: float
    is_cip_proxy: bool
    proxy_reason: Optional[str]
    annual_hedge_cost: float
    hedge_cost_vs_lcu: float
    breakeven_annual_depreciation: float
    p_hedging_beats_unhedged: float
    forward_rates: Dict[int, float]

    @property
    def forward_source(self) -> str:
        """Source description of forward rates."""
        return "Covered Interest Parity Proxy" if self.is_cip_proxy else "Quoted Forwards"

    @property
    def hedge_cost_drag(self) -> float:
        """Alias for annual_hedge_cost."""
        return self.annual_hedge_cost

    @property
    def p_hedge_outperforms(self) -> float:
        """Alias for p_hedging_beats_unhedged."""
        return self.p_hedging_beats_unhedged

    def to_dict(self) -> Dict[str, Any]:
        """Convert scalar metrics to dictionary."""
        return {
            "hedged_usd_irr": self.hedged_usd_irr,
            "hedged_usd_moic": self.hedged_usd_moic,
            "unhedged_usd_irr_mean": self.unhedged_usd_irr_mean,
            "unhedged_usd_moic_mean": self.unhedged_usd_moic_mean,
            "is_cip_proxy": self.is_cip_proxy,
            "proxy_reason": self.proxy_reason,
            "annual_hedge_cost": self.annual_hedge_cost,
            "hedge_cost_vs_lcu": self.hedge_cost_vs_lcu,
            "breakeven_annual_depreciation": self.breakeven_annual_depreciation,
            "p_hedging_beats_unhedged": self.p_hedging_beats_unhedged,
        }


def get_forward_rates_for_deal(
    periods: List[int],
    spot_rate: float,
    forwards_df: Optional[pd.DataFrame] = None,
    country: Optional[str] = None,
    local_rate_pct: float = 12.0,
    us_rate_pct: float = 4.5,
) -> Tuple[Dict[int, float], bool, Optional[str]]:
    """Determine forward rates for deal periods using forwards_df or CIP proxy fallback.

    Parameters
    ----------
    periods : List[int]
        Deal month indices (e.g. [0, 12, 24, 36]).
    spot_rate : float
        Spot exchange rate at period 0.
    forwards_df : Optional[pd.DataFrame], default None
        Optional DataFrame loaded from forwards.csv.
    country : Optional[str], default None
        Country name filter for forwards_df.
    local_rate_pct : float, default 12.0
        Domestic policy / money market rate for CIP proxy (%).
    us_rate_pct : float, default 4.5
        US policy rate for CIP proxy (%).

    Returns
    -------
    Tuple[Dict[int, float], bool, Optional[str]]
        Dictionary of {period: forward_rate}, is_cip_proxy flag, and reason string.
    """
    forward_rates: Dict[int, float] = {0: spot_rate}
    is_cip_proxy = False
    proxy_reason: Optional[str] = None

    # Check if quoted forwards are available in forwards_df
    quotes_found = False
    if forwards_df is not None and not forwards_df.empty:
        df_c = forwards_df
        if country is not None and "country" in df_c.columns:
            df_c = df_c[df_c["country"].str.lower() == country.lower()]

        if not df_c.empty and "tenor_months" in df_c.columns and "forward_rate" in df_c.columns:
            tenor_map = dict(zip(df_c["tenor_months"], df_c["forward_rate"]))
            # Check if quotes cover our deal periods
            deal_periods_to_cover = [p for p in periods if p > 0]
            can_cover = all(p in tenor_map for p in deal_periods_to_cover)
            if can_cover:
                for p in deal_periods_to_cover:
                    forward_rates[p] = float(tenor_map[p])
                quotes_found = True

    if not quotes_found:
        is_cip_proxy = True
        proxy_reason = (
            f"No quoted forwards available for {country or 'country'}; using Covered Interest Parity (CIP) "
            f"proxy based on policy rate spread ({local_rate_pct:.1f}% vs {us_rate_pct:.1f}%)."
        )
        # Covered Interest Parity formula:
        # F_t = S_0 * ((1 + r_local) / (1 + r_us))^(t / 12)
        r_loc = local_rate_pct / 100.0
        r_us = us_rate_pct / 100.0
        cip_ratio = (1.0 + r_loc) / (1.0 + r_us)

        for p in periods:
            if p == 0:
                forward_rates[0] = spot_rate
            else:
                forward_rates[p] = float(spot_rate * (cip_ratio ** (p / 12.0)))

    return forward_rates, is_cip_proxy, proxy_reason


def compute_breakeven_annual_depreciation(
    deal_df: pd.DataFrame,
    deal_config: DealConfig,
    hedged_usd_irr: float,
    spot_rate: float,
    parallel_premium: float = 0.0,
) -> float:
    """Find the annual currency depreciation rate at which unhedged IRR equals hedged IRR."""
    max_p = int(deal_df["period"].max())
    if max_p == 0:
        return 0.0

    wht_factor = 1.0 - deal_config.withholding_tax_pct
    haircut_factor = 1.0 - deal_config.blocked_funds_haircut_pct

    # Pre-extract inflows and outflows per period
    equity_lcu = np.zeros(max_p + 1, dtype=float)
    proceeds_lcu = np.zeros(max_p + 1, dtype=float)

    for _, row in deal_df.iterrows():
        p = int(row["period"])
        equity_lcu[p] += float(row["equity_invested_lcu"])
        dist = float(row["distribution_lcu"]) * wht_factor * haircut_factor
        exit_p = float(row["exit_proceeds_lcu"]) * haircut_factor
        proceeds_lcu[p] += dist + exit_p

    t = np.arange(max_p + 1, dtype=float)

    def irr_at_depreciation(delta_annual: float) -> float:
        # S_t = S_0 * (1 + delta_annual)^(t / 12)
        path_rates = spot_rate * ((1.0 + max(-0.90, delta_annual)) ** (t / 12.0))
        # Outflow at spot/official, inflow at repat rate
        repat_rates = path_rates
        if deal_config.repatriation_rate == RepatriationRateMode.PARALLEL:
            repat_rates = path_rates * (1.0 + parallel_premium)
        elif deal_config.repatriation_rate == RepatriationRateMode.BLEND:
            w = deal_config.repatriation_blend_weight
            repat_rates = w * path_rates + (1.0 - w) * (path_rates * (1.0 + parallel_premium))

        usd_cfs = (proceeds_lcu / repat_rates) - (equity_lcu / path_rates)
        return calculate_irr_single(usd_cfs) - hedged_usd_irr

    # Solve for delta_annual
    try:
        sol = optimize.root_scalar(irr_at_depreciation, bracket=[-0.50, 1.50], method="brentq")
        return float(sol.root)
    except Exception:
        try:
            res = optimize.minimize(lambda d: irr_at_depreciation(float(d[0])) ** 2, x0=[0.10], bounds=[(-0.80, 5.0)])
            return float(res.x[0])
        except Exception:
            return 0.0


def analyze_hedging(
    deal_df: pd.DataFrame,
    deal_config: DealConfig,
    unhedged_result: DealAnalysisResult,
    spot_rate: float,
    forwards_df: Optional[pd.DataFrame] = None,
    local_rate_pct: float = 12.0,
    us_rate_pct: float = 4.5,
    parallel_premium: float = 0.0,
) -> HedgeAnalysisResult:
    """Analyze hedged vs unhedged USD returns for a deal.

    Parameters
    ----------
    deal_df : pd.DataFrame
        Deal cash flows dataframe.
    deal_config : DealConfig
        Deal configuration.
    unhedged_result : DealAnalysisResult
        Results from analyze_deal on simulated paths.
    spot_rate : float
        Current spot rate (LCU per USD).
    forwards_df : Optional[pd.DataFrame], default None
        Optional forwards dataframe from forwards.csv.
    local_rate_pct : float, default 12.0
        Domestic policy rate (%).
    us_rate_pct : float, default 4.5
        US policy rate (%).
    parallel_premium : float, default 0.0
        Parallel market spread.

    Returns
    -------
    HedgeAnalysisResult
        Comparative hedging analytics.
    """
    periods = sorted(deal_df["period"].unique().tolist())
    forward_rates, is_cip_proxy, proxy_reason = get_forward_rates_for_deal(
        periods=periods,
        spot_rate=spot_rate,
        forwards_df=forwards_df,
        country=deal_config.country,
        local_rate_pct=local_rate_pct,
        us_rate_pct=us_rate_pct,
    )

    max_p = int(deal_df["period"].max())
    hedged_usd_cfs = np.zeros(max_p + 1, dtype=float)

    wht_factor = 1.0 - deal_config.withholding_tax_pct
    haircut_factor = 1.0 - deal_config.blocked_funds_haircut_pct

    for _, row in deal_df.iterrows():
        p = int(row["period"])
        fwd_rate = forward_rates.get(p, spot_rate)

        # Equity invested at spot rate at entry
        entry_rate = spot_rate if p == 0 else fwd_rate
        outflow_usd = float(row["equity_invested_lcu"]) / entry_rate

        # Proceeds converted at locked forward contract rate
        net_dist = float(row["distribution_lcu"]) * wht_factor * haircut_factor
        net_exit = float(row["exit_proceeds_lcu"]) * haircut_factor
        inflow_usd = (net_dist + net_exit) / fwd_rate

        hedged_usd_cfs[p] += inflow_usd - outflow_usd

    hedged_usd_irr = calculate_irr_single(hedged_usd_cfs)
    hedged_usd_moic = calculate_moic_single(hedged_usd_cfs)

    # Cost of hedging
    annual_hedge_cost = unhedged_result.usd_irr_mean - hedged_usd_irr
    hedge_cost_vs_lcu = unhedged_result.headline_lcu_irr - hedged_usd_irr

    # Probability that hedging beats unhedged simulation
    p_hedging_beats_unhedged = float(np.mean(hedged_usd_irr > unhedged_result.usd_irrs))

    # Breakeven annual depreciation
    breakeven_deprec = compute_breakeven_annual_depreciation(
        deal_df=deal_df,
        deal_config=deal_config,
        hedged_usd_irr=hedged_usd_irr,
        spot_rate=spot_rate,
        parallel_premium=parallel_premium,
    )

    return HedgeAnalysisResult(
        hedged_usd_irr=hedged_usd_irr,
        hedged_usd_moic=hedged_usd_moic,
        unhedged_usd_irr_mean=unhedged_result.usd_irr_mean,
        unhedged_usd_moic_mean=unhedged_result.usd_moic_mean,
        is_cip_proxy=is_cip_proxy,
        proxy_reason=proxy_reason,
        annual_hedge_cost=annual_hedge_cost,
        hedge_cost_vs_lcu=hedge_cost_vs_lcu,
        breakeven_annual_depreciation=breakeven_deprec,
        p_hedging_beats_unhedged=p_hedging_beats_unhedged,
        forward_rates=forward_rates,
    )
