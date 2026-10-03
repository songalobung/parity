"""Deal engine: converts local-currency cash flows to USD distributions along simulated FX paths."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from parity.fxpaths import SimulationResult, simulate_fx_paths
from parity.schemas import DataValidationError, DealConfig, RepatriationRateMode


@dataclass
class DealAnalysisResult:
    """Encapsulates headline LCU returns and simulated USD return distributions.

    Attributes
    ----------
    headline_lcu_irr : float
        Annualized local-currency headline IRR.
    headline_lcu_moic : float
        Headline local-currency MOIC.
    usd_irr_mean : float
        Expected (mean) annualized USD IRR across simulated paths.
    usd_irr_median : float
        Median annualized USD IRR.
    usd_irr_p10 : float
        10th percentile annualized USD IRR.
    usd_irr_p90 : float
        90th percentile annualized USD IRR.
    usd_moic_mean : float
        Mean USD MOIC.
    usd_moic_median : float
        Median USD MOIC.
    usd_moic_p10 : float
        10th percentile USD MOIC.
    usd_moic_p90 : float
        90th percentile USD MOIC.
    hurdle_usd_irr : float
        Target USD IRR hurdle.
    p_below_hurdle : float
        Probability that USD IRR is below hurdle_usd_irr: P(USD IRR < hurdle).
    expected_fx_drag : float
        Expected FX drag (headline_lcu_irr - usd_irr_mean).
    breakeven_lcu_irr_50 : float
        Required headline local-currency IRR to achieve hurdle_usd_irr at 50% confidence.
    breakeven_lcu_irr_75 : float
        Required headline local-currency IRR to achieve hurdle_usd_irr at 75% confidence.
    usd_irrs : np.ndarray
        Array of shape (n_paths,) containing all path USD IRRs.
    usd_moics : np.ndarray
        Array of shape (n_paths,) containing all path USD MOICs.
    """

    headline_lcu_irr: float
    headline_lcu_moic: float
    usd_irr_mean: float
    usd_irr_median: float
    usd_irr_p10: float
    usd_irr_p90: float
    usd_moic_mean: float
    usd_moic_median: float
    usd_moic_p10: float
    usd_moic_p90: float
    hurdle_usd_irr: float
    p_below_hurdle: float
    expected_fx_drag: float
    breakeven_lcu_irr_50: float
    breakeven_lcu_irr_75: float
    usd_irrs: np.ndarray
    usd_moics: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert scalar summary metrics to dictionary."""
        return {
            "headline_lcu_irr": self.headline_lcu_irr,
            "headline_lcu_moic": self.headline_lcu_moic,
            "usd_irr_mean": self.usd_irr_mean,
            "usd_irr_median": self.usd_irr_median,
            "usd_irr_p10": self.usd_irr_p10,
            "usd_irr_p90": self.usd_irr_p90,
            "usd_moic_mean": self.usd_moic_mean,
            "usd_moic_median": self.usd_moic_median,
            "usd_moic_p10": self.usd_moic_p10,
            "usd_moic_p90": self.usd_moic_p90,
            "hurdle_usd_irr": self.hurdle_usd_irr,
            "p_below_hurdle": self.p_below_hurdle,
            "expected_fx_drag": self.expected_fx_drag,
            "breakeven_lcu_irr_50": self.breakeven_lcu_irr_50,
            "breakeven_lcu_irr_75": self.breakeven_lcu_irr_75,
        }


def calculate_irr_single(cfs: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> float:
    """Calculate annualized IRR for a single monthly cash flow array.

    Parameters
    ----------
    cfs : np.ndarray
        1D array where index corresponds to month index (0, 1, ..., T).

    Returns
    -------
    float
        Annualized IRR. Returns -1.0 if complete loss.
    """
    cfs = np.asarray(cfs, dtype=float)
    inflows = np.sum(cfs[cfs > 0])
    outflows = -np.sum(cfs[cfs < 0])

    if outflows <= 0:
        return 0.0
    if inflows <= 0:
        return -1.0

    t = np.arange(len(cfs), dtype=float)

    def npv(r_m: float) -> float:
        return float(np.sum(cfs / ((1.0 + r_m) ** t)))

    # Try bracketing root between -0.999 and 10.0
    try:
        r_m = optimize.brentq(npv, -0.999, 10.0, maxiter=max_iter, xtol=tol)
        return float((1.0 + r_m) ** 12.0 - 1.0)
    except Exception:
        pass

    # Fallback to least squares minimization
    try:
        res = optimize.minimize(
            lambda r: npv(float(r[0])) ** 2,
            x0=[(inflows / outflows) ** (1.0 / max(1, len(cfs) - 1)) - 1.0],
            bounds=[(-0.9999, 10.0)],
        )
        r_m = float(res.x[0])
        return float((1.0 + r_m) ** 12.0 - 1.0)
    except Exception:
        return -1.0


def calculate_moic_single(cfs: np.ndarray) -> float:
    """Calculate MOIC (Total Inflows / Total Outflows)."""
    inflows = float(np.sum(cfs[cfs > 0]))
    outflows = float(-np.sum(cfs[cfs < 0]))
    if outflows <= 0:
        return 0.0
    return inflows / outflows


def calculate_irrs_vectorized(
    cfs_mat: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-5,
) -> np.ndarray:
    """Vectorized calculation of annualized IRR across N paths.

    Parameters
    ----------
    cfs_mat : np.ndarray
        Array of shape (N, T + 1) of monthly cash flows.

    Returns
    -------
    np.ndarray
        Array of shape (N,) containing annualized IRRs.
    """
    N, T = cfs_mat.shape
    t = np.arange(T, dtype=float)

    # Inflows and Outflows
    pos_mask = cfs_mat > 0
    neg_mask = cfs_mat < 0
    inflows = np.sum(np.where(pos_mask, cfs_mat, 0.0), axis=1)
    outflows = np.sum(np.where(neg_mask, -cfs_mat, 0.0), axis=1)

    is_loss = (inflows <= 0) | (outflows <= 0)

    # Initial guess for monthly rate based on MOIC
    moic = np.where(outflows > 0, np.maximum(inflows / np.maximum(outflows, 1e-6), 0.001), 1.0)
    r_m = np.where(is_loss, -0.9999, moic ** (1.0 / max(1, T - 1)) - 1.0)
    r_m = np.clip(r_m, -0.95, 2.0)

    # Vectorized Newton-Raphson
    for _ in range(max_iter):
        denom = (1.0 + r_m[:, None]) ** t[None, :]
        npv = np.sum(cfs_mat / denom, axis=1)
        d_npv = -np.sum(t[None, :] * cfs_mat / (denom * (1.0 + r_m[:, None])), axis=1)

        step = np.where(np.abs(d_npv) > 1e-8, npv / d_npv, 0.0)
        r_m_new = np.clip(r_m - step, -0.999, 10.0)

        if np.max(np.abs(r_m_new - r_m)) < tol:
            r_m = r_m_new
            break
        r_m = r_m_new

    irr_ann = (1.0 + r_m) ** 12.0 - 1.0
    irr_ann[is_loss] = -1.0
    return irr_ann


def calculate_moics_vectorized(cfs_mat: np.ndarray) -> np.ndarray:
    """Calculate MOIC across N paths."""
    pos_mask = cfs_mat > 0
    neg_mask = cfs_mat < 0
    inflows = np.sum(np.where(pos_mask, cfs_mat, 0.0), axis=1)
    outflows = np.sum(np.where(neg_mask, -cfs_mat, 0.0), axis=1)
    return np.where(outflows > 0, inflows / outflows, 0.0)


def convert_lcu_flows_to_usd(
    deal_df: pd.DataFrame,
    deal_config: DealConfig,
    sim_result: SimulationResult,
    parallel_premium: float = 0.0,
) -> np.ndarray:
    """Convert local-currency deal cash flows to USD along simulated FX paths.

    Applies repatriation rate regime (official, parallel, blend),
    withholding tax on distributions, and blocked funds haircut.

    Parameters
    ----------
    deal_df : pd.DataFrame
        DataFrame containing period, equity_invested_lcu, distribution_lcu, exit_proceeds_lcu.
    deal_config : DealConfig
        Deal configuration parameters.
    sim_result : SimulationResult
        Simulated FX paths.
    parallel_premium : float, default 0.0
        Parallel market spread for repatriation rate calculation.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, max_period + 1) of net USD cash flows.
    """
    max_deal_period = int(deal_df["period"].max())
    n_paths = sim_result.n_paths

    if max_deal_period > sim_result.n_months:
        raise DataValidationError(
            f"Deal cash flows extend to month {max_deal_period}, but simulation only has {sim_result.n_months} months."
        )

    # Initialize monthly vectors for LCU
    equity_lcu = np.zeros(max_deal_period + 1, dtype=float)
    dist_lcu = np.zeros(max_deal_period + 1, dtype=float)
    exit_lcu = np.zeros(max_deal_period + 1, dtype=float)

    for _, row in deal_df.iterrows():
        p = int(row["period"])
        equity_lcu[p] += float(row["equity_invested_lcu"])
        dist_lcu[p] += float(row["distribution_lcu"])
        exit_lcu[p] += float(row["exit_proceeds_lcu"])

    # Apply frictions to positive proceeds
    wht_factor = 1.0 - deal_config.withholding_tax_pct
    haircut_factor = 1.0 - deal_config.blocked_funds_haircut_pct

    # Net LCU proceeds before FX conversion
    net_dist_lcu = dist_lcu * wht_factor * haircut_factor
    net_exit_lcu = exit_lcu * haircut_factor

    # Prepare exchange rate matrices: shape (n_paths, max_deal_period + 1)
    rates_slice = sim_result.paths[:, : max_deal_period + 1]

    # Entry conversion rate for equity invested (standard official rate at entry)
    entry_rates = rates_slice

    # Exit / repatriation conversion rate
    repat_mode = deal_config.repatriation_rate
    if repat_mode == RepatriationRateMode.OFFICIAL:
        repat_rates = rates_slice
    elif repat_mode == RepatriationRateMode.PARALLEL:
        repat_rates = rates_slice * (1.0 + parallel_premium)
    elif repat_mode == RepatriationRateMode.BLEND:
        w = deal_config.repatriation_blend_weight
        parallel_rates = rates_slice * (1.0 + parallel_premium)
        repat_rates = w * rates_slice + (1.0 - w) * parallel_rates
    else:
        repat_rates = rates_slice

    # Compute USD cash flows: shape (n_paths, max_deal_period + 1)
    usd_outflows = equity_lcu[None, :] / entry_rates
    usd_inflows = (net_dist_lcu[None, :] + net_exit_lcu[None, :]) / repat_rates

    usd_cfs = usd_inflows - usd_outflows
    return usd_cfs


def calculate_breakeven_lcu_irr(
    deal_df: pd.DataFrame,
    deal_config: DealConfig,
    sim_result: SimulationResult,
    target_confidence: float = 0.50,
    parallel_premium: float = 0.0,
) -> float:
    """Calculate the required headline local-currency IRR needed to hit hurdle_usd_irr.

    Parameters
    ----------
    deal_df : pd.DataFrame
        Deal cash flows.
    deal_config : DealConfig
        Deal configuration.
    sim_result : SimulationResult
        Simulated paths.
    target_confidence : float, default 0.50
        Required confidence level (0.50 for median path, 0.75 for 75% confidence).
    parallel_premium : float, default 0.0
        Parallel premium.

    Returns
    -------
    float
        Required headline local-currency IRR.
    """
    hurdle = deal_config.hurdle_usd_irr

    # Base USD flows
    base_usd_cfs = convert_lcu_flows_to_usd(
        deal_df, deal_config, sim_result, parallel_premium=parallel_premium
    )
    N, T = base_usd_cfs.shape

    # Separate positive and negative flows
    pos_mask = base_usd_cfs > 0
    neg_mask = base_usd_cfs < 0

    base_inflows = np.where(pos_mask, base_usd_cfs, 0.0)
    base_outflows = np.where(neg_mask, base_usd_cfs, 0.0)

    # Percentile quantile for confidence:
    # If target_confidence = 0.50 -> 50th percentile (median)
    # If target_confidence = 0.75 -> 25th percentile (75% of paths beat the hurdle)
    q = (1.0 - target_confidence) * 100.0

    # For large N (e.g. 100k paths), subsample up to 10k paths for the iterative root solver
    if N > 10000:
        sub_idx = np.linspace(0, N - 1, 10000, dtype=int)
        sub_inflows = base_inflows[sub_idx]
        sub_outflows = base_outflows[sub_idx]
    else:
        sub_inflows = base_inflows
        sub_outflows = base_outflows

    def objective(k: float) -> float:
        scaled_cfs = sub_outflows + float(k) * sub_inflows
        irrs = calculate_irrs_vectorized(scaled_cfs)
        achieved_irr = float(np.percentile(irrs, q))
        return achieved_irr - hurdle

    # Solve for multiplier k on inflows
    try:
        sol = optimize.root_scalar(objective, bracket=[0.2, 20.0], method="brentq")
        k_opt = float(sol.root)
    except Exception:
        try:
            res = optimize.minimize(lambda k: objective(float(k[0])) ** 2, x0=[1.5], bounds=[(0.1, 50.0)])
            k_opt = float(res.x[0])
        except Exception:
            k_opt = 1.0

    # Build scaled local cash flows to compute corresponding headline LCU IRR
    max_p = int(deal_df["period"].max())
    lcu_cfs = np.zeros(max_p + 1, dtype=float)
    for _, row in deal_df.iterrows():
        p = int(row["period"])
        outflow = float(row["equity_invested_lcu"])
        inflow = float(row["distribution_lcu"] + row["exit_proceeds_lcu"])
        lcu_cfs[p] += (k_opt * inflow) - outflow

    return calculate_irr_single(lcu_cfs)


def analyze_deal(
    deal_df: pd.DataFrame,
    deal_config: DealConfig,
    sim_result: Optional[SimulationResult] = None,
    spot_rate: Optional[float] = None,
    parallel_premium: float = 0.0,
    n_paths: int = 100000,
    seed: Optional[int] = 42,
) -> DealAnalysisResult:
    """Run end-to-end deal return stress-testing against simulated FX paths.

    Parameters
    ----------
    deal_df : pd.DataFrame
        Deal cash flow DataFrame matching deal.csv schema.
    deal_config : DealConfig
        Deal configuration matching deal.yaml schema.
    sim_result : Optional[SimulationResult], default None
        Pre-simulated FX paths. If None, simulated automatically.
    spot_rate : Optional[float], default None
        Spot exchange rate if sim_result is generated on the fly.
    parallel_premium : float, default 0.0
        Parallel market spread.
    n_paths : int, default 100000
        Number of Monte Carlo paths if simulating.
    seed : Optional[int], default 42
        RNG seed for reproducibility.

    Returns
    -------
    DealAnalysisResult
        Comprehensive returns analysis, distributions, FX drag, and breakevens.
    """
    max_deal_period = int(deal_df["period"].max())

    # Build headline LCU cash flows
    lcu_cfs = np.zeros(max_deal_period + 1, dtype=float)
    for _, row in deal_df.iterrows():
        p = int(row["period"])
        lcu_cfs[p] += (
            float(row["distribution_lcu"])
            + float(row["exit_proceeds_lcu"])
            - float(row["equity_invested_lcu"])
        )

    headline_lcu_irr = calculate_irr_single(lcu_cfs)
    headline_lcu_moic = calculate_moic_single(lcu_cfs)

    # Simulate paths if not provided
    if sim_result is None:
        if spot_rate is None or spot_rate <= 0:
            spot_rate = 100.0  # Normalized spot rate fallback
        sim_result = simulate_fx_paths(
            spot_rate=spot_rate,
            n_months=max(60, max_deal_period),
            n_paths=n_paths,
            seed=seed,
        )

    # Convert LCU flows to USD across all paths
    usd_cfs_mat = convert_lcu_flows_to_usd(
        deal_df, deal_config, sim_result, parallel_premium=parallel_premium
    )

    # Calculate USD returns across all paths
    usd_irrs = calculate_irrs_vectorized(usd_cfs_mat)
    usd_moics = calculate_moics_vectorized(usd_cfs_mat)

    # Summary distributions
    usd_irr_mean = float(np.mean(usd_irrs))
    usd_irr_median = float(np.median(usd_irrs))
    usd_irr_p10 = float(np.percentile(usd_irrs, 10))
    usd_irr_p90 = float(np.percentile(usd_irrs, 90))

    usd_moic_mean = float(np.mean(usd_moics))
    usd_moic_median = float(np.median(usd_moics))
    usd_moic_p10 = float(np.percentile(usd_moics, 10))
    usd_moic_p90 = float(np.percentile(usd_moics, 90))

    # Probability below hurdle
    hurdle = deal_config.hurdle_usd_irr
    p_below_hurdle = float(np.mean(usd_irrs < hurdle))

    # Expected FX drag
    expected_fx_drag = headline_lcu_irr - usd_irr_mean

    # Breakeven local-currency IRR needed for hurdle
    breakeven_50 = calculate_breakeven_lcu_irr(
        deal_df,
        deal_config,
        sim_result,
        target_confidence=0.50,
        parallel_premium=parallel_premium,
    )
    breakeven_75 = calculate_breakeven_lcu_irr(
        deal_df,
        deal_config,
        sim_result,
        target_confidence=0.75,
        parallel_premium=parallel_premium,
    )

    return DealAnalysisResult(
        headline_lcu_irr=headline_lcu_irr,
        headline_lcu_moic=headline_lcu_moic,
        usd_irr_mean=usd_irr_mean,
        usd_irr_median=usd_irr_median,
        usd_irr_p10=usd_irr_p10,
        usd_irr_p90=usd_irr_p90,
        usd_moic_mean=usd_moic_mean,
        usd_moic_median=usd_moic_median,
        usd_moic_p10=usd_moic_p10,
        usd_moic_p90=usd_moic_p90,
        hurdle_usd_irr=hurdle,
        p_below_hurdle=p_below_hurdle,
        expected_fx_drag=expected_fx_drag,
        breakeven_lcu_irr_50=breakeven_50,
        breakeven_lcu_irr_75=breakeven_75,
        usd_irrs=usd_irrs,
        usd_moics=usd_moics,
    )
