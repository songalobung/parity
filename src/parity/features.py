"""Feature engineering for Parity devaluation probability and FX path models."""

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from parity.loaders import load_priors


FEATURE_COLUMNS: List[str] = [
    "parallel_premium",
    "reserves_cover",
    "reserves_change_3m",
    "ca_pct_gdp",
    "rer_misalignment",
    "rate_differential",
    "imf_program",
    "imf_review_off_track",
]

TARGET_COLUMNS: List[str] = [
    "target_devaluation_12m",
    "target_devaluation_24m",
]


def compute_parallel_premium(
    official_rate: pd.Series,
    parallel_rate: pd.Series,
    fallback: float = 0.0,
) -> pd.Series:
    """Compute parallel market premium: (parallel_rate / official_rate) - 1.

    If parallel_rate is missing or contains NaNs, fills with fallback and warns.

    Parameters
    ----------
    official_rate : pd.Series
        Official spot exchange rate (LCU per USD).
    parallel_rate : pd.Series
        Parallel market exchange rate (LCU per USD), nullable.
    fallback : float, default 0.0
        Fallback value when parallel rate is missing (0.0 represents zero premium).

    Returns
    -------
    pd.Series
        Parallel market premium spread in decimal (e.g. 0.15 = 15%).
    """
    if parallel_rate.isna().all():
        warnings.warn(
            "parallel_rate is entirely missing; falling back to 0.0 premium across all observations.",
            UserWarning,
            stacklevel=2,
        )
        return pd.Series(fallback, index=official_rate.index, dtype=float)

    if parallel_rate.isna().any():
        num_missing = int(parallel_rate.isna().sum())
        warnings.warn(
            f"parallel_rate is missing in {num_missing} rows; falling back to {fallback}.",
            UserWarning,
            stacklevel=2,
        )

    # Compute raw ratio
    premium = (parallel_rate / official_rate) - 1.0
    premium = premium.fillna(fallback)
    # Premium cannot be negative in economic substance (parallel >= official)
    return premium.clip(lower=0.0)


def compute_rer_misalignment(
    df: pd.DataFrame,
    base_date: Optional[Union[str, dt.date]] = None,
) -> pd.Series:
    """Calculate real exchange rate (RER) misalignment per country.

    RER misalignment = cumulative inflation differential vs nominal depreciation
    since a base date. Positive values indicate real overvaluation (domestic inflation
    has outpaced nominal currency depreciation), increasing devaluation pressure.

    Parameters
    ----------
    df : pd.DataFrame
        FX panel containing country, date, official_rate, inflation_local_pct, inflation_us_pct.
    base_date : Optional[Union[str, dt.date]]
        Base date for calculating cumulative changes. If None or not found for a country,
        defaults to that country's earliest observation date.

    Returns
    -------
    pd.Series
        RER misalignment series indexed matching df.
    """
    misalignment = pd.Series(0.0, index=df.index, dtype=float)

    for country, group in df.groupby("country"):
        group = group.sort_values("date")
        idx = group.index

        # Determine base index
        if base_date is not None:
            parsed_base = pd.to_datetime(base_date).date()
            base_matches = group[group["date"] == parsed_base]
            base_idx = base_matches.index[0] if not base_matches.empty else idx[0]
        else:
            base_idx = idx[0]

        s0 = group.loc[base_idx, "official_rate"]

        # Nominal depreciation relative to base: (S_t / S_0) - 1
        nominal_deprec = (group["official_rate"] / s0) - 1.0

        # Monthly inflation differential approx: (pi_local - pi_us) / 1200
        inf_diff_monthly = (group["inflation_local_pct"] - group["inflation_us_pct"]) / 1200.0

        # Cumulative inflation differential relative to base date
        cum_inf = inf_diff_monthly.cumsum()
        cum_inf_rel = cum_inf - cum_inf.loc[base_idx]

        # Misalignment = cum_inf_diff - nominal_depreciation
        misalign_group = cum_inf_rel - nominal_deprec
        misalignment.loc[idx] = misalign_group

    return misalignment


def compute_rate_differential(
    policy_rate_pct: pd.Series,
    us_rate_pct: pd.Series,
) -> pd.Series:
    """Compute interest rate differential (local policy rate minus US rate)."""
    return policy_rate_pct - us_rate_pct


def compute_imf_review_off_track(
    imf_program: pd.Series,
    imf_review_on_track: pd.Series,
) -> pd.Series:
    """Compute IMF review off-track flag.

    Returns 1 if an active IMF program exists (imf_program == 1)
    and review is explicitly off track (imf_review_on_track == 0).
    Otherwise returns 0.
    """
    is_active = imf_program == 1
    is_off_track = imf_review_on_track == 0
    off_track_flag = (is_active & is_off_track).astype(int)
    return off_track_flag


def compute_devaluation_events(
    panel_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    monthly_threshold: float = 0.15,
    three_month_threshold: float = 0.25,
) -> pd.Series:
    """Identify discrete devaluation events from exchange rate moves or external events.

    An event occurs if:
    - 1-month depreciation > monthly_threshold, OR
    - 3-month cumulative depreciation > three_month_threshold, OR
    - An explicit event is listed in events_df for that country and date.

    Parameters
    ----------
    panel_df : pd.DataFrame
        FX panel sorted by country and date.
    events_df : Optional[pd.DataFrame]
        Optional historical events dataframe.
    monthly_threshold : float, default 0.15
        Depreciation threshold over 1 month (from priors.yaml).
    three_month_threshold : float, default 0.25
        Depreciation threshold over 3 months (from priors.yaml).

    Returns
    -------
    pd.Series
        Binary series (0 or 1) indicating devaluation events.
    """
    is_event = pd.Series(0, index=panel_df.index, dtype=int)

    for _, group in panel_df.groupby("country"):
        idx = group.index
        rates = group["official_rate"]

        # 1-month depreciation: S_t / S_{t-1} - 1
        deprec_1m = rates.pct_change(1)
        # 3-month depreciation: S_t / S_{t-3} - 1
        deprec_3m = rates.pct_change(3)

        cond = (deprec_1m > monthly_threshold) | (deprec_3m > three_month_threshold)
        is_event.loc[idx[cond.fillna(False)]] = 1

    # Overlay external events if provided
    if events_df is not None and not events_df.empty:
        event_set = set(zip(events_df["country"], events_df["date"]))
        for i, row in panel_df.iterrows():
            if (row["country"], row["date"]) in event_set:
                is_event.at[i] = 1

    return is_event


def compute_forward_targets(
    panel_df: pd.DataFrame,
    event_series: pd.Series,
    horizons: Sequence[int] = (12, 24),
) -> pd.DataFrame:
    """Compute forward-looking binary target indicators for devaluation events.

    Target = 1 if an event occurs in (t, t + h] months forward.
    Target = 0 if no event occurs and at least h forward months are observed.
    Target = NaN if fewer than h forward months exist and no event occurred (right-censored).

    Parameters
    ----------
    panel_df : pd.DataFrame
        FX panel dataframe.
    event_series : pd.Series
        Binary indicator of devaluation events at observation date t.
    horizons : Sequence[int], default (12, 24)
        Forward forecasting horizons in months.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns f'target_devaluation_{h}m'.
    """
    targets = pd.DataFrame(index=panel_df.index)

    for h in horizons:
        col_name = f"target_devaluation_{h}m"
        target_col = pd.Series(np.nan, index=panel_df.index, dtype=float)

        for _, group in panel_df.groupby("country"):
            idx = group.index
            n_obs = len(idx)
            events = event_series.loc[idx].to_numpy()

            for pos in range(n_obs):
                window = events[pos + 1 : pos + 1 + h]
                if len(window) == 0:
                    continue
                if np.any(window == 1):
                    target_col.loc[idx[pos]] = 1.0
                elif len(window) >= h:
                    target_col.loc[idx[pos]] = 0.0
                else:
                    # Right-censored without an event
                    target_col.loc[idx[pos]] = np.nan

        targets[col_name] = target_col

    return targets


def build_feature_matrix(
    panel_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    priors_path: Optional[Union[str, Path]] = None,
    base_date: Optional[Union[str, dt.date]] = None,
) -> pd.DataFrame:
    """Generate complete feature matrix and target labels from an FX panel.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Validated FX panel DataFrame from load_fx_panel.
    events_df : Optional[pd.DataFrame], default None
        Optional validated FX events DataFrame.
    priors_path : Optional[Union[str, Path]], default None
        Path to priors.yaml. If None, loaded from default config.
    base_date : Optional[Union[str, dt.date]], default None
        Base date for RER misalignment calculation.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame containing original metadata, all FEATURE_COLUMNS,
        and TARGET_COLUMNS.
    """
    priors = load_priors(priors_path)
    event_priors = priors.get("devaluation_event", {})
    monthly_thresh = float(event_priors.get("monthly_depreciation_threshold", 0.15))
    three_m_thresh = float(event_priors.get("three_month_depreciation_threshold", 0.25))

    df = panel_df.copy().sort_values(by=["country", "date"]).reset_index(drop=True)

    # 1. Parallel market premium
    if "parallel_premium" not in df.columns:
        if "parallel_rate" in df.columns:
            df["parallel_premium"] = compute_parallel_premium(
                df["official_rate"], df["parallel_rate"]
            )
        else:
            warnings.warn("No parallel_rate column in panel; setting parallel_premium to 0.0.", UserWarning)
            df["parallel_premium"] = 0.0

    # 2. Reserves cover
    if "reserves_cover" not in df.columns:
        df["reserves_cover"] = df["reserves_months_imports"].astype(float)

    # 3. Reserves change 3m
    if "reserves_change_3m" not in df.columns:
        df["reserves_change_3m"] = df["reserves_change_3m_pct"].astype(float)

    # 4. Current account balance
    if "ca_pct_gdp" not in df.columns:
        df["ca_pct_gdp"] = df["current_account_pct_gdp"].astype(float)

    # 5. RER Misalignment
    if "rer_misalignment" not in df.columns:
        df["rer_misalignment"] = compute_rer_misalignment(df, base_date=base_date)

    # 6. Interest rate differential
    if "rate_differential" not in df.columns:
        df["rate_differential"] = compute_rate_differential(
            df["policy_rate_pct"], df["us_rate_pct"]
        )

    # 7. IMF program
    df["imf_program"] = df["imf_program"].astype(int)

    # 8. IMF review off-track
    if "imf_review_off_track" not in df.columns:
        imf_review_col = df["imf_review_on_track"] if "imf_review_on_track" in df.columns else pd.Series(None, index=df.index)
        df["imf_review_off_track"] = compute_imf_review_off_track(
            df["imf_program"], imf_review_col
        )

    # 9. Events & Targets
    df["is_devaluation_event"] = compute_devaluation_events(
        df,
        events_df=events_df,
        monthly_threshold=monthly_thresh,
        three_month_threshold=three_m_thresh,
    )

    target_df = compute_forward_targets(df, df["is_devaluation_event"], horizons=(12, 24))
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = target_df[col]

    return df


def generate_synthetic_panel(
    n_countries: int = 4,
    n_months: int = 72,
    seed: int = 42,
    start_date: dt.date = dt.date(2020, 1, 31),
    known_coefficients: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate synthetic FX panel for unit testing and coefficient recovery verification.

    Parameters
    ----------
    n_countries : int, default 4
        Number of synthetic countries.
    n_months : int, default 72
        Number of monthly observations per country (e.g. 6 years).
    seed : int, default 42
        RNG seed for reproducibility.
    start_date : dt.date, default 2020-01-31
        First observation date.
    known_coefficients : Optional[Dict[str, float]]
        True logistic regression coefficients used to generate events.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        Synthetic fx_panel DataFrame, and dictionary of true known coefficients.
    """
    rng = np.random.default_rng(seed)

    country_names = [f"Country_{chr(65 + i)}" for i in range(n_countries)]
    currency_names = [f"CU{chr(65 + i)}" for i in range(n_countries)]

    # Default ground-truth coefficients matching priors signs
    if known_coefficients is None:
        known_coefficients = {
            "intercept": -1.5,
            "parallel_premium": 3.5,
            "reserves_cover": -0.4,
            "reserves_change_3m": -2.0,
            "ca_pct_gdp": -0.12,
            "rer_misalignment": 1.5,
            "rate_differential": -0.06,
            "imf_program": -0.6,
            "imf_review_off_track": 1.0,
        }

    records = []
    dates = pd.date_range(start=start_date, periods=n_months, freq="ME").date

    for c_idx in range(n_countries):
        country = country_names[c_idx]
        currency = currency_names[c_idx]
        base_rate = float(rng.uniform(50.0, 500.0))
        current_rate = base_rate

        for m_idx, current_date in enumerate(dates):
            # Macro fundamentals
            parallel_premium = float(rng.choice([0.0, rng.uniform(0.08, 0.40)], p=[0.6, 0.4]))
            reserves_cover = float(rng.uniform(1.5, 6.0))
            reserves_change_3m = float(rng.uniform(-0.15, 0.10))
            ca_pct_gdp = float(rng.uniform(-7.0, 1.0))
            rer_misalign = float(rng.uniform(-0.15, 0.35))
            inf_local = float(rng.uniform(5.0, 25.0))
            inf_us = float(rng.uniform(2.0, 3.5))
            policy_rate = float(inf_local + rng.uniform(1.0, 5.0))
            us_rate = float(rng.uniform(3.5, 5.5))
            rate_diff = policy_rate - us_rate
            imf_prog = int(rng.choice([0, 1], p=[0.5, 0.5]))
            imf_review = int(rng.choice([0, 1])) if imf_prog == 1 else None
            imf_off_track = 1 if (imf_prog == 1 and imf_review == 0) else 0

            # Compute latent probability and targets
            logit = (
                known_coefficients["intercept"]
                + known_coefficients["parallel_premium"] * parallel_premium
                + known_coefficients["reserves_cover"] * reserves_cover
                + known_coefficients["reserves_change_3m"] * reserves_change_3m
                + known_coefficients["ca_pct_gdp"] * ca_pct_gdp
                + known_coefficients["rer_misalignment"] * rer_misalign
                + known_coefficients["rate_differential"] * rate_diff
                + known_coefficients["imf_program"] * imf_prog
                + known_coefficients["imf_review_off_track"] * imf_off_track
            )
            prob_12m = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
            target_12m = 1.0 if rng.uniform() < prob_12m else 0.0

            logit_24m = logit + 0.8
            prob_24m = 1.0 / (1.0 + np.exp(-np.clip(logit_24m, -30.0, 30.0)))
            target_24m = 1.0 if (target_12m == 1.0 or rng.uniform() < prob_24m) else 0.0

            # Right-censored observations without events
            if m_idx >= n_months - 12 and target_12m == 0.0:
                target_12m = np.nan
            if m_idx >= n_months - 24 and target_24m == 0.0:
                target_24m = np.nan

            # Exchange rate evolution
            crawl_pct = float(rng.uniform(0.003, 0.01))
            current_rate *= (1.0 + crawl_pct)
            parallel_rate = current_rate * (1.0 + parallel_premium) if parallel_premium > 0.0 else None

            records.append({
                "country": country,
                "currency": currency,
                "date": current_date,
                "official_rate": round(current_rate, 4),
                "parallel_rate": round(parallel_rate, 4) if parallel_rate is not None else None,
                "reserves_months_imports": round(reserves_cover, 2),
                "reserves_change_3m_pct": round(reserves_change_3m, 4),
                "current_account_pct_gdp": round(ca_pct_gdp, 2),
                "inflation_local_pct": round(inf_local, 2),
                "inflation_us_pct": round(inf_us, 2),
                "policy_rate_pct": round(policy_rate, 2),
                "us_rate_pct": round(us_rate, 2),
                "imf_program": imf_prog,
                "imf_review_on_track": imf_review,
                "parallel_premium": round(parallel_premium, 4),
                "reserves_cover": round(reserves_cover, 2),
                "reserves_change_3m": round(reserves_change_3m, 4),
                "ca_pct_gdp": round(ca_pct_gdp, 2),
                "rer_misalignment": round(rer_misalign, 4),
                "rate_differential": round(rate_diff, 4),
                "imf_review_off_track": imf_off_track,
                "target_devaluation_12m": target_12m,
                "target_devaluation_24m": target_24m,
            })

    df = pd.DataFrame.from_records(records)
    return df, known_coefficients
