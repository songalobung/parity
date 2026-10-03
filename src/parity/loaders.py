"""Data loaders for Parity CSV and YAML inputs."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml
from pydantic import ValidationError

from parity.schemas import (
    DEAL_CASH_FLOW_COLUMNS,
    DEAL_CONFIG_KEYS,
    FORWARDS_COLUMNS,
    FX_EVENTS_COLUMNS,
    FX_PANEL_COLUMNS,
    DataFileNotFoundError,
    DataValidationError,
    DealCashFlowSchema,
    DealConfig,
    EventType,
    MissingColumnError,
)


def _check_file_exists(file_path: Path, expected_columns: Sequence[str]) -> None:
    """Ensure the file exists; if missing, raise DataFileNotFoundError naming file and columns."""
    if not file_path.is_file():
        raise DataFileNotFoundError(file_path=file_path, expected_columns=expected_columns)


def _check_missing_columns(
    df_columns: Sequence[str],
    expected_columns: Sequence[str],
    file_path: Path,
) -> None:
    """Ensure all required columns exist in the DataFrame header."""
    missing = [col for col in expected_columns if col not in df_columns]
    if missing:
        raise MissingColumnError(
            file_path=file_path,
            missing_columns=missing,
            expected_columns=expected_columns,
        )


def load_fx_panel(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate fx_panel.csv.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the FX panel CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned, typed, and sorted panel dataframe.
    """
    path = Path(file_path)
    _check_file_exists(path, FX_PANEL_COLUMNS)

    df = pd.read_csv(path)
    _check_missing_columns(df.columns, FX_PANEL_COLUMNS, path)

    if df.empty:
        # Return empty structured dataframe matching dtypes
        typed_empty = pd.DataFrame(columns=FX_PANEL_COLUMNS)
        typed_empty["date"] = pd.to_datetime(typed_empty["date"])
        return typed_empty

    # Parse dates
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception as exc:
        raise DataValidationError(f"Invalid date format in {path}: {exc}") from exc

    # Parse and validate official rate
    df["official_rate"] = pd.to_numeric(df["official_rate"], errors="coerce")
    if (df["official_rate"] <= 0).any() or df["official_rate"].isna().any():
        bad = df[df["official_rate"].isna() | (df["official_rate"] <= 0)]
        raise DataValidationError(
            f"official_rate must be strictly positive in {path}. Invalid rows: {bad.index.tolist()}"
        )

    # Parse parallel rate (nullable)
    if "parallel_rate" in df.columns:
        df["parallel_rate"] = pd.to_numeric(df["parallel_rate"], errors="coerce")
        invalid_parallel = df["parallel_rate"].notna() & (df["parallel_rate"] <= 0)
        if invalid_parallel.any():
            bad = df[invalid_parallel]
            raise DataValidationError(
                f"parallel_rate must be strictly positive when present in {path}. "
                f"Invalid rows: {bad.index.tolist()}"
            )

    # Parse reserves and macro columns
    numeric_cols = [
        "reserves_months_imports",
        "reserves_change_3m_pct",
        "current_account_pct_gdp",
        "inflation_local_pct",
        "inflation_us_pct",
        "policy_rate_pct",
        "us_rate_pct",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = df[df[col].isna()]
            raise DataValidationError(f"Column '{col}' contains non-numeric or NaN values in {path} at rows: {bad.index.tolist()}")

    if (df["reserves_months_imports"] < 0).any():
        bad = df[df["reserves_months_imports"] < 0]
        raise DataValidationError(
            f"reserves_months_imports cannot be negative in {path}. Invalid rows: {bad.index.tolist()}"
        )

    # Parse imf_program (must be 0 or 1)
    df["imf_program"] = pd.to_numeric(df["imf_program"], errors="coerce")
    if not df["imf_program"].isin([0, 1]).all():
        raise DataValidationError(f"imf_program must contain only 0 or 1 in {path}")
    df["imf_program"] = df["imf_program"].astype(int)

    # Parse imf_review_on_track (0, 1, or nullable)
    def _parse_imf_review(val: Any) -> Optional[int]:
        if pd.isna(val) or val is None or str(val).strip().upper() in ("NA", "NAN", ""):
            return None
        try:
            num = int(float(val))
            if num in (0, 1):
                return num
        except (ValueError, TypeError):
            pass
        raise DataValidationError(f"Invalid imf_review_on_track value '{val}' in {path}; expected 0, 1, or NA")

    df["imf_review_on_track"] = df["imf_review_on_track"].apply(_parse_imf_review)

    # Clean country and currency strings
    df["country"] = df["country"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.strip()

    # Sort deterministically
    df = df.sort_values(by=["country", "date"]).reset_index(drop=True)
    return df


def load_fx_events(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate fx_events.csv.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the FX events CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned, typed, and sorted events dataframe.
    """
    path = Path(file_path)
    _check_file_exists(path, FX_EVENTS_COLUMNS)

    df = pd.read_csv(path)
    _check_missing_columns(df.columns, FX_EVENTS_COLUMNS, path)

    if df.empty:
        typed_empty = pd.DataFrame(columns=FX_EVENTS_COLUMNS)
        typed_empty["date"] = pd.to_datetime(typed_empty["date"])
        return typed_empty

    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception as exc:
        raise DataValidationError(f"Invalid date format in {path}: {exc}") from exc

    df["country"] = df["country"].astype(str).str.strip()

    # Validate event_type
    valid_events = {e.value for e in EventType}
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    invalid_events = ~df["event_type"].isin(valid_events)
    if invalid_events.any():
        bad = df[invalid_events]
        raise DataValidationError(
            f"Invalid event_type in {path}. Permitted values: {sorted(valid_events)}. "
            f"Found: {bad['event_type'].unique().tolist()}"
        )

    # Validate size_pct (nullable float)
    df["size_pct"] = pd.to_numeric(df["size_pct"], errors="coerce")

    df = df.sort_values(by=["country", "date"]).reset_index(drop=True)
    return df


def load_forwards(file_path: Optional[Union[str, Path]]) -> Optional[pd.DataFrame]:
    """Load and validate forwards.csv if provided.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]]
        Path to the forwards CSV file. If None, returns None.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame if file was provided and loaded, or None if file_path is None.
    """
    if file_path is None:
        return None

    path = Path(file_path)
    _check_file_exists(path, FORWARDS_COLUMNS)

    df = pd.read_csv(path)
    _check_missing_columns(df.columns, FORWARDS_COLUMNS, path)

    if df.empty:
        typed_empty = pd.DataFrame(columns=FORWARDS_COLUMNS)
        typed_empty["date"] = pd.to_datetime(typed_empty["date"])
        return typed_empty

    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception as exc:
        raise DataValidationError(f"Invalid date format in {path}: {exc}") from exc

    df["country"] = df["country"].astype(str).str.strip()

    # Validate tenor_months
    df["tenor_months"] = pd.to_numeric(df["tenor_months"], errors="coerce")
    if (df["tenor_months"] <= 0).any() or df["tenor_months"].isna().any():
        bad = df[df["tenor_months"].isna() | (df["tenor_months"] <= 0)]
        raise DataValidationError(f"tenor_months must be positive integers in {path}. Rows: {bad.index.tolist()}")
    df["tenor_months"] = df["tenor_months"].astype(int)

    # Validate forward_rate
    df["forward_rate"] = pd.to_numeric(df["forward_rate"], errors="coerce")
    if (df["forward_rate"] <= 0).any() or df["forward_rate"].isna().any():
        bad = df[df["forward_rate"].isna() | (df["forward_rate"] <= 0)]
        raise DataValidationError(f"forward_rate must be positive in {path}. Rows: {bad.index.tolist()}")

    df = df.sort_values(by=["country", "date", "tenor_months"]).reset_index(drop=True)
    return df


def load_deal_cash_flows(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate deal.csv cash flows.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to deal cash flow CSV.

    Returns
    -------
    pd.DataFrame
        Validated, sorted cash flow dataframe.
    """
    path = Path(file_path)
    _check_file_exists(path, DEAL_CASH_FLOW_COLUMNS)

    df = pd.read_csv(path)
    _check_missing_columns(df.columns, DEAL_CASH_FLOW_COLUMNS, path)

    if df.empty:
        raise DataValidationError(f"Deal cash flows file '{path}' is empty. At least period 0 is required.")

    # Validate period
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    if df["period"].isna().any() or (df["period"] < 0).any():
        raise DataValidationError(f"period must be non-negative integers in {path}")
    df["period"] = df["period"].astype(int)

    # Check non-negative cash flows
    cf_cols = ["equity_invested_lcu", "distribution_lcu", "exit_proceeds_lcu"]
    for col in cf_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or (df[col] < 0).any():
            bad = df[df[col].isna() | (df[col] < 0)]
            raise DataValidationError(f"Cash flow column '{col}' must be non-negative in {path}. Rows: {bad.index.tolist()}")

    # Validate sequence with Pydantic model
    records = df.to_dict(orient="records")
    try:
        DealCashFlowSchema(rows=records)  # type: ignore[arg-type]
    except (ValidationError, ValueError) as exc:
        raise DataValidationError(f"Deal cash flow validation failed for {path}: {exc}") from exc

    df = df.sort_values(by="period").reset_index(drop=True)
    return df


def load_deal_config(file_path: Union[str, Path]) -> DealConfig:
    """Load and validate deal.yaml.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to deal configuration YAML.

    Returns
    -------
    DealConfig
        Validated Pydantic configuration object.
    """
    path = Path(file_path)
    _check_file_exists(path, DEAL_CONFIG_KEYS)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    except Exception as exc:
        raise DataValidationError(f"Failed to parse YAML file '{path}': {exc}") from exc

    if not isinstance(raw_cfg, dict):
        raise DataValidationError(f"Expected YAML file '{path}' to be a mapping/dict, got {type(raw_cfg).__name__}")

    missing_keys = [k for k in DEAL_CONFIG_KEYS if k not in raw_cfg]
    if missing_keys:
        raise MissingColumnError(
            file_path=path,
            missing_columns=missing_keys,
            expected_columns=DEAL_CONFIG_KEYS,
        )

    try:
        return DealConfig(**raw_cfg)
    except ValidationError as exc:
        raise DataValidationError(f"Validation failed for deal config '{path}': {exc}") from exc


DEFAULT_PRIORS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "priors.yaml"


def load_priors(file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load model priors from config/priors.yaml.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]]
        Path to priors YAML file. If None, defaults to config/priors.yaml.

    Returns
    -------
    Dict[str, Any]
        Dictionary of documented priors.
    """
    path = Path(file_path) if file_path is not None else DEFAULT_PRIORS_PATH
    if not path.is_file():
        raise DataFileNotFoundError(
            file_path=path,
            expected_columns=["devaluation_event", "devaluation_probability"],
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            priors = yaml.safe_load(f)
    except Exception as exc:
        raise DataValidationError(f"Failed to parse priors YAML file '{path}': {exc}") from exc

    if not isinstance(priors, dict):
        raise DataValidationError(f"Expected priors YAML to be a dict, got {type(priors).__name__}")
    return priors

