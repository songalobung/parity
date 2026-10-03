"""Pydantic schemas and exception types for Parity data inputs."""

import datetime as dt
from enum import Enum
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class ParityError(Exception):
    """Base exception for all Parity runtime and data errors."""


class DataFileNotFoundError(ParityError, FileNotFoundError):
    """Raised when an expected data input file is missing.

    Strict rule: Never invent historical data. If a file is missing,
    fail with a clear error naming the file and expected columns.
    """

    def __init__(self, file_path: Union[str, Path], expected_columns: Sequence[str]):
        self.file_path = str(file_path)
        self.expected_columns = list(expected_columns)
        message = (
            f"Missing required data file: '{self.file_path}'. "
            f"Required columns for this file are: {', '.join(self.expected_columns)}"
        )
        super().__init__(message)


class MissingColumnError(ParityError, ValueError):
    """Raised when a data file is missing one or more required columns."""

    def __init__(
        self,
        file_path: Union[str, Path],
        missing_columns: Sequence[str],
        expected_columns: Sequence[str],
    ):
        self.file_path = str(file_path)
        self.missing_columns = list(missing_columns)
        self.expected_columns = list(expected_columns)
        message = (
            f"File '{self.file_path}' is missing required column(s): {', '.join(self.missing_columns)}. "
            f"All expected columns: {', '.join(self.expected_columns)}"
        )
        super().__init__(message)


class DataValidationError(ParityError, ValueError):
    """Raised when data values fail semantic, type, or boundary validation."""


# -----------------------------------------------------------------------------
# Column Lists
# -----------------------------------------------------------------------------

FX_PANEL_COLUMNS: List[str] = [
    "country",
    "currency",
    "date",
    "official_rate",
    "parallel_rate",
    "reserves_months_imports",
    "reserves_change_3m_pct",
    "current_account_pct_gdp",
    "inflation_local_pct",
    "inflation_us_pct",
    "policy_rate_pct",
    "us_rate_pct",
    "imf_program",
    "imf_review_on_track",
]

FX_EVENTS_COLUMNS: List[str] = [
    "country",
    "date",
    "event_type",
    "size_pct",
]

FORWARDS_COLUMNS: List[str] = [
    "country",
    "date",
    "tenor_months",
    "forward_rate",
]

DEAL_CASH_FLOW_COLUMNS: List[str] = [
    "period",
    "equity_invested_lcu",
    "distribution_lcu",
    "exit_proceeds_lcu",
]

DEAL_CONFIG_KEYS: List[str] = [
    "country",
    "close_date",
    "hurdle_usd_irr",
    "repatriation_rate",
    "repatriation_blend_weight",
    "blocked_funds_haircut_pct",
    "withholding_tax_pct",
]


# -----------------------------------------------------------------------------
# Enumerations
# -----------------------------------------------------------------------------

class EventType(str, Enum):
    """Permitted devaluation / exchange rate regime shock event types."""
    DEVALUATION = "devaluation"
    FLOAT = "float"
    PEG_BREAK = "peg_break"
    MANAGED_CRAWL_CHANGE = "managed_crawl_change"


class RepatriationRateMode(str, Enum):
    """Exchange rate mode used for repatriating foreign investor capital."""
    OFFICIAL = "official"
    PARALLEL = "parallel"
    BLEND = "blend"


# -----------------------------------------------------------------------------
# Row-Level Schemas
# -----------------------------------------------------------------------------

class FXPanelRow(BaseModel):
    """Schema for a single monthly observation in fx_panel.csv."""
    model_config = ConfigDict(extra="ignore")

    country: str = Field(..., min_length=1, description="Country identifier")
    currency: str = Field(..., min_length=1, description="Currency code (e.g. KES, NGN)")
    date: dt.date = Field(..., description="Monthly observation date")
    official_rate: float = Field(..., gt=0, description="Official spot rate (LCU per USD)")
    parallel_rate: Optional[float] = Field(None, description="Parallel market spot rate (LCU per USD)")
    reserves_months_imports: float = Field(..., ge=0, description="Gross reserves in months of imports")
    reserves_change_3m_pct: float = Field(..., description="3-month percentage change in reserves")
    current_account_pct_gdp: float = Field(..., description="Current account balance as % of GDP")
    inflation_local_pct: float = Field(..., description="Local YoY CPI inflation rate (%)")
    inflation_us_pct: float = Field(..., description="US YoY CPI inflation rate (%)")
    policy_rate_pct: float = Field(..., description="Local central bank policy rate (%)")
    us_rate_pct: float = Field(..., description="US policy rate or Treasury yield proxy (%)")
    imf_program: int = Field(..., ge=0, le=1, description="Active IMF program flag (0 or 1)")
    imf_review_on_track: Optional[int] = Field(
        None, description="IMF review status (1=on track, 0=off track, None=NA)"
    )

    @field_validator("parallel_rate", mode="before")
    @classmethod
    def validate_parallel_rate(cls, v: Any) -> Optional[float]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        v_float = float(v)
        if v_float <= 0:
            raise ValueError(f"parallel_rate must be positive if present; got {v_float}")
        return v_float

    @field_validator("imf_review_on_track", mode="before")
    @classmethod
    def validate_imf_review_on_track(cls, v: Any) -> Optional[int]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            v_int = int(v)
        except (ValueError, TypeError):
            raise ValueError(f"imf_review_on_track must be 0, 1, or None; got {v}")
        if v_int not in (0, 1):
            raise ValueError(f"imf_review_on_track must be 0, 1, or None; got {v_int}")
        return v_int


class FXPanelSchema(BaseModel):
    """Collection wrapper for validating multiple FXPanel rows."""
    rows: List[FXPanelRow]


class FXEventRow(BaseModel):
    """Schema for a discrete event in fx_events.csv."""
    model_config = ConfigDict(extra="ignore")

    country: str = Field(..., min_length=1)
    date: dt.date = Field(...)
    event_type: EventType = Field(...)
    size_pct: Optional[float] = Field(None, description="Magnitude of devaluation (e.g. 0.30 for 30%)")

    @field_validator("size_pct", mode="before")
    @classmethod
    def validate_size_pct(cls, v: Any) -> Optional[float]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)


class FXEventSchema(BaseModel):
    """Collection wrapper for validating FXEvent rows."""
    rows: List[FXEventRow]


class ForwardRow(BaseModel):
    """Schema for a single forward rate observation in forwards.csv."""
    model_config = ConfigDict(extra="ignore")

    country: str = Field(..., min_length=1)
    date: dt.date = Field(...)
    tenor_months: int = Field(..., gt=0, description="Tenor in months (e.g. 3, 6, 12, 24)")
    forward_rate: float = Field(..., gt=0, description="Quoted forward rate (LCU per USD)")


class ForwardSchema(BaseModel):
    """Collection wrapper for validating Forward rows."""
    rows: List[ForwardRow]


class DealCashFlowRow(BaseModel):
    """Schema for a single cash flow row in deal.csv."""
    model_config = ConfigDict(extra="ignore")

    period: int = Field(..., ge=0, description="Month index from close (0, 1, 2, ...)")
    equity_invested_lcu: float = Field(..., ge=0, description="Equity invested in LCU")
    distribution_lcu: float = Field(..., ge=0, description="Interim distribution in LCU")
    exit_proceeds_lcu: float = Field(..., ge=0, description="Terminal exit proceeds in LCU")


class DealCashFlowSchema(BaseModel):
    """Collection wrapper for validating deal cash flows."""
    rows: List[DealCashFlowRow]

    @model_validator(mode="after")
    def validate_sequence(self) -> "DealCashFlowSchema":
        if not self.rows:
            raise ValueError("Deal cash flows cannot be empty")
        periods = [r.period for r in self.rows]
        sorted_periods = sorted(periods)
        if sorted_periods[0] != 0:
            raise ValueError(f"Deal cash flows must start at period 0; got period {sorted_periods[0]}")
        # Check uniqueness
        if len(set(periods)) != len(periods):
            raise ValueError("Duplicate period indices found in deal cash flows")
        return self


class DealConfig(BaseModel):
    """Schema for deal execution parameters loaded from deal.yaml."""
    model_config = ConfigDict(extra="ignore")

    country: str = Field(..., min_length=1, description="Deal country")
    close_date: dt.date = Field(..., description="Deal closing date")
    hurdle_usd_irr: float = Field(..., description="Target USD IRR hurdle (e.g. 0.15 for 15%)")
    repatriation_rate: RepatriationRateMode = Field(
        default=RepatriationRateMode.BLEND,
        description="Repatriation rate regime: official, parallel, or blend"
    )
    repatriation_blend_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight given to official rate if repatriation_rate is 'blend'"
    )
    blocked_funds_haircut_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Haircut on proceeds due to trapped or illiquid funds"
    )
    withholding_tax_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Withholding tax applied to distributions (0.0 to 1.0)"
    )
