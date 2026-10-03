"""Tests for Parity schemas, exceptions, and validation rules."""

from datetime import date
import pytest
from pydantic import ValidationError

from parity.schemas import (
    DataFileNotFoundError,
    DataValidationError,
    DealCashFlowRow,
    DealCashFlowSchema,
    DealConfig,
    EventType,
    FXEventRow,
    FXPanelRow,
    ForwardRow,
    MissingColumnError,
    ParityError,
    RepatriationRateMode,
    FX_PANEL_COLUMNS,
    DEAL_CASH_FLOW_COLUMNS,
)


class TestExceptions:
    def test_custom_exception_hierarchy(self):
        assert issubclass(DataFileNotFoundError, ParityError)
        assert issubclass(DataFileNotFoundError, FileNotFoundError)
        assert issubclass(MissingColumnError, ParityError)
        assert issubclass(MissingColumnError, ValueError)
        assert issubclass(DataValidationError, ParityError)

    def test_data_file_not_found_error_message(self):
        err = DataFileNotFoundError(
            file_path="data/missing_fx_panel.csv",
            expected_columns=FX_PANEL_COLUMNS,
        )
        msg = str(err)
        assert "data/missing_fx_panel.csv" in msg
        assert "official_rate" in msg
        assert "imf_program" in msg
        assert err.expected_columns == FX_PANEL_COLUMNS

    def test_missing_column_error_message(self):
        err = MissingColumnError(
            file_path="data/deal.csv",
            missing_columns=["equity_invested_lcu"],
            expected_columns=DEAL_CASH_FLOW_COLUMNS,
        )
        msg = str(err)
        assert "equity_invested_lcu" in msg
        assert "data/deal.csv" in msg


class TestFXPanelSchema:
    def test_valid_fx_panel_row(self):
        row = FXPanelRow(
            country="Kenya",
            currency="KES",
            date=date(2026, 1, 31),
            official_rate=130.5,
            parallel_rate=135.0,
            reserves_months_imports=4.2,
            reserves_change_3m_pct=-0.02,
            current_account_pct_gdp=-4.5,
            inflation_local_pct=6.8,
            inflation_us_pct=2.5,
            policy_rate_pct=12.5,
            us_rate_pct=4.75,
            imf_program=1,
            imf_review_on_track=1,
        )
        assert row.country == "Kenya"
        assert row.official_rate == 130.5
        assert row.parallel_rate == 135.0
        assert row.imf_review_on_track == 1

    def test_fx_panel_row_nullable_parallel_and_imf_review(self):
        row = FXPanelRow(
            country="Uganda",
            currency="UGX",
            date=date(2026, 2, 28),
            official_rate=3800.0,
            parallel_rate=None,
            reserves_months_imports=3.8,
            reserves_change_3m_pct=0.01,
            current_account_pct_gdp=-3.2,
            inflation_local_pct=3.5,
            inflation_us_pct=2.5,
            policy_rate_pct=9.5,
            us_rate_pct=4.75,
            imf_program=0,
            imf_review_on_track=None,
        )
        assert row.parallel_rate is None
        assert row.imf_review_on_track is None

    def test_invalid_official_rate(self):
        with pytest.raises(ValidationError):
            FXPanelRow(
                country="Kenya",
                currency="KES",
                date=date(2026, 1, 31),
                official_rate=-10.0,  # Negative
                reserves_months_imports=4.0,
                reserves_change_3m_pct=0.0,
                current_account_pct_gdp=-4.0,
                inflation_local_pct=5.0,
                inflation_us_pct=2.0,
                policy_rate_pct=10.0,
                us_rate_pct=4.0,
                imf_program=1,
            )

    def test_invalid_reserves_cover(self):
        with pytest.raises(ValidationError):
            FXPanelRow(
                country="Kenya",
                currency="KES",
                date=date(2026, 1, 31),
                official_rate=100.0,
                reserves_months_imports=-1.0,  # Cannot be negative
                reserves_change_3m_pct=0.0,
                current_account_pct_gdp=-4.0,
                inflation_local_pct=5.0,
                inflation_us_pct=2.0,
                policy_rate_pct=10.0,
                us_rate_pct=4.0,
                imf_program=1,
            )

    def test_invalid_imf_program_flag(self):
        with pytest.raises(ValidationError):
            FXPanelRow(
                country="Kenya",
                currency="KES",
                date=date(2026, 1, 31),
                official_rate=100.0,
                reserves_months_imports=3.0,
                reserves_change_3m_pct=0.0,
                current_account_pct_gdp=-4.0,
                inflation_local_pct=5.0,
                inflation_us_pct=2.0,
                policy_rate_pct=10.0,
                us_rate_pct=4.0,
                imf_program=2,  # Must be 0 or 1
            )


class TestFXEventsSchema:
    def test_valid_event(self):
        event = FXEventRow(
            country="Nigeria",
            date=date(2023, 6, 14),
            event_type=EventType.DEVALUATION,
            size_pct=0.36,
        )
        assert event.event_type == EventType.DEVALUATION
        assert event.size_pct == 0.36

    def test_valid_event_from_string(self):
        event = FXEventRow(
            country="Egypt",
            date=date(2024, 3, 6),
            event_type="float",  # type: ignore[arg-type]
            size_pct=0.38,
        )
        assert event.event_type == EventType.FLOAT

    def test_invalid_event_type(self):
        with pytest.raises(ValidationError):
            FXEventRow(
                country="Ghana",
                date=date(2024, 1, 1),
                event_type="hyperinflation",  # type: ignore[arg-type]
            )


class TestForwardsSchema:
    def test_valid_forward(self):
        fwd = ForwardRow(
            country="Kenya",
            date=date(2026, 1, 31),
            tenor_months=12,
            forward_rate=145.2,
        )
        assert fwd.tenor_months == 12
        assert fwd.forward_rate == 145.2

    def test_invalid_forward_tenor_or_rate(self):
        with pytest.raises(ValidationError):
            ForwardRow(
                country="Kenya",
                date=date(2026, 1, 31),
                tenor_months=0,  # Must be > 0
                forward_rate=100.0,
            )
        with pytest.raises(ValidationError):
            ForwardRow(
                country="Kenya",
                date=date(2026, 1, 31),
                tenor_months=12,
                forward_rate=-5.0,  # Must be > 0
            )


class TestDealSchema:
    def test_valid_deal_cash_flows(self):
        rows = [
            DealCashFlowRow(period=0, equity_invested_lcu=1000.0, distribution_lcu=0.0, exit_proceeds_lcu=0.0),
            DealCashFlowRow(period=12, equity_invested_lcu=0.0, distribution_lcu=100.0, exit_proceeds_lcu=0.0),
            DealCashFlowRow(period=24, equity_invested_lcu=0.0, distribution_lcu=100.0, exit_proceeds_lcu=1500.0),
        ]
        schema = DealCashFlowSchema(rows=rows)
        assert len(schema.rows) == 3

    def test_deal_cash_flow_must_start_at_zero(self):
        rows = [
            DealCashFlowRow(period=1, equity_invested_lcu=1000.0, distribution_lcu=0.0, exit_proceeds_lcu=0.0),
        ]
        with pytest.raises(ValidationError, match="period 0"):
            DealCashFlowSchema(rows=rows)

    def test_duplicate_period_rejected(self):
        rows = [
            DealCashFlowRow(period=0, equity_invested_lcu=1000.0, distribution_lcu=0.0, exit_proceeds_lcu=0.0),
            DealCashFlowRow(period=0, equity_invested_lcu=500.0, distribution_lcu=0.0, exit_proceeds_lcu=0.0),
        ]
        with pytest.raises(ValidationError, match="Duplicate period"):
            DealCashFlowSchema(rows=rows)

    def test_deal_config(self):
        cfg = DealConfig(
            country="Nigeria",
            close_date=date(2026, 1, 31),
            hurdle_usd_irr=0.15,
            repatriation_rate=RepatriationRateMode.BLEND,
            repatriation_blend_weight=0.6,
            blocked_funds_haircut_pct=0.05,
            withholding_tax_pct=0.10,
        )
        assert cfg.country == "Nigeria"
        assert cfg.repatriation_blend_weight == 0.6
        assert cfg.repatriation_rate == RepatriationRateMode.BLEND

    def test_invalid_deal_config_haircut_range(self):
        with pytest.raises(ValidationError):
            DealConfig(
                country="Nigeria",
                close_date=date(2026, 1, 31),
                hurdle_usd_irr=0.15,
                repatriation_rate="blend",  # type: ignore[arg-type]
                blocked_funds_haircut_pct=1.5,  # > 1.0 invalid
            )
