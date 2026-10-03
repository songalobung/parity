"""Tests for Parity loaders: file handling, CSV/YAML parsing, schema validation, and templates."""

from datetime import date
from pathlib import Path
import pandas as pd
import pytest
import yaml

from parity.loaders import (
    load_deal_cash_flows,
    load_deal_config,
    load_forwards,
    load_fx_events,
    load_fx_panel,
)
from parity.schemas import (
    DEAL_CASH_FLOW_COLUMNS,
    DEAL_CONFIG_KEYS,
    FORWARDS_COLUMNS,
    FX_EVENTS_COLUMNS,
    FX_PANEL_COLUMNS,
    DataFileNotFoundError,
    DataValidationError,
    DealConfig,
    MissingColumnError,
    RepatriationRateMode,
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "data" / "templates"


# -----------------------------------------------------------------------------
# Template Validation Tests
# -----------------------------------------------------------------------------

class TestTemplates:
    def test_fx_panel_template_exists_and_loads(self):
        tpl = TEMPLATES_DIR / "fx_panel.csv"
        assert tpl.is_file(), f"Template not found at {tpl}"
        df = load_fx_panel(tpl)
        assert list(df.columns) == FX_PANEL_COLUMNS
        assert len(df) == 0

    def test_fx_events_template_exists_and_loads(self):
        tpl = TEMPLATES_DIR / "fx_events.csv"
        assert tpl.is_file(), f"Template not found at {tpl}"
        df = load_fx_events(tpl)
        assert list(df.columns) == FX_EVENTS_COLUMNS
        assert len(df) == 0

    def test_forwards_template_exists_and_loads(self):
        tpl = TEMPLATES_DIR / "forwards.csv"
        assert tpl.is_file(), f"Template not found at {tpl}"
        df = load_forwards(tpl)
        assert df is not None
        assert list(df.columns) == FORWARDS_COLUMNS
        assert len(df) == 0

    def test_deal_csv_template_headers(self):
        tpl = TEMPLATES_DIR / "deal.csv"
        assert tpl.is_file(), f"Template not found at {tpl}"
        # Empty deal template without rows should fail cash flow check (requires at least period 0)
        with pytest.raises(DataValidationError, match="empty"):
            load_deal_cash_flows(tpl)

    def test_deal_yaml_template_loads(self):
        tpl = TEMPLATES_DIR / "deal.yaml"
        assert tpl.is_file(), f"Template not found at {tpl}"
        cfg = load_deal_config(tpl)
        assert isinstance(cfg, DealConfig)
        assert cfg.country == "Nigeria"
        assert cfg.hurdle_usd_irr == 0.15
        assert cfg.repatriation_rate == RepatriationRateMode.BLEND


# -----------------------------------------------------------------------------
# Missing File Tests (Charter / hi.md rule: fail naming file and columns)
# -----------------------------------------------------------------------------

class TestMissingFiles:
    def test_missing_fx_panel_file_error(self, tmp_path):
        missing_file = tmp_path / "non_existent_fx_panel.csv"
        with pytest.raises(DataFileNotFoundError) as exc_info:
            load_fx_panel(missing_file)
        err = exc_info.value
        assert str(missing_file) in str(err)
        for col in ["country", "official_rate", "reserves_months_imports", "imf_program"]:
            assert col in str(err)

    def test_missing_fx_events_file_error(self, tmp_path):
        missing_file = tmp_path / "non_existent_events.csv"
        with pytest.raises(DataFileNotFoundError) as exc_info:
            load_fx_events(missing_file)
        assert str(missing_file) in str(exc_info.value)
        assert "event_type" in str(exc_info.value)

    def test_missing_forwards_file_error(self, tmp_path):
        missing_file = tmp_path / "non_existent_forwards.csv"
        with pytest.raises(DataFileNotFoundError) as exc_info:
            load_forwards(missing_file)
        assert str(missing_file) in str(exc_info.value)
        assert "forward_rate" in str(exc_info.value)

    def test_optional_forwards_file_none(self):
        assert load_forwards(None) is None

    def test_missing_deal_csv_error(self, tmp_path):
        missing_file = tmp_path / "non_existent_deal.csv"
        with pytest.raises(DataFileNotFoundError) as exc_info:
            load_deal_cash_flows(missing_file)
        assert str(missing_file) in str(exc_info.value)
        assert "equity_invested_lcu" in str(exc_info.value)

    def test_missing_deal_yaml_error(self, tmp_path):
        missing_file = tmp_path / "non_existent_deal.yaml"
        with pytest.raises(DataFileNotFoundError) as exc_info:
            load_deal_config(missing_file)
        assert str(missing_file) in str(exc_info.value)
        assert "hurdle_usd_irr" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Missing Column Tests
# -----------------------------------------------------------------------------

class TestMissingColumns:
    def test_fx_panel_missing_columns(self, tmp_path):
        csv_file = tmp_path / "panel_incomplete.csv"
        csv_file.write_text("country,currency,date,official_rate\nKenya,KES,2026-01-31,130.0\n")
        with pytest.raises(MissingColumnError) as exc_info:
            load_fx_panel(csv_file)
        err = exc_info.value
        assert "reserves_months_imports" in err.missing_columns
        assert "imf_program" in err.missing_columns

    def test_deal_yaml_missing_keys(self, tmp_path):
        yaml_file = tmp_path / "deal_incomplete.yaml"
        yaml_file.write_text("country: Nigeria\nhurdle_usd_irr: 0.15\n")
        with pytest.raises(MissingColumnError) as exc_info:
            load_deal_config(yaml_file)
        assert "close_date" in exc_info.value.missing_columns


# -----------------------------------------------------------------------------
# Data Parsing & Validation Tests
# -----------------------------------------------------------------------------

class TestDataParsing:
    def test_load_valid_fx_panel(self, tmp_path):
        csv_file = tmp_path / "fx_panel.csv"
        csv_file.write_text(
            "country,currency,date,official_rate,parallel_rate,reserves_months_imports,"
            "reserves_change_3m_pct,current_account_pct_gdp,inflation_local_pct,inflation_us_pct,"
            "policy_rate_pct,us_rate_pct,imf_program,imf_review_on_track\n"
            "Nigeria,NGN,2026-02-28,1450.0,1520.0,3.5,-0.04,-2.5,28.5,2.4,26.0,4.75,1,1\n"
            "Kenya,KES,2026-01-31,129.5,,4.1,0.01,-4.2,6.5,2.4,12.75,4.75,1,NA\n"
        )
        df = load_fx_panel(csv_file)
        assert len(df) == 2
        # Sorting check: Kenya comes before Nigeria
        assert df.iloc[0]["country"] == "Kenya"
        assert pd.isna(df.iloc[0]["parallel_rate"])
        assert pd.isna(df.iloc[0]["imf_review_on_track"])
        assert df.iloc[1]["country"] == "Nigeria"
        assert df.iloc[1]["official_rate"] == 1450.0
        assert df.iloc[1]["imf_review_on_track"] == 1
        assert isinstance(df.iloc[0]["date"], date)

    def test_fx_panel_invalid_values(self, tmp_path):
        # Negative official rate
        csv_file = tmp_path / "fx_panel_bad.csv"
        csv_file.write_text(
            "country,currency,date,official_rate,parallel_rate,reserves_months_imports,"
            "reserves_change_3m_pct,current_account_pct_gdp,inflation_local_pct,inflation_us_pct,"
            "policy_rate_pct,us_rate_pct,imf_program,imf_review_on_track\n"
            "Kenya,KES,2026-01-31,-129.5,,4.1,0.01,-4.2,6.5,2.4,12.75,4.75,1,1\n"
        )
        with pytest.raises(DataValidationError, match="official_rate"):
            load_fx_panel(csv_file)

    def test_load_valid_fx_events(self, tmp_path):
        csv_file = tmp_path / "fx_events.csv"
        csv_file.write_text(
            "country,date,event_type,size_pct\n"
            "Nigeria,2023-06-14,devaluation,0.36\n"
            "Egypt,2024-03-06,float,\n"
        )
        df = load_fx_events(csv_file)
        assert len(df) == 2
        assert df.iloc[0]["country"] == "Egypt"  # Sorted alphabetically
        assert df.iloc[0]["event_type"] == "float"
        assert pd.isna(df.iloc[0]["size_pct"])
        assert df.iloc[1]["country"] == "Nigeria"
        assert df.iloc[1]["size_pct"] == 0.36

    def test_load_fx_events_invalid_type(self, tmp_path):
        csv_file = tmp_path / "fx_events_bad.csv"
        csv_file.write_text(
            "country,date,event_type,size_pct\n"
            "Kenya,2024-01-01,unrecognized_event,0.1\n"
        )
        with pytest.raises(DataValidationError, match="Invalid event_type"):
            load_fx_events(csv_file)

    def test_load_valid_forwards(self, tmp_path):
        csv_file = tmp_path / "forwards.csv"
        csv_file.write_text(
            "country,date,tenor_months,forward_rate\n"
            "Kenya,2026-01-31,12,142.5\n"
            "Kenya,2026-01-31,3,133.0\n"
        )
        df = load_forwards(csv_file)
        assert df is not None
        assert len(df) == 2
        # Sorted by tenor_months
        assert df.iloc[0]["tenor_months"] == 3
        assert df.iloc[1]["tenor_months"] == 12

    def test_load_valid_deal_cash_flows(self, tmp_path):
        csv_file = tmp_path / "deal.csv"
        csv_file.write_text(
            "period,equity_invested_lcu,distribution_lcu,exit_proceeds_lcu\n"
            "0,10000000,0,0\n"
            "12,0,500000,0\n"
            "24,0,500000,0\n"
            "36,0,500000,15000000\n"
        )
        df = load_deal_cash_flows(csv_file)
        assert len(df) == 4
        assert df.iloc[0]["period"] == 0
        assert df.iloc[3]["exit_proceeds_lcu"] == 15000000

    def test_load_deal_cash_flows_missing_period_zero(self, tmp_path):
        csv_file = tmp_path / "deal_bad.csv"
        csv_file.write_text(
            "period,equity_invested_lcu,distribution_lcu,exit_proceeds_lcu\n"
            "1,10000000,0,0\n"
        )
        with pytest.raises(DataValidationError, match="period 0"):
            load_deal_cash_flows(csv_file)

    def test_load_valid_deal_config(self, tmp_path):
        yaml_file = tmp_path / "deal.yaml"
        yaml_content = {
            "country": "Kenya",
            "close_date": "2026-03-31",
            "hurdle_usd_irr": 0.18,
            "repatriation_rate": "official",
            "repatriation_blend_weight": 0.5,
            "blocked_funds_haircut_pct": 0.0,
            "withholding_tax_pct": 0.05,
        }
        with open(yaml_file, "w") as f:
            yaml.safe_dump(yaml_content, f)

        cfg = load_deal_config(yaml_file)
        assert cfg.country == "Kenya"
        assert cfg.close_date == date(2026, 3, 31)
        assert cfg.hurdle_usd_irr == 0.18
        assert cfg.repatriation_rate == RepatriationRateMode.OFFICIAL
