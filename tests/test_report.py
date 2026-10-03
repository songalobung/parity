"""Unit tests for report generation (Markdown and Excel)."""

import datetime as dt
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from parity.deal import analyze_deal
from parity.devaluation import DevaluationModel
from parity.hedging import analyze_hedging
from parity.loaders import load_deal_cash_flows, load_deal_config, load_forwards
from parity.report import (
    compute_top_risk_drivers,
    generate_excel_report,
    generate_markdown_report,
    save_report,
)
from parity.scenarios import run_scenario_switchboard


@pytest.fixture
def sample_deal_data():
    deal_df = pd.DataFrame({
        "period": [0, 12, 24, 36],
        "date": ["2026-01-31", "2027-01-31", "2028-01-31", "2029-01-31"],
        "equity_invested_lcu": [100.0, 0.0, 0.0, 0.0],
        "distribution_lcu": [0.0, 10.0, 10.0, 0.0],
        "exit_proceeds_lcu": [0.0, 0.0, 0.0, 150.0],
    })
    deal_cfg = load_deal_config("data/templates/deal.yaml")
    deal_res = analyze_deal(deal_df=deal_df, deal_config=deal_cfg, spot_rate=100.0, n_paths=200, seed=42)
    return deal_df, deal_cfg, deal_res


class TestReportGeneration:
    def test_compute_top_risk_drivers(self):
        features = {
            "parallel_premium": 0.25,
            "reserves_months_imports": 2.0,
            "rer_misalignment": 0.15,
            "imf_review_off_track": 1,
            "current_account_pct_gdp": -0.05,
        }
        drivers = compute_top_risk_drivers(features, n_top=3)
        assert len(drivers) == 3
        # Ensure drivers are sorted descending by contribution
        assert drivers[0][1] >= drivers[1][1] >= drivers[2][1]
        assert all(len(d) == 3 for d in drivers)

    def test_generate_markdown_report_content(self, sample_deal_data):
        deal_df, deal_cfg, deal_res = sample_deal_data
        features = {"parallel_premium": 0.15, "reserves_months_imports": 3.0}
        drivers = compute_top_risk_drivers(features, n_top=3)
        hedge_res = analyze_hedging(
            deal_df=deal_df,
            deal_config=deal_cfg,
            forwards_df=None,
            spot_rate=100.0,
            unhedged_result=deal_res,
        )

        md = generate_markdown_report(
            deal_res=deal_res,
            deal_config=deal_cfg,
            dev_prob_12m=0.28,
            dev_prob_24m=0.45,
            risk_drivers=drivers,
            hedge_res=hedge_res,
            as_of_date=dt.date(2026, 9, 30),
            country="Nigeria",
        )

        # Check required SPEC sections
        assert "Parity Stress-Test Report: Nigeria" in md
        assert "Data Freshness Warning" in md
        assert "28.0%" in md  # 12m prob
        assert "45.0%" in md  # 24m prob
        assert "Annualized IRR" in md
        assert "Expected FX Drag" in md
        assert "Hurdle Shortfall Risk" in md
        assert "Required Local-Currency IRR (Breakeven)" in md
        assert "Hedging Comparison" in md
        assert "Top Drivers of Devaluation Risk" in md
        assert "Model Limitations & Governance Disclaimers" in md
        assert "Sparse Historical Devaluation Episodes" in md

    def test_generate_excel_report(self, sample_deal_data, tmp_path):
        deal_df, deal_cfg, deal_res = sample_deal_data
        features = {"parallel_premium": 0.20, "reserves_months_imports": 2.5}
        drivers = compute_top_risk_drivers(features, n_top=3)
        hedge_res = analyze_hedging(
            deal_df=deal_df,
            deal_config=deal_cfg,
            forwards_df=None,
            spot_rate=100.0,
            unhedged_result=deal_res,
        )

        comp_df, _ = run_scenario_switchboard(
            deal_df=deal_df,
            deal_config=deal_cfg,
            base_features=features,
            selected_scenarios=["base", "step_devaluation"],
            n_paths=100,
            seed=42,
        )

        out_xlsx = tmp_path / "test_report.xlsx"
        saved = generate_excel_report(
            deal_res=deal_res,
            deal_config=deal_cfg,
            dev_prob_12m=0.32,
            dev_prob_24m=0.50,
            risk_drivers=drivers,
            deal_df=deal_df,
            hedge_res=hedge_res,
            comparison_df=comp_df,
            as_of_date="2026-09-30",
            country="Nigeria",
            output_path=out_xlsx,
        )

        assert saved.is_file()
        assert saved.stat().st_size > 0

        # Load back with openpyxl and verify sheets and charts
        wb = openpyxl.load_workbook(saved)
        sheet_names = wb.sheetnames
        assert "Executive Summary" in sheet_names
        assert "Scenario Stress-Testing" in sheet_names
        assert "FX Simulation Paths" in sheet_names
        assert "Cash Flows" in sheet_names
        assert "Governance & Disclaimers" in sheet_names

        # Verify charts were created in Scenario and FX sheets
        ws_scen = wb["Scenario Stress-Testing"]
        assert len(ws_scen._charts) >= 1

        ws_fx = wb["FX Simulation Paths"]
        assert len(ws_fx._charts) >= 1

    def test_save_report_dispatch(self, sample_deal_data, tmp_path):
        deal_df, deal_cfg, deal_res = sample_deal_data
        drivers = [("parallel_premium", 0.5, "Spread")]

        # Markdown dispatch
        md_path = tmp_path / "report.md"
        saved_md = save_report(
            deal_res=deal_res,
            deal_config=deal_cfg,
            dev_prob_12m=0.20,
            dev_prob_24m=0.35,
            risk_drivers=drivers,
            deal_df=deal_df,
            output_path=md_path,
        )
        assert saved_md.suffix == ".md"
        assert saved_md.is_file()
        with open(saved_md, "r", encoding="utf-8") as f:
            content = f.read()
            assert "# Parity Stress-Test Report" in content

        # Excel dispatch
        xlsx_path = tmp_path / "report.xlsx"
        saved_xlsx = save_report(
            deal_res=deal_res,
            deal_config=deal_cfg,
            dev_prob_12m=0.20,
            dev_prob_24m=0.35,
            risk_drivers=drivers,
            deal_df=deal_df,
            output_path=xlsx_path,
        )
        assert saved_xlsx.suffix == ".xlsx"
        assert saved_xlsx.is_file()
