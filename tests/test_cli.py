"""Unit tests for the Parity CLI."""

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from parity.cli import app
from parity.features import generate_synthetic_panel

runner = CliRunner()


@pytest.fixture
def sample_panel_csv(tmp_path):
    panel_df, _ = generate_synthetic_panel(n_countries=2, n_months=36, seed=42)
    panel_file = tmp_path / "fx_panel.csv"
    panel_df.to_csv(panel_file, index=False)
    return panel_file


@pytest.fixture
def sample_deal_csv(tmp_path):
    deal_df = pd.DataFrame({
        "period": [0, 12, 24, 36],
        "date": ["2026-01-31", "2027-01-31", "2028-01-31", "2029-01-31"],
        "equity_invested_lcu": [100.0, 0.0, 0.0, 0.0],
        "distribution_lcu": [0.0, 10.0, 10.0, 0.0],
        "exit_proceeds_lcu": [0.0, 0.0, 0.0, 150.0],
    })
    deal_file = tmp_path / "deal.csv"
    deal_df.to_csv(deal_file, index=False)
    return deal_file


class TestParityCLI:
    def test_cli_fit(self, sample_panel_csv):
        result = runner.invoke(
            app,
            [
                "fit",
                "--panel",
                str(sample_panel_csv),
                "--horizon",
                "12",
            ],
        )
        assert result.exit_code == 0
        assert "DEVALUATION MODEL FIT SUMMARY" in result.output
        assert "Intercept:" in result.output
        assert "Brier Score:" in result.output

    def test_cli_devalue(self, sample_panel_csv):
        result = runner.invoke(
            app,
            [
                "devalue",
                "--country",
                "Country_0",
                "--panel",
                str(sample_panel_csv),
            ],
        )
        assert result.exit_code == 0
        assert "DEVALUATION HAZARD ASSESSMENT: COUNTRY_0" in result.output
        assert "12-Month Devaluation Probability:" in result.output
        assert "TOP MACRO RISK DRIVERS:" in result.output

    def test_cli_deal(self, sample_deal_csv, sample_panel_csv):
        result = runner.invoke(
            app,
            [
                "deal",
                "--deal",
                str(sample_deal_csv),
                "--config",
                "data/templates/deal.yaml",
                "--panel",
                str(sample_panel_csv),
                "--paths",
                "100",
            ],
        )
        assert result.exit_code == 0
        assert "PARITY DEAL STRESS-TEST RESULTS: NIGERIA" in result.output
        assert "Annualized IRR" in result.output
        assert "Expected FX Drag:" in result.output
        assert "HEDGING SUMMARY:" in result.output

    def test_cli_compare(self, sample_deal_csv, sample_panel_csv):
        result = runner.invoke(
            app,
            [
                "compare",
                "--deal",
                str(sample_deal_csv),
                "--config",
                "data/templates/deal.yaml",
                "--panel",
                str(sample_panel_csv),
                "--scenarios",
                "base,step_devaluation",
                "--paths",
                "50",
            ],
        )
        assert result.exit_code == 0
        assert "SCENARIO COMPARISON TABLE: NIGERIA" in result.output
        assert "base" in result.output
        assert "step_devaluation" in result.output

    def test_cli_report_markdown(self, sample_deal_csv, sample_panel_csv, tmp_path):
        out_file = tmp_path / "test_report.md"
        result = runner.invoke(
            app,
            [
                "report",
                "--deal",
                str(sample_deal_csv),
                "--config",
                "data/templates/deal.yaml",
                "--panel",
                str(sample_panel_csv),
                "--out",
                str(out_file),
                "--paths",
                "100",
            ],
        )
        assert result.exit_code == 0
        assert "Report generated successfully:" in result.output
        assert out_file.is_file()
        content = out_file.read_text(encoding="utf-8")
        assert "# Parity Stress-Test Report: Nigeria" in content

    def test_cli_report_excel(self, sample_deal_csv, sample_panel_csv, tmp_path):
        out_file = tmp_path / "test_deal_report.xlsx"
        result = runner.invoke(
            app,
            [
                "report",
                "--deal",
                str(sample_deal_csv),
                "--config",
                "data/templates/deal.yaml",
                "--panel",
                str(sample_panel_csv),
                "--out",
                str(out_file),
                "--paths",
                "100",
            ],
        )
        assert result.exit_code == 0
        assert "Report generated successfully:" in result.output
        assert out_file.is_file()
        assert out_file.stat().st_size > 1000
