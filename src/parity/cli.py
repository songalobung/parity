"""Command-line interface (CLI) for Parity.

Provides commands to fit devaluation models, evaluate country devaluation risk,
stress-test deal returns, compare multi-scenario paths, and generate investment reports
in Markdown and institutional Excel format.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import typer

from parity.deal import DealAnalysisResult, analyze_deal
from parity.devaluation import DevaluationModel
from parity.features import (
    build_feature_matrix,
    compute_imf_review_off_track,
    compute_parallel_premium,
    compute_rate_differential,
    compute_rer_misalignment,
)
from parity.hedging import HedgeAnalysisResult, analyze_hedging
from parity.loaders import (
    load_deal_cash_flows,
    load_deal_config,
    load_forwards,
    load_fx_events,
    load_fx_panel,
)
from parity.magnitude import MagnitudeModel
from parity.report import (
    compute_top_risk_drivers,
    generate_excel_report,
    generate_markdown_report,
    save_report,
)
from parity.scenarios import run_scenario, run_scenario_switchboard
from parity.schemas import DealConfig

app = typer.Typer(
    name="parity",
    help="Parity: Hard-Currency Return Model for Frontier Deals.",
    add_completion=False,
)


def _extract_country_macro_features(
    panel_df: Optional[pd.DataFrame],
    country: str,
    as_of_date: Optional[Union[str, dt.date]] = None,
) -> Tuple[float, Dict[str, float], Optional[dt.date]]:
    """Extract spot rate and macro features from panel DataFrame for a country.

    If panel is None or country is missing, returns sensible defaults.
    """
    default_spot = 100.0
    default_features: Dict[str, float] = {
        "parallel_premium": 0.05,
        "reserves_months_imports": 4.0,
        "reserves_change_3m_pct": -0.02,
        "current_account_pct_gdp": -0.03,
        "rer_misalignment": 0.05,
        "rate_differential": 0.08,
        "imf_program": 1,
        "imf_review_off_track": 0,
    }

    if panel_df is None or panel_df.empty:
        return default_spot, default_features, None

    c_mask = panel_df["country"].astype(str).str.lower() == country.lower()
    c_df = panel_df[c_mask].copy()

    if c_df.empty:
        return default_spot, default_features, None

    c_df["date"] = pd.to_datetime(c_df["date"])
    c_df = c_df.sort_values("date")

    if as_of_date is not None:
        target_dt = pd.to_datetime(as_of_date)
        past_df = c_df[c_df["date"] <= target_dt]
        row = past_df.iloc[-1] if not past_df.empty else c_df.iloc[-1]
    else:
        row = c_df.iloc[-1]

    obs_date = row["date"].date() if hasattr(row["date"], "date") else row["date"]
    spot = float(row["official_rate"])
    parallel_rate = float(row["parallel_rate"]) if pd.notna(row.get("parallel_rate")) else None
    reserves_m = float(row.get("reserves_months_imports", 4.0))
    reserves_chg = float(row.get("reserves_change_3m_pct", 0.0))
    ca_gdp = float(row.get("current_account_pct_gdp", -0.02))
    dom_rate = float(row.get("domestic_policy_rate", 0.12))
    us_rate = float(row.get("us_policy_rate", 0.05))
    imf_prog = int(row.get("imf_program", 0))
    imf_rev = int(row.get("imf_review_on_track", 1)) if pd.notna(row.get("imf_review_on_track")) else None
    rer_idx = float(row.get("rer_index", 100.0)) if pd.notna(row.get("rer_index")) else None

    features: Dict[str, float] = {
        "parallel_premium": compute_parallel_premium(spot, parallel_rate),
        "reserves_months_imports": reserves_m,
        "reserves_change_3m_pct": reserves_chg,
        "current_account_pct_gdp": ca_gdp,
        "rer_misalignment": compute_rer_misalignment(rer_idx),
        "rate_differential": compute_rate_differential(dom_rate, us_rate),
        "imf_program": imf_prog,
        "imf_review_off_track": compute_imf_review_off_track(imf_prog, imf_rev),
    }

    return spot, features, obs_date


@app.command()
def fit(
    panel: Path = typer.Option(..., "--panel", "-p", help="Path to fx_panel.csv file."),
    events: Optional[Path] = typer.Option(None, "--events", "-e", help="Path to fx_events.csv file."),
    horizon: int = typer.Option(12, "--horizon", "-h", help="Prediction horizon in months (12 or 24)."),
    priors: Optional[Path] = typer.Option(None, "--priors", help="Path to custom priors.yaml."),
) -> None:
    """Fit devaluation logistic regression model on historical FX panel data."""
    typer.echo(f"Loading FX panel from: {panel}")
    panel_df = load_fx_panel(panel)

    events_df = None
    if events is not None and events.is_file():
        typer.echo(f"Loading external events from: {events}")
        events_df = load_fx_events(events)

    typer.echo(f"Building feature matrix (horizon = {horizon} months)...")
    feat_df = build_feature_matrix(
        panel_df=panel_df,
        events_df=events_df,
        priors_path=priors,
    )

    target_col = f"devaluation_{horizon}m"
    n_events = int(feat_df[target_col].sum()) if target_col in feat_df else int(feat_df.get("is_devaluation_event", pd.Series(0)).sum())
    typer.echo(f"Total observations: {len(feat_df)} | Total devaluation events: {n_events}")

    model = DevaluationModel(horizon=horizon, priors_path=priors)
    typer.echo("Fitting model (prior-regularized L2 logistic regression)...")
    model.fit(feat_df)

    eval_res = model.evaluate(feat_df)

    typer.echo("\n" + "=" * 65)
    typer.echo(f"DEVALUATION MODEL FIT SUMMARY ({horizon}-MONTH HORIZON)")
    typer.echo("=" * 65)
    typer.echo(f"Intercept: {model.intercept_:.4f} (Prior: {model.prior_intercept:.4f})")
    typer.echo(f"Brier Score: {eval_res.get('brier_score', 0.0):.4f}")
    if "auc" in eval_res:
        typer.echo(f"ROC-AUC:     {eval_res['auc']:.4f}")
    typer.echo("-" * 65)
    typer.echo(f"{'Feature':<28} {'Fitted Coef':>12} {'Prior Coef':>12}")
    typer.echo("-" * 65)
    for feat, coef in model.coef_.items():
        prior_val = model.priors_.get(f"coef_{feat}", np.nan)
        typer.echo(f"{feat:<28} {coef:>12.4f} {prior_val:>12.4f}")
    typer.echo("=" * 65)
    typer.echo("Fit complete.")


@app.command()
def devalue(
    country: str = typer.Option(..., "--country", "-c", help="Country name (e.g. Nigeria)."),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="As-of evaluation date (YYYY-MM-DD)."),
    panel: Optional[Path] = typer.Option(None, "--panel", "-p", help="Path to fx_panel.csv file."),
    scenario: str = typer.Option("base", "--scenario", "-s", help="Scenario name from scenarios.yaml."),
    scenarios_config: Optional[Path] = typer.Option(None, "--scenarios-config", help="Path to scenarios.yaml."),
    priors: Optional[Path] = typer.Option(None, "--priors", help="Path to priors.yaml."),
) -> None:
    """Evaluate 12-month and 24-month devaluation odds and top risk drivers."""
    panel_df = None
    if panel is not None and panel.is_file():
        panel_df = load_fx_panel(panel)
    elif Path("data/templates/fx_panel.csv").is_file():
        panel_df = load_fx_panel("data/templates/fx_panel.csv")

    spot, features, obs_date = _extract_country_macro_features(panel_df, country, date)

    # Initialize models
    model_12 = DevaluationModel(horizon=12, priors_path=priors)
    model_24 = DevaluationModel(horizon=24, priors_path=priors)

    prob_12 = model_12.predict_proba(features)
    prob_24 = model_24.predict_proba(features)

    drivers = compute_top_risk_drivers(features, model_12, n_top=3)

    typer.echo("\n" + "=" * 65)
    typer.echo(f"DEVALUATION HAZARD ASSESSMENT: {country.upper()}")
    typer.echo("=" * 65)
    typer.echo(f"As-of Date: {obs_date or date or 'Latest'} | Spot Rate: {spot:.2f} | Scenario: {scenario}")
    typer.echo("-" * 65)
    typer.echo(f"  * 12-Month Devaluation Probability: {prob_12 * 100:.1f}%")
    typer.echo(f"  * 24-Month Devaluation Probability: {prob_24 * 100:.1f}%")
    typer.echo("-" * 65)
    typer.echo("TOP MACRO RISK DRIVERS:")
    for rank, (feat, contrib, desc) in enumerate(drivers, 1):
        val = features.get(feat, 0.0)
        typer.echo(f"  {rank}. {feat} = {val:+.2f} (log-odds impact: {contrib:+.2f})")
        typer.echo(f"     -> {desc}")
    typer.echo("=" * 65 + "\n")


@app.command()
def deal(
    deal: Path = typer.Option(Path("data/templates/deal.csv"), "--deal", help="Path to deal cash flows CSV."),
    config: Path = typer.Option(Path("data/templates/deal.yaml"), "--config", help="Path to deal configuration YAML."),
    panel: Optional[Path] = typer.Option(None, "--panel", help="Path to fx_panel.csv."),
    forwards: Optional[Path] = typer.Option(None, "--forwards", help="Path to forwards.csv."),
    scenario: str = typer.Option("base", "--scenario", "-s", help="Scenario name."),
    scenarios_config: Optional[Path] = typer.Option(None, "--scenarios-config", help="Path to scenarios.yaml."),
    priors: Optional[Path] = typer.Option(None, "--priors", help="Path to priors.yaml."),
    paths: int = typer.Option(10000, "--paths", "-n", help="Number of Monte Carlo paths."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
) -> None:
    """Stress-test deal cash flows against simulated FX trajectories."""
    deal_df = load_deal_cash_flows(deal)
    deal_cfg = load_deal_config(config)

    panel_df = None
    if panel is not None and panel.is_file():
        panel_df = load_fx_panel(panel)
    elif Path("data/templates/fx_panel.csv").is_file():
        panel_df = load_fx_panel("data/templates/fx_panel.csv")

    forwards_df = None
    if forwards is not None and forwards.is_file():
        forwards_df = load_forwards(forwards)
    elif Path("data/templates/forwards.csv").is_file():
        forwards_df = load_forwards("data/templates/forwards.csv")

    spot, features, _ = _extract_country_macro_features(panel_df, deal_cfg.country, deal_cfg.close_date)

    typer.echo(f"Running simulation for '{deal_cfg.country}' under scenario '{scenario}' ({paths:,} paths)...")
    res = run_scenario(
        scenario_name=scenario,
        deal_df=deal_df,
        deal_config=deal_cfg,
        base_features=features,
        spot_rate=spot,
        scenarios_path=scenarios_config,
        n_paths=paths,
        seed=seed,
    )

    hedge_res = analyze_hedging(
        deal_df=deal_df,
        deal_config=deal_cfg,
        unhedged_result=res,
        spot_rate=spot,
        forwards_df=forwards_df,
        local_rate_pct=(features.get("rate_differential", 0.08) + 0.05) * 100.0,
        us_rate_pct=5.0,
        parallel_premium=features.get("parallel_premium", 0.0),
    )

    typer.echo("\n" + "=" * 70)
    typer.echo(f"PARITY DEAL STRESS-TEST RESULTS: {deal_cfg.country.upper()}")
    typer.echo("=" * 70)
    typer.echo(f"Close Date: {deal_cfg.close_date} | Repatriation: {deal_cfg.repatriation_rate} | Hurdle: {deal_cfg.hurdle_usd_irr * 100:.1f}%")
    typer.echo("-" * 70)
    typer.echo(f"{'Metric':<25} {'LCU Headline':>14} {'USD Expected':>14} {'USD Median':>14}")
    typer.echo("-" * 70)
    typer.echo(f"{'Annualized IRR':<25} {res.headline_lcu_irr * 100:>13.1f}% {res.usd_irr_mean * 100:>13.1f}% {res.usd_irr_median * 100:>13.1f}%")
    typer.echo(f"{'MOIC':<25} {res.headline_lcu_moic:>13.2f}x {res.usd_moic_mean:>13.2f}x {res.usd_moic_median:>13.2f}x")
    typer.echo("-" * 70)
    typer.echo("STRESS PERCENTILES & RISK:")
    typer.echo(f"  * USD IRR P10 (Stress):     {res.usd_irr_p10 * 100:.1f}%")
    typer.echo(f"  * USD IRR P90 (Upside):     {res.usd_irr_p90 * 100:.1f}%")
    typer.echo(f"  * Expected FX Drag:         {res.expected_fx_drag * 100:.1f}% ({res.expected_fx_drag * 10000:.0f} bps)")
    typer.echo(f"  * P(USD IRR < Hurdle):      {res.p_below_hurdle * 100:.1f}%")
    typer.echo(f"  * Breakeven LCU IRR (50%):  {res.breakeven_lcu_irr_50 * 100:.1f}%")
    typer.echo(f"  * Breakeven LCU IRR (75%):  {res.breakeven_lcu_irr_75 * 100:.1f}%")
    typer.echo("-" * 70)
    typer.echo("HEDGING SUMMARY:")
    typer.echo(f"  * Fully Hedged USD IRR:     {hedge_res.hedged_usd_irr * 100:.1f}% (Source: {hedge_res.forward_source})")
    typer.echo(f"  * Annual Hedge Cost Drag:   {hedge_res.hedge_cost_drag * 100:.1f}%")
    typer.echo(f"  * Breakeven FX Move:        {hedge_res.breakeven_annual_depreciation * 100:.1f}% / yr")
    typer.echo(f"  * P(Hedge Outperforms):     {hedge_res.p_hedge_outperforms * 100:.1f}%")
    typer.echo("=" * 70 + "\n")


@app.command()
def compare(
    deal: Path = typer.Option(Path("data/templates/deal.csv"), "--deal", help="Path to deal cash flows CSV."),
    config: Path = typer.Option(Path("data/templates/deal.yaml"), "--config", help="Path to deal configuration YAML."),
    scenarios: str = typer.Option(
        "base,step_devaluation,imf_program_lost,premium_widens,slow_crawl,float_and_overshoot",
        "--scenarios",
        "-s",
        help="Comma-separated scenario names.",
    ),
    panel: Optional[Path] = typer.Option(None, "--panel", help="Path to fx_panel.csv."),
    scenarios_config: Optional[Path] = typer.Option(None, "--scenarios-config", help="Path to scenarios.yaml."),
    paths: int = typer.Option(5000, "--paths", "-n", help="Number of simulated paths."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
) -> None:
    """Compare deal performance across multiple stress scenarios."""
    deal_df = load_deal_cash_flows(deal)
    deal_cfg = load_deal_config(config)

    panel_df = None
    if panel is not None and panel.is_file():
        panel_df = load_fx_panel(panel)
    elif Path("data/templates/fx_panel.csv").is_file():
        panel_df = load_fx_panel("data/templates/fx_panel.csv")

    spot, features, _ = _extract_country_macro_features(panel_df, deal_cfg.country, deal_cfg.close_date)
    scen_list = [s.strip() for s in scenarios.split(",") if s.strip()]

    typer.echo(f"Running scenario switchboard across {len(scen_list)} scenarios...")
    comp_df, _ = run_scenario_switchboard(
        deal_df=deal_df,
        deal_config=deal_cfg,
        base_features=features,
        spot_rate=spot,
        selected_scenarios=scen_list,
        scenarios_path=scenarios_config,
        n_paths=paths,
        seed=seed,
    )

    typer.echo("\n" + "=" * 90)
    typer.echo(f"SCENARIO COMPARISON TABLE: {deal_cfg.country.upper()} (Hurdle = {deal_cfg.hurdle_usd_irr * 100:.1f}%)")
    typer.echo("=" * 90)
    header = f"{'Scenario':<20} {'Mean USD':>10} {'Median':>10} {'P10':>10} {'MOIC':>8} {'P(<Hurdle)':>12} {'FX Drag':>10}"
    typer.echo(header)
    typer.echo("-" * 90)
    for _, row in comp_df.iterrows():
        s_name = row["scenario"]
        mean_s = f"{row['usd_irr_mean'] * 100:.1f}%"
        med_s = f"{row['usd_irr_median'] * 100:.1f}%"
        p10_s = f"{row['usd_irr_p10'] * 100:.1f}%"
        moic_s = f"{row['usd_moic_mean']:.2f}x"
        ph_s = f"{row['p_below_hurdle'] * 100:.1f}%"
        drag_s = f"{row['expected_fx_drag'] * 100:.1f}%"
        typer.echo(f"{s_name:<20} {mean_s:>10} {med_s:>10} {p10_s:>10} {moic_s:>8} {ph_s:>12} {drag_s:>10}")
    typer.echo("=" * 90 + "\n")


@app.command()
def report(
    deal: Path = typer.Option(Path("data/templates/deal.csv"), "--deal", help="Path to deal cash flows CSV."),
    config: Path = typer.Option(Path("data/templates/deal.yaml"), "--config", help="Path to deal configuration YAML."),
    out: Path = typer.Option(Path("reports/deal.md"), "--out", "-o", help="Output file path (.md or .xlsx)."),
    panel: Optional[Path] = typer.Option(None, "--panel", help="Path to fx_panel.csv."),
    forwards: Optional[Path] = typer.Option(None, "--forwards", help="Path to forwards.csv."),
    scenario: str = typer.Option("base", "--scenario", "-s", help="Base scenario for main simulation."),
    scenarios: str = typer.Option(
        "base,step_devaluation,imf_program_lost,float_and_overshoot",
        "--scenarios",
        help="Scenarios to include in comparison table.",
    ),
    scenarios_config: Optional[Path] = typer.Option(None, "--scenarios-config", help="Path to scenarios.yaml."),
    priors: Optional[Path] = typer.Option(None, "--priors", help="Path to priors.yaml."),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format: 'markdown' or 'excel'."),
    paths: int = typer.Option(10000, "--paths", "-n", help="Number of Monte Carlo paths."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
) -> None:
    """Generate an institutional Investment Committee report in Markdown or Excel."""
    deal_df = load_deal_cash_flows(deal)
    deal_cfg = load_deal_config(config)

    panel_df = None
    if panel is not None and panel.is_file():
        panel_df = load_fx_panel(panel)
    elif Path("data/templates/fx_panel.csv").is_file():
        panel_df = load_fx_panel("data/templates/fx_panel.csv")

    forwards_df = None
    if forwards is not None and forwards.is_file():
        forwards_df = load_forwards(forwards)
    elif Path("data/templates/forwards.csv").is_file():
        forwards_df = load_forwards("data/templates/forwards.csv")

    spot, features, obs_date = _extract_country_macro_features(panel_df, deal_cfg.country, deal_cfg.close_date)

    typer.echo(f"Executing base simulation for '{deal_cfg.country}' ({paths:,} paths)...")
    base_res = run_scenario(
        scenario_name=scenario,
        deal_df=deal_df,
        deal_config=deal_cfg,
        base_features=features,
        spot_rate=spot,
        scenarios_path=scenarios_config,
        n_paths=paths,
        seed=seed,
    )

    # Devaluation models
    dev_model_12 = DevaluationModel(horizon=12, priors_path=priors)
    dev_model_24 = DevaluationModel(horizon=24, priors_path=priors)
    prob_12 = dev_model_12.predict_proba(features)
    prob_24 = dev_model_24.predict_proba(features)
    drivers = compute_top_risk_drivers(features, dev_model_12, n_top=3)

    # Hedging analysis
    hedge_res = analyze_hedging(
        deal_df=deal_df,
        deal_config=deal_cfg,
        unhedged_result=base_res,
        spot_rate=spot,
        forwards_df=forwards_df,
        local_rate_pct=(features.get("rate_differential", 0.08) + 0.05) * 100.0,
        us_rate_pct=5.0,
        parallel_premium=features.get("parallel_premium", 0.0),
    )

    # Scenarios comparison
    scen_list = [s.strip() for s in scenarios.split(",") if s.strip()]
    comp_df, _ = run_scenario_switchboard(
        deal_df=deal_df,
        deal_config=deal_cfg,
        base_features=features,
        spot_rate=spot,
        selected_scenarios=scen_list,
        scenarios_path=scenarios_config,
        n_paths=min(5000, paths),
        seed=seed,
    )

    # Generate and save report
    saved_path = save_report(
        deal_res=base_res,
        deal_config=deal_cfg,
        dev_prob_12m=prob_12,
        dev_prob_24m=prob_24,
        risk_drivers=drivers,
        deal_df=deal_df,
        output_path=out,
        hedge_res=hedge_res,
        comparison_df=comp_df,
        as_of_date=obs_date,
        country=deal_cfg.country,
        format_type=format,
    )

    typer.echo(f"Report generated successfully: {saved_path}")


if __name__ == "__main__":
    app()
