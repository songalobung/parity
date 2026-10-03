"""Scenario switchboard executing macroeconomic and FX regime stress scenarios."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

import pandas as pd

from parity.deal import DealAnalysisResult, analyze_deal
from parity.devaluation import DevaluationModel
from parity.fxpaths import simulate_fx_paths
from parity.magnitude import MagnitudeModel
from parity.schemas import DataFileNotFoundError, DataValidationError, DealConfig

DEFAULT_SCENARIOS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "scenarios.yaml"
)


def load_scenarios(
    scenarios_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load scenario definitions from scenarios.yaml.

    Parameters
    ----------
    scenarios_path : Optional[Union[str, Path]], default None
        Path to scenarios YAML configuration.

    Returns
    -------
    Dict[str, Any]
        Dictionary of scenario specifications.
    """
    path = Path(scenarios_path) if scenarios_path is not None else DEFAULT_SCENARIOS_PATH
    if not path.is_file():
        raise DataFileNotFoundError(
            file_path=path,
            expected_columns=["scenarios", "base", "step_devaluation"],
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise DataValidationError(f"Failed to parse scenarios file '{path}': {exc}") from exc

    if not isinstance(data, dict) or "scenarios" not in data:
        raise DataValidationError(f"Expected top-level 'scenarios' key in '{path}'")

    return data["scenarios"]


def run_scenario(
    scenario_name: str,
    deal_df: pd.DataFrame,
    deal_config: DealConfig,
    base_features: Optional[Dict[str, float]] = None,
    spot_rate: float = 100.0,
    scenarios_path: Optional[Union[str, Path]] = None,
    devaluation_model: Optional[DevaluationModel] = None,
    magnitude_model: Optional[MagnitudeModel] = None,
    n_paths: int = 10000,
    seed: Optional[int] = 42,
) -> DealAnalysisResult:
    """Run deal stress testing pipeline under a specific named scenario.

    Parameters
    ----------
    scenario_name : str
        Scenario identifier (e.g. 'base', 'imf_program_lost', 'step_devaluation').
    deal_df : pd.DataFrame
        Deal cash flow dataframe.
    deal_config : DealConfig
        Deal configuration.
    base_features : Optional[Dict[str, float]], default None
        Baseline macro features dictionary.
    spot_rate : float, default 100.0
        Starting spot exchange rate.
    scenarios_path : Optional[Union[str, Path]], default None
        Path to scenarios.yaml.
    devaluation_model : Optional[DevaluationModel], default None
        Devaluation model instance.
    magnitude_model : Optional[MagnitudeModel], default None
        Magnitude model instance.
    n_paths : int, default 10000
        Number of Monte Carlo paths.
    seed : Optional[int], default 42
        Random seed for reproducibility.

    Returns
    -------
    DealAnalysisResult
        Results under this scenario.
    """
    all_scenarios = load_scenarios(scenarios_path)
    if scenario_name not in all_scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. Available: {list(all_scenarios.keys())}"
        )

    scen_def = all_scenarios[scenario_name]
    overrides = scen_def.get("overrides", {})

    # Copy features and apply overrides
    features = dict(base_features or {})

    if "imf_program" in overrides:
        features["imf_program"] = int(overrides["imf_program"])
    if "imf_review_on_track" in overrides:
        review_on = int(overrides["imf_review_on_track"])
        prog = int(features.get("imf_program", 1))
        features["imf_review_off_track"] = 1 if (prog == 1 and review_on == 0) else 0

    if "parallel_premium_delta" in overrides:
        current_prem = float(features.get("parallel_premium", 0.0))
        features["parallel_premium"] = max(0.0, current_prem + float(overrides["parallel_premium_delta"]))

    # Simulation parameter overrides
    forced_jump_m = overrides.get("force_jump_month", None)
    if forced_jump_m is not None:
        forced_jump_m = int(forced_jump_m)

    forced_jump_s = overrides.get("force_jump_size", None)
    if forced_jump_s is not None:
        forced_jump_s = float(forced_jump_s)

    forced_ann_crawl = overrides.get("forced_annual_depreciation", None)
    if forced_ann_crawl is not None:
        forced_ann_crawl = float(forced_ann_crawl)

    post_jump_retrace = overrides.get("post_jump_retracement", None)
    if post_jump_retrace is not None:
        post_jump_retrace = float(post_jump_retrace)

    max_p = int(deal_df["period"].max())
    sim_months = max(60, max_p)

    sim_result = simulate_fx_paths(
        spot_rate=spot_rate,
        n_months=sim_months,
        n_paths=n_paths,
        devaluation_model=devaluation_model,
        magnitude_model=magnitude_model,
        features_obs=features,
        forced_jump_month=forced_jump_m,
        forced_jump_size=forced_jump_s,
        forced_annual_depreciation=forced_ann_crawl,
        post_jump_retracement=post_jump_retrace,
        seed=seed,
    )

    parallel_prem = float(features.get("parallel_premium", 0.0))
    return analyze_deal(
        deal_df=deal_df,
        deal_config=deal_config,
        sim_result=sim_result,
        parallel_premium=parallel_prem,
    )


def run_scenario_switchboard(
    deal_df: pd.DataFrame,
    deal_config: DealConfig,
    base_features: Optional[Dict[str, float]] = None,
    spot_rate: float = 100.0,
    selected_scenarios: Optional[List[str]] = None,
    scenarios_path: Optional[Union[str, Path]] = None,
    n_paths: int = 10000,
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, Dict[str, DealAnalysisResult]]:
    """Execute scenario switchboard across multiple scenarios and construct comparison table.

    Parameters
    ----------
    deal_df : pd.DataFrame
        Deal cash flows.
    deal_config : DealConfig
        Deal configuration.
    base_features : Optional[Dict[str, float]], default None
        Baseline macro features.
    spot_rate : float, default 100.0
        Starting spot exchange rate.
    selected_scenarios : Optional[List[str]], default None
        List of scenario names to evaluate. If None, runs all available in scenarios.yaml.
    scenarios_path : Optional[Union[str, Path]], default None
        Path to scenarios.yaml.
    n_paths : int, default 10000
        Number of simulated paths.
    seed : Optional[int], default 42
        RNG seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, DealAnalysisResult]]
        Formatted comparison DataFrame and map of {scenario_name: DealAnalysisResult}.
    """
    all_scenarios = load_scenarios(scenarios_path)
    if selected_scenarios is None:
        selected_scenarios = list(all_scenarios.keys())

    results_map: Dict[str, DealAnalysisResult] = {}
    rows: List[Dict[str, Any]] = []

    for name in selected_scenarios:
        if name not in all_scenarios:
            raise ValueError(f"Unknown scenario '{name}'. Available: {list(all_scenarios.keys())}")

        desc = all_scenarios[name].get("description", "")
        res = run_scenario(
            scenario_name=name,
            deal_df=deal_df,
            deal_config=deal_config,
            base_features=base_features,
            spot_rate=spot_rate,
            scenarios_path=scenarios_path,
            n_paths=n_paths,
            seed=seed,
        )
        results_map[name] = res

        rows.append({
            "scenario": name,
            "description": desc,
            "headline_lcu_irr": res.headline_lcu_irr,
            "headline_lcu_moic": res.headline_lcu_moic,
            "usd_irr_mean": res.usd_irr_mean,
            "usd_irr_median": res.usd_irr_median,
            "usd_irr_p10": res.usd_irr_p10,
            "usd_irr_p90": res.usd_irr_p90,
            "usd_moic_mean": res.usd_moic_mean,
            "usd_moic_median": res.usd_moic_median,
            "p_below_hurdle": res.p_below_hurdle,
            "expected_fx_drag": res.expected_fx_drag,
            "breakeven_lcu_irr_50": res.breakeven_lcu_irr_50,
            "breakeven_lcu_irr_75": res.breakeven_lcu_irr_75,
        })

    comparison_df = pd.DataFrame(rows)
    return comparison_df, results_map
