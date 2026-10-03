"""Parity: Hard-Currency Return Model for Frontier Deals."""

__version__ = "0.1.0"

from parity.schemas import (
    DataFileNotFoundError,
    DataValidationError,
    DealCashFlowRow,
    DealCashFlowSchema,
    DealConfig,
    EventType,
    FXEventRow,
    FXEventSchema,
    FXPanelRow,
    FXPanelSchema,
    ForwardRow,
    ForwardSchema,
    MissingColumnError,
    ParityError,
    RepatriationRateMode,
)
from parity.loaders import (
    load_deal_cash_flows,
    load_deal_config,
    load_forwards,
    load_fx_events,
    load_fx_panel,
    load_priors,
)
from parity.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_feature_matrix,
    compute_devaluation_events,
    compute_forward_targets,
    compute_imf_review_off_track,
    compute_parallel_premium,
    compute_rate_differential,
    compute_rer_misalignment,
    generate_synthetic_panel,
)
from parity.devaluation import DevaluationModel, sigmoid
from parity.magnitude import MagnitudeModel
from parity.fxpaths import SimulationResult, simulate_fx_paths
from parity.deal import (
    DealAnalysisResult,
    analyze_deal,
    calculate_irr_single,
    calculate_moic_single,
    convert_lcu_flows_to_usd,
)
from parity.hedging import (
    HedgeAnalysisResult,
    analyze_hedging,
    get_forward_rates_for_deal,
)
from parity.scenarios import (
    load_scenarios,
    run_scenario,
    run_scenario_switchboard,
)
from parity.report import (
    compute_top_risk_drivers,
    generate_excel_report,
    generate_markdown_report,
    save_report,
)
from parity.cli import app


__all__ = [
    "ParityError",
    "DataFileNotFoundError",
    "MissingColumnError",
    "DataValidationError",
    "EventType",
    "RepatriationRateMode",
    "FXPanelRow",
    "FXPanelSchema",
    "FXEventRow",
    "FXEventSchema",
    "ForwardRow",
    "ForwardSchema",
    "DealCashFlowRow",
    "DealCashFlowSchema",
    "DealConfig",
    "load_fx_panel",
    "load_fx_events",
    "load_forwards",
    "load_deal_cash_flows",
    "load_deal_config",
    "load_priors",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "compute_parallel_premium",
    "compute_rer_misalignment",
    "compute_rate_differential",
    "compute_imf_review_off_track",
    "compute_devaluation_events",
    "compute_forward_targets",
    "build_feature_matrix",
    "generate_synthetic_panel",
    "DevaluationModel",
    "sigmoid",
    "MagnitudeModel",
    "SimulationResult",
    "simulate_fx_paths",
    "DealAnalysisResult",
    "analyze_deal",
    "calculate_irr_single",
    "calculate_moic_single",
    "convert_lcu_flows_to_usd",
    "HedgeAnalysisResult",
    "analyze_hedging",
    "get_forward_rates_for_deal",
    "load_scenarios",
    "run_scenario",
    "run_scenario_switchboard",
    "compute_top_risk_drivers",
    "generate_markdown_report",
    "generate_excel_report",
    "save_report",
    "app",
]
