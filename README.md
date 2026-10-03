<div align="center">

# Parity

### Institutional Hard-Currency Return & FX Stress-Testing Engine for Frontier Deals

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Tests: 99 Passing](https://img.shields.io/badge/Tests-99%20Passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Architecture: Prior--Driven](https://img.shields.io/badge/Architecture-Prior--Driven-navy)](config/priors.yaml)
[![Excel: Interactive Charts](https://img.shields.io/badge/Excel%20Output-Charts%20%26%20KPIs-darkgreen?logo=microsoftexcel&logoColor=white)](reports/)

<p align="center">
  <b>Evaluate the real dollar performance of frontier and emerging market investments.</b><br>
  Simulate 100,000+ macroeconomic FX trajectories, model discrete devaluation hazards, incorporate parallel-market repatriation and trapped capital haircuts, and generate board-ready Investment Committee reports.
</p>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Analytical Pipeline](#architecture--analytical-pipeline)
- [Target Currencies](#target-currencies)
- [Installation](#installation)
- [Data Onboarding Guide](#data-onboarding-guide)
  - [Directory Setup & Privacy Guard](#directory-setup--privacy-guard)
  - [File 1: Monthly FX & Macro Panel (`fx_panel.csv`)](#file-1-monthly-fx--macro-panel-fx_panelcsv)
  - [File 2: Historical Devaluation Events (`fx_events.csv`)](#file-2-historical-devaluation-events-fx_eventscsv)
  - [File 3: Market Forward & NDF Quotes (`forwards.csv`)](#file-3-market-forward--ndf-quotes-forwardscsv)
  - [File 4: Deal Cash Flows (`deal.csv`)](#file-4-deal-cash-flows-dealcsv)
  - [File 5: Underwriting Terms (`deal.yaml`)](#file-5-underwriting-terms-dealyaml)
- [Running the Application (CLI Reference)](#running-the-application-cli-reference)
  - [1. Recalibrate Model Parameters (`parity fit`)](#1-recalibrate-model-parameters-parity-fit)
  - [2. Evaluate Devaluation Hazard (`parity devalue`)](#2-evaluate-devaluation-hazard-parity-devalue)
  - [3. Stress-Test Deal Cash Flows (`parity deal`)](#3-stress-test-deal-cash-flows-parity-deal)
  - [4. Multi-Scenario Switchboard (`parity compare`)](#4-multi-scenario-switchboard-parity-compare)
  - [5. Generate Board-Ready Reports (`parity report`)](#5-generate-board-ready-reports-parity-report)
- [Excel Workbook Deliverable](#excel-workbook-deliverable)
- [Python SDK Reference](#python-sdk-reference)
- [Public Data Sourcing Guide](#public-data-sourcing-guide)
- [Model Limitations & Governance Disclaimers](#model-limitations--governance-disclaimers)
- [Release Notes (v0.1.0)](#release-notes-v010)
- [Contributing & License](#contributing--license)

---

## Overview

Most private equity, venture capital, and project finance models in frontier markets evaluate returns in local currency and convert to US Dollars (USD) at the spot rate at model exit.

In reality, **headline local returns quietly evaporate** due to:
- **Discrete Currency Devaluations**: Sudden 20% to 50% policy resets that break pegs or crawling bands.
- **Parallel-Market Spreads**: Repatriating proceeds through official windows may take months or require settling at unofficial parallel rates.
- **Trapped Liquidity & Capital Controls**: Exit delays or central bank rationing impose implicit or explicit haircuts on realized capital.
- **Withholding Taxes (WHT)**: Cross-border leakage on equity distributions.

**Parity** bridges the gap between local-currency projections and realized hard-currency returns. It consumes your local cash flow schedules, evaluates macroeconomic vulnerability, simulates thousands of regime-switching FX paths, and produces hard-currency return distributions with exact risk metrics.

---

## Architecture & Analytical Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PARITY ANALYTICAL PIPELINE                      │
└────────────────────────────────────────────────────────────────────────┘

  1. Macro Inputs & Feature Matrix
     [Official / Parallel Spot Rates, Reserves Cover, RER Misalignment, CA Deficit, IMF Status]
                                  │
                                  ▼
  2. Devaluation Hazard Engine (12m & 24m)
     [Regularized Logistic Regression + Documented Priors + TimeSeriesSplit Validation]
                                  │
                                  ▼
  3. Magnitude & Regime Switch Simulator
     [Lognormal / Gamma Jumps + Baseline Peg / Crawl / Float Baselines + 2-Regime Monte Carlo]
                                  │
                                  ▼
  4. Deal Conversion & Frictions Engine
     [Official / Parallel / Blend Repatriation + Trapped Cash Haircut + Withholding Tax]
                                  │
                                  ▼
  5. Hedging Analytics (forwards.csv or CIP Proxy Fallback)
     [Hedged IRR, Carry Drag bps, Breakeven Annual Depreciation Rate]
                                  │
                                  ▼
  6. Deliverable Outputs
     ├─► Terminal CLI Dashboard
     ├─► Investment Committee Markdown Memo (.md)
     └─► Institutional 5-Tab Financial Model (.xlsx with Charts & Green/Red Indicators)
```

---

## Target Currencies

Parity is mathematical and **country-agnostic**, but was specifically designed around the structural foreign exchange dynamics of frontier African and emerging economies:

| Currency | Country | Typical FX Regime & Key Risk Dynamics |
| :--- | :--- | :--- |
| **KES** | Kenya | Managed float; import-cover reserve drains and Eurobond refinancing cycles. |
| **NGN** | Nigeria | Multi-window FX structure, NAFEM convergence, significant parallel market premiums. |
| **ETB** | Ethiopia | Transition from crawling peg to market-determined float; reserve constraints. |
| **GHS** | Ghana | Free floating with periodic stabilization under IMF ECF extended facility. |
| **EGP** | Egypt | Periodic discrete step devaluations followed by central bank crawl management. |
| **ZMW** | Zambia | Floating copper-driven currency; debt restructuring sensitivity. |
| **AOA** | Angola | Oil-revenue linked managed float with repatriation queues. |
| **UGX** | Uganda | Liquid managed float with central bank intervention smoothing. |
| **TZS** | Tanzania | Crawling band with tight administrative controls. |

---

## Installation

Parity requires **Python 3.11+**.

```powershell
# 1. Clone the repository
git clone https://github.com/songalobung/parity.git
cd parity

# 2. Initialize virtual environment (uv or standard python venv)
uv venv --python 3.11 .venv

# On Windows PowerShell:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate

# 3. Install Parity in editable mode with development dependencies
uv pip install -e ".[dev]"
```

Verify installation:
```powershell
parity --help
pytest
```

---

## Data Onboarding Guide

Parity never invents historical data. You supply your own data matching Parity's transparent schemas.

### Directory Setup & Privacy Guard

Parity includes pre-configured CSV templates in `data/templates/`. When adding proprietary or licensed data:
- Store production data in `data/` (e.g. `data/my_deal.csv`, `data/fx_panel.csv`).
- **Data Privacy Guarantee**: The repository's `.gitignore` explicitly excludes all files under `data/*` (except templates and `data/schemas.md`). Your internal deal terms and proprietary vendor series will never be committed to source control.

---

### File 1: Monthly FX & Macro Panel (`fx_panel.csv`)

Tracks monthly macroeconomic indicators for one or more countries.

**Template Location:** `data/templates/fx_panel.csv`

#### Schema & Column Definitions

| Column | Type | Nullable? | Description & Allowed Values |
| :--- | :---: | :---: | :--- |
| `country` | `string` | No | Name of the country (e.g., `Nigeria`, `Kenya`, `Egypt`). |
| `currency` | `string` | No | 3-letter ISO currency code (e.g., `NGN`, `KES`, `EGP`). |
| `date` | `date` | No | Monthly date formatted as `YYYY-MM-DD` (typically month-end). |
| `official_rate` | `float` | No | Official central bank exchange rate (LCU per USD). Must be > 0. |
| `parallel_rate` | `float` | **Yes** | Unofficial street / parallel market exchange rate. If empty, premium defaults to 0. |
| `reserves_months_imports` | `float` | No | Gross liquid foreign exchange reserves in months of imports (e.g. `3.4`). |
| `reserves_change_3m_pct` | `float` | No | 3-month percentage change in reserves as decimal (e.g. `-0.08` for -8%). |
| `current_account_pct_gdp` | `float` | No | Current account balance as percentage of GDP (e.g. `-0.035` for -3.5%). |
| `inflation_local_pct` | `float` | No | Annual headline CPI inflation rate (e.g. `0.285` for 28.5%). |
| `inflation_us_pct` | `float` | No | Annual US CPI inflation rate (e.g. `0.025` for 2.5%). |
| `policy_rate_pct` | `float` | No | Domestic central bank policy interest rate (e.g. `0.22` for 22.0%). |
| `us_rate_pct` | `float` | No | US policy rate / Fed Funds upper bound (e.g. `0.05` for 5.0%). |
| `imf_program` | `int` | No | Binary flag: `1` if an active IMF program exists; `0` otherwise. |
| `imf_review_on_track` | `int` | **Yes** | `1` if latest IMF review completed on schedule; `0` if stalled/off-track; empty if no program. |

#### Example `fx_panel.csv`:
```csv
country,currency,date,official_rate,parallel_rate,reserves_months_imports,reserves_change_3m_pct,current_account_pct_gdp,inflation_local_pct,inflation_us_pct,policy_rate_pct,us_rate_pct,imf_program,imf_review_on_track
Nigeria,NGN,2025-10-31,1480.50,1540.00,3.6,-0.04,-0.021,0.295,0.024,0.2675,0.0475,0,
Nigeria,NGN,2025-11-30,1510.00,1580.00,3.4,-0.05,-0.023,0.302,0.025,0.2675,0.0475,0,
Nigeria,NGN,2025-12-31,1545.00,1625.00,3.1,-0.08,-0.025,0.315,0.026,0.2725,0.0450,0,
Kenya,KES,2025-12-31,129.50,,4.2,0.02,-0.038,0.048,0.026,0.1200,0.0450,1,1
```

---

### File 2: Historical Devaluation Events (`fx_events.csv`)

Optional registry of discrete devaluation shocks to calibrate jump size distributions.

**Template Location:** `data/templates/fx_events.csv`

#### Schema & Column Definitions

| Column | Type | Nullable? | Allowed Values |
| :--- | :---: | :---: | :--- |
| `country` | `string` | No | Country name. |
| `date` | `date` | No | Date of the discrete devaluation event (`YYYY-MM-DD`). |
| `event_type` | `string` | No | `devaluation`, `float`, `peg_break`, or `managed_crawl_change`. |
| `size_pct` | `float` | **Yes** | Discrete percentage jump in LCU per USD (e.g. `0.35` for 35% reset). |

#### Example `fx_events.csv`:
```csv
country,date,event_type,size_pct
Nigeria,2023-06-14,float,0.45
Egypt,2024-03-06,peg_break,0.38
Ethiopia,2024-07-29,float,0.30
```

---

### File 3: Market Forward & NDF Quotes (`forwards.csv`)

Optional quotes for market forward contracts or Non-Deliverable Forwards (NDFs). If this file is omitted, Parity automatically synthesizes forward rates using the Covered Interest Parity (CIP) proxy.

**Template Location:** `data/templates/forwards.csv`

#### Example `forwards.csv`:
```csv
country,date,tenor_months,forward_rate
Nigeria,2026-01-31,12,1750.00
Nigeria,2026-01-31,24,2050.00
Nigeria,2026-01-31,36,2400.00
```

---

### File 4: Deal Cash Flows (`deal.csv`)

The cash flow schedule of your investment in local currency (LCU).

**Template Location:** `data/templates/deal.csv`

#### Rules
- Period `0` is strictly required and represents initial investment close.
- All subsequent periods represent months elapsed from close (e.g. `12`, `24`, `36`, `60`).
- Figures are in local currency units (not in thousands or millions).

#### Example `deal.csv` (3-Year Private Equity Deal):
```csv
period,date,equity_invested_lcu,distribution_lcu,exit_proceeds_lcu
0,2026-01-31,100000000.0,0.0,0.0
12,2027-01-31,0.0,8000000.0,0.0
24,2028-01-31,0.0,10000000.0,0.0
36,2029-01-31,0.0,12000000.0,165000000.0
```

---

### File 5: Underwriting Terms (`deal.yaml`)

Configuration file specifying target hurdle, cross-border tax, and repatriation mechanics.

**Template Location:** `data/templates/deal.yaml`

```yaml
country: "Nigeria"
close_date: "2026-01-31"
hurdle_usd_irr: 0.15                  # 15.0% target net USD IRR

# Repatriation FX Window
repatriation_rate: "blend"            # Options: official | parallel | blend
repatriation_blend_weight: 0.5        # 50% official window, 50% parallel market

# Frictions & Capital Controls
blocked_funds_haircut_pct: 0.05       # 5.0% penalty for central bank illiquidity / queueing
withholding_tax_pct: 0.10             # 10.0% statutory cross-border WHT on distributions
```

---

## Running the Application (CLI Reference)

Parity provides a unified command line interface via the `parity` executable.

```
Usage: parity [OPTIONS] COMMAND [ARGS]...

Parity: Hard-Currency Return Model for Frontier Deals.

Commands:
  fit      Fit devaluation logistic regression model on historical FX panel data.
  devalue  Evaluate 12-month and 24-month devaluation odds and top risk drivers.
  deal     Stress-test deal cash flows against simulated FX trajectories.
  compare  Compare deal performance across multiple stress scenarios.
  report   Generate an institutional Investment Committee report in Markdown or Excel.
```

---

### 1. Recalibrate Model Parameters (`parity fit`)

Calibrates the regularized logistic regression model against historical panel data using time-series cross-validation (`TimeSeriesSplit`), reporting fitted coefficients versus documented priors:

```powershell
parity fit --panel data/fx_panel.csv --events data/fx_events.csv --horizon 12
```

**Terminal Output:**
```
=================================================================
DEVALUATION MODEL FIT SUMMARY (12-MONTH HORIZON)
=================================================================
Intercept: -1.9421 (Prior: -2.0000)
Brier Score: 0.0841
ROC-AUC:     0.8125
-----------------------------------------------------------------
Feature                       Fitted Coef   Prior Coef
-----------------------------------------------------------------
parallel_premium                   3.8124       3.5000
reserves_cover                    -0.4215      -0.4000
reserves_change_3m                -1.4810      -1.5000
ca_pct_gdp                        -0.0820      -0.0800
rer_misalignment                   1.1890       1.2000
rate_differential                 -0.0480      -0.0500
imf_program                       -0.5890      -0.6000
imf_review_off_track               1.0420       1.0000
=================================================================
Fit complete.
```

---

### 2. Evaluate Devaluation Hazard (`parity devalue`)

Assesses 12-month and 24-month devaluation probabilities and isolates the top 3 macro risk drivers:

```powershell
parity devalue --country Nigeria --date 2026-01-31
```

**Terminal Output:**
```
=================================================================
DEVALUATION HAZARD ASSESSMENT: NIGERIA
=================================================================
As-of Date: 2026-01-31 | Spot Rate: 1545.00 | Scenario: base
-----------------------------------------------------------------
  * 12-Month Devaluation Probability: 31.8%
  * 24-Month Devaluation Probability: 49.2%
-----------------------------------------------------------------
TOP MACRO RISK DRIVERS:
  1. parallel_premium = +0.18 (log-odds impact: +0.63)
     -> Parallel market exchange rate spread over official peg
  2. reserves_months_imports = +3.10 (log-odds impact: -1.24)
     -> Depletion of foreign exchange reserves cover
  3. rer_misalignment = +0.12 (log-odds impact: +0.14)
     -> Real effective exchange rate (REER) overvaluation
=================================================================
```

---

### 3. Stress-Test Deal Cash Flows (`parity deal`)

Simulates 100,000 Monte Carlo paths and converts deal flows under the specified repatriation mechanics:

```powershell
parity deal --deal data/templates/deal.csv --config data/templates/deal.yaml --paths 100000
```

---

### 4. Multi-Scenario Switchboard (`parity compare`)

Runs the deal through predefined macro stress scenarios:

```powershell
parity compare --deal data/templates/deal.csv --config data/templates/deal.yaml \
  --scenarios base,step_devaluation,imf_program_lost,premium_widens,slow_crawl,float_and_overshoot
```

**Terminal Output:**
```
==========================================================================================
SCENARIO COMPARISON TABLE: NIGERIA (Hurdle = 15.0%)
==========================================================================================
Scenario               Mean USD     Median        P10     MOIC   P(<Hurdle)    FX Drag
------------------------------------------------------------------------------------------
base                      11.2%      10.8%      -4.2%    1.42x        68.4%      12.2%
step_devaluation           4.8%       4.2%      -8.1%    1.18x        89.1%      18.6%
imf_program_lost           7.9%       7.4%      -6.8%    1.29x        78.2%      15.5%
premium_widens             8.4%       8.1%      -5.9%    1.31x        76.5%      15.0%
slow_crawl                14.1%      13.9%       5.2%    1.52x        58.2%       9.3%
float_and_overshoot        2.1%       1.5%     -11.4%    1.08x        94.2%      21.3%
==========================================================================================
```

---

### 5. Generate Board-Ready Reports (`parity report`)

#### Option A: Markdown Investment Committee Memo (`.md`)
```powershell
parity report --deal data/templates/deal.csv --config data/templates/deal.yaml --out reports/deal_memo.md
```

#### Option B: Institutional Excel Workbook with Charts & Colors (`.xlsx`)
```powershell
parity report --deal data/templates/deal.csv --config data/templates/deal.yaml --out reports/deal_memo.xlsx
```

---

## Excel Workbook Deliverable

When exporting to `.xlsx`, Parity generates an executive financial model formatted according to institutional investment banking standards:

| Worksheet Tab | Contents & Highlights |
| :--- | :--- |
| **`Executive Summary`** | - **KPI Summary Cards**: Headline LCU IRR, Expected USD IRR, Hurdle Shortfall, FX Drag bps, Breakeven LCU IRR (75%).<br>- **Return Percentiles**: Mean, Median, P10 Stress, P90 Upside.<br>- **Green / Red Financial Formatting**: Returns meeting hurdle are formatted in bold green; negative returns or capital impairment formatted in red.<br>- **Devaluation Hazard & Top Drivers**: 12m/24m probabilities and driver attributions.<br>- **Hedging Comparison**: Unhedged vs hedged IRR and breakeven FX move. |
| **`Scenario Stress-Testing`** | - **Multi-Scenario Switchboard**: Base, Step Devaluation, IMF Lost, Spread Widening, Slow Crawl, Float Overshoot.<br>- **Embedded Bar Chart**: Visual comparison of Expected USD IRR by scenario plotted directly against your target hurdle. |
| **`FX Simulation Paths`** | - **Monthly Fan Distribution**: Month 0 through Exit percentile rates (P10, Median, P90, Mean).<br>- **Embedded Line Chart**: Currency depreciation trajectory fan illustrating stress envelopes. |
| **`Cash Flows`** | - Full period-by-period cash flow schedule (Equity Invested, Distribution, Exit Proceeds, Net Cash Flow) with conditional green indicators on distributions. |
| **`Governance & Disclaimers`** | - Documentation of model assumptions, data freshness warning, and Apache 2.0 disclaimer. |

---

## Python SDK Reference

Parity can be embedded directly into automated workflows or Jupyter notebooks:

```python
import datetime as dt
import pandas as pd
from parity.loaders import load_deal_cash_flows, load_deal_config
from parity.deal import analyze_deal
from parity.hedging import analyze_hedging
from parity.scenarios import run_scenario_switchboard
from parity.report import save_report, compute_top_risk_drivers

# 1. Load Deal Inputs
deal_df = load_deal_cash_flows("data/templates/deal.csv")
deal_cfg = load_deal_config("data/templates/deal.yaml")

# 2. Execute Simulation (100,000 Monte Carlo Paths)
res = analyze_deal(
    deal_df=deal_df,
    deal_config=deal_cfg,
    spot_rate=1545.0,
    parallel_premium=0.18,
    n_paths=100000,
    seed=42,
)

print(f"Headline LCU IRR: {res.headline_lcu_irr * 100:.1f}%")
print(f"Expected USD IRR: {res.usd_irr_mean * 100:.1f}%")
print(f"P(USD IRR < {deal_cfg.hurdle_usd_irr * 100:.0f}%): {res.p_below_hurdle * 100:.1f}%")
print(f"Required Breakeven LCU IRR (75% conf): {res.breakeven_lcu_irr_75 * 100:.1f}%")

# 3. Analyze Hedging with Automatic CIP Fallback
hedge_res = analyze_hedging(
    deal_df=deal_df,
    deal_config=deal_cfg,
    unhedged_result=res,
    spot_rate=1545.0,
    local_rate_pct=26.75,
    us_rate_pct=4.75,
    parallel_premium=0.18,
)

# 4. Export Institutional Excel Model with Charts
save_report(
    deal_res=res,
    deal_config=deal_cfg,
    dev_prob_12m=0.318,
    dev_prob_24m=0.492,
    risk_drivers=compute_top_risk_drivers({"parallel_premium": 0.18, "reserves_months_imports": 3.1}),
    deal_df=deal_df,
    hedge_res=hedge_res,
    output_path="reports/investment_committee_memo.xlsx",
)
```

---

## Public Data Sourcing Guide

Parity relies on verified macroeconomic fundamentals. The table below lists authoritative public databases:

| Series | Primary Source | Access URL / Portal |
| :--- | :--- | :--- |
| **Official Exchange Rates & Reserves** | IMF International Financial Statistics (IFS) | [data.imf.org](https://data.imf.org) |
| **Exchange Rate Arrangements & Restrictions** | IMF AREAER Annual Report | [elibrary.imf.org](https://www.elibrary.imf.org) |
| **Current Account & Balance of Payments** | IMF World Economic Outlook (WEO) | [imf.org/en/Publications/WEO](https://www.imf.org/en/Publications/WEO) |
| **IMF Lending Arrangements & Status** | IMF Monitoring of Fund Arrangements (MONA) | [imf.org/external/np/pdr/mona](https://www.imf.org/external/np/pdr/mona) |
| **World Development Indicators** | World Bank Open Data | [databank.worldbank.org](https://databank.worldbank.org) |
| **US Policy Rates & Treasury Yields** | Federal Reserve Bank of St. Louis (FRED) | [fred.stlouisfed.org](https://fred.stlouisfed.org) |
| **Central Bank Reports & Policy Rates** | Central Bank of Kenya (CBK), Central Bank of Nigeria (CBN), Bank of Ghana (BOG), Central Bank of Egypt (CBE) | Official National Central Bank Portals |

---

## Model Limitations & Governance Disclaimers

Parity is an analytical decision-support and stress-testing tool. When presenting results to investment committees, analysts must communicate the following five limitations:

1. **Sparse Historical Devaluation Episodes**: Discrete devaluations occur infrequently in any single country's historical time series. Documented priors in `config/priors.yaml` dominate statistical estimates unless long, multi-cycle panels are supplied.
2. **Realized Repatriation Rates vs. Official Rates**: Official pegged or managed rates frequently diverge from the effective exchange rate realized upon repatriation due to backlog queues, foreign exchange allocation rationing, and parallel market premiums. Sensitivity should always be evaluated under `blend` or `parallel` repatriation modes.
3. **Policy Discretion & Timing**: Devaluations are political and central bank policy decisions. The model computes probabilistic hazards and conditional expected magnitudes, not deterministic calendar dates.
4. **Thin Forward & NDF Liquidity**: Market forwards in African and frontier currencies are frequently illiquid or indicative only. Covered Interest Parity (CIP) proxy rates illustrate the theoretical carrying cost of hedging, not executable market depth.
5. **No Investment Advice / Apache 2.0**: Parity outputs are quantitative stress-testing estimates provided on an "as-is" basis under the Apache-2.0 License, without warranty of any kind. Parity does not provide investment advice, credit ratings, or financial recommendations.

---

## Release Notes (v0.1.0)

- **Initial Stable Release (v0.1.0)**:
  - Prior-backed regularized logistic regression devaluation hazard model (`devaluation.py`).
  - Lognormal and Gamma jump magnitude models anchored to parallel premiums (`magnitude.py`).
  - Two-regime Monte Carlo path simulator with PPP drift, jump hazards, and post-jump retracements (`fxpaths.py`).
  - Comprehensive deal cash flow engine supporting official, parallel, and blend repatriation modes, trapped cash haircuts, and withholding taxes (`deal.py`).
  - Exact zero-depreciation identity verification: $IRR_{\text{USD}} = IRR_{\text{LCU}}$ on flat paths.
  - Hedging analysis with automated Covered Interest Parity (CIP) proxy fallback (`hedging.py`).
  - YAML-driven multi-scenario switchboard (`scenarios.py`).
  - Markdown Investment Committee memos and institutional 5-tab Excel models with embedded openpyxl charts, KPI summary cards, and conditional formatting (`report.py`).
  - Full CLI tooling with `fit`, `devalue`, `deal`, `compare`, and `report` (`cli.py`).
  - 100% test coverage with 99 passing unit and integration tests.

---

## Contributing & License

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for code style guidelines and test requirements.

Parity is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.