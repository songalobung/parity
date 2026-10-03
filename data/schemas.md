# Parity Data Schemas

This document defines the schema, types, constraints, and requirements for all input datasets consumed by Parity.

Parity does not bundle proprietary data. Users supply their own data files adhering strictly to these formats.

---

## 1. FX Panel (`fx_panel.csv`)

Monthly panel of exchange rates, macro fundamentals, and policy indicators by country.

| Column | Type | Nullable | Description / Units |
|---|---|---|---|
| `country` | string | No | Country name or standard ISO country identifier (e.g., `Kenya`, `Nigeria`). |
| `currency` | string | No | Three-letter ISO currency code (e.g., `KES`, `NGN`, `ETB`). |
| `date` | ISO Date (YYYY-MM-DD or YYYY-MM) | No | Monthly observation date (typically month-end). |
| `official_rate` | float | No | Official spot exchange rate in Local Currency Units (LCU) per 1 USD (> 0). |
| `parallel_rate` | float | Yes | Parallel or unofficial market exchange rate in LCU per USD. Blank/NA if no active parallel market. |
| `reserves_months_imports` | float | No | Gross foreign exchange reserves expressed in months of import cover (>= 0). |
| `reserves_change_3m_pct` | float | No | 3-month percentage change in gross reserves (e.g., `-0.05` for -5% or `-5.0` based on percentage scale; standard decimal representation preferred, e.g. -0.05). |
| `current_account_pct_gdp` | float | No | Current account balance as a percentage of GDP (e.g., `-4.2` for -4.2% of GDP). |
| `inflation_local_pct` | float | No | Year-over-year domestic consumer price inflation rate (%). |
| `inflation_us_pct` | float | No | Year-over-year US CPI inflation rate (%). |
| `policy_rate_pct` | float | No | Central bank domestic policy rate (annualized %). |
| `us_rate_pct` | float | No | US policy rate / Federal Funds rate / Treasury proxy (annualized %). |
| `imf_program` | integer (`0` or `1`) | No | Indicator whether an active IMF lending program is in place (`1` = yes, `0` = no). |
| `imf_review_on_track` | integer/string (`0`, `1`, `NA`) | Yes | Status of IMF program review (`1` = on track, `0` = delayed/off track, `NA` or blank if no program or unrated). |

---

## 2. FX Events (`fx_events.csv`)

Historical discrete exchange rate regime shocks or formal devaluation events.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `country` | string | No | Country identifier matching `fx_panel.csv`. |
| `date` | ISO Date | No | Date of the devaluation or regime event. |
| `event_type` | string | No | One of: `devaluation`, `float`, `peg_break`, `managed_crawl_change`. |
| `size_pct` | float | Yes | Magnitude of discrete spot devaluation as a percentage (e.g., `0.30` for +30% LCU/USD depreciation). Nullable if unquantified. |

---

## 3. Forwards (`forwards.csv`) - *Optional*

Quoted forward or Non-Deliverable Forward (NDF) points/rates. If omitted, Parity falls back to a Covered Interest Parity (CIP) proxy using domestic and US policy rates.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `country` | string | No | Country identifier matching `fx_panel.csv`. |
| `date` | ISO Date | No | Observation date of the forward quote. |
| `tenor_months` | integer | No | Tenor in months (e.g., `3`, `6`, `12`, `24`). Must be positive. |
| `forward_rate` | float | No | Quoted outright forward rate in LCU per USD (> 0). |

---

## 4. Deal Cash Flows (`deal.csv`)

Monthly deal-level cash flows expressed in Local Currency Units (LCU).

| Column | Type | Nullable | Description |
|---|---|---|---|
| `period` | integer | No | Month index from deal close (`0` = close / initial investment, `1`, `2`, ...). Must be strictly non-negative and sequential. |
| `equity_invested_lcu` | float | No | Equity capital deployed by the sponsor in LCU during this period (>= 0). Typically positive at `period = 0`. |
| `distribution_lcu` | float | No | Dividends, interest, or interim distributions received in LCU (>= 0). |
| `exit_proceeds_lcu` | float | No | Net exit proceeds received at terminal exit in LCU (>= 0). |

---

## 5. Deal Configuration (`deal.yaml`)

Scenario and execution parameters for evaluating a specific investment deal.

```yaml
country: "Nigeria"
close_date: "2026-01-31"
hurdle_usd_irr: 0.15 # Minimum required USD IRR (e.g. 15%)
repatriation_rate: "blend" # One of: 'official', 'parallel', 'blend'
repatriation_blend_weight: 0.5 # Weight on official rate when repatriation_rate is 'blend' (0.0 to 1.0)
blocked_funds_haircut_pct: 0.05 # Haircut applied to distributions due to trapped/delayed liquidity (e.g. 5%)
withholding_tax_pct: 0.10 # Cross-border withholding tax rate applied to repatriated distributions (e.g. 10%)
```
