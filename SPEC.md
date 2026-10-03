# SPEC

## Repo layout
parity/
  CLAUDE.md
  CHARTER.md
  LICENSE
  NOTICE
  CONTRIBUTING.md
  README.md
  .gitignore
  config/priors.yaml
  config/scenarios.yaml
  data/schemas.md
  data/templates/*.csv        (headers only)
  src/parity/
    schemas.py
    loaders.py
    features.py
    devaluation.py
    magnitude.py
    fxpaths.py
    deal.py
    hedging.py
    scenarios.py
    report.py
    cli.py
  tests/

## Data inputs (user supplies CSVs)
1. fx_panel.csv: country, currency, date (monthly), official_rate (LCU per USD),
   parallel_rate (nullable), reserves_months_imports, reserves_change_3m_pct,
   current_account_pct_gdp, inflation_local_pct, inflation_us_pct, policy_rate_pct,
   us_rate_pct, imf_program (0/1), imf_review_on_track (0/1/NA)
2. fx_events.csv: country, date, event_type (devaluation / float / peg_break /
   managed_crawl_change), size_pct (nullable)
3. forwards.csv (optional): country, date, tenor_months, forward_rate (NDF or quoted forward)
4. deal.csv: period (month index from close), equity_invested_lcu, distribution_lcu,
   exit_proceeds_lcu
5. deal.yaml: country, close_date, hurdle_usd_irr, repatriation_rate
   (official / parallel / blend), repatriation_blend_weight, blocked_funds_haircut_pct,
   withholding_tax_pct
Suggested public sources: IMF IFS and AREAER, World Bank, central bank publications.
Verify all figures against primary sources before use.

## Models

### A. Devaluation probability (devaluation.py)
- Event definition in config: depreciation > X% in one month OR > Y% over 3 months
  (defaults flagged as priors).
- Logistic regression on lagged features, horizons 12 and 24 months.
- Features: parallel-market premium (parallel/official - 1), reserves cover, reserve change,
  current account balance, real exchange rate misalignment (cumulative inflation differential
  vs nominal move since a base date), policy rate minus US rate, IMF program flag,
  IMF off-track flag.
- Expected signs documented in priors.yaml: premium +, reserves -, CA deficit +,
  misalignment +, IMF program - (off-track +).
- Methods: predict_proba(), fit() with regularisation and time-series CV (no random splits),
  calibration curve, Brier score, AUC.

### B. Devaluation size (magnitude.py)
- Size given an event: lognormal or Gamma. Anchor: a fraction of the parallel premium closes
  on the official rate. Other drivers: regime (peg / managed / float), reserves cover.
- Output mean, median, 10th and 90th percentile.

### C. FX path simulator (fxpaths.py)
- Monthly paths, N=10,000, fixed seed option.
- Calm regime: drift = inflation differential (PPP-style) times a pass-through parameter,
  plus small volatility.
- Jump regime: hazard from model A, jump size from model B, then a post-jump adjustment
  period (partial retracement or continued crawl, parameterised).
- Output: paths, terminal-rate distribution, annualised depreciation distribution.

### D. Deal engine (deal.py)
- Convert local-currency flows to USD along each path.
- Apply repatriation_rate, blocked_funds_haircut_pct and withholding_tax_pct from deal.yaml.
  Parallel-market repatriation and trapped cash are central in NGN, ETB and EGP.
- Output per deal: LCU IRR and MOIC (headline), USD IRR and MOIC distribution
  (mean, median, P10, P90), P(USD IRR < hurdle), expected FX drag (LCU IRR minus USD IRR),
  breakeven local-currency IRR needed to hit hurdle_usd_irr at 50% and 75% confidence.

### E. Hedging comparison (hedging.py)
- Hedged vs unhedged USD IRR using forwards.csv when supplied, else a covered-interest-parity
  proxy (local rate minus US rate), flagged as a proxy.
- Show hedge cost as annual drag, and the breakeven devaluation at which hedging pays.

### F. Scenario switchboard (scenarios.py)
- YAML scenarios: base, imf_program_lost, premium_widens (+15 pts),
  step_devaluation (-30% at month 12), slow_crawl (-8% a year),
  float_and_overshoot (-40% then 15% retrace).
- Each scenario overrides features or forces a path, reruns the pipeline, and outputs
  a comparison table.

## CLI (typer)
parity fit --panel data/fx_panel.csv --events data/fx_events.csv
parity devalue --country Nigeria --date 2026-09-30 --scenario base
parity deal --deal data/deal.csv --config data/deal.yaml --scenario base
parity compare --deal data/deal.csv --config data/deal.yaml --scenarios base,step_devaluation
parity report --deal data/deal.csv --config data/deal.yaml --out reports/deal.md

## Output report (report.py)
Markdown with: devaluation probability (12m and 24m), headline LCU IRR vs USD IRR
distribution, FX drag, P(USD IRR < hurdle), required local IRR, hedge comparison,
top 3 drivers of devaluation risk, data freshness warning, and model limitations.

## Tests
- Synthetic panel generator to verify fit() recovers known coefficients.
- Monotonicity: higher parallel premium or lower reserves raises devaluation probability.
- Deal engine: a zero-depreciation path reproduces the LCU IRR exactly.
- Edge cases: missing parallel_rate, NA imf_review, negative-IRR deals, no-forward fallback.
- Simulation reproducibility with a fixed seed.

## Build order
1. Repo scaffold, schemas, loaders, CSV templates, LICENSE, NOTICE, CONTRIBUTING, .gitignore, tests
2. features.py
3. devaluation.py + tests
4. magnitude.py + fxpaths.py + tests
5. deal.py + tests
6. hedging.py + scenarios.py
7. cli.py + report.py
8. README with data-sourcing guide and limitations, v0.1 release notes

## Limitations to state in README
- Few devaluation episodes per country. Priors dominate unless data is rich.
- Official rates may not be the rate investors actually get. Parallel and repatriation
  rates matter, and data on them is patchy.
- Devaluations are policy decisions and can be abrupt. The model gives odds, not timing.
- Forward and NDF liquidity is thin in many of these currencies.
- Not investment advice. Outputs are model estimates, with no warranty.