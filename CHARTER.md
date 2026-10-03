# Parity: Project Charter

**Project:** Parity, a hard-currency return model for frontier deals
**Owner:** [Your name]
**License:** Apache 2.0
**Status:** Pre-build

## Problem
Most PE and IB models for African deals run in local currency and convert to USD at the end.
Devaluation, parallel-market rates and trapped cash quietly cut USD returns, and the headline
IRR hides it. Existing tools are built for developed markets, are expensive, or don't handle
frontier FX regimes.

## Objective
Build an open-source Python library and CLI that takes a local-currency deal and shows its
USD IRR and MOIC distribution under simulated FX paths, with devaluation probabilities driven
by reserves, parallel-market premium, current account and IMF status.

## Target users
- Foreign PE and VC funds evaluating African deals
- Investment banks building frontier-market valuations
- Asset managers and DFIs with local-currency exposure
- Analysts and students who can't afford terminal-grade tools

## Scope (in)
- Devaluation probability model (12 and 24 months)
- Devaluation size model
- Monte Carlo FX path simulator with calm and jump regimes
- Deal engine: local-currency flows to USD, with repatriation rate, blocked funds and withholding tax
- Hedged vs unhedged comparison
- YAML scenario switchboard
- CLI and Markdown report
- Currencies: KES, NGN, ETB, GHS, EGP, ZMW, AOA, UGX, TZS (code stays country-agnostic)

## Out of scope (v1)
- Live data feeds or scraping
- Web UI or dashboard
- Full LBO or DCF modeling (Parity consumes deal cash flows, it doesn't build them)
- Sovereign default modeling
- Trading signals or investment recommendations
- Bundled datasets (users supply their own CSVs)

## Deliverables
1. Python package `parity` with tests
2. CLI: fit, devalue, deal, compare, report
3. Documented priors in config/priors.yaml
4. CSV templates and data schemas
5. README with data-sourcing guide and limitations
6. CONTRIBUTING.md, LICENSE, NOTICE

## Success criteria
- Zero-depreciation path reproduces the local-currency IRR exactly
- fit() recovers known coefficients on synthetic data
- All tests pass, simulations reproducible with a fixed seed
- A user can go from a deal CSV to a report in under 10 minutes
- Report states P(USD IRR < hurdle), FX drag and required local IRR
- At least one real deal case study run end to end, with inputs from primary data

## Milestones
Follows the build order in SPEC.md. Each step ends with passing tests before the next begins.
1. Scaffold, schemas, loaders, templates
2. Feature engineering
3. Devaluation probability model
4. Size model and FX path simulator
5. Deal engine
6. Hedging and scenarios
7. CLI and report
8. README, docs, first public release (v0.1)

## Key risks
| Risk | Impact | Mitigation |
|---|---|---|
| Few devaluation episodes per country | Priors dominate output | Label priors clearly, pool countries when fitting, say so in the report |
| Parallel-market data is patchy | Weakest input to the strongest signal | Make parallel_rate optional, fall back with a warning |
| Official rate differs from the rate investors get | USD returns overstated | Repatriation blend and blocked-funds inputs in deal.yaml |
| Thin forward and NDF markets | Hedge comparison unreliable | CIP proxy fallback, flagged as proxy |
| Users treat output as forecasts | Reputational and legal exposure | Disclaimers in README and every report |
| Licensed data committed to repo | Legal breach | .gitignore on data/, never commit vendor data |

## Assumptions
- Users can source monthly FX, reserves and macro data from IMF, World Bank and central banks
- Devaluations are discrete events plus drift, which the two-regime path model can approximate
- Monthly frequency is enough for deal-level analysis

## Governance
- Owner sets scope and approves merges to main
- Contributions via pull request, tests required
- Good first issues: country data loaders, new scenarios, calibration notebooks
- Semantic versioning, changelog kept in the repo

## Disclaimer
Parity produces model estimates, not forecasts or investment advice. Outputs depend on
user-supplied data and documented priors. No warranty.