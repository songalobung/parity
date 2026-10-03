# Parity: Hard-Currency Return Model for Frontier Deals

Project name: Parity. Package name: parity.
License: Apache 2.0. Add LICENSE, NOTICE, and CONTRIBUTING.md.

Python library + CLI that takes a local-currency deal and stress-tests its USD returns
against simulated FX paths. Core outputs:
1. Devaluation probability over 12 and 24 months
2. USD IRR and MOIC distribution versus the headline local-currency IRR
3. Required local-currency IRR to hit a USD target

## Rules
- Read CHARTER.md for scope and success criteria. Do not build anything listed under Out of scope.
- Read SPEC.md before writing code. Follow it.
- Python 3.11+, numpy, pandas, scipy, statsmodels, scikit-learn, pydantic, typer, pytest, matplotlib.
- Type hints everywhere. Small pure functions. No notebooks as source of truth.
- Every model coefficient lives in config/priors.yaml with a comment saying it is a prior, not a fitted value.
- Never hardcode data. All data comes from CSVs in data/ matching the schemas in SPEC.md.
- Never invent historical data. If a file is missing, fail with a clear error naming the file and columns.
- Add a .gitignore that excludes data/ (except templates and schemas.md). Never commit licensed data.
- Write tests alongside each module. Run pytest before declaring any step done.
- Build in the order under "Build order" in SPEC.md. Stop after each step and show test results.