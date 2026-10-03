# Parity: Developer Guide

Parity is an open-source Python library and CLI to stress-test local-currency frontier deals against simulated FX paths.

## Key Rules & Conventions

- **Country Agnostic**: All core logic remains country-agnostic (supporting KES, NGN, ETB, GHS, EGP, ZMW, AOA, UGX, TZS, etc.).
- **Prior Transparency**: All model coefficients reside in `config/priors.yaml` with explicit comments stating they are priors, not fitted values. Never hardcode model coefficients into Python modules.
- **Data Integrity**: Never invent historical data. Users supply CSVs matching `data/schemas.md`. Missing files fail with clear errors naming the missing file and expected columns.
- **Privacy & Licensing**: Never commit proprietary or licensed vendor data. `.gitignore` excludes `data/*` except `data/templates/` and `data/schemas.md`.
- **Code Style**:
  - Python 3.11+
  - Complete type hints
  - Pure, functional components where possible
  - Strict validation using Pydantic schemas

## Common Commands

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run single test module
pytest tests/test_loaders.py
```

## Build Milestones

1. Scaffold, schemas, loaders, CSV templates, LICENSE, NOTICE, CONTRIBUTING, .gitignore, tests
2. features.py
3. devaluation.py + tests
4. magnitude.py + fxpaths.py + tests
5. deal.py + tests
6. hedging.py + scenarios.py
7. cli.py + report.py
8. README with data-sourcing guide, limitations, v0.1 release
