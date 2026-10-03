# Contributing to Parity

Thank you for your interest in contributing to Parity!

Parity is an open-source Python library and CLI that takes a local-currency deal and stress-tests its USD returns against simulated FX paths in frontier markets.

## Principles & Rules

1. **Country Agnostic**: Parity models apply across frontier economies (KES, NGN, ETB, GHS, EGP, ZMW, AOA, UGX, TZS, etc.). Never write country-specific logic into the core models.
2. **Transparent Priors**: Every model coefficient lives in `config/priors.yaml` with a comment explaining that it is a prior, not a fitted value. Never bury model parameters in Python code.
3. **No Hardcoded or Licensed Data**:
   - All input data comes from user-supplied CSV files matching the schemas in `data/schemas.md`.
   - Never invent historical data. If required data is missing, fail with an explicit error naming the missing file and columns.
   - Never commit proprietary or licensed vendor data to the repository. The `.gitignore` excludes all files under `data/` except `data/templates/` and `data/schemas.md`.
4. **Code Quality**:
   - Python 3.11+.
   - 100% type hints across all public and internal interfaces.
   - Small, pure, testable functions with minimal state.
   - No Jupyter notebooks as sources of truth.
5. **Testing**:
   - Write tests alongside each module.
   - All tests must pass cleanly (`pytest`) before submitting a pull request.
   - Use synthetic data or template-derived data in unit tests.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/parity.git
   cd parity
   ```

2. Create a virtual environment using `uv` or Python 3.11+:
   ```bash
   uv venv --python 3.11 .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install editable package with dev dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Pull Request Process

1. Fork the repo and create your feature branch: `git checkout -b feature/my-feature`.
2. Ensure new features or bug fixes have corresponding tests in `tests/`.
3. Check that test coverage remains comprehensive and all tests pass.
4. Submit a pull request describing the change, motivation, and test coverage.

## License

By contributing to Parity, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).
