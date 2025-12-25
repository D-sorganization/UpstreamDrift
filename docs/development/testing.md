# Testing

## Validation Suite
The main entry point for verification is `validate_suite.py`.

```bash
python validate_suite.py
```

This script checks:
- Directory structure.
- Importability of modules.
- Basic functionality of shared components.

## Unit Tests
Tests are located in `tests/`.

```bash
pytest tests/
```

## Pre-commit Checks
We use `pre-commit` hooks to ensure code quality.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
