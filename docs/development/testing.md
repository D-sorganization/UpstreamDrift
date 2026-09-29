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

## Integration Test Tiers

UpstreamDrift uses separate expectations for core CI and dedicated
native-engine validation:

- Core PR lane: fast, dependency-light, and allowed to skip optional engines
- Dedicated native-engine lanes: expected to provision engines intentionally and
  fail if those environments are broken

Useful commands:

```bash
# Core cross-engine validator logic only
pytest tests/integration/test_cross_engine_validation.py::TestCrossEngineValidator -v

# Real cross-engine path with strict fixture behavior
UPSTREAM_DRIFT_STRICT_ENGINE_PROBES=true \
pytest tests/integration/test_cross_engine_validation.py -v

# Native import check used by nightly workflow
python scripts/check_native_engine_imports.py \
  --json-output /tmp/native-engine-imports.json \
  --markdown-output /tmp/native-engine-imports.md
```

For the longer-form strategy, see `docs/testing/integration-testing-strategy.md`.

## Pre-commit Checks

We use `pre-commit` hooks to ensure code quality.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
