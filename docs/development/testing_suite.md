# UpstreamDrift Testing Suite Documentation

## Overview

This document describes the testing infrastructure, conventions, and coverage
goals for the UpstreamDrift repository. Our quality standard is that tests
should be held to the same rigor as production code.

## Test Organization

```
tests/
├── acceptance/          # End-to-end acceptance tests
├── analytical/          # Analytical/mathematical validation tests
├── api/                 # (in unit/) API route and middleware tests
├── architecture/        # Structural dependency direction tests
├── benchmarks/          # Performance benchmarks
├── cross_engine/        # Cross-engine validation tests
├── fixtures/            # Shared test data and fixture files
├── headless/            # Headless mode smoke tests
├── integration/         # Integration and E2E tests
├── learning/            # ML/RL learning pipeline tests
├── parity/              # Parity tests across implementations
├── physics_validation/  # Physics model validation tests
├── security/            # Security-focused tests
├── unit/                # Unit tests (primary coverage focus)
│   ├── ai/              # AI assistant and workflow tests
│   ├── api/             # API endpoint tests
│   ├── core/            # Core utilities (error, datetime, type)
│   ├── data_io/         # Data I/O path utilities tests
│   ├── dbc/             # Design-by-Contract runtime tests
│   ├── engines/         # Physics engine tests
│   ├── injury/          # Injury risk model tests
│   ├── robotics/        # Robotics module tests
│   └── signal_toolkit/  # Signal processing tests
└── tools/               # Tool-specific tests
```

## Running Tests

### Full test suite

```bash
python -m pytest tests/ -v
```

### Unit tests only

```bash
python -m pytest tests/unit/ -v
```

### With coverage report

```bash
python -m pytest tests/unit/ --cov=src/shared/python --cov-report=term-missing
```

### Specific package

```bash
python -m pytest tests/unit/core/ -v
python -m pytest tests/unit/signal_toolkit/ -v
```

## Test Conventions

### Naming

- Test files: `test_<module_name>.py`
- Extended tests: `test_<module_name>_extended.py` (supplements existing tests)
- Test classes: `Test<ClassName>` or `Test<FunctionName>`
- Test methods: `test_<behavior_being_tested>`

### Structure

Each test class tests one public class or function group. Tests follow
the Arrange-Act-Assert pattern:

```python
class TestComputePSD:
    """Tests for Power Spectral Density computation."""

    def test_peak_frequency(self) -> None:
        """PSD should peak near the signal's frequency."""
        # Arrange
        fs = 1000.0
        data = np.sin(2 * np.pi * 50.0 * np.arange(0, 1.0, 1.0 / fs))

        # Act
        freqs, psd = compute_psd(data, fs=fs, nperseg=256)
        peak_freq = freqs[np.argmax(psd)]

        # Assert
        assert abs(peak_freq - 50.0) < 5.0
```

### Type Hints

All test methods include return type annotations (`-> None`).

### Design by Contract Tests

DbC contract tests live in `tests/unit/dbc/` and verify that precondition,
postcondition, and invariant checks fire correctly with `PreconditionError`
and `PostconditionError`.

## Coverage Goals

### Priority Levels

| Priority | Package            | Target | Status         |
| -------- | ------------------ | ------ | -------------- |
| P0       | `core/`            | >90%   | Wave 1 ✓       |
| P0       | `signal_toolkit/`  | >60%   | Wave 1 ✓       |
| P1       | `analysis/`        | >80%   | Planned Wave 2 |
| P1       | `biomechanics/`    | >80%   | Planned Wave 2 |
| P1       | `spatial_algebra/` | >80%   | Planned Wave 2 |
| P2       | `physics/`         | >50%   | Planned Wave 3 |
| P2       | `data_io/`         | >50%   | Planned Wave 3 |
| P3       | `optimization/`    | >40%   | Planned Wave 4 |
| P3       | `validation_pkg/`  | >40%   | Planned Wave 4 |

### Exclusions

GUI/UI modules (`ui/`, `dashboard/`, `gui_pkg/`, `theme/`) are lower priority
for unit testing as they require Qt/display infrastructure. They are covered
by integration and headless tests instead.

## CI/CD Integration

Tests run automatically via GitHub Actions on every push and PR:

- Python 3.11 matrix
- Parallel execution via `pytest-xdist`
- Coverage reports uploaded to Codecov
- Quality gate enforced: CI must pass before merge

## Wave History

### Wave 1 (Feb 2026) — Core & Signal Processing

- `core/error_decorators.py`: 0% → ~90% (30+ new tests)
- `core/error_utils.py`: 60% → ~95% (50+ new tests)
- `core/datetime_utils.py`: 53% → ~95% (40+ new tests)
- `signal_toolkit/signal_processing.py`: 24% → ~55% (32+ new tests)
- **Total new tests**: 169
