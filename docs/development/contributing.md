# Contributing Guide

Welcome to the Golf Modeling Suite development team! We follow strict engineering standards to ensure reliability and reproducibility.

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- Git
- Docker (optional, but recommended for full physics validation)
- MATLAB (optional, for Simscape models)

### Installation

```bash
# Clone the repository
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite

# Install with development dependencies
pip install -e ".[dev,engines,analysis]"

# Install pre-commit hooks
pre-commit install
```

## 🧪 Testing Standards

We use `pytest` for testing. All PRs must pass the test suite.

```bash
# Run unit tests
pytest tests/unit/

# Run physics validation (requires engines)
pytest tests/physics_validation/

# Run coverage report
pytest --cov=shared --cov-report=html
```

## 🎨 Code Style

We enforce strict formatting and typing:
- **Black**: Code formatter (line length 88)
- **Ruff**: Linter (E, F, B, I rules)
- **MyPy**: Static type checking (strict mode)

Run the full quality gate before committing:
```bash
black . && ruff check . && mypy .
```

## 📝 Pull Request Process

1. Create a branch from `main` (e.g., `feat/my-feature`).
2. Implement your changes.
3. Ensure all tests pass.
4. Update documentation if API changes.
5. Create PR with descriptive title and body.
