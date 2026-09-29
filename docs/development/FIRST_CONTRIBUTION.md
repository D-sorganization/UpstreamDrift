# Your First Contribution

Welcome to UpstreamDrift! This guide will help you go from zero to your first PR.

## Prerequisites

- Python 3.11+ (Python 3.13 recommended)
- Git with Git LFS
- A GitHub account with fork permissions

## Setup (10 minutes)

1. Clone the repository:

   ```bash
   git clone https://github.com/D-sorganization/UpstreamDrift.git
   cd UpstreamDrift
   git lfs install && git lfs pull
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:

   ```bash
   pre-commit install
   pre-commit install --hook-type pre-push
   ```

5. Run the test suite to verify your setup:

   ```bash
   pytest tests/unit -x -q --tb=short
   ```

   If all tests pass, you are ready to contribute.

## Architecture Overview

The codebase is organized into layers:

- **src/api/**: FastAPI server with REST and WebSocket endpoints
- **src/engines/**: Physics engine adapters (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite)
- **src/shared/**: Common utilities, contracts, logging, physics models
- **src/tools/**: Development and analysis tools
- **src/launchers/**: Application entry points (unified launcher)
- **tests/**: Unit, integration, and API tests

For more detail, see [Architecture](architecture.md) and the
[ADRs](../adr/README.md) for key design decisions.

## Making Your First Change

1. Create a feature branch from `main`:

   ```bash
   git checkout main && git pull
   git checkout -b feature/your-change-description
   ```

2. Make your changes following our code style (enforced by pre-commit hooks).

3. Run tests locally:

   ```bash
   pytest tests/unit -x -q
   ```

4. Commit and push:

   ```bash
   git add your-files
   git commit -m "feat: description of your change"
   git push -u origin feature/your-change-description
   ```

5. Create a PR on GitHub. The CI pipeline will run automatically.

## Commit Message Format

We use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code refactoring (no behavior change)
- `test:` Adding or updating tests
- `ci:` CI/CD configuration changes
- `chore:` Maintenance tasks (dependency updates, etc.)

## Common Development Tasks

The Makefile provides shortcuts for frequent operations:

```bash
make help      # Show all available targets
make install   # Install dependencies
make check     # Run linters and tests
make format    # Format code with ruff
```

## Good First Issues

Look for issues labeled
[`good first issue`](https://github.com/D-sorganization/UpstreamDrift/labels/good%20first%20issue)
on GitHub. These are specifically chosen to be approachable for new contributors.

## Need Help?

- Check the [Troubleshooting Guide](../troubleshooting/README.md) for common issues
- Read through existing [docs/](../README.md) for more guides
- Open a GitHub issue if you are stuck
