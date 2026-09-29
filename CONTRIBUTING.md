# Contributing to UpstreamDrift

Thank you for your interest in contributing to UpstreamDrift.

`CLAUDE.md` is the authoritative source for repository rules and quality gates.
This guide focuses on contribution flow and the minimum local steps to prepare a PR.

## Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make changes** following our coding standards
5. **Test** your changes: `pytest`
6. **Commit** with a descriptive message
7. **Push** and create a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install the supported default surface
pip install -e ".[dev]"

# Optional: add Drake + Pinocchio for cross-engine work
pip install -e ".[dev,all-engines]"
```

See `docs/engines/support_tiers.md` before enabling heavier or experimental
engine combinations.

### Where dependencies live

- Python runtime: `pyproject.toml [project] dependencies`
- Python optional features: `pyproject.toml [project.optional-dependencies]`
- Python pinned for CI: `requirements.lock` and `requirements-dev.lock`
  generated from `pyproject.toml`
- Conda convenience: `environment.yml` generated from `pyproject.toml`
- Rust extension: `Cargo.toml`, `Cargo.lock`, and `rust_core/upstream-physics/`
- UI build: `ui/package.json` only; the deprecated root Create React App
  build was removed in favor of the Vite + Tauri surface
- DB migrations: `alembic.ini`

Adding a new Python dependency starts in `pyproject.toml`. Run
`make sync-deps` to regenerate the lockfiles and `environment.yml`, then commit
the canonical source and generated artifacts together.

### Rust kernel development

The Rust workspace no longer requires a sibling `../Tools` checkout just to
build `upstream-physics`. `tools-core` is fetched automatically from a pinned
git revision of `D-sorganization/Tools`, so the clean-clone workflow is:

```bash
cargo build

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip maturin

cd rust_core/upstream-physics
python -m maturin develop --features python
```

If you need editable cross-repository Python integration code from
`D-sorganization/Tools`, run `scripts/setup_tools_workspace.sh` to attach the
optional sibling workspace and helper `PYTHONPATH` entries.

### Building the UI

The UI lives under `ui/`. It is built with Vite, packaged with Tauri 2,
and bundled into the Python wheel via the hatch custom build hook
(`build_hooks.py`).

To develop:

```bash
cd ui && npm install && npm run dev
```

To build for distribution:

```bash
cd ui && npm run build
# ui/dist is then included in `pip install -e .` builds via build_hooks.py.
```

## Code Standards

### Python

- **Formatter**: Ruff format
- **Linter**: Ruff check
- **Type Checker**: MyPy (see note below)
- Use type hints for all new functions
- Use `logging` instead of `print()`
- Follow existing patterns in the codebase

> **Note on Type Checking**: While MyPy is part of our quality toolchain, strict type checking
> is not yet fully enforced across the legacy codebase. New code should include type hints.

### Before Committing

Use the Makefile for convenience:

```bash
make format   # Format with Ruff
make lint     # Run ruff and mypy
make test     # Run pytest
make check    # Run all checks
```

Or run commands directly:

```bash
python3 -m ruff format .
python3 -m ruff check .
python3 -m mypy .
python3 -m pytest
```

> **Note on CI parity**: `CLAUDE.md` is authoritative for required commands, and
> `.github/workflows/ci-standard.yml` is the canonical enforcement surface.
> The coverage threshold is the `fail_under` value in `pyproject.toml [tool.coverage.report]`.

## Physics Engine Guidelines

Current support tiers:

- **Supported default**: MuJoCo
- **Extended cross-engine**: Drake, Pinocchio
- **Experimental / stub**: OpenSim, MyoSuite

See `docs/engines/support_tiers.md` and
`docs/engines/engine_capabilities.md` before widening an engine-facing change.

When adding engine-specific code:

- Follow the existing adapter pattern
- Implement the PhysicsEngine protocol
- Add corresponding tests
- **Use the canonical target loader.** Engine-specific target loaders are
  forbidden by `src/engines/CROSS_ENGINE_PARITY_SPEC.md` §2.1. Every engine
  imports `load_club_target` from
  `src.shared.python.motion_matching.load_club_target`; this is enforced by
  `scripts/ci/check_no_engine_loader.py` and is a CI gate (issue #4254).

## Testing

- 1,563+ tests in the test suite
- Add tests for new functionality
- Run `pytest` before submitting PR
- Use existing test fixtures where possible

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Testing changes

Example: `feat(mujoco): Add contact force visualization`

## Documentation

- Update CHANGELOG.md under [Unreleased]
- Add docstrings with parameter descriptions
- Update engine-specific docs if applicable

## Pull Request Process

1. Ensure CI passes (ruff check, ruff format, mypy, pytest)
2. Update documentation
3. Request review from maintainers
