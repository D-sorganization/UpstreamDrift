# Clean-Environment Install Smoke Specification

## Context and Objective

As part of the deployment qualification epic ([#14276](https://github.com/D-sorganization/UpstreamDrift/issues/14276)),
issue [#14275](https://github.com/D-sorganization/UpstreamDrift/issues/14275) establishes
an automated clean-environment install smoke test. Following the deployment boundary
decision recorded in [ADR-0021](../adr/0021-container-strategy.md) and
[14272-deployment-boundary.md](14272-deployment-boundary.md), clean execution targets
(such as base container runtimes, isolated virtualenvs, and user installations) must
be qualified against the core runtime invariants without optional heavy engine dependencies.

## Qualification Invariants

A clean-environment installation is qualified when all of the following conditions hold:

1. **Python Version Boundary**: The runtime Python interpreter satisfies the declared
   `requires-python` constraint (`>= 3.11, < 3.13`).
2. **Core Dependencies**: All mandatory core packages declared in `pyproject.toml`
   (`numpy`, `scipy`, `fastapi`, `uvicorn`, `pydantic`, `mujoco`, `sqlalchemy`,
   `bcrypt`, `jwt`, `defusedxml`, `pandas`, `platformdirs`, `structlog`) are
   importable and available.
3. **Core Public Surface**: The core public interfaces (`src.api.local_server`,
   `src.engines.physics_engines.mujoco`, `src.shared.python.physics`,
   `src.shared.python.spatial_algebra`, and `src.shared.python.engine_core`) load
   cleanly without missing symbol references.
4. **Engine Isolation**: Optional heavy engines (`drake`, `pinocchio`, `opensim`,
   `myosuite`, `jaxsim`) are neither loaded into memory nor required by the core
   surface.
5. **CLI Entrypoint Operability**: The primary entrypoints (`upstream-drift` via
   `launch_upstream_drift.py --help` and `ud-sim` via `src.sim_run.cli --help`)
   respond successfully with exit code 0.

## Execution and Automation

The smoke test is executable directly via the command line or CI pipeline:

```bash
# Human-readable summary output
python scripts/ci/check_clean_install_smoke.py

# Machine-readable JSON output for automated gating
python scripts/ci/check_clean_install_smoke.py --json

# Explicit entrypoint subprocess validation
python scripts/ci/check_clean_install_smoke.py --check-cli
```

Automated regression coverage is maintained in `tests/ci/test_clean_environment_install_smoke.py`
under the `core_only` and `unit` test markers.
