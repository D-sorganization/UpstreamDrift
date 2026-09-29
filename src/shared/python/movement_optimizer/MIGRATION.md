# Movement Optimizer — vendored into Tools

This directory is the **canonical home** of the Movement Optimizer biomechanics
application. It was migrated from the standalone repository
`D-sorganization/Movement_Optimizer`, which is **archived** in favour of this
copy so that all future development happens here, in the shared Tools monorepo
consumed by UpstreamDrift.

See the migration epic: **D-sorganization/Tools#3407**.

## What this is

A biomechanics barbell-trajectory optimizer using Lagrangian inverse dynamics
(3-link sagittal chain: shank, thigh, trunk). Supports squat, full squat,
deadlift, bench press, snatch, clean, and jerk; spinal-load (L5/S1) and
Hill-muscle analysis; a PyQt6 GUI and a headless CLI; an optional Rust/PyO3
acceleration backend with a pure-NumPy fallback.

## How it is integrated

- **Self-contained sub-app.** Like `src/pendulum_simulator`, this tree carries
  its own quality bar and is excluded from the monorepo's ruff / mypy / coverage
  delta gates (see `ruff.toml`, `mypy.ini`, `.coveragerc`, `pyproject.toml`, and
  the CI filter lists in `.github/workflows/ci-standard.yml`). Its own test suite
  lives under `tests/` here and is preserved verbatim from the origin repo.
- **Launcher / UpstreamDrift discovery.** Published via `model_pack.yaml`
  (`pack_id: tools-movement-optimizer`, route `/tools/movement-optimizer`), validated by
  `scripts/movement_optimizer_provider_manifest.py` and
  `tests/test_movement_optimizer_provider_manifest.py`.
- **Imports.** The package is importable as `movement_optimizer` because Tools
  puts `src/` on `sys.path` (see `_bootstrap.py`).

## Known follow-ups (tracked under Tools#3407)

- **#3410** — unify the UpstreamDrift surface: this app and the older generic
  `src/optimizer_gui` tool both currently advertise a Movement-Optimizer route;
  collapse to one canonical route and retire the duplicate.
- **#3411** — carry over the code-quality findings from the origin-repo audit
  (elbow-bias bug, bench-press inertia convention, parity-only physics tests,
  GUI-thread blocking, scipy<1.16 pin, etc.) and fix them here.
- The divergent swingset/chain model API in `src/optimizer_gui` should be
  reconciled against this package's richer `models/swingset.py` /
  `models/chain_forces.py` and then de-duplicated.
