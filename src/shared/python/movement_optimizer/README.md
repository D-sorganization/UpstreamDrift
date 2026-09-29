# Movement Optimizer

[![CI](https://github.com/D-sorganization/Movement-Optimizer/actions/workflows/ci-standard.yml/badge.svg)](https://github.com/D-sorganization/Movement-Optimizer/actions/workflows/ci-standard.yml)
[![DCO](https://github.com/D-sorganization/Movement-Optimizer/actions/workflows/dco.yml/badge.svg)](https://github.com/D-sorganization/Movement-Optimizer/actions/workflows/dco.yml)

A biomechanics tool for optimizing barbell exercise trajectories using Lagrangian inverse dynamics. The body is modeled as a 3-link planar chain (shin, thigh, trunk) in the sagittal plane. An optional Rust-accelerated backend provides high-performance computation for the hot-path inverse dynamics.

## Used by UpstreamDrift

This repository publishes a `tool_pack/v1` manifest (`tool_pack.yaml` at the
repo root) and a `biomech.tool_pack` entry point
(`movement_optimizer.tool_pack`) so the UpstreamDrift launcher can spawn this
optimizer as part of the unified Biomechanics category. See the umbrella
[D-sorganization/UpstreamDrift#5179](https://github.com/D-sorganization/UpstreamDrift/issues/5179);
the launcher tile that consumes this manifest is tracked in
[D-sorganization/UpstreamDrift#5182](https://github.com/D-sorganization/UpstreamDrift/issues/5182).

```python
from movement_optimizer.tool_pack import manifest, list_exercises, run_headless

manifest()             # parsed tool_pack.yaml as a dict
list_exercises()       # ['squat', 'full_squat', 'deadlift', ...]
run_headless("squat", "/tmp/squat.json")
```

```bash
movement-optimizer --gui                                  # default
movement-optimizer --headless --exercise squat \
                   --output /tmp/squat.json
movement-optimizer --list-exercises
```

The `Launch-Movement-Optimizer.bat` and `launch-movement-optimizer.sh`
shortcuts forward through to the same entry point, so existing users see no
behavior change.

## Features

- **Multi-exercise support**: Squat, Full Squat, Deadlift, Bench Press, Clean, Jerk, Snatch
- **2D model**: Production-ready sagittal-plane 3-link chain with a disabled placeholder selector reserved for future 3D work
- **Real-time animation**: Stick-figure visualization with playback controls
- **Spinal load analysis**: L5/S1 compression and shear estimation with NIOSH limits
- **Trajectory optimization**: Multi-start parallel SLSQP with COM balance constraints
- **Hill muscle model**: Torque-angle-velocity capacity with sticking-point detection
- **Trial comparison**: Side-by-side overlay of multiple optimization runs
- **Export**: CSV data, PNG/PDF plots, animated GIF
- **Session persistence**: Auto-save/restore of parameters and solutions
- **CLI mode**: Headless batch optimization for scripting and automation
- **Optional Rust accelerator**: PyO3/maturin extension for hot-path dynamics

## Installation

```bash
# Clone the repository
git clone https://github.com/D-sorganization/Movement-Optimizer.git
cd Movement-Optimizer

# Install in development mode
pip install -e ".[dev]"

# (Optional) Build Rust accelerator
cd rust_core && maturin develop --release && cd ..
```

### Requirements

- Python 3.10+
- NumPy, SciPy, matplotlib, PyQt6
- (Optional) Rust toolchain + maturin for the accelerator

## Usage

### GUI Mode

```bash
# Launch the GUI
python3 -m movement_optimizer

# Or use the launch scripts
./launch-movement-optimizer.sh        # Linux/macOS
Launch-Movement-Optimizer.bat         # Windows
```

The GUI provides:

- **Left sidebar**: Body parameters (mass, height, segment multipliers), barbell mass, optimization controls
- **Right area**: Tabbed exercise views with animation and analysis plots
- **Bottom bar**: Playback controls (play/pause, step, rewind, speed)

### CLI Mode

```bash
# Run a single optimization
python3 -m movement_optimizer.cli --exercise squat --body-mass 80 --height 1.80 --bar-mass 100

# Save results to JSON
python3 -m movement_optimizer.cli --exercise deadlift --body-mass 90 --bar-mass 150 --output results.json

# Customize duration and smoothness
python3 -m movement_optimizer.cli --exercise snatch --body-mass 75 --bar-mass 60 --duration 3.0 --smoothness 1.5
```

### Runtime Configuration

The application keeps its GUI/session state in `~/.movement_optimizer` by default.
To override that location for local development, CI, or sandboxed runs, set:

```bash
export MOVEMENT_OPTIMIZER_STATE_DIR=/tmp/movement-optimizer-state
```

An example override is included in [.env.example](.env.example).

### Docker

```bash
docker build -t movement-optimizer .
docker run --rm movement-optimizer --exercise squat --body-mass 80 --bar-mass 100
```

## Architecture

```
src/movement_optimizer/
    __init__.py          # Package init, __version__
    __main__.py          # GUI entry point
    backend.py           # Abstract PhysicsBackend interface
    cli.py               # CLI entry point for batch optimization
    models.py            # BodyModel, LagrangianDynamics, exercise config factories
    trajectory/          # TrajectoryOptimizer (multi-start SLSQP)
    constants.py         # All physical constants and tuning parameters
    gui/                 # PyQt6 GUI package
        __init__.py      # Re-exports MainWindow
        main_window.py   # MainWindow top-level widget
        exercise_tab.py  # ExerciseTab with animation and plots
        comparison.py    # ComparisonDialog for trial overlay
        widgets.py       # LabelledSlider, ParameterSidebar, PlaybackControls
    rendering.py         # Matplotlib rendering helpers
    comparison.py        # ComparisonStore and metrics
    persistence.py       # JSON save/load for solutions and state
    export.py            # GIF, PNG, PDF export
    spine_loads.py       # Spinal compression/shear estimation
    exercises/           # Exercise config factories (clean, jerk, snatch)
rust_core/               # Optional PyO3/maturin Rust extension
    src/lib.rs           # Vectorized inverse dynamics and COM batch
    Cargo.toml

tests/                   # pytest test suite
```

### Key Design Decisions

- **Architecture Decision Records**: Durable architecture decisions are indexed in [docs/adr/README.md](docs/adr/README.md)
- **Inner BOS constraint**: The optimizer enforces that the whole-body COM stays within the inner 60% of the base of support for realistic balance
- **No Black**: Formatting is handled exclusively by `ruff format`
- **DBC (Design by Contract)**: Public methods check preconditions and raise `ValueError`/`TypeError` on violation
- **Logging over print**: `src/` code uses `logging.getLogger(__name__)`; a pre-commit hook blocks `print()` in `src/`
- **Thread parallelism**: scipy's SLSQP releases the GIL, enabling real parallelism via `ThreadPoolExecutor`

### 3D Backend Status

The shipped optimizer is currently 2D-only. The GUI still shows a disabled 3D selector to make
that roadmap explicit, but the unfinished 3D modules were removed so unsupported paths fail by
construction instead of drifting into misleading or physically incorrect behavior.

### Rust Accelerator Status

`rust_core/` is a **real, compiled PyO3 extension** (vectorized inverse dynamics and COM
batch evaluation on the optimizer hot path) — not a placeholder. It is _optional at build
time_: when the extension is not built, the dynamics transparently fall back to an
equivalent NumPy implementation, so results are identical and only performance differs.
Build it with `cd rust_core && maturin develop --release`. A failed extension _load_ on a
machine that should have it is logged so the silent-fallback case is observable.

## Known Limitations

- **SLSQP convergence edge cases.** A small number of extreme configurations are known to be
  numerically unstable for SLSQP on some platforms and may return `success=False`:

  - very long durations (≈30 s movements),
  - near-zero / zero range-of-motion (identical start and end angles),
  - very large multi-start counts.

  These are covered by `xfail`-marked tests in `tests/test_edge_cases.py`. If you hit one,
  reduce the duration / range or use a moderate multi-start count.

## Development

```bash
# Run tests
python3 -m pytest tests/ -v --tb=short

# Run tests with coverage
python3 -m pytest tests/ -v --cov=movement_optimizer --cov-report=term-missing

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy --ignore-missing-imports src/movement_optimizer/
```

## Testing

Tests are located in `tests/` and use shared fixtures from `conftest.py`. To run:

```bash
python3 -m pytest tests/ -v
```

Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/) for fuzzing body model parameters and joint angles.

## Benchmarking

Performance regression tests are in `tests/test_benchmarks.py`:

```bash
python3 -m pytest tests/test_benchmarks.py -v --tb=short
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.
