# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST UPDATED: 2026-03-28

  This is the canonical specification template for all repositories in the
  D-sorganization fleet. Every repo MUST have a SPEC.md at its root.

  INSTRUCTIONS:
  1. Copy this template to the root of your repository as SPEC.md
  2. Fill in every section — leave nothing as "[TODO]"
  3. Keep this document updated with every PR that changes functionality
  4. CI will block merges if SPEC.md is stale (source changed but spec didn't)

  AUDIENCE: This document is designed for both human developers AND AI agents.
  Write clearly, use concrete examples, and avoid ambiguity.
-->

## 1. Identity

| Field | Value |
|-------|-------|
| **Repository Name** | `UpstreamDrift` |
| **GitHub URL** | `https://github.com/D-sorganization/UpstreamDrift` |
| **Owner** | D-sorganization |
| **Primary Language(s)** | Python 3.10+, Rust, TypeScript |
| **License** | MIT |
| **Current Version** | 2.1.0 |
| **Spec Version** | 1.0.3 |
| **Last Spec Update** | 2026-03-30 |

## 2. Purpose & Mission

UpstreamDrift is a multi-physics golf swing biomechanical simulation platform that consolidates five leading physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) for cross-validated biomechanical analysis. It enables researchers and biomechanists to simulate human movement across models ranging from simplified 2-DOF pendulums to complex 290-muscle musculoskeletal systems, providing a unified interface for comparative physics analysis and professional-grade visualization.

## 3. Goals & Non-Goals

### Goals

- Integrate and cross-validate five physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) for biomechanical analysis
- Provide biomechanical analysis tools including inverse kinematics (IK), inverse dynamics (ID), and muscle dynamics modeling
- Enable motion capture integration and trajectory optimization
- Offer multiple control schemes (impedance, admittance, hybrid) for simulated systems
- Deliver professional GUI with real-time 3D rendering for simulation visualization
- Expose FastAPI REST backend for programmatic access and integration
- Support desktop deployment via Tauri application framework
- Provide MATLAB/Simulink integration for cross-platform workflow compatibility
- Implement reinforcement learning integration for learning-based control policies
- Support models ranging from educational 2-DOF pendulums to complex 290-muscle systems

### Non-Goals

- Not a general-purpose physics engine; focused exclusively on biomechanical simulation
- Not intended for non-biomechanical simulations (rigid body dynamics, fluid dynamics, etc.)
- Not a replacement for domain-specific tools (OpenSim for clinical analysis, MATLAB for controls research)

## 4. Architecture Overview

### System Context

UpstreamDrift sits at the center of a biomechanical simulation ecosystem. It depends on five external physics engines as pluggable backends and exposes its functionality through three primary interfaces: a professional PyQt6 GUI for interactive simulation, a FastAPI REST API for programmatic access, and a Tauri desktop application for cross-platform deployment. The system integrates with motion capture systems (via MediaPipe and custom importers), optimization libraries (SciPy, Sympy), and machine learning frameworks (scikit-learn for RL integration). The Rust core (`rust_core/upstream-physics/`) provides high-performance physics kernels for compute-intensive operations.

### Module Map

```
UpstreamDrift/
├── src/
│   ├── engines/
│   │   ├── physics_engines/        # Engine adapters and integrations
│   │   │   ├── mujoco_engine.py    # MuJoCo backend (supported)
│   │   │   ├── drake_engine.py     # Drake backend (extended)
│   │   │   ├── pinocchio_engine.py # Pinocchio backend (extended)
│   │   │   ├── opensim_engine.py   # OpenSim backend (experimental)
│   │   │   └── myosuite_engine.py  # MyoSuite backend (experimental)
│   │   └── pendulum_models/        # Simplified educational models
│   │       ├── twodof_pendulum.py
│   │       └── biomechanical_pendulum.py
│   ├── launchers/                  # GUI/CLI entry points
│   │   ├── gui_launcher.py         # PyQt6 professional GUI
│   │   └── cli_launcher.py         # Command-line interface
│   ├── api/                        # FastAPI REST backend
│   │   ├── main.py                 # API entry point
│   │   ├── endpoints/              # REST endpoint definitions
│   │   └── models.py               # Pydantic request/response models
│   ├── config/                     # Configuration management
│   │   └── configuration.py        # Config loading and validation
│   ├── shared/                     # Cross-engine utilities
│   │   ├── validators.py           # Shared validation logic
│   │   ├── utilities.py            # Helper functions
│   │   └── exceptions.py           # Exception definitions
│   └── tools/                      # Development and analysis tools
│       ├── analysis_tools.py       # Biomechanical analysis utilities
│       └── validation_tools.py     # Cross-engine validation
├── rust_core/
│   └── upstream-physics/           # Rust physics kernels
│       ├── src/
│       │   ├── lib.rs
│       │   └── physics.rs
│       └── Cargo.toml
├── ui/
│   ├── src/
│   │   ├── main.ts                 # Tauri app entry point
│   │   └── components/             # React/Vue components
│   ├── tauri.conf.json
│   └── package.json
├── shared/
│   └── models/                     # URDF/model definitions
│       ├── golf_swing_models/
│       ├── human_body_models/
│       └── pendulum_models/
├── tests/
│   ├── unit/                       # Unit tests per module
│   ├── integration/                # Cross-engine integration tests
│   ├── acceptance/                 # End-to-end scenario tests
│   ├── cross_engine/               # Cross-validation tests
│   ├── physics_validation/         # Physics accuracy tests
│   ├── benchmarks/                 # Performance benchmarks
│   └── conftest.py                 # Pytest fixtures and configuration
├── .github/
│   └── workflows/
│       ├── ci-standard.yml         # Standard CI checks
│       ├── heavy-tests-opt-in.yml  # Heavy tests (custom runner)
│       ├── nightly-cross-validation.yml
│       ├── tauri-build.yml
│       ├── vendor-freshness.yml
│       └── docker-size-gates.yml
├── pyproject.toml
├── poetry.lock
├── SPEC.md                         # This file
└── README.md
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| MuJoCo Engine Adapter | `src/engines/physics_engines/mujoco_engine.py` | Primary physics engine integration with full support for contact dynamics and muscle models |
| Drake Engine Adapter | `src/engines/physics_engines/drake_engine.py` | Extended Drake support for trajectory optimization and manipulation tasks |
| Pinocchio Engine Adapter | `src/engines/physics_engines/pinocchio_engine.py` | Extended Pinocchio support for efficient rigid-body dynamics computation |
| OpenSim Engine Adapter | `src/engines/physics_engines/opensim_engine.py` | Experimental OpenSim integration for clinical biomechanics workflows |
| MyoSuite Engine Adapter | `src/engines/physics_engines/myosuite_engine.py` | Experimental MyoSuite integration for detailed muscle physiology simulation |
| Pendulum Models | `src/engines/pendulum_models/` | Educational simplified models for learning and quick prototyping |
| FastAPI Backend | `src/api/` | REST API exposing simulation, IK/ID, trajectory optimization, and control endpoints |
| PyQt6 GUI | `src/launchers/gui_launcher.py` | Professional interactive GUI with real-time 3D visualization |
| Tauri Desktop App | `ui/` | Cross-platform desktop application wrapper (Windows, macOS, Linux) |
| Rust Physics Kernels | `rust_core/upstream-physics/` | High-performance compiled physics routines for critical paths |
| Configuration Manager | `src/config/` | Centralized configuration loading, validation, and environment management |
| Shared Utilities | `src/shared/` | Cross-engine validators, helpers, and exception definitions |
| URDF Models | `shared/models/` | Canonical model definitions (URDF format) for golf swings, human body, pendulums |

## 5. Desired Functionality

### Core Features

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| F1 | MuJoCo engine integration | ✅ | Full support for MuJoCo 3.3.0+ with contact dynamics, muscle actuators, and sensor simulation |
| F2 | Drake engine integration | ✅ | Extended Drake support for trajectory optimization, manipulation, and planning problems |
| F3 | Pinocchio engine integration | ✅ | Extended Pinocchio support for efficient rigid-body dynamics and jacobian computation |
| F4 | OpenSim engine integration | 🔄 | Experimental OpenSim integration for clinical biomechanics and musculoskeletal analysis |
| F5 | MyoSuite engine integration | 🔄 | Experimental MyoSuite integration for detailed muscle physiology and motor control |
| F6 | Cross-engine validation | ✅ | Automated cross-validation framework comparing results across all engines with tolerance thresholds |
| F7 | FastAPI REST API | ✅ | Programmatic access to simulation, IK/ID, trajectory optimization, and control endpoints |
| F8 | PyQt6 professional GUI | ✅ | Interactive desktop GUI with real-time 3D rendering, parameter adjustment, and result export |
| F9 | Tauri desktop application | 🔄 | Cross-platform desktop app bundling the GUI and API with native OS integration |
| F10 | MATLAB/Simulink integration | ✅ | Export models to MATLAB format and integrate with Simulink via MEX interface |
| F11 | Trajectory optimization | ✅ | SciPy-based trajectory optimization with constraint support and custom cost functions |
| F12 | Muscle dynamics analysis | ✅ | IK, ID, and muscle dynamics computation with Hill-type and Millard muscle models |
| F13 | Motion capture integration | 🔄 | Import and track motion capture data (C3D, BVH, TRC formats) and compare with simulation |
| F14 | Reinforcement learning integration | 🔄 | Gym-compatible interface for RL-based controller learning and policy optimization |

### API / Interface Contract

**REST API Endpoints (FastAPI)**:
- `GET /health` — Health check
- `POST /simulate` — Run single simulation with specified engine and parameters
- `POST /cross-validate` — Run multi-engine cross-validation and return results
- `POST /ik` — Solve inverse kinematics given target pose
- `POST /id` — Solve inverse dynamics given trajectory
- `POST /trajectory-optimize` — Optimize trajectory subject to constraints
- `GET /engines` — List available physics engines and their status
- `POST /export` — Export simulation model to URDF, MATLAB, or other formats

**GUI Interface (PyQt6)**:
- Model loader and parameter editor
- Real-time 3D simulation viewer with playback controls
- Cross-engine comparison visualizer
- IK/ID solver interface with result tables
- Trajectory optimization GUI with constraint editor
- Data export and report generation

**CLI Interface**:
- `upstream-drift simulate --engine mujoco --model golf_swing.urdf`
- `upstream-drift cross-validate --models model1.urdf model2.urdf`
- `upstream-drift ik --model human.urdf --target-pose [...] --engine pinocchio`

**Desktop App (Tauri)**:
- Native window management and file dialogs
- System menu integration
- Automated updates and crash reporting

## 6. Data & Configuration

### Input Data

| Input | Format | Source | Schema |
|-------|--------|--------|--------|
| Biomechanical Models | URDF | `shared/models/` | URDF 1.0 standard with custom muscle actuator extensions |
| Motion Capture Data | C3D, BVH, TRC | External mocap systems or files | Standard formats with marker sets and frame data |
| Optimization Constraints | JSON | User input or configuration | Custom constraint schema in `src/config/schemas/` |
| Control Parameters | YAML/JSON | Configuration files or API | Engine-specific parameter maps validated against schemas |

### Output Data

| Output | Format | Destination | Description |
|--------|--------|-------------|-------------|
| Simulation Trajectories | JSON/HDF5 | API response or file export | Joint angles, muscle activations, forces over time |
| Cross-Validation Reports | JSON/PDF | File export or API | Engine comparison metrics, error margins, validation status |
| IK/ID Solutions | JSON/MATLAB | API response or file | Joint angles (IK) and joint torques (ID) with confidence metrics |
| Optimized Trajectories | URDF/MATLAB | File export | Trajectory-optimized model definitions with optimal control inputs |
| Visualization Data | JSON (Three.js format) | GUI or web client | 3D geometry, animation keyframes, and rendering parameters |

### Configuration

Configuration is managed through:
- **Environment Variables**: `UPSTREAM_DRIFT_ENGINE` (default: mujoco), `UPSTREAM_DRIFT_API_PORT` (default: 8000)
- **YAML Config Files**: `~/.upstream_drift/config.yaml` with engine-specific sections
- **API Request Parameters**: Engine selection, model path, solver options passed as JSON
- **GUI Settings**: Stored in `~/.upstream_drift/gui_settings.json` (viewport, window size, recent files)

Example config.yaml:
```yaml
default_engine: mujoco
api:
  host: 0.0.0.0
  port: 8000
engines:
  mujoco:
    model_path: /path/to/models
    timestep: 0.001
  drake:
    use_simulator: true
visualization:
  default_camera: third_person
  background_color: [0.1, 0.1, 0.1, 1.0]
```

## 7. Testing Specification

### Testing Strategy

UpstreamDrift employs a comprehensive test pyramid with multiple specialized categories:
- **Unit Tests**: Test individual engine adapters, utilities, and validators in isolation
- **Integration Tests**: Test workflows combining multiple modules (e.g., load model → simulate → export)
- **Acceptance Tests**: End-to-end scenarios (e.g., full golf swing simulation with visualization)
- **Cross-Engine Tests**: Validate physics consistency across multiple engines with tolerance thresholds
- **Physics Validation Tests**: Verify results against known ground truth (analytical solutions, published benchmarks)
- **Benchmark Tests**: Performance regression detection and optimization validation
- **Property-Based Tests**: Hypothesis-driven fuzzing for robustness

### Test Organization

| Category | Location | Framework | Markers |
|----------|----------|-----------|---------|
| Unit | `tests/unit/` | pytest | `@pytest.mark.unit` |
| Integration | `tests/integration/` | pytest | `@pytest.mark.integration` |
| Acceptance | `tests/acceptance/` | pytest | `@pytest.mark.acceptance` |
| Cross-Engine | `tests/cross_engine/` | pytest | `@pytest.mark.cross_engine` |
| Physics Validation | `tests/physics_validation/` | pytest | `@pytest.mark.physics_validation` |
| Benchmarks | `tests/benchmarks/` | pytest-benchmark | `@pytest.mark.benchmark` |
| Property-Based | `tests/property/` | hypothesis + pytest | `@pytest.mark.property` |

### Coverage Requirements

| Scope | Minimum | Current | Enforced By |
|-------|---------|---------|-------------|
| Overall | 70% | ~75% | CI (`--cov-fail-under=70`) |
| Engine adapters | 80% | ~82% | CI per-module checks |
| API layer | 75% | ~78% | CI per-module checks |
| Shared utilities | 85% | ~87% | CI per-module checks |

### Required Test Scenarios

- [ ] Unit creation with valid URDF returns expected topology (chain, mass distribution)
- [ ] MuJoCo engine simulation produces reasonable trajectories with gravity effects
- [ ] Cross-engine validation identifies discrepancies >5% between engines
- [ ] IK solver converges within 10 iterations for standard human poses
- [ ] ID computation returns physically plausible torques (within 2-sigma of analytical)
- [ ] FastAPI endpoints return 200 for valid requests and 400 for invalid schema
- [ ] GUI loads model and renders 3D visualization without crashing
- [ ] Trajectory optimization improves cost function by >20% over initial guess
- [ ] Muscle dynamics simulation produces realistic activation patterns
- [ ] Cross-platform build (Windows, macOS, Linux) produces functional binaries

## 8. Quality Standards

### Code Quality Tools

| Tool | Version | Purpose | Blocking? |
|------|---------|---------|-----------|
| ruff | latest | Linting and formatting | Yes |
| mypy | 1.7+ | Static type checking | Yes |
| pytest | 7.0+ | Testing framework | Yes |
| pytest-cov | 4.0+ | Coverage measurement | Yes |
| bandit | 1.7+ | Security scanning | Yes |
| hypothesis | 6.0+ | Property-based testing | No |

### Design Principles

- **TDD**: Unit tests written before implementation; minimum 70% coverage enforced
- **Design by Contract (DbC)**: Explicit preconditions and postconditions in engine adapters
- **DRY**: Cross-engine utilities in `src/shared/` prevent code duplication
- **Orthogonality**: Engines are loosely coupled; each can be used independently
- **Explicit is Better**: Function signatures include type hints; no magic string parameters

### Custom Quality Gates (CI)

Beyond standard tools, CI enforces custom checks:
- **Dependency Direction**: No reverse dependencies (leaf → branch → root)
- **File Size Budget**: No module exceeds 500 lines; classes capped at 200 LOC
- **Import Depth**: Maximum 4 import levels to prevent circular dependencies
- **Physics Fitness**: Cross-engine validation must pass with <5% tolerance
- **Docker Size Gate**: Built images must not exceed 800 MB

### CI/CD Pipeline

| Workflow | Trigger | Purpose | Blocking? |
|----------|---------|---------|-----------|
| `ci-standard.yml` | Push/PR | Lint, type check, unit/integration tests | Yes |
| `heavy-tests-opt-in.yml` | Manual dispatch or `/heavy-test` label | Cross-engine and physics validation (long-running) | No (opt-in) |
| `nightly-cross-validation.yml` | Daily 2:00 UTC | Full multi-engine validation suite against all model variations | No (informational) |
| `tauri-build.yml` | Tag release | Build desktop apps for Windows/macOS/Linux | Yes (for releases) |
| `vendor-freshness.yml` | Weekly | Check for stale dependencies and security updates | No (warning-only) |
| `docker-size-gates.yml` | Push | Ensure Docker image size stays <800 MB | Yes |

## 9. Dependencies

### Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.20+ | Numerical computation |
| scipy | 1.7+ | Scientific algorithms (optimization, linalg) |
| fastapi | 0.95+ | REST API framework |
| uvicorn | 0.20+ | ASGI server for FastAPI |
| pydantic | 2.0+ | Request/response validation |
| mujoco | 3.3.0+ | Primary physics engine (required) |
| PyQt6 | 6.0+ | Professional GUI framework |
| tauri-py | 1.0+ | Tauri bridge for Python backend |

### Optional Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| drake | 1.0+ | Drake physics engine integration |
| pinocchio | 2.6+ | Pinocchio rigid-body dynamics |
| myosuite | 2.0+ | MyoSuite muscle simulation |
| opensim | 4.4+ | OpenSim musculoskeletal models |
| mediapipe | 0.9+ | Motion capture integration (pose detection) |
| scikit-learn | 1.0+ | RL policy learning and clustering |
| sympy | 1.11+ | Symbolic trajectory optimization |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | 7.0+ | Testing framework |
| pytest-cov | 4.0+ | Coverage measurement |
| hypothesis | 6.0+ | Property-based testing |
| ruff | latest | Linting and formatting |
| mypy | 1.7+ | Type checking |
| bandit | 1.7+ | Security scanning |
| black | 23.0+ | Code formatter |

### Fleet Dependencies

| Repo | Relationship | Description |
|------|-------------|-------------|
| (none currently) | — | UpstreamDrift is currently a standalone fleet repository |

## 10. Deployment & Operations

### How to Run

```bash
# Prerequisites
- Python 3.10 or later
- MuJoCo 3.3.0+ with license (community or pro)
- Optional: Drake, Pinocchio, OpenSim binaries on PATH
- For Tauri desktop app: Node.js 16+, Rust toolchain

# Installation
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
python -m pip install -e ".[dev]"  # Include dev dependencies
# For desktop app: cargo install tauri-cli

# Running the FastAPI Server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Running the PyQt6 GUI
python -m src.launchers.gui_launcher

# Running the CLI
upstream-drift simulate --engine mujoco --model shared/models/golf_swing.urdf

# Building the Tauri Desktop App
cd ui && npm install && npm run tauri build
# Outputs: UpstreamDrift.exe (Windows), UpstreamDrift.app (macOS), UpstreamDrift.AppImage (Linux)

# Running Tests
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/ --cov=src --cov-fail-under=70
```

### Build Artifacts

| Artifact | Format | Destination |
|----------|--------|-------------|
| Python Package | .whl | PyPI (on release) |
| FastAPI Server | Docker image | Docker Hub (on release) |
| Desktop App (Windows) | .msi installer | GitHub releases |
| Desktop App (macOS) | .dmg bundle | GitHub releases |
| Desktop App (Linux) | .AppImage | GitHub releases |
| Documentation | HTML | GitHub Pages |

## 11. Roadmap & Open Issues

### Current Phase

**Active Development**: Core engine integrations complete; expanding experimental OpenSim and MyoSuite support. Tauri desktop app in active development. Motion capture integration and RL control schemes are in-progress.

### Planned Work

| Priority | Item | Issue/PR | Target Date |
|----------|------|----------|-------------|
| P0 | Complete OpenSim integration (F4) | #45 | Q2 2026 |
| P0 | Complete MyoSuite integration (F5) | #46 | Q2 2026 |
| P1 | Motion capture import and tracking (F13) | #78 | Q3 2026 |
| P1 | RL controller learning framework (F14) | #92 | Q3 2026 |
| P1 | Tauri desktop app release (F9) | #101 | Q2 2026 |
| P2 | Extended MATLAB integration (export/import) | #112 | Q4 2026 |
| P2 | Performance profiling and GPU acceleration | #130 | Q4 2026 |

### Known Limitations

- OpenSim and MyoSuite integrations are experimental; API may change
- Cross-engine validation only enforces tolerances on kinematic outputs; dynamics comparison still in development
- Motion capture import limited to marker-based systems (no IMU data yet)
- RL integration currently supports basic Gym environments; no hierarchical or multi-agent support
- Tauri app Windows builds require MSVC toolchain (no MinGW support)
- Performance scaling beyond 100-muscle models not yet tested

## 12. Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-03-30 | 1.0.3 | Performance optimization in validation metrics: explicitly computing 3D marker RMSE via element-wise `np.sqrt` to avoid `np.linalg.norm(..., axis=2)` overhead. |
| 2026-03-30 | 1.0.2 | Performance optimization in SwingOptimizer: explicitly computing clubhead velocity magnitude via `np.sqrt` to avoid `np.linalg.norm(..., axis=1)` overhead. |
| 2026-03-29 | 1.0.1 | Performance optimization in validation package: explicitly computing magnitudes instead of using `np.linalg.norm` to avoid NumPy reduction overhead on small axes. |
| 2026-03-29 | 1.0.1 | Performance optimization: Replaced `np.linalg.norm(..., axis=1)` with explicit element-wise arithmetic (`np.sqrt` and `np.hypot`) in physics ground reaction forces calculations for a ~5-10x speedup |
| 2026-03-28 | 1.0.0 | Initial specification for UpstreamDrift v2.1.0; documented all 14 features, architecture, testing strategy, and CI/CD pipeline |

---

<!--
  SPEC MAINTENANCE RULES:

  1. WHEN TO UPDATE: Any PR that adds, removes, or changes functionality
     described in this spec MUST include a corresponding spec update.

  2. WHO UPDATES: The PR author (human or agent) is responsible.

  3. CI ENFORCEMENT: The spec-check workflow will flag PRs where source
     files changed but SPEC.md did not. This is a blocking check.

  4. REVIEW: Spec changes should be reviewed with the same rigor as code.

  5. VERSION: Bump the Spec Version field when making substantive changes.
     Use semver: major (structure change), minor (new features), patch (corrections).
-->
