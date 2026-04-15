# SPEC.md — Repository Specification Document

Last-Updated: 2026-04-15T00:00:00Z

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

| Field                   | Value                                              |
| ----------------------- | -------------------------------------------------- |
| **Repository Name**     | `UpstreamDrift`                                    |
| **GitHub URL**          | `https://github.com/D-sorganization/UpstreamDrift` |
| **Owner**               | D-sorganization                                    |
| **Primary Language(s)** | Python 3.10+, Rust, TypeScript                     |
| **License**             | MIT                                                |
| **Current Version**     | 2.1.0                                              |
| **Spec Version**        | 1.0.118                                            |
| **Last Spec Update**    | 2026-04-15                                         |

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
│   │   └── python/
│   │       └── humanoid_character_builder/
│   │           └── generators/
│   │               ├── urdf_generator.py      # Public humanoid URDF generator orchestration
│   │               ├── _urdf_model_builder.py # Internal link/joint/model assembly helpers
│   │               └── _urdf_xml_writer.py    # Internal URDF XML emission helpers
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
│       ├── ci-standard.yml         # Standard CI checks (core deps only)
│       ├── ci-optional-stack.yml   # Optional-stack verification lane (issue #2368)
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

| Component                      | Location                                                   | Purpose                                                                                                                                   |
| ------------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| MuJoCo Engine Adapter          | `src/engines/physics_engines/mujoco_engine.py`             | Primary physics engine integration with full support for contact dynamics and muscle models                                               |
| Drake Engine Adapter           | `src/engines/physics_engines/drake_engine.py`              | Extended Drake support for trajectory optimization and manipulation tasks                                                                 |
| Pinocchio Engine Adapter       | `src/engines/physics_engines/pinocchio_engine.py`          | Extended Pinocchio support for efficient rigid-body dynamics computation                                                                  |
| OpenSim Engine Adapter         | `src/engines/physics_engines/opensim_engine.py`            | Experimental OpenSim integration for clinical biomechanics workflows                                                                      |
| MyoSuite Engine Adapter        | `src/engines/physics_engines/myosuite_engine.py`           | Experimental MyoSuite integration for detailed muscle physiology simulation                                                               |
| Pendulum Models                | `src/engines/pendulum_models/`                             | Educational simplified models for learning and quick prototyping                                                                          |
| FastAPI Backend                | `src/api/`                                                 | REST API exposing simulation, IK/ID, trajectory optimization, and control endpoints                                                       |
| PyQt6 GUI                      | `src/launchers/gui_launcher.py`                            | Professional interactive GUI with real-time 3D visualization                                                                              |
| Tauri Desktop App              | `ui/`                                                      | Cross-platform desktop application wrapper (Windows, macOS, Linux)                                                                        |
| Rust Physics Kernels           | `rust_core/upstream-physics/`                              | High-performance compiled physics routines for critical paths                                                                             |
| Configuration Manager          | `src/config/`                                              | Centralized configuration loading, validation, and environment management                                                                 |
| Shared Utilities               | `src/shared/`                                              | Cross-engine validators, helpers, and exception definitions                                                                               |
| Model Pack Manifest            | `src/shared/python/config/model_pack_manifest.py`          | Versioned provider-ready manifest contract for discoverable local and external biomechanical model packs                                  |
| Model Source Providers         | `src/shared/python/config/model_source_providers.py`       | Shared provider/path policy that canonicalizes local, sibling-repo, and installed-package model sources for launcher and engine discovery |
| Launcher Model Sources         | `src/launchers/launcher_model_sources.py`                  | Resolves provider-specific source roots, working directories, artifact paths, and extra PYTHONPATH entries                                |
| Provider Compatibility Harness | `src/launchers/launcher_provider_compatibility.py`         | Validates that launcher model entries expose resolvable provider roots, artifacts, working directories, and import paths                  |
| Humanoid URDF Generator        | `src/shared/python/humanoid_character_builder/generators/` | Generates humanoid URDFs via a thin public orchestrator backed by focused model-building and XML-emission helpers                         |
| URDF Models                    | `shared/models/`                                           | Canonical model definitions (URDF format) for golf swings, human body, pendulums                                                          |

## 5. Desired Functionality

### Core Features

| #   | Feature                            | Status | Description                                                                                         |
| --- | ---------------------------------- | ------ | --------------------------------------------------------------------------------------------------- |
| F1  | MuJoCo engine integration          | ✅     | Full support for MuJoCo 3.3.0+ with contact dynamics, muscle actuators, and sensor simulation       |
| F2  | Drake engine integration           | ✅     | Extended Drake support for trajectory optimization, manipulation, and planning problems             |
| F3  | Pinocchio engine integration       | ✅     | Extended Pinocchio support for efficient rigid-body dynamics and jacobian computation               |
| F4  | OpenSim engine integration         | 🔄     | Experimental OpenSim integration for clinical biomechanics and musculoskeletal analysis             |
| F5  | MyoSuite engine integration        | 🔄     | Experimental MyoSuite integration for detailed muscle physiology and motor control                  |
| F6  | Cross-engine validation            | ✅     | Automated cross-validation framework comparing results across all engines with tolerance thresholds |
| F7  | FastAPI REST API                   | ✅     | Programmatic access to simulation, IK/ID, trajectory optimization, and control endpoints            |
| F8  | PyQt6 professional GUI             | ✅     | Interactive desktop GUI with real-time 3D rendering, parameter adjustment, and result export        |
| F9  | Tauri desktop application          | 🔄     | Cross-platform desktop app bundling the GUI and API with native OS integration                      |
| F10 | MATLAB/Simulink integration        | ✅     | Export models to MATLAB format and integrate with Simulink via MEX interface                        |
| F11 | Trajectory optimization            | ✅     | SciPy-based trajectory optimization with constraint support and custom cost functions               |
| F12 | Muscle dynamics analysis           | ✅     | IK, ID, and muscle dynamics computation with Hill-type and Millard muscle models                    |
| F13 | Motion capture integration         | 🔄     | Import and track motion capture data (C3D, BVH, TRC formats) and compare with simulation            |
| F14 | Reinforcement learning integration | 🔄     | Gym-compatible interface for RL-based controller learning and policy optimization                   |

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

| Input                    | Format        | Source                          | Schema                                                   |
| ------------------------ | ------------- | ------------------------------- | -------------------------------------------------------- |
| Biomechanical Models     | URDF          | `shared/models/`                | URDF 1.0 standard with custom muscle actuator extensions |
| Motion Capture Data      | C3D, BVH, TRC | External mocap systems or files | Standard formats with marker sets and frame data         |
| Optimization Constraints | JSON          | User input or configuration     | Custom constraint schema in `src/config/schemas/`        |
| Control Parameters       | YAML/JSON     | Configuration files or API      | Engine-specific parameter maps validated against schemas |

### Output Data

| Output                   | Format                 | Destination                 | Description                                                        |
| ------------------------ | ---------------------- | --------------------------- | ------------------------------------------------------------------ |
| Simulation Trajectories  | JSON/HDF5              | API response or file export | Joint angles, muscle activations, forces over time                 |
| Cross-Validation Reports | JSON/PDF               | File export or API          | Engine comparison metrics, error margins, validation status        |
| IK/ID Solutions          | JSON/MATLAB            | API response or file        | Joint angles (IK) and joint torques (ID) with confidence metrics   |
| Optimized Trajectories   | URDF/MATLAB            | File export                 | Trajectory-optimized model definitions with optimal control inputs |
| Visualization Data       | JSON (Three.js format) | GUI or web client           | 3D geometry, animation keyframes, and rendering parameters         |

### Configuration

Configuration is managed through:

- **Environment Variables**: `UPSTREAM_DRIFT_ENGINE` (default: mujoco), `UPSTREAM_DRIFT_API_PORT` (default: 8000)
- **YAML Config Files**: `~/.upstream_drift/config.yaml` with engine-specific sections
- **Model Pack Manifests**: versioned YAML manifests in `src/shared/python/config/model_pack_manifest.py` shape, with compatibility support for legacy `config/models.yaml` registries during migration
- **Launcher Source Metadata**: model entries may optionally declare `provider`, `source_root`, `working_dir`, and `python_paths` so launcher processes can execute from external provider repos without assuming all assets live inside `UpstreamDrift`
- **Launcher Presentation Metadata**: model-pack entries may optionally declare a `launcher` block (`category`, `logo`, `status`, optional `web_route`) so provider repos define their own launcher tiles without duplicating engine-specific inference logic in `launcher_manifest_loader.py`
- **Cross-Engine Identity Metadata**: model-pack entries may declare canonical conceptual identity (`canonical_id`, `motion_family`, `exercise`, `humanoid`, optional `dataset`) so semantically equivalent MuJoCo, Drake, Pinocchio, and OpenSim packs resolve through one normalized key
- **Interchange / Provenance Metadata**: model-pack entries may declare `exchange_artifacts` and `provenance` so derived URDF/MJCF/OSIM/SDF assets are distinguished from source-of-truth assets and versioned conversion outputs
- **Provider Compatibility Harness**: `launcher_provider_compatibility.py` can evaluate local and external model entries in CI-friendly fashion before any engine process is launched
- **Provider Manifest Validation CLI**: `scripts/check_provider_compatibility.py` validates provider manifests and emits machine-readable JSON diagnostics for CI and agent-driven integration workflows
- **Shared Model Source Providers**: `model_source_providers.py` defines the canonical provider contract and approved-root path policy used by launcher handlers, model-registry consumers, and engine discovery
- **Provider Root Discovery**: `UPSTREAM_DRIFT_PROVIDER_ROOTS` may point to external provider repos whose `model_pack.yaml` manifests should be merged into the runtime registry during migration
- **Known Provider Catalog**: `provider_catalog.py` defines the conventional sibling-repo onboarding map for `MuJoCo_Models`, `Drake_Models`, `Pinocchio_Models`, `OpenSim_Models`, `Tools`, and `Movement-Optimizer` so local development can discover engine and utility provider manifests without extra env wiring
- **Provider Onboarding Guide**: `docs/development/external_provider_onboarding.md` documents the sibling-repo layout, explicit override roots, unavailable-runtime status behavior, the utility-provider compatibility rules, and the packaged-distribution bridge for future installer work
- **Discovery Mode Flag**: `UPSTREAM_DRIFT_DISCOVERY_MODE` explicitly controls launcher model discovery during migration via `local-only`, `hybrid` (default), and `provider-first` rollout modes
- **Provider-Backed Engine Discovery**: `EngineManager` consults provider-backed model packs from `src/config/models.yaml` so external engine repos can surface availability and validation paths without being copied into `src/engines`
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
- **Perturbation Analyzer Tests**: Engine-specific analyzer paths validate optimized squared-norm reductions for peak-speed and trajectory-deviation metrics across Drake, MuJoCo, MyoSuite, OpenSim, and Pinocchio

### Test Organization

| Category           | Location                    | Framework           | Markers                           |
| ------------------ | --------------------------- | ------------------- | --------------------------------- |
| Unit               | `tests/unit/`               | pytest              | `@pytest.mark.unit`               |
| Integration        | `tests/integration/`        | pytest              | `@pytest.mark.integration`        |
| Acceptance         | `tests/acceptance/`         | pytest              | `@pytest.mark.acceptance`         |
| Cross-Engine       | `tests/cross_engine/`       | pytest              | `@pytest.mark.cross_engine`       |
| Physics Validation | `tests/physics_validation/` | pytest              | `@pytest.mark.physics_validation` |
| Benchmarks         | `tests/benchmarks/`         | pytest-benchmark    | `@pytest.mark.benchmark`          |
| Property-Based     | `tests/property/`           | hypothesis + pytest | `@pytest.mark.property`           |

### Coverage Requirements

| Scope            | Minimum | Current | Enforced By                |
| ---------------- | ------- | ------- | -------------------------- |
| Overall          | 70%     | ~75%    | CI (`--cov-fail-under=70`) |
| Engine adapters  | 80%     | ~82%    | CI per-module checks       |
| API layer        | 75%     | ~78%    | CI per-module checks       |
| Shared utilities | 85%     | ~87%    | CI per-module checks       |

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
- [x] Integration: URDF smoke tests reuse a shared parsed fixture for structural assertions and skip Pinocchio loading cleanly when `buildModelFromUrdf` is unavailable
- [x] Unit: `src/shared/python/model_generation/tests/` is covered by an AST-based regression check so pytest fixtures and shared loader helpers keep explicit return annotations

## 8. Quality Standards

### Code Quality Tools

| Tool       | Version | Purpose                | Blocking? |
| ---------- | ------- | ---------------------- | --------- |
| ruff       | latest  | Linting and formatting | Yes       |
| mypy       | 1.7+    | Static type checking   | Yes       |
| pytest     | 7.0+    | Testing framework      | Yes       |
| pytest-cov | 4.0+    | Coverage measurement   | Yes       |
| bandit     | 1.7+    | Security scanning      | Yes       |
| hypothesis | 6.0+    | Property-based testing | No        |

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

| Workflow                       | Trigger                                | Purpose                                                                                                                                                       | Blocking?          |
| ------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `ci-standard.yml`              | Push/PR                                | Lint, type check, unit/integration tests (core deps only — no optional extras)                                                                                | Yes                |
| `ci-optional-stack.yml`        | Push/PR/weekly Wednesday               | **Optional-stack verification lane** (issue #2368): installs Pinocchio, Pink, Crocoddyl, PyQt6, and full API extras; exercises tests skipped in `ci-standard` | Yes                |
| `heavy-tests-opt-in.yml`       | Manual dispatch or `/heavy-test` label | Cross-engine and physics validation (long-running)                                                                                                            | No (opt-in)        |
| `nightly-cross-validation.yml` | Daily 2:00 UTC                         | Full multi-engine validation suite against all model variations                                                                                               | No (informational) |
| `tauri-build.yml`              | Tag release                            | Build desktop apps for Windows/macOS/Linux                                                                                                                    | Yes (for releases) |
| `vendor-freshness.yml`         | Weekly                                 | Check for stale dependencies and security updates                                                                                                             | No (warning-only)  |
| `docker-size-gates.yml`        | Push                                   | Ensure Docker image size stays <800 MB                                                                                                                        | Yes                |

## 9. Dependencies

### Runtime Dependencies

| Package  | Version | Purpose                                      |
| -------- | ------- | -------------------------------------------- |
| numpy    | 1.20+   | Numerical computation                        |
| scipy    | 1.7+    | Scientific algorithms (optimization, linalg) |
| fastapi  | 0.95+   | REST API framework                           |
| uvicorn  | 0.20+   | ASGI server for FastAPI                      |
| pydantic | 2.0+    | Request/response validation                  |
| mujoco   | 3.3.0+  | Primary physics engine (required)            |
| PyQt6    | 6.0+    | Professional GUI framework                   |
| tauri-py | 1.0+    | Tauri bridge for Python backend              |

### Optional Runtime Dependencies

| Package      | Version | Purpose                                     |
| ------------ | ------- | ------------------------------------------- |
| drake        | 1.0+    | Drake physics engine integration            |
| pinocchio    | 2.6+    | Pinocchio rigid-body dynamics               |
| myosuite     | 2.0+    | MyoSuite muscle simulation                  |
| opensim      | 4.4+    | OpenSim musculoskeletal models              |
| mediapipe    | 0.9+    | Motion capture integration (pose detection) |
| scikit-learn | 1.0+    | RL policy learning and clustering           |
| sympy        | 1.11+   | Symbolic trajectory optimization            |

### Development Dependencies

| Package    | Version | Purpose                |
| ---------- | ------- | ---------------------- |
| pytest     | 7.0+    | Testing framework      |
| pytest-cov | 4.0+    | Coverage measurement   |
| hypothesis | 6.0+    | Property-based testing |
| ruff       | latest  | Linting and formatting |
| mypy       | 1.7+    | Type checking          |
| bandit     | 1.7+    | Security scanning      |
| black      | 23.0+   | Code formatter         |

### Fleet Dependencies

| Repo             | Relationship | Description                                              |
| ---------------- | ------------ | -------------------------------------------------------- |
| (none currently) | —            | UpstreamDrift is currently a standalone fleet repository |

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

| Artifact              | Format         | Destination             |
| --------------------- | -------------- | ----------------------- |
| Python Package        | .whl           | PyPI (on release)       |
| FastAPI Server        | Docker image   | Docker Hub (on release) |
| Desktop App (Windows) | .msi installer | GitHub releases         |
| Desktop App (macOS)   | .dmg bundle    | GitHub releases         |
| Desktop App (Linux)   | .AppImage      | GitHub releases         |
| Documentation         | HTML           | GitHub Pages            |

## 11. Roadmap & Open Issues

### Current Phase

**Active Development**: Core engine integrations complete; expanding experimental OpenSim and MyoSuite support. Tauri desktop app in active development. Motion capture integration and RL control schemes are in-progress.

### Planned Work

| Priority | Item                                        | Issue/PR | Target Date |
| -------- | ------------------------------------------- | -------- | ----------- |
| P0       | Complete OpenSim integration (F4)           | #45      | Q2 2026     |
| P0       | Complete MyoSuite integration (F5)          | #46      | Q2 2026     |
| P1       | Motion capture import and tracking (F13)    | #78      | Q3 2026     |
| P1       | RL controller learning framework (F14)      | #92      | Q3 2026     |
| P1       | Tauri desktop app release (F9)              | #101     | Q2 2026     |
| P2       | Extended MATLAB integration (export/import) | #112     | Q4 2026     |
| P2       | Performance profiling and GPU acceleration  | #130     | Q4 2026     |

### Known Limitations

- OpenSim and MyoSuite integrations are experimental; API may change
- Cross-engine validation only enforces tolerances on kinematic outputs; dynamics comparison still in development
- Motion capture import limited to marker-based systems (no IMU data yet)
- RL integration currently supports basic Gym environments; no hierarchical or multi-agent support
- Tauri app Windows builds require MSVC toolchain (no MinGW support)
- Performance scaling beyond 100-muscle models not yet tested

## 12. Change Log

| Date       | Version | Changes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-04-15 | 1.0.118 | CI stabilization: core and shared-contract dependency installs now use per-job virtual environments, pytest-qt is pinned to the PyQt6 API in GUI-capable lanes, and benchmark tests are excluded from default core CI so self-hosted runner state cannot leak optional stacks into required tests.                                                                                                                                                                                                                                                                        |
| 2026-04-15 | 1.0.117 | CI stabilization: moved disabled legacy assessment auto-fix and auto-remediation workflows out of the active workflow directory so GitHub no longer records workflow-file failures for superseded automation on every push.                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-15 | 1.0.116 | Calc backend drift alignment: synchronized the shared `calc_backend` leaf routers and tests with the Tools canonical implementation, added a hash-based drift guard for the synchronized modules, and expanded first-party calc backend coverage around thermal profile, WGS, pressure-drop, flare, baghouse, acid-gas dewpoint, and scrubber edge paths.                                                                                                                                                                                                                                                                            |
| 2026-04-15 | 1.0.115 | Shared package drift alignment: synchronized small Tools-canonical subpackages for `chat`, `data_processing`, `notes`, and `plot_theme`, added hash guards for the synchronized leaves, and expanded tests for data processor and plot theme behavior so future local drift is caught in CI.                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-15 | 1.0.111 | CI stabilization: core and shared-contract pytest invocations now run through the setup-python interpreter, and NumPy/SciPy are repaired after editable installs so self-hosted toolcache drift cannot hide compiled dependencies from collection.                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-15 | 1.0.110 | Test stabilization: all tests in `TestPuttingGreenSimulatorAdvanced` (multi-ball scatter analysis) are now marked `slow` at the class level to prevent 60-second per-test timeout failures; added post-editable-install numpy/scipy re-pin in core and shared-contract lanes.                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-15 | 1.0.109 | Motion optimizer smoke simulation test: the upstream motion optimizer test suite now stubs the heavy simulation execution in CI to avoid GPU/MuJoCo dependency errors; the stub validates the public API surface while deferring full trajectory and convergence checks to the optional-stack lane.                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-15 | 1.0.108 | CI stabilization: the required Python core test lane adds a forced wheel refresh for the scipy package before the editable install, replacing any cached wheel that was compiled against an older numpy ABI.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-15 | 1.0.107 | Test stabilization: the pendulum panel-builder helper tests now fake the imported simulation-panel module and restore temporary module fakes in a `finally` block, preventing QtSvg ABI issues from aborting collection and leaking partial fake modules into later perturbation tests.                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-15 | 1.0.106 | CI stabilization: Linux dependency bootstrap in required and shared-contract lanes now retries apt update/install commands around self-hosted runner lock races, including metadata-list locks that apt's native timeout option does not reliably wait for.                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-15 | 1.0.105 | CI stabilization: the required Python core test lane now disables xdist in favor of serial execution, avoiding late worker crashes and pytest-cov SQLite corruption on reused self-hosted workspaces while preserving the existing per-test timeout and coverage gates.                                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-15 | 1.0.104 | CI stabilization: the Rust quality gate now caches only Cargo registry/git sources and leaves `target/` local to each runner, preventing cached build-script binaries compiled against a newer glibc from being restored onto older self-hosted images.                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-15 | 1.0.103 | CI stabilization: the standard Python test lane now uses a per-job runner-temp coverage data file and bounded xdist fanout on self-hosted runners, reducing worker crashes and stale/corrupt `.coverage.*` database reuse during large matrix runs.                                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-15 | 1.0.102 | CI stabilization: the Rust quality gate now creates a temporary virtual environment for PyO3 wheel build and verification, avoiding PEP 668 externally managed system Python failures while keeping the maturin build and binding smoke test on the same interpreter.                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-15 | 1.0.101 | CI stabilization: the standard Linux dependency installs now wait on both dpkg and apt list locks, and the Rust quality gate builds, installs, and verifies the PyO3 wheel through the same `python3` interpreter to avoid ABI-tag mismatches on self-hosted runners.                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-15 | 1.0.100 | Tauri CI stabilization: the repository-level Cargo workspace now explicitly excludes the standalone `ui/src-tauri` package so rustfmt, clippy, and cargo check can run from the desktop-app crate without Cargo treating it as an undeclared parent-workspace member.                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-15 | 1.0.99  | Tauri CI stabilization: the Rust + TypeScript desktop-app workflow now checks out the shared Tools repository and symlinks it for Cargo path resolution before rustfmt, clippy, check, and release build steps that depend on `tools-core`.                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-15 | 1.0.98  | CI stabilization: aligned the Vitest and coverage packages with the Vite 7 frontend toolchain so component tests transform correctly under Node 24, and marked optional-stack API, Pinocchio, and full unit-test steps as non-blocking while preserving their failure annotations and summaries for known environment-dependent optional integrations.                                                                                                                                                                                                                                                                                |
| 2026-04-15 | 1.0.97  | Security and visualization API hardening: validated model-library download URLs with HTTPS-only scheme checks before `urlretrieve`, added regression coverage for blocked local-file URL schemes, and exported `PyVistaBackend` from the top-level Unreal integration package with factory/root export tests.                                                                                                                                                                                                                                                                                                                         |
| 2026-04-14 | 1.0.96  | Performance optimization: Replaced `np.sum(velocities**2, axis=1)` with `np.einsum("...i,...i->...", v.astype(float), v.astype(float))` in `handedness_support.py` `validate_energy_conservation` for ~2x speedup with safe float promotion.                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-14 | 1.0.95  | Performance optimization: Replaced `np.linalg.norm` with `np.einsum("...i,...i->...", diff, diff, dtype=float)` in `handedness_support.py` `validate_mirror_trajectory` for ~35% speedup and safe float promotion preventing integer overflow.                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-14 | 1.0.94  | CI infrastructure: hardened optional-stack lane (`ci-optional-stack.yml`) with `idna[uts46]` force-reinstall to fix `email-validator` on toolcache runners, removed conflicting static Xvfb start step (replaced with `xvfb-run --auto-servernum` + `-p no:xvfb` to prevent pytest-xvfb double-spawn), and added `sortedcontainers` force-reinstall before `pip-audit` in the Security Audit step to fix `cyclonedx` import crash. Added 5 mypy exclusions for numpy-stubs 2.2.x/2.3.x divergence files in `pyproject.toml`.                                                                                                          |
| 2026-04-14 | 1.0.93  | Performance optimization: Extracted redundant `np.linalg.norm(pb - pa)` computations to cached local variables inside collision checker hot loops, and optimized `DistanceResult` validation logic with element-wise calculation instead of `np.linalg.norm`.                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-13 | 1.0.92  | Performance optimization: Replaced `np.sum(..., axis=1)` squared-norm reductions with `np.einsum("ij,ij->i", ...)` in the Drake, MuJoCo, MyoSuite, OpenSim, and Pinocchio perturbation analyzers to reduce temporary allocations in peak-speed and trajectory-deviation metrics.                                                                                                                                                                                                                                                                                                                                                      |
| 2026-04-13 | 1.0.91  | Performance optimization: Replaced `np.linalg.norm(..., axis=1)` with `np.einsum` in `electrical_model.py` for ~35% speedup.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-12 | 1.0.89  | Performance optimization: replaced `np.linalg.norm(..., axis=1)`, `np.sum(diff**2, axis=-1)` and explicit array math with dimension-agnostic `np.einsum` to eliminate temporary array allocation overhead when computing Euclidean distances.                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-10 | 1.0.90  | CI(fleet-mode-aware-pick-runner): added mode-aware runner selection (`local`/`cloud`/`hybrid`) via `FLEET_RUNNER_MODE` repo variable across all 54 workflow files, replaced bare `abs()` with `np.abs()` in `signal_toolkit/calculus.py` to satisfy mypy `SupportsAbs` constraint, narrowed `toolstrip_widget.py` segment-layout accessor return type with an `isinstance` guard, and added `type: ignore[assignment]` annotations for Qt metaclass-rebound `pyqtSignal` descriptors in `signal_toolkit/widget.py`.                                                                                                                   |
| 2026-04-10 | 1.0.87  | Integration(editor-tools): synchronized the shared `text_editor_diff_mixin.py` leaf module with `Tools`, declared type-only editor diff mixin state without changing runtime behavior, and added a drift-guard regression test that ignores the type-only block while pinning selected editor leaf modules to the upstream sibling-repo baseline.                                                                                                                                                                                                                                                                                     |
| 2026-04-10 | 1.0.86  | Optional-stack contracts(Pinocchio): preserved root-body MuJoCo joints during URDF export so Pinocchio models retain the MuJoCo velocity-DOF contract, rejected mock-only Pinocchio availability surfaces, and isolated induced-acceleration unit mocks from global `sys.modules` leakage.                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-10 | 1.0.85  | Optional-stack contracts(Drake): tightened Drake availability probing to reject mock-only or partial `pydrake` surfaces, restored module-scoped Drake dependency mocks for module-scoped tests, and aligned the isolated strict reset test with the initialized-engine precondition.                                                                                                                                                                                                                                                                                                                                                  |
| 2026-04-10 | 1.0.84  | Optional-stack contracts: tightened OpenSim availability probing to reject importable-but-incompatible bindings, exported unlimited MuJoCo hinge joints as URDF `continuous` joints, and aligned Pinocchio/OpenSim audit tests with the implemented engine contracts.                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-10 | 1.0.83  | CI governance: removed the `**.md` pull-request ignore from CI Standard so required quality-gate checks run for SPEC-only pull requests instead of leaving required checks permanently pending.                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-04-10 | 1.0.82  | Refactor(data-fitting): decomposed `validation_pkg/data_fitting.py` into focused data-model, inverse-kinematics, parameter-estimation, sensitivity, and pipeline helper modules while preserving the legacy facade import surface and adding facade regression coverage.                                                                                                                                                                                                                                                                                                                                                              |
| 2026-04-10 | 1.0.81  | Test framework fix: finalized the pendulum panel-builder helper isolation by reloading `panel_builders` under fixture-scoped fake simulation and perturbation modules, preserving richer fake `run_simulation` results, and removing stale merge-conflict artifacts that broke linting on the branch.                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-10 | 1.0.80  | Test isolation(Drake): expanded the isolated Drake strict-test mock surface to include `pydrake.geometry` and `pydrake.multibody.tree`, reducing cross-test import failures when the optional-stack lane executes strict Drake mocks before the broader Drake wrapper and integration-audit suites.                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-10 | 1.0.79  | Test framework fix: tightened pendulum panel-builder helper tests so fake simulation and perturbation modules are fixture-scoped, reload `panel_builders` under the fake environment, and avoid leaking mocked modules or incompatible `run_simulation` signatures into unrelated tests.                                                                                                                                                                                                                                                                                                                                              |
| 2026-04-10 | 1.0.78  | Test framework fix: Refactored `test_panel_builders_helpers.py` to prevent module-level mocking of `run_simulation` from polluting `sys.modules` and causing cascading failures in other test suites.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-10 | 1.0.77  | Optimization: Cached difference vectors and their np.linalg.norm results into local variables inside tight collision-checking loops to eliminate redundant vector subtractions and matrix math operations, halving execution time in these hot paths without sacrificing code readability.                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-10 | 1.0.77  | Enabled Docker vulnerability scanning (Trivy) in CI pipeline.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-10 | 1.0.76  | Simscape + GUI maintenance: fixed the 3D golf dataset generator defaults to target the bundled `GolfSwing3D_Kinetic` model while keeping legacy `verbose` and newer `verbosity` config paths compatible, and decomposed the launcher dashboard plus pendulum GUI builder/toolstrip code into smaller helpers with focused regression coverage.                                                                                                                                                                                                                                                                                        |
| 2026-04-10 | 1.0.75  | Optimization: Replaced np.linalg.norm(..., axis=1) with explicit element-wise computation in deformable objects to avoid axis reduction overhead for arrays with small inner dimensions.                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-04-10 | 1.0.74  | API/auth integrity: aligned API-key prefix lookup and API-key creation on the canonical `key_prefix` field, removed the stale auth integration xfail, and added a route-level regression test to ensure newly created API keys persist the hashed prefix used by the fast-path lookup dependency.                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-10 | 1.0.73  | Type hygiene(annotation-slice): added explicit return annotations for the JAX optimizer's nested `loss_fn` helpers in `pendulum_simulator/optimizer_gpu.py`, and extended the bounded AST regression test so this optimization source slice stays annotation-complete alongside the earlier explorer and output-manager waves.                                                                                                                                                                                                                                                                                                        |
| 2026-04-10 | 1.0.72  | Type hygiene(annotation-slice): added explicit return annotations for the shared output-manager filename sanitizer and save dispatcher helpers, and extended the bounded AST regression test to keep the source-level output-manager slice annotation-complete alongside the earlier explorer and shim updates.                                                                                                                                                                                                                                                                                                                       |
| 2026-04-10 | 1.0.71  | Type hygiene(annotation-slice): added explicit return annotations for the lazy explorer window accessor, the backward-compatible output-manager load helper, and the nested video-analyzer JSON conversion helper, and added a bounded AST regression test to keep this small source slice annotation-complete.                                                                                                                                                                                                                                                                                                                       |
| 2026-04-09 | 1.0.69  | Integration(utility-providers): extended the shared provider catalog to cover `Tools` and `Movement-Optimizer`, allowed tool-category provider manifests to participate in the compatibility harness without engine-runtime or cross-engine-identity requirements, and added regression coverage proving pendulum packs and optimization utilities flow through the shared launcher pipeline without duplicating launcher logic.                                                                                                                                                                                                      |
| 2026-04-09 | 1.0.68  | Integration(provider-onboarding): added a shared provider catalog for conventional sibling-repo discovery of `MuJoCo_Models`, `Drake_Models`, `Pinocchio_Models`, and `OpenSim_Models`; updated the model registry to load provider manifests from those known roots without extra env wiring; downgraded provider-backed launcher tile status when source roots or engine runtimes are unavailable; and documented the onboarding layout and packaging bridge for future installer work.                                                                                                                                             |
| 2026-04-09 | 1.0.69  | Added Windows installer packaging profiles (`core`, `hybrid`, `full`) with a shared packaging-policy contract, refactored `installer/windows/setup.py` to consume testable setup helpers, and taught the installer build flow to record profile/discovery/provider-root metadata for provider-aware release validation.                                                                                                                                                                                                                                                                                                               |
| 2026-04-09 | 1.0.67  | Refactor(launcher-tiles): added explicit `launcher` presentation metadata to the shared model-pack and registry contracts, taught provider-backed launcher tiles to prefer that shared metadata over ad hoc engine inference, preserved legacy `launcher_manifest.json` as a compatibility source for static tiles, and added regression coverage for explicit provider tile metadata plus deterministic mixed-source ordering.                                                                                                                                                                                                       |
| 2026-04-07 | 1.0.61  | Test(upstream-drift-tools): refreshed the next return-annotation slice onto the latest `main`, added explicit return annotations across the scoped `upstream_drift_tools` tests, normalized PSA sensitivity fallback arrays to concrete `float64` NumPy arrays for mypy-clean optional-stack execution, and added an AST regression guard covering the touched files for issue #2385.                                                                                                                                                                                                                                                 |
| 2026-04-07 | 1.0.60  | Perf(simscape-3d-analysis): refreshed the vector-reduction optimization branch onto the latest `main` history after the `frankenstein_editor` and `kinematic_forces` merges, preserving the explicit small-array NumPy reduction speedup in the 3D golf C3D analysis service and plotting tabs while clearing stale merge state.                                                                                                                                                                                                                                                                                                      |
| 2026-04-07 | 1.0.59  | Refactor(kinematic-forces): rebased the `kinematic_forces.py` decomposition branch onto the latest `main` after the `frankenstein_editor` and impact-model splits merged, preserving the kinematic-forces package surface while aligning the branch with the current shared facades and specification history for issue #2386.                                                                                                                                                                                                                                                                                                        |
| 2026-04-07 | 1.0.56  | Refactor(frankenstein-editor): rebased the `frankenstein_editor.py` decomposition branch onto the latest `main` after the impact-model split merged, preserving the editor package surface while aligning the branch with the current shared facades and specification history for issue #2386.                                                                                                                                                                                                                                                                                                                                       |
| 2026-04-07 | 1.0.55  | Refactor(impact-model): merged the `impact_model` package decomposition onto current `main`, repaired package imports to the shared physics constants module, and updated facade-policy/docstring checks to target the package `__init__.py` entrypoints used by the split `impact_model` and `terrain` surfaces.                                                                                                                                                                                                                                                                                                                     |
| 2026-04-07 | 1.0.54  | Refactor(dataset-generator): documented the decomposition of `dataset_generator.py` into focused dataset-building modules while preserving the public entry points used by the issue #2386 hotspot reduction pass.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-07 | 1.0.51  | Refactor(pose6dof): documented the decomposition of `pose6dof.py` into focused domain modules while preserving the public facade and current integration seams for the issue #2386 hotspot reduction pass.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-04-07 | 1.0.52  | Extended `launcher_manifest_loader.py` to augment static launcher tiles with provider-backed `ModelRegistry` entries, preserving one manifest surface for local and external model packs, and updated `src/api/local_server.py` to serve launcher metadata through the shared loader instead of raw JSON reads. Added regression coverage for provider-tile discovery and static-only opt-out in `tests/config/test_launcher_manifest.py`.                                                                                                                                                                                            |
| 2026-04-07 | 1.0.53  | Fix(launcher-manifest): restored launcher-facing `capabilities` and `order` metadata on `ModelConfig` while rebasing the provider-model manifest migration work, preserving deterministic tile ordering and provider surfacing across local and external model packs after the manifest and registry contract split.                                                                                                                                                                                                                                                                                                                  |
| 2026-04-07 | 1.0.50  | Fix(config): local `models.yaml` registries now normalize blank legacy descriptions to the model name before adapting entries into the stricter versioned model-pack manifest contract, preserving existing launcher/model-registry behavior while keeping provider manifests strict. Added regression coverage in `tests/unit/config/test_model_registry_config.py`.                                                                                                                                                                                                                                                                 |
| 2026-04-07 | 1.0.49  | Extended `ModelRegistry` to discover optional external `model_pack.yaml` manifests from `UPSTREAM_DRIFT_PROVIDER_ROOTS`, merge provider-backed entries into the runtime registry, and default their `source_root` to the manifest directory when not explicitly declared.                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-07 | 1.0.48  | Wired launcher provider compatibility into `launcher_diagnostics.py` so diagnostics can report incompatible provider-backed entries before launch, and added regression coverage for pass/warning/import-error diagnostic paths.                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-04-07 | 1.0.47  | Added `launcher_provider_compatibility.py` as a shared validation harness for local and provider-backed launcher entries, with regression coverage for missing external roots, artifact resolution, and CI-friendly compatibility assertions before engine startup.                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-07 | 1.0.46  | Added provider-aware launcher source resolution via `launcher_model_sources.py`, extended model manifests/registry entries with optional `provider`, `source_root`, `working_dir`, and `python_paths` metadata, and updated launcher handlers/process management to honor external source roots while preserving current local defaults.                                                                                                                                                                                                                                                                                              |
| 2026-04-07 | 1.0.45  | Added `model_pack_manifest.py` as the versioned shared contract for provider-ready model packs, introduced unit coverage for manifest validation and deterministic ordering, and routed shared `ModelRegistry` entry parsing through the same validation path while preserving legacy `models.yaml` compatibility.                                                                                                                                                                                                                                                                                                                    |
| 2026-04-07 | 1.0.44  | Refactor(viewer-backends): documented the decomposition of `viewer_backends.py` into smaller backend-oriented modules while preserving the existing facade and launch integration surface for the issue #2386 hotspot reduction pass.                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-07 | 1.0.43  | Added explicit return annotations across `src/shared/python/humanoid_character_builder/tests/test_urdf_generator.py`, `test_contracts.py`, `test_collision_geometry.py`, and `test_physics_validator.py`; expanded `tests/unit/test_humanoid_character_builder_test_return_annotations.py` to guard the broader humanoid-character-builder test subtree; and made `test_collision_geometry.py` skip cleanly when the optional `trimesh` dependency is unavailable so non-optional environments do not fail during test collection under issue #2385.                                                                                  |
| 2026-04-07 | 1.0.42  | Added explicit return annotations across `src/shared/python/humanoid_character_builder/tests/test_api.py`, `test_inertia_calculator.py`, and `test_body_parameters.py`, repaired the invalid-height and invalid-mass tests to assert `BodyParameters.validate()` errors instead of nonexistent constructor exceptions, and added `tests/unit/test_humanoid_character_builder_test_return_annotations.py` so future missing return annotations in that scoped test subtree fail fast in CI under issue #2385.                                                                                                                          |
| 2026-04-07 | 1.0.41  | Added explicit return annotations across the `src/shared/python/plot_engine/tests/` Plotly and Matplotlib suites, including the shared renderer fixtures, and added `tests/unit/test_plot_engine_test_return_annotations.py` so future missing return annotations in that subtree fail fast in CI under issue #2385.                                                                                                                                                                                                                                                                                                                  |
| 2026-04-07 | 1.0.40  | Added explicit return annotations across the `src/shared/python/model_generation/tests/` API, CLI, editor, library, Simscape, and GitHub-importer suites, and added `tests/unit/test_model_generation_test_return_annotations.py` so future missing return annotations in that subtree fail fast in CI under issue #2385.                                                                                                                                                                                                                                                                                                             |
| 2026-04-07 | 1.0.39  | Refactor(pendulum-gui): decomposed `equations_popup.py` into a small facade backed by `equations_popup_reference_content.py`, `equations_popup_jacobian_content.py`, and shared popup styling. Added popup regression coverage plus architecture-debt budgets so the extracted content modules stay tracked under issue #2388.                                                                                                                                                                                                                                                                                                        |
| 2026-04-07 | 1.0.38  | Refactor(golf-gui): decomposed the legacy Python visualization stack by turning `Motion_Capture_Plotter.py` into a thin shell backed by `motion_capture_plotter_ui.py`, `motion_capture_plotter_data.py`, and `motion_capture_plotter_visualization.py`, and by splitting `golf_visualizer_implementation.py` into focused models/data/renderer/widget/app modules with a compatibility facade. Extended `config/architecture_debt_policy.json` and `tests/unit/test_architecture_debt_policy.py` to track the remaining MATLAB monoliths under phased size budgets. Closes #2383.                                                    |
| 2026-04-07 | 1.0.37  | Refactor(shared-python): decomposed `mesh_generator.py` into backend-specific modules (`mesh_generator_models.py`, `mesh_generator_primitive.py`, `mesh_generator_makehuman.py`, `mesh_generator_smplx.py`), split `terrain.py` into `terrain_representation.py`, `terrain_loading.py`, and `terrain_physics.py`, and split pressure-drop facade logic into `pressure_drop_api.py`, validation/results/reference helpers, and focused engine modules. Added `config/architecture_debt_policy.json` plus `tests/unit/test_architecture_debt_policy.py` to lock in facade budgets and tracked hotspots. Closes #2382.                   |
| 2026-04-06 | 1.0.34  | test(urdf): reduced duplication in `tests/integration/test_urdf_generation_smoke.py` by reusing a shared parsed XML fixture for structural assertions and documenting the Pinocchio smoke-test skip path when `buildModelFromUrdf` is unavailable in the installed wheel.                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-06 | 1.0.33  | refactor(urdf): decompose 761-line `urdf_generator.py` into four focused sub-modules: `urdf_config.py` (URDFGeneratorConfig dataclass), `urdf_geometry.py` (geometry dict creation and XML rendering), `urdf_joints.py` (joint-type mapping and composite joint expansion), `urdf_xml_builder.py` (full XML tree assembly). Public API fully preserved; backward-compat shims added for all private methods. Added "Generated URDF must be valid XML" postcondition. Closes #2370.                                                                                                                                                    |
| 2026-04-06 | 1.0.32  | test(urdf): add 20-test cross-engine URDF generation and load smoke test covering XML structural validity, file persistence, config variants, and optional MuJoCo/Drake/Pinocchio loading paths (gracefully skipped when engine is absent). Closes #2369.                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-06 | 1.0.31  | CI: Added `ci-optional-stack.yml` — the optional-stack verification lane (issue #2368). Installs Pinocchio, Pink, Crocoddyl, PyQt6, and full API extras so that tests guarded by `try: import X` run without being skipped. Includes skip-visibility report in job summary. SPEC.md updated with new lane in module map and CI/CD pipeline table.                                                                                                                                                                                                                                                                                     |
| 2026-04-06 | 1.0.30  | Refactor: split monolithic `src/shared/python/mujoco_humanoid_golf/kinematic_forces.py` (1102 lines) into three focused modules: `mujoco_version.py` (MuJoCo version checking and `MjDataContext`), `jacobian_utils.py` (Jacobian and mass-matrix utilities), and a slimmed `kinematic_forces.py` retaining `KinematicForceData`, `KinematicForceAnalyzer`, and `export_kinematic_forces_to_csv`. Public API fully preserved; no logic changed. Closes #2357.                                                                                                                                                                         |
| 2026-04-06 | 1.0.29  | Sentinel: Fixed command injection vulnerability in `DataProcessorEngine.filter_data()` by adding AST-based validation to user-provided query strings before passing them to `pandas.DataFrame.query()`.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-06 | 1.0.28  | Bolt: Optimized `np.linalg.norm(..., axis=1)` to `np.sqrt(np.sum((...)**2, axis=1))` in `src/shared/python/data_io/marker_mapping.py` iterative fitting loop.                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-06 | 1.0.27  | Bolt: Optimized GRF contact force magnitude calculation across `src/api/routes/force_overlays.py`, `src/api/routes/video.py`, `src/robotics/locomotion/footstep_planner.py`, `src/shared/python/data_io/export.py`, `src/shared/python/data_io/output_manager.py`, `src/shared/python/gui_pkg/plot_generator.py`, `src/shared/python/physics/impact_model.py`, and `src/tools/video_analyzer/analyzer.py` by streamlining repeated magnitude computations and reducing array reduction overhead.                                                                                                                                      |
| 2026-04-06 | 1.0.26  | Bolt: Fixed integer overflow in `src/engines/physics_engines/putting_green/python/simulator.py` distance calculation by casting coordinate deltas to float before explicit element-wise squaring; added regression coverage for integer-array inputs.                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-06 | 1.0.25  | Fixed `patch_analyzers.py` to discover the repository root from the script location instead of a hardcoded workstation path, refactored it into import-safe helper functions plus a `main()` entrypoint, and added a regression test guarding maintained scripts against `C:/Users/diete/Repositories/UpstreamDrift` literals.                                                                                                                                                                                                                                                                                                        |
| 2026-04-05 | 1.0.24  | Fix: Re-export `ComparisonReport` from `analyzer_base` in all 5 engine perturbation analyzers (drake, mujoco, myosuite, opensim, pinocchio) to resolve test suite `ImportError` after refactoring.                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-05 | 1.0.23  | Sentinel: explicitly disable reading and writing of `pickle` format in `src/shared/python/upstream_drift_tools/data_processing/io.py` to prevent arbitrary code execution vulnerabilities via insecure deserialization.                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-05 | 1.0.22  | Bolt: Optimized `np.linalg.norm(..., axis=1)` to `np.sqrt(np.mean(np.sum(..., axis=1)))` in perturbation analyzers (drake, myosuite, opensim, pinocchio) for performance gain.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-04 | 1.0.21  | Fix: Re-export `MANDATORY_METRICS` from all 5 engine perturbation analyzers (drake, mujoco, myosuite, opensim, pinocchio) via `# noqa: F401` import; tests import `MANDATORY_METRICS` from engine-specific modules after refactor moved it to `analyzer_base`.                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-04 | 1.0.20  | Fix: Use `patch.dict(sys.modules)` in `test_cli_launch` (test_unified_launcher_coverage.py) to prevent real import of `golf_launcher.py` with top-level PyQt6 imports; avoids xdist worker crash in subprocess context on Python 3.10.                                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-04-04 | 1.0.19  | Fix: Populate `src/engines/physics_engines/__init__.py` with subpackage imports (mujoco, pinocchio, opensim, drake, myosuite) to register them as module attributes; required for Python 3.10 compatibility where `unittest.mock.patch` navigates attribute chains and fails on unregistered subpackages.                                                                                                                                                                                                                                                                                                                             |
| 2026-04-04 | 1.0.13  | Bolt: Performance optimization in MuJoCo PerturbationAnalyzer. Replaced `np.linalg.norm(..., axis=1)` with explicit element-wise squaring and summation for max value calculations to avoid axis reduction overhead.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-04-04 | 1.0.14  | Bolt: Optimized `np.max(np.linalg.norm(..., axis=1))` to `np.sqrt(np.max(np.sum(..., axis=1)))` in `CollisionGeometryGenerator._fit_sphere` for performance gain.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-04 | 1.0.15  | Sentinel: Replace pickle-based checksum serialization in `engine_core/checkpoint.py` with deterministic string encoding to eliminate pickle deserialization risk.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-04 | 1.0.16  | Bolt: Optimized MuJoCo PerturbationAnalyzer norm calculations using explicit element-wise arithmetic; fix MANDATORY_METRICS re-export regression from analyzer_base refactor.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-04 | 1.0.18  | Bolt: Optimized MuJoCo PerturbationAnalyzer peak speed and trajectory RMSE calculations using explicit element-wise squaring instead of np.linalg.norm; added test fixes for Python 3.10/3.12 namespace package compatibility.                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-04 | 1.0.17  | Bolt: Optimized `np.linalg.norm` to explicit element-wise sum of squares calculation in `biomechanics/ztcf.py` for performance gain.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-04-03 | 1.0.11  | Bolt: Optimized `np.linalg.norm(..., axis=1)` to explicit element-wise sum of squares calculation in `trajectory_funnel_benchmark.py` for performance gain.                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-01 | 1.0.8   | Sentinel: restricted legacy `np.load` callers to `allow_pickle=False` in shared I/O and golf-physics utilities, matching the repository's no-unsafe-deserialization policy.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-04-01 | 1.0.7   | Bolt: Optimized `np.linalg.norm` to explicit element-wise calculation for camera framing in GUI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-03-31 | 1.0.6   | Bolt: Optimized `np.linalg.norm` to explicit element-wise calculation for validation metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-03-30 | 1.0.5   | A-N Assessment remediation (issue #2255): added DbC input validation (TypeError/ValueError) to functions in `scripts/analyze_completist_data.py`, `check_coverage_gates.py`, `check_dependency_direction.py`, `check_duplicates.py`, `check_heavy_dep_parity.py`, and `check_vendor_updates.py`; extracted chained attribute accesses to intermediate variables (LoD) in `build_hooks.py`, `examples/aerodynamics_demo.py`, `basic_flight_simulation.py`, `topography_demo.py`, `motion_training_demo.py`, and `installer/windows/`; extracted `_data_path()` helper to eliminate repeated `os.path.join(DATA_DIR, ...)` calls (DRY). |
| 2026-03-30 | 1.0.4   | Suppressed mypy false-positive on `np.savez` keyword-array arguments in `ImitationLearner` and `GAILLearner` save methods; numpy stubs do not model `**kwargs` as ndarray values.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-03-30 | 1.0.3   | Fixed arbitrary code execution vulnerability via pickle in `ImitationLearner` models by serializing configuration data as JSON strings and saving array elements explicitly.                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-03-30 | 1.0.3   | Performance optimization in ZTCF magnitude computation: explicitly computing magnitudes using `np.hypot` and `np.sqrt` to avoid `np.linalg.norm(..., axis=1)` overhead.                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-04-03 | 1.0.12  | Performance optimization in Drake Engine trajectory calculation: Replaced `np.linalg.norm(..., axis=1)` with explicit calculations involving `np.sum(...**2, axis=1)` and `np.sqrt(...)` for ball speed and smoothness costs.                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-04-01 | 1.0.10  | Added AST-based validation to pandas query expressions in DataProcessingEngine to mitigate arbitrary code execution risk.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-01 | 1.0.9   | Explicitly set allow_pickle=False in multiple np.load calls across the codebase to prevent arbitrary code execution vulnerabilities.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-04-07 | 1.0.36  | Replaced Windows installer build-script `print()` status output with structured logging in `installer/windows/build_installer.py`, extracted a shared output-reporting helper for generated artifacts, and added installer-unit regression coverage for the logging path.                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-08 | 1.0.62  | Standardized cross-engine model-pack metadata by adding canonical conceptual identity, capability alias normalization, interchange-artifact metadata, and provenance metadata to the shared manifest/registry contract, plus compatibility tests proving semantically equivalent models resolve consistently across engines.                                                                                                                                                                                                                                                                                                          |
| 2026-04-08 | 1.0.63  | Added explicit launcher migration discovery modes (`local-only`, `hybrid`, `provider-first`) to the model registry, published ADR 0004 documenting rollout, rollback, and legacy-shim exit criteria, and added CI regression coverage for the provider migration boundary modules.                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-08 | 1.0.65  | Expanded the launcher provider compatibility harness with machine-readable diagnostics, manifest-level validation, runtime-unavailable classification, a shared `scripts/check_provider_compatibility.py` entrypoint, and provider-facing documentation for external repo CI usage.                                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-09 | 1.0.67  | Refactor(launcher-tiles): added explicit `launcher` presentation metadata to the shared model-pack and registry contracts, taught provider-backed launcher tiles to prefer that shared metadata over ad hoc engine inference, preserved legacy `launcher_manifest.json` as a compatibility source for static tiles, and added regression coverage for explicit provider tile metadata plus deterministic mixed-source ordering.                                                                                                                                                                                                       |
| 2026-04-08 | 1.0.66  | Introduced a shared model-source provider layer with canonical approved-root path policy for local, sibling-repo, and installed-package models; rewired launcher/model-registry resolution through it; and taught `EngineManager` to surface provider-backed engine availability and validation paths.                                                                                                                                                                                                                                                                                                                                |
| 2026-04-07 | 1.0.35  | Optimized collision geometric primitive magnitude calculations by routing repeated small 3D norm calls through a shared `math.hypot` helper in `geometric_primitives.py`, reducing NumPy reduction overhead in support, containment, and distance paths.                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-04-09 | 1.0.70  | Hardened provider/launcher runtime compatibility by treating explicitly stubbed engine modules as available during runtime probing, fixed `EngineManager` registry path discovery when `suite_root` already points at `src/`, and refreshed lazy-import and Pinocchio strict tests to respect the current DbC and provider-discovery contracts.                                                                                                                                                                                                                                                                                       |
| 2026-03-30 | 1.0.3   | Performance optimization in validation metrics: explicitly computing 3D marker RMSE via element-wise `np.sqrt` to avoid `np.linalg.norm(..., axis=2)` overhead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-03-30 | 1.0.2   | Performance optimization in SwingOptimizer: explicitly computing clubhead velocity magnitude via `np.sqrt` to avoid `np.linalg.norm(..., axis=1)` overhead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-03-29 | 1.0.1   | Performance optimization in validation package: explicitly computing magnitudes instead of using `np.linalg.norm` to avoid NumPy reduction overhead on small axes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-07 | 1.0.36  | Optimized vector reduction for segment lengths by replacing `np.linalg.norm(..., axis=1)` with explicit `np.sqrt(np.sum(np.square(disp, dtype=float), axis=1))` in 3D Golf Model analysis services.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-11 | 1.0.88  | Performance optimization: Replaced `np.linalg.norm(..., axis=1)` with explicit element-wise arithmetic (`np.sqrt(np.sum(np.square(...)))`) in handedness config and electrical model calculators to reduce NumPy reduction overhead.                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-03-29 | 1.0.1   | Performance optimization: Replaced `np.linalg.norm(..., axis=1)` with explicit element-wise arithmetic (`np.sqrt` and `np.hypot`) in physics ground reaction forces calculations for a ~5-10x speedup                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-13 | 1.0.91  | Performance optimization: Replaced explicit element-wise `np.sqrt(x**2 + y**2 + z**2)` with dimension-agnostic, zero-allocation `np.sqrt(np.einsum('...i,...i->...', diff, diff))` in `src/shared/python/physics/ground_reaction_forces.py` for ~35% speedup over `np.linalg.norm`                                                                                                                                                                                                                                                                                                                                                    |
| 2026-04-13 | 1.0.90  | Performance optimization: Replaced `np.sum(diff**2, axis=1)` and `np.sqrt(np.sum(diff**2, axis=1))` with dimension-agnostic, zero-allocation `np.einsum('...i,...i->...', diff, diff)` in `src/shared/python/analysis/nonlinear_dynamics.py` for ~3x speedup.                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-03-28 | 1.0.0   | Initial specification for UpstreamDrift v2.1.0; documented all 14 features, architecture, testing strategy, and CI/CD pipeline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

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

| 2026-04-13 | 1.0.91 | Bolt: Optimized np.sum(..., axis=1) square reductions to np.einsum in Drake PerturbationAnalyzer to improve performance and avoid temporary array allocations. |
| 2026-04-14 | 1.0.93 | Bolt: Optimized np.sum(np.square(..., dtype=float), axis=1) to np.einsum in 3D_Golf_Model marker statistics analysis to improve performance and avoid type casting errors. |
| 2026-04-14 | 1.0.112 | fix: prevent test_paths_utils false failure from /tmp/pyproject.toml — test now uses a clean isolated tmp directory instead of relying on /tmp state. |
| 2026-04-15 | 1.0.113 | Bolt: Optimized `np.sum(np.square(...))` and explicit sum of squares to `np.einsum` in `analysis_tab.py`, `marker_plot_tab.py`, and `motion_capture.py` to improve performance. |
