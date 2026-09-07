<p align="center">
  <img src="assets/branding/logo.png" alt="UpstreamDrift" width="180"/>
</p>

<h1 align="center">UpstreamDrift</h1>

<p align="center">
  Biomechanical golf swing analysis across multiple physics engines,
  from two-degree-of-freedom pendulums to muscle-driven musculoskeletal models.
</p>

<p align="center">
  <a href="https://github.com/D-sorganization/UpstreamDrift/actions/workflows/ci-standard.yml"><img src="https://github.com/D-sorganization/UpstreamDrift/actions/workflows/ci-standard.yml/badge.svg" alt="CI Standard"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg" alt="Python 3.11 and 3.12"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

---

UpstreamDrift is a simulation and analysis platform for the golf swing. It runs
the same model definitions through several physics engines, so a result can be
checked against an independent implementation rather than taken on trust. It
covers forward and inverse dynamics, inverse kinematics, trajectory
optimization, motion-capture retargeting, and cross-engine comparison, with a
web interface and a desktop application over the top.

- **Audience**: biomechanics researchers, robotics engineers, and golf
  instruction technologists.
- **Platform**: Windows, macOS, and Linux. Python 3.11 or 3.12.
- **Status**: actively developed. MuJoCo is the supported engine; other engines
  carry the support levels stated in [Engine Support](#engine-support).

## Contents

| Section                                       | Purpose                                  |
| --------------------------------------------- | ---------------------------------------- |
| [Start Here](#start-here)                     | Choose a path for your first hour        |
| [Capabilities](#capabilities)                 | What the platform can do                 |
| [Installation](#installation)                 | Get a working environment                |
| [Running the Platform](#running-the-platform) | Entry points and what each one opens     |
| [Engine Support](#engine-support)             | Which engines are validated, and how far |
| [Documentation](#documentation)               | Guides, API reference, and architecture  |
| [Repository Layout](#repository-layout)       | Where things live                        |
| [Contributing](#contributing)                 | Development workflow                     |
| [Citation](#citation)                         | How to cite this work                    |

## Start Here

Pick the entry point that matches what you want to do.

| If you want to                           | Read                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------- |
| See one reproducible result end to end   | [Guided walkthrough](docs/portfolio/golf_modeling_demo.md)                      |
| Install and launch the platform          | [Installation](#installation), below                                            |
| Understand the whole system              | [Project map](docs/architecture/PROJECT_MAP.md)                                 |
| Choose a physics engine for your problem | [Engine selection guide](docs/engines/engine_selection_guide.md)                |
| Go from video to tracked motion          | [Motion pipeline guide](docs/motion_pipeline/README.md)                         |
| Build a humanoid model                   | [Character builder quickstart](docs/user_guide/character_builder_quickstart.md) |
| Add support for a new engine             | [Adapter authoring guide](docs/adapters/authoring_guide.md)                     |
| Browse everything                        | [Documentation hub](docs/README.md)                                             |

## Capabilities

### Modeling

- Models spanning 2 to 28 degrees of freedom, including flexible-shaft
  formulations, up to musculoskeletal models with several hundred muscle
  actuators.
- Parametric humanoid generation with anthropometric scaling and URDF export.
- MATLAB Simscape Multibody models, maintained as research references rather
  than production artifacts.

### Analysis

- **Inverse kinematics** with nullspace optimization for redundant chains.
- **Inverse dynamics** with full torque computation and force decomposition.
- **Kinematic force analysis** separating Coriolis, centrifugal, and
  gravitational contributions.
- **Trajectory optimization** for comparing candidate swing objectives across
  speed, accuracy, and efficiency.
- **Cross-engine validation** running one model definition through several
  engines and reporting the deviation.

### Control and Robotics

Impedance, admittance, hybrid force-position, and operational-space control;
parallel-mechanism analysis of the two-handed grip; manipulability and
singularity characterization; task-space control with redundancy resolution.

### Motion Capture

Load and retarget motion data in CSV, JSON, and C3D formats. Markerless tracking
runs through OpenPose or MediaPipe. See the
[motion pipeline guide](docs/motion_pipeline/README.md) for the video-to-motion
workflow.

### Visualization and Export

Real-time three-dimensional rendering with multiple camera views and
force-torque vector overlays; more than ten plot types including energy
breakdowns, phase diagrams, and three-dimensional trajectories; CSV and JSON
export for external analysis.

## Installation

### Prerequisites

- Python 3.11 or 3.12.
- Git. The shared Tools layer (`theme`, `sidekick`, `chat`, `utils`, ...) is a
  pinned submodule at `vendor/ud-tools`; the platform will not import without
  it. The repository does not use Git LFS.
- MATLAB R2023a or later with Simulink and Simscape Multibody, only for the
  MATLAB models.

The supported combinations of Python version, operating system, engine tier, and
hardware are recorded in the
[production readiness matrix](docs/operations/production-readiness.md).

### Install

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
git submodule update --init --recursive vendor/ud-tools

pip install -e ".[dev]"

python scripts/ci/verify_installation.py
```

Only the `vendor/ud-tools` submodule is required. The three model submodules
(`shared/models/opensim/opensim-models`, `shared/models/myosuite/myo_sim`,
`src/shared/tools/human-gazebo`) are optional and only needed for the
experimental OpenSim/MyoSuite engines.

`pyproject.toml` is the canonical dependency source. A Conda wrapper is
generated from it:

```bash
conda env create -f environment.yml
conda activate upstream-drift
```

Edit dependencies in `pyproject.toml` and run `make sync-deps` to regenerate
`environment.yml`.

For interface development without the physics engines:

```bash
pip install -e .
export GOLF_USE_MOCK_ENGINE=1
```

If installation fails, see
[installation troubleshooting](docs/troubleshooting/installation.md).

### Rust Kernels

The Rust build works from a clean clone. The shared `tools-core` crate is
fetched from a pinned `D-sorganization/Tools` revision, so no sibling checkout is
required.

```bash
cargo build

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip maturin

cd rust_core/upstream-physics
python -m maturin develop --features python
python -c "import upstream_physics; print(upstream_physics.IntegratorConfig())"
```

To develop against a local `Tools` checkout instead of the pinned revision, run
`scripts/setup_tools_workspace.sh`.

## Running the Platform

```bash
python launch_upstream_drift.py
```

This starts the local API server on port 8000 and opens the web interface in
your browser. The other entry points:

| Command                                           | Opens                        |
| ------------------------------------------------- | ---------------------------- |
| `python launch_upstream_drift.py`                 | Web interface, recommended   |
| `python launch_upstream_drift.py --classic`       | PyQt6 desktop application    |
| `python launch_upstream_drift.py --api-only`      | API server with no interface |
| `python launch_upstream_drift.py --engine <name>` | A single engine directly     |
| `python -m src.tools.pose_studio`                 | Pose Studio, standalone      |

`upstream-drift` is installed as a console script and accepts the same
arguments. The desktop application remains supported for users who prefer a
native window.

`launch_golf_suite.py` is a deprecated compatibility shim retained for existing
scripts. New work should use `launch_upstream_drift.py`.

Development tasks run through the Makefile:

```bash
make help      # List available targets
make check     # Run linters and tests
make format    # Apply Ruff formatting
```

## Launcher Tiles

<!-- BEGIN GENERATED: launcher tiles (scripts/registry/generate_registry_artifacts.py) -->

68 visible tiles from `src/config/models.yaml` (maturity: 41 ready, 6 beta, 21 experimental, 3 hidden). Regenerate with `python -m scripts.registry.generate_registry_artifacts`.

| Tile | Category | Maturity | Surfaces | Web | Help |
| --- | --- | --- | --- | --- | --- |
| Model Explorer (`model_explorer`) | tool | ready | pyqt, web | route `/tools/model-explorer` | [help](docs/help/model_explorer.md) |
| MuJoCo (`mujoco_unified`) | physics_engine | ready | pyqt, web | native-window | [help](docs/help/mujoco_unified.md) |
| Drake (`drake_golf`) | physics_engine | ready | pyqt, web | native-window | [help](docs/help/drake_golf.md) |
| Pinocchio (`pinocchio_golf`) | physics_engine | ready | pyqt, web | native-window | [help](docs/help/pinocchio_golf.md) |
| OpenSim (`opensim_golf`) | physics_engine | ready | pyqt, web | native-window | [help](docs/help/opensim_golf.md) |
| MyoSuite (`myosim_suite`) | physics_engine | ready | pyqt, web | native-window | [help](docs/help/myosim_suite.md) |
| Rate of Closure Impact Explorer (`rate_of_closure`) | simulation | ready | pyqt, web | route `/tools/impact-explorer` | [help](docs/help/rate_of_closure.md) |
| Putting Green (`putting_green`) | simulation | ready | pyqt, web | route `/tools/putting-green` | [help](docs/help/putting_green.md) |
| Simscape (`matlab_suite`) | physics_engine | ready | pyqt, web | native-window | [help](docs/help/matlab_suite.md) |
| Motion-Match Preview (`motion_target_preview`) | motion_matching | ready | pyqt, web | native-window | [help](docs/help/motion_target_preview.md) |
| Motion Capture (`motion_capture`) | motion_capture | beta | pyqt, web | route `/tools/motion-capture` | [help](docs/help/motion_capture.md) |
| Video Analyzer (`video_analyzer`) | motion_capture | experimental | pyqt, web | route `/tools/video-analyzer` | [help](docs/help/motion_capture.md) |
| Video Processor (`video_processor`) | motion_capture | ready | pyqt, web | native-window | [help](docs/help/video_processor.md) |
| Data Explorer (`data_explorer`) | tool | ready | pyqt, web | route `/tools/data-explorer` | [help](docs/help/data_explorer.md) |
| Data Processor (`data_processor`) | tool | ready | pyqt, web | native-window | [help](docs/help/data_processor.md) |
| Project Map (`project_map`) | documentation | ready | pyqt, web | unavailable | [help](docs/help/project_map.md) |
| Cross-Engine Dashboard (`cross_engine_dashboard`) | simulation | ready | pyqt, web | native-window | [help](docs/help/cross_engine_dashboard.md) |
| Exercise Dashboard (`biomech_exercise`) | biomechanics | ready | pyqt, web | native-window | [help](docs/help/biomech_exercise.md) |
| Shot Tracer (`shot_tracer`) | simulation | ready | pyqt, web | route `/ball-flight` | [help](docs/help/shot_tracer.md) |
| Pose Studio (`pose_studio`) | motion_matching | beta | pyqt, web | native-window | [help](docs/help/pose_studio.md) |
| Simulator (`golf_simulation_suite`) | simulation | ready | pyqt, web | native-window | [help](docs/help/golf_simulation_suite.md) |
| Swing Optimizer (`swing_optimizer`) | simulation | experimental | pyqt, web | native-window | [help](docs/help/simulation_controls.md) |
| Injury Risk Analysis (`injury_analysis`) | biomechanics | experimental | pyqt, web | native-window | — |
| Terrain Engine (`terrain_engine`) | simulation | ready | pyqt, web | route `/tools/terrain` | [help](docs/help/terrain_engine.md) |
| Dataset Generator (`dataset_generator`) | tool | ready | web | route `/tools/dataset` | [help](docs/help/dataset_generator.md) |
| BunkerShot3D Designer Workbench (`bunkershot3d`) | simulation | experimental | pyqt, web | native-window | [help](docs/help/simulation_controls.md) |
| Pendulum Simulator (`pendulum_simulator`) | simulation | ready | pyqt, web | native-window | [help](docs/help/pendulum_simulator.md) |
| AI Assistant (`chat_assistant`) | tool | ready | web | route `/chat` | [help](docs/help/chat_assistant.md) |
| Character Builder (`character_builder`) | tool | ready | web | route `/tools/character-builder` | [help](docs/help/character_builder.md) |
| Drake Dashboard (`drake_dashboard`) | physics_engine | experimental | pyqt, web | native-window | [help](docs/help/engine_selection.md) |
| MuJoCo Dashboard (`mujoco_dashboard`) | physics_engine | experimental | pyqt, web | native-window | [help](docs/help/engine_selection.md) |
| Pinocchio Dashboard (`pinocchio_dashboard`) | physics_engine | experimental | pyqt, web | native-window | [help](docs/help/engine_selection.md) |
| Canonical-Core Estimation (`canonical_core_estimation`) | biomechanics | experimental | pyqt, web | route `/tools/canonical-core/estimation` | — |
| Canonical-Core Comparison (`canonical_core_comparison`) | biomechanics | experimental | pyqt, web | route `/tools/canonical-core/comparison` | — |
| Analysis Tools (`analysis_tools_api`) | analysis | ready | web | route `/tools/analysis` | [help](docs/help/analysis_tools_api.md) |
| Motion Pipeline (`motion_pipeline`) | motion_matching | experimental | web | unavailable | — |
| Perturbation Analysis (`perturbation_analysis`) | analysis | experimental | web | unavailable | [help](docs/help/analysis_tools.md) |
| Force Overlays (`force_overlays`) | tool | experimental | web | unavailable | — |
| Realtime WebSocket (`realtime_ws`) | tool | experimental | web | unavailable | — |
| AI Protocol (AIP) (`aip`) | tool | experimental | web | unavailable | — |
| Actuator Controls (`actuator_controls`) | tool | experimental | web | unavailable | — |
| Unreal Integration (`unreal_integration`) | tool | experimental | web | unavailable | — |
| Robotics Module (`robotics_module`) | tool | experimental | web | unavailable | — |
| Tools Calculator Suite (`tools_calculator_hub`) | tool | beta | web | native-window | [help](docs/help/tools_calculator_hub.md) |
| P&ID Generator (`pid_generator`) | tool | experimental | web | unavailable | — |
| Simulation Backends (`simulation_backends`) | simulation | ready | pyqt, web | native-window | [help](docs/help/simulation_backends.md) |
| MuJoCo Models (`mujoco_models_shared`) | simulation | ready | pyqt, web | native-window | [help](docs/help/mujoco_models_shared.md) |
| Launch Monitor Analytics (`launch_monitor_analytics`) | tool | ready | pyqt, web | native-window | [help](docs/help/launch_monitor_analytics.md) |
| Drake Models (`drake_models_shared`) | simulation | ready | pyqt, web | native-window | [help](docs/help/drake_models_shared.md) |
| Pinocchio Models (`pinocchio_models_shared`) | simulation | ready | pyqt, web | native-window | [help](docs/help/pinocchio_models_shared.md) |
| OpenSim Models (`opensim_models_shared`) | simulation | ready | pyqt, web | native-window | [help](docs/help/opensim_models_shared.md) |
| Golf Environment (`golf_environment`) | simulation | ready | pyqt, web | native-window | [help](docs/help/golf_environment.md) |
| Swing → Flight Pipeline (`swing_flight_pipeline`) | simulation | ready | pyqt, web | native-window | [help](docs/help/swing_flight_pipeline.md) |
| Ball Flight Simulator (`ball_flight_simulator`) | simulation | ready | pyqt, web | native-window | [help](docs/help/ball_flight_simulator.md) |
| C3D Viewer (`c3d_viewer`) | motion_capture | ready | pyqt, web | native-window | [help](docs/help/c3d_viewer.md) |
| OpenPose (`openpose_analysis`) | motion_capture | ready | pyqt, web | native-window | [help](docs/help/openpose_analysis.md) |
| MediaPipe (`mediapipe_analysis`) | motion_capture | ready | pyqt, web | native-window | [help](docs/help/mediapipe_analysis.md) |
| Sidekick (`sidekick`) | tool | beta | pyqt, web | native-window | [help](docs/help/sidekick.md) |
| Training Controller (`training_controller`) | tool | beta | pyqt, web | native-window | [help](docs/help/training_controller.md) |
| Setup Wizard (`config_setup_wizard`) | tool | beta | pyqt, web | native-window | [help](docs/help/config_setup_wizard.md) |
| Pose Subscriber (demo) (`pose_subscriber_demo`) | motion_matching | experimental | pyqt, web | native-window | — |
| Library (`library_tool`) | documentation | ready | pyqt, web | native-window | [help](docs/help/library_tool.md) |
| Gait Model (`biomech_gait`) | biomechanics | ready | pyqt, web | native-window | [help](docs/help/biomech_gait.md) |
| Sit-to-Stand Model (`biomech_sit_to_stand`) | biomechanics | ready | pyqt, web | native-window | [help](docs/help/biomech_sit_to_stand.md) |
| Movement Optimizer (`movement_optimizer`) | biomechanics | ready | pyqt, web | native-window | [help](docs/help/movement_optimizer.md) |
| Pure Rigid Body (`pinn_pure_rigid`) | biomechanics | experimental | pyqt, web | native-window | — |
| PINN Hybrid (`pinn_hybrid`) | biomechanics | experimental | pyqt, web | native-window | — |
| Swing Objective Lab (`swing_objective_lab`) | simulation | ready | pyqt, web | route `/tools/swing-objective-lab` | [help](docs/help/swing_objective_lab.md) |

<!-- END GENERATED: launcher tiles -->

## Engine Support

Support level determines what is validated in continuous integration, not what
is implemented. Engines outside the supported tier work, but regressions in them
are found later.

| Tier         | Engines           | Install profile                        | Validation                         |
| ------------ | ----------------- | -------------------------------------- | ---------------------------------- |
| Supported    | MuJoCo            | `pip install -e ".[dev]"`              | Required on every pull request     |
| Extended     | Drake, Pinocchio  | `pip install -e ".[dev,all-engines]"`  | Nightly cross-engine validation    |
| Experimental | OpenSim, MyoSuite | `pip install -e ".[dev,biomechanics]"` | Best-effort, local validation only |

| Engine                                                       | Strengths                                                                   |
| ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| [MuJoCo](src/engines/physics_engines/mujoco/README.md)       | Contact-rich dynamics, ground and ball contact, motion capture workflow     |
| [Drake](src/engines/physics_engines/drake/README.md)         | Trajectory optimization, contact modeling, system analysis, URDF            |
| [Pinocchio](src/engines/physics_engines/pinocchio/README.md) | Fast rigid-body algorithms, analytical derivatives, PINK inverse kinematics |
| [OpenSim](src/engines/physics_engines/opensim/README.md)     | Biomechanics validation surface, experimental                               |
| [MyoSuite](src/engines/physics_engines/myosuite/README.md)   | Muscle modeling surface, experimental                                       |

The full contract is in [support tiers](docs/engines/support_tiers.md);
feature-level coverage is in
[engine capabilities](docs/engines/engine_capabilities.md).

## Documentation

The [documentation hub](docs/README.md) is the entry point. Frequently used
sections:

- [Project map](docs/architecture/PROJECT_MAP.md) — every feature, module, and integration.
- [User guide](docs/user_guide/README.md) — installation, running simulations, using the interface.
- [Engines](docs/engines/README.md) — engine documentation and comparison.
- [API reference](docs/api/README.md) — code interfaces and REST endpoints.
- [Adapters](docs/adapters/authoring_guide.md) — adding a physics engine.
- [Architecture decisions](docs/adr/) — durable design records.
- [Specification](SPEC.md) — the platform specification.
- [Troubleshooting](docs/troubleshooting/) — installation, configuration, and cross-engine deviations.
- [Development](docs/development/README.md) — architecture, contributing, and testing.

## Repository Layout

```text
UpstreamDrift/
├── launch_upstream_drift.py     Canonical entry point
├── src/
│   ├── launchers/               Launch applications
│   ├── engines/
│   │   ├── physics_engines/     MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite
│   │   ├── Simscape_Multibody_Models/   MATLAB and Simulink models
│   │   └── pendulum_models/     Reduced-order models
│   ├── shared/                  Code with more than one consumer
│   └── tools/                   Standalone utilities, including Pose Studio
├── rust_core/                   Rust physics kernels
├── apps/                        Web interface
├── shared/                      Model assets and vendored dependencies
├── docs/                        Documentation
└── tests/                       Test suite
```

## Contributing

Contributions are welcome. Start with the
[contributing guide](docs/development/contributing.md), then the
[development guidelines](docs/development/README.md) and the
[testing guide](docs/testing/testing-guide.md).

Before adding code, read [AGENTS.md](AGENTS.md). It maps the shared
infrastructure and gives a discovery workflow that prevents reimplementing
something the repository already provides.

Report vulnerabilities through [SECURITY.md](SECURITY.md), not through public
issues.

## Citation

```bibtex
@software{upstream_drift,
  title  = {UpstreamDrift: A Unified Platform for Biomechanical Golf Swing Analysis},
  author = {Dieter Olson},
  year   = {2026},
  url    = {https://github.com/D-sorganization/UpstreamDrift}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This project builds on
[MuJoCo](https://mujoco.org/) for physics simulation,
[Drake](https://drake.mit.edu/) for model-based design and control,
[Pinocchio](https://stack-of-tasks.github.io/pinocchio/) for rigid-body dynamics,
[MyoSuite](https://github.com/MyoHub/myosuite) for musculoskeletal models, and
[OpenSim](https://opensim.stanford.edu/) for biomechanical modeling.
