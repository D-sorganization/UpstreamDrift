# UpstreamDrift - Complete Project Map

> **Version 2.1** | Last updated: February 2026
>
> This document is the single comprehensive reference for every feature,
> module, and tool in the UpstreamDrift Golf Modeling Suite. It is designed
> to give users (and developers) full visibility into what the platform can
> do, including features that are not yet exposed as launcher tiles.

---

## Table of Contents

1. [Launcher Tiles (User-Facing)](#1-launcher-tiles-user-facing)
2. [Physics Engines](#2-physics-engines)
3. [Model Gait & Locomotion System](#3-model-gait--locomotion-system)
4. [Robotics Module](#4-robotics-module)
5. [Learning & AI](#5-learning--ai)
6. [Research Modules](#6-research-modules)
7. [Deployment & Real-Time](#7-deployment--real-time)
8. [Unreal Engine Integration](#8-unreal-engine-integration)
9. [Shared Analysis Library](#9-shared-analysis-library)
10. [Tools & Utilities](#10-tools--utilities)
11. [Visualization & Plotting](#11-visualization--plotting)
12. [API & Web UI](#12-api--web-ui)
13. [Examples & Tutorials](#13-examples--tutorials)
14. [Hidden / Unexposed Features](#14-hidden--unexposed-features-summary)
15. [Deprecated / Archived Code](#15-deprecated--archived-code)

---

## 1. Launcher Tiles (User-Facing)

These are the 11 tiles currently visible in the launcher (PyQt6 classic or
Tauri/React web UI). Defined in `src/config/launcher_manifest.json` and
`src/config/models.yaml`.

| # | Tile | Category | Status | Description |
|---|------|----------|--------|-------------|
| 1 | **Model Explorer** | Tool | Utility | Browse, select, and build URDF/MJCF models |
| 2 | **MuJoCo** | Physics Engine | GUI Ready | Unified humanoid simulation & analytics dashboard |
| 3 | **Drake** | Physics Engine | GUI Ready | Drake robotics with MeshCat visualization |
| 4 | **Pinocchio** | Physics Engine | GUI Ready | Rigid body dynamics with analytical derivatives |
| 5 | **OpenSim** | Physics Engine | Engine Ready | Musculoskeletal modeling for biomechanics |
| 6 | **MyoSuite** | Physics Engine | Engine Ready | Muscle-actuated simulation (290 muscles) |
| 7 | **Putting Green** | Physics Engine | Simulator | Ball rolling physics with terrain |
| 8 | **Matlab Models** | External | External | 2D/3D Simscape models and analysis tools |
| 9 | **Motion Capture** | Tool | Utility | C3D viewer, OpenPose, MediaPipe analysis |
| 10 | **Video Analyzer** | Tool | Utility | Video-based motion analysis with pose estimation |
| 11 | **Data Explorer** | Tool | Utility | Import, filter, and visualize simulation datasets |

### How to customize tiles

- **Classic launcher**: Click "Layout: Locked" to unlock, then "Edit Tiles"
  to add/remove tiles. Drag-and-drop to reorder.
- **Web UI**: Tiles are loaded from the manifest API (`/api/launcher/tiles`).

---

## 2. Physics Engines

Located in `src/engines/physics_engines/`. Each engine has its own Python
implementation, documentation, and models.

### MuJoCo (Recommended)
- **Path**: `src/engines/physics_engines/mujoco/`
- **Capabilities**: Rigid body, contact, tendons, actuators, mass matrix,
  Jacobian, force/torque vectors, wrench, screw theory, video & dataset export
- **Models**: 2-28 DOF humanoids, flexible shaft, MyoSuite muscle integration
- **Key features**:
  - Humanoid launcher with simulation dashboard
  - Linkage mechanisms (4-bar, Watt, Stephenson, scissor)
  - Rigid body dynamics (CRBA, RNEA, ABA algorithms)
  - Inverse kinematics with nullspace optimization
  - Playground experiments (MoCapAct demo, custom scenes)
  - Motion capture workflow (OpenPose & MediaPipe)

### Drake
- **Path**: `src/engines/physics_engines/drake/`
- **Capabilities**: Rigid body, optimization, contact, mass matrix, Jacobian
- **Key features**: Trajectory optimization, MeshCat visualization, URDF support

### Pinocchio
- **Path**: `src/engines/physics_engines/pinocchio/`
- **Capabilities**: Rigid body, inverse kinematics, mass matrix, Jacobian
- **Key features**: PINK IK, fast recursive algorithms, ZTCF/ZVCF analysis

### OpenSim
- **Path**: `src/engines/physics_engines/opensim/`
- **Capabilities**: Musculoskeletal, IK, muscle analysis, mass matrix, Jacobian
- **Bundled models**: Gait2392, Gait2354, Gait10dof18musc, Rajagopal, Hamner,
  Leg39, Leg6Dof9Musc, DynamicJumper, ToyLanding
- **Pipelines**: CMC, RRA, Static Optimization, Joint Reaction analysis

### MyoSuite
- **Path**: `src/engines/physics_engines/myosuite/`
- **Capabilities**: Musculoskeletal, muscle control, neural activation
- **Key features**: 290-muscle full body models, Hill-type muscle dynamics

### Putting Green
- **Path**: `src/engines/physics_engines/putting_green/`
- **Capabilities**: Surface modeling, ball physics, terrain
- **Key features**: Realistic putting simulation with topographic surface models

### Pendulum (Educational)
- **Path**: `src/engines/physics_engines/pendulum/`
- **Key features**: 2-DOF and 3-DOF simplified swing models for learning

---

## 3. Model Gait & Locomotion System

**Location**: `src/robotics/locomotion/`
**Status**: Fully implemented but NOT exposed in the launcher

This is the recently upgraded gait system providing bipedal locomotion control:

### Gait Types (`gait_types.py`)
- **GaitType**: Stand, Walk, Trot, Run, Crawl, Bound, Gallop
- **GaitPhase**: Double Support, Left/Right Support, Flight, Left/Right Swing
- **LegState**: Stance, Swing, Early Contact, Late Contact, Loading, Unloading
- **SupportState**: Double support (left/right/centered), single support, flight
- **GaitParameters**: Configurable step length, width, height, duration,
  double support ratio, swing height profile, CoM height, max foot velocity

### Gait State Machine (`gait_state_machine.py`)
- Finite state machine for managing gait transitions
- Events: Step Complete, Foot Contact, Foot Liftoff, Balance Lost, Stop/Start,
  Speed Change, Emergency Stop
- Deterministic transitions with callback system
- Phase progress tracking, foot trajectory interpolation

### Footstep Planner (`footstep_planner.py`)
- Footstep generation for walking and running
- Position and orientation planning with reachability validation
- Yaw extraction from orientation quaternions

### Factory Functions
- `create_walk_parameters()` - Standard walking gait
- `create_run_parameters()` - Running gait (no double support)
- `create_stand_parameters()` - Static standing

### ZMP Computer
- Zero Moment Point computation for balance analysis
- Stability margin calculation

---

## 4. Robotics Module

**Location**: `src/robotics/`
**Status**: Fully implemented but NOT exposed in the launcher

### Contact Dynamics (`contact/`)
- `ContactManager` - Multi-contact detection and management
- `FrictionCone` - Coulomb friction modeling
- Grasp matrix computation and force closure checking
- Grasp quality metrics

### Whole-Body Control (`control/`)
- `WholeBodyController` - Hierarchical task-priority QP solver
- Task types: CoM tracking, end-effector positioning, posture regulation
- `QPSolver` - Quadratic programming for constrained optimization
- Configurable task weights and priorities

### Sensing (`sensing/`)
- `ForceTorqueSensor` - 6-axis F/T sensor with configurable noise
- `IMUSensor` - IMU with gyroscope and accelerometer
- Noise models: Gaussian, Brownian, Quantization, Composite

### Planning (`planning/`)
- Collision avoidance planning
- Motion planning algorithms

### Core Protocols (`core/`)
- `RoboticsCapable` - Base protocol for robotics engines
- `ContactCapable` - Contact detection protocol
- `HumanoidCapable` - Humanoid-specific protocol
- `ManipulationCapable` - Manipulation protocol

---

## 5. Learning & AI

### Reinforcement Learning (`src/learning/rl/`)
**Status**: NOT exposed in launcher

- `RoboticsGymEnv` - Base Gymnasium-compatible environment
- `HumanoidWalkEnv` - Humanoid walking environment
- `HumanoidStandEnv` - Humanoid standing balance environment
- `ManipulationPickPlaceEnv` - Pick-and-place manipulation
- `DualArmManipulationEnv` - Bimanual coordination
- Configurable observations, actions, rewards, and tasks

### Imitation Learning (`src/learning/imitation/`)
**Status**: NOT exposed in launcher

- `BehaviorCloning` - Supervised learning from demonstrations
- `DAgger` - Dataset Aggregation with expert queries
- `GAIL` - Generative Adversarial Imitation Learning
- `DemonstrationDataset` - Dataset management for demo data

### Motion Retargeting (`src/learning/retargeting/`)
**Status**: NOT exposed in launcher

- `MotionRetargeter` - Transfer motion between different skeletons
- `SkeletonConfig` - Skeleton configuration and joint mapping

### Sim-to-Real Transfer (`src/learning/sim2real/`)
**Status**: NOT exposed in launcher

- `DomainRandomizer` - Domain randomization for sim-to-real transfer
- `SystemIdentifier` - System identification from real data
- Reality gap analysis

### AI Assistant (`src/shared/python/ai/`)
**Status**: Available via "AI Chat" button (if API key configured)

- Agent-agnostic architecture (OpenAI, Anthropic, Gemini, Ollama)
- Tool registry with scientific validation
- Workflow engine with guided multi-step workflows:
  - First analysis workflow
  - C3D import workflow
  - Cross-engine validation workflow
  - Drift control decomposition workflow
  - Inverse dynamics workflow
- Education system with glossary and expertise-level adaptation
- RAG (Retrieval-Augmented Generation) for documentation
- GUI panel integrated into the launcher

---

## 6. Research Modules

**Location**: `src/research/`
**Status**: NOT exposed in launcher

### Model Predictive Control (`mpc/`)
- `ModelPredictiveController` - Generic nonlinear MPC
- `CentroidalMPC` - Centroidal dynamics MPC for locomotion
- `WholeBodyMPC` - Whole-body MPC for manipulation
- Cost function framework

### Differentiable Physics (`differentiable/`)
- `DifferentiableEngine` - Automatic differentiation through simulation
- `ContactDifferentiableEngine` - Contact-aware differentiation
- Gradient-based trajectory optimization

### Deformable Objects (`deformable/`)
- `SoftBody` - FEM-based soft body simulation
- `Cable` - Cable and rope simulation
- `Cloth` - Cloth and fabric simulation
- `MaterialProperties` - Configurable material parameters

### Multi-Robot Coordination (`multi_robot/`)
- `MultiRobotSystem` - Multi-robot management
- `FormationController` - Formation control
- `CooperativeManipulation` - Cooperative manipulation tasks
- `TaskCoordinator` - Task allocation and coordination

---

## 7. Deployment & Real-Time

**Location**: `src/deployment/`
**Status**: NOT exposed in launcher

### Digital Twin (`digital_twin/`)
- `DigitalTwin` - Synchronized simulation mirroring real robot state
- `StateEstimator` - State estimation from sensor data
- Anomaly detection (comparing real vs simulated behavior)
- Predictive simulation for planning

### Real-Time Control (`realtime/`)
- `RealTimeController` - High-frequency control loops (up to 1kHz)
- Communication protocols: EtherCAT, ROS2, UDP
- Timing statistics and jitter monitoring
- Robot state and control command interfaces

### Safety System (`safety/`)
- `SafetyMonitor` - Real-time safety monitoring and enforcement
- `CollisionAvoidance` - Artificial potential field collision avoidance
- Human detection and tracking
- Configurable safety limits

### Teleoperation (`teleoperation/`)
- `TeleoperationInterface` - Human-in-the-loop control
- Input devices: SpaceMouse, VR controllers, haptic devices, keyboard/mouse
- Workspace mapping and scaling
- Demonstration recording

---

## 8. Unreal Engine Integration

**Location**: `src/unreal_integration/`
**Status**: NOT exposed in launcher

- **Streaming**: WebSocket and REST API streaming server for real-time data
- **Mesh Loading**: Multi-format support (GLTF/GLB/FBX/OBJ)
- **Skeleton Mapping**: Gaming skeleton to physics model mapping
- **Visualization**: Force vectors, trajectories, HUD data
- **VR Interaction**: VR controller support, gestures, locomotion modes
- **Viewer Backends**: Meshcat, Mock, and extensible backend system

---

## 9. Shared Analysis Library

**Location**: `src/shared/python/`

These modules are used by engines but also available standalone:

### Biomechanics
| Module | Description |
|--------|-------------|
| `activation_dynamics.py` | Muscle activation dynamics |
| `aerodynamics.py` | Ball aerodynamics |
| `ball_flight_physics.py` | Ball flight trajectory models |
| `biomechanics_data.py` | Biomechanics data structures |
| `flexible_shaft.py` | Flexible golf shaft dynamics |
| `grip_contact_model.py` | Two-handed grip contact model |
| `ground_reaction_forces.py` | GRF processing |
| `hill_muscle.py` | Hill-type muscle model |
| `impact_model.py` | Club-ball impact physics |
| `injury/` | Injury risk, joint stress, spinal load analysis, swing modifications |
| `kinematic_sequence.py` | Kinematic sequencing analysis |
| `manipulability.py` | Manipulability ellipsoid analysis |
| `multi_muscle.py` | Multi-muscle system modeling |
| `muscle_analysis.py` | Muscle force analysis |
| `muscle_equilibrium.py` | Muscle equilibrium solver |
| `myoconverter_integration.py` | MyoConverter model conversion |
| `myosuite_adapter.py` | MyoSuite adapter |
| `swing_plane_analysis.py` | Swing plane fitting and analysis |
| `swing_comparison.py` | Cross-swing comparison |

### Analysis
| Module | Description |
|--------|-------------|
| `analysis/angular_momentum.py` | Angular momentum computation |
| `analysis/energy_metrics.py` | Kinetic/potential energy metrics |
| `analysis/grf_metrics.py` | Ground reaction force metrics |
| `analysis/phase_detection.py` | Swing phase detection |
| `analysis/stability_metrics.py` | Balance and stability metrics |
| `analysis/swing_metrics.py` | Comprehensive swing metrics |
| `comparative_analysis.py` | Cross-engine comparison |
| `statistical_analysis.py` | Statistical analysis tools |
| `kaggle_validation.py` | Kaggle dataset validation |

### Signal Processing
| Module | Description |
|--------|-------------|
| `signal_toolkit/core.py` | Core signal operations |
| `signal_toolkit/filters.py` | Butterworth, Savitzky-Golay filters |
| `signal_toolkit/calculus.py` | Numerical differentiation/integration |
| `signal_toolkit/fitting.py` | Curve fitting |
| `signal_toolkit/noise.py` | Noise analysis |
| `signal_toolkit/io.py` | Signal I/O |

### Spatial Algebra
| Module | Description |
|--------|-------------|
| `spatial_algebra/spatial_vectors.py` | 6D spatial vectors |
| `spatial_algebra/transforms.py` | Spatial transforms |
| `spatial_algebra/inertia.py` | Spatial inertia |
| `spatial_algebra/joints.py` | Joint models |
| `spatial_algebra/pose6dof.py` | 6-DOF pose representation |

### Pose Estimation
| Module | Description |
|--------|-------------|
| `pose_estimation/openpose_estimator.py` | OpenPose integration |
| `pose_estimation/mediapipe_estimator.py` | MediaPipe integration |
| `pose_estimation/openpose_gui.py` | OpenPose GUI |
| `pose_estimation/mediapipe_gui.py` | MediaPipe GUI |
| `video_pose_pipeline.py` | End-to-end video-to-pose pipeline |

### Infrastructure
| Module | Description |
|--------|-------------|
| `engine_manager.py` | Engine availability and lifecycle |
| `cross_engine_validator.py` | Cross-engine validation framework |
| `configuration_manager.py` | App-wide configuration |
| `checkpoint.py` | Simulation checkpointing |
| `provenance.py` | Data provenance tracking |
| `reproducibility.py` | Reproducibility helpers |
| `terrain.py` / `terrain_engine.py` | Terrain modeling |
| `equipment.py` | Golf equipment database |
| `handedness_support.py` | Left/right-handed support |

---

## 10. Tools & Utilities

### Model Explorer (Launcher Tile)
- **Path**: `src/tools/model_explorer/`
- Browse humanoid, pendulum, and robotic models
- URDF/MJCF code editor with syntax highlighting
- Frankenstein editor for mixing model parts
- Joint manipulator and chain manipulation
- Mesh browser and MuJoCo viewer
- Segment panel and end-effector manager

### Humanoid Character Builder
- **Path**: `src/tools/humanoid_character_builder/`
- **Status**: NOT exposed as a launcher tile
- Anthropometry-based body parameter generation
- Segment definitions with standard body proportions
- URDF generation from body parameters
- Mesh generation for visualization
- Inertia calculation (primitive shapes and mesh-based)
- Presets for common body types

### Model Generation System
- **Path**: `src/tools/model_generation/`
- **Status**: NOT exposed as a launcher tile
- Parametric URDF builder
- Manual URDF builder
- MJCF to URDF converter
- Simscape to URDF converter
- Model library with caching and repository
- Frankenstein editor for hybrid models
- CLI interface and REST API
- Pendulum putter model generator

### Video Analyzer (Launcher Tile)
- **Path**: `src/tools/video_analyzer/`
- Video processing pipeline
- Pose estimation integration
- Motion tracking and frame analysis

### Data Explorer (Launcher Tile)
- **Path**: `src/tools/data_explorer/`
- Data import and filtering
- Visualization and export

### Shot Tracer
- **Path**: `src/launchers/shot_tracer.py`
- **Status**: NOT exposed as a launcher tile
- Multi-model ball flight visualization
- Compares Waterloo/Penner, MacDonald-Hanzely, and Nathan models
- 3D trajectory rendering with PyQtGraph

### Scientific Auditor
- **Path**: `src/shared/python/tools/scientific_auditor.py`
- **Status**: NOT exposed in UI
- Audits simulation results for scientific validity

---

## 11. Visualization & Plotting

**Location**: `src/shared/python/plotting/`

### Renderers
| Renderer | Description |
|----------|-------------|
| `renderers/kinematics.py` | Joint angles, velocities, accelerations |
| `renderers/kinetics.py` | Forces, torques, moments |
| `renderers/energy.py` | Kinetic, potential, total energy |
| `renderers/stability.py` | Balance and stability plots |
| `renderers/coordination.py` | Inter-joint coordination |
| `renderers/club.py` | Club path and face angle |
| `renderers/comparison.py` | Cross-engine comparison plots |
| `renderers/dashboard.py` | Summary dashboard |
| `renderers/signal.py` | Signal analysis plots |
| `renderers/vectors.py` | Force/torque vector overlays |

### Additional Visualization
| Module | Description |
|--------|-------------|
| `ellipsoid_visualization.py` | Manipulability ellipsoids |
| `swing_plane_visualization.py` | 3D swing plane rendering |
| `comparative_plotting.py` | Multi-engine comparison plots |
| `plotting/animation.py` | Animated visualizations |
| `plotting/export.py` | Plot export to PNG/SVG/PDF |
| `theme/` | Unified theme system (colors, typography, matplotlib) |

---

## 12. API & Web UI

### REST API (`src/api/`)
| Route | Description |
|-------|-------------|
| `/api/launcher/manifest` | Full launcher manifest |
| `/api/launcher/tiles` | All tiles in display order |
| `/api/launcher/engines` | Physics engine tiles only |
| `/api/launcher/tools` | Tool tiles only |
| `/api/launcher/engines/capabilities` | Engine capability profiles |
| `/api/simulation/*` | Simulation control (start, stop, step) |
| `/api/simulation/ws` | WebSocket for real-time simulation |
| `/api/analysis/*` | Analysis endpoints |
| `/api/export/*` | Data export endpoints |
| `/api/video/*` | Video processing endpoints |
| `/api/terrain/*` | Terrain management |
| `/api/dataset/*` | Dataset management |
| `/api/auth/*` | Authentication (bcrypt, JWT) |

### Web UI (`ui/`)
- React + Tauri desktop application
- LauncherDashboard with responsive tile grid
- 3D Scene visualization (Three.js)
- Live plotting with real-time updates
- Engine selector, parameter panel, simulation controls
- Connection status, diagnostics panel, toast notifications

---

## 13. Examples & Tutorials

**Location**: `examples/`

| Example | Description |
|---------|-------------|
| `01_basic_simulation.py` | Getting started with basic simulation |
| `02_parameter_sweeps.py` | Running parameter sweep studies |
| `03_injury_risk_tutorial.py` | Injury risk assessment workflow |
| `motion_training_demo.py` | Motion training demonstration |

**Documentation tutorials**: `docs/tutorials/`

---

## 14. Hidden / Unexposed Features Summary

The following features are fully implemented but have **no launcher tile or
menu entry** for user access. These are candidates for exposure:

| Feature | Location | Maturity | Recommendation |
|---------|----------|----------|----------------|
| **Gait/Locomotion System** | `src/robotics/locomotion/` | Production | Expose as tile or submenu |
| **Whole-Body Controller** | `src/robotics/control/` | Production | Expose under robotics section |
| **Contact Dynamics** | `src/robotics/contact/` | Production | Expose under robotics section |
| **F/T & IMU Sensors** | `src/robotics/sensing/` | Production | Expose under robotics section |
| **RL Environments** | `src/learning/rl/` | Production | Expose as "Training" tile |
| **Imitation Learning** | `src/learning/imitation/` | Production | Expose as "Training" tile |
| **Motion Retargeting** | `src/learning/retargeting/` | Production | Expose under Motion Capture |
| **Sim-to-Real Transfer** | `src/learning/sim2real/` | Production | Expose as "Training" tile |
| **MPC Framework** | `src/research/mpc/` | Research | Expose as "Research" section |
| **Differentiable Physics** | `src/research/differentiable/` | Research | Expose as "Research" section |
| **Deformable Objects** | `src/research/deformable/` | Research | Expose as "Research" section |
| **Multi-Robot System** | `src/research/multi_robot/` | Research | Expose as "Research" section |
| **Digital Twin** | `src/deployment/digital_twin/` | Production | Expose as tile |
| **Real-Time Control** | `src/deployment/realtime/` | Production | Expose under deployment |
| **Safety System** | `src/deployment/safety/` | Production | Expose under deployment |
| **Teleoperation** | `src/deployment/teleoperation/` | Production | Expose as tile |
| **Unreal Integration** | `src/unreal_integration/` | Production | Expose as tile |
| **Shot Tracer** | `src/launchers/shot_tracer.py` | Production | Expose as tile |
| **Humanoid Character Builder** | `src/tools/humanoid_character_builder/` | Production | Expose as tile |
| **Model Generation CLI/API** | `src/tools/model_generation/` | Production | Integrate into Model Explorer |
| **Scientific Auditor** | `src/shared/python/tools/` | Utility | Expose in Tools menu |
| **Swing Optimizer** | `src/shared/python/optimization/` | Production | Expose in analysis |
| **Signal Toolkit** | `src/shared/python/signal_toolkit/` | Production | Expose in analysis |
| **Pose Editor** | `src/shared/python/pose_editor/` | Production | Expose in Tools |
| **Equipment Database** | `src/shared/python/equipment.py` | Production | Expose in Tools |

---

## 15. Deprecated / Archived Code

| Item | Location | Status |
|------|----------|--------|
| Pre-refactor launcher | `src/launchers/_archive/` | Archived (keep for reference) |
| Archive directory | `archive/` | Contains README only |
| Pre-Jan-2026 assessments | `docs/archive/assessments_jan2026/` | Archived |
| Old phase plans | `docs/archive/phase_plans/` | Archived |
| Legacy pendulum archive | `src/engines/pendulum_models/archive/` | Security warning (eval usage) |

---

## Quick Reference: Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F1` | Open User Manual |
| `Ctrl+?` | Keyboard shortcuts overlay |
| `Ctrl+,` | Preferences |
| `Ctrl+F` | Search models |
| `Ctrl+Q` | Quit |
| `Ctrl+`` ` | Toggle process output console |
| `Space` | Play/pause simulation |
| `R` | Reset simulation |
| `Ctrl+S` | Save state |
| `Ctrl+E` | Export data |

---

## Quick Reference: CLI Launch Options

```bash
# Default web UI
python3 launch_golf_suite.py

# Classic PyQt6 desktop launcher
python3 launch_golf_suite.py --classic

# API server only (for development)
python3 launch_golf_suite.py --api-only

# Direct engine launch
python3 launch_golf_suite.py --engine mujoco
python3 launch_golf_suite.py --engine drake
python3 launch_golf_suite.py --engine pinocchio
python3 launch_golf_suite.py --engine pendulum

# Custom port
python3 launch_golf_suite.py --port 9000
```
