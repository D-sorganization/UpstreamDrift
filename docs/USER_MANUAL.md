# UpstreamDrift User Manual

**Version 2.1** | **Biomechanical Golf Swing Analysis Platform**

---

> **Document Status**: Official User Reference
> **Last Updated**: February 2026
> **Applicable Version**: UpstreamDrift v2.1.x

---

## Table of Contents

1. [Introduction and Overview](#1-introduction-and-overview)
2. [Installation Guide](#2-installation-guide)
3. [Getting Started](#3-getting-started)
4. [Physics Engines Guide](#4-physics-engines-guide)
5. [Core Features](#5-core-features)
6. [API Reference](#6-api-reference)
7. [Visualization and Analysis](#7-visualization-and-analysis)
8. [Advanced Features](#8-advanced-features)
9. [Configuration](#9-configuration)
10. [Troubleshooting](#10-troubleshooting)
11. [Appendices](#11-appendices)

---

## 1. Introduction and Overview

### 1.1 What is UpstreamDrift?

**UpstreamDrift** (formerly Golf Modeling Suite) is a unified platform for biomechanical golf swing analysis that integrates multiple physics engines to provide comprehensive simulation, analysis, and visualization capabilities. The platform enables researchers, coaches, and engineers to:

- Simulate golf swings using research-grade physics engines
- Analyze biomechanical forces, torques, and muscle activations
- Compare results across different physics modeling approaches
- Process motion capture data for swing analysis
- Optimize swing trajectories for performance or injury prevention

### 1.2 Key Features and Capabilities

| Category | Features |
|----------|----------|
| **Physics Simulation** | 5 integrated physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) |
| **Model Complexity** | From 2-DOF educational pendulums to 290-muscle musculoskeletal models |
| **Biomechanics** | Forward/inverse dynamics, muscle activation, joint forces |
| **Motion Capture** | CSV, JSON, C3D support with automatic retargeting |
| **Visualization** | Real-time 3D rendering, force vectors, phase diagrams |
| **API** | REST API with JWT authentication, WebSocket support |
| **Export** | CSV, JSON data export for external analysis |

### 1.3 System Requirements

#### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10/11, macOS 12+, Ubuntu 20.04+ |
| **Python** | 3.11 or higher (3.13 recommended) |
| **RAM** | 8 GB |
| **Storage** | 5 GB free space |
| **Graphics** | OpenGL 3.3+ compatible GPU |

#### Recommended Requirements

| Component | Recommendation |
|-----------|----------------|
| **RAM** | 16 GB or more |
| **Storage** | SSD with 10 GB+ free space |
| **Graphics** | Dedicated GPU with 4 GB+ VRAM |
| **Display** | 1920x1080 or higher resolution |

#### Optional Components

| Component | Purpose |
|-----------|---------|
| **MATLAB R2023a+** | For Simscape Multibody models |
| **Git LFS** | For large model files |
| **Docker** | For containerized deployment |
| **CUDA 11+** | For GPU-accelerated simulation (optional) |

---

## 2. Installation Guide

### 2.1 Prerequisites

Before installing UpstreamDrift, ensure you have:

1. **Python 3.11+** installed and accessible from command line
2. **Git** with Git LFS extension installed
3. **Conda** (recommended) or **pip** package manager

#### Verify Prerequisites

```bash
# Check Python version
python --version
# Expected: Python 3.11.x or higher

# Check Git
git --version
git lfs version

# Check Conda (if using)
conda --version
```

### 2.2 Installation via Conda (Recommended)

Conda installation handles binary dependencies (MuJoCo, PyQt6) automatically:

```bash
# Clone the repository
git clone https://github.com/dieterolson/UpstreamDrift.git
cd UpstreamDrift

# Initialize Git LFS and pull large files
git lfs install
git lfs pull

# Create and activate conda environment
conda env create -f environment.yml
conda activate golf-suite

# Verify installation
python scripts/verify_installation.py
```

### 2.3 Installation via Pip

For pip-based installation:

```bash
# Clone the repository
git clone https://github.com/dieterolson/UpstreamDrift.git
cd UpstreamDrift
git lfs install && git lfs pull

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install with all dependencies
pip install -e ".[dev,engines]"

# Verify installation
python scripts/verify_installation.py
```

### 2.4 Light Installation (UI Development)

For UI development without heavy physics engines:

```bash
pip install -e .
export GOLF_USE_MOCK_ENGINE=1  # or set in Windows
python launch_golf_suite.py
```

### 2.5 Verifying Installation

Run the verification script to check all components:

```bash
python scripts/verify_installation.py
```

Expected output:

```
UpstreamDrift Installation Verification
========================================
[OK] Python 3.13.1
[OK] MuJoCo 3.2.0
[OK] PyQt6 6.6.1
[OK] NumPy 1.26.4
[OK] SciPy 1.12.0
[WARNING] Drake not installed (optional)
[WARNING] Pinocchio not installed (optional)

Core: 5/5 passed
Optional: 2/4 available
```

### 2.6 Installing Optional Physics Engines

#### Drake Installation

```bash
# Via conda (recommended)
conda install -c conda-forge drake

# Via pip (limited platforms)
pip install drake
```

#### Pinocchio Installation

```bash
# Via conda (required)
conda install -c conda-forge pinocchio
```

#### MyoSuite Installation

```bash
pip install -U myosuite
```

#### OpenSim Installation

```bash
# Via conda
conda install -c opensim-org opensim

# Or download from https://opensim.stanford.edu/
```

---

## 3. Getting Started

### 3.1 Launching the Application

UpstreamDrift provides multiple launch modes:

```bash
# Web UI (recommended) - opens browser automatically
python launch_golf_suite.py

# Classic PyQt6 desktop launcher
python launch_golf_suite.py --classic

# API server only (development)
python launch_golf_suite.py --api-only

# Specific engine directly
python launch_golf_suite.py --engine mujoco
```

#### Launch Options

| Option | Description |
|--------|-------------|
| `--classic` | Use PyQt6 desktop launcher |
| `--api-only` | Start API server without UI |
| `--engine ENGINE` | Launch specific engine (mujoco, drake, pinocchio, etc.) |
| `--port PORT` | Specify server port (default: 8000) |
| `--no-browser` | Don't auto-open browser |

### 3.2 Understanding the UI Layout

#### Web UI Layout

```
+----------------------------------------------------------+
|  Header: Navigation | Engine Status | Settings           |
+----------------------------------------------------------+
|  Sidebar       |  Main Content Area                      |
|  - Engine      |  +----------------------------------+   |
|  - Models      |  |  3D Visualization Panel          |   |
|  - Analysis    |  |                                  |   |
|  - Export      |  +----------------------------------+   |
|                |  |  Controls | Plots | Data         |   |
|                |  +----------------------------------+   |
+----------------------------------------------------------+
|  Status Bar: Engine Status | Time | FPS                  |
+----------------------------------------------------------+
```

#### Classic PyQt6 Layout

The classic launcher features a tabbed interface:

1. **Controls Tab**: Model selection, simulation controls, torque sliders
2. **Visualization Tab**: Camera views, force/torque vector rendering
3. **Analysis Tab**: Real-time metrics, data export options
4. **Plotting Tab**: Time series, phase diagrams, 3D trajectories

### 3.3 Your First Simulation

#### Step 1: Select a Physics Engine

From the launcher, select your preferred engine:
- **MuJoCo** (recommended for beginners)
- **Drake** (for trajectory optimization)
- **Pinocchio** (for fast algorithms)

#### Step 2: Load a Model

```python
# Using the GUI: Click "Load Model" and select from library

# Using Python API:
from shared.python.engine_manager import EngineManager, EngineType

manager = EngineManager()
engine = manager.get_engine(EngineType.MUJOCO)
engine.load_model("double_pendulum")
```

#### Step 3: Configure Parameters

Set initial conditions:
- Joint angles (q)
- Joint velocities (v)
- Applied torques (tau)

#### Step 4: Run Simulation

```python
# GUI: Click "Start Simulation"

# Python API:
dt = 0.002  # 2ms timestep
for i in range(1000):
    engine.step(dt)
    q, v = engine.get_state()
    print(f"Time: {i*dt:.3f}s, Position: {q}")
```

#### Step 5: Analyze Results

Access built-in analysis tools:
- View force/torque plots
- Export data to CSV/JSON
- Generate phase diagrams

---

## 4. Physics Engines Guide

### 4.1 Engine Selection Decision Tree

```
START: What is your primary goal?
|
+-> Muscle-driven biomechanics?
|   YES --> MuJoCo + MyoSuite (ONLY OPTION)
|
+-> Trajectory optimization?
|   YES --> Drake (best optimization tools)
|
+-> Fast prototyping?
|   YES --> MuJoCo (easiest setup)
|
+-> Research reproducibility?
|   YES --> Pinocchio (lightweight, algorithmic)
|
+-> Contact-heavy simulation?
|   YES --> MuJoCo (most mature contact model)
|
+-> Musculoskeletal validation?
    YES --> OpenSim (gold standard in biomechanics)
```

### 4.2 MuJoCo (Primary Engine)

**Status**: Fully Implemented

MuJoCo is the recommended primary engine for most use cases.

#### Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Forward Dynamics | Complete | RK4 integration |
| Inverse Dynamics | Complete | RNEA algorithm |
| Contact Dynamics | Complete | Soft contact model |
| Flexible Shafts | Complete | Multi-segment beam |
| Musculoskeletal | Complete | Via MyoSuite |
| Motion Capture | Complete | OpenPose, MediaPipe |

#### When to Use MuJoCo

- General biomechanical simulation
- Contact-intensive scenarios (ball impact, ground reaction)
- Musculoskeletal analysis (with MyoSuite)
- Real-time visualization requirements
- Easiest installation and setup

#### Model Library

| Model | DOF | Description |
|-------|-----|-------------|
| Double Pendulum | 2 | Educational: shoulder + wrist |
| Triple Pendulum | 3 | Educational: shoulder + elbow + wrist |
| Upper Body | 10 | Realistic: pelvis + spine + arms + club |
| Full Body | 15 | Complete: legs + pelvis + torso + arms + club |
| Advanced | 28 | Research: scapulae + 3-DOF shoulders + flexible shaft |

#### Quick Start with MuJoCo

```python
from engines.physics_engines.mujoco.python.mujoco_physics_engine import (
    MuJoCoPhysicsEngine
)

engine = MuJoCoPhysicsEngine()
engine.load_model("upper_body_golf_swing")

# Run simulation
for _ in range(1000):
    engine.step(dt=0.002)
    q, v = engine.get_state()

# Get clubhead speed
clubhead_velocity = engine.get_body_velocity("club_head")
print(f"Clubhead speed: {np.linalg.norm(clubhead_velocity):.2f} m/s")
```

### 4.3 Drake (Trajectory Optimization)

**Status**: Fully Implemented

Drake excels at trajectory optimization and model-based design.

#### Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Forward Dynamics | Complete | RK3 integration |
| Inverse Dynamics | Complete | High precision |
| Trajectory Optimization | Complete | TrajOpt solver |
| Contact Planning | Complete | Advanced constraint handling |
| URDF Support | Complete | Full parsing |

#### When to Use Drake

- Optimal swing trajectory generation
- Multi-objective optimization (speed, accuracy, efficiency)
- Contact-rich motion planning
- System identification and control design

#### Quick Start with Drake

```python
from engines.physics_engines.drake.python.src.drake_golf_model import (
    DrakeGolfModel
)

model = DrakeGolfModel()
model.load_urdf("models/golf_swing.urdf")

# Optimize for maximum clubhead speed
from engines.physics_engines.drake.python.src.trajectory_optimization import (
    SwingOptimizer
)

optimizer = SwingOptimizer(model)
optimal_trajectory = optimizer.optimize(
    objective="clubhead_speed",
    constraints={"joint_limits": True, "energy_limit": 500.0}
)
```

### 4.4 Pinocchio (Fast Rigid Body Dynamics)

**Status**: Fully Implemented

Pinocchio provides high-performance rigid body algorithms.

#### Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Forward Dynamics | Complete | ABA algorithm |
| Inverse Dynamics | Complete | RNEA algorithm |
| Jacobians | Complete | Analytical derivatives |
| ZTCF/ZVCF | Complete | Counterfactual analysis |
| Drift-Control | Complete | State-of-the-art decomposition |

#### When to Use Pinocchio

- Counterfactual analysis (ZTCF/ZVCF)
- Drift-control decomposition
- Lightweight simulations
- Research requiring algorithmic differentiation

#### Quick Start with Pinocchio

```python
from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine
)

engine = PinocchioPhysicsEngine()
engine.load_urdf("models/golf_swing.urdf")

# Compute drift-control decomposition
q, v = engine.get_state()
drift_accel = engine.compute_drift_acceleration()
control_effect = engine.compute_control_acceleration(tau)
```

### 4.5 OpenSim (Musculoskeletal Modeling)

**Status**: Partial Implementation

OpenSim provides gold-standard musculoskeletal modeling.

#### Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Model Loading | Complete | .osim files |
| Forward Dynamics | Complete | |
| Inverse Dynamics | Complete | |
| Muscle Analysis | Complete | Hill-type models |
| Full GUI | Pending | Basic interface available |

#### Implementation Notes

OpenSim integration is partially complete. Core dynamics functionality works, but the full GUI and some advanced features are under development.

#### When to Use OpenSim

- Validation against published biomechanics research
- Muscle force estimation from motion data
- Clinical or rehabilitation applications

#### Quick Start with OpenSim

```python
from engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine
)

engine = OpenSimPhysicsEngine()
engine.load_from_path("shared/models/opensim/opensim-models/Models/Arm26/arm26.osim")

# Set muscle activations
activations = np.zeros(26)
activations[0] = 0.5  # Biceps at 50%

engine.set_control(activations)
engine.step(dt=0.001)

# Get muscle forces
analyzer = engine.get_muscle_analyzer()
forces = analyzer.get_muscle_forces()
```

### 4.6 MyoSuite (290-Muscle Models)

**Status**: Partial Implementation

MyoSuite provides realistic muscle-driven simulation via MuJoCo.

#### Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| Environment Loading | Complete | Gym-compatible |
| Muscle Control | Complete | 0-1 activation |
| Hill-Type Muscles | Complete | Force-length-velocity |
| RL Training | Complete | Stable-Baselines3 |
| Full GUI | Pending | Basic interface available |

#### Implementation Notes

MyoSuite is MuJoCo-exclusive. The 290-muscle model provides the most physiologically realistic simulation but requires significant computational resources.

#### Available Models

| Environment ID | Description | Muscles |
|----------------|-------------|---------|
| myoElbowPose1D6MRandom-v0 | Elbow control | 6 |
| myoHandPose100Random-v0 | Hand control | 39 |
| myoArmPose-v0 | Full arm | Multiple |
| myoLegWalk-v0 | Walking | Lower body |

#### Quick Start with MyoSuite

```python
from engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine
)

engine = MyoSuitePhysicsEngine()
engine.load_from_path("myoElbowPose1D6MRandom-v0")

# Set muscle activations
activations = {
    "BIClong": 0.5,
    "TRIlong": 0.2,
}
engine.set_muscle_activations(activations)
engine.step(dt=0.002)

# Get muscle forces
analyzer = engine.get_muscle_analyzer()
forces = analyzer.get_muscle_forces()
```

### 4.7 Engine Capability Matrix

| Feature | MuJoCo | Drake | Pinocchio | OpenSim | MyoSuite |
|---------|--------|-------|-----------|---------|----------|
| Forward Dynamics | Full | Full | Full | Stub | Stub |
| Inverse Dynamics | Full | Full | Full | Stub | Stub |
| URDF Import | Full | Full | Full | N/A | N/A |
| Jacobians | Full | Full | Full | Partial | N/A |
| Contact Dynamics | Full | Full | Partial | N/A | N/A |
| Muscle Models | Via MyoSuite | N/A | N/A | Full | Full |
| Trajectory Opt | Basic | Full | Partial | N/A | N/A |
| ZTCF/ZVCF | N/A | N/A | Full | N/A | N/A |

**Legend**:
- **Full**: Fully implemented and tested
- **Partial**: Implemented with limitations
- **Stub**: Placeholder only
- **N/A**: Not applicable

---

## 5. Core Features

### 5.1 Forward and Inverse Dynamics

#### Forward Dynamics

Compute accelerations given forces/torques:

$$\ddot{q} = M(q)^{-1}(\tau - C(q, \dot{q}) - G(q))$$

```python
# Forward dynamics simulation
engine.set_control(tau)
engine.step(dt)
q, v = engine.get_state()
```

#### Inverse Dynamics

Compute required torques for a desired motion:

$$\tau = M(q)\ddot{q} + C(q, \dot{q}) + G(q)$$

```python
# Inverse dynamics computation
tau = engine.compute_inverse_dynamics(q, v, a)
```

### 5.2 Motion Capture Integration

UpstreamDrift supports multiple motion capture formats and pose estimation systems.

#### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | .csv | Custom column mapping |
| JSON | .json | Hierarchical joint data |
| C3D | .c3d | Standard biomechanics format |

#### Pose Estimation Systems

| System | Status | Notes |
|--------|--------|-------|
| OpenPose | Supported | 25-body keypoints |
| MediaPipe | Supported | 33 landmarks |
| MoveNet | Supported | Lightning/Thunder models |

#### Loading Motion Capture Data

```python
from shared.python.motion_capture import MotionCaptureLoader, MotionRetargeting

# Load motion capture
mocap_seq = MotionCaptureLoader.load_csv('player_swing.csv')

# Retarget to model
retargeting = MotionRetargeting(model, data, marker_set)
times, joint_traj, success = retargeting.retarget_sequence(mocap_seq)
```

### 5.3 Flexible Shaft Modeling

Multi-segment beam model for realistic club dynamics:

```python
# Configure flexible shaft
engine.load_model("advanced_biomechanical")

# Shaft parameters
shaft_config = {
    "segments": 5,
    "stiffness_gradient": [100, 150, 200, 250, 300],  # N/m
    "damping": 0.1
}
engine.configure_flexible_shaft(shaft_config)
```

### 5.4 Ground Reaction Forces

Analyze forces between feet and ground during the swing:

```python
# Get ground reaction forces
grf_left = engine.get_contact_force("left_foot", "ground")
grf_right = engine.get_contact_force("right_foot", "ground")

# Compute center of pressure
cop = engine.compute_center_of_pressure()
```

### 5.5 Ball Impact Analysis

Analyze the moment of club-ball contact:

```python
# Setup impact analysis
impact_analyzer = engine.get_impact_analyzer()

# Run simulation through impact
for _ in range(frames):
    engine.step(dt)
    if impact_analyzer.detect_impact():
        impact_data = impact_analyzer.get_impact_data()
        print(f"Impact velocity: {impact_data.clubhead_velocity} m/s")
        print(f"Face angle: {impact_data.face_angle} deg")
        print(f"Attack angle: {impact_data.attack_angle} deg")
        break
```

### 5.6 Kinematic Sequence Analysis

Analyze the proximal-to-distal sequencing of the golf swing:

```python
from shared.python.analysis import KinematicSequenceAnalyzer

analyzer = KinematicSequenceAnalyzer(engine)

# Compute sequence
sequence = analyzer.compute_kinematic_sequence()

# Get peak angular velocities and timing
print(f"Pelvis peak: {sequence.pelvis_peak_velocity:.1f} deg/s at {sequence.pelvis_peak_time:.3f}s")
print(f"Torso peak: {sequence.torso_peak_velocity:.1f} deg/s at {sequence.torso_peak_time:.3f}s")
print(f"Arm peak: {sequence.arm_peak_velocity:.1f} deg/s at {sequence.arm_peak_time:.3f}s")
print(f"Club peak: {sequence.club_peak_velocity:.1f} deg/s at {sequence.club_peak_time:.3f}s")
```

---

## 6. API Reference

### 6.1 API Architecture

UpstreamDrift provides a local-first REST API built on FastAPI:

```
http://localhost:8000/          # API root
http://localhost:8000/docs      # OpenAPI documentation
http://localhost:8000/redoc     # ReDoc documentation
http://localhost:8000/health    # Health check
```

### 6.2 Authentication

Authentication is optional in local mode but required for cloud deployment.

#### Local Mode (Default)

No authentication required:

```bash
curl http://localhost:8000/health
```

#### Cloud Mode

JWT-based authentication:

```bash
# Login to obtain token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Response
{
  "access_token": "eyJ0eXAi...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Use token in requests
curl http://localhost:8000/engines \
  -H "Authorization: Bearer eyJ0eXAi..."
```

### 6.3 Core Endpoints

#### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "engines": {
    "mujoco": "available",
    "drake": "not_installed",
    "pinocchio": "available"
  }
}
```

#### List Engines

```http
GET /engines
```

Response:
```json
{
  "engines": [
    {
      "type": "mujoco",
      "status": "available",
      "version": "3.2.0",
      "models_loaded": 0
    },
    {
      "type": "drake",
      "status": "not_installed",
      "version": null
    }
  ]
}
```

#### Load Engine

```http
POST /engines/{type}/load
```

Request:
```json
{
  "model_path": "models/golf_swing.urdf",
  "config": {
    "timestep": 0.002,
    "gravity": [0, 0, -9.81]
  }
}
```

### 6.4 Simulation Endpoints

#### Run Simulation

```http
POST /simulate
```

Request:
```json
{
  "engine_type": "mujoco",
  "duration": 2.0,
  "timestep": 0.002,
  "initial_state": {
    "q": [0.0, 0.1, 0.2],
    "v": [0.0, 0.0, 0.0]
  },
  "control_sequence": [
    {"time": 0.0, "tau": [0.0, 0.0, 0.0]},
    {"time": 1.0, "tau": [10.0, 5.0, 2.0]}
  ]
}
```

Response:
```json
{
  "task_id": "sim_abc123",
  "status": "completed",
  "results": {
    "time": [0.0, 0.002, 0.004, ...],
    "positions": [[0.0, 0.1, 0.2], ...],
    "velocities": [[0.0, 0.0, 0.0], ...],
    "clubhead_speed_max": 45.2
  }
}
```

#### Async Simulation

```http
POST /simulate/async
```

Returns immediately with task ID:
```json
{
  "task_id": "sim_def456",
  "status": "running"
}
```

Check status:
```http
GET /simulate/status/sim_def456
```

### 6.5 Analysis Endpoints

#### Biomechanical Analysis

```http
POST /analyze/biomechanics
```

Request:
```json
{
  "trajectory": {
    "time": [0.0, 0.1, 0.2],
    "positions": [[...], [...], [...]],
    "velocities": [[...], [...], [...]]
  },
  "analysis_type": ["inverse_dynamics", "kinematic_sequence", "energy"]
}
```

#### Video Analysis

```http
POST /analyze/video
Content-Type: multipart/form-data

file: <video_file>
pose_estimator: "mediapipe"
```

### 6.6 Export Endpoints

```http
GET /export/{task_id}?format=csv
```

Supported formats: `csv`, `json`

### 6.7 Example API Requests

#### Python requests

```python
import requests

# Run a simulation
response = requests.post(
    "http://localhost:8000/simulate",
    json={
        "engine_type": "mujoco",
        "duration": 1.0,
        "timestep": 0.002
    }
)
result = response.json()
print(f"Max clubhead speed: {result['results']['clubhead_speed_max']} m/s")
```

#### cURL

```bash
# Health check
curl http://localhost:8000/health

# Run simulation
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"engine_type": "mujoco", "duration": 1.0}'

# Export results
curl "http://localhost:8000/export/sim_abc123?format=csv" > results.csv
```

---

## 7. Visualization and Analysis

### 7.1 3D Scene Controls

#### Camera Controls

| Control | Action |
|---------|--------|
| Left-click + drag | Rotate view |
| Right-click + drag | Pan view |
| Scroll wheel | Zoom in/out |
| Middle-click | Reset view |

#### Preset Views

| View | Shortcut | Description |
|------|----------|-------------|
| Side | `1` | View from golfer's right side |
| Front | `2` | Face-on view |
| Top | `3` | Bird's eye view |
| Down-the-line | `4` | Behind golfer, toward target |
| Follow | `5` | Camera follows clubhead |

### 7.2 Energy Analysis Plots

The energy analysis panel shows:

1. **Kinetic Energy**: $$KE = \frac{1}{2}\dot{q}^T M(q) \dot{q}$$
2. **Potential Energy**: $$PE = -q^T G(q)$$
3. **Total Energy**: $$E_{total} = KE + PE$$

```python
# Generate energy plot
from shared.python.plotting import EnergyPlotter

plotter = EnergyPlotter(simulation_data)
fig = plotter.plot_energy_components()
fig.savefig("energy_analysis.png")
```

### 7.3 Phase Diagrams

Phase diagrams show position vs. velocity for each joint:

```python
from shared.python.plotting import PhaseDiagramPlotter

plotter = PhaseDiagramPlotter(simulation_data)
fig = plotter.plot_phase_space(joint_indices=[0, 1, 2])
```

### 7.4 Jacobian Visualization

Visualize the manipulability ellipsoid at the end-effector:

```python
# Compute manipulability
J = engine.compute_jacobian()
manipulability = np.sqrt(np.linalg.det(J @ J.T))

# Visualize ellipsoid
from shared.python.visualization import EllipsoidVisualizer

viz = EllipsoidVisualizer(engine)
viz.draw_manipulability_ellipsoid()
```

### 7.5 Force/Torque Vector Visualization

Enable force visualization in the 3D view:

```python
# GUI: Toggle in Visualization tab

# API:
engine.set_visualization_options({
    "show_forces": True,
    "force_scale": 0.01,  # 1cm per Newton
    "show_torques": True,
    "torque_scale": 0.1   # 10cm per N-m
})
```

---

## 8. Advanced Features

### 8.1 Cross-Engine Validation

Compare results across different physics engines:

```python
from shared.python.cross_engine_validator import CrossEngineValidator

validator = CrossEngineValidator()

# Run same simulation on multiple engines
results_mujoco = run_simulation(EngineType.MUJOCO, params)
results_drake = run_simulation(EngineType.DRAKE, params)

# Compare results
comparison = validator.compare_states(
    "MuJoCo", results_mujoco,
    "Drake", results_drake,
    metric="torque"
)

if comparison.passed:
    print(f"Engines agree within tolerance")
else:
    print(f"Max deviation: {comparison.max_deviation:.2e}")
```

#### Tolerance Guidelines

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| Positions | +/-1e-6 m | 1 micrometer |
| Velocities | +/-1e-5 m/s | 10 micrometers/sec |
| Accelerations | +/-1e-4 m/s^2 | 0.1 mm/s^2 |
| Torques (absolute) | +/-1e-3 N-m | 1 millinewton-meter |
| Torques (RMS) | <10% | For large magnitudes |

### 8.2 Multi-Engine Comparison

Run simulations in parallel across multiple engines:

```python
from shared.python.multi_engine import MultiEngineRunner

runner = MultiEngineRunner()
runner.add_engine(EngineType.MUJOCO)
runner.add_engine(EngineType.DRAKE)
runner.add_engine(EngineType.PINOCCHIO)

# Run identical simulation on all engines
results = runner.run_parallel(simulation_params)

# Generate comparison report
report = runner.generate_comparison_report(results)
report.save("multi_engine_comparison.html")
```

### 8.3 Motion Retargeting

Retarget motion capture data to different models:

```python
from shared.python.motion_capture import MotionRetargeting

# Load source motion
source_motion = MotionCaptureLoader.load_c3d("professional_swing.c3d")

# Create retargeting pipeline
retargeter = MotionRetargeting(
    source_skeleton="vicon_full_body",
    target_model=engine.model
)

# Map markers to joints
marker_mapping = {
    "LSHO": "left_shoulder",
    "LELB": "left_elbow",
    "LWRI": "left_wrist",
    # ... more markers
}
retargeter.set_marker_mapping(marker_mapping)

# Perform retargeting
joint_trajectory = retargeter.retarget(source_motion)

# Apply to simulation
for t, q in joint_trajectory:
    engine.set_state(q, v=None)
    engine.step(dt)
```

---

## 9. Configuration

### 9.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOLF_SUITE_MODE` | `local` | Authentication mode (`local`, `cloud`) |
| `GOLF_AUTH_DISABLED` | `false` | Force disable authentication |
| `GOLF_PORT` | `8000` | API server port |
| `GOLF_USE_MOCK_ENGINE` | `false` | Use mock engine for testing |
| `CORS_ORIGINS` | `localhost` | Allowed CORS origins |
| `GOLF_API_SECRET_KEY` | - | JWT signing key (cloud mode) |
| `DATABASE_URL` | `sqlite:///.db` | Database connection string |
| `GOLF_LOG_LEVEL` | `INFO` | Logging level |

### 9.2 Engine Parameters

#### MuJoCo Configuration

```python
mujoco_config = {
    "timestep": 0.002,          # Simulation timestep (seconds)
    "integrator": "RK4",        # RK4, Euler, implicit
    "iterations": 100,          # Solver iterations
    "gravity": [0, 0, -9.81],   # Gravity vector
    "cone_type": "elliptic",    # Friction cone type
    "jacobian_type": "dense",   # Jacobian computation
}
engine.configure(mujoco_config)
```

#### Drake Configuration

```python
drake_config = {
    "timestep": 0.001,
    "integrator": "runge_kutta3",
    "contact_model": "hydroelastic",
    "solver": "SNOPT",          # For optimization
}
```

#### Pinocchio Configuration

```python
pinocchio_config = {
    "integrator": "euler",
    "compute_derivatives": True,
    "use_local_frame": True,
}
```

### 9.3 Logging Configuration

Configure logging via environment or code:

```python
import logging

# Set log level
logging.getLogger("golf_suite").setLevel(logging.DEBUG)

# Or via environment
# GOLF_LOG_LEVEL=DEBUG

# Log to file
handler = logging.FileHandler("golf_suite.log")
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.getLogger("golf_suite").addHandler(handler)
```

### 9.4 Configuration Files

```
UpstreamDrift/
├── .env                    # Environment overrides (not in git)
├── pyproject.toml          # Project configuration
├── environment.yml         # Conda environment
└── src/
    └── api/
        └── config.py       # API configuration
```

---

## 10. Troubleshooting

### 10.1 Error Code Reference

All errors use structured codes with the format `GMS-XXX-YYY`:

| Prefix | Category | Description |
|--------|----------|-------------|
| GMS-GEN | General | Internal errors, rate limits |
| GMS-ENG | Engine | Load failures, invalid state |
| GMS-SIM | Simulation | Timeout, invalid parameters |
| GMS-VID | Video | Invalid format, processing errors |
| GMS-ANL | Analysis | Service not ready |
| GMS-AUT | Auth | Token expired, quota exceeded |
| GMS-VAL | Validation | Missing field, invalid value |
| GMS-RES | Resource | Not found, access denied |
| GMS-SYS | System | Database, configuration |

### 10.2 Common Error Codes

#### GMS-ENG-001: Engine Not Found

**Cause**: Requested physics engine is not installed.

**Solution**:
```bash
# Check available engines
python -c "from shared.python import EngineManager; print(EngineManager().list_available())"

# Install missing engine
pip install mujoco  # or conda install drake
```

#### GMS-ENG-003: Failed to Load Physics Engine

**Cause**: Engine dependencies missing or incompatible.

**Solution**:
```bash
# Reinstall engine
pip uninstall mujoco && pip install mujoco

# Check for dependency conflicts
pip check
```

#### GMS-SIM-001: Simulation Timeout

**Cause**: Simulation exceeded time limit.

**Solution**:
- Reduce simulation duration
- Increase timestep (may reduce accuracy)
- Use async simulation endpoint

#### GMS-VAL-001: Invalid Request Parameters

**Cause**: Request body missing required fields.

**Solution**: Check API documentation for required fields:
```bash
curl http://localhost:8000/docs
```

### 10.3 Frequently Asked Questions

#### Q: GUI crashes on startup

**A**: Try running with verbose logging:
```bash
python launch_golf_suite.py --debug
```

For headless environments:
```bash
export QT_QPA_PLATFORM=offscreen
python launch_golf_suite.py
```

#### Q: Simulation runs slowly

**A**: Try these optimizations:

1. Use headless mode for batch processing:
   ```python
   engine.set_render_mode("offscreen")
   ```

2. Reduce visualization frequency:
   ```python
   engine.step(render_every=10)
   ```

3. Use a lighter model (fewer DOF)

#### Q: Cross-engine results don't match

**A**: Small differences are expected due to:
- Different integration methods
- Contact model variations
- Numerical precision

Check that deviations are within tolerance guidelines (see Section 8.1).

#### Q: MuJoCo not found on Windows

**A**: MuJoCo is pip-installable:
```bash
pip install mujoco
python -c "import mujoco; print(mujoco.__version__)"
```

#### Q: Drake installation fails

**A**: Drake has specific platform requirements:
```bash
# macOS/Linux via conda (recommended)
conda install -c conda-forge drake

# Windows: Use WSL2 or see https://drake.mit.edu
```

#### Q: How do I use MyoSuite muscles with Drake?

**A**: MyoSuite is MuJoCo-exclusive. For Drake, convert muscle forces to equivalent torques:
```python
# In MuJoCo: compute muscle torques
muscle_forces = analyzer.get_muscle_forces()
moment_arms = analyzer.compute_moment_arms()
joint_torques = {m: f * r for m, f, r in zip(muscles, muscle_forces, moment_arms)}

# Apply in Drake as external torques
drake_engine.set_external_torques(joint_torques)
```

### 10.4 Getting Help

1. **Check documentation**: http://localhost:8000/docs
2. **Search GitHub Issues**: https://github.com/dieterolson/UpstreamDrift/issues
3. **Run diagnostics**:
   ```python
   from shared.python import EngineManager
   print(EngineManager().get_diagnostic_report())
   ```
4. **Open a new issue** with:
   - Python version (`python --version`)
   - OS and version
   - Full error traceback
   - Steps to reproduce

---

## 11. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| **DOF** | Degrees of Freedom - number of independent motion parameters |
| **Forward Dynamics** | Computing accelerations given forces |
| **Inverse Dynamics** | Computing forces given desired accelerations |
| **Jacobian** | Matrix relating joint velocities to end-effector velocities |
| **RNEA** | Recursive Newton-Euler Algorithm for inverse dynamics |
| **ABA** | Articulated Body Algorithm for forward dynamics |
| **URDF** | Unified Robot Description Format |
| **ZTCF** | Zero Torque Counterfactual - analysis with all torques set to zero |
| **ZVCF** | Zero Velocity Counterfactual - analysis with velocities set to zero |
| **Hill-Type Muscle** | Muscle model with force-length and force-velocity relationships |

### Appendix B: Physical Constants

| Constant | Value | Units |
|----------|-------|-------|
| Gravity | 9.81 | m/s^2 |
| Golf ball mass | 0.0459 | kg |
| Golf ball diameter | 0.0427 | m |
| Driver club length | 1.143 | m |
| Driver head mass | 0.200 | kg |

### Appendix C: Model File Locations

```
UpstreamDrift/
├── shared/
│   └── models/
│       ├── mujoco/           # MuJoCo XML models
│       ├── urdf/             # URDF robot descriptions
│       ├── myosuite/         # MyoSuite muscle models
│       │   └── myo_sim/      # Official MyoSim (submodule)
│       └── opensim/          # OpenSim .osim models
│           └── opensim-models/  # Official models (submodule)
└── engines/
    └── physics_engines/
        └── mujoco/
            └── python/
                └── mujoco_golf_pendulum/
                    └── models.py  # Inline XML definitions
```

### Appendix D: Citation

If you use UpstreamDrift in your research, please cite:

```bibtex
@software{upstream_drift,
  title = {UpstreamDrift: A Unified Platform for Biomechanical Golf Swing Analysis},
  author = {Dieter Olson},
  year = {2026},
  url = {https://github.com/dieterolson/UpstreamDrift}
}
```

### Appendix E: License

MIT License

Copyright (c) 2026 UpstreamDrift Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**Document Revision History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.1 | Feb 2026 | UpstreamDrift Team | Complete rewrite for v2.1 |
| 2.0 | Jan 2026 | UpstreamDrift Team | Added multi-engine support |
| 1.0 | Dec 2025 | UpstreamDrift Team | Initial release |

---

*End of Document*
