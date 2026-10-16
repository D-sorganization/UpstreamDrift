# Chapter 4: Quick Start Guide

This chapter walks through the essential workflows to get you productive with
UpstreamDrift as quickly as possible. Each section includes executable commands
and expected output.

## 4.1 Running Your First Simulation

The fastest way to run a simulation is through the example scripts in the
`examples/` directory.

### Example 1: Basic Simulation

```bash
python examples/01_basic_simulation.py
```

This script:

1. Initializes the `EngineManager` and selects the MuJoCo engine (default).
2. Loads a 7 DOF humanoid golf model.
3. Applies a sequence of joint torques representing a simplified golf swing.
4. Runs the simulation for 2 seconds at $\Delta t = 0.001$ s (2000 time steps).
5. Records joint positions, velocities, and energies.
6. Generates plots of the swing trajectory and energy profile.

### Example 2: Parameter Sweeps

```bash
python examples/02_parameter_sweeps.py
```

This script demonstrates how to sweep over model parameters (e.g., club mass,
shaft stiffness, grip friction) and observe their effect on club head velocity
at impact. The parameter sweep uses the kinetic energy at impact as the
objective function:

$$E_{\text{impact}} = \frac{1}{2} m_{\text{club}} v_{\text{club}}^2$$

where $m_{\text{club}}$ is the club head mass and $v_{\text{club}}$ is the
club head velocity at the moment of ball contact.

### Example 3: Injury Risk Tutorial

```bash
python examples/03_injury_risk_tutorial.py
```

This script demonstrates the injury risk assessment pipeline, computing joint
stress metrics, spinal load estimates, and soft tissue strain indicators across
a full swing cycle.

### Programmatic Simulation

For custom simulations, use the Python API directly:

```python
#!/usr/bin/env python3
"""Minimal simulation example."""

import numpy as np
from src.shared.python.engine_core.engine_manager import EngineManager

# 1. Initialize the engine
manager = EngineManager()
engine = manager.get_engine("mujoco")

# 2. Load a model
engine.load_model("src/shared/models/urdf/humanoid_7dof.urdf")

# 3. Set initial conditions
engine.reset()

# 4. Simulation loop
dt = 0.001  # 1 ms time step
duration = 2.0  # 2 seconds
n_steps = int(duration / dt)

trajectory = []
for step in range(n_steps):
    # Apply torques (example: sinusoidal torque on shoulder joint)
    t = step * dt
    torques = np.zeros(7)
    torques[0] = 50.0 * np.sin(2 * np.pi * t / duration)  # Shoulder

    engine.set_control(torques)
    engine.step(dt)

    state = engine.get_state()
    trajectory.append(state)

# 5. Access results
print(f"Simulation complete: {n_steps} steps")
print(f"Final joint positions: {trajectory[-1]['q']}")
print(f"Final joint velocities: {trajectory[-1]['qdot']}")
```

## 4.2 Launching the GUI (PyQt6 Classic Launcher)

The classic PyQt6 launcher provides a native desktop interface with engine
selection, model browsing, visualization, and analysis tools.

### Launch Command

```bash
# Via the main entry point
python launch_upstream_drift.py --classic

# Or equivalently
upstream-drift --classic
```

### GUI Features

The classic launcher provides:

- **Engine Selection Panel**: Choose from available engines (MuJoCo, Drake,
  Pinocchio, etc.) with capability indicators.
- **Model Browser**: Browse and select from available models organized by
  complexity (pendulum, intermediate, advanced, musculoskeletal).
- **Simulation Controls**: Start, pause, reset, and step through simulations.
  Configure time step, duration, and integration parameters.
- **Visualization Viewport**: Real-time 3D rendering with multiple camera views,
  force/torque vector overlays, and manipulability ellipsoids.
- **Analysis Panel**: Access inverse kinematics, inverse dynamics, kinematic
  sequence analysis, energy plots, and data export tools.
- **Engine Dashboards**: Dedicated dashboards for MuJoCo, Drake, and Pinocchio
  with engine-specific controls and visualizations.

### Startup Sequence

The launcher uses an async startup system with a splash screen:

1. **Splash screen displayed immediately** (< 100 ms after launch).
2. **Background worker thread** loads heavy modules (MuJoCo, engine probes).
3. **Progress bar updates** reflect actual loading progress.
4. **Main window opens** with pre-loaded resources (no duplicate loading).

### Keyboard Shortcuts

| Shortcut | Action                   |
| -------- | ------------------------ |
| `Space`  | Start / pause simulation |
| `R`      | Reset simulation         |
| `S`      | Step forward one frame   |
| `Ctrl+E` | Export current data      |
| `Ctrl+P` | Capture screenshot       |
| `1`-`5`  | Switch camera view       |
| `Ctrl+Q` | Quit application         |

## 4.3 Launching the Web UI (Tauri/React)

The modern web-based desktop application uses Tauri (Rust backend) and React
(TypeScript frontend) for a contemporary user experience.

### Launch Command (Default)

```bash
# Default launch mode (recommended)
python launch_upstream_drift.py

# Or equivalently
golf-suite
```

This starts the FastAPI backend server and opens the web UI in your default
browser at `http://localhost:8000`.

### Development Mode

For active UI development with hot-reload:

```bash
cd ui
npm install
npm run dev
```

This starts the Vite development server with hot module replacement (HMR).
The UI source code is in `ui/src/` and uses:

- **React** (v18+) with hooks
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Vite** for fast builds and HMR
- **Vitest** for unit testing

### Building the Desktop Application

To build a native desktop application using Tauri:

```bash
cd ui
npm install
npm run tauri build
```

This produces a standalone executable for your platform (Linux AppImage, macOS
.dmg, Windows .exe) that bundles the FastAPI backend and React frontend into
a single distributable application.

## 4.4 Starting the API Server

The REST API provides programmatic access to all simulation and analysis
capabilities.

### Launch the API Server

```bash
# Via the main entry point
python launch_upstream_drift.py --api-only

# Or directly
python start_api_server.py

# Or via the upstream-drift CLI
upstream-drift --api-only --port 8000
```

**Local Mode vs. Authenticated Mode**

By default, the local server runs with authentication disabled (`GOLF_AUTH_DISABLED=true`)
so that offline workflows work without user accounts. For a JWT-authenticated deployment,
run the production server entry point (`src/api/server.py`) or set
`GOLF_AUTH_DISABLED=false` in your environment before startup.

### API Documentation

Once the server is running, interactive documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/api/health`

### Quick API Test

```bash
# Check server health
curl http://localhost:8000/api/health

# Authenticate (get JWT token)
curl -X POST http://localhost:8000/api/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username": "admin", "password": "your-password"}'

# List available engines (authenticated)
curl http://localhost:8000/api/engines \
    -H "Authorization: Bearer <your-jwt-token>"

# Start a simulation
curl -X POST http://localhost:8000/api/simulation/start \
    -H "Authorization: Bearer <your-jwt-token>" \
    -H "Content-Type: application/json" \
    -d '{
        "engine": "mujoco",
        "model": "humanoid_7dof",
        "duration": 2.0,
        "dt": 0.001
    }'
```

### API Server Configuration

The API server accepts the following command-line arguments and environment
variables:

| Argument       | Environment Variable | Default       | Description                |
| -------------- | -------------------- | ------------- | -------------------------- |
| `--port`       | `API_PORT`           | `8000`        | Server port                |
| (n/a)          | `API_HOST`           | `127.0.0.1`   | Server bind address        |
| (n/a)          | `ENVIRONMENT`        | `development` | `development`/`production` |
| (n/a)          | `SECRET_KEY`         | (generated)   | JWT signing key            |
| `--no-browser` | `GOLF_NO_BROWSER`    | (unset)       | Suppress browser launch    |

In development mode (`ENVIRONMENT=development`), the server enables auto-reload,
which watches for file changes and automatically restarts.

## 4.5 Basic Workflows

### Workflow 1: Compare Engines on the Same Model

This workflow runs the same simulation on multiple engines and compares results:

```python
from src.shared.python.engine_core.engine_manager import EngineManager
from src.shared.python.comparative_analysis import ComparativeAnalysis

manager = EngineManager()
model_path = "src/shared/models/urdf/humanoid_7dof.urdf"

results = {}
for engine_name in ["mujoco", "drake", "pinocchio"]:
    engine = manager.get_engine(engine_name)
    engine.load_model(model_path)
    engine.reset()

    # Run simulation
    for _ in range(2000):
        engine.step(dt=0.001)

    results[engine_name] = engine.get_state()

# Compare
analyzer = ComparativeAnalysis()
comparison = analyzer.compare(results)
analyzer.plot_comparison(comparison)
```

### Workflow 2: Motion Capture to Simulation

This workflow imports motion capture data and drives a simulation model:

```python
from src.shared.python.video_pose_pipeline import (
    VideoPosePipeline,
    VideoProcessingConfig,
)

# Configure the pipeline
config = VideoProcessingConfig(
    input_path="data/swing_video.mp4",
    pose_method="mediapipe",
    output_format="json",
)

# Run pose estimation
pipeline = VideoPosePipeline(config)
pose_data = pipeline.process()

# Retarget to simulation model
from src.learning.retargeting import MotionRetargeter

retargeter = MotionRetargeter(
    source_skeleton="mediapipe",
    target_model="humanoid_28dof",
)
joint_angles = retargeter.retarget(pose_data)
```

### Workflow 3: Injury Risk Assessment

```python
from src.shared.python.injury import InjuryRiskAssessor

assessor = InjuryRiskAssessor()

# Load simulation results (from any engine)
trajectory = load_simulation_results("output/swing_data.json")

# Assess injury risk
report = assessor.assess(trajectory)

print(f"Overall risk level: {report.overall_risk}")
print(f"Peak spinal load: {report.spinal_load_peak:.1f} N")
print(f"Shoulder stress: {report.shoulder_stress:.1f} Nm")
print(f"Wrist strain index: {report.wrist_strain:.3f}")
```

### Workflow 4: Ball Flight Prediction

Given club head state at impact, predict the full ball flight trajectory:

```python
from src.shared.python.aerodynamics import BallFlightModel
from src.shared.python.impact_model import ImpactModel

# Compute ball launch conditions from impact
impact = ImpactModel()
launch = impact.compute(
    club_speed=45.0,          # m/s
    attack_angle=-3.0,        # degrees (negative = descending)
    club_path=2.0,            # degrees (positive = in-to-out)
    face_angle=1.0,           # degrees (positive = open)
    dynamic_loft=12.0,        # degrees
)

# Simulate ball flight
flight = BallFlightModel()
trajectory = flight.simulate(
    launch_speed=launch.ball_speed,     # m/s
    launch_angle=launch.launch_angle,   # degrees
    spin_rate=launch.spin_rate,         # rpm
    spin_axis=launch.spin_axis,         # degrees
    wind_speed=5.0,                     # m/s
    wind_direction=180.0,               # degrees (headwind)
    altitude=0.0,                       # meters above sea level
)

print(f"Carry distance: {trajectory.carry:.1f} m")
print(f"Total distance: {trajectory.total:.1f} m")
print(f"Max height: {trajectory.apex:.1f} m")
print(f"Flight time: {trajectory.time:.2f} s")
```

The aerodynamic model accounts for:

- Drag force: $F_D = \frac{1}{2} \rho C_D A v^2$
- Lift force (Magnus effect): $F_L = \frac{1}{2} \rho C_L A v^2$
- Gravitational force: $F_g = m g$

where $\rho$ is air density, $C_D$ and $C_L$ are drag and lift coefficients
(dependent on Reynolds number $Re = \frac{\rho v d}{\mu}$ and spin ratio
$S = \frac{\omega r}{v}$), $A$ is the cross-sectional area, and $v$ is the
ball velocity.

## 4.6 Exploring the Example Scripts

The `examples/` directory contains annotated scripts for common use cases:

| Script                       | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `01_basic_simulation.py`     | Simple forward dynamics simulation                |
| `02_parameter_sweeps.py`     | Sweep over model parameters and visualize effects |
| `03_injury_risk_tutorial.py` | Injury risk assessment pipeline walkthrough       |
| `motion_training_demo.py`    | Motion capture import and retargeting demo        |

Each script includes detailed comments explaining the workflow, parameters,
and expected output. They are designed to be run directly:

```bash
python examples/01_basic_simulation.py
```

## 4.7 Understanding Output and Results

### Output Directory

Simulation results are stored in the `output/` directory by default. Each
simulation run creates a timestamped subdirectory containing:

```
output/
├── 2026-02-05_143022_mujoco_7dof/
│   ├── trajectory.csv          # Joint positions and velocities over time
│   ├── dynamics.csv            # Forces, torques, energies over time
│   ├── metadata.json           # Simulation parameters and configuration
│   ├── plots/                  # Generated visualization plots
│   │   ├── joint_angles.png
│   │   ├── energy_profile.png
│   │   ├── phase_diagram.png
│   │   └── 3d_trajectory.png
│   └── analysis/               # Derived analysis results
│       ├── kinematic_sequence.json
│       ├── peak_torques.csv
│       └── injury_risk_report.json
```

### Database Storage

The SQLite database (`golf_modeling_suite.db`) in the repository root stores
structured simulation results, user configurations, and analysis metadata.
It is accessed by the API server and can be queried directly:

```bash
sqlite3 golf_modeling_suite.db ".tables"
sqlite3 golf_modeling_suite.db "SELECT * FROM simulations LIMIT 5;"
```

### Data Export Formats

Results can be exported in two formats:

- **CSV**: Tabular data with headers, suitable for spreadsheet analysis and
  external tools (MATLAB, R, Excel).
- **JSON**: Hierarchical data with full metadata, suitable for programmatic
  consumption and API responses.

Export is available through the GUI (File > Export), the API (`/api/export/`
endpoints), and programmatically via the `src/shared/python/export.py` module.

---

_This concludes Part I: Getting Started. The subsequent parts of this manual
cover the physics engines in depth (Part II), analysis and simulation tools
(Part III), robotics and control (Part IV), research tools (Part V), API and
integration (Part VI), development and operations (Part VII), and appendices
(Part VIII)._

---

_UpstreamDrift v2.1.0 -- User Manual -- Part I_

_Document generated for UpstreamDrift by the documentation team._

---
