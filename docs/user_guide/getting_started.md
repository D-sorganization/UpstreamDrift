# Getting Started

## Installation

```bash
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required (`requires-python = ">=3.11"` in
`pyproject.toml`).

## Running the Unified Launcher

The easiest way to explore the suite is the `upstream-drift` console script
installed by the editable install above:

```bash
upstream-drift             # web UI (opens in a browser)
upstream-drift --classic   # classic PyQt6 desktop launcher
upstream-drift --api-only  # FastAPI server only, no UI
```

Without installing, the same entry point is available as a script from the
repository root:

```bash
python launch_upstream_drift.py --classic
```

Either way you get an interface that lets you:

- Select a physics engine (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite, ...).
- Choose a specific model (e.g. 2D Golf, Humanoid).
- Configure simulation parameters.
- Launch the simulation.

## Running Specific Engines Directly

The launcher can jump straight to one engine:

```bash
upstream-drift --engine mujoco --no-browser
```

`--engine` accepts the values of `EngineType`: `mujoco`, `drake`, `pinocchio`,
`jaxsim`, `opensim`, `myosim`, `matlab_2d`, `matlab_3d`, `pendulum`,
`golf_swing_pendulum`, `putting_green`.

The individual engine GUIs can also be started directly. The canonical entry
points are declared in `src/config/models.yaml`; run these from the repository
root:

### MuJoCo

```bash
python -m src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf
```

### Drake

```bash
python -m src.engines.physics_engines.drake.python
```

### Pinocchio

```bash
python -m src.engines.physics_engines.pinocchio.python
```

## MATLAB Models

1. Open MATLAB.
2. Add `src/shared/matlab/` to the path and run `setup_golf_suite()`.
3. Open the desired `.slx` file from
   `src/engines/Simscape_Multibody_Models/` (for example
   `2D_Golf_Model/matlab/GolfSwing.slx`).
4. Press Run in Simulink.
