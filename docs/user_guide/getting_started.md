# Getting Started

## Running the Unified Launcher

The easiest way to explore the suite is using the Python-based unified launcher.

```bash
python launchers/golf_launcher.py
```

This will open a graphical interface allowing you to:
- Select a Physics Engine (MuJoCo, Drake, Pinocchio).
- Choose a specific model (e.g., 2D Golf, Humanoid).
- Configure simulation parameters.
- Launch the simulation.

## Running Specific Engines directly

### MuJoCo
```bash
python engines/physics_engines/mujoco/python/main.py
```

### Drake
```bash
# Ensure you are in the root directory
python -m engines.physics_engines.drake.python.main
```

### Pinocchio
```bash
python engines/physics_engines/pinocchio/python/main.py
```

## MATLAB Models

1. Open MATLAB.
2. Run `setup_golf_suite()`.
3. Open the desired `.slx` file from `engines/Simscape_Multibody_Models/`.
4. Press Run in Simulink.
