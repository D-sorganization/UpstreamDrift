# Examples

This directory contains executable scripts and reference workflows demonstrating the core capabilities of UpstreamDrift.

## Core Simulation and Parameter Studies

- **01_basic_simulation.py**: Demonstrates initializing the engine manager, loading MuJoCo (gracefully handling absence), running a simulation loop, and saving results.
- **02_parameter_sweeps.py**: Demonstrates accessing the Physics Parameter Registry, running a parameter sweep, and exporting analysis reports.
- **03_injury_risk_tutorial.py**: Demonstrates spinal/joint loading and injury-risk metric evaluation across a swing trajectory.

## Domain and Physics Demos

- **aerodynamics_demo.py**: Demonstrates aerodynamic drag, Magnus lift, and wind effects on ball flight.
- **basic_flight_simulation.py**: Demonstrates minimal projectile and golf-ball flight integration.
- **motion_training_demo.py**: Demonstrates motion-training workflows and feedback signals.
- **topography_demo.py**: Demonstrates terrain elevation and sloped landing surface queries.

## Subdirectory Workflows

- **motion_capture/**: Motion-capture driving and parameter sweep examples (`parameter_sweep.py`, `reference_driver.json`).
- **simulation_backends/**: Backend-specific simulation examples across supported physics engines.

## Running Examples

Ensure your environment is set up (see `docs/development/contributing.md` and `docs/user_guide/installation.md`), then run:

```bash
python examples/01_basic_simulation.py
python examples/02_parameter_sweeps.py
python examples/03_injury_risk_tutorial.py
```

Results are saved to `output/`.
