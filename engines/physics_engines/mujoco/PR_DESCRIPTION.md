# Migrate Docker Environment from Repository_Management

## Overview

This PR migrates the Docker environment and GUI components from the `Repository_Management` repository to `MuJoCo_Golf_Swing_Model`. This consolidates the simulation environment with the model code.

## Changes

- **Added**: `docker/` directory containing:
  - `Dockerfile`, `requirements.txt`: Environment definition.
  - `gui/`: Tkinter-based launcher (`deepmind_control_suite_MuJoCo_GUI.py`).
  - `src/`: Simulation package (`humanoid_golf`).
  - `run_with_viewer.sh`, `build_and_run.bat`: Helper scripts.
  - Examples: `example_dynamic_stance.py`, etc.

## Rationale

Previously, the Docker tools were in a separate repo. Moving them here allows for a self-contained repository where users can build and run the model immediately.

## Impact

- ✅ Self-contained repository.
- ✅ Reproducible environment via Docker.
- ✅ Includes GUI for ease of use.
