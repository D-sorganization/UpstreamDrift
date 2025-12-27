# Pendulums Model (double/triple pendulum toolkit)

This folder is a relocated copy of the redesigned pendulum toolkit so the original `Double Pendulum Model` folder stays untouched. It bundles the PyQt6 and Tkinter desktop apps, browser playground, and physics engines for double and triple pendulums.

## Contents
- `double_pendulum_model/physics/double_pendulum.py`: control-affine dynamics, parameter helpers, and safe expression parsing.
- `double_pendulum_model/ui/double_pendulum_gui.py`: desktop GUI for configuring and visualizing the pendulum.
- `double_pendulum_model/visualization/double_pendulum_web/`: HTML/JS playground for browser demos.
- `double_pendulum_model/tests/`: unit tests for the physics utilities.

## Running the GUI

You can run the GUI in several ways:

### Option 1: Run the launcher script (Easiest for IDE)
```bash
python "Pendulums_Model/run_pendulum.py"
```
Or simply click the play button in your IDE when `run_pendulum.py` is open.

### Option 2: Run as a module (from this folder)
```bash
cd Pendulums_Model
python -m double_pendulum_model
```

### Option 3: Run the GUI module directly (from this folder)
```bash
cd Pendulums_Model
python -m double_pendulum_model.ui.double_pendulum_gui
```

## Where did the rotatable/advanced GUI go?

The 3D rotatable Tkinter GUI with gravity toggles, plane constraints, and logging tools is still present in this relocated copy. Launch the GUI via the commands above to access the rotatable view without affecting the original folder.

## Running the tests
```bash
python -m pytest "Pendulums_Model/double_pendulum_model/tests"
```
