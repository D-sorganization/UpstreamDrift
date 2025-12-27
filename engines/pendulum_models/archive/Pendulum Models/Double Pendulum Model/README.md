# Double Pendulum Model

This folder contains a standalone, driven double pendulum toolkit with both a Tkinter GUI and a browser-based playground. It was separated from the solar system simulator to keep the projects independent.

## Contents
- `double_pendulum_model/physics/double_pendulum.py`: control-affine dynamics, parameter helpers, and safe expression parsing.
- `double_pendulum_model/ui/double_pendulum_gui.py`: desktop GUI for configuring and visualizing the pendulum.
- `double_pendulum_model/visualization/double_pendulum_web/`: HTML/JS playground for browser demos.
- `double_pendulum_model/tests/`: unit tests for the physics utilities.

## Running the GUI

You can run the GUI in several ways:

### Option 1: Run the launcher script (Easiest for IDE)
```bash
python "Double Pendulum Model/run_pendulum.py"
```
Or simply click the play button in your IDE when `run_pendulum.py` is open.

### Option 2: Run as a module
```bash
python -m double_pendulum_model
```

### Option 3: Run the GUI module directly
```bash
python -m double_pendulum_model.ui.double_pendulum_gui
```

## Where did the rotatable/advanced GUI go?

The 3D rotatable Tkinter GUI with gravity toggles, plane constraints, and logging tools is still present. It was introduced in commit `92eb6a4` ("Enhance double pendulum GUI with 3D visualization and advanced features") and has only received physics fixes since (`98d6c88`, `1f5017f`, `f7da94c`). The recent "golf swing" update (`12f3420`) touched only the browser playground (`visualization/double_pendulum_web/app.js`) and did not remove the advanced desktop GUI. Launch the GUI via the commands above to access the rotatable view.

## Running the tests
```bash
python -m pytest "Double Pendulum Model/double_pendulum_model/tests"
```
