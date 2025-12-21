# Running Simulations

The Golf Modeling Suite offers multiple ways to run simulations, from a graphical user interface to programmatic control.

## 🚀 Unified Launcher

The easiest way to start is using the Unified Launcher:

```bash
python launch_golf_suite.py
```

This opens a GUI where you can:
- specify specific physics engine (MuJoCo, Drake, etc.)
- Configure simulation parameters
- Visualize results
- Export data

## 💻 Command Line Interface

You can also run simulations directly from Python scripts.

### Basic Example

```python
from shared.python.output_manager import OutputManager

# Setup output
manager = OutputManager()
manager.create_output_structure()

# Run a simulation (conceptual)
# engine = MujocoEngine()
# result = engine.simulate(duration=3.0)

# Save results
# manager.save_simulation_results(result, "swing_test")
```
