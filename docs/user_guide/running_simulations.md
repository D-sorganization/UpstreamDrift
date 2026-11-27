# Running Simulations

UpstreamDrift offers multiple ways to run simulations, from the unified launcher to headless CLI and Python APIs.

## Unified Launcher

The easiest way to start interactively is via the unified launcher:

```bash
upstream-drift
# or from source:
python launch_upstream_drift.py
```

This starts the local API server and opens the interface where you can:

- Select a physics engine (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite, Pendulum, etc.)
- Configure simulation parameters
- Visualize results and diagnostics
- Export data

## Headless and Programmatic Execution

For headless batch simulations, use `ud-sim`:

```bash
ud-sim run tests/fixtures/sim_run/smoke_case.json --format summary
```

You can also manage simulation outputs directly from Python scripts.

### Basic Example

```python
from src.shared.python.output_manager import OutputManager

# Setup output
manager = OutputManager()
manager.create_output_structure()

# Run a simulation (conceptual)
# engine = MujocoPhysicsEngine()
# engine.load_from_path("path/to/model.xml")
# result = engine.step(0.002)

# Save results
# manager.save_simulation_results(result, "swing_test")
```
