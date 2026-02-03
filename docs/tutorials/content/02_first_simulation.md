# Tutorial 2: Your First Simulation

**Estimated Time:** 45 minutes
**Difficulty:** Beginner

## Prerequisites
- Completed [Tutorial 1: Getting Started](01_getting_started.md)
- Golf Modeling Suite installed and verified
- Basic Python knowledge

## Learning Objectives
By the end of this tutorial, you will:
- Load and configure a biomechanical model
- Run a physics simulation using MuJoCo
- Analyze joint kinematics and dynamics
- Export simulation results for further analysis

## Overview

The Golf Modeling Suite uses a unified physics engine interface, allowing you to run the same simulation across different physics backends. In this tutorial, we'll use MuJoCo as our primary engine - it's the most comprehensive option for biomechanical golf swing simulation.

## Step 1: Understanding the Engine Architecture

The suite supports multiple physics engines through a common interface:

| Engine | Best For | Key Strengths |
|--------|----------|---------------|
| **MuJoCo** | Production simulations | Contact physics, muscle simulation, flexible shafts |
| **Drake** | Control optimization | Multi-body dynamics, trajectory optimization |
| **Pinocchio** | Fast kinematics | Efficient recursive algorithms, URDF support |
| **OpenSim** | Biomechanics research | Muscle-driven simulation, .osim models |
| **MyoSuite** | RL training | Gym environment, muscle activation |
| **Pendulum** | Testing/validation | Analytical solutions, 6-DOF model |

## Step 2: Create Your First Simulation Script

Create a new file `my_first_simulation.py`:

```python
"""My first simulation with the Golf Modeling Suite."""

from pathlib import Path
from src.shared.python.engine_manager import EngineManager, EngineType

# Define project paths
project_root = Path(__file__).parent.parent
model_path = project_root / "shared/models/mujoco/humanoid/humanoid.xml"

def main():
    # Initialize the engine manager
    engine_manager = EngineManager(project_root)

    # List available engines on your system
    available = engine_manager.get_available_engines()
    print(f"Available engines: {[e.name for e in available]}")

    # Switch to MuJoCo (default engine)
    engine_manager.switch_engine(EngineType.MUJOCO)
    print("Activated MuJoCo engine")

    # Get the physics engine instance
    physics_engine = engine_manager.get_active_physics_engine()

    # Load the humanoid model
    physics_engine.load_from_path(str(model_path))
    print(f"Loaded model from {model_path}")

    return physics_engine, engine_manager

if __name__ == "__main__":
    physics_engine, _ = main()
    print("Model loaded successfully!")
```

Run the script:
```bash
python my_first_simulation.py
```

Expected output:
```
Available engines: ['MUJOCO', 'PINOCCHIO', 'PENDULUM']
Activated MuJoCo engine
Loaded model from shared/models/mujoco/humanoid/humanoid.xml
Model loaded successfully!
```

## Step 3: Configure Simulation Parameters

Extend your script to configure simulation parameters:

```python
import numpy as np

def run_simulation(physics_engine, duration=2.0, timestep=0.002):
    """Run a forward simulation.

    Args:
        physics_engine: Initialized physics engine instance
        duration: Total simulation time in seconds
        timestep: Physics integration step size in seconds

    Returns:
        Dictionary containing time series data
    """
    # Calculate number of steps
    num_steps = int(duration / timestep)

    # Pre-allocate arrays for results
    times = np.zeros(num_steps)
    positions = []
    velocities = []

    # Reset simulation to initial state
    physics_engine.reset()

    print(f"Running simulation: {duration}s at {timestep}s timestep ({num_steps} steps)")

    # Main simulation loop
    for step in range(num_steps):
        # Record current time
        times[step] = physics_engine.get_time()

        # Get current state (positions and velocities)
        q, v = physics_engine.get_state()
        positions.append(q.copy())
        velocities.append(v.copy())

        # Step the simulation forward
        physics_engine.step(timestep)

        # Progress indicator
        if step % (num_steps // 10) == 0:
            print(f"  Progress: {100 * step / num_steps:.0f}%")

    print("Simulation complete!")

    return {
        "times": times,
        "positions": np.array(positions),
        "velocities": np.array(velocities),
    }
```

## Step 4: Analyze Joint Kinematics

Add analysis functions to examine joint behavior:

```python
def analyze_kinematics(results, joint_indices=None):
    """Analyze joint kinematics from simulation results.

    Args:
        results: Dictionary from run_simulation()
        joint_indices: List of joint indices to analyze (default: first 3)

    Returns:
        Dictionary of kinematic analysis
    """
    if joint_indices is None:
        joint_indices = [0, 1, 2]  # Analyze first 3 joints

    positions = results["positions"]
    velocities = results["velocities"]
    times = results["times"]

    analysis = {}

    for i, idx in enumerate(joint_indices):
        joint_pos = positions[:, idx]
        joint_vel = velocities[:, idx]

        # Compute acceleration via finite differences
        dt = times[1] - times[0]
        joint_acc = np.gradient(joint_vel, dt)

        analysis[f"joint_{idx}"] = {
            "position": joint_pos,
            "velocity": joint_vel,
            "acceleration": joint_acc,
            "pos_range": (joint_pos.min(), joint_pos.max()),
            "max_velocity": np.abs(joint_vel).max(),
            "max_acceleration": np.abs(joint_acc).max(),
        }

        print(f"Joint {idx}:")
        print(f"  Position range: {joint_pos.min():.4f} to {joint_pos.max():.4f} rad")
        print(f"  Max velocity: {np.abs(joint_vel).max():.4f} rad/s")
        print(f"  Max acceleration: {np.abs(joint_acc).max():.4f} rad/s^2")

    return analysis
```

## Step 5: Compute Dynamics Quantities

Access dynamics information from the physics engine:

```python
def compute_dynamics(physics_engine):
    """Compute key dynamics quantities at current state.

    Args:
        physics_engine: Physics engine instance

    Returns:
        Dictionary of dynamics data
    """
    # Get mass matrix M(q)
    mass_matrix = physics_engine.compute_mass_matrix()
    print(f"Mass matrix shape: {mass_matrix.shape}")

    # Get current state
    q, v = physics_engine.get_state()

    # Compute inverse dynamics: tau = ID(q, v, a)
    # With zero acceleration, this gives gravity + Coriolis forces
    zero_acc = np.zeros_like(v)
    passive_torques = physics_engine.compute_inverse_dynamics(zero_acc)
    print(f"Passive torques (gravity + Coriolis): {passive_torques[:3]}")

    # Compute drift acceleration (what happens with zero input)
    drift_acc = physics_engine.compute_drift_acceleration()
    print(f"Drift acceleration: {drift_acc[:3]}")

    # Compute Jacobian for a specific body
    try:
        jacobian = physics_engine.compute_jacobian("torso")
        print(f"Torso Jacobian shape: {jacobian.shape}")
    except Exception as e:
        print(f"Jacobian computation: {e}")

    return {
        "mass_matrix": mass_matrix,
        "passive_torques": passive_torques,
        "drift_acceleration": drift_acc,
    }
```

## Step 6: Export Results

Save your simulation data for further analysis:

```python
import json
from datetime import datetime

def export_results(results, analysis, output_path="simulation_results"):
    """Export simulation results to files.

    Args:
        results: Raw simulation data
        analysis: Kinematic analysis
        output_path: Output directory name
    """
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)

    # Save time series as NumPy arrays
    np.save(output_dir / "times.npy", results["times"])
    np.save(output_dir / "positions.npy", results["positions"])
    np.save(output_dir / "velocities.npy", results["velocities"])

    # Save analysis summary as JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "duration": float(results["times"][-1]),
        "num_steps": len(results["times"]),
        "joints_analyzed": list(analysis.keys()),
    }

    # Add scalar metrics (convert numpy types to Python types)
    for joint_name, data in analysis.items():
        summary[joint_name] = {
            "pos_range": [float(data["pos_range"][0]), float(data["pos_range"][1])],
            "max_velocity": float(data["max_velocity"]),
            "max_acceleration": float(data["max_acceleration"]),
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results exported to {output_dir}/")
    print(f"  - times.npy ({results['times'].nbytes} bytes)")
    print(f"  - positions.npy ({results['positions'].nbytes} bytes)")
    print(f"  - velocities.npy ({results['velocities'].nbytes} bytes)")
    print(f"  - summary.json")
```

## Step 7: Complete Example

Here's the complete script putting it all together:

```python
"""Complete first simulation example."""

from pathlib import Path
import numpy as np
import json
from datetime import datetime

from src.shared.python.engine_manager import EngineManager, EngineType


def main():
    # Setup
    project_root = Path(__file__).parent.parent
    model_path = project_root / "shared/models/mujoco/humanoid/humanoid.xml"

    # Initialize engine
    engine_manager = EngineManager(project_root)
    engine_manager.switch_engine(EngineType.MUJOCO)
    physics_engine = engine_manager.get_active_physics_engine()
    physics_engine.load_from_path(str(model_path))

    # Run simulation
    results = run_simulation(physics_engine, duration=2.0, timestep=0.002)

    # Analyze results
    analysis = analyze_kinematics(results, joint_indices=[0, 1, 2])

    # Compute dynamics at final state
    dynamics = compute_dynamics(physics_engine)

    # Export
    export_results(results, analysis)

    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
```

## Troubleshooting

### Model Not Found
Ensure you've pulled the Git LFS files:
```bash
git lfs pull
```

### Engine Not Available
Check which engines are installed:
```python
from src.shared.python.engine_availability import check_availability
print(check_availability())
```

### Numerical Instabilities
If the simulation becomes unstable:
- Reduce the timestep (try 0.001s or smaller)
- Check initial conditions are physically reasonable
- Verify model file is not corrupted

## Next Steps
- [Tutorial 3: Engine Comparison](03_engine_comparison.md) - Run the same simulation across multiple engines
- [Tutorial 4: Video Analysis](04_video_analysis.md) - Extract poses from video and map to simulation
- [API Reference](../../api/physics_engine.md) - Complete PhysicsEngine interface documentation
