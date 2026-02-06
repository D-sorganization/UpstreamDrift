# Tutorial 3: Engine Comparison and Cross-Validation

**Estimated Time:** 60 minutes
**Difficulty:** Intermediate

## Prerequisites

- Completed [Tutorial 2: Your First Simulation](02_first_simulation.md)
- At least two physics engines installed (MuJoCo + one of Drake/Pinocchio)
- Understanding of basic rigid body dynamics

## Learning Objectives

By the end of this tutorial, you will:

- Understand the strengths of each physics engine
- Run identical simulations across multiple engines
- Use the cross-engine validator to ensure consistency
- Interpret validation results and tolerance thresholds

## Overview

The Golf Modeling Suite supports 6 physics engines through a unified interface. This tutorial shows how to run the same simulation across different engines and validate that results are consistent within specified tolerances.

## Step 1: Understanding Engine Characteristics

### Engine Comparison Matrix

| Feature             | MuJoCo            | Drake             | Pinocchio         | OpenSim      | MyoSuite          |
| ------------------- | ----------------- | ----------------- | ----------------- | ------------ | ----------------- |
| **Primary Use**     | Production        | Optimization      | Fast kinematics   | Biomechanics | RL training       |
| **Contact Physics** | Excellent         | Good              | Basic             | Basic        | Excellent         |
| **Muscle Models**   | Built-in          | Manual            | No                | Excellent    | Excellent         |
| **Speed**           | Fast              | Medium            | Very Fast         | Slow         | Medium            |
| **Model Formats**   | XML, MJCF         | URDF, SDF         | URDF              | .osim        | MuJoCo XML        |
| **Jacobians**       | [Linear; Angular] | [Angular; Linear] | [Linear; Angular] | Limited      | [Linear; Angular] |

### When to Use Each Engine

**MuJoCo** - Default choice for:

- Golf swing simulation with contact
- Flexible shaft modeling (Euler-Bernoulli beam)
- Counterfactual motion analysis
- High-fidelity biomechanics

**Drake** - Preferred for:

- Trajectory optimization problems
- Control system design
- Multi-body dynamics with constraints
- Robotics integration

**Pinocchio** - Best for:

- Rapid prototyping
- Inverse kinematics/dynamics
- When speed is critical
- URDF-based workflows

**OpenSim** - Ideal for:

- Clinical biomechanics research
- Muscle-driven simulations
- Existing .osim model libraries
- Comparative studies with published research

**MyoSuite** - Use for:

- Reinforcement learning training
- Muscle activation optimization
- Gym-compatible environments
- Grip force modeling

## Step 2: Multi-Engine Simulation Setup

Create a multi-engine comparison script:

```python
"""Cross-engine simulation comparison."""

from pathlib import Path
import numpy as np
from typing import Optional

from src.shared.python.engine_manager import EngineManager, EngineType


def run_simulation_on_engine(
    engine_manager: EngineManager,
    engine_type: EngineType,
    model_path: Path,
    duration: float = 1.0,
    timestep: float = 0.001,
    initial_state: Optional[tuple] = None,
) -> dict:
    """Run a simulation on a specific engine.

    Args:
        engine_manager: Initialized EngineManager
        engine_type: Which engine to use
        model_path: Path to model file
        duration: Simulation duration in seconds
        timestep: Integration timestep
        initial_state: Optional (q0, v0) tuple for initial conditions

    Returns:
        Dictionary with times, positions, velocities
    """
    # Switch to the requested engine
    try:
        engine_manager.switch_engine(engine_type)
    except Exception as e:
        print(f"Engine {engine_type.name} not available: {e}")
        return None

    # Get physics engine instance
    physics_engine = engine_manager.get_active_physics_engine()

    # Load model
    physics_engine.load_from_path(str(model_path))
    physics_engine.reset()

    # Set initial conditions if provided
    if initial_state is not None:
        q0, v0 = initial_state
        physics_engine.set_state(q0, v0)

    # Run simulation
    num_steps = int(duration / timestep)
    times = np.zeros(num_steps)
    positions = []
    velocities = []

    for step in range(num_steps):
        times[step] = physics_engine.get_time()
        q, v = physics_engine.get_state()
        positions.append(q.copy())
        velocities.append(v.copy())
        physics_engine.step(timestep)

    return {
        "engine": engine_type.name,
        "times": times,
        "positions": np.array(positions),
        "velocities": np.array(velocities),
    }


def compare_engines(
    engine_types: list,
    model_path: Path,
    duration: float = 1.0,
) -> dict:
    """Run identical simulation across multiple engines.

    Args:
        engine_types: List of EngineType values to compare
        model_path: Path to shared model file
        duration: Simulation duration

    Returns:
        Dictionary mapping engine names to results
    """
    project_root = Path(__file__).parent.parent
    engine_manager = EngineManager(project_root)

    # Get initial state from first engine
    engine_manager.switch_engine(engine_types[0])
    first_engine = engine_manager.get_active_physics_engine()
    first_engine.load_from_path(str(model_path))
    initial_state = first_engine.get_state()

    results = {}
    for engine_type in engine_types:
        print(f"\nRunning simulation on {engine_type.name}...")
        result = run_simulation_on_engine(
            engine_manager,
            engine_type,
            model_path,
            duration=duration,
            initial_state=initial_state,
        )
        if result is not None:
            results[engine_type.name] = result

    return results
```

## Step 3: Using the Cross-Engine Validator

The suite includes a built-in validator for comparing engine outputs:

```python
from src.shared.python.cross_engine_validator import CrossEngineValidator

def validate_results(results: dict) -> dict:
    """Validate results across engines using the CrossEngineValidator.

    Args:
        results: Dictionary mapping engine names to simulation results

    Returns:
        Dictionary of validation results
    """
    validator = CrossEngineValidator()
    validation_results = {}

    engine_names = list(results.keys())

    # Compare each pair of engines
    for i, engine_a in enumerate(engine_names):
        for engine_b in engine_names[i + 1:]:
            pair_key = f"{engine_a}_vs_{engine_b}"
            print(f"\nValidating {pair_key}...")

            res_a = results[engine_a]
            res_b = results[engine_b]

            # Compare positions
            pos_result = validator.compare_states(
                engine_a, res_a["positions"],
                engine_b, res_b["positions"],
                metric="position",
            )

            # Compare velocities
            vel_result = validator.compare_states(
                engine_a, res_a["velocities"],
                engine_b, res_b["velocities"],
                metric="velocity",
            )

            validation_results[pair_key] = {
                "position": {
                    "passed": pos_result.passed,
                    "max_deviation": pos_result.max_deviation,
                    "tolerance": pos_result.tolerance,
                    "severity": pos_result.severity,
                },
                "velocity": {
                    "passed": vel_result.passed,
                    "max_deviation": vel_result.max_deviation,
                    "tolerance": vel_result.tolerance,
                    "severity": vel_result.severity,
                },
            }

            # Print summary
            print(f"  Position: {pos_result.severity} (max dev: {pos_result.max_deviation:.2e})")
            print(f"  Velocity: {vel_result.severity} (max dev: {vel_result.max_deviation:.2e})")

    return validation_results
```

## Step 4: Understanding Tolerance Thresholds

The cross-engine validator uses the following tolerances (defined in Guideline P3):

| Metric       | Tolerance        | Units  |
| ------------ | ---------------- | ------ |
| Position     | 1e-6             | meters |
| Velocity     | 1e-5             | m/s    |
| Acceleration | 1e-4             | m/s^2  |
| Torque       | 1e-3 or <10% RMS | N\*m   |
| Jacobian     | 1e-8             | -      |

### Severity Classification

| Severity    | Deviation Range | Interpretation          |
| ----------- | --------------- | ----------------------- |
| **PASSED**  | <= 1x tolerance | Results are consistent  |
| **WARNING** | 1-2x tolerance  | Acceptable with caution |
| **ERROR**   | 2-10x tolerance | Investigation required  |
| **BLOCKER** | >100x tolerance | Fundamental model error |

## Step 5: Jacobian Convention Differences

Different engines use different Jacobian conventions. The suite handles this automatically, but it's important to understand:

```python
def demonstrate_jacobian_conventions(engine_manager, model_path):
    """Show Jacobian convention differences between engines."""

    # MuJoCo convention: [Linear (3x n); Angular (3x n)]
    engine_manager.switch_engine(EngineType.MUJOCO)
    mujoco_engine = engine_manager.get_active_physics_engine()
    mujoco_engine.load_from_path(str(model_path))

    J_mujoco = mujoco_engine.compute_jacobian("torso")
    print(f"MuJoCo Jacobian shape: {J_mujoco.shape}")
    print("Convention: [Linear; Angular]")

    # Drake convention: [Angular (3x n); Linear (3x n)]
    try:
        engine_manager.switch_engine(EngineType.DRAKE)
        drake_engine = engine_manager.get_active_physics_engine()
        drake_engine.load_from_path(str(model_path))

        J_drake = drake_engine.compute_jacobian("torso")
        print(f"\nDrake Jacobian shape: {J_drake.shape}")
        print("Convention: [Angular; Linear]")

        # To compare, reorder Drake's Jacobian
        J_drake_reordered = np.vstack([J_drake[3:], J_drake[:3]])
        print("\nReordered for comparison")
    except Exception as e:
        print(f"Drake not available: {e}")
```

## Step 6: Complete Comparison Example

```python
"""Complete multi-engine comparison example."""

from pathlib import Path
import numpy as np

from src.shared.python.engine_manager import EngineManager, EngineType
from src.shared.python.cross_engine_validator import CrossEngineValidator


def main():
    project_root = Path(__file__).parent.parent

    # Use a simple model compatible with multiple engines
    model_path = project_root / "shared/models/urdf/double_pendulum.urdf"

    # Engines to compare
    engines_to_test = [
        EngineType.MUJOCO,
        EngineType.PINOCCHIO,
        # EngineType.DRAKE,  # Uncomment if installed
    ]

    # Check availability
    engine_manager = EngineManager(project_root)
    available = engine_manager.get_available_engines()
    print(f"Available engines: {[e.name for e in available]}")

    # Filter to only available engines
    engines_to_test = [e for e in engines_to_test if e in available]

    if len(engines_to_test) < 2:
        print("Need at least 2 engines for comparison. Install more engines.")
        return

    # Run simulations
    results = compare_engines(engines_to_test, model_path, duration=1.0)

    # Validate
    validation = validate_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for pair, metrics in validation.items():
        print(f"\n{pair}:")
        for metric, data in metrics.items():
            status = "PASS" if data["passed"] else "FAIL"
            all_passed = all_passed and data["passed"]
            print(f"  {metric}: {status} ({data['severity']})")

    print("\n" + "=" * 60)
    if all_passed:
        print("All validations PASSED - engines are consistent")
    else:
        print("Some validations FAILED - investigate discrepancies")


if __name__ == "__main__":
    main()
```

## Step 7: Debugging Validation Failures

When validation fails, use these debugging steps:

### 1. Check Model Compatibility

```python
# Ensure models are equivalent
def check_model_compatibility(engine_a, engine_b):
    """Verify models have same DOF and structure."""
    q_a, v_a = engine_a.get_state()
    q_b, v_b = engine_b.get_state()

    print(f"Engine A: {len(q_a)} DOF")
    print(f"Engine B: {len(q_b)} DOF")

    assert len(q_a) == len(q_b), "DOF mismatch!"
```

### 2. Compare Mass Matrices

```python
def compare_mass_matrices(engine_a, engine_b, tol=1e-6):
    """Compare inertia matrices."""
    M_a = engine_a.compute_mass_matrix()
    M_b = engine_b.compute_mass_matrix()

    diff = np.abs(M_a - M_b)
    max_diff = diff.max()

    print(f"Mass matrix max difference: {max_diff:.2e}")
    if max_diff > tol:
        print("Mass matrices differ - check model parameters")
```

### 3. Isolate Time of Divergence

```python
def find_divergence_point(results_a, results_b, tol=1e-4):
    """Find when trajectories start diverging."""
    for i, (pa, pb) in enumerate(zip(results_a["positions"], results_b["positions"])):
        diff = np.linalg.norm(pa - pb)
        if diff > tol:
            t = results_a["times"][i]
            print(f"Trajectories diverge at t={t:.4f}s (step {i})")
            return i
    print("Trajectories remain consistent")
    return -1
```

## Troubleshooting

### Different Number of DOF

Ensure both engines load the same model definition. Some engines may add implicit DOF (e.g., for contacts).

### Numerical Precision Differences

Try smaller timesteps. Different integrators accumulate errors differently.

### Jacobian Sign Conventions

Some engines use body-fixed vs. world-fixed frames. The suite normalizes these, but custom code may need adjustment.

## Next Steps

- [Tutorial 4: Video Analysis](04_video_analysis.md) - Extract poses from video
- [API Reference: CrossEngineValidator](../../api/cross_engine_validator.md)
- [Guideline P3: Tolerance Specifications](../../guidelines/P3_tolerances.md)
