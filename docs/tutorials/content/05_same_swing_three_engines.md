# Tutorial 5: Same Swing in Three Engines

**Estimated Time:** 60 minutes
**Difficulty:** Intermediate

## Prerequisites

- Completed [Tutorial 3: Engine Comparison](03_engine_comparison.md)
- MuJoCo and Pinocchio installed
- JaxSim installed from the pinned optional extra if you want to run the
  differentiable backend locally
- A single exported swing trajectory or the bundled golfer URDF sample

## Learning Objectives

By the end of this tutorial, you will:

- Run one swing input through three engine paths without changing the source data
- Record each engine's position, velocity, and units convention before comparison
- Normalize outputs into the suite convention before checking tolerances
- Decide whether a mismatch is a modeling issue, a units issue, or an engine gap

## Why This Tutorial Exists

The same golf swing can look different across engines when the raw outputs use
different state layouts, velocity conventions, or unit metadata. The comparison
workflow is intentionally explicit: capture the native result, document the
native convention, normalize once, and only then compare.

Use this tutorial as the capstone checklist for a three-engine learner workflow.
It is safe to complete with two engines when JaxSim is unavailable; in that case,
record the skip reason instead of changing the input data.

## Engine Paths

| Path              | Role                           | Native model format | Velocity convention                                 | Units to confirm         |
| ----------------- | ------------------------------ | ------------------- | --------------------------------------------------- | ------------------------ |
| MuJoCo            | Contact-rich reference run     | MJCF/XML            | Linear then angular where exposed by MuJoCo helpers | meters, radians, seconds |
| Pinocchio         | Fast kinematics/dynamics check | URDF                | Linear/angular ordering depends on API call         | meters, radians, seconds |
| JaxSim-compatible | Differentiable gradient check  | URDF/JaxSim model   | JaxSim-native generalized velocity layout           | meters, radians, seconds |

## Step 1: Pick One Input

Choose exactly one input trajectory and keep it unchanged for all engines.

Recommended starter inputs:

- A short golfer swing exported from Tutorial 2.
- A normalized pose-interchange swing saved from Pose Studio.
- A full-body motion-matching target only after the simple sample passes.

Record these values before running any engine:

| Field                | Value |
| -------------------- | ----- |
| Source file          |       |
| Sampling rate        |       |
| Position units       |       |
| Angle units          |       |
| Initial state source |       |

## Step 2: Run the Reference Engine

Start with MuJoCo when it is available because it is the usual reference path for
contact-rich golf simulations.

The example below points at the checked-in golfer URDF used by the Pinocchio
path. If your MuJoCo installation requires MJCF input, use the equivalent
exported MJCF while keeping the same initial state and trajectory source.

```python
from pathlib import Path

from src.shared.python.engine_core.engine_manager import EngineManager
from src.shared.python.engine_core.engine_registry import EngineType


project_root = Path(__file__).resolve().parents[2]
model_path = (
    project_root
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "generated"
    / "golfer.urdf"
)

manager = EngineManager(project_root)
manager.switch_engine(EngineType.MUJOCO)
engine = manager.get_active_physics_engine()
engine.load_from_path(str(model_path))
engine.reset()

q0, v0 = engine.get_state()
print({"engine": "mujoco", "q_dof": len(q0), "v_dof": len(v0)})
```

Save the initial state from the first successful engine and reuse it for every
subsequent engine. Do not let each backend choose its own default state.

## Step 3: Run the Fast Dynamics Check

Run Pinocchio with the same model and initial state. If the model loader reports
missing joints or a different degree-of-freedom count, stop and fix the model
mapping before comparing trajectories.

```python
manager.switch_engine(EngineType.PINOCCHIO)
engine = manager.get_active_physics_engine()
engine.load_from_path(str(model_path))
engine.set_state(q0, v0)

q_pin, v_pin = engine.get_state()
print({"engine": "pinocchio", "q_dof": len(q_pin), "v_dof": len(v_pin)})
```

## Step 4: Run the Differentiable Path

When JaxSim is installed, run the JaxSim-compatible path with the same initial
state. If it is not installed, mark the run as skipped and keep the MuJoCo and
Pinocchio comparison results.

Install JaxSim through the repository extra so the version matches the upgrade
guard:

```bash
python3 -m pip install -e ".[jaxsim]"
```

```python
try:
    jaxsim_engine_type = EngineType("jaxsim")
except ValueError:
    print("JaxSim engine type is not registered in this checkout")
else:
    manager.switch_engine(jaxsim_engine_type)
    engine = manager.get_active_physics_engine()
    engine.load_from_path(str(model_path))
    engine.set_state(q0, v0)
    q_jaxsim, v_jaxsim = engine.get_state()
    print({"engine": "jaxsim", "q_dof": len(q_jaxsim), "v_dof": len(v_jaxsim)})
```

## Step 5: Normalize Before Comparing

Before calculating errors, write down each native convention and normalize to the
suite comparison convention.

| Engine            | Native position layout | Native velocity layout | Normalized? |
| ----------------- | ---------------------- | ---------------------- | ----------- |
| MuJoCo            |                        |                        |             |
| Pinocchio         |                        |                        |             |
| JaxSim-compatible |                        |                        |             |

For trajectory comparison, use the same tolerances as Tutorial 3:

| Metric       | Starter tolerance | Units                                           |
| ------------ | ----------------- | ----------------------------------------------- |
| Position     | 1e-6              | meters                                          |
| Velocity     | 1e-5              | meters/second or radians/second                 |
| Acceleration | 1e-4              | meters/second squared or radians/second squared |

If raw results disagree but normalized results agree, the engines are consistent
and the discrepancy was convention metadata. If normalized results disagree,
check model parameters, initial state order, and integration timestep.

## Step 6: Record the Result

Use this summary table in notes or PR descriptions:

| Check                                       | Result |
| ------------------------------------------- | ------ |
| Same source trajectory used for all engines |        |
| Same initial state reused                   |        |
| Degree-of-freedom counts match              |        |
| Units documented per engine                 |        |
| Velocity convention documented per engine   |        |
| Normalized position tolerance passed        |        |
| Normalized velocity tolerance passed        |        |
| JaxSim skip reason, if skipped              |        |

## Troubleshooting

### JaxSim Is Not Installed

Keep the two-engine comparison and record the missing optional dependency. Do not
replace it with a different source trajectory.

### Degree-of-Freedom Counts Differ

Compare the model conversion step first. A trajectory comparison is not valid
until each engine is using the same joint set and initial state ordering.

### Positions Match but Velocities Do Not

Check whether one API reports spatial velocity as angular then linear while
another reports linear then angular. Normalize the layout before treating the
result as a physics mismatch.

## Step 7 (Optional): Roll Out a Forward Simulation

Each engine adapter exposes a canonical `rollout(controls, horizon, dt)` that
returns the shared `Trace` schema (`t`, `q`, `v`, `u`, `meta`). For the JaxSim
path this is validated against an analytic torque-free body in
`tests/cross_engine/test_jaxsim_forward_sim.py` (issue #6655):

```python
# JaxSim path (Linux only); skips cleanly when jaxsim is unavailable.
from src.engines.physics_engines.jaxsim import JaxSimBackend

backend = JaxSimBackend()
backend.load_from_path(str(model_path))
backend.set_state(q0, v0)
trace = backend.rollout(controls=None, horizon=100, dt=1e-3)
print({"backend": trace.backend, "steps": trace.num_steps, "nv": trace.meta["nv"]})
```

Compare `trace.q`/`trace.v` against the MuJoCo and Pinocchio traces only after
normalizing each to the suite canonical `[angular; linear]` inertial convention
(Step 5).

## Next Steps

- The JaxSim upgrade-guard from issue #6660 now ships as
  `scripts/jaxsim/check_jaxsim_pin.py` and runs as a CI step in
  `cross-engine-equivalence.yml`; run it locally before bumping the pin.
- Extend this tutorial with a checked-in, runnable sample once the JaxSim backend
  API is stable in the repository.
- Read the [JaxSim version policy](../../development/jaxsim_version_policy.md)
  before changing the optional dependency pin.
- Use [Tutorial 4: Video Analysis](04_video_analysis.md) to generate a real
  swing trajectory for the same workflow.
