# Motion Pipeline — Troubleshooting Guide

> Common failure modes and how to fix them. Part of epic #4558.

---

## Unit Conversion Errors (mm vs m)

### Symptom
```
ValueError: Marker positions exceed plausible human range
  Got max distance 1500.0 m between markers
```

### Cause
Your mocap data is in millimeters but the pipeline expects meters.

### Fix
```python
from src.shared.python.motion_pipeline.converters import UnitConverter

converter = UnitConverter()
motion = converter.load("your_data.trc")

# Check current units
print(f"Current units: {motion.units}")  # Likely 'mm'

# Convert to meters
motion.to_meters()  # or motion.scale(0.001)
```

### Prevention
Always specify units when loading:
```python
motion = load_mocap("data.trc", units="mm")  # Auto-converts to meters
```

---

## Marker Occlusion Patterns

### Symptom
```
Warning: 47% of markers have zero confidence in frames 120-150
IK solver failed: insufficient constraints
```

### Cause
Markers were occluded during capture (e.g., club passing in front of body).

### Fix Options

**Option 1: Interpolate missing markers**
```python
from src.shared.python.motion_pipeline.cleaning import interpolate_gaps

motion = load_mocap("data.json")
motion = interpolate_gaps(motion, max_gap=10)  # Interpolate up to 10 frames
```

**Option 2: Use virtual markers**
```python
from src.shared.python.motion_pipeline.cleaning import create_virtual_markers

# Create virtual markers from existing ones
motion = create_virtual_markers(motion, 
    virtual_marker_name="CLUB_HEAD",
    source_markers=["WRIST_R", "ELBOW_R"],
    method="extrapolate"
)
```

**Option 3: Reduce IK weights for missing markers**
```python
ik_config = IKConfig(
    marker_weights={"default": 1.0, "occluded": 0.1}
)
```

---

## IK Convergence Failures

### Symptom
```
IK solver did not converge after 100 iterations
  Final error: 0.156 m (target: 0.01 m)
```

### Cause
The solver cannot find a pose that satisfies all constraints.

### Fix Options

**Option 1: Adjust cost weights**
```python
from src.shared.python.motion_pipeline import IKConfig

config = IKConfig(
    position_weight=1.0,
    orientation_weight=0.5,  # Reduce orientation priority
    regularization_weight=0.1,  # Add smoothness prior
    initial_guess="previous_frame"  # Warm start
)
```

**Option 2: Increase iterations**
```python
config = IKConfig(max_iterations=500, tolerance=0.05)  # Relax tolerance
```

**Option 3: Use multi-stage solving**
```python
# Stage 1: Coarse solve (root + pelvis only)
coarse_config = IKConfig(free_joints=["pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt"])
coarse_motion = ik_solver.solve(motion, coarse_config)

# Stage 2: Fine solve (all joints)
fine_config = IKConfig(free_joints="all")
final_motion = ik_solver.solve(coarse_motion, fine_config)
```

---

## Residual Reduction Reports

### How to Read Residual Reports

After running RRA (Residual Reduction Algorithm):

```
=== Residual Reduction Analysis ===
Initial residuals:
  FX: 45.2 N   FY: 12.3 N   FZ: 78.9 N
  MX: 12.4 Nm  MY: 8.7 Nm   MZ: 15.2 Nm

After RRA:
  FX: 2.1 N    FY: 1.8 N    FZ: 3.4 N
  MX: 0.8 Nm   MY: 0.5 Nm   MZ: 1.2 Nm

Reduction: 95.3%
```

**Acceptable thresholds:**
- Forces: < 5% of body weight
- Moments: < 1% of body weight × height

**If residuals remain high:**

```python
from src.shared.python.motion_pipeline import RRAConfig

rra_config = RRAConfig(
    mass_adjustment=True,  # Allow mass scaling
    com_adjustment=True,   # Allow center of mass adjustment
    residual_weights={"force": 1.0, "moment": 0.5}
)

corrected_motion = rra_solver.solve(motion, rra_config)
```

---

## Engine-Specific Issues

### MuJoCo

**Symptom:** `mujoco.mujoco.MjDataError: contact set is full`

**Fix:** Increase contact buffer:
```python
model = mujoco.MjModel("your_model.xml")
model.nconmax = 500  # Default is 50
```

### Drake

**Symptom:** `RuntimeError: MultibodyPlant: geometric queries require a SceneGraph`

**Fix:** Ensure SceneGraph is connected:
```python
from pydrake.systems.framework import DiagramBuilder

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
```

### Pinocchio

**Symptom:** `AttributeError: 'pinocchio.RobotWrapper' object has no attribute 'computeTotalEnergy'`

**Fix:** Use separate energy computations:
```python
kinetic = robot.kineticEnergy(q, v)
potential = robot.potentialEnergy(q)
total = kinetic + potential  # NOT robot.computeTotalEnergy()
```

---

## Quick Diagnostic Script

Run this to diagnose common issues:

```bash
python3 scripts/motion_pipeline/diagnose.py --input your_motion.json
```

Output:
```
=== Motion Diagnostic Report ===
File: your_motion.json
Frames: 300
Markers: 25

Issues found:
  ⚠ 12 frames with occluded markers (frames 45-56)
  ⚠ Unit mismatch: data in mm, expected m
  ✓ Temporal continuity: OK
  ✓ Marker naming convention: OK

Recommended actions:
  1. Run: motion.to_meters()
  2. Run: interpolate_gaps(motion, max_gap=15)
```

---

## Related Documents

- [User Workflow Guide](README.md) — How to run the pipeline
- [Format Matrix](formats.md) — Mocap format support
- [ADR 0007](../adr/0007-motion-pipeline-architecture.md) — Architecture decisions