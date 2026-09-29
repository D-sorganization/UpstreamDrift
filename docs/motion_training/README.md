# Motion Training from Club Trajectory Data

## Overview

This module implements inverse kinematics (IK) based motion training for golf swing analysis. The system uses captured club trajectory data to generate humanoid body configurations that satisfy the constraint of maintaining both hands on the club grip while the club follows a prescribed motion path.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Motion Training Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ Club Trajectory │───▶│  IK Solver      │───▶│ Joint Configurations    │  │
│  │ Data Parser     │    │  (Pink/Pinocchio)│    │ (q trajectory)          │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│         │                       │                         │                  │
│         ▼                       ▼                         ▼                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ Excel/CSV       │    │ Dual End-Effector│   │ Visualization           │  │
│  │ Motion Capture  │    │ Task (hands)    │    │ (Meshcat/Matplotlib)    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Club Trajectory Data Format

The motion capture data is stored in Excel format with the following structure:

| Field                | Description                   | Units       |
| -------------------- | ----------------------------- | ----------- |
| Sample #             | Frame index                   | integer     |
| Time                 | Time relative to impact (T=0) | seconds     |
| Mid-hands (X,Y,Z)    | Grip center position          | centimeters |
| Mid-hands (Xx,Xy,Xz) | Grip X-axis orientation       | unit vector |
| Mid-hands (Yx,Yy,Yz) | Grip Y-axis orientation       | unit vector |
| Club face (X,Y,Z)    | Club face center position     | centimeters |
| Club face (Xx,Xy,Xz) | Club face X-axis orientation  | unit vector |
| Club face (Yx,Yy,Yz) | Club face Y-axis orientation  | unit vector |

**Event Markers:**

- **A (Address)**: Starting position
- **T (Top)**: Top of backswing
- **I (Impact)**: Ball contact
- **F (Finish)**: End of follow-through

### 2. Inverse Kinematics Approach

We use **Pink** (a task-based IK solver for Pinocchio) with the following task formulation:

#### Primary Tasks: Hand Placement

- **Left Hand Task**: SE(3) frame task tracking the left grip position
- **Right Hand Task**: SE(3) frame task tracking the right grip position

The hand positions are computed from the mid-hands grip center plus offsets along the local coordinate axes:

```
left_hand_pos  = mid_hands_pos + offset_left  @ local_rotation
right_hand_pos = mid_hands_pos + offset_right @ local_rotation
```

Where offsets are typically:

- Left hand: +3-4 cm along grip axis (bottom hand for right-handed golfer)
- Right hand: -3-4 cm along grip axis (top hand for right-handed golfer)

#### Secondary Tasks: Posture Regularization

- **Posture Task**: Low-weight regularization toward a reference pose
- **Joint Limit Task**: Soft constraints to keep joints within limits

### 3. Humanoid Model Requirements

The golfer model must have:

- **Pelvis as root** (can be fixed or floating base)
- **Full arm kinematic chains**: Shoulder (3DOF) → Elbow (2DOF) → Wrist (2DOF) → Hand
- **Spine articulation**: For torso rotation during swing
- **Leg chains**: For weight transfer and balance

Key frames (end-effectors):

- `hand_left` - Left hand link
- `hand_right` - Right hand link

### 4. Coordinate System Conventions

```
Global Coordinate System (GCS):
  X: Forward (toward target)
  Y: Left (lateral)
  Z: Up (vertical)

Local Club Coordinate System:
  X: Along shaft (grip to head)
  Y: Perpendicular to face
  Z: Tangent to face (toe direction)
```

## Implementation Details

### File Structure

```
src/engines/physics_engines/pinocchio/python/motion_training/
├── __init__.py
├── club_trajectory_parser.py     # Parse Excel/CSV motion data
├── dual_hand_ik_solver.py        # IK solver with hand constraints
├── motion_visualizer.py          # Meshcat visualization
├── training_pipeline.py          # Main training orchestration
└── trajectory_exporter.py        # Export to MuJoCo/Drake formats
```

### Usage Example

```python
from motion_training.club_trajectory_parser import ClubTrajectoryParser
from motion_training.dual_hand_ik_solver import DualHandIKSolver
from motion_training.motion_visualizer import MotionVisualizer

# 1. Parse club trajectory
parser = ClubTrajectoryParser("data/Wiffle_ProV1_club_3D_data.xlsx")
trajectory = parser.parse(sheet_name="TW_wiffle")

# 2. Initialize IK solver with golfer model
solver = DualHandIKSolver(
    urdf_path="models/generated/golfer_ik.urdf",
    left_hand_frame="hand_left",
    right_hand_frame="hand_right",
)

# 3. Generate body configurations
configurations = solver.solve_trajectory(trajectory)

# 4. Visualize
viz = MotionVisualizer()
viz.show_motion(configurations, trajectory)
```

### Algorithm: Differential IK Loop

```
Input: Club trajectory T = {(t_i, pose_i)} for i = 0..N
Output: Joint trajectory Q = {q_i}

1. Initialize q_0 to reference pose (address position)
2. For each frame i = 1 to N:
   a. Compute left_hand_target  = transform_to_left_hand(pose_i)
   b. Compute right_hand_target = transform_to_right_hand(pose_i)
   c. Set task targets
   d. Solve IK: v = solve_ik(q_{i-1}, tasks, dt)
   e. Integrate: q_i = integrate(q_{i-1}, v * dt)
   f. Check convergence and constraints
3. Return Q
```

## Configuration Parameters

### Solver Settings

| Parameter               | Default    | Description                       |
| ----------------------- | ---------- | --------------------------------- |
| `solver`                | "quadprog" | QP solver backend                 |
| `damping`               | 1e-6       | Levenberg-Marquardt damping       |
| `dt`                    | 0.01       | Integration timestep              |
| `max_iterations`        | 100        | Max IK iterations per frame       |
| `position_tolerance`    | 1e-3       | Position error tolerance (m)      |
| `orientation_tolerance` | 1e-2       | Orientation error tolerance (rad) |

### Task Weights

| Task                   | Weight | Description                        |
| ---------------------- | ------ | ---------------------------------- |
| Left hand position     | 10.0   | High priority hand tracking        |
| Left hand orientation  | 5.0    | Moderate orientation tracking      |
| Right hand position    | 10.0   | High priority hand tracking        |
| Right hand orientation | 5.0    | Moderate orientation tracking      |
| Posture regularization | 1e-3   | Soft preference for reference pose |

### Hand Grip Offsets

For a standard golf grip (right-handed golfer):

```python
LEFT_HAND_OFFSET = [0.0, 0.0, 0.04]   # 4cm below grip center (lead hand)
RIGHT_HAND_OFFSET = [0.0, 0.0, -0.04]  # 4cm above grip center (trail hand)
```

## Visualization

The visualizer shows:

1. **Club trajectory path**: As a line/ribbon through space
2. **Club at each frame**: Semi-transparent club model
3. **Humanoid model**: Full body following the motion
4. **Hand targets**: Spheres showing IK target positions
5. **Frame axes**: Local coordinate systems

## Exporting Trajectories

### For MuJoCo

```python
exporter = TrajectoryExporter(format="mujoco")
exporter.export(configurations, "motion_data/swing_trajectory.json")
```

Output format:

```json
{
  "model": "golfer.xml",
  "timestep": 0.004166,
  "frames": [
    {"time": 0.0, "qpos": [...], "qvel": [...]},
    ...
  ]
}
```

### For Drake

```python
exporter = TrajectoryExporter(format="drake")
exporter.export(configurations, "motion_data/swing_trajectory.yaml")
```

## Troubleshooting

### Common Issues

1. **IK fails to converge**

   - Increase max iterations
   - Check if target is reachable (within arm length)
   - Reduce target velocity (smaller dt)

2. **Joint limits violated**

   - Add joint limit tasks with higher weight
   - Check model joint limit definitions

3. **Discontinuous motion**

   - Use smaller dt steps
   - Add velocity smoothing
   - Check for singularities

4. **Hands separate from club**
   - Increase hand task weights
   - Verify grip offset parameters
   - Check coordinate system alignment

## Future Extensions

1. **Collision Avoidance**: Add self-collision constraints
2. **Dynamic Feasibility**: Verify torque limits in dynamics
3. **Optimization**: Trajectory optimization for smoother motion
4. **Multi-target**: Track additional body points (head, hip)
5. **Learning**: Train neural network policy from IK trajectories

## References

- Pink: Task-based inverse kinematics for Pinocchio
- Pinocchio: Rigid body dynamics library
- Motion Capture Data: GEARS/Trackman format
