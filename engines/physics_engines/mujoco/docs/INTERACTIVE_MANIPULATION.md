# Interactive Drag-and-Pose Manipulation Guide

## Overview

The MuJoCo Golf Swing Model now includes a comprehensive interactive manipulation system that allows you to:

- **Click and drag** body parts to position them in 3D space
- **Fix segments** in space or relative to other components
- **Save and load poses** for reuse
- **Interpolate between poses** for smooth transitions
- **Fine-tune IK solver** parameters for optimal performance

## Quick Start

1. **Launch the Application**:
   ```bash
   cd python
   python3 -m mujoco_golf_pendulum.advanced_gui
   ```

2. **Navigate to the "Interactive Pose" Tab** in the right panel

3. **Start Dragging**:
   - Simply **click and drag** any body part in the 3D view
   - The body will follow your mouse using inverse kinematics
   - Release the mouse to stop dragging

4. **Zoom Camera**:
   - Use **mouse wheel** to zoom in/out

## Features

### 1. Drag Mode Controls

#### Enable/Disable Dragging
- **Enable Drag Manipulation**: Toggle to enable/disable click-and-drag functionality
- **Default**: Enabled

#### Maintain Orientation
- **Maintain Orientation While Dragging**: When enabled, the body maintains its original orientation during dragging
- **Use case**: Useful when you want to translate a body without rotating it
- **Default**: Disabled

#### Nullspace Posture Optimization
- **Use Nullspace Posture Optimization**: Automatically returns joints to original configuration when possible
- **Use case**: Helps maintain natural-looking poses
- **Default**: Enabled

### 2. Body Constraints

Constraints allow you to fix body segments in space or relative to other components. This is extremely useful for investigating model motion with certain parts held fixed.

#### Fixed in Space
1. Select a **Body** from the dropdown
2. Choose **"Fixed in Space"** as the constraint type
3. Click **"Add Constraint"**
4. The selected body will now remain at its current position during simulation or dragging

#### Relative to Body
1. Select a **Body** from the dropdown
2. Choose **"Relative to Body"** as the constraint type
3. Select a **Reference** body
4. Click **"Add Constraint"**
5. The selected body will maintain its current relative position to the reference body

#### Managing Constraints
- **Remove Constraint**: Select a body and click "Remove Constraint"
- **Clear All Constraints**: Click "Clear All Constraints" to remove all active constraints
- **Active Constraints List**: See all currently constrained bodies in the list

#### Visual Feedback
- **Selected bodies**: Highlighted with a **cyan circle**
- **Constrained bodies**: Marked with a **magenta square** and "FIXED" label
- *Note: Visual overlays require OpenCV (opencv-python). If not installed, functionality works without visuals.*

### 3. Pose Library

The pose library allows you to save, load, and manage different model configurations.

#### Saving Poses
1. Position the model as desired using drag manipulation
2. Enter a **name** in the "Pose name..." field
3. Click **"Save Pose"**
4. The pose is now saved to the library

#### Loading Poses
1. Select a pose from the **Pose List**
2. Click **"Load"**
3. The model will instantly assume the saved configuration

#### Deleting Poses
1. Select a pose from the **Pose List**
2. Click **"Delete"**
3. Confirm deletion

#### Export/Import Pose Libraries
- **Export Library**: Save all poses to a JSON file for backup or sharing
- **Import Library**: Load poses from a previously exported JSON file

### 4. Pose Interpolation

Create smooth transitions between two saved poses.

#### How to Use
1. **Select exactly two poses** from the Pose List (Ctrl+Click to select multiple)
2. Move the **Blend slider** (0% = first pose, 100% = second pose)
3. The model will smoothly interpolate between the two configurations

#### Use Cases
- Creating animation keyframes
- Finding intermediate configurations
- Exploring the configuration space between known poses

### 5. IK Solver Settings (Advanced)

Fine-tune the inverse kinematics solver for optimal performance.

#### Damping (0.01 - 1.00)
- **What it does**: Controls numerical stability near singularities
- **Lower values** (0.01-0.05): More accurate, may be unstable
- **Higher values** (0.1-1.0): More stable, less accurate
- **Default**: 0.05
- **Recommendation**: Start with default, increase if you see jittery motion

#### Step Size (0.01 - 1.00)
- **What it does**: Controls how aggressively the solver moves toward the target
- **Lower values** (0.1-0.3): Smoother, slower convergence
- **Higher values** (0.5-1.0): Faster, may overshoot
- **Default**: 0.30
- **Recommendation**: Increase for faster response, decrease if motion is erratic

## Mouse Controls

| Action | Control |
|--------|---------|
| Select body | **Left click** |
| Drag body | **Left click + drag** |
| Zoom camera | **Mouse wheel** |
| Release selection | **Release left button** |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Play/Pause simulation |
| **R** | Reset simulation |

## Use Cases and Workflows

### Setting Initial Pose for Simulation

1. **Pause the simulation** (Space or Pause button)
2. **Drag body parts** to desired positions
3. *Optional*: **Save the pose** for later reuse
4. **Resume simulation** to see the motion from this configuration

### Investigating Constrained Motion

1. Select bodies to constrain (e.g., fix feet in space)
2. Add **"Fixed in Space"** constraints
3. Drag other body parts to see how the model responds with constraints
4. Useful for understanding kinematic chains and dependencies

### Creating Animation Keyframes

1. Create and **save multiple poses** representing key animation frames
2. Use **pose interpolation** to preview transitions
3. Export poses for use in animation software or motion planning

### Exploring Joint Limits

1. Enable drag mode
2. Try to move bodies to extreme positions
3. The IK solver respects joint limits automatically
4. Useful for understanding workspace and reachability

## Technical Details

### Inverse Kinematics (IK) Solver

- **Algorithm**: Damped Least-Squares (DLS)
- **Max iterations**: 20 per drag update
- **Tolerance**: 1e-3 (1mm position error)
- **Features**:
  - Singularity robust (damped)
  - Joint limit respecting
  - Nullspace optimization for posture maintenance

### Mouse Picking

- **Algorithm**: Ray-casting with bounding sphere intersection
- **Projection**: Perspective projection with 45° FOV
- **Selection radius**: 1.5x body bounding sphere (for easier selection)

### Constraints

- **Implementation**: IK-based constraint satisfaction
- **Types**:
  - Fixed in space: Maintains absolute world position
  - Relative to body: Maintains relative transform to reference

### Performance

- **Real-time updates**: 60 FPS simulation + drag updates
- **IK solver speed**: ~1ms per drag update (typical)
- **Recommended models**: Works with all models, best with <30 DOF

## Troubleshooting

### Problem: Body won't move when dragging

**Possible causes**:
1. Drag mode is disabled - Check "Enable Drag Manipulation"
2. Body is constrained - Check Active Constraints list
3. Joint limits preventing motion - Try different drag direction

### Problem: Dragging is jittery or unstable

**Solutions**:
1. Increase IK damping (0.1 or higher)
2. Decrease step size (0.1-0.2)
3. Disable nullspace posture optimization temporarily

### Problem: Selected body not highlighted

**Cause**: OpenCV not installed

**Solution**:
```bash
pip install opencv-python
```
*Note: Visual feedback is optional - drag functionality works without it*

### Problem: IK solver not converging

**Solutions**:
1. Increase max iterations (modify `ik_max_iterations` in code)
2. Increase damping for more stability
3. Try dragging smaller distances
4. Check if target is outside reachable workspace

## Advanced Features

### Programmatic Access

You can access the manipulator programmatically:

```python
from mujoco_golf_pendulum.advanced_gui import AdvancedGolfAnalysisWindow

# Get manipulator instance
app = AdvancedGolfAnalysisWindow()
manipulator = app.sim_widget.get_manipulator()

# Save current pose
manipulator.save_pose("my_pose", "Description here")

# Load pose
manipulator.load_pose("my_pose")

# Add constraint
from mujoco_golf_pendulum.interactive_manipulation import ConstraintType
manipulator.add_constraint(body_id=5, constraint_type=ConstraintType.FIXED_IN_SPACE)

# Export poses
manipulator.export_pose_library("poses.json")
```

### Custom IK Parameters

Modify solver parameters for specific use cases:

```python
manipulator = app.sim_widget.get_manipulator()

# High-accuracy, slow
manipulator.ik_damping = 0.01
manipulator.ik_max_iterations = 100
manipulator.ik_tolerance = 1e-5

# Fast, stable
manipulator.ik_damping = 0.1
manipulator.ik_max_iterations = 10
manipulator.ik_step_size = 0.8
```

## Model Compatibility

The interactive manipulation system works with all models:

| Model | DOF | Notes |
|-------|-----|-------|
| Double Pendulum | 2 | Very responsive |
| Triple Pendulum | 3 | Very responsive |
| Upper Body Golf | 10 | Responsive |
| Full Body Golf | 15 | Responsive |
| Advanced Biomech | 28 | Good performance |
| MyoUpperBody | 19 | Good (muscles passive) |
| MyoBody Full | 52 | Slower (complex) |

*Performance may vary based on hardware*

## Best Practices

1. **Start with simple models** (2-10 DOF) to learn the interface
2. **Pause simulation** before dragging for cleaner poses
3. **Save poses frequently** when exploring configurations
4. **Use constraints** to investigate specific motion patterns
5. **Adjust IK settings** based on model complexity
6. **Export pose libraries** for important model configurations

## Future Enhancements

Potential future additions:
- Trajectory recording during drag
- Collision-aware dragging
- Multi-body simultaneous manipulation
- Pose interpolation animation export
- Virtual springs/dampers for drag
- Haptic feedback integration

## Support and Feedback

For questions, issues, or feature requests related to interactive manipulation:
1. Check this documentation
2. Review code in `python/mujoco_golf_pendulum/interactive_manipulation.py`
3. Open an issue on the project repository

## License

This feature is part of the MuJoCo Golf Swing Model project and follows the same license.

---

**Happy Posing!** 🎮⛳
