# Force/Torque Visualization & Trajectory Tracers

## Overview

The humanoid golf simulation now includes comprehensive visualization features:
- **Contact force arrows** showing ground reaction forces
- **Joint torque indicators** for all actuators
- **Persistent trajectory tracers** for key body parts
- **Configurable visualization options**

---

## Features

### 1. Contact Force Visualization

Displays real-time contact forces as colored arrows:

- **Red arrows**: Normal forces (perpendicular to surface)
- **Orange arrows**: Friction forces (tangential to surface)
- **Arrow length**: Proportional to force magnitude
- **Only shows forces > 0.1 N** (filters noise)

**Data Available:**
```python
{
    "position": [x, y, z],           # Contact point location
    "normal": [nx, ny, nz],          # Contact normal direction
    "normal_force": float,           # Normal force magnitude (N)
    "friction_force": float,         # Friction force magnitude (N)
    "total_force": float,            # Total force magnitude (N)
    "body1": "foot",                 # First body in contact
    "body2": "ground"                # Second body in contact
}
```

### 2. Joint Torque Visualization

Monitors torques at all actuated joints:

- **Green indicators**: Positive torque
- **Blue indicators**: Negative torque
- **Real-time updates** during simulation

**Data Available:**
```python
{
    "actuator_name": torque_value  # N⋅m for each actuator
}
```

### 3. Trajectory Tracers

Persistent traces showing motion paths of key body parts:

**Default Traced Bodies:**
- **Pelvis** (Yellow): Center of mass trajectory
- **Torso** (Cyan): Upper body motion
- **Head** (Red): Head path through swing
- **Right Hand** (Green): Club/hand path
- **Left Hand** (Blue): Lead hand path

**Features:**
- Traces persist throughout simulation
- Color-coded by body part
- Maximum 1000 points per trace (configurable)
- Can add/remove bodies dynamically

---

## Configuration

### In GUI (humanoid_launcher.py)

The launcher now includes visualization options in the config:

```python
config = {
    # ... other settings ...
    
    # Simulation
    "simulation_duration": 3.0,  # Golf swing duration (seconds)
    
    # Visualization
    "show_contact_forces": True,   # Show force arrows
    "show_joint_torques": True,    # Show torque indicators
    "show_tracers": True,          # Show trajectory traces
    "tracer_bodies": [             # Bodies to trace
        "pelvis",
        "torso", 
        "head",
        "r_hand",
        "l_hand"
    ]
}
```

### In simulation_config.json

```json
{
  "simulation_duration": 3.0,
  "show_contact_forces": true,
  "show_joint_torques": true,
  "show_tracers": true,
  "tracer_bodies": ["pelvis", "torso", "head", "r_hand", "l_hand"]
}
```

---

## Usage

### Basic Usage

1. **Launch the simulation** with live view enabled
2. **Visualization is automatic** - forces, torques, and traces appear
3. **Traces persist** as the simulation runs

### Customizing Traced Bodies

To trace different body parts, modify `tracer_bodies`:

```python
# Trace feet instead of hands
config["tracer_bodies"] = ["pelvis", "r_foot", "l_foot"]

# Trace club head
config["tracer_bodies"] = ["pelvis", "head", "club_head"]

# Trace all major joints
config["tracer_bodies"] = [
    "pelvis", "torso", "head",
    "r_shoulder", "r_elbow", "r_hand",
    "l_shoulder", "l_elbow", "l_hand"
]
```

### Disabling Visualizations

Turn off specific visualizations:

```python
config["show_contact_forces"] = False  # Hide force arrows
config["show_joint_torques"] = False   # Hide torque indicators
config["show_tracers"] = False         # Hide trajectory traces
```

---

## API Reference

### TrajectoryTracer

Manages persistent trajectory traces:

```python
from humanoid_golf.visualization import TrajectoryTracer

# Create tracer
tracer = TrajectoryTracer(max_points=1000)

# Add point to trace
tracer.add_point("pelvis", position=[0.0, 0.0, 1.0])

# Get trace for a body
trace = tracer.get_trace("pelvis")  # Returns list of [x, y, z] positions

# Clear traces
tracer.clear("pelvis")  # Clear specific body
tracer.clear()          # Clear all traces
```

### ForceVisualizer

Extracts force and torque data:

```python
from humanoid_golf.visualization import ForceVisualizer

# Create visualizer
visualizer = ForceVisualizer(physics)

# Get contact forces
contacts = visualizer.get_contact_forces()
for contact in contacts:
    print(f"Force at {contact['position']}: {contact['total_force']} N")

# Get joint torques
torques = visualizer.get_joint_torques()
for actuator, torque in torques.items():
    print(f"{actuator}: {torque} N⋅m")

# Get center of mass
com = visualizer.get_center_of_mass()
print(f"COM position: {com['position']}")
print(f"COM velocity: {com['velocity']}")
```

---

## Data Export

All visualization data can be exported for analysis:

### Contact Forces

```python
import json

# Get contact data
contacts = visualizer.get_contact_forces()

# Export to JSON
with open("contact_forces.json", "w") as f:
    json.dump(contacts, f, indent=2)
```

### Trajectory Traces

```python
# Get all traces
traces = {}
for body_name in ["pelvis", "head", "r_hand"]:
    traces[body_name] = tracer.get_trace(body_name)

# Export to JSON
with open("trajectories.json", "w") as f:
    json.dump(traces, f, indent=2)
```

### Joint Torques

```python
# Get torque data
torques = visualizer.get_joint_torques()

# Export to JSON
with open("joint_torques.json", "w") as f:
    json.dump(torques, f, indent=2)
```

---

## Color Schemes

### Force Colors

```python
FORCE_COLORS = {
    "contact_normal": [1.0, 0.0, 0.0, 0.8],      # Red
    "contact_friction": [1.0, 0.5, 0.0, 0.8],    # Orange
    "joint_torque_positive": [0.0, 1.0, 0.0, 0.8],  # Green
    "joint_torque_negative": [0.0, 0.0, 1.0, 0.8],  # Blue
}
```

### Trace Colors

```python
TRACE_COLORS = {
    "pelvis": [1.0, 1.0, 0.0, 0.6],   # Yellow
    "torso": [0.0, 1.0, 1.0, 0.6],    # Cyan
    "head": [1.0, 0.0, 0.0, 0.6],     # Red
    "r_hand": [0.0, 1.0, 0.0, 0.6],   # Green
    "l_hand": [0.0, 0.0, 1.0, 0.6],   # Blue
    "r_foot": [1.0, 0.5, 0.0, 0.6],   # Orange
    "l_foot": [0.5, 0.0, 1.0, 0.6],   # Purple
}
```

---

## Performance

### Optimization Tips

1. **Limit traced bodies**: More traces = more memory
2. **Reduce max_points**: Default 1000 is usually sufficient
3. **Disable unused features**: Turn off visualizations you don't need

```python
# Minimal visualization (best performance)
config["show_contact_forces"] = True   # Essential for golf analysis
config["show_joint_torques"] = False   # Disable if not needed
config["show_tracers"] = True
config["tracer_bodies"] = ["club_head"]  # Only trace club

# Maximum visualization (comprehensive analysis)
config["show_contact_forces"] = True
config["show_joint_torques"] = True
config["show_tracers"] = True
config["tracer_bodies"] = [
    "pelvis", "torso", "head",
    "r_shoulder", "r_elbow", "r_hand",
    "l_shoulder", "l_elbow", "l_hand",
    "r_hip", "r_knee", "r_foot",
    "l_hip", "l_knee", "l_foot"
]
```

---

## Troubleshooting

### Traces Not Appearing

1. Check `show_tracers` is `True`
2. Verify body names exist in model
3. Ensure simulation is running (not paused)

### Force Arrows Not Visible

1. Check `show_contact_forces` is `True`
2. Verify contacts are occurring (check ground contact)
3. Forces < 0.1 N are filtered out

### Performance Issues

1. Reduce `max_points` in TrajectoryTracer
2. Limit number of traced bodies
3. Disable unused visualizations

---

## Future Enhancements

Planned features:
- [ ] Real-time plotting dashboard
- [ ] Export to video with overlays
- [ ] Interactive trace playback
- [ ] Force magnitude heatmaps
- [ ] Joint angle overlays
- [ ] Energy visualization

---

## Examples

### Golf Swing Analysis

```python
# Optimal config for golf swing analysis
config = {
    "simulation_duration": 3.0,
    "show_contact_forces": True,  # See ground reaction forces
    "show_joint_torques": True,   # Monitor joint loads
    "show_tracers": True,
    "tracer_bodies": [
        "pelvis",      # Body rotation
        "head",        # Head stability
        "r_hand",      # Club path
        "club_head"    # Ball contact point
    ]
}
```

### Balance Analysis

```python
# Config for balance and stability analysis
config = {
    "simulation_duration": 3.0,
    "show_contact_forces": True,  # Ground reaction forces
    "show_joint_torques": False,
    "show_tracers": True,
    "tracer_bodies": [
        "pelvis",      # Center of mass
        "r_foot",      # Right foot path
        "l_foot"       # Left foot path
    ]
}
```

---

## See Also

- [INTERACTIVE_VISUALIZATION_GUIDE.md](INTERACTIVE_VISUALIZATION_GUIDE.md) - Complete MuJoCo visualization guide
- [GUI_ARCHITECTURE.md](GUI_ARCHITECTURE.md) - GUI architecture documentation
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference guide
