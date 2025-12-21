# MuJoCo Interactive Visualization & Data Analysis Guide

## Overview

This guide explains how to leverage MuJoCo's powerful built-in features for interactive visualization, force/torque analysis, and real-time data monitoring of the humanoid golf model.

---

## 🎮 Built-in MuJoCo Features

### 1. **Interactive Viewer Controls**

MuJoCo's viewer provides extensive built-in controls:

#### Mouse Controls
- **Left Click + Drag**: Rotate camera
- **Right Click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out
- **Double Click**: Select body (shows info)

#### Keyboard Shortcuts
- **Space**: Pause/Resume simulation
- **Right Arrow**: Step forward one frame
- **Left Arrow**: Step backward (if recording)
- **Backspace**: Reset simulation
- **Ctrl+P**: Toggle profiler
- **Ctrl+F**: Toggle frame rate display
- **Ctrl+S**: Save screenshot
- **F1**: Toggle help overlay

#### Visualization Options
- **Ctrl+1-9**: Toggle visualization groups
  - Contact forces
  - Joint axes
  - Center of mass
  - Inertia boxes
  - Perturbation forces
  - Constraint forces

---

## 📊 Force & Torque Visualization

### Contact Forces

MuJoCo can visualize contact forces in real-time:

```python
# In your simulation code
import mujoco
import mujoco.viewer

# Enable contact force visualization
viewer_options = mujoco.MjvOption()
viewer_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
viewer_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

# Set force arrow scaling
viewer_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
```

**What You'll See:**
- Red arrows showing contact force magnitude and direction
- Contact points highlighted
- Force magnitude proportional to arrow length

### Joint Torques

Display actuator forces and joint torques:

```python
# Enable joint visualization
viewer_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
viewer_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

# Access torque data
for i in range(model.nu):
    actuator_force = data.actuator_force[i]
    print(f"Actuator {i}: {actuator_force} N⋅m")
```

### Constraint Forces

Visualize constraint forces (joint limits, contacts):

```python
viewer_options.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
```

---

## 📈 Real-Time Data Access

### Available Data Streams

MuJoCo provides extensive sensor and state data:

#### 1. **Joint States**
```python
# Position (angles)
qpos = data.qpos  # All generalized positions
joint_angles = data.qpos[7:]  # Skip free joint (first 7 DOF)

# Velocity
qvel = data.qvel
joint_velocities = data.qvel[6:]  # Skip free joint velocities

# Acceleration
qacc = data.qacc
```

#### 2. **Contact Forces**
```python
# Number of active contacts
n_contacts = data.ncon

# Iterate through contacts
for i in range(data.ncon):
    contact = data.contact[i]
    
    # Contact position
    pos = contact.pos
    
    # Contact frame (normal, tangent directions)
    frame = contact.frame
    
    # Contact force (in contact frame)
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, force)
    
    normal_force = force[0]  # Normal component
    friction_force = np.linalg.norm(force[1:3])  # Tangential
```

#### 3. **Actuator Data**
```python
# Actuator forces (torques)
actuator_forces = data.actuator_force

# Control signals
ctrl = data.ctrl

# Actuator lengths (for tendons)
actuator_length = data.actuator_length
```

#### 4. **Energy & Dynamics**
```python
# Kinetic energy
kinetic_energy = data.energy[0]

# Potential energy
potential_energy = data.energy[1]

# Total energy
total_energy = kinetic_energy + potential_energy

# Center of mass
com_pos = data.subtree_com[0]  # Root body COM
com_vel = data.cvel[0]  # COM velocity
```

#### 5. **Sensor Readings**
```python
# If you have sensors defined in XML
for i in range(model.nsensor):
    sensor_data = data.sensordata[i]
    sensor_name = model.sensor(i).name
    print(f"{sensor_name}: {sensor_data}")
```

---

## 🎨 Custom Visualization Overlays

### Adding Text Overlays

Display real-time data on the viewer:

```python
def add_text_overlay(viewer, data):
    """Add custom text overlay to viewer."""
    # This requires accessing the viewer's internal text buffer
    # MuJoCo viewer supports custom rendering callbacks
    
    # Example: Display joint angles
    text = f"Time: {data.time:.2f}s\n"
    text += f"Energy: {data.energy[0] + data.energy[1]:.2f} J\n"
    text += f"Contacts: {data.ncon}\n"
    
    # Add to viewer (implementation depends on viewer type)
    viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "Stats", text)
```

### Custom Rendering

Add custom geometric primitives:

```python
def render_com_trajectory(viewer, com_history):
    """Render center of mass trajectory."""
    # Add line segments for trajectory
    for i in range(len(com_history) - 1):
        viewer.add_marker(
            pos=com_history[i],
            size=[0.01, 0.01, 0.01],
            rgba=[1, 0, 0, 0.5],
            type=mujoco.mjtGeom.mjGEOM_SPHERE
        )
```

---

## 📉 Real-Time Plotting

### Option 1: Matplotlib Integration

Plot data alongside simulation:

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RealtimePlotter:
    def __init__(self):
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8))
        self.time_data = []
        self.angle_data = []
        self.torque_data = []
        self.energy_data = []
        
    def update(self, data):
        self.time_data.append(data.time)
        self.angle_data.append(data.qpos[7:].copy())
        self.torque_data.append(data.actuator_force.copy())
        self.energy_data.append(data.energy[0] + data.energy[1])
        
        # Update plots
        self.axes[0].clear()
        self.axes[0].plot(self.time_data, self.angle_data)
        self.axes[0].set_ylabel('Joint Angles (rad)')
        
        self.axes[1].clear()
        self.axes[1].plot(self.time_data, self.torque_data)
        self.axes[1].set_ylabel('Torques (N⋅m)')
        
        self.axes[2].clear()
        self.axes[2].plot(self.time_data, self.energy_data)
        self.axes[2].set_ylabel('Total Energy (J)')
        self.axes[2].set_xlabel('Time (s)')
        
        plt.pause(0.001)
```

### Option 2: MuJoCo Profiler

Use built-in profiler for performance metrics:

```python
# Enable profiler
mujoco.mjv_defaultOption(viewer_options)
viewer_options.flags[mujoco.mjtVisFlag.mjVIS_PROFILER] = True

# Profiler shows:
# - Step time
# - Forward dynamics time
# - Constraint solver time
# - Collision detection time
```

---

## 🎬 Recording & Playback

### Record Simulation

```python
class SimulationRecorder:
    def __init__(self, model):
        self.states = []
        self.times = []
        
    def record_frame(self, data):
        """Record current state."""
        self.states.append({
            'qpos': data.qpos.copy(),
            'qvel': data.qvel.copy(),
            'ctrl': data.ctrl.copy(),
            'time': data.time
        })
        self.times.append(data.time)
    
    def playback(self, model, data, viewer):
        """Play back recorded simulation."""
        for state in self.states:
            data.qpos[:] = state['qpos']
            data.qvel[:] = state['qvel']
            data.ctrl[:] = state['ctrl']
            data.time = state['time']
            
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
```

---

## 🔬 Advanced Analysis Features

### 1. **Contact Analysis**

Detailed contact force analysis:

```python
def analyze_contacts(model, data):
    """Analyze all active contacts."""
    contact_info = []
    
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Get contact force
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        
        # Get body names
        geom1 = contact.geom1
        geom2 = contact.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]
        
        contact_info.append({
            'body1': model.body(body1).name,
            'body2': model.body(body2).name,
            'position': contact.pos.copy(),
            'normal_force': force[0],
            'friction_force': np.linalg.norm(force[1:3]),
            'penetration': contact.dist
        })
    
    return contact_info
```

### 2. **Joint Load Analysis**

Calculate joint loads:

```python
def compute_joint_loads(model, data):
    """Compute loads on each joint."""
    # Inverse dynamics to get required torques
    mujoco.mj_inverse(model, data)
    
    joint_loads = {}
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        qfrc = data.qfrc_inverse[i]
        joint_loads[joint_name] = qfrc
    
    return joint_loads
```

### 3. **Stability Metrics**

Analyze balance and stability:

```python
def compute_stability_metrics(model, data):
    """Compute balance and stability metrics."""
    # Center of mass
    com = data.subtree_com[0]
    
    # Zero moment point (ZMP)
    # Simplified calculation
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    
    for i in range(data.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        contact_pos = data.contact[i].pos
        
        total_force += force[:3]
        total_moment += np.cross(contact_pos - com, force[:3])
    
    if total_force[2] > 0:
        zmp = com[:2] - total_moment[:2] / total_force[2]
    else:
        zmp = com[:2]
    
    return {
        'com': com,
        'zmp': zmp,
        'com_height': com[2],
        'is_stable': True  # Add stability criteria
    }
```

---

## 🚀 Quick Start: Enable Motion

To get your model moving immediately:

```python
# 1. Set initial pose (standing position)
data.qpos[2] = 1.0  # Raise pelvis to standing height

# 2. Enable a simple controller
def simple_pd_controller(model, data):
    """Simple PD controller to maintain standing pose."""
    kp = 100.0  # Proportional gain
    kd = 10.0   # Derivative gain
    
    target_pos = np.zeros(model.nu)  # Target joint angles
    
    for i in range(model.nu):
        error = target_pos[i] - data.qpos[7 + i]
        error_dot = -data.qvel[6 + i]
        data.ctrl[i] = kp * error + kd * error_dot

# 3. Run simulation with controller
while viewer.is_running():
    simple_pd_controller(model, data)
    mujoco.mj_step(model, data)
    viewer.sync()
```

---

## 📚 References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Visualization Guide](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [Contact Force Tutorial](https://mujoco.readthedocs.io/en/stable/programming/samples.html#contact-forces)

---

## 🎯 Next Steps

1. **Immediate**: Add default controller to enable motion
2. **Short-term**: Implement force visualization overlays
3. **Medium-term**: Add real-time plotting dashboard
4. **Long-term**: Build comprehensive analysis suite

See implementation in `humanoid_golf/sim.py` for working examples.
