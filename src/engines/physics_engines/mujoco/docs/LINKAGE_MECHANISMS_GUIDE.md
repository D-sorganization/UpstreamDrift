# Linkage Mechanisms Library Guide

## Overview

The MuJoCo Golf Swing Model now includes a comprehensive library of classic linkage mechanisms and parallel manipulators for physics exploration and education. This library provides 16 different mechanisms across 5 major categories, allowing users to explore fundamental concepts in kinematics, dynamics, and mechanical design.

## Purpose

This library transforms the golf swing analysis tool into a full-featured physics exploration package. While golf swing analysis remains the primary feature, these mechanisms provide:

- **Educational value**: Learn about classic mechanical engineering designs
- **Physics exploration**: Experiment with different types of motion conversion
- **Historical context**: Study mechanisms that powered the industrial revolution
- **Modern robotics**: Explore parallel manipulators used in industry today

## Categories of Mechanisms

### 1. Four-Bar Linkages

Four-bar linkages are fundamental planar mechanisms consisting of four rigid links connected by revolute (hinge) joints. They are classified based on the Grashof condition, which determines whether the mechanism can achieve full rotation.

**Grashof Condition**: `s + l < p + q`
- Where s = shortest link, l = longest link, p & q = intermediate links
- If satisfied, at least one link can rotate fully 360°

#### Available Four-Bar Mechanisms:

**Four-Bar: Grashof Crank-Rocker**
- Description: Classic crank-rocker mechanism with Grashof condition satisfied
- Behavior: Input crank rotates fully, output rocker oscillates
- Applications: Windshield wipers, rocking chairs, oscillating fans
- Link lengths: [4.0, 1.0, 3.5, 3.0] (ground, crank, coupler, follower)

**Four-Bar: Double Crank**
- Description: Both input and output links can rotate fully (Grashof)
- Behavior: Continuous rotation transmission with varying velocity ratio
- Applications: Parallel-crank steering linkages, some conveyor systems
- Link lengths: [4.0, 2.0, 4.5, 3.5]

**Four-Bar: Double Rocker**
- Description: Both output links oscillate (no full rotation)
- Behavior: Input oscillation creates output oscillation
- Applications: Rocking mechanisms, certain toggle mechanisms
- Link lengths: [4.0, 3.8, 2.5, 3.0]

**Four-Bar: Parallelogram**
- Description: Special case with parallel opposite links (equal length)
- Behavior: Output link maintains same orientation as input (parallel motion)
- Applications: Truck suspensions, drafting tables, folding mechanisms
- Link lengths: [4.0, 2.0, 4.0, 2.0]

### 2. Slider Mechanisms

Slider mechanisms convert rotary motion to linear motion (or vice versa) and are foundational to many engines and machines.

**Slider-Crank (Horizontal)**
- Description: Piston engine mechanism converting rotation to linear motion
- Behavior: Crank rotation drives slider in linear reciprocating motion
- Applications: Internal combustion engines, reciprocating pumps, compressors
- Parameters: Crank length = 1.0, Rod length = 3.0
- Physics: Non-sinusoidal motion due to connecting rod angularity

**Slider-Crank (Vertical)**
- Description: Vertical orientation of slider-crank mechanism
- Behavior: Same kinematics as horizontal, different gravity effects
- Applications: Vertical compressors, pile drivers
- Parameters: Crank length = 1.0, Rod length = 3.0

**Scotch Yoke**
- Description: Perfect simple harmonic motion generator
- Behavior: Produces exact sinusoidal motion (no connecting rod angularity)
- Applications: Control valves, function generators, mechanical computers
- Parameters: Crank radius = 1.0
- Unique property: Output displacement = R·sin(θ) exactly

### 3. Special Mechanisms

These mechanisms serve specialized purposes in mechanical systems.

**Geneva Mechanism**
- Description: Intermittent motion converter (continuous input → indexed output)
- Behavior: Drive pin engages slots to create stepped rotational motion
- Applications: Film projectors, indexing tables, automatic tools
- Parameters: 6 slots (can be 4, 6, or 8), Drive radius = 2.0
- Dwell time: Output rests while drive completes rotation

**Oldham Coupling**
- Description: Couples two parallel shafts with lateral offset
- Behavior: Transmits rotation between offset parallel shafts
- Applications: Shaft couplings, eccentric drives, scroll compressors
- Parameters: Offset = 0.5
- Unique: Maintains constant velocity ratio despite misalignment

### 4. Straight-Line Mechanisms

These remarkable mechanisms generate approximate or exact straight-line motion using only rotary joints—a significant achievement in the history of mechanical engineering.

**Peaucellier-Lipkin Linkage**
- Description: First planar linkage to produce mathematically perfect straight-line motion
- Behavior: Converts circular motion into exact linear motion
- Historical significance: Solved a major engineering problem (1864)
- Applications: Historical beam engines, mechanical computing
- Mathematical basis: Inverse geometry transformation

**Chebyshev Linkage**
- Description: Three-bar linkage producing approximate straight-line motion
- Behavior: Foot point traces nearly straight path over portion of cycle
- Applications: Walking mechanisms, leg prosthetics, mechanical calculators
- Discovered by: Pafnuty Chebyshev (1850s)
- Advantage: Simpler than exact mechanisms, sufficient for many applications

**Watt's Linkage**
- Description: James Watt's parallel motion linkage for steam engines
- Behavior: Center point traces approximate straight vertical line
- Historical significance: Enabled practical double-acting steam engines (1784)
- Applications: Steam engine beam guidance, vehicle suspensions
- Impact: Critical to the Industrial Revolution

### 5. Parallel Mechanisms

Parallel manipulators have multiple kinematic chains connecting the base to the end effector, providing high stiffness and accuracy.

**Delta Robot (3-DOF Parallel)**
- Description: High-speed pick-and-place parallel robot
- Behavior: Three arms control platform position in 3D space
- Applications: Food packaging, electronics assembly, medical devices
- DOF: 3 translational (X, Y, Z)
- Advantages: High speed, good acceleration, large workspace
- Parameters: Base radius = 2.0, Platform radius = 0.5

**5-Bar Parallel Manipulator**
- Description: Two-degree-of-freedom planar parallel robot
- Behavior: Two actuated arms control end effector position in plane
- Applications: Drawing machines, laser cutting, assembly tasks
- DOF: 2 translational (planar motion)
- Advantages: High stiffness, good accuracy, compact design
- Parameters: Link length = 1.5, Base width = 2.0

**Stewart Platform (6-DOF)**
- Description: Six-degree-of-freedom parallel manipulator (hexapod)
- Behavior: Six actuated legs control platform in all spatial directions
- Applications: Flight simulators, tire testing, telescope mounts, CNC machines
- DOF: 6 (3 translational + 3 rotational) - full spatial mobility
- Advantages: Very high stiffness, excellent load capacity, precise positioning
- Parameters: Base radius = 1.5, Platform radius = 0.8
- Complexity: Most versatile parallel mechanism

### 6. Scaling Mechanisms

**Pantograph**
- Description: Geometric scaling and copying mechanism
- Behavior: Output point traces path similar to input but scaled
- Applications: Engraving machines, photocopy machines, sign writing
- Parameters: Scale factor = 2.0 (adjustable)
- Principle: Similar triangles maintain proportional distances

## Using the Mechanisms in the GUI

### Loading a Mechanism

1. Launch the application:
   ```bash
   python -m mujoco_golf_pendulum
   ```

2. In the GUI, locate the "Physics Models & Mechanisms" dropdown in the Control Panel

3. Mechanisms are listed with the prefix "Mechanism:" followed by their name
   - Example: "Mechanism: Four-Bar: Grashof Crank-Rocker"

4. Select any mechanism to load it into the simulation

### Controlling Mechanisms

Each mechanism has one or more actuators (motors) that you can control:

- **Sliders**: Use the actuator sliders to apply torque/force to the mechanism
- **Range**: Most motors are limited to ±5 to ±15 Nm depending on mechanism size
- **Real-time**: Adjustments take effect immediately in the physics simulation

### Simulation Controls

- **Play/Pause**: Start or pause the physics simulation
- **Reset**: Return mechanism to initial configuration
- **Speed Control**: Adjust simulation speed for detailed observation

### Visualization Features

The mechanisms include:
- **Color-coded links**: Different colors for different link types
  - Red/Orange: Input cranks or drivers
  - Green: Couplers and intermediate links
  - Blue: Output links or followers
  - Yellow: Special tracer points

- **Pivot markers**: Spheres indicate fixed pivot points (gray)
- **Tracer points**: Special markers that show the path of interest
  - Straight-line mechanisms have red tracers
  - End effectors on parallel robots are marked in orange

### Analysis and Data Export

All standard analysis features work with mechanisms:

1. **Force/Torque Analysis**: View constraint forces and actuator torques
2. **Joint Analysis**: Examine joint positions, velocities, accelerations
3. **Plotting**: Generate phase portraits, energy plots, trajectory plots
4. **Data Export**: Export kinematic data to CSV or JSON

## Educational Applications

### Recommended Exploration Sequence

For learning about mechanisms, we recommend this exploration order:

1. **Start with Four-Bar Linkages**: Understand basic planar kinematics
   - Compare Grashof vs Non-Grashof behavior
   - Observe how link length ratios affect motion

2. **Explore Slider Mechanisms**: Learn rotation-to-translation conversion
   - Compare Slider-Crank vs Scotch Yoke motion quality
   - Understand connecting rod effects on kinematics

3. **Study Straight-Line Mechanisms**: Appreciate historical engineering achievements
   - Observe Peaucellier's perfect straight line
   - Compare approximate mechanisms (Chebyshev, Watt)

4. **Examine Special Mechanisms**: Understand specialized motion types
   - Geneva mechanism intermittent motion
   - Oldham coupling offset transmission

5. **Advanced: Parallel Mechanisms**: Explore modern robotics
   - 5-bar planar workspace
   - Delta robot 3D workspace
   - Stewart platform 6-DOF motion

### Physics Concepts Demonstrated

**Kinematics**:
- Position, velocity, acceleration relationships
- Degrees of freedom and mobility
- Grashof condition and kinematic inversions
- Instantaneous centers of rotation

**Dynamics**:
- Constraint forces in joints
- Inertial effects and accelerations
- Energy conservation and dissipation
- Mechanical advantage and transmission

**Control**:
- Torque-based actuation
- Position control challenges
- Parallel mechanism singularities
- Workspace limitations

## Technical Details

### MuJoCo Physics Parameters

All mechanisms use:
- **Timestep**: 0.001 - 0.002 seconds
- **Gravity**: -9.81 m/s² (Earth gravity)
- **Integrator**: Implicit Euler (stable for constrained systems)
- **Joint damping**: 0.05 - 2.0 Nm·s/rad (varies by mechanism)

### Coordinate Systems

- **World frame**: Z-axis points upward (against gravity)
- **Visualization plane**: Planar mechanisms operate in X-Y plane at Z ≈ 0.5-1.0m
- **Units**: SI units (meters, kilograms, seconds, Newtons)

### Joint Types Used

- **Hinge joints**: Revolute joints for rotational motion
- **Slider joints**: Prismatic joints for linear motion
- **Ball joints**: Spherical joints (3-DOF) for spatial linkages
- **Free joints**: 6-DOF unconstrained bodies (end effectors)

### Constraint Implementation

- **Connect constraints**: Weld specified points on different bodies
- **Equality constraints**: Enforce joint coupling relationships
- **Solver parameters**: Tuned for stability and accuracy

## Programming Interface

### Accessing Mechanisms Programmatically

```python
from mujoco_golf_pendulum.linkage_mechanisms import LINKAGE_CATALOG

# List all available mechanisms
for name, config in LINKAGE_CATALOG.items():
    print(f"{name}: {config['description']}")

# Get specific mechanism XML
from mujoco_golf_pendulum.linkage_mechanisms import generate_four_bar_linkage_xml

xml = generate_four_bar_linkage_xml(
    link_lengths=[4.0, 1.0, 3.5, 3.0],
    link_type="grashof_crank_rocker"
)
```

### Available Generator Functions

- `generate_four_bar_linkage_xml(link_lengths, link_type)`
- `generate_slider_crank_xml(crank_length, rod_length, orientation)`
- `generate_scotch_yoke_xml(crank_radius)`
- `generate_geneva_mechanism_xml(num_slots, drive_radius)`
- `generate_peaucellier_linkage_xml(scale)`
- `generate_chebyshev_linkage_xml()`
- `generate_watt_linkage_xml()`
- `generate_pantograph_xml(scale_factor)`
- `generate_delta_robot_xml(base_radius, platform_radius)`
- `generate_five_bar_parallel_xml(link_length)`
- `generate_stewart_platform_xml(base_radius, platform_radius)`
- `generate_oldham_coupling_xml(offset)`

### Customizing Mechanisms

You can modify mechanism parameters by editing the generator functions or creating new ones:

```python
# Custom four-bar with specific link lengths
custom_xml = generate_four_bar_linkage_xml(
    link_lengths=[5.0, 1.5, 4.0, 3.5],
    link_type="grashof_crank_rocker"
)

# Custom pantograph with 3x scaling
custom_pantograph = generate_pantograph_xml(scale_factor=3.0)
```

## References and Further Reading

### Classic Mechanism Theory
- Hartenberg, R. S., & Denavit, J. (1964). *Kinematic Synthesis of Linkages*
- Erdman, A. G., & Sandor, G. N. (1997). *Mechanism Design: Analysis and Synthesis*
- Norton, R. L. (2011). *Design of Machinery*

### Historical Context
- Ferguson, E. S. (1962). "Kinematics of Mechanisms from the Time of Watt"
- Moon, F. C. (2007). *The Machines of Leonardo da Vinci and Franz Reuleaux*

### Parallel Mechanisms
- Merlet, J. P. (2006). *Parallel Robots*
- Gosselin, C., & Angeles, J. (1990). "Singularity Analysis of Closed-Loop Kinematic Chains"

### Straight-Line Linkages
- Artobolevskii, I. I. (1975). *Mechanisms in Modern Engineering Design*
- Kempe, A. B. (1877). *How to Draw a Straight Line* (historic classic)

## Troubleshooting

### Mechanism Not Moving
- Check that simulation is running (Play button pressed)
- Verify actuator sliders are moved from zero position
- Some mechanisms have limited range - try both directions

### Unstable Behavior
- Reduce actuator torque/force inputs
- Check that mechanism hasn't reached a singularity
- Reset simulation and try again

### Poor Performance
- Some complex mechanisms (Stewart platform) are computationally intensive
- Reduce simulation speed if needed
- Close other applications to free resources

## Contributing New Mechanisms

To add a new mechanism to the library:

1. Create a generator function in `linkage_mechanisms/__init__.py`
2. Follow the existing XML structure pattern
3. Add entry to `LINKAGE_CATALOG` dictionary
4. Include proper description and category
5. Test in GUI for stability and correct physics

Example structure:
```python
def generate_your_mechanism_xml(param1, param2):
    """
    Generate your mechanism.

    Parameters:
    -----------
    param1 : type
        Description
    """
    xml = f"""
    <mujoco model="your_mechanism">
        <!-- MuJoCo XML structure -->
    </mujoco>
    """
    return xml
```

## Conclusion

This linkage mechanisms library provides a rich environment for exploring mechanical engineering principles through interactive physics simulation. Whether you're a student learning kinematics, an educator teaching mechanism design, or simply curious about how machines work, these mechanisms offer hands-on experience with fundamental concepts that have shaped our technological world.

Enjoy exploring the fascinating world of mechanisms!
