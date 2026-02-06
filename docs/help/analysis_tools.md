# Analysis Tools

Analyze simulation results with plots, metrics, and export capabilities.

## Overview

UpstreamDrift provides comprehensive analysis tools to extract insights from simulation data, including energy analysis, phase diagrams, kinematic sequences, and force/torque profiles.

## Energy Analysis

### Energy Components

The energy panel displays:

| Component | Formula | Description |
|-----------|---------|-------------|
| Kinetic Energy (KE) | KE = (1/2) * v^T * M * v | Energy of motion |
| Potential Energy (PE) | PE = m * g * h | Energy from height |
| Total Energy | E = KE + PE | Conservation check |

### Energy Plots

**Time Series:**
- KE, PE, and Total energy over time
- Energy conservation verification
- Energy transfer visualization

**Energy Distribution:**
- Energy by segment/body
- Peak energy timing
- Energy flow direction

### Interpreting Energy

| Observation | Meaning |
|-------------|---------|
| Total energy constant | Energy conserved (expected) |
| Total energy decreasing | Damping/friction present |
| KE spike | Rapid movement phase |
| PE spike | Height gained |

### Using Energy Analysis

```python
from shared.python.analysis import EnergyAnalyzer

analyzer = EnergyAnalyzer(simulation_data)

# Get energy time series
ke, pe, total = analyzer.compute_energy()

# Check conservation
is_conserved = analyzer.check_conservation(tolerance=0.01)

# Find peak kinetic energy
peak_time, peak_ke = analyzer.find_peak_kinetic()
```

## Phase Diagrams

### What are Phase Diagrams?

Phase diagrams plot position vs. velocity for each degree of freedom, revealing:
- Dynamic stability
- Limit cycles
- Chaotic behavior
- System characteristics

### Reading Phase Diagrams

| Pattern | Meaning |
|---------|---------|
| Closed loop | Periodic motion |
| Spiral inward | Damped oscillation |
| Spiral outward | Unstable growth |
| Fixed point | Equilibrium |
| Strange attractor | Chaotic dynamics |

### Creating Phase Plots

```python
from shared.python.plotting import PhaseDiagramPlotter

plotter = PhaseDiagramPlotter(simulation_data)

# Plot single joint
fig = plotter.plot_joint(joint_index=0)

# Plot multiple joints
fig = plotter.plot_joints([0, 1, 2])

# 3D phase space
fig = plotter.plot_3d_phase_space(joint_index=0)
```

## Kinematic Sequence Analysis

### Proximal-to-Distal Sequencing

The kinematic sequence is crucial for golf swing efficiency:

1. **Pelvis** rotates first
2. **Torso** follows
3. **Arms** accelerate
4. **Club** reaches peak velocity at impact

### Metrics

| Metric | Description | Optimal |
|--------|-------------|---------|
| Pelvis peak velocity | Max angular velocity | 550-650 deg/s |
| Torso peak velocity | Max angular velocity | 750-900 deg/s |
| Arm peak velocity | Max angular velocity | 1000-1200 deg/s |
| Club peak velocity | Max angular velocity | 2000+ deg/s |
| Sequence timing | Time between peaks | 0.05-0.1s |

### X-Factor

The X-factor measures torso-pelvis separation:
- **Address:** Initial separation
- **Top of backswing:** Maximum separation
- **X-factor stretch:** Increase during transition

### Using Kinematic Sequence

```python
from shared.python.analysis import KinematicSequenceAnalyzer

analyzer = KinematicSequenceAnalyzer(simulation_data)

# Compute full sequence
sequence = analyzer.compute_sequence()

print(f"Pelvis peak: {sequence.pelvis_peak_velocity:.0f} deg/s")
print(f"Torso peak: {sequence.torso_peak_velocity:.0f} deg/s")
print(f"Club peak: {sequence.club_peak_velocity:.0f} deg/s")

# Check sequencing order
is_correct = analyzer.verify_sequence_order()
```

## Force and Torque Analysis

### Joint Torques

Display and analyze torques at each joint:

**Plots:**
- Torque vs. time
- Torque vs. joint angle
- Peak torque identification
- Torque direction changes

### Ground Reaction Forces

Analyze foot-ground interaction:

| Component | Description |
|-----------|-------------|
| Vertical (Fz) | Weight support, push-off |
| Anterior-Posterior (Fx) | Forward/backward shear |
| Medial-Lateral (Fy) | Side-to-side forces |
| COP | Center of pressure path |

### Contact Forces

Analyze ball-club impact:
- Impact force magnitude
- Contact duration
- Force direction
- Impulse calculation

### Force Analysis API

```python
from shared.python.analysis import ForceAnalyzer

analyzer = ForceAnalyzer(simulation_data)

# Joint torques
torques = analyzer.get_joint_torques()

# Ground reaction forces
grf = analyzer.get_ground_reaction_forces()

# Contact forces
impact = analyzer.get_impact_forces()
```

## Jacobian Analysis

### What is the Jacobian?

The Jacobian matrix relates joint velocities to end-effector (clubhead) velocities:

```
v_clubhead = J(q) * q_dot
```

### Manipulability

Manipulability measures how effectively joint motion translates to clubhead motion:

| Metric | Description |
|--------|-------------|
| Manipulability index | sqrt(det(J * J^T)) |
| Condition number | Ratio of max/min singular values |
| Singular directions | Directions of low manipulability |

### Manipulability Ellipsoid

The manipulability ellipsoid visualizes:
- Easy-to-move directions (long axes)
- Difficult directions (short axes)
- Singular configurations (collapsed ellipsoid)

### Using Jacobian Analysis

```python
from shared.python.analysis import JacobianAnalyzer

analyzer = JacobianAnalyzer(engine)

# Get Jacobian at current configuration
J = analyzer.get_jacobian()

# Compute manipulability
manip = analyzer.compute_manipulability()

# Find singular configurations
singulars = analyzer.find_singular_configurations()
```

## Comparative Analysis

### Swing Comparison

Compare multiple swings:

**Metrics:**
- Joint angle differences
- Velocity profiles
- Timing differences
- Peak values comparison

### Cross-Engine Validation

Compare results across physics engines:

```python
from shared.python.cross_engine_validator import CrossEngineValidator

validator = CrossEngineValidator()

# Run same simulation on multiple engines
results = validator.compare_engines(
    engines=["mujoco", "drake", "pinocchio"],
    params=simulation_params
)

# Check deviations
for engine, data in results.items():
    print(f"{engine}: max deviation = {data.max_deviation}")
```

## Data Export

### Export Formats

| Format | Best For | Includes |
|--------|----------|----------|
| CSV | Spreadsheets, MATLAB | Raw numerical data |
| JSON | Programming, APIs | Structured with metadata |
| MAT | MATLAB | Native MATLAB format |
| HDF5 | Large datasets | Efficient binary storage |

### Export Options

- Time range selection
- Variable selection
- Downsampling
- Include metadata

### Quick Export

```python
from shared.python.export import DataExporter

exporter = DataExporter(simulation_data)

# Export to CSV
exporter.to_csv("results.csv", columns=["time", "q", "v", "tau"])

# Export to JSON with metadata
exporter.to_json("results.json", include_metadata=True)

# Export plots
exporter.export_plots("plots/", format="png", dpi=300)
```

## Custom Analysis

### Writing Custom Analyzers

Extend the analysis framework:

```python
from shared.python.analysis import BaseAnalyzer

class MyAnalyzer(BaseAnalyzer):
    def compute(self):
        # Access simulation data
        time = self.data.time
        positions = self.data.q

        # Your analysis
        result = my_custom_analysis(time, positions)

        return result
```

### Registering Custom Plots

Add custom plots to the analysis panel:

```python
from shared.python.plotting import register_custom_plot

@register_custom_plot("My Custom Plot")
def my_plot(data, ax):
    ax.plot(data.time, data.q[:, 0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
```

## Troubleshooting

### No Data to Analyze

- Ensure recording was enabled before simulation
- Check that simulation ran for sufficient time
- Verify data was saved correctly

### Unexpected Results

- Verify simulation parameters
- Check for numerical instability
- Compare with simple test cases
- Validate against known solutions

### Export Failures

- Check write permissions
- Verify disk space
- Try different format
- Check for special characters in filenames

---

*See also: [Full User Manual](../USER_MANUAL.md) | [Simulation Controls](simulation_controls.md) | [Visualization](visualization.md)*
