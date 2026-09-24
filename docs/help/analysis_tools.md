# Analysis Tools

Analyze simulation results with plots, metrics, and export capabilities.

## Overview

UpstreamDrift provides comprehensive analysis tools to extract insights from simulation data, including energy analysis, phase diagrams, kinematic sequences, force/torque profiles, and Jacobian conditioning diagnostics.

Headless and GUI analyses are powered by `src.shared.python.analysis` (`AnalysisOrchestrator`), `src.shared.python.biomechanics.kinematic_sequence` (`SegmentTimingAnalyzer`), `src.shared.python.spatial_algebra.manipulability`, and engine-level validation tools.

## Energy Analysis

### Energy Components

The energy panel displays:

| Component             | Formula                   | Description        |
| --------------------- | ------------------------- | ------------------ |
| Kinetic Energy (KE)   | KE = (1/2) _ v^T _ M \* v | Energy of motion   |
| Potential Energy (PE) | PE = m _ g _ h            | Energy from height |
| Total Energy          | E = KE + PE               | Conservation check |

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

| Observation             | Meaning                     |
| ----------------------- | --------------------------- |
| Total energy constant   | Energy conserved (expected) |
| Total energy decreasing | Damping/friction present    |
| KE spike                | Rapid movement phase        |
| PE spike                | Height gained               |

### Using Energy Analysis

Retrieve structured energy time series via `AnalysisOrchestrator` or evaluate metrics directly with `EnergyMetricsMixin`:

```python
from src.shared.python.analysis import AnalysisOrchestrator
from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin

# Extract structured plot data from recorder
orchestrator = AnalysisOrchestrator(recorder)
plot_data = orchestrator.get_plot_data("energies")

# Access series
for series in plot_data.series:
    print(f"{series.name}: {len(series.y)} points ({series.units})")

# Compute numerical energy metrics directly
energy_mixin = EnergyMetricsMixin()
times, ke = recorder.get_time_series("kinetic_energy")
_, pe = recorder.get_time_series("potential_energy")
metrics = energy_mixin.compute_energy_metrics(ke, pe)
print(
    f"Max KE: {metrics['max_kinetic_energy']:.2f} J, Variation: {metrics['energy_variation']:.2f} J"
)
```

## Phase Diagrams

### What Are Phase Diagrams?

Phase diagrams plot position vs. velocity for each degree of freedom, revealing:

- Dynamic stability
- Limit cycles
- Chaotic behavior
- System characteristics

### Reading Phase Diagrams

| Pattern           | Meaning            |
| ----------------- | ------------------ |
| Closed loop       | Periodic motion    |
| Spiral inward     | Damped oscillation |
| Spiral outward    | Unstable growth    |
| Fixed point       | Equilibrium        |
| Strange attractor | Chaotic dynamics   |

### Creating Phase Plots

```python
from src.shared.python.analysis import AnalysisOrchestrator

orchestrator = AnalysisOrchestrator(recorder, joint_names=["Pelvis", "Thorax", "Arm"])

# Get phase diagram for Joint 0 (Pelvis)
phase_data = orchestrator.get_plot_data("phase_diagram")

angles = phase_data.series[0].x  # degrees
velocities = phase_data.series[0].y  # deg/s

# Compute 3D Poincaré map section crossings
poincare_data = orchestrator.get_plot_data("poincare_map_3d")
```

## Kinematic Sequence Analysis

### Proximal-to-Distal Sequencing

The kinematic sequence is crucial for golf swing efficiency:

1. **Pelvis** rotates first
2. **Torso** follows
3. **Arms** accelerate
4. **Club** reaches peak velocity at impact

### Metrics

| Metric               | Description          | Optimal         |
| -------------------- | -------------------- | --------------- |
| Pelvis peak velocity | Max angular velocity | 550-650 deg/s   |
| Torso peak velocity  | Max angular velocity | 750-900 deg/s   |
| Arm peak velocity    | Max angular velocity | 1000-1200 deg/s |
| Club peak velocity   | Max angular velocity | 2000+ deg/s     |
| Sequence timing      | Time between peaks   | 0.05-0.1s       |

### X-Factor

The X-factor measures torso-pelvis separation:

- **Address:** Initial separation
- **Top of backswing:** Maximum separation
- **X-factor stretch:** Increase during transition

### Using Kinematic Sequence Analysis

Use `SegmentTimingAnalyzer` from `src.shared.python.biomechanics.kinematic_sequence` to analyze peak timing across segments:

```python
import numpy as np
from src.shared.python.biomechanics.kinematic_sequence import (
    SegmentTimingAnalyzer,
)

# Define expected proximal-to-distal order
analyzer = SegmentTimingAnalyzer(
    expected_order=["pelvis", "torso", "lead_arm", "club"]
)

# Analyze timing from angular velocity series
result = analyzer.analyze(
    segment_velocities={
        "pelvis": pelvis_vel,
        "torso": torso_vel,
        "lead_arm": arm_vel,
        "club": club_vel,
    },
    times=timestamps,
)

print(f"Sequence consistency: {result.sequence_consistency:.2f}")
print(f"Observed peak order: {result.sequence_order}")
print(f"Valid sequence: {result.is_valid_sequence}")
for peak in result.peaks:
    print(f"{peak.name}: peak = {peak.peak_velocity:.1f} at t = {peak.time:.3f} s")
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

| Component               | Description              |
| ----------------------- | ------------------------ |
| Vertical (Fz)           | Weight support, push-off |
| Anterior-Posterior (Fx) | Forward/backward shear   |
| Medial-Lateral (Fy)     | Side-to-side forces      |
| COP                     | Center of pressure path  |

### Contact Forces

Analyze ball-club impact:

- Impact force magnitude
- Contact duration
- Force direction
- Impulse calculation

### Force and Torque Analysis API

```python
from src.shared.python.analysis import AnalysisOrchestrator

orchestrator = AnalysisOrchestrator(recorder)

# Joint torques time series
torque_plot = orchestrator.get_plot_data("joint_torques")

# Ground reaction forces (butterfly diagram with vector arrows)
grf_butterfly = orchestrator.get_plot_data("grf_butterfly_diagram")
vectors = grf_butterfly.metadata["vectors"]

# Center of Pressure trajectory and Stability (CoM vs CoP)
cop_plot = orchestrator.get_plot_data("cop_trajectory")
stability_plot = orchestrator.get_plot_data("stability_diagram")

# Joint power curves and impulse accumulation
power_plot = orchestrator.get_plot_data("joint_power_curves")
impulse_plot = orchestrator.get_plot_data("impulse_accumulation")
```

## Jacobian Analysis

### What Is the Jacobian?

The Jacobian matrix relates joint velocities to end-effector (clubhead) velocities:

```
v_clubhead = J(q) * q_dot
```

### Manipulability

Manipulability measures how effectively joint motion translates to clubhead motion:

| Metric               | Description                      |
| -------------------- | -------------------------------- |
| Manipulability index | sqrt(det(J \* J^T))              |
| Condition number     | Ratio of max/min singular values |
| Singular directions  | Directions of low manipulability |

### Manipulability Ellipsoid

The manipulability ellipsoid visualizes:

- Easy-to-move directions (long axes)
- Difficult directions (short axes)
- Singular configurations (collapsed ellipsoid)

### Using Jacobian Analysis

Use diagnostic utilities from `src.shared.python.spatial_algebra.manipulability`:

```python
from src.shared.python.spatial_algebra.manipulability import (
    check_jacobian_conditioning,
    compute_manipulability_ellipsoid,
    compute_manipulability_index,
)

# Spatial Jacobian (6 x n) computed by physics engine for clubhead
J = engine.compute_jacobian("clubhead")["spatial"]

# Yoshikawa manipulability index: mu = sqrt(det(J * J^T))
manip_index = compute_manipulability_index(J)

# Condition number: kappa = sigma_max / sigma_min
condition_number = check_jacobian_conditioning(J, body_name="clubhead", warn=True)

# Principal ellipsoid axes and radii (singular values)
radii, axes = compute_manipulability_ellipsoid(J)
print(
    f"Manipulability Index: {manip_index:.3e}, Condition Number: {condition_number:.2e}"
)
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

Validate numerical consistency across physics engines using `CrossEngineValidator`:

```python
import numpy as np
from src.shared.python.engine_core.cross_engine_validator import (
    CrossEngineValidator,
)

validator = CrossEngineValidator()

# Validate consistency between MuJoCo and Drake trajectories
result = validator.compare_states(
    "MuJoCo",
    mujoco_positions,
    "Drake",
    drake_positions,
    metric="position",
)

print(f"Validation passed: {result.passed}")
print(
    f"Max deviation: {result.max_deviation:.2e} m (tolerance: {result.tolerance:.2e} m)"
)
```

## Data Export

### Export Formats

| Format | Best For             | Includes                 |
| ------ | -------------------- | ------------------------ |
| CSV    | Spreadsheets, MATLAB | Raw numerical data       |
| JSON   | Programming, APIs    | Structured with metadata |
| MAT    | MATLAB               | Native MATLAB format     |
| HDF5   | Large datasets       | Efficient binary storage |

### Export Options

- Time range selection
- Variable selection
- Downsampling
- Include metadata

### Quick Export

Export structured analysis data to JSON or CSV:

```python
import json
from src.shared.python.analysis import AnalysisOrchestrator
from src.shared.python.analysis.reporting import ReportingMixin

# 1. Export PlotData to JSON via AnalysisOrchestrator
orchestrator = AnalysisOrchestrator(recorder)
plot_data = orchestrator.get_plot_data("energies")
with open("energy_plot.json", "w", encoding="utf-8") as f:
    json.dump(plot_data.to_dict(), f, indent=2)

# 2. Export Statistical Report to CSV via StatisticalAnalyzer
# (inherits ReportingMixin.export_statistics_csv)
from src.shared.python.validation_pkg.statistical_analysis import (
    StatisticalAnalyzer,
)

analyzer = StatisticalAnalyzer(
    times=times,
    joint_positions=q,
    joint_velocities=v,
    joint_torques=tau,
    club_head_speed=club_speed,
)
analyzer.export_statistics_csv("swing_statistics.csv")
```

## Custom Analysis

### Extending Analysis With Mixins

Custom analysis routines can integrate with UpstreamDrift's modular analysis framework:

```python
import numpy as np
from src.shared.python.analysis import BasicStatsMixin, EnergyMetricsMixin


class CustomBiomechanicsAnalyzer(BasicStatsMixin, EnergyMetricsMixin):

    def __init__(self, times: np.ndarray, joint_positions: np.ndarray) -> None:
        self.times = times
        self.joint_positions = joint_positions
        self.dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.0

    def compute_custom_metrics(self) -> dict[str, float]:
        summary = self.compute_summary_stats(self.joint_positions[:, 0])
        return {
            "mean_pos": summary.mean,
            "max_pos": summary.max_val,
            "rom": summary.range_of_motion,
        }
```

### Extending Orchestrator Plot Types

The headless `AnalysisOrchestrator` maps plot-type identifiers to extractor methods returning structured `PlotData`:

```python
from src.shared.python.analysis.plot_data import PlotData, PlotSeries


def make_custom_plot_data(
    name: str, x: list[float], y: list[float], units: str = "deg"
) -> PlotData:
    return PlotData(
        plot_type="custom_metric",
        title=f"Custom Metric: {name}",
        x_label="Time (s)",
        y_label=f"Value ({units})",
        series=[PlotSeries(name=name, x=x, y=y, units=units)],
        metadata={"custom": True},
    )
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

_See also: [Full User Manual](../user_guide/user_manual.md) | [Simulation Controls](simulation_controls.md) | [Visualization](visualization.md)_
