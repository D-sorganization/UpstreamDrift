# Chapter 22: Visualization System

UpstreamDrift provides a layered visualization architecture. At the
lowest level, the MuJoCo 3-D viewer renders force arrows, trajectory
traces, and torque spheres directly into the scene. Above that, a
comprehensive matplotlib-based plotting framework generates publication-
quality figures and animations. A Qt-based dashboard ties everything
together for interactive use.

```
Visualization layers
====================
Layer 3:  Dashboard (Qt)      -- UnifiedDashboardWindow, LivePlotWidget
Layer 2:  Plotting (matplotlib) -- 11 specialised renderers, animation, export
Layer 1:  3-D Overlays (MuJoCo) -- arrows, spheres, trajectory lines in-scene
```

---

## 22.1 MuJoCo 3-D Visualization

The file
`src/engines/physics_engines/mujoco/docker/src/humanoid_golf/visualization.py`
implements real-time 3-D overlays rendered through the native MuJoCo
`mjvGeom` API. No external renderer is needed; geometry primitives are
injected directly into the MuJoCo scene buffer.

### 22.1.1 TrajectoryTracer

`TrajectoryTracer` records body positions over time in bounded deques
(default `max_points=1000`):

```python
tracer = TrajectoryTracer(max_points=1000)
tracer.add_point("club_head", np.array([0.5, 0.1, 1.2]))
```

The tracer also supports a **desired-trajectory overlay** via
`set_desired_trajectory(body_name, positions)`. This registers a
reference path that is drawn alongside the actual trace for comparison.
`get_desired_trace(body_name)` retrieves the reference.

`clear(body_name=None)` removes traces for one body (or all bodies if
`body_name` is omitted).

### 22.1.2 ForceVisualizer

`ForceVisualizer` extracts simulation state from a MuJoCo model/data
pair:

| Method                 | Returns                                                           |
| ---------------------- | ----------------------------------------------------------------- |
| `get_contact_forces()` | List of dicts with position, normal, force magnitudes, body names |
| `get_joint_torques()`  | Dict mapping actuator name to torque value                        |
| `get_center_of_mass()` | Dict with CoM position and velocity vectors                       |

Contact forces are extracted via `mujoco.mj_contactForce(model, data, i, force)`,
which populates a 6-D vector: the first element is the normal force
magnitude and elements 1-2 are tangential friction forces.

### 22.1.3 Geometry Helpers

Three low-level functions populate `mjvGeom` structures for the 3-D
viewer:

#### `_init_arrow_geom(geom, start, end, radius, rgba)`

Creates a capsule (arrow shaft) aligned from `start` to `end`. The
rotation matrix is constructed so that the capsule's z-axis aligns with
the direction vector $\mathbf{d} = \text{end} - \text{start}$:

$$
\hat{z} = \frac{\mathbf{d}}{\|\mathbf{d}\|}, \qquad
  \hat{x} = \frac{\hat{z} \times \hat{e}}{\|\hat{z} \times \hat{e}\|}, \qquad
  \hat{y} = \hat{z} \times \hat{x}
$$

where $\hat{e}$ is an arbitrary reference vector chosen to avoid
degeneracy (the x-axis if $|\hat{z} \cdot \hat{x}| < 0.9$, otherwise
the y-axis).

The geometry is initialised via `mujoco.mjv_initGeom()` with type
`mjGEOM_CAPSULE`.

#### `_init_sphere_geom(geom, pos, radius, rgba)`

Creates a sphere at position `pos`. Used for torque visualisation where
the radius is proportional to torque magnitude: $r \propto |\tau|$.

#### `_init_line_geom(geom, p0, p1, radius, rgba)`

Creates a thin cylinder between two trajectory points. Implemented as
a delegate to `_init_arrow_geom` with a small radius.

### 22.1.4 Overlay Configuration

The `add_visualization_overlays()` entry point (not shown in full)
renders the following:

| Overlay                  | Geometry      | Colour Convention             |
| ------------------------ | ------------- | ----------------------------- | ---- | --- |
| Contact normal forces    | Arrow         | Red (`rgba=[1,0,0,0.8]`)      |
| Contact friction forces  | Arrow         | Orange (`rgba=[1,0.5,0,0.6]`) |
| Joint torques (positive) | Sphere        | Green, radius $\propto        | \tau | $   |
| Joint torques (negative) | Sphere        | Blue, radius $\propto         | \tau | $   |
| Actual trajectory        | Line segments | Per-body colour cycle         |
| Desired trajectory       | Line segments | Cyan, alternating-gap style   |

Toggles (`show_contacts`, `show_torques`, `show_trajectories`, etc.) and
scale factors (`force_scale`, `torque_scale`) are configurable at
runtime.

---

## 22.2 Matplotlib Plotting Framework

The plotting framework lives under `src/shared/python/plotting/` and is
structured as follows:

```
plotting/
    __init__.py
    config.py          # PlotConfig, ColorPalette
    base.py            # Base classes and interfaces
    core.py            # GolfSwingPlotter orchestrator
    animation.py       # SwingAnimator, AnimationConfig
    export.py          # ExportConfig, export_figure, export_plot_data
    kinematics.py      # Kinematics-specific helpers
    energy.py          # Energy-specific helpers
    transforms.py      # Coordinate transforms
    renderers/
        base.py        # BaseRenderer
        kinematics.py  # KinematicsRenderer
        kinetics.py    # KineticsRenderer
        energy.py      # EnergyRenderer
        club.py        # ClubRenderer
        signal.py      # SignalRenderer
        coordination.py# CoordinationRenderer
        stability.py   # StabilityRenderer
        comparison.py  # ComparisonRenderer
        dashboard.py   # DashboardRenderer
        vectors.py     # VectorOverlayRenderer
```

### 22.2.1 PlotConfig

`PlotConfig` is the single source of truth for all figure styling.
Every plotting function accepts an optional `PlotConfig`; the module
default `DEFAULT_CONFIG` is used when none is supplied.

| Attribute      | Type           | Default | Description                 |
| -------------- | -------------- | ------- | --------------------------- | ---------------------- |
| `width`        | `float`        | 10.0    | Figure width (inches)       |
| `height`       | `float`        | 6.0     | Figure height (inches)      |
| `dpi`          | `int`          | 100     | Dots per inch               |
| `line_width`   | `float`        | 1.5     | Default line width          |
| `marker_size`  | `float`        | 20.0    | Default scatter marker size |
| `title_size`   | `float`        | 14.0    | Title font size             |
| `label_size`   | `float`        | 12.0    | Axis label font size        |
| `tick_size`    | `float`        | 10.0    | Tick label font size        |
| `legend_size`  | `float`        | 10.0    | Legend font size            |
| `show_grid`    | `bool`         | `True`  | Display grid lines          |
| `grid_alpha`   | `float`        | 0.3     | Grid line opacity           |
| `tight_layout` | `bool`         | `True`  | Apply tight-layout          |
| `colors`       | `ColorPalette` | default | Colour palette instance     |
| `style`        | `str           | None`   | `None`                      | Matplotlib style sheet |

Helper methods:

- `create_figure(nrows, ncols, **kwargs)` -- Returns a pre-styled
  `(Figure, Axes)` tuple.
- `apply_to_axes(ax)` -- Apply tick sizes and grid settings to an
  existing Axes.

#### Preset Factories

| Factory                     | Purpose               | Key Differences                  |
| --------------------------- | --------------------- | -------------------------------- |
| `PlotConfig.presentation()` | Slides / posters      | 14x8 in, DPI 150, large fonts    |
| `PlotConfig.publication()`  | Journal figures       | 7x4.5 in, DPI 300, small fonts   |
| `PlotConfig.dashboard()`    | Multi-panel real-time | 5x3.5 in, DPI 100, compact fonts |

### 22.2.2 ColorPalette

`ColorPalette` is a frozen (immutable) dataclass providing named colours
and an indexed cycle:

```python
palette = ColorPalette()
palette.primary     # "#1f77b4"
palette.secondary   # "#ff7f0e"
palette.get_color(0)  # "#1f77b4"
palette.get_color(8)  # wraps to "#1f77b4"
```

The cycle contains eight colours: primary through senary, plus accent
and dark. `get_color(index)` wraps around automatically.

### 22.2.3 Specialised Renderers

All renderers inherit from `BaseRenderer` and receive recorded
simulation data and a colour dictionary. The following eleven renderers
are available:

#### KinematicsRenderer

Generates plots of joint-space motion:

- Joint angles vs. time (degrees)
- Joint angular velocities vs. time
- Joint angular accelerations vs. time
- Phase diagrams (angle vs. angular velocity)
- 3-D phase-space trajectories
- Poincare maps (stroboscopic sections)

#### KineticsRenderer

Generates dynamics-related plots:

- Joint torques vs. time (per actuator)
- Joint powers vs. time ($P = \tau \cdot \dot{q}$)
- Work loops (torque vs. angle, closed curves)
- Joint stiffness profiles
- 3-D angular momentum vector

#### EnergyRenderer

Plots energy flow through the system:

- Kinetic energy (KE), potential energy (PE), and total energy vs. time
- Power flow diagrams (rate of energy change)
- Segmental energy breakdown by body segment

#### ClubRenderer

Golf-specific club analysis:

- Club head speed vs. time (m/s and mph)
- 3-D club head trajectory
- Swing plane fitting (least-squares plane fit to trajectory points)

#### SignalRenderer

Advanced signal analysis visualisations:

- Jerk profiles (time derivative of acceleration)
- Power Spectral Density (PSD) using Welch's method
- Spectrograms (time-frequency representation)
- Wavelet scalograms (continuous wavelet transform)
- Lyapunov exponent estimation (orbital divergence)

#### CoordinationRenderer

Inter-joint coordination analysis:

- Coupling angle plots (relative phase between joints)
- Cross-correlation matrix (heatmap)
- Synergy trajectory plots (principal component projections)

#### StabilityRenderer

Balance and stability analysis:

- Centre of Pressure (CoP) trajectory
- CoM-CoP distance vs. time
- Ground Reaction Force (GRF) butterfly plots
- Local dynamic stability metrics
- 3-D vector field plots (velocity and force fields)

#### ComparisonRenderer

Counterfactual analysis tools:

- ZTCF (Zero-Torque Counterfactual) vs. ZVCF (Zero-Velocity
  Counterfactual) comparison plots
- Side-by-side actual vs. counterfactual trajectories

#### DashboardRenderer

Summary visualisation:

- 6-subplot summary grid (club speed, energy, angular momentum, joint
  angles, CoP, ground reaction forces)
- Radar/spider charts for multi-metric performance profiles

#### VectorOverlayRenderer

Spatial vector overlays:

- Force arrow plots in 2-D/3-D
- Torque vector plots
- Trajectory comparison with desired overlay
- GRF vector visualisation at foot contacts

---

## 22.3 Animation System

The animation subsystem in `src/shared/python/plotting/animation.py`
creates matplotlib `FuncAnimation` objects from simulation recordings.

### 22.3.1 AnimationConfig

| Attribute        | Type               | Default     | Description                              |
| ---------------- | ------------------ | ----------- | ---------------------------------------- |
| `fps`            | `int`              | 30          | Frames per second                        |
| `interval_ms`    | `int`              | 0           | Override ms between frames (0 = use fps) |
| `trail_length`   | `int`              | 60          | Number of past positions to keep visible |
| `figsize`        | `(float, float)`   | (10.0, 8.0) | Figure size in inches                    |
| `dpi`            | `int`              | 100         | Output resolution                        |
| `show_vectors`   | `bool`             | `True`      | Overlay force/velocity vectors           |
| `vector_scale`   | `float`            | 0.01        | Vector magnitude scale factor            |
| `skeleton_links` | `list[(str, str)]` | `[]`        | Body pairs for stick-figure links        |
| `desired_color`  | `str`              | `"#00CED1"` | Desired trajectory colour (cyan)         |
| `actual_color`   | `str`              | `"#FF4500"` | Actual trajectory colour (red-orange)    |

The `effective_interval` property computes `max(1, int(1000 / fps))`
when `interval_ms` is zero.

### 22.3.2 SwingAnimator

`SwingAnimator` is constructed with a `RecorderInterface` and an optional
`AnimationConfig`.

#### `create_trajectory_animation(body_names, desired_positions)`

Creates a 3-D trajectory playback animation. For each named body, the
recorded positions are retrieved via `recorder.get_time_series()` and
plotted with a trailing line of length `trail_length`.

If `desired_positions` is provided (a dictionary mapping body names to
`(N, 3)` arrays), the desired paths are drawn as semi-transparent static
lines for overlay comparison.

Axis limits are computed from the data extents with a 10% margin. A
timestamp annotation updates on each frame.

#### `create_stick_figure_animation(body_positions, links)`

Animates a skeleton by connecting body positions via straight lines.
Each entry in `links` is a `(body_a, body_b)` pair. The animation
updates all link lines on each frame.

#### `create_vector_field_animation(positions, vectors, times, label)`

Animates time-varying vectors (forces, velocities) at spatial positions
using matplotlib's `quiver` plotting. The old quiver is removed and
replaced on each frame, and a time annotation is displayed.

#### `save_animation(anim, path, writer, fps, dpi)` (Static)

Convenience wrapper around `anim.save()`. Supports:

| Writer     | File Extension | Requirement      |
| ---------- | -------------- | ---------------- |
| `"ffmpeg"` | `.mp4`         | FFmpeg installed |
| `"pillow"` | `.gif`         | Pillow installed |

The output directory is created automatically if it does not exist.

---

## 22.4 Export System

The export module (`src/shared/python/plotting/export.py`) provides
functions for persisting figures and their underlying data.

### 22.4.1 ExportConfig

| Attribute          | Type   | Default   | Description                     |
| ------------------ | ------ | --------- | ------------------------------- | --------------------- |
| `output_dir`       | `str   | Path`     | `"exports"`                     | Root export directory |
| `image_format`     | `str`  | `"png"`   | Default raster format           |
| `vector_format`    | `str`  | `"pdf"`   | Default vector format           |
| `dpi`              | `int`  | 300       | Raster export resolution        |
| `transparent`      | `bool` | `False`   | Transparent background          |
| `bbox_inches`      | `str`  | `"tight"` | Bounding box mode               |
| `include_metadata` | `bool` | `True`    | Embed timestamp and source info |

### 22.4.2 `export_figure`

```python
paths = export_figure(fig, name="club_speed", config=cfg)
# Returns list of Paths: [exports/club_speed.png, exports/club_speed.pdf]
```

Saves a matplotlib `Figure` to one or more formats. By default both the
raster and vector formats from the config are used. Custom formats can
be passed via the `formats` parameter.

### 22.4.3 `export_plot_data`

```python
path = export_plot_data(
    data={"time": times, "speed": speeds},
    name="club_speed_data",
    config=cfg,
    fmt="json",   # or "csv"
)
```

Exports the raw numerical data behind a plot.

- **JSON** mode: wraps the data in a dictionary, converting NumPy arrays
  to lists. If `include_metadata` is set, a `_meta` key is added with
  the export timestamp and source identifier (`"UpstreamDrift"`).

- **CSV** mode: flattens arrays into columns. Two-dimensional arrays are
  split into separate columns named `key_0`, `key_1`, etc. The first
  row is the header.

### 22.4.4 `export_all_figures`

```python
results = export_all_figures(
    figures={"speed": fig1, "energy": fig2},
    config=cfg,
)
# results == {"speed": [Path(...), ...], "energy": [Path(...), ...]}
```

Batch export of multiple named figures, calling `export_figure()` for
each entry.

---

## 22.5 Dashboard

The Qt-based dashboard in `src/shared/python/dashboard/` provides an
interactive GUI for simulation control, live plotting, post-hoc analysis,
and data export.

### 22.5.1 Architecture

```
dashboard/
    window.py         # UnifiedDashboardWindow (QMainWindow)
    widgets.py        # LivePlotWidget, ControlPanel, FrequencyAnalysisDialog
    runner.py         # SimulationRunner (QThread)
    recorder.py       # GenericPhysicsRecorder
    launcher.py       # launch_dashboard() entry point
    advanced_analysis.py  # Counterfactual analysis dialog
```

### 22.5.2 UnifiedDashboardWindow

The main application window (`QMainWindow`) is divided into two panels:

**Left panel:**

- `ControlPanel` -- Start, Stop, Reset buttons for simulation control.
- `LivePlotWidget` -- Real-time metric plotting with a metric selector
  (positions, velocities, torques, ZTCF, induced accelerations, frequency
  analysis).

**Right panel** (tabbed):

- **Plotting** tab -- Drop-down menu for selecting plot types (summary
  dashboard, joint angles, torques, energies, club analysis, etc.) with
  a `MplCanvas` rendering area.
- **Counterfactuals** tab -- ZTCF/ZVCF counterfactual analysis dialog.
- **Export** tab -- Export simulation data to multiple formats (CSV, JSON,
  MAT, C3D, HDF5).

The window resolves joint names from the engine if available and passes
them to both the live plot widget and the plotter for labelling.

### 22.5.3 LivePlotWidget

`LivePlotWidget` (`QWidget`) displays real-time plots that update on each
simulation frame. It supports:

- Multiple metric types via combo-box selection.
- Per-joint plotting with individual joint selectors.
- Parametric (X-Y) plotting.
- Frequency-domain analysis via `FrequencyAnalysisDialog` (PSD computed
  using Welch's method, displayed in dB/Hz with semilog scaling).

The widget embeds a `MplCanvas` (matplotlib `FigureCanvasQTAgg`
subclass) for rendering.

### 22.5.4 ControlPanel

`ControlPanel` provides three buttons:

| Button | Action                          |
| ------ | ------------------------------- |
| Start  | Start `SimulationRunner` thread |
| Stop   | Stop simulation loop            |
| Reset  | Reset engine state and recorder |

### 22.5.5 SimulationRunner

`SimulationRunner` is a `QThread` that runs the physics simulation loop
in a background thread, preventing the GUI from freezing.

Signals emitted:

| Signal                | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `frame_ready`         | Emitted after each physics step; triggers GUI update |
| `simulation_finished` | Emitted when simulation ends or is stopped           |
| `status_message(str)` | Status text for the status bar                       |

The runner targets a configurable frame rate (default 60 FPS) by sleeping
between steps. A maximum step count (default 10,000) prevents infinite
runs.

### 22.5.6 GenericPhysicsRecorder

`GenericPhysicsRecorder` records simulation data from any engine
implementing the `PhysicsEngine` interface.

| Parameter          | Default | Description                                       |
| ------------------ | ------- | ------------------------------------------------- |
| `max_samples`      | 100,000 | Hard ceiling on buffer size                       |
| `initial_capacity` | 1,000   | Starting buffer size                              |
| `growth_factor`    | 1.5     | Buffer growth multiplier when capacity is reached |

When the buffer is full, recording stops automatically with a warning.
The recorder supports real-time counterfactual analysis configuration
via `analysis_config` (ZTCF, ZVCF, drift tracking, induced
accelerations).

### 22.5.7 Launching the Dashboard

Engine-specific launcher scripts in `src/launchers/` provide a
one-command entry point:

```python
# src/launchers/mujoco_dashboard.py
from src.shared.python.dashboard.launcher import launch_dashboard
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
    MuJoCoPhysicsEngine,
)

launch_dashboard(engine_class=MuJoCoPhysicsEngine, title="MuJoCo Dashboard")
```

Similar launchers exist for Drake and Pinocchio engines.

---
