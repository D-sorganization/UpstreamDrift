# Chapter 25: Tools and Utilities

The `src/tools/` package and `src/shared/python/` provide utility
modules for model generation, model exploration, video analysis, injury
risk assessment, signal processing, and spatial algebra.

---

## 25.1 Model Generation

The model generation subsystem (`src/tools/model_generation/`) is
organised into eight sub-packages:

```
model_generation/
    builders/     # Parametric model building
    humanoid/     # Humanoid-specific generation
    mesh/         # Mesh generation and assembly
    inertia/      # Inertia tensor computation
    export/       # URDF, MJCF XML export
    core/         # Core data types and utilities
    library/      # Pre-built model templates
    converters/   # Format conversion (URDF <-> MJCF)
    plugins/      # Extension points
    editor/       # Interactive model editing
    cli/          # Command-line interface
    api/          # Programmatic generation API
```

### 25.1.1 Inertia Computation

The `inertia/` sub-package provides analytical formulae for computing
inertia tensors of standard geometric primitives.

#### Cylinder

For a solid cylinder of mass $m$, radius $r$, and height $h$ aligned
along the z-axis:

$$I_{xx} = I_{yy} = \frac{m}{12}(3r^2 + h^2), \qquad I_{zz} = \frac{mr^2}{2}$$

```python
from model_generation.inertia import cylinder_inertia
result = cylinder_inertia(mass=2.0, radius=0.05, length=0.3, axis="z")
# result == {"ixx": ..., "iyy": ..., "izz": ..., "ixy": 0, "ixz": 0, "iyz": 0}
```

When the cylinder axis is changed (e.g., `axis="x"`), the diagonal
entries are permuted accordingly so that the axial moment of inertia
aligns with the specified axis.

#### Sphere

For a solid sphere of mass $m$ and radius $r$:

$$I_{xx} = I_{yy} = I_{zz} = \frac{2mr^2}{5}$$

```python
from model_generation.inertia import sphere_inertia
result = sphere_inertia(mass=1.0, radius=0.1)
```

#### Box (Cuboid)

For a solid box of mass $m$ and full side lengths $a$, $b$, $c$:

$$I_{xx} = \frac{m}{12}(b^2 + c^2), \qquad I_{yy} = \frac{m}{12}(a^2 + c^2), \qquad I_{zz} = \frac{m}{12}(a^2 + b^2)$$

```python
from model_generation.inertia import box_inertia
result = box_inertia(mass=5.0, size_x=0.3, size_y=0.2, size_z=0.1)
```

#### Capsule and Ellipsoid

`capsule_inertia(mass, radius, length)` and
`ellipsoid_inertia(mass, semi_x, semi_y, semi_z)` compute the inertia
for capsule (cylinder with hemispherical caps) and ellipsoid primitives
respectively.

#### Unified InertiaCalculator

`InertiaCalculator` provides a high-level interface supporting four
computation modes:

| Mode (`InertiaMode`) | Description                                   |
| -------------------- | --------------------------------------------- |
| `PRIMITIVE`          | Use analytical formulas for standard shapes   |
| `MESH_DENSITY`       | Compute from mesh volume with uniform density |
| `MESH_MASS`          | Compute from mesh volume with specified mass  |
| `MANUAL`             | User provides inertia tensor directly         |

Returns an `InertiaResult` containing `ixx`, `iyy`, `izz`, `ixy`,
`ixz`, `iyz`, and `mass`.

#### Spatial Inertia

The `spatial.py` module provides tools for spatial (6-D) inertia:

- `mcI(mass, com, inertia)` -- Construct the $6 \times 6$ spatial
  inertia matrix.
- `transform_spatial_inertia(I_spatial, T)` -- Transform inertia to a
  new reference frame using a homogeneous transformation matrix.
- `spatial_inertia_to_urdf(I_spatial)` -- Convert to URDF-compatible
  inertia representation.

### 25.1.2 Export

The `export/` sub-package generates model files in standard formats:

- **URDF** (Unified Robot Description Format) -- XML format for ROS
  and many physics engines.
- **MJCF** (MuJoCo XML) -- Native MuJoCo model format.

Both exporters accept the internal model representation and produce
validated XML output.

### 25.1.3 Builders

The `builders/` sub-package provides parametric model construction:

- Segment-by-segment body assembly.
- Joint type selection (revolute, prismatic, ball, free).
- Collision geometry attachment.
- Visual geometry attachment.
- Actuator definition.

### 25.1.4 Humanoid Generation

The `humanoid/` sub-package specialises model generation for humanoid
characters:

- Anthropometric scaling from body measurements.
- Standard joint configurations for human motion.
- Segment mass and inertia estimation from body mass (de Leva tables).

---

## 25.2 Model Explorer

The model explorer (`src/tools/model_explorer/`) provides an interactive
browser for inspecting robot and humanoid models:

- URDF file loading and parsing.
- Joint tree visualisation with kinematic structure display.
- Interactive joint angle sliders for posing.
- Collision and visual geometry rendering.
- Inertia property inspection.
- Bundled human models and reference meshes.

The visualisation widget (`visualization_widget.py`) renders 3-D model
previews and supports camera manipulation (orbit, pan, zoom).

---

## 25.3 Video Analyzer

The video analysis pipeline (`src/tools/video_analyzer/` and
`src/shared/python/video_pose_pipeline.py`) extracts pose data from
video recordings.

### 25.3.1 Pipeline Architecture

```
video_analyzer/
    video_processor.py     # Frame extraction, pre-processing
    pose_estimator.py      # MediaPipe/OpenPose integration
    analyzer.py            # Swing detection, joint angle computation
    types.py               # Data types for video analysis results
```

### 25.3.2 Video Processing Config

```python
config = VideoProcessingConfig(
    estimator_type="mediapipe",   # or "openpose"
    min_confidence=0.5,
    enable_temporal_smoothing=True,
)
pipeline = VideoPosePipeline(config)
```

### 25.3.3 Pose Estimation

The pipeline supports two backends:

| Backend   | Library   | Landmarks | Description                   |
| --------- | --------- | --------- | ----------------------------- |
| MediaPipe | mediapipe | 33 body   | Google's real-time pose model |
| OpenPose  | openpose  | 25 body   | CMU's multi-person detector   |

Each frame is processed to extract 2-D and 3-D landmark positions with
per-landmark confidence scores. Frames with average confidence below
`min_confidence` are marked as invalid.

### 25.3.4 Temporal Smoothing

When `enable_temporal_smoothing` is set, landmark positions are smoothed
across frames using a one-dimensional low-pass filter to reduce jitter.

### 25.3.5 Swing Sequence Detection

The analyzer (`analyzer.py`) includes heuristics for detecting golf
swing phases (address, backswing, transition, downswing, impact,
follow-through) from the sequence of body landmarks.

---

## 25.4 Injury Risk Assessment

The spinal load analysis module
(`src/shared/python/analysis/` area) provides injury risk
assessment for golf swing motions.

### 25.4.1 Spinal Load Analysis

Spinal loading is estimated by modelling the lumbar spine as a rigid
column subjected to the forces and torques generated during the swing.
Key metrics computed:

| Metric               | Unit | Description                       |
| -------------------- | ---- | --------------------------------- |
| Compressive force    | N    | Axial load on lumbar vertebrae    |
| Anterior shear force | N    | Forward-directed shear at L4/L5   |
| Lateral shear force  | N    | Side-directed shear at L4/L5      |
| Resultant moment     | N m  | Net moment about the lumbar spine |

### 25.4.2 Risk Thresholds

Risk thresholds are based on NIOSH (National Institute for Occupational
Safety and Health) guidelines and biomechanical literature:

| Threshold               | Value  | Interpretation                  |
| ----------------------- | ------ | ------------------------------- |
| Compressive force (AL)  | 3400 N | Action limit -- increased risk  |
| Compressive force (MPL) | 6400 N | Maximum permissible limit       |
| Shear force limit       | 1000 N | Threshold for shear injury risk |

Loads exceeding the action limit trigger a warning; loads exceeding the
maximum permissible limit indicate a high injury risk.

### 25.4.3 Usage

```python
from src.shared.python.analysis import spinal_load_analysis

results = spinal_load_analysis.analyze_spinal_loads(
    joint_angles=joint_angles,
    joint_torques=joint_torques,
    body_masses=segment_masses,
)
# results contains peak forces, risk flags, and time-series data
```

---

## 25.5 Signal Processing Toolkit

The signal processing module (`src/shared/python/signal_processing.py`)
provides a comprehensive set of functions for processing biomechanical
time-series data.

### 25.5.1 Filtering

| Function              | Description                                              |
| --------------------- | -------------------------------------------------------- |
| Low-pass filter       | Butterworth IIR filter for removing high-frequency noise |
| Band-pass filter      | Retain signal components within a frequency band         |
| Savitzky-Golay filter | Polynomial smoothing preserving peak shapes              |

The `savgol_filter` from SciPy is used for Savitzky-Golay smoothing,
which fits successive sub-sets of data points with a low-degree
polynomial by the method of linear least squares.

### 25.5.2 Numerical Differentiation

Two methods are available:

**Central differences:**

$$\dot{x}(t) \approx \frac{x(t + \Delta t) - x(t - \Delta t)}{2\,\Delta t}$$

$$\ddot{x}(t) \approx \frac{x(t + \Delta t) - 2\,x(t) + x(t - \Delta t)}{(\Delta t)^2}$$

**Savitzky-Golay differentiation:** Fits a polynomial to a window of
data points and differentiates the polynomial analytically. This
produces smoother derivatives than finite differences, especially for
noisy data.

### 25.5.3 Frequency Analysis

| Function          | Description                                  |
| ----------------- | -------------------------------------------- |
| `compute_psd()`   | Power Spectral Density via Welch's method    |
| Spectrogram       | Time-frequency representation via STFT       |
| Coherence         | Cross-spectral coherence between two signals |
| Cross-correlation | Sliding dot-product between signals          |

The PSD function returns frequencies and power values; the dashboard
converts to decibels: $P_{\text{dB}} = 10 \log_{10}(P + \epsilon)$ with
$\epsilon = 10^{-12}$.

### 25.5.4 Time Normalisation

Movement data can be normalised to a percentage scale (0-100%) to enable
comparison across trials of different durations. This is essential for
ensemble averaging and inter-subject comparison.

### 25.5.5 Feature Extraction

| Feature        | Description                                                   |
| -------------- | ------------------------------------------------------------- |
| Peak detection | Local maxima and minima with configurable prominence          |
| Zero crossings | Time points where signal changes sign                         |
| RMS            | Root mean square: $\text{RMS} = \sqrt{\frac{1}{N}\sum x_i^2}$ |
| DTW distance   | Dynamic Time Warping for shape comparison                     |

The DTW implementation includes an optional **Numba JIT** acceleration
(approximately 100x speedup) and an $\mathcal{O}(M)$ space optimisation
that stores only two rows of the cost matrix instead of the full
$\mathcal{O}(NM)$ matrix.

### 25.5.6 Continuous Wavelet Transform

The CWT decomposes a signal into time-scale components using Morlet
wavelets. Wavelet generation is cached with `functools.lru_cache` for
performance.

---

## 25.6 Spatial Algebra

Spatial algebra utilities are distributed across several modules in the
codebase (primarily `src/shared/python/reference_frames.py` and the
plotting transforms module).

### 25.6.1 Rotation Matrices

Elementary rotation matrices about the coordinate axes:

$$R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}$$

$$R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

$$R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 25.6.2 Euler Angle Conventions

Conversion between Euler angles (ZYX, XYZ, ZXZ, etc.) and $3 \times 3$
rotation matrices. The ZYX convention is the most common in the
codebase:

$$R_{\text{ZYX}}(\alpha, \beta, \gamma) = R_z(\alpha)\, R_y(\beta)\, R_x(\gamma)$$

Extraction of Euler angles from a rotation matrix handles the
gimbal-lock singularity when $\cos\beta = 0$ by falling back to a
two-angle decomposition.

### 25.6.3 Quaternion Operations

Quaternions are represented as $q = [w, x, y, z]$ (scalar-first
convention, consistent with MuJoCo).

| Operation          | Formula                                              |
| ------------------ | ---------------------------------------------------- |
| Multiply           | Hamilton product: $q_1 \otimes q_2$                  |
| Inverse            | $q^{-1} = \bar{q} / \|q\|^2$                         |
| To rotation matrix | See Section 21.4.4 (implemented in multiple modules) |
| SLERP              | Spherical linear interpolation (see Section 21.4.5)  |

**Quaternion to rotation matrix:**

$$R = \begin{bmatrix} 1 - 2(y^2 + z^2) & 2(xy - zw) & 2(xz + yw) \\ 2(xy + zw) & 1 - 2(x^2 + z^2) & 2(yz - xw) \\ 2(xz - yw) & 2(yz + xw) & 1 - 2(x^2 + y^2) \end{bmatrix}$$

This conversion appears in `FormationController`, `CooperativeManipulation`,
and the MuJoCo visualization module.

**SLERP (Spherical Linear Interpolation):**

$$q(t) = \frac{\sin((1 - t)\theta)}{\sin\theta}\, q_0 + \frac{\sin(t\theta)}{\sin\theta}\, q_1$$

where $\theta = \arccos(q_0 \cdot q_1)$. When the quaternions are
nearly identical ($\text{dot} > 0.9995$), linear interpolation with
normalisation is used as a numerically stable fallback. When the dot
product is negative, one quaternion is negated to take the short path.

### 25.6.4 Homogeneous Transformations

A rigid-body transformation is represented as a $4 \times 4$ homogeneous
matrix:

$$T = \begin{bmatrix} R & t \\ 0_{1 \times 3} & 1 \end{bmatrix}$$

where $R \in \text{SO}(3)$ is the rotation matrix and
$t \in \mathbb{R}^3$ is the translation vector.

Transformation of a point $p$ in the body frame to the world frame:

$$p_{\text{world}} = R\, p + t$$

Inverse transformation:

$$T^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$$

Composition of transformations:

$$T_{AC} = T_{AB}\, T_{BC}$$

### 25.6.5 Skew-Symmetric Matrix

The skew-symmetric (hat) operator converts a 3-D vector to a
$3 \times 3$ matrix that encodes the cross product:

$$[\mathbf{v}]_\times = \begin{bmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix}$$

This satisfies $[\mathbf{v}]_\times\, \mathbf{w} = \mathbf{v} \times \mathbf{w}$
for any vector $\mathbf{w}$.

The skew-symmetric matrix is used extensively in the grasp matrix
computation (Section 21.4.5), the MuJoCo arrow geometry alignment
(Section 22.1.3), and the soft-body FEM stress computation
(Section 21.3.3).

### 25.6.6 Rodrigues' Rotation Formula

Given a unit axis $\hat{k}$ and angle $\theta$, the rotation matrix is:

$$R = I + \sin\theta\, [\hat{k}]_\times + (1 - \cos\theta)\, [\hat{k}]_\times^2$$

This is equivalent to the matrix exponential $R = \exp(\theta\, [\hat{k}]_\times)$
and is used for axis-angle to rotation matrix conversion.

### 25.6.7 Parallel Axis Theorem

When transferring an inertia tensor from the centre of mass to a
parallel axis displaced by vector $d$:

$$I' = I_{\text{cm}} + m\, (d^T d\, I_3 - d\, d^T)$$

This is implemented in the spatial inertia module
(`transform_spatial_inertia`) and used during model generation to express
link inertias in joint frames.

---

## Summary of Key File Paths

| Component                | Path                                                                           |
| ------------------------ | ------------------------------------------------------------------------------ |
| MPC Controller           | `src/research/mpc/controller.py`                                               |
| MPC Specialisations      | `src/research/mpc/specialized.py`                                              |
| Differentiable Engine    | `src/research/differentiable/engine.py`                                        |
| Deformable Objects       | `src/research/deformable/objects.py`                                           |
| Multi-Robot System       | `src/research/multi_robot/system.py`                                           |
| Multi-Robot Coordination | `src/research/multi_robot/coordination.py`                                     |
| Plot Config              | `src/shared/python/plotting/config.py`                                         |
| Plot Renderers           | `src/shared/python/plotting/renderers/`                                        |
| Animation                | `src/shared/python/plotting/animation.py`                                      |
| Export                   | `src/shared/python/plotting/export.py`                                         |
| Dashboard Window         | `src/shared/python/dashboard/window.py`                                        |
| Dashboard Widgets        | `src/shared/python/dashboard/widgets.py`                                       |
| Simulation Runner        | `src/shared/python/dashboard/runner.py`                                        |
| Physics Recorder         | `src/shared/python/dashboard/recorder.py`                                      |
| MuJoCo 3-D Visualization | `src/engines/physics_engines/mujoco/docker/src/humanoid_golf/visualization.py` |
| API Server               | `src/api/server.py`                                                            |
| API Routes (Simulation)  | `src/api/routes/simulation.py`                                                 |
| API Routes (WebSocket)   | `src/api/routes/simulation_ws.py`                                              |
| API Routes (Engines)     | `src/api/routes/engines.py`                                                    |
| API Routes (Auth)        | `src/api/routes/auth.py`                                                       |
| API Routes (Export)      | `src/api/routes/export.py`                                                     |
| API Request Models       | `src/api/models/requests.py`                                                   |
| API Response Models      | `src/api/models/responses.py`                                                  |
| JWT Security             | `src/api/auth/security.py`                                                     |
| Motion Retargeting       | `src/learning/retargeting/retargeter.py`                                       |
| Model Generation         | `src/tools/model_generation/`                                                  |
| Inertia Primitives       | `src/tools/model_generation/inertia/primitives.py`                             |
| Model Explorer           | `src/tools/model_explorer/`                                                    |
| Video Analyzer           | `src/tools/video_analyzer/`                                                    |
| Video Pose Pipeline      | `src/shared/python/video_pose_pipeline.py`                                     |
| Signal Processing        | `src/shared/python/signal_processing.py`                                       |

---
