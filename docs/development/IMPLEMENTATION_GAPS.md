# Implementation Gaps Tracking Document

> **Generated**: 2026-02-05
> **Status**: Active tracking document for implementation gap management
> **Purpose**: GitHub issue creation reference and development prioritization

---

## Table of Contents

1. [Critical Issues (Must Fix)](#critical-issues-must-fix)
2. [Feature Gaps (Incomplete Implementation)](#feature-gaps-incomplete-implementation)
3. [Physics Model Gaps](#physics-model-gaps)
4. [UI/UX Gaps](#uiux-gaps)
5. [Code Quality Issues](#code-quality-issues)
6. [Summary Matrix](#summary-matrix)

---

## Critical Issues (Must Fix)

### CRITICAL-001: Kinematic Sequence Patent Risk

| Field | Value |
|-------|-------|
| **Issue Title** | `refactor(analysis): Replace TPI-patented kinematic sequence with neutral SegmentTimingAnalyzer` |
| **Severity** | CRITICAL |
| **Labels** | `legal`, `refactor`, `breaking-change` |

**Affected Files/Modules:**
- `src/shared/python/kinematic_sequence.py`
- `src/api/services/analysis_service.py`
- `src/shared/python/plotting/renderers/coordination.py`
- `src/shared/python/plotting/core.py`
- `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/advanced_gui_methods.py`
- `tests/unit/test_kinematic_sequence.py`

**Description:**

The current `KinematicSequenceAnalyzer` class implements analysis methods that closely mirror TPI (Titleist Performance Institute) patented "kinematic sequence" analysis. Key concerns:
- Uses exact terminology ("Pelvis", "Torso", "Arm", "Club" ordering)
- Implements peak velocity detection with TPI-style timing analysis
- Speed gain ratios match patented methodology

**Recommended Fix:**

1. Rename class to `SegmentTimingAnalyzer`
2. Use generic terminology: "proximal_segment", "intermediate_segment", "distal_segment"
3. Document as "general biomechanical segment timing analysis" without golf-specific claims
4. Remove any references to "kinematic sequence" as a branded term
5. Implement as generic proximal-to-distal timing analysis applicable to any multi-segment motion

**Estimated Effort:** 3-5 days (includes test updates and documentation)

---

### CRITICAL-002: XML Entity Expansion Vulnerability (B314)

| Field | Value |
|-------|-------|
| **Issue Title** | `security(xml): Replace xml.etree.ElementTree with defusedxml to prevent XXE attacks` |
| **Severity** | CRITICAL |
| **Labels** | `security`, `jules:sentinel`, `cve-risk` |

**Affected Files/Modules:**
- `src/tools/model_explorer/urdf_builder.py`
- `src/tools/model_explorer/visualization_widget.py`
- `src/tools/model_explorer/urdf_code_editor.py`
- `src/tools/model_explorer/mesh_browser.py`
- `src/tools/model_explorer/joint_manipulator.py`
- `src/tools/model_explorer/frankenstein_editor.py`
- `src/tools/model_explorer/end_effector_manager.py`
- `src/tools/model_explorer/chain_manipulation.py`
- `src/tools/model_explorer/component_library.py`
- `src/tools/model_generation/converters/urdf_parser.py`
- `src/tools/model_generation/converters/mjcf_converter.py`
- `src/tools/model_generation/converters/format_utils.py`
- `src/tools/model_generation/editor/text_editor.py`
- `src/tools/humanoid_character_builder/generators/urdf_generator.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/urdf_io.py`
- `src/engines/physics_engines/drake/python/src/drake_golf_model.py`

**Description:**

Using `xml.etree.ElementTree.fromstring()` to parse untrusted XML data exposes the application to:
- Billion Laughs attack (exponential entity expansion)
- External Entity Injection (XXE)
- Denial of Service through malformed XML

**Recommended Fix:**

```python
# Before (vulnerable)
import xml.etree.ElementTree as ET
root = ET.fromstring(xml_content)

# After (secure)
import defusedxml.ElementTree as ET
root = ET.fromstring(xml_content)
```

Note: `defusedxml>=0.7.0` is already in optional dependencies (`[urdf]`). Add to core dependencies and migrate all XML parsing.

**Estimated Effort:** 2-3 days

---

### CRITICAL-003: Insecure Temporary File Usage (B108/B377)

| Field | Value |
|-------|-------|
| **Issue Title** | `security(files): Replace hardcoded /tmp paths with secure tempfile.mkdtemp()` |
| **Severity** | CRITICAL |
| **Labels** | `security`, `jules:sentinel` |

**Affected Files/Modules:**
- `src/engines/physics_engines/opensim/python/opensim_physics_engine.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py`
- `src/tools/model_generation/api/rest_api.py`
- `src/tools/model_generation/library/repository.py`
- `src/tools/humanoid_character_builder/interfaces/api.py`
- `src/api/routes/video.py`
- `src/launchers/docker_manager.py`
- `src/shared/python/test_utils.py`
- `src/tools/model_explorer/mujoco_viewer.py`
- `src/shared/python/signal_toolkit/tests/test_signal_toolkit.py`

**Description:**

Hardcoded `/tmp` paths create security vulnerabilities:
- Race conditions (TOCTOU attacks)
- Symlink attacks
- Predictable file names enabling targeted attacks
- Shared namespace on multi-user systems

**Recommended Fix:**

```python
# Before (insecure)
tmp_path = "/tmp/my_file.xml"
with open(tmp_path, "w") as f:
    f.write(content)

# After (secure)
import tempfile
with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
    f.write(content)
    tmp_path = f.name
try:
    # use tmp_path
finally:
    os.unlink(tmp_path)
```

**Estimated Effort:** 2-3 days

---

### CRITICAL-004: Binding All Network Interfaces (B104)

| Field | Value |
|-------|-------|
| **Issue Title** | `security(network): Replace 0.0.0.0 binding with explicit interface configuration` |
| **Severity** | CRITICAL |
| **Labels** | `security`, `jules:sentinel`, `api` |

**Affected Files/Modules:**
- `src/api/server.py` (primary)
- `src/engines/physics_engines/mujoco/python/golf_suite_launcher.py`
- `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py`
- `src/engines/physics_engines/drake/python/src/drake_gui_app.py`
- `src/launchers/docker_manager.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/meshcat_adapter.py`

**Description:**

Binding to `0.0.0.0` exposes services to all network interfaces, potentially allowing unauthorized access from:
- External networks
- Other containers in shared environments
- Compromised adjacent systems

**Recommended Fix:**

```python
# Before (insecure)
uvicorn.run(app, host="0.0.0.0", port=8000)

# After (secure - configurable)
import os
host = os.environ.get("API_HOST", "127.0.0.1")
uvicorn.run(app, host=host, port=8000)
```

Add environment variable documentation and default to localhost for development.

**Estimated Effort:** 1 day

---

### CRITICAL-005: URL Open Without Scheme Validation (B310)

| Field | Value |
|-------|-------|
| **Issue Title** | `security(network): Add URL scheme validation for urllib.request.urlopen()` |
| **Severity** | HIGH |
| **Labels** | `security`, `jules:sentinel` |

**Affected Files/Modules:**
- `src/tools/model_generation/library/repository.py`
- `src/tools/model_generation/library/model_library.py`
- `src/tools/model_explorer/model_library.py`
- `src/shared/python/standard_models.py`

**Description:**

Using `urllib.request.urlopen()` without validating URL schemes can lead to:
- SSRF (Server-Side Request Forgery) via `file://` scheme
- Protocol smuggling via non-HTTP schemes
- Information disclosure through local file access

**Recommended Fix:**

```python
from urllib.parse import urlparse

def safe_urlopen(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    return urllib.request.urlopen(url)
```

**Estimated Effort:** 1 day

---

### CRITICAL-006: SQL Injection Risk (B608)

| Field | Value |
|-------|-------|
| **Issue Title** | `security(database): Replace string concatenation with parameterized queries` |
| **Severity** | CRITICAL |
| **Labels** | `security`, `jules:sentinel`, `database` |

**Affected Files/Modules:**
- `src/api/database.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/recording_library.py`

**Description:**

String-based SQL query construction allows injection attacks:
```python
# Vulnerable pattern
query = f"SELECT * FROM users WHERE name = '{user_input}'"
```

**Recommended Fix:**

```python
# Secure pattern (parameterized)
query = "SELECT * FROM users WHERE name = ?"
cursor.execute(query, (user_input,))
```

**Estimated Effort:** 1-2 days

---

### CRITICAL-007: Subprocess Shell=True (B604)

| Field | Value |
|-------|-------|
| **Issue Title** | `security(subprocess): Remove shell=True from subprocess calls` |
| **Severity** | HIGH |
| **Labels** | `security`, `jules:sentinel` |

**Affected Files/Modules:**
- `src/shared/python/secure_subprocess.py`
- `scripts/check_integrations.py`

**Description:**

Using `subprocess.run(..., shell=True)` enables shell injection attacks when user input is included in commands.

**Recommended Fix:**

```python
# Before (vulnerable)
subprocess.run(f"process {user_file}", shell=True)

# After (secure)
subprocess.run(["process", user_file], shell=False)
```

Note: `secure_subprocess.py` exists but may not be used consistently.

**Estimated Effort:** 1 day

---

### CRITICAL-008: Use of exec() (B102)

| Field | Value |
|-------|-------|
| **Issue Title** | `security(code): Replace exec() with safe alternatives` |
| **Severity** | HIGH |
| **Labels** | `security`, `jules:sentinel` |

**Affected Files/Modules:**
- Multiple UI files using `exec()` for dynamic widget creation
- `src/shared/python/dashboard/widgets.py`

**Description:**

`exec()` enables arbitrary code execution. Even with controlled inputs, it's a security anti-pattern.

**Recommended Fix:**

Use `simpleeval` library (already in dependencies) or factory patterns instead of dynamic code execution.

**Estimated Effort:** 2-3 days

---

### CRITICAL-009: Permissive File Permissions (B103)

| Field | Value |
|-------|-------|
| **Issue Title** | `security(files): Replace permissive chmod with restrictive permissions` |
| **Severity** | MEDIUM |
| **Labels** | `security`, `jules:sentinel` |

**Affected Files/Modules:**
- `scripts/migrate_api_keys.py`

**Description:**

Using `chmod 777` or similar permissive modes exposes files to all users on the system.

**Recommended Fix:**

```python
# Before (insecure)
os.chmod(path, 0o777)

# After (secure)
os.chmod(path, 0o600)  # Owner read/write only
```

**Estimated Effort:** 0.5 days

---

## Feature Gaps (Incomplete Implementation)

### FEATURE-001: OpenSim Engine - Stub Implementation

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(engine): Complete OpenSim physics engine forward/inverse dynamics` |
| **Severity** | HIGH |
| **Labels** | `enhancement`, `engine`, `opensim` |

**Affected Files/Modules:**
- `src/engines/physics_engines/opensim/python/opensim_physics_engine.py`
- `src/engines/physics_engines/opensim/python/muscle_analysis.py`
- `src/engines/physics_engines/opensim/python/opensim_golf/core.py`

**Description:**

The OpenSim physics engine has a functional wrapper but limited dynamics:
- Forward dynamics rely on OpenSim's internal integrator (not configurable)
- Inverse dynamics implemented but untested with complex models
- Muscle analysis stubs exist but need validation
- Jacobian computation uses numerical differentiation (slow, inaccurate)

**Current State:**
- Model loading: Working
- State get/set: Working
- Forward dynamics: Partial (uses OpenSim Manager)
- Inverse dynamics: Implemented but limited
- Mass matrix: Working via SimbodyMatterSubsystem
- Jacobian: Numerical finite differences (not analytical)

**Recommended Fix:**

1. Implement analytical Jacobian via SimbodyMatterSubsystem calcSystemJacobian
2. Add configurable integration methods
3. Validate inverse dynamics against known solutions
4. Complete muscle-induced acceleration analysis

**Estimated Effort:** 2-3 weeks

---

### FEATURE-002: MyoSuite Engine - 290-Muscle Model Integration

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(engine): Complete MyoSuite 290-muscle model dynamics integration` |
| **Severity** | HIGH |
| **Labels** | `enhancement`, `engine`, `myosuite`, `muscles` |

**Affected Files/Modules:**
- `src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py`
- `src/engines/physics_engines/myosuite/python/muscle_analysis.py`
- `src/shared/models/myosuite/myo_sim/`

**Description:**

MyoSuite integration works for basic simulations but lacks:
- Full 290-muscle model activation dynamics
- Muscle-induced acceleration decomposition
- Proper Gym/MuJoCo protocol bridging for custom dt
- Gravity force isolation

**Current State:**
- Environment loading: Working (via Gym make)
- Basic stepping: Working (sim.step())
- Muscle activations: Interface exists, not fully tested
- compute_gravity_forces(): Returns empty array
- Custom timestep: Hacky implementation

**Recommended Fix:**

1. Implement proper muscle state extraction from MjData
2. Add gravity force computation via MuJoCo APIs
3. Create muscle-induced acceleration analysis
4. Document Gym action space mapping for control

**Estimated Effort:** 2-3 weeks

---

### FEATURE-003: Drift-Control Decomposition - Cross-Engine Parity

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(physics): Implement drift-control decomposition in all physics engines` |
| **Severity** | HIGH |
| **Labels** | `enhancement`, `physics`, `guideline-F` |

**Affected Files/Modules:**
- `src/engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py` (complete)
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/drift_control.py` (complete)
- `src/engines/physics_engines/drake/python/drake_physics_engine.py` (partial)
- `src/engines/physics_engines/opensim/python/opensim_physics_engine.py` (basic)
- `src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py` (basic)

**Description:**

Project design guideline Section F requires drift-control decomposition (ZTCF/ZVCF) in all engines. Currently:
- Pinocchio: Full implementation with ABA
- MuJoCo: Full implementation
- Drake: Basic implementation
- OpenSim: Basic (using forward dynamics)
- MyoSuite: Basic (using forward dynamics)

**Recommended Fix:**

1. Validate Drake implementation against Pinocchio
2. Add direct bias computation for OpenSim/MyoSuite
3. Create cross-engine verification tests
4. Document numerical tolerances for comparisons

**Estimated Effort:** 1-2 weeks

---

### FEATURE-004: Constrained Systems Support

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(physics): Add complex joint constraint support (4-bar linkages, closed loops)` |
| **Severity** | MEDIUM |
| **Labels** | `enhancement`, `physics`, `constraints` |

**Affected Files/Modules:**
- All physics engine implementations
- `src/shared/python/interfaces.py`

**Description:**

Current implementation supports:
- Simple revolute joints
- Prismatic joints
- Ball joints
- Fixed joints

Missing support for:
- 4-bar linkages (club shaft flexibility model)
- Closed kinematic loops
- Holonomic constraints
- Contact constraints with friction cones

**Recommended Fix:**

1. Extend PhysicsEngine interface with constraint methods
2. Implement constraint handling per engine capabilities
3. Add constraint validation in URDF/MJCF loading
4. Document constraint support matrix

**Estimated Effort:** 3-4 weeks

---

### FEATURE-005: Closed-Form IK Solutions

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(kinematics): Implement analytical IK for common arm configurations` |
| **Severity** | MEDIUM |
| **Labels** | `enhancement`, `kinematics`, `performance` |

**Affected Files/Modules:**
- `src/engines/physics_engines/pinocchio/python/motion_training/dual_hand_ik_solver.py`
- `src/engines/physics_engines/pinocchio/python/dtack/ik/pink_solver.py`
- `src/shared/python/manipulability.py`

**Description:**

Current IK implementation uses iterative numerical methods (PINK/Pinocchio). For real-time applications, analytical solutions are faster for:
- 6-DOF arm configurations
- Dual-arm coordination
- Redundancy resolution

**Current State:**
- PINK-based iterative IK: Working
- Analytical IK: Not implemented

**Recommended Fix:**

1. Implement Pieper's solution for 6-DOF arms
2. Add configuration-space optimization for redundant arms
3. Create hybrid solver (analytical + numerical refinement)
4. Benchmark against iterative methods

**Estimated Effort:** 2-3 weeks

---

## Physics Model Gaps

### PHYSICS-001: Ball Spin Decay Model

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(physics): Implement realistic ball spin decay model` |
| **Severity** | MEDIUM |
| **Labels** | `enhancement`, `physics`, `ball-flight` |

**Affected Files/Modules:**
- `src/shared/python/ball_flight_physics.py`
- `src/shared/python/flight_models.py`
- `src/shared/python/aerodynamics.py`

**Description:**

Current implementation uses constant spin or simple linear decay. Real ball spin decay involves:
- Magnus force interaction
- Surface roughness effects
- Dimple pattern aerodynamics
- Spin axis precession

**Recommended Fix:**

1. Implement empirical spin decay model from golf ball literature
2. Add dimple-dependent drag coefficients
3. Validate against TrackMan/FlightScope data
4. Document physical assumptions

**Estimated Effort:** 1-2 weeks

---

### PHYSICS-002: Shaft Frequency Response

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(physics): Complete shaft frequency response model` |
| **Severity** | MEDIUM |
| **Labels** | `enhancement`, `physics`, `equipment` |

**Affected Files/Modules:**
- `src/shared/python/physics_parameters.py`
- `src/engines/common/physics.py`
- `src/engines/physics_engines/putting_green/python/putter_stroke.py`

**Description:**

Shaft flex modeling is incomplete:
- Static deflection: Implemented
- Dynamic response: Partial
- Frequency-dependent damping: Missing
- Kick point effects: Not modeled

**Recommended Fix:**

1. Add Euler-Bernoulli beam model for shaft
2. Implement modal superposition for dynamics
3. Include damping ratio parameters
4. Validate against shaft testing data

**Estimated Effort:** 2-3 weeks

---

### PHYSICS-003: Impact MOI Approximation

| Field | Value |
|-------|-------|
| **Issue Title** | `fix(physics): Improve impact MOI calculation accuracy` |
| **Severity** | MEDIUM |
| **Labels** | `bug`, `physics`, `impact` |

**Affected Files/Modules:**
- `src/shared/python/impact_model.py`
- `src/engines/common/physics.py`

**Description:**

Current impact MOI uses point-mass approximation which is inaccurate for:
- Off-center hits (gear effect)
- High-MOI club heads
- Putter face inserts

**Recommended Fix:**

1. Implement full rigid body impact dynamics
2. Add face normal computation for angled impacts
3. Include club head mass distribution
4. Validate against high-speed camera data

**Estimated Effort:** 1-2 weeks

---

### PHYSICS-004: Gear Effect Model

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(physics): Replace heuristic gear effect with physics-based model` |
| **Severity** | MEDIUM |
| **Labels** | `enhancement`, `physics`, `impact` |

**Affected Files/Modules:**
- `src/shared/python/impact_model.py`
- `src/shared/python/ball_flight_physics.py`

**Description:**

Current gear effect uses lookup tables/heuristics. Physics-based model should include:
- Club head MOI tensor
- Impact point location
- Face bulge/roll geometry
- Ball deformation effects

**Recommended Fix:**

1. Implement rigid body angular momentum transfer
2. Add face curvature parameters
3. Calculate induced spin from moment arm
4. Validate against launch monitor data

**Estimated Effort:** 2 weeks

---

### PHYSICS-005: Eccentric Work - Muscle Efficiency

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(physics): Implement muscle eccentric work efficiency model` |
| **Severity** | LOW |
| **Labels** | `enhancement`, `physics`, `biomechanics` |

**Affected Files/Modules:**
- `src/shared/python/muscle_analysis.py`
- `src/engines/physics_engines/opensim/python/muscle_analysis.py`
- `src/engines/physics_engines/myosuite/python/muscle_analysis.py`

**Description:**

Current muscle models assume constant efficiency. Real muscles have:
- Velocity-dependent efficiency
- Eccentric vs concentric differences
- Fatigue effects
- Temperature dependence

**Recommended Fix:**

1. Implement Hill-type muscle model efficiency
2. Add eccentric efficiency factor (typically higher than concentric)
3. Include metabolic cost calculation
4. Validate against EMG data

**Estimated Effort:** 2-3 weeks

---

### PHYSICS-006: Air Density - Environmental Compensation

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(physics): Implement comprehensive air density model` |
| **Severity** | LOW |
| **Labels** | `enhancement`, `physics`, `environment` |

**Affected Files/Modules:**
- `src/shared/python/aerodynamics.py`
- `src/shared/python/physics_constants.py`
- `src/shared/python/flight_model_options.py`

**Description:**

Current air density uses simple altitude correction. Complete model needs:
- Temperature effects (ideal gas law)
- Humidity effects (water vapor)
- Barometric pressure
- Course-specific presets

**Recommended Fix:**

1. Implement full ISA (International Standard Atmosphere) model
2. Add psychrometric calculations for humidity
3. Create course database with typical conditions
4. Add real-time weather API integration option

**Estimated Effort:** 1 week

---

## UI/UX Gaps

### UX-001: Context-Sensitive Help System

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(ui): Add context-sensitive help buttons to all UI components` |
| **Severity** | MEDIUM |
| **Labels** | `enhancement`, `ui`, `documentation` |

**Affected Files/Modules:**
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/`
- `src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py`
- `src/engines/physics_engines/drake/python/src/drake_gui_app.py`
- `src/tools/model_explorer/main_window.py`
- `src/shared/python/dashboard/window.py`
- `src/engines/pendulum_models/python/double_pendulum_model/ui/`

**Description:**

No UI components have context-sensitive help. Users need:
- "?" buttons linking to relevant documentation
- Inline help tooltips
- Keyboard shortcut for help overlay (F1)
- Tutorial mode for new users

**Current State:**
- Some tooltips exist (see `test_ui_tooltips.py`)
- No help button infrastructure
- No F1 help system

**Recommended Fix:**

1. Create `HelpButton` widget component
2. Map help topics to documentation URLs
3. Implement F1 context help system
4. Add tutorial overlay mode

**Estimated Effort:** 1-2 weeks

---

### UX-002: Engine Parameter Panel Guidance

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(ui): Add parameter guidance tooltips to engine configuration panels` |
| **Severity** | MEDIUM |
| **Labels** | `enhancement`, `ui`, `usability` |

**Affected Files/Modules:**
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/physics_tab.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/tabs/controls_tab.py`
- `src/shared/python/dashboard/widgets.py`

**Description:**

Engine parameter panels lack user guidance:
- No explanation of what parameters do
- No valid range indicators
- No "reset to defaults" option
- No preset configurations

**Recommended Fix:**

1. Add tooltip descriptions for all parameters
2. Show valid ranges and units
3. Implement preset system (beginner/advanced/expert)
4. Add "restore defaults" functionality

**Estimated Effort:** 1 week

---

### UX-003: Simulation Controls Documentation

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(ui): Add documentation for simulation control parameters` |
| **Severity** | LOW |
| **Labels** | `enhancement`, `ui`, `documentation` |

**Affected Files/Modules:**
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/sim_widget.py`
- All GUI simulation controls

**Description:**

Users don't understand:
- Frame rate vs physics timestep relationship
- Duration settings impact on accuracy
- Substep count effects

**Recommended Fix:**

1. Add explanatory labels/tooltips
2. Create "simulation settings" help dialog
3. Show real-time performance metrics
4. Warn when settings may cause instability

**Estimated Effort:** 0.5 weeks

---

### UX-004: Interactive Plot Legends

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(ui): Add interactive legends to all plot types` |
| **Severity** | LOW |
| **Labels** | `enhancement`, `ui`, `visualization` |

**Affected Files/Modules:**
- `src/shared/python/plotting/core.py`
- `src/shared/python/plotting/renderers/`
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/ui/tabs/`

**Description:**

Plot legends should support:
- Click to hide/show series
- Double-click to isolate series
- Drag to reorder
- Color picker for customization

**Recommended Fix:**

1. Upgrade to plotly or bokeh for interactivity
2. Implement legend click handlers
3. Add series customization dialog
4. Persist user preferences

**Estimated Effort:** 1-2 weeks

---

### UX-005: Motion Retargeting Tab Parameters

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(ui): Add parameter explanations to motion retargeting interface` |
| **Severity** | LOW |
| **Labels** | `enhancement`, `ui`, `motion-training` |

**Affected Files/Modules:**
- `src/engines/physics_engines/pinocchio/python/motion_training/`
- Related GUI components

**Description:**

Motion retargeting parameters are unexplained:
- IK solver tolerances
- Smoothing factors
- Constraint weights
- Bone mapping options

**Recommended Fix:**

1. Add parameter documentation
2. Create visual feedback for retargeting quality
3. Implement "auto-tune" based on source/target skeletons
4. Add preset configurations

**Estimated Effort:** 1 week

---

### UX-006: Model Explorer Filter Documentation

| Field | Value |
|-------|-------|
| **Issue Title** | `feat(ui): Document model explorer filter options` |
| **Severity** | LOW |
| **Labels** | `enhancement`, `ui`, `documentation` |

**Affected Files/Modules:**
- `src/tools/model_explorer/main_window.py`
- `src/tools/model_explorer/component_library.py`

**Description:**

Model explorer filters lack documentation:
- What each filter does
- Filter syntax (wildcards, regex)
- Combining filters
- Saving filter presets

**Recommended Fix:**

1. Add filter help tooltip
2. Show filter syntax examples
3. Implement filter history
4. Add "save filter as preset" option

**Estimated Effort:** 0.5 weeks

---

## Code Quality Issues

### QUALITY-001: Mypy Exclusion Directories

| Field | Value |
|-------|-------|
| **Issue Title** | `refactor(types): Add type annotations to excluded mypy directories` |
| **Severity** | MEDIUM |
| **Labels** | `refactor`, `typing`, `technical-debt` |

**Affected Files/Modules:**

Per `pyproject.toml`, these directories are excluded from mypy checks:
1. `src/engines/Simscape_Multibody_Models/2D_Golf_Model`
2. `src/engines/Simscape_Multibody_Models/3D_Golf_Model`
3. `src/shared/models/opensim/opensim-models`
4. `src/shared/python/signal_toolkit/`
5. `src/shared/python/plotting/`
6. `src/shared/python/tests/`
7. `src/shared/python/ui/qt/widgets/`
8. `src/shared/python/dashboard/tests/`
9. `src/engines/physics_engines/pinocchio/python/motion_training/`
10. `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/`
11. `src/api/`
12. `src/launchers/`
13. `src/engines/common/`
14. `src/shared/python/pose_editor/`
15. `src/shared/python/ai/gui/`
16. `src/shared/python/pose_estimation/`
17. `src/shared/python/ui/adapters/`
18. `src/shared/python/ui/`

**Description:**

12+ directories are excluded from type checking due to incomplete annotations. This hides potential bugs and makes the codebase harder to maintain.

**Recommended Fix:**

1. Prioritize `src/api/` and `src/launchers/` (user-facing)
2. Add `py.typed` markers as directories are cleaned up
3. Use `# type: ignore` sparingly with justification
4. Consider `--strict` mode for new code

**Estimated Effort:** 4-6 weeks (incremental)

---

### QUALITY-002: Pose Estimation Module Docstrings

| Field | Value |
|-------|-------|
| **Issue Title** | `docs(pose): Add comprehensive docstrings to pose_estimation modules` |
| **Severity** | LOW |
| **Labels** | `documentation`, `pose-estimation` |

**Affected Files/Modules:**
- `src/shared/python/pose_estimation/__init__.py`
- `src/shared/python/pose_estimation/openpose_gui.py`
- `src/shared/python/pose_estimation/openpose_estimator.py`
- `src/shared/python/pose_estimation/mediapipe_estimator.py`
- `src/shared/python/pose_estimation/interface.py`
- `src/shared/python/pose_estimation/mediapipe_gui.py`

**Description:**

Pose estimation modules have minimal documentation:
- `interface.py`: Has class docstrings
- `mediapipe_estimator.py`: Module docstring present, method docs sparse
- `openpose_estimator.py`: Basic docstrings
- GUI modules: Minimal documentation

**Recommended Fix:**

1. Add comprehensive module-level docstrings
2. Document all public methods with Args/Returns/Raises
3. Add usage examples in docstrings
4. Create tutorial documentation

**Estimated Effort:** 1 week

---

### QUALITY-003: Control Algorithm Documentation

| Field | Value |
|-------|-------|
| **Issue Title** | `docs(control): Document robotics control algorithms` |
| **Severity** | MEDIUM |
| **Labels** | `documentation`, `robotics`, `control` |

**Affected Files/Modules:**
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/inverse_dynamics.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/drift_control.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/manipulability.py`
- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/power_flow.py`
- `src/engines/physics_engines/pinocchio/python/pinocchio_golf/manipulability.py`
- `src/shared/python/reference_frames.py`
- `src/shared/python/indexed_acceleration.py`

**Description:**

Control algorithms need documentation:
- Mathematical background
- Algorithm assumptions
- Numerical stability considerations
- Usage examples

**Recommended Fix:**

1. Add LaTeX equations in docstrings
2. Reference relevant papers/textbooks
3. Document numerical parameters
4. Add visualization examples

**Estimated Effort:** 1-2 weeks

---

### QUALITY-004: MATLAB Integration User Guides

| Field | Value |
|-------|-------|
| **Issue Title** | `docs(matlab): Create comprehensive MATLAB integration guides` |
| **Severity** | LOW |
| **Labels** | `documentation`, `matlab`, `integration` |

**Affected Files/Modules:**
- `src/engines/physics_engines/pinocchio/tools/matlab_utilities/`
- `src/engines/physics_engines/drake/tools/matlab_utilities/`
- `src/engines/pendulum_models/tools/matlab_utilities/`
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab_utilities/`
- `src/tools/matlab_utilities/`

**Description:**

MATLAB integration tools exist but lack:
- Setup instructions
- Data format documentation
- Example workflows
- Troubleshooting guides

**Recommended Fix:**

1. Create README.md in each matlab_utilities folder
2. Document data exchange formats
3. Add example scripts
4. Create video tutorials

**Estimated Effort:** 1-2 weeks

---

### QUALITY-005: Legacy Launcher Refactoring

| Field | Value |
|-------|-------|
| **Issue Title** | `refactor(launchers): Clean up legacy launcher code` |
| **Severity** | MEDIUM |
| **Labels** | `refactor`, `technical-debt`, `launchers` |

**Affected Files/Modules:**
- `src/launchers/_archive/golf_launcher_pre_refactor_ce85e6ec.py`
- `src/launchers/golf_launcher.py`
- `src/launchers/golf_suite_launcher.py`
- `src/launchers/base.py`
- `src/launchers/ui_components.py`
- `src/launchers/launcher_diagnostics.py`
- `src/launchers/drake_dashboard.py`
- `src/launchers/shot_tracer.py`

**Description:**

Launcher code has accumulated technical debt:
- Archived pre-refactor file still present
- Duplicate code between launchers
- Inconsistent error handling
- Missing type annotations

**Recommended Fix:**

1. Remove or archive legacy code properly
2. Extract common launcher base class
3. Implement consistent error handling
4. Add type annotations
5. Write unit tests

**Estimated Effort:** 1-2 weeks

---

### QUALITY-006: Mesh Generator Pass Statements

| Field | Value |
|-------|-------|
| **Issue Title** | `refactor(mesh): Implement mesh generator stub methods` |
| **Severity** | LOW |
| **Labels** | `refactor`, `mesh`, `stub` |

**Affected Files/Modules:**
- `src/tools/humanoid_character_builder/generators/mesh_generator.py`
- `src/tools/humanoid_character_builder/mesh/mesh_processor.py`

**Description:**

Mesh generation code contains stub implementations with `pass` statements:
- `mesh_generator.py`: 1 pass statement
- `mesh_processor.py`: 2 pass statements

These indicate incomplete error handling or feature implementation.

**Recommended Fix:**

1. Replace `pass` with proper implementation or explicit `NotImplementedError`
2. Document why stubs exist if intentional
3. Add TODO comments with tracking issue numbers
4. Implement missing functionality

**Estimated Effort:** 1-2 days

---

## Summary Matrix

| ID | Category | Severity | Effort | Status |
|----|----------|----------|--------|--------|
| CRITICAL-001 | Legal | CRITICAL | 3-5 days | Open |
| CRITICAL-002 | Security | CRITICAL | 2-3 days | Open |
| CRITICAL-003 | Security | CRITICAL | 2-3 days | Open |
| CRITICAL-004 | Security | CRITICAL | 1 day | Open |
| CRITICAL-005 | Security | HIGH | 1 day | Open |
| CRITICAL-006 | Security | CRITICAL | 1-2 days | Open |
| CRITICAL-007 | Security | HIGH | 1 day | Open |
| CRITICAL-008 | Security | HIGH | 2-3 days | Open |
| CRITICAL-009 | Security | MEDIUM | 0.5 days | Open |
| FEATURE-001 | Engine | HIGH | 2-3 weeks | Open |
| FEATURE-002 | Engine | HIGH | 2-3 weeks | Open |
| FEATURE-003 | Physics | HIGH | 1-2 weeks | Open |
| FEATURE-004 | Physics | MEDIUM | 3-4 weeks | Open |
| FEATURE-005 | Kinematics | MEDIUM | 2-3 weeks | Open |
| PHYSICS-001 | Ball Flight | MEDIUM | 1-2 weeks | Open |
| PHYSICS-002 | Equipment | MEDIUM | 2-3 weeks | Open |
| PHYSICS-003 | Impact | MEDIUM | 1-2 weeks | Open |
| PHYSICS-004 | Impact | MEDIUM | 2 weeks | Open |
| PHYSICS-005 | Biomechanics | LOW | 2-3 weeks | Open |
| PHYSICS-006 | Environment | LOW | 1 week | Open |
| UX-001 | UI | MEDIUM | 1-2 weeks | Open |
| UX-002 | UI | MEDIUM | 1 week | Open |
| UX-003 | UI | LOW | 0.5 weeks | Open |
| UX-004 | UI | LOW | 1-2 weeks | Open |
| UX-005 | UI | LOW | 1 week | Open |
| UX-006 | UI | LOW | 0.5 weeks | Open |
| QUALITY-001 | Types | MEDIUM | 4-6 weeks | Open |
| QUALITY-002 | Docs | LOW | 1 week | Open |
| QUALITY-003 | Docs | MEDIUM | 1-2 weeks | Open |
| QUALITY-004 | Docs | LOW | 1-2 weeks | Open |
| QUALITY-005 | Refactor | MEDIUM | 1-2 weeks | Open |
| QUALITY-006 | Refactor | LOW | 1-2 days | Open |

---

## Quick Reference: GitHub Labels

| Label | Color | Description |
|-------|-------|-------------|
| `security` | `#d73a4a` | Security vulnerability |
| `jules:sentinel` | `#0e8a16` | Auto-tracked by Jules Sentinel |
| `legal` | `#fbca04` | Legal/patent concerns |
| `enhancement` | `#a2eeef` | New feature or improvement |
| `bug` | `#d73a4a` | Something isn't working |
| `refactor` | `#5319e7` | Code refactoring |
| `documentation` | `#0075ca` | Documentation improvements |
| `technical-debt` | `#c5def5` | Code quality issues |
| `breaking-change` | `#b60205` | Breaking API change |

---

*Document maintained by: Development Team*
*Last updated: 2026-02-05*
