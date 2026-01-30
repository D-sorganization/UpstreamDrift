# Humanoid Character Builder for URDF Generation

## Implementation Plan & Technical Specification

**Author**: Claude Code
**Date**: 2026-01-30
**Status**: Planning Phase

---

## 1. Executive Summary

This document outlines a comprehensive plan to implement a video game-style character builder for the URDF generator. The feature will enable users to create customizable humanoid models with:

- Parametric body shape control (height, weight, build, proportions)
- Facial customization and appearance options
- Individual body part mesh export
- Automatic inertia calculation from mesh geometry
- Manual inertia override mode
- Integration with open source character generation resources

---

## 2. Current Architecture Analysis

### 2.1 Existing URDF Generation Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `URDFBuilder` | `src/tools/model_explorer/urdf_builder.py` | Core URDF segment management with physics validation |
| `GolfURDFGenerator` | `src/engines/physics_engines/drake/python/src/drake_golf_model.py` | Drake-based specialized golf model generation |
| `URDFExporter` (MuJoCo) | `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/urdf_io.py` | Bidirectional URDF/MJCF conversion |
| `URDFExporter` (Pinocchio) | `src/engines/physics_engines/pinocchio/python/dtack/utils/urdf_exporter.py` | YAML-based canonical URDF export |
| `MeshBrowser` | `src/tools/model_explorer/mesh_browser.py` | Mesh file browsing and copying |
| `FrankensteinEditor` | `src/tools/model_explorer/frankenstein_editor.py` | Multi-URDF component composition |

### 2.2 Existing Inertia Calculation

**Current Implementation** (`src/shared/python/spatial_algebra/inertia.py`):
- `mcI()`: Constructs 6x6 spatial inertia matrix from mass, COM, and rotational inertia
- `transform_spatial_inertia()`: Transforms inertia between reference frames
- Default inertia: 0.1 kg*m^2 for unspecified segments

**Drake Integration**:
- `make_cylinder_inertia()`: Uses `pydrake.UnitInertia.SolidCylinder`
- Analytical formulas for primitive shapes

**Limitations**:
- No mesh-based inertia calculation
- Only primitive shapes (box, cylinder, sphere, capsule)
- No convex hull decomposition

### 2.3 Model Explorer GUI Architecture

```
MainWindow
  |-- SegmentPanel (left dock)
  |-- VisualizationWidget (center)
  |-- PropertiesPanel (bottom)
  |-- URDFCodeEditor (tab)
  |-- FrankensteinEditor (tab)
  |-- MeshBrowser (tab)
```

The GUI uses PyQt6 with dockable widgets, making it extensible for new character builder panels.

---

## 3. Open Source Resource Evaluation

### 3.1 Character Generation Libraries

| Resource | License | Python Support | Mesh Export | Body Parameters | Faces | Recommendation |
|----------|---------|----------------|-------------|-----------------|-------|----------------|
| **MakeHuman** | AGPL-3.0 (code), CC0 (exports) | Yes (PyQt) | OBJ, FBX, DAE | Excellent | Yes | **PRIMARY** |
| **MPFB2** | GPL-3.0 (code), CC0 (assets) | Blender Python | All Blender formats | Excellent | Yes | **PRIMARY** |
| **SMPL/SMPL-X** | Custom (research) | PyTorch | OBJ, PLY | Good | Limited | **SECONDARY** |
| **Anny** | Apache-2.0 | PyTorch/Warp | Multiple | Excellent | No | Consider |
| **CharMorph** | GPL-3.0 | Blender Python | All Blender formats | Good | Yes | Alternative |
| **MB-Lab** | AGPL-3.0 | Blender Python | All Blender formats | Good | Yes | Legacy |

### 3.2 Recommended Primary Integration: MakeHuman + MPFB2

**MakeHuman** advantages:
- Mature project (15+ years development)
- Extensive body parameter system (height, weight, age, gender, ethnicity, muscularity)
- Rich face customization (eyes, nose, mouth, chin, etc.)
- Clothing and hair assets available
- CC0 license on exported models - no restrictions
- Python-based with Qt GUI
- Standalone application + Blender plugin

**MPFB2** advantages:
- Modern Blender 4.x integration
- Same asset compatibility as MakeHuman
- Direct mesh manipulation in Blender
- Better export pipeline options

### 3.3 Secondary Integration: SMPL-X

**SMPL-X** provides:
- Differentiable body model (good for optimization)
- Learned shape space from 10,000+ body scans
- Hand and face articulation
- PyTorch integration
- Used in research for motion capture fitting

**Limitation**: Research license requires registration; not fully open for commercial use.

### 3.4 Mesh Processing: Trimesh

**Trimesh** (`pip install trimesh`) is the recommended library for:
- Loading STL, OBJ, PLY meshes
- Computing mass properties (volume, COM, inertia tensor)
- Convex hull generation
- Mesh repair and watertight checks
- Boolean operations

Key functions:
```python
import trimesh

mesh = trimesh.load('body_part.stl')
if mesh.is_watertight:
    inertia_tensor = mesh.moment_inertia  # 3x3 matrix at COM
    volume = mesh.volume
    center_mass = mesh.center_mass
```

---

## 4. Proposed Architecture

### 4.1 New Module Structure

```
src/tools/model_explorer/
  |-- character_builder/
  |     |-- __init__.py
  |     |-- character_builder_widget.py    # Main GUI widget
  |     |-- body_parameters.py             # Parametric body control
  |     |-- appearance_panel.py            # Face/skin/clothing customization
  |     |-- segment_preview.py             # 3D preview of body parts
  |     |-- mesh_exporter.py               # Export individual segments
  |     |-- inertia_calculator.py          # Mesh-based inertia computation
  |     |-- makehuman_bridge.py            # MakeHuman integration
  |     |-- templates/
  |     |     |-- base_humanoid.yaml       # Default humanoid template
  |     |     |-- athletic.yaml            # Athletic body type preset
  |     |     |-- average.yaml             # Average body type preset
  |     |     |-- heavy.yaml               # Heavy build preset
  |
src/shared/python/
  |-- mesh_inertia.py                      # Mesh-to-inertia calculation utilities
  |-- body_segment_library.py              # Reusable body segment definitions
```

### 4.2 Core Classes

```python
# character_builder/body_parameters.py
@dataclass
class BodyParameters:
    """Parametric body shape control."""
    # Demographics
    height_m: float = 1.75           # Total height in meters
    mass_kg: float = 75.0            # Total body mass
    gender_factor: float = 0.5       # 0=feminine, 1=masculine
    age_years: float = 30.0          # Affects proportions

    # Build
    muscularity: float = 0.5         # 0=lean, 1=muscular
    body_fat: float = 0.2            # Body fat percentage
    shoulder_width_factor: float = 1.0
    hip_width_factor: float = 1.0
    limb_length_factor: float = 1.0
    torso_length_factor: float = 1.0

    # Individual segment scaling
    head_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    neck_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # ... per-segment scaling

    def to_makehuman_targets(self) -> dict[str, float]:
        """Convert to MakeHuman morph target values."""
        ...

    def compute_segment_dimensions(self) -> dict[str, SegmentDimensions]:
        """Compute dimensions for each body segment."""
        ...

    def estimate_segment_masses(self) -> dict[str, float]:
        """Estimate mass distribution using anthropometric data."""
        # Based on: de Leva (1996) anthropometric tables
        ...
```

```python
# character_builder/inertia_calculator.py
class MeshInertiaCalculator:
    """Calculate inertia tensors from mesh geometry."""

    def __init__(self, density: float = 1000.0):  # kg/m^3, default ~tissue density
        self.density = density

    def compute_from_mesh(
        self,
        mesh_path: Path,
        mass: float | None = None,  # Override mass (uniform density if None)
    ) -> InertiaResult:
        """
        Compute inertia tensor from mesh file.

        Returns:
            InertiaResult with ixx, iyy, izz, ixy, ixz, iyz,
            center_of_mass, volume, computed_mass
        """
        import trimesh
        mesh = trimesh.load(mesh_path)

        if not mesh.is_watertight:
            mesh = self._repair_mesh(mesh)

        # Get raw inertia assuming unit density
        raw_inertia = mesh.moment_inertia
        volume = mesh.volume
        center_mass = mesh.center_mass

        if mass is not None:
            # Scale inertia for specified mass
            computed_density = mass / volume
            inertia = raw_inertia * computed_density
        else:
            # Use uniform density
            inertia = raw_inertia * self.density
            mass = volume * self.density

        return InertiaResult(
            ixx=inertia[0, 0], iyy=inertia[1, 1], izz=inertia[2, 2],
            ixy=inertia[0, 1], ixz=inertia[0, 2], iyz=inertia[1, 2],
            center_of_mass=center_mass,
            volume=volume,
            mass=mass
        )

    def compute_from_primitives(
        self,
        shape: str,
        dimensions: dict,
        mass: float
    ) -> InertiaResult:
        """Compute inertia for primitive shapes (fallback)."""
        ...
```

```python
# character_builder/mesh_exporter.py
class SegmentMeshExporter:
    """Export individual body segment meshes."""

    def export_segment(
        self,
        segment_name: str,
        body_params: BodyParameters,
        output_path: Path,
        format: str = 'stl',  # stl, obj, dae
        include_collision_mesh: bool = True,
        collision_simplification: float = 0.5,  # Reduce mesh complexity
    ) -> ExportResult:
        """Export a single body segment as mesh file."""
        ...

    def export_all_segments(
        self,
        body_params: BodyParameters,
        output_dir: Path,
        format: str = 'stl',
    ) -> list[ExportResult]:
        """Export all body segments to separate files."""
        ...

    def create_convex_hull(
        self,
        mesh_path: Path
    ) -> Path:
        """Generate convex hull for collision geometry."""
        import trimesh
        mesh = trimesh.load(mesh_path)
        hull = mesh.convex_hull
        hull_path = mesh_path.with_suffix('.collision.stl')
        hull.export(hull_path)
        return hull_path
```

### 4.3 GUI Widget Design

```python
# character_builder/character_builder_widget.py
class CharacterBuilderWidget(QWidget):
    """
    Video game-style character builder interface.

    Layout:
    +--------------------------------------------------+
    |  [Presets v]  [Load]  [Save]  [Export URDF]      |
    +--------------------------------------------------+
    |                    |                             |
    |   BODY PARAMETERS  |      3D PREVIEW             |
    |   ---------------  |      ----------             |
    |   Height: [====]   |      [3D Viewport]          |
    |   Weight: [====]   |                             |
    |   Build:  [====]   |      [Rotate] [Zoom]        |
    |   Gender: [====]   |                             |
    |   Age:    [====]   |                             |
    |                    |                             |
    |   PROPORTIONS      |                             |
    |   -----------      |                             |
    |   Shoulders: [==]  |                             |
    |   Hips:      [==]  |                             |
    |   Legs:      [==]  |                             |
    |   Arms:      [==]  |                             |
    |                    |                             |
    +--------------------------------------------------+
    |                  SEGMENTS                        |
    +--------------------------------------------------+
    |  [x] Head    [x] Torso   [x] L.Arm   [x] R.Arm   |
    |  [Mesh: auto v]  [Inertia: computed v]           |
    |  [Export Selected]  [Export All]                 |
    +--------------------------------------------------+
    |                                                  |
    |   APPEARANCE          |    INERTIA MODE          |
    |   ----------          |    -----------           |
    |   Skin Tone: [===]    |    (o) Compute from mesh |
    |   Face Type: [===]    |    ( ) Manual override   |
    |   Hair:      [===]    |    Density: [1000] kg/m3 |
    |                       |                          |
    +--------------------------------------------------+
    ```
"""
```

### 4.4 Inertia Modes

**Mode 1: Automatic Mesh-Based Computation**
```python
class InertiaMode(Enum):
    MESH_UNIFORM_DENSITY = "mesh_uniform"      # Compute from mesh with uniform density
    MESH_SPECIFIED_MASS = "mesh_with_mass"     # Scale mesh inertia to match specified mass
    PRIMITIVE_APPROXIMATION = "primitive"       # Use primitive shape approximation
    MANUAL = "manual"                           # User specifies all inertia values

@dataclass
class SegmentInertiaConfig:
    mode: InertiaMode = InertiaMode.MESH_UNIFORM_DENSITY
    density_kg_m3: float = 1000.0              # Tissue density (~1000 kg/m^3)
    mass_override: float | None = None         # Override mass (kg)
    manual_inertia: dict | None = None         # Manual ixx, iyy, izz, etc.
```

**Mode 2: Manual Override**
```python
@dataclass
class ManualInertia:
    """User-specified inertia values."""
    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0
    com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
```

---

## 5. MakeHuman Integration Strategy

### 5.1 Option A: Subprocess Integration (Recommended for Phase 1)

Use MakeHuman as an external process:

```python
class MakeHumanBridge:
    """Bridge to MakeHuman for mesh generation."""

    def __init__(self, makehuman_path: Path | None = None):
        self.mh_path = makehuman_path or self._find_makehuman()

    def generate_mesh(
        self,
        body_params: BodyParameters,
        output_path: Path,
        subdivisions: int = 0,
    ) -> Path:
        """
        Generate mesh using MakeHuman CLI/scripting.

        1. Convert body_params to MakeHuman target values
        2. Write .mhx2 or Python script
        3. Run MakeHuman in batch mode
        4. Export mesh (OBJ/FBX)
        """
        ...

    def segment_mesh(
        self,
        full_mesh_path: Path,
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Segment full body mesh into individual parts.
        Uses vertex groups or predefined cutting planes.
        """
        ...
```

### 5.2 Option B: Direct Python Integration (Phase 2)

Extract MakeHuman's mesh generation logic:

```python
# Future: Direct integration with MakeHuman Python modules
from makehuman.core import HumanMesh
from makehuman.modifiers import MacroModifier

class DirectMakeHumanIntegration:
    def __init__(self):
        self.mesh = HumanMesh()
        self.modifiers = self._load_modifiers()

    def set_parameters(self, params: BodyParameters):
        # Apply morph targets directly
        self.mesh.setHeight(params.height_m)
        self.mesh.setWeight(params.mass_kg)
        # etc.
```

### 5.3 Body Segmentation

For URDF generation, we need to split the full body mesh into individual link meshes:

```python
HUMANOID_SEGMENTS = {
    # Segment name -> vertex group or cutting definition
    "pelvis": {"vertex_group": "pelvis", "parent": None},
    "lumbar": {"vertex_group": "spine-lower", "parent": "pelvis"},
    "thorax": {"vertex_group": "spine-upper", "parent": "lumbar"},
    "head": {"vertex_group": "head", "parent": "neck"},
    "neck": {"vertex_group": "neck", "parent": "thorax"},
    "left_shoulder": {"vertex_group": "shoulder.L", "parent": "thorax"},
    "right_shoulder": {"vertex_group": "shoulder.R", "parent": "thorax"},
    "left_upper_arm": {"vertex_group": "upperarm.L", "parent": "left_shoulder"},
    "right_upper_arm": {"vertex_group": "upperarm.R", "parent": "right_shoulder"},
    "left_forearm": {"vertex_group": "forearm.L", "parent": "left_upper_arm"},
    "right_forearm": {"vertex_group": "forearm.R", "parent": "right_upper_arm"},
    "left_hand": {"vertex_group": "hand.L", "parent": "left_forearm"},
    "right_hand": {"vertex_group": "hand.R", "parent": "right_forearm"},
    "left_thigh": {"vertex_group": "thigh.L", "parent": "pelvis"},
    "right_thigh": {"vertex_group": "thigh.R", "parent": "pelvis"},
    "left_shin": {"vertex_group": "shin.L", "parent": "left_thigh"},
    "right_shin": {"vertex_group": "shin.R", "parent": "right_thigh"},
    "left_foot": {"vertex_group": "foot.L", "parent": "left_shin"},
    "right_foot": {"vertex_group": "foot.R", "parent": "right_shin"},
}
```

---

## 6. Anthropometric Data Integration

### 6.1 Mass Distribution (de Leva 1996)

Reference: de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.

```python
# Segment mass as percentage of total body mass
SEGMENT_MASS_RATIOS = {
    # Males (average)
    "male": {
        "head": 0.0694,
        "trunk": 0.4346,  # Will subdivide into pelvis, lumbar, thorax
        "upper_arm": 0.0271,
        "forearm": 0.0162,
        "hand": 0.0061,
        "thigh": 0.1416,
        "shank": 0.0433,
        "foot": 0.0137,
    },
    # Females (average)
    "female": {
        "head": 0.0668,
        "trunk": 0.4257,
        "upper_arm": 0.0255,
        "forearm": 0.0138,
        "hand": 0.0056,
        "thigh": 0.1478,
        "shank": 0.0481,
        "foot": 0.0129,
    }
}

# Segment length as percentage of stature
SEGMENT_LENGTH_RATIOS = {
    "male": {
        "head_height": 0.1395,
        "neck": 0.052,
        "trunk": 0.288,
        "upper_arm": 0.186,
        "forearm": 0.146,
        "hand": 0.108,
        "thigh": 0.245,
        "shank": 0.246,
        "foot_length": 0.152,
    }
}
```

### 6.2 Inertia Approximation Formulas

For primitive-based fallback:

```python
def approximate_segment_inertia(
    segment_type: str,
    mass: float,
    length: float,
    width: float | None = None,
) -> dict:
    """
    Approximate inertia using standard formulas.
    Most limb segments can be approximated as cylinders or frustums.
    """
    if segment_type in ["upper_arm", "forearm", "thigh", "shank"]:
        # Cylinder approximation
        # I_xx = I_yy = (1/12)*m*(3*r^2 + L^2)
        # I_zz = (1/2)*m*r^2
        radius = width / 2 if width else length * 0.08  # Approximate
        ixx = iyy = (1/12) * mass * (3 * radius**2 + length**2)
        izz = 0.5 * mass * radius**2
        return {"ixx": ixx, "iyy": iyy, "izz": izz}

    elif segment_type == "head":
        # Sphere approximation
        # I = (2/5)*m*r^2
        radius = length / 2
        i = 0.4 * mass * radius**2
        return {"ixx": i, "iyy": i, "izz": i}

    # ... other segments
```

---

## 7. URDF Generation Pipeline

### 7.1 Complete Pipeline

```
User Input (GUI sliders)
         |
         v
+-------------------+
| BodyParameters    |
| - height, weight  |
| - build factors   |
| - appearance      |
+-------------------+
         |
         v
+-------------------+
| MakeHuman Bridge  |
| - Generate mesh   |
| - Apply morphs    |
+-------------------+
         |
         v
+-------------------+
| Mesh Segmentation |
| - Split by vertex |
|   groups          |
| - Export per-link |
+-------------------+
         |
         v
+-------------------+
| Inertia Calc      |
| - trimesh load    |
| - Mass properties |
| - Scale to mass   |
+-------------------+
         |
         v
+-------------------+
| URDFBuilder       |
| - Add segments    |
| - Add joints      |
| - Validate        |
+-------------------+
         |
         v
+-------------------+
| URDF Output       |
| + Mesh files      |
+-------------------+
```

### 7.2 Output Structure

```
output/
  |-- humanoid_model/
  |     |-- humanoid.urdf
  |     |-- meshes/
  |     |     |-- visual/
  |     |     |     |-- pelvis.dae
  |     |     |     |-- thorax.dae
  |     |     |     |-- head.dae
  |     |     |     |-- ... (textured meshes)
  |     |     |-- collision/
  |     |     |     |-- pelvis.stl
  |     |     |     |-- thorax.stl
  |     |     |     |-- head.stl
  |     |     |     |-- ... (simplified collision)
  |     |-- textures/
  |     |     |-- skin_diffuse.png
  |     |     |-- skin_normal.png
  |     |-- config/
  |     |     |-- body_params.yaml
  |     |     |-- inertia_params.yaml
```

---

## 8. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

**Deliverables:**
1. `mesh_inertia.py` - Trimesh-based inertia calculation
2. `body_parameters.py` - Parameter dataclasses and anthropometric data
3. `inertia_calculator.py` - Both mesh-based and manual modes
4. Unit tests for inertia calculations

**Key Milestones:**
- [ ] Trimesh integration working
- [ ] Inertia validation against known shapes
- [ ] Anthropometric mass distribution tables

### Phase 2: MakeHuman Integration (Weeks 3-4)

**Deliverables:**
1. `makehuman_bridge.py` - Subprocess/script integration
2. Mesh segmentation pipeline
3. Parameter mapping (BodyParameters -> MakeHuman targets)

**Key Milestones:**
- [ ] Generate full body mesh from parameters
- [ ] Segment mesh into individual parts
- [ ] Export collision-simplified meshes

### Phase 3: GUI Implementation (Weeks 5-6)

**Deliverables:**
1. `character_builder_widget.py` - Main GUI
2. `body_sliders_panel.py` - Parameter sliders
3. `appearance_panel.py` - Face/skin customization
4. `segment_list_widget.py` - Segment selection and export

**Key Milestones:**
- [ ] Working slider interface
- [ ] Real-time preview updates
- [ ] Segment selection UI

### Phase 4: URDF Integration (Weeks 7-8)

**Deliverables:**
1. Integration with existing `URDFBuilder`
2. Mesh path resolution in URDF
3. Export wizard for complete model packages
4. Template presets (athletic, average, etc.)

**Key Milestones:**
- [ ] Complete URDF generation from character builder
- [ ] Mesh files properly referenced
- [ ] Works with existing physics engines

### Phase 5: Polish & Advanced Features (Weeks 9-10)

**Deliverables:**
1. Clothing/accessory support
2. Save/load character configurations
3. Batch export for parameter studies
4. Documentation and examples

---

## 9. Dependencies

### 9.1 Required Python Packages

```toml
# pyproject.toml additions
[project.dependencies]
trimesh = ">=4.0.0"      # Mesh processing and inertia
numpy = ">=1.24.0"       # Already present
scipy = ">=1.10.0"       # Convex hull, optimization

[project.optional-dependencies]
character_builder = [
    "pyglet",            # For trimesh visualization
    "networkx",          # Mesh graph operations
    "Pillow",            # Texture handling
]
```

### 9.2 External Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| MakeHuman | Character generation | Download from makehuman.org |
| MPFB2 | Blender integration | Blender extension |
| Blender (optional) | Mesh processing | blender.org |

### 9.3 Asset Requirements

- MakeHuman base mesh and targets
- Default texture set (skin, eyes)
- Preset configurations (athletic, average, etc.)

---

## 10. API Design

### 10.1 Python API

```python
from golf_modeling_suite.character_builder import CharacterBuilder, BodyParameters

# Create character with parameters
params = BodyParameters(
    height_m=1.80,
    mass_kg=80.0,
    muscularity=0.7,
    gender_factor=0.8,
)

# Build character
builder = CharacterBuilder()
character = builder.create(params)

# Export as URDF with meshes
character.export_urdf(
    output_dir="./my_humanoid",
    inertia_mode=InertiaMode.MESH_UNIFORM_DENSITY,
    density=1050.0,  # kg/m^3
    include_textures=True,
)

# Or export individual segment mesh
pelvis_mesh = character.get_segment_mesh("pelvis")
pelvis_inertia = character.compute_segment_inertia("pelvis", mass=12.0)
```

### 10.2 REST API Extensions

```yaml
# New API endpoints
POST /api/character/create:
  description: Create character from parameters
  body:
    height_m: float
    mass_kg: float
    build: string (athletic|average|heavy)
    # ... other params
  returns:
    character_id: string
    preview_url: string

GET /api/character/{id}/segments:
  description: List all segments
  returns:
    segments: list[SegmentInfo]

POST /api/character/{id}/export:
  description: Export as URDF
  body:
    format: urdf|mjcf|sdf
    inertia_mode: mesh|manual
  returns:
    download_url: string
```

---

## 11. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MakeHuman compatibility issues | Medium | High | Maintain fallback primitive-based generation |
| Mesh segmentation quality | Medium | Medium | Manual vertex group definitions as fallback |
| Inertia calculation accuracy | Low | Medium | Validate against analytical solutions |
| Performance with large meshes | Medium | Low | Level-of-detail options, caching |
| License compliance | Low | High | Document all licenses, use CC0 exports |

---

## 12. Success Criteria

### 12.1 Functional Requirements

- [ ] User can adjust body parameters via sliders
- [ ] Preview updates in real-time (or near real-time)
- [ ] Individual body part meshes can be exported
- [ ] Inertia is correctly computed from meshes
- [ ] Manual inertia override works
- [ ] Generated URDF loads in MuJoCo, Drake, and Pinocchio
- [ ] Mesh files are correctly referenced in URDF

### 12.2 Performance Requirements

- Character generation: < 30 seconds
- Inertia calculation per segment: < 1 second
- Preview update: < 500ms
- Total export time: < 60 seconds

### 12.3 Quality Requirements

- Inertia within 5% of analytical solution for primitives
- Mesh is watertight (validated by trimesh)
- URDF passes validation
- No physics simulation instabilities

---

## 13. Open Questions

1. **Finger articulation**: Should we include individual finger joints, or treat hands as single rigid bodies?
   - Recommendation: Start with rigid hands, add finger option later

2. **Facial expressions**: Do we need blend shapes for faces?
   - Recommendation: Static faces for Phase 1, blend shapes for future

3. **Clothing collision**: Should clothing affect collision geometry?
   - Recommendation: Clothing visual only, use body for collision

4. **Real-time preview**: OpenGL preview in Qt, or separate visualizer?
   - Recommendation: Use existing MuJoCo viewer integration

---

## 14. References

1. de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters. Journal of Biomechanics, 29(9), 1223-1230.

2. MakeHuman Documentation: http://www.makehumancommunity.org/wiki/

3. MPFB2 Documentation: https://static.makehumancommunity.org/mpfb.html

4. Trimesh Documentation: https://trimesh.org/

5. SMPL: A Skinned Multi-Person Linear Model, Loper et al., SIGGRAPH Asia 2015

6. URDF Specification: http://wiki.ros.org/urdf/XML

---

## 15. Appendix: Example URDF Output

```xml
<?xml version="1.0" ?>
<robot name="humanoid_character">

  <!-- Materials -->
  <material name="skin">
    <color rgba="0.87 0.72 0.53 1.0"/>
  </material>

  <!-- Pelvis (root link) -->
  <link name="pelvis">
    <visual>
      <geometry>
        <mesh filename="package://humanoid_model/meshes/visual/pelvis.dae"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://humanoid_model/meshes/collision/pelvis.stl"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.0 0.0 0.05"/>
      <mass value="11.7"/>
      <inertia ixx="0.0892" ixy="0.0" ixz="0.0"
               iyy="0.0654" iyz="0.0" izz="0.0743"/>
    </inertial>
  </link>

  <!-- Lumbar spine -->
  <link name="lumbar">
    <visual>
      <geometry>
        <mesh filename="package://humanoid_model/meshes/visual/lumbar.dae"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://humanoid_model/meshes/collision/lumbar.stl"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.0 0.0 0.08"/>
      <mass value="8.5"/>
      <inertia ixx="0.0456" ixy="0.0" ixz="0.0"
               iyy="0.0312" iyz="0.0" izz="0.0521"/>
    </inertial>
  </link>

  <joint name="pelvis_to_lumbar" type="universal">
    <parent link="pelvis"/>
    <child link="lumbar"/>
    <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="2"/>
  </joint>

  <!-- ... additional links and joints ... -->

</robot>
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-30*
