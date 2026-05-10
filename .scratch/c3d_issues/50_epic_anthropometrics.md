# [EPIC] anthropometric properties pipeline — segment props (length / mass / inertia / CoM) end-to-end across mocap → URDF → engines

## Vision

Today the pipeline has **algorithms** for anthropometric scaling (`humanoid_character_builder/core/anthropometry.py` with de Leva 1996 ratios; `motion_pipeline/scaling/anthropometric.py` for length estimation; multiple `inertia.py` modules computing tensors per engine), but they are **not unified, not exposed in the GUIs, and not bridged across engines**. A user cannot:

1. **Select a segment in the C3D Viewer** and see its length / mass / inertia tensor / center of mass.
2. **Calibrate** a model's anthropometrics from observed mocap data (subject-specific rather than population-average).
3. **Export** anthropometrics from one engine's model and import them into another's (the URDF format is the natural interchange but neither side bridges to it).
4. **Persist** subject anthropometrics for re-use across analyses.

The user has flagged this as critical because **engine simulations only match reality when the anthropometrics match the subject** — a 6-foot 220-lb golfer simulated with default 5'9" 165-lb anthropometrics produces results that don't transfer.

This epic creates a unified anthropometrics pipeline:

```
   ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
   │ MoCap C3D    ├───►│ SegmentProps├───►│ URDF (canon) │
   │ (markers)    │    │ (canonical) │    │ <inertial>   │
   └──────────────┘    └──────┬──────┘    └──────┬───────┘
                              │                   │
                              ▼                   ▼
                ┌─────────────────────────┐
                │ Per-engine adapters     │
                │ • Drake                 │
                │ • MuJoCo                │
                │ • Pinocchio             │
                │ • OpenSim (.osim)       │
                │ • Simscape (input MAT)  │
                │ • MyoSuite              │
                └──────────┬──────────────┘
                           │
                           ▼
                    Engine simulation
                    that matches the subject
```

## Architecture

A new shared package `src/shared/python/anthropometrics/` that **integrates and extends** existing modules without duplicating them:

```
src/shared/python/anthropometrics/
├── __init__.py
├── contracts.py            # Protocols + dataclasses
├── segment_properties.py   # canonical SegmentProperties dataclass
├── estimators/
│   ├── from_mocap.py       # length from marker pairs; mass via published regression
│   ├── from_de_leva.py     # de Leva 1996 ratios (uses humanoid_character_builder.core.anthropometry)
│   ├── from_dempster.py    # Dempster (1955) classical ratios
│   ├── from_zatsiorsky.py  # Zatsiorsky-Seluyanov ratios
│   └── from_inertia_calc.py # uses model_generation.inertia.calculator (vendor inertia GUI)
├── readers/
│   ├── c3d_subject_info.py # reads any SUBJECT_INFO / PROCESSING params from C3D
│   ├── urdf_inertial.py    # reads <inertial> from URDF
│   ├── osim_body.py        # reads <Body> + <inertia> from OpenSim .osim
│   └── mjcf_body.py        # reads <body> + <inertial> from MJCF
├── writers/
│   ├── urdf_inertial.py    # writes canonical <inertial> URDF block
│   ├── osim_body.py        # writes canonical OpenSim Body
│   └── mjcf_body.py        # writes canonical MJCF body
├── engine_adapters/
│   ├── drake.py
│   ├── mujoco.py
│   ├── pinocchio.py
│   ├── opensim.py
│   ├── simscape.py
│   └── myosuite.py
├── ui/
│   ├── segment_properties_panel.py    # Qt widget (Length / Mass / Inertia tensor / CoM display)
│   └── calibration_dialog.py          # subject-specific calibration UI
├── persistence.py          # SubjectAnthropometrics JSON (per-subject saved record)
└── pipeline.py             # high-level "load mocap → compute → export to engine" orchestrator
```

**Existing modules to integrate (DO NOT duplicate):**

- `src/shared/python/humanoid_character_builder/core/anthropometry.py` — de Leva ratios. Use, don't reimplement.
- `src/shared/python/motion_pipeline/scaling/anthropometric.py` — already does length estimation. Wrap.
- `src/shared/python/model_generation/inertia/{calculator,primitives,spatial}.py` — already computes inertia for primitive shapes. Use.
- `src/shared/python/spatial_algebra/inertia.py` — canonical inertia spatial-algebra utilities.
- `src/engines/physics_engines/{drake,mujoco}/python/.../spatial_algebra/inertia.py` — engine-specific. Adapt, don't duplicate.

## Children (15 issues)

| # | Title | Type | Priority |
|---|---|---|---|
| 1 | feat(anthropometrics): canonical `SegmentProperties` + `SubjectAnthropometrics` dataclasses + Protocols | architecture | high |
| 2 | feat(anthropometrics): `from_de_leva` + `from_dempster` + `from_zatsiorsky` regression estimators | feature | high |
| 3 | feat(anthropometrics): `from_mocap` segment-length estimator (already partially in motion_pipeline; consolidate) | feature | high |
| 4 | feat(anthropometrics): `from_inertia_calc` regression-based inertia tensor + CoM (uses existing inertia calculator) | feature | high |
| 5 | feat(anthropometrics): C3D `SUBJECT_INFO` / `PROCESSING` parameter group reader | feature | medium |
| 6 | feat(anthropometrics): URDF `<inertial>` reader + writer (round-trip canonical SegmentProperties) | feature | high |
| 7 | feat(anthropometrics): OpenSim `.osim` `<Body>` reader + writer | feature | high |
| 8 | feat(anthropometrics): MJCF `<body><inertial>` reader + writer | feature | high |
| 9 | feat(anthropometrics): Drake / Pinocchio / MyoSuite / Simscape engine adapters | feature | high |
| 10 | feat(anthropometrics): high-level `pipeline.py` (load → compute → export to chosen engine) | feature | high |
| 11 | feat(c3d-viewer + matcher): SegmentPropertiesPanel UI (select segment → see length / mass / inertia / CoM) | integration | high |
| 12 | feat(matcher): subject-anthropometrics calibration dialog (slider-driven, mocap-grounded) | integration | medium |
| 13 | feat(anthropometrics): SubjectAnthropometrics JSON persistence (save / load subject records) | feature | high |
| 14 | test(anthropometrics): comprehensive TDD coverage + validation against published anthropometric tables | testing | high |
| 15 | docs(anthropometrics): ADR + user guide + cross-engine pipeline guide | docs | medium |

## Cross-cutting principles (binding for every child PR)

### TDD

- Every child PR: failing tests first.
- ≥85% line / ≥75% branch coverage on new code.
- **Validation tests**: assert canonical published values (e.g. de Leva's 14% upper-arm-mass ratio) are reproduced exactly from our estimators.

### DbC

- All dataclasses validate in `__post_init__`.
- All public callables use `precondition` / `postcondition` from `core.contracts.decorators`.

### LOD, DRY, Orthogonality

- Standard rules.
- **Critical**: do NOT duplicate inertia / anthropometric algorithms. Refer to existing modules and import.

### Generic naming

No vendor / lab / person / study names in code. References to published anthropometric tables (de Leva, Dempster, Zatsiorsky) are fine because they are public scientific citations.

### Performance

- Inertia tensor calc for one segment in ≤ 1 ms.
- Bulk pipeline (load C3D → compute 16 segments → export URDF) in ≤ 100 ms.

## Canonical data model — `SegmentProperties`

```python
@dataclass(frozen=True)
class SegmentProperties:
    """All anthropometric properties of a single body segment.

    Stored in **SI units**: metres, kilograms, kg·m². No imperial.
    """

    # Identity
    name: str                          # e.g. "left_upper_arm"
    body_part_id: str                  # canonical id (head/torso/upper_arm/...)

    # Geometry
    length_m: float                    # > 0
    proximal_marker: str | None        # canonical marker name at proximal end
    distal_marker: str | None          # canonical marker name at distal end

    # Mass
    mass_kg: float                     # > 0

    # Center of mass (in segment-local frame; +x along proximal-to-distal)
    com_xyz_m: np.ndarray              # shape (3,)

    # Inertia tensor (in segment-local frame, expressed at CoM)
    inertia_tensor: np.ndarray         # shape (3, 3); symmetric, positive-definite

    # Provenance
    source_method: str                 # "de_leva" / "dempster" / "from_mocap" / "manual"
    source_subject_height_m: float     # what subject height the regression was anchored at
    source_subject_mass_kg: float      # what subject mass

    # Validation: __post_init__ checks
    # - all positive masses / lengths
    # - inertia tensor is 3x3 symmetric
    # - inertia eigenvalues all > 0 (positive-definite)
    # - eigenvalue triangle inequality (Ix + Iy >= Iz, etc.) — ENFORCED
    # - com_xyz_m within bounding box of segment
```

## Cross-engine pipeline

The pipeline is a one-line user experience:

```python
from src.shared.python.anthropometrics.pipeline import run_pipeline

run_pipeline(
    mocap_file="data/C3D_TA_Driver.c3d",
    subject_height_m=1.83,
    subject_mass_kg=82.0,
    estimator="de_leva",                    # one of: de_leva / dempster / zatsiorsky
    target_engines=["drake", "mujoco", "pinocchio", "opensim"],
    output_dir="output/calibrated_subject/",
)
```

**The pipeline produces:**

```
output/calibrated_subject/
├── subject.json                            # SubjectAnthropometrics record (canonical)
├── humanoid.urdf                           # canonical URDF (for Drake / Pinocchio / MyoSuite / generic)
├── humanoid.osim                           # OpenSim .osim
├── humanoid.xml                            # MJCF for MuJoCo
├── humanoid.slx-input.mat                  # Simscape input MAT
└── report.html                             # validation report (mass closure ±1%,
                                            #   inertia spectral check, etc.)
```

**Round-trip guarantee**: loading any of the engine-specific files back through its reader and the canonical pipeline produces a `SubjectAnthropometrics` record identical to `subject.json` within numerical tolerance (`rtol=1e-9`, `atol=1e-6`).

## C3D file metadata sources

The pipeline reads anthropometrics from C3D when present:

- `PROCESSING:Bodymass` (kg)
- `PROCESSING:Height` (mm — converted to m)
- `PROCESSING:LeftLegLength`, `RightLegLength`, etc.
- `SUBJECTS:NAMES` (excluded from logs per generic-naming policy; used only as session identifier)

When C3D doesn't provide them, fall back to user-provided `subject_height_m` / `subject_mass_kg` arguments. Default to "default subject" (1.75 m, 75 kg) only when neither is available, with a clear log warning.

## Inertia matrix surfacing in the UI

A new "Segment Properties" panel in both the C3D Viewer and the Motion-Match Preview matcher shows for the currently-selected segment:

```
Segment: left_upper_arm
─────────────────────────────────
Length      : 0.2891 m
Mass        : 1.967 kg                    (de Leva 14.0% × 80 kg × 0.176 ratio)
CoM offset  : (+0.0987, +0.0035, -0.0041) m  (from proximal end)
Inertia (in CoM frame, kg·m²):
   ┌                                  ┐
   │  0.01024  -0.00006  +0.00012  │
   │ -0.00006   0.00198  +0.00003  │
   │ +0.00012  +0.00003   0.01001  │
   └                                  ┘
   Principal moments (eigenvalues): [0.00198, 0.01001, 0.01024] kg·m²

Source     : de_leva
Anchored at: subject 1.83 m, 82 kg
```

## Out of scope for v1

- Inertia from voxel scans / DXA — separate medical-imaging effort.
- Optimization-based subject-mass refinement from torque data — v2.
- Per-trial (rather than per-subject) anthropometrics — same person across multiple trials uses one subject record.
- Children's anthropometrics tables (de Leva is adult-only) — v2.
- Non-rigid (soft tissue) wobble masses — v2.

## Definition of done (epic)

- All 15 children closed.
- A user can:
  - Select a segment in the C3D Viewer → see length / mass / inertia tensor / CoM.
  - Run the pipeline on a C3D file with subject mass + height → get URDF / OSIM / MJCF / Simscape outputs.
  - Load any engine-specific file back and recover the same canonical record.
- ≥ 85% line coverage on `anthropometrics`; ≥ 75% branch.
- Validation tests reproduce de Leva, Dempster, Zatsiorsky published numbers.
- Cross-engine round-trip test passes for all 5 engine formats.
- ADR + user guide + cross-engine pipeline guide on `main`.
