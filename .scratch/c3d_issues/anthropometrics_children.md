# anthropometrics child issues — full bodies for all 15 children

## Child 01 — Contracts

### feat(anthropometrics): canonical SegmentProperties + SubjectAnthropometrics + Protocols

First issue. Pure data model + Protocols.

Files in `src/shared/python/anthropometrics/`:
- `contracts.py` — `Estimator`, `Reader`, `Writer`, `EngineAdapter` Protocols.
- `segment_properties.py` — `SegmentProperties` frozen dataclass (full spec in epic body).
- `_subject_anthropometrics.py` — `SubjectAnthropometrics(subject_id, height_m, mass_kg, segments: dict[str, SegmentProperties], source_method, ...)`.
- `__init__.py` re-exports.

Validation:
- All masses, lengths > 0.
- inertia_tensor 3×3 symmetric, positive-definite.
- Eigenvalue triangle inequality (Ix + Iy ≥ Iz, Iy + Iz ≥ Ix, Ix + Iz ≥ Iy) — strict invariant of physical realisability.
- `com_xyz_m` within bounding box of segment (length × radius bounds).

Tests: ≥95% coverage. Each validation rule has a happy + fail test.

---

## Child 02 — de Leva / Dempster / Zatsiorsky estimators

### feat(anthropometrics): from_de_leva + from_dempster + from_zatsiorsky regression estimators

Three estimator classes in `src/shared/python/anthropometrics/estimators/`. Each implements `Estimator` Protocol:

```python
class Estimator(Protocol):
    method_name: str

    def estimate(
        self,
        subject_height_m: float,
        subject_mass_kg: float,
        segment_lengths_m: dict[str, float] | None = None,
    ) -> dict[str, SegmentProperties]:
        ...
```

`from_de_leva.py` reuses `humanoid_character_builder.core.anthropometry` ratios — DO NOT DUPLICATE.

`from_dempster.py` and `from_zatsiorsky.py` ship their own ratio tables (committed JSON) since `humanoid_character_builder` only ships de Leva.

**Validation tests**: assert published numbers reproduce. Examples:
- de Leva upper-arm mass ratio = 0.0271 of total body mass for males.
- de Leva forearm CoM = 45.7% of segment length from proximal end.
- These published values must reproduce exactly to ≤1e-3.

≥90% coverage.

---

## Child 03 — From-mocap segment-length estimator

### feat(anthropometrics): from_mocap segment-length estimator

`src/shared/python/anthropometrics/estimators/from_mocap.py`:

```python
def estimate_segment_lengths_from_markers(
    markers: dict[str, np.ndarray],   # marker_name -> (T, 3)
    segment_definitions: list[SegmentDef],
    *,
    method: Literal["mean_distance", "median_distance", "min_distance"] = "median_distance",
) -> dict[str, float]:
    """Return per-segment length in metres."""
```

Where `SegmentDef = (segment_name, proximal_marker, distal_marker)`.

Reuses logic from `motion_pipeline/scaling/anthropometric.py` — wrap, don't duplicate.

Tests:
- Synthetic 100-frame trajectory with known segment length recovers within 1e-9.
- NaN-tolerant: half-NaN markers still produce a length via remaining frames.
- Method comparison: median is robust to outliers; min is conservative.

≥90% coverage.

---

## Child 04 — Inertia tensor estimator

### feat(anthropometrics): from_inertia_calc regression-based inertia tensor + CoM

`src/shared/python/anthropometrics/estimators/from_inertia_calc.py`:

Wraps `model_generation.inertia.calculator` (already in repo) and the de Leva radius-of-gyration ratios to produce `SegmentProperties.inertia_tensor` for each segment.

For a cylinder approximation:
- `Ix` (long axis) = mass × r² / 2
- `Iy = Iz` (transverse) = mass × (3r² + L²) / 12

For the de Leva approach: `I = mass × (gyration_radius × length)²` per axis.

Mix-and-match: cylinder for limbs, ellipsoid for head/torso.

Tests: each inertia computation reproduces the published gyration-radius-derived value to 1e-9.

≥90% coverage.

---

## Child 05 — C3D SUBJECT_INFO reader

### feat(anthropometrics): C3D SUBJECT_INFO / PROCESSING parameter reader

`src/shared/python/anthropometrics/readers/c3d_subject_info.py`:

```python
@dataclass(frozen=True)
class C3DSubjectMetadata:
    subject_id: str | None
    height_m: float | None       # PROCESSING:Height (mm) -> m
    mass_kg: float | None        # PROCESSING:Bodymass (kg)
    age_years: float | None      # SUBJECTS:AGE
    sex: Literal["M", "F", "unspecified"]
    leg_length_m: float | None
    arm_length_m: float | None

def read_c3d_subject_metadata(path: Path) -> C3DSubjectMetadata:
    """Returns C3DSubjectMetadata; missing keys → None fields."""
```

Real-data note: our existing `data/C3D_TA_*.c3d` files do NOT carry `SUBJECT_INFO` — fields all return None. Test against synthesised C3D fixtures with the params populated.

≥95% coverage.

---

## Child 06 — URDF inertial reader/writer

### feat(anthropometrics): URDF <inertial> reader + writer (round-trip canonical SegmentProperties)

`src/shared/python/anthropometrics/{readers,writers}/urdf_inertial.py`.

Reader:
```python
def read_urdf_inertial(visual_element: ET.Element) -> SegmentProperties:
    """Parse a <link><inertial> block into SegmentProperties.

    URDF inertia is at CoM; tensor is symmetric (xx, xy, xz, yy, yz, zz).
    """
```

Writer:
```python
def write_urdf_inertial(props: SegmentProperties) -> ET.Element:
    """Produce <inertial><origin xyz="..."/><mass value="..."/><inertia ...></inertia></inertial>"""
```

Round-trip test: every `SegmentProperties` instance written then re-read recovers identically (rtol=1e-9, atol=1e-12).

≥95% coverage.

---

## Child 07 — OpenSim .osim reader/writer

### feat(anthropometrics): OpenSim .osim Body reader + writer

`src/shared/python/anthropometrics/{readers,writers}/osim_body.py`.

OpenSim `<Body>` schema:
- `<mass>kg</mass>`
- `<mass_center>x y z</mass_center>`
- `<inertia>Ixx Iyy Izz Ixy Ixz Iyz</inertia>`

Round-trip test required.

≥95% coverage.

---

## Child 08 — MJCF body reader/writer

### feat(anthropometrics): MJCF <body><inertial> reader + writer

`src/shared/python/anthropometrics/{readers,writers}/mjcf_body.py`.

MJCF `<body><inertial>` schema:
- `pos="x y z"`
- `mass="kg"`
- `diaginertia="Ix Iy Iz"` OR `fullinertia="Ixx Iyy Izz Ixy Ixz Iyz"`

Round-trip test required.

≥95% coverage.

---

## Child 09 — Engine adapters

### feat(anthropometrics): Drake / Pinocchio / MyoSuite / Simscape engine adapters

Five adapter modules in `src/shared/python/anthropometrics/engine_adapters/`. Each implements `EngineAdapter` Protocol:

```python
class EngineAdapter(Protocol):
    engine_name: str

    def export(self, anthropometrics: SubjectAnthropometrics, output_path: Path) -> None:
        """Write engine-native model file with the given anthropometrics."""

    def import_back(self, input_path: Path) -> SubjectAnthropometrics:
        """Inverse of export."""
```

- Drake — emits URDF (Drake reads URDF natively).
- Pinocchio — emits URDF.
- MyoSuite — emits URDF + MJCF (MyoSuite uses MuJoCo).
- OpenSim — emits .osim.
- Simscape — emits an input MAT file conforming to the existing Simscape input-MAT editor schema.

Tests: round-trip per adapter. Skip per-engine tests if engine wheel not installed.

≥85% coverage per adapter.

---

## Child 10 — High-level pipeline

### feat(anthropometrics): high-level pipeline.py orchestrator

`src/shared/python/anthropometrics/pipeline.py`:

```python
def run_pipeline(
    mocap_file: Path | str,
    *,
    subject_height_m: float | None = None,
    subject_mass_kg: float | None = None,
    estimator: Literal["de_leva", "dempster", "zatsiorsky"] = "de_leva",
    target_engines: Sequence[str] = ("drake", "mujoco", "pinocchio", "opensim"),
    output_dir: Path | str,
) -> SubjectAnthropometrics:
    """End-to-end: load mocap → compute anthropometrics → export per engine."""
```

Validation report: HTML at `output_dir/report.html` showing:
- Mass closure (sum of segment masses ÷ subject mass — should be 1.00 ± 1%).
- Inertia spectral check (all eigenvalues positive, triangle inequality holds).
- Length closure (sum of axial lengths ÷ subject height — sanity check).

Tests: end-to-end on `data/C3D_TA_Driver.c3d` produces all 4 engine outputs; loading each back recovers the canonical record.

≥85% coverage.

---

## Child 11 — Segment Properties Panel UI

### feat(c3d-viewer + matcher): SegmentPropertiesPanel UI

`src/shared/python/anthropometrics/ui/segment_properties_panel.py`:

`QGroupBox` showing for currently-selected segment:
- Length (m), Mass (kg), CoM offset (m), Inertia tensor (3×3 monospaced grid), Principal moments (sorted eigenvalues), Source method, Source subject params.

Wired into both:
- C3D Viewer's `viewer_3d_tab` — "Segments" sub-pane.
- Motion-Match Preview's `live_view_controller` — sidebar.

Headless tests + visual regression snapshot.

≥85% coverage.

---

## Child 12 — Calibration dialog

### feat(matcher): subject-anthropometrics calibration dialog

`src/shared/python/anthropometrics/ui/calibration_dialog.py`:

User-facing modal. Shows:
- Subject height + mass spinboxes (auto-filled from C3D SUBJECT_INFO if present).
- Estimator combobox (de Leva / Dempster / Zatsiorsky).
- "Compute" button → live-updates the SegmentPropertiesPanel below.
- "Save subject record…" → writes to `~/.golf_modeling_suite/subjects/<id>.json`.
- "Export to engine…" → runs `pipeline.run_pipeline` for chosen engine.

Tests: headless; programmatic state-change reaches export.

≥80% coverage.

---

## Child 13 — Subject persistence

### feat(anthropometrics): SubjectAnthropometrics JSON persistence

`src/shared/python/anthropometrics/persistence.py`:
- `save_subject(record, path)` / `load_subject(path) -> SubjectAnthropometrics`.
- Schema v1.
- Default location `~/.golf_modeling_suite/subjects/`.

≥95% coverage.

---

## Child 14 — Tests + validation

### test(anthropometrics): comprehensive TDD coverage + validation against published tables

In addition to per-child unit tests (≥85% each), this issue ships:

1. **`tests/integration/anthropometrics/test_published_table_reproduction.py`** — for each of the de Leva / Dempster / Zatsiorsky tables, reproduce the published (mass_ratio, length_ratio, com_proximal_ratio, gyration_radii) within 1e-3. Reference: published papers' exact numerical tables.

2. **`tests/integration/anthropometrics/test_cross_engine_round_trip.py`** — for each of 4 engine adapters: export → re-import → recover identical canonical record (rtol=1e-9, atol=1e-12).

3. **`tests/integration/anthropometrics/test_pipeline_end_to_end.py`** — full pipeline against `data/C3D_TA_Driver.c3d` produces sane outputs (mass closure ±1%, all inertia eigenvalues positive).

≥80% line coverage on `anthropometrics`; ≥70% branch.

---

## Child 15 — Docs + ADR + cross-engine pipeline guide

### docs(anthropometrics): ADR + user guide + cross-engine pipeline guide

Three new docs:

1. **ADR** at `docs/adr/00<next>-anthropometrics-pipeline.md`:
   - Context: algorithms scattered, no UI exposure, no cross-engine bridge.
   - Decision: canonical `SegmentProperties` + Protocol-driven readers/writers/adapters + URDF as common interchange.
   - Consequences: cross-model anthropometrics pipeline; engines all work from the same subject record.

2. **User guide** `docs/user_guide/anthropometrics/quickstart.md`:
   - Pick subject mass + height in the matcher's calibration dialog → "Run pipeline" → see SegmentPropertiesPanel populate → "Export to Drake".
   - Worked example using `data/C3D_TA_Driver.c3d`.

3. **Cross-engine pipeline guide** `docs/user_guide/anthropometrics/cross_engine.md`:
   - Step-by-step: export from Drake URDF → import into Pinocchio → roundtrip to OpenSim .osim → reflect in MJCF.
   - The canonical `SubjectAnthropometrics` record is the source of truth.
   - Validation report HTML interpretation.

AGENTS.md updated.
