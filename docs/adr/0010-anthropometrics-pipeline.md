# ADR-0010: Anthropometrics Pipeline Orchestrator and Cross-Engine Bridge

- Status: Accepted
- Date: 2026-05-08
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: EPIC [#4797](https://github.com/D-sorganization/UpstreamDrift/issues/4797),
  closes [#4827](https://github.com/D-sorganization/UpstreamDrift/issues/4827).
  Extends [ADR-0009](0009-anthropometrics-pipeline.md), which introduced the
  canonical record and Protocol surface; this ADR documents the
  user-facing pipeline orchestrator, the calibration GUI, and URDF as
  the cross-engine interchange.

## Context

Anthropometric scaling code was scattered across the codebase before
the EPIC #4797 work landed:

- `humanoid_character_builder/core/anthropometry.py` carried a de Leva
  (1996) ratio table that only the character-builder GUI consumed.
- `motion_pipeline/scaling/anthropometric.py` produced segment lengths
  from mocap markers but only the matcher used them.
- Each per-engine wrapper (`drake/`, `pinocchio/`, `myosuite/`,
  `opensim/`, `simscape/`) carried its own inertia computation with
  subtly different conventions, axis orderings, and unit handling.

The visible symptoms:

- The same subject (1.78 m, 72 kg) produced different masses, segment
  lengths, and inertia tensors across Drake, Pinocchio, OpenSim, and
  MJCF runs — sometimes off by 5 % on principal moments.
- No GUI exposed the underlying numbers; users could not audit or
  override estimator output.
- A subject calibrated in the matcher could not be exported to OpenSim
  or re-imported from MJCF — there was no canonical bridge.
- Triangle-inequality and positive-definiteness violations slipped
  through silently when ratios were applied to extreme heights or
  masses.

Constraints carried over from ADR-0009:

- SI units only, frozen public dataclasses, Design-by-Contract
  validation at every boundary.
- No vendor / lab / person names in code identifiers (citations in
  docstrings only).
- Per-segment inertia tensor calc ≤ 1 ms; whole-subject pipeline
  ≤ 100 ms end-to-end.
- ≥ 85 % line coverage on new code; validation tests must reproduce
  published table values to documented precision.

## Decision

Adopt a single canonical record, a small set of runtime-checkable
Protocols, a high-level pipeline orchestrator, and URDF as the lingua
franca for cross-engine exchange.

**Canonical record.** `src/shared/python/anthropometrics/segment_properties.py`
defines the frozen `SegmentProperties` dataclass (mass, length, CoM,
3 × 3 inertia tensor in SI units). `SubjectAnthropometrics` bundles
the per-segment records with subject scalars (`subject_id`,
`height_m`, `mass_kg`, `sex`, `age_years`, `source_method`). Both
classes validate every invariant on construction:

- positive scalar lengths and masses,
- inertia tensor symmetric, positive-definite, with the triangle
  inequality on principal moments,
- `|com_xyz_m| <= 2 * length_m` (CoM cannot leave the segment).

**Protocols** (`anthropometrics.contracts`):

| Protocol        | Purpose                                              |
| --------------- | ---------------------------------------------------- |
| `Estimator`     | `(height, mass, sex, age) -> SubjectAnthropometrics` |
| `Reader`        | `path -> SubjectAnthropometrics`                     |
| `Writer`        | `(SubjectAnthropometrics, path) -> None`             |
| `EngineAdapter` | `SubjectAnthropometrics <-> engine-native file`      |

Every Protocol is `@runtime_checkable`; call sites can fail fast with
`isinstance` checks.

**Pipeline orchestrator.** `anthropometrics.pipeline.run_pipeline()`
is the single public entry point that drives the whole flow:

1. Load a C3D mocap file, fall back to `SUBJECT_INFO`/`PROCESSING`
   parameter blocks if scalars are not supplied.
2. Estimate per-segment lengths from mocap markers (best-effort).
3. Apply the chosen regression estimator (`de_leva` / `dempster` /
   `zatsiorsky`) to materialise a `SubjectAnthropometrics`.
4. Export to every engine in `target_engines` via the
   `ADAPTER_REGISTRY`.
5. Persist the canonical record to `output_dir/subject.json`.
6. Emit a deterministic `output_dir/report.html` validation report
   (mass closure, inertia spectral check, length closure).

**UI surface.**
`anthropometrics.ui.calibration_dialog.SubjectCalibrationDialog`
wraps `run_pipeline()` for end users: pick a C3D file, confirm
height/mass, choose an estimator and target engines, click *Compute*,
inspect every segment in `SegmentPropertiesPanel`, then *Save* /
*Export*. The dialog is a thin shell — every callable surface is also
available as a scripting API.

**URDF as interchange.** Drake and Pinocchio read URDF natively;
OpenSim `.osim` and MJCF have paired adapters in
`engine_adapters/_urdf_io.py` and `engine_adapters/_mjcf_io.py`.
URDF is verbose but universal — it is the *interchange* format, not
the in-memory canonical. Round-trips through every adapter satisfy
`numpy.allclose(a, b, rtol=1e-9, atol=1e-12)` on inertia tensors.

## Alternatives Considered

1. **Per-engine ad-hoc scaling (status quo).** Rejected — the symptoms
   above are inherent: every engine grew its own conventions because
   nothing forced agreement.
2. **URDF as the in-memory canonical record.** Rejected — URDF is
   ~30× larger than the dataclass on disk, and round-tripping through
   `xml.etree` on every internal call would blow the 1 ms / segment
   budget. URDF is right for *interchange*; a frozen dataclass is
   right for *compute*.
3. **OpenSim `.osim` as canonical interchange.** Rejected — `.osim`
   is tightly coupled to the OpenSim wrapped-muscle / Body / Joint
   model. Engines without that machinery (Pinocchio, plain Drake
   `MultibodyPlant`) would need a lossy translation layer.
4. **Single God-class subject record.** Rejected — would couple I/O,
   estimation, and engine translation. Protocol-driven separation
   lets each subsystem evolve independently.
5. **Vendor a third-party library** (e.g. `opensim-core`,
   `biomechanics-toolkit`). Rejected — license incompatibility,
   install footprint, and none cover all four engine targets.

## Consequences

- **Positive**
  - One subject record; identical numbers across Drake, Pinocchio,
    MyoSuite/MuJoCo, OpenSim, and Simscape.
  - GUI (`SubjectCalibrationDialog`) and scripting API
    (`run_pipeline`) read and write the *same* canonical record —
    no GUI/CLI drift.
  - Round-trip URDF / `.osim` / MJCF guaranteed by the
    `read(write(x)) == x` test pairs in
    `tests/anthropometrics/test_roundtrip_*.py`.
  - DbC catches non-physical inputs at the dataclass boundary, not
    deep inside an engine integrator.
  - Validation report is deterministic — snapshot-testable against
    `tests/fixtures/anthropometrics/expected_report.html`.
- **Negative**
  - Adding a new engine requires writing an `EngineAdapter`. The
    [cross-engine pipeline guide](../user_guide/anthropometrics/cross_engine.md)
    walks through this.
  - Estimator ratio tables are versioned data files
    (`estimators/ratios/*.json`); editing them requires a regression
    test update.
- **Follow-ups**
  - Subject-specific MRI / CT scan ingestion as an additional
    `Estimator`.
  - Optional EMG-driven inertia perturbation for athlete-specific
    studies.

## Validation

- `tests/anthropometrics/test_validation_published_tables.py` — every
  estimator must reproduce its source paper's tabulated values to
  documented precision (de Leva: 4 sig figs; Dempster: 3; Zatsiorsky: 4).
- `tests/anthropometrics/test_roundtrip_*.py` —
  `adapter.import_back(adapter.export(x)) == x` for URDF, `.osim`,
  and MJCF.
- `tests/anthropometrics/test_pipeline.py` —
  `run_pipeline()` against `data/C3D_TA_Driver.c3d` produces a
  deterministic `SubjectAnthropometrics` and a byte-stable
  `report.html`.
- Property-based tests (`hypothesis`) on
  `SegmentProperties.__post_init__` confirm DbC invariants reject
  every class of malformed input.
- CI gates: `ruff check`, `ruff format --check`, file-size budget,
  pytest with the coverage `fail_under` from `pyproject.toml`.

## See also

- [ADR-0009](0009-anthropometrics-pipeline.md) — original canonical
  record + Protocol surface.
- [Quickstart user guide](../user_guide/anthropometrics/quickstart.md)
- [Cross-engine pipeline guide](../user_guide/anthropometrics/cross_engine.md)
