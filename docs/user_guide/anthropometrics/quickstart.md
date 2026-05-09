# Anthropometrics — Quickstart

Build a subject-specific biomechanical model from nothing more than a
height and a mass, view every segment's mass / length / inertia in
the UI, and export a URDF that any URDF-aware engine can load.

> **Background.** Design rationale lives in
> [ADR-0010](../../adr/0010-anthropometrics-pipeline.md). For the
> cross-engine round-trip flow (Drake → Pinocchio → OpenSim → MJCF)
> see the [cross-engine pipeline guide](cross_engine.md).

This guide covers the GUI flow first, then the scripting API for
users who prefer code.

> **Screenshots.** Image references in this doc point at
> `docs/user_guide/anthropometrics/img/`. Screenshots will be added
> in a follow-up PR — the placeholder paths below are the contract
> for that follow-up to satisfy.

---

## A. GUI flow — Subject Calibration Dialog

The dialog ships in
`src/shared/python/anthropometrics/ui/calibration_dialog.py` and is
launched from the matcher (Tools → *Subject Calibration*) or directly
from a tile in the launcher.

### 1. Pick the mocap file and subject scalars

![Calibration dialog — initial screen](img/calibration_dialog.png)

- **C3D file.** Click *Browse…* and select `data/C3D_TA_Driver.c3d`
  (the bundled worked example). The dialog reads
  `SUBJECT_INFO`/`PROCESSING` parameters and pre-fills any of
  `subject_id`, `height_m`, `mass_kg`, `sex`, `age_years` it finds.
- **Height (m)** and **Mass (kg).** Spinboxes are bounded to
  physically realistic ranges (height 0.5 – 2.5 m, mass 10 – 300 kg).
  For the worked example, set **height = 1.75 m** and **mass = 75 kg**
  if not auto-filled.
- **Sex.** *M*, *F*, or *unspecified*. de Leva publishes only male
  and female tables; *unspecified* falls back to the male table.
- **Estimator.** *de_leva* (default), *dempster*, or *zatsiorsky*.
  See the [cookbook in the consolidated user guide](../anthropometrics.md#pick-the-right-estimator)
  for guidance on which to pick.
- **Target engines.** Multi-select; defaults to all four
  (`drake`, `mujoco`, `pinocchio`, `opensim`).

### 2. Run the pipeline

Click **Run pipeline**. The dialog calls
`anthropometrics.pipeline.run_pipeline()` under the hood and
populates the segment list on the left. The full call takes well
under 100 ms for a 16-segment subject.

### 3. Inspect the segments

![Segment properties panel populated](img/segment_properties_panel.png)

The right-hand `SegmentPropertiesPanel` displays the currently
selected segment:

- mass (kg), length (m), CoM (x, y, z) in m,
- 3 × 3 inertia tensor (kg · m²),
- principal moments and triangle-inequality status badge,
- estimator citation + table page reference.

Iterate through the segments — every value should be physically
plausible. If a value looks wrong, the DbC layer would have raised
during construction; the panel will not display non-physical
records.

### 4. Export to Drake (or any other engine)

Click **Export to Drake** to write the URDF. The same button row
exposes *Export to Pinocchio*, *Export to OpenSim*, *Export to
MJCF*. Files land in the directory shown at the bottom of the
dialog (default `~/.golf_modeling_suite/subjects/<subject_id>/`).

Click **Save** to persist the canonical `subject.json` (schema
version `SCHEMA_VERSION`). Reloading via *Open subject…* re-runs
every DbC invariant — corrupt files fail loudly at load time,
never mid-simulation.

---

## B. Worked example — `data/C3D_TA_Driver.c3d`

Bundled with the repo. The `SUBJECT_INFO` block in this file does
not encode height or mass, so the dialog will leave the spinboxes at
their defaults. Set:

| Field      | Value           |
| ---------- | --------------- |
| height_m   | 1.75            |
| mass_kg    | 75.0            |
| sex        | M               |
| estimator  | de_leva         |

Expected outputs after clicking *Run pipeline* with all four target
engines selected:

| Field                          | Approximate value | Source                |
| ------------------------------ | ----------------- | --------------------- |
| total mass (Σ segment masses)  | 75.0 kg ± 1 %     | mass closure          |
| total axial length             | ≈ 1.75 m          | length closure        |
| segments produced              | 16                | de Leva male table    |
| trunk mass                     | ≈ 33.3 kg         | de Leva male, 0.4346  |
| thigh mass (each)              | ≈ 10.5 kg         | de Leva male, 0.1416  |
| `report.html` mass_ratio cell  | green / *OK*      | within 1 % tolerance  |

Files written to `output_dir`:

```
output_dir/
├── subject.json        # canonical record (schema-versioned)
├── report.html         # deterministic validation report
├── drake.urdf          # URDF (consumed by Drake & Pinocchio)
├── pinocchio.urdf      # same content, distinct filename
├── mujoco.xml          # MJCF (MyoSuite / plain MuJoCo)
└── opensim.osim        # OpenSim model file
```

Open `report.html` in a browser to see the validation tables — see
the [cross-engine pipeline guide § Reading the validation report](cross_engine.md#f-reading-the-validation-report)
for how to interpret each section.

---

## C. Scripting API — same flow without the GUI

For batch jobs, CI fixtures, or notebook work, call
`run_pipeline()` directly:

```python
from pathlib import Path

from anthropometrics import run_pipeline

subject = run_pipeline(
    mocap_file="data/C3D_TA_Driver.c3d",
    subject_height_m=1.75,
    subject_mass_kg=75.0,
    estimator="de_leva",
    target_engines=("drake", "mujoco", "pinocchio", "opensim"),
    output_dir=Path("./out/ta_driver"),
)

print(f"{subject.subject_id}: {len(subject.segments)} segments")
print(f"total mass = {sum(p.mass_kg for _, p in subject.segments):.2f} kg")
print(f"de Leva citation: {subject.source_method}")
```

The returned `SubjectAnthropometrics` is the same canonical record
the GUI shows. Every downstream consumer (engine adapters, the
matcher, the C3D Viewer) reads from this single source of truth.

### Skipping the C3D file

If you only have height + mass and no mocap, build the subject from
the estimator directly:

```python
from anthropometrics import save_subject
from anthropometrics.estimators import DeLevaEstimator

subject = DeLevaEstimator().estimate(
    subject_id="demo_subject_01",
    height_m=1.75,
    mass_kg=75.0,
    sex="M",
)
save_subject(subject, "out/demo_subject_01.json")
```

Then export to engines via the registry:

```python
from anthropometrics import ADAPTER_REGISTRY

ADAPTER_REGISTRY["drake"].export(subject, "out/demo.urdf")
```

The engine adapter is the only piece that knows about the on-disk
format. The canonical record is engine-agnostic.

---

## D. Where to go next

- [Cross-engine pipeline guide](cross_engine.md) — moving the same
  subject between Drake, Pinocchio, OpenSim, and MJCF.
- [Anthropometrics consolidated guide](../anthropometrics.md) —
  estimator cookbook, mocap-derived lengths, primitive-geometry
  inertia helpers, persistence, troubleshooting.
- [ADR-0009](../../adr/0009-anthropometrics-pipeline.md) — original
  canonical record + Protocols.
- [ADR-0010](../../adr/0010-anthropometrics-pipeline.md) — pipeline
  orchestrator and cross-engine bridge.
