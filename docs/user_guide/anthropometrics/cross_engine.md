# Anthropometrics — Cross-Engine Pipeline

How to move a single subject between **Drake**, **Pinocchio**,
**OpenSim**, and **MJCF / MuJoCo / MyoSuite** without losing a single
significant figure.

> **Prerequisite.** Read the [quickstart](quickstart.md) first to
> produce a `SubjectAnthropometrics` and an output directory of
> per-engine files. This guide picks up from there.

The canonical `SubjectAnthropometrics` record is the *source of
truth*. URDF is the cross-engine *interchange* format because Drake
and Pinocchio read it natively; OpenSim and MJCF have paired adapters
in `engine_adapters/`. Every adapter pair satisfies
`numpy.allclose(a, b, rtol=1e-9, atol=1e-12)` on inertia tensors —
round-trips are lossless to numerical precision.

> **Background.** [ADR-0010](../../adr/0010-anthropometrics-pipeline.md)
> documents the design decision; this guide is the operational
> recipe.

---

## A. Export from `run_pipeline()` to URDF (Drake adapter)

The pipeline orchestrator emits all four formats in one call:

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
```

Or invoke a single adapter directly when you only need one format:

```python
from anthropometrics import ADAPTER_REGISTRY

drake = ADAPTER_REGISTRY["drake"]
drake.export(subject, Path("./out/ta_driver/drake.urdf"))
```

The Drake adapter writes a URDF whose `<inertial>` blocks express
mass, CoM, and the 3 × 3 inertia tensor in SI units, expressed at
the link CoM (the URDF spec requirement). Drake's
`MultibodyPlant.AddModelFromUrdf` consumes it without further
massaging.

---

## B. Import the URDF into Pinocchio

The Pinocchio adapter ships in
`anthropometrics.engine_adapters.pinocchio`. Use it directly to
recover an in-memory `SubjectAnthropometrics`:

```python
from anthropometrics import ADAPTER_REGISTRY

pinocchio_adapter = ADAPTER_REGISTRY["pinocchio"]
recovered = pinocchio_adapter.import_back(Path("./out/ta_driver/drake.urdf"))

# Numerically identical to the original record.
import numpy as np
for (n_a, p_a), (n_b, p_b) in zip(subject.segments, recovered.segments):
    assert n_a == n_b
    assert np.allclose(
        p_a.inertia_tensor, p_b.inertia_tensor, rtol=1e-9, atol=1e-12
    )
```

If you only need the Pinocchio `Model` for dynamics work (and not the
canonical record), bypass the adapter:

```python
import pinocchio

model = pinocchio.buildModelFromUrdf(str(Path("./out/ta_driver/drake.urdf")))
data = model.createData()
```

The URDF emitted by the Drake adapter is a drop-in for any
URDF-aware Pinocchio entry point.

---

## C. Round-trip to OpenSim `.osim`

OpenSim does not consume URDF directly; the OpenSim adapter writes
its own `.osim` file:

```python
from anthropometrics import ADAPTER_REGISTRY

opensim_adapter = ADAPTER_REGISTRY["opensim"]
opensim_adapter.export(subject, Path("./out/ta_driver/opensim.osim"))

# Re-import to confirm the round-trip.
recovered = opensim_adapter.import_back(Path("./out/ta_driver/opensim.osim"))
```

The `.osim` writer maps each `SegmentProperties` to an OpenSim
`Body`: `mass_kg → mass`, `com_xyz_m → mass_center`,
`inertia_tensor → inertia` (`Ixx`, `Iyy`, `Izz`, `Ixy`, `Ixz`,
`Iyz`). The reader inverts that mapping. Wrapped-muscle definitions
are preserved verbatim if they were already present in the source
`.osim` template — the adapter only touches `<Body>` inertials.

---

## D. Reflect in MJCF (MuJoCo / MyoSuite)

The MJCF writer in `engine_adapters/_mjcf_io.py` emits a
`<body><inertial …/></body>` tree consumable by both plain MuJoCo
and MyoSuite:

```python
from anthropometrics import ADAPTER_REGISTRY

myosuite_adapter = ADAPTER_REGISTRY["myosuite"]
myosuite_adapter.export(subject, Path("./out/ta_driver/mujoco.xml"))

recovered = myosuite_adapter.import_back(Path("./out/ta_driver/mujoco.xml"))
```

MJCF expresses inertia at the body's CoM by default, matching the
canonical record. The adapter writes the diagonal in
`<inertial diaginertia=…>` when the off-diagonals are below
`1e-12`, otherwise it falls back to the full `fullinertia` form.

---

## E. End-to-end round-trip script

Confirms losslessness across all four engines:

```python
from pathlib import Path

import numpy as np

from anthropometrics import ADAPTER_REGISTRY, run_pipeline

out = Path("./out/ta_driver")
subject = run_pipeline(
    mocap_file="data/C3D_TA_Driver.c3d",
    subject_height_m=1.75,
    subject_mass_kg=75.0,
    estimator="de_leva",
    target_engines=("drake", "mujoco", "pinocchio", "opensim"),
    output_dir=out,
)

paths = {
    "drake": out / "drake.urdf",
    "pinocchio": out / "pinocchio.urdf",
    "myosuite": out / "mujoco.xml",
    "opensim": out / "opensim.osim",
}

for engine, path in paths.items():
    recovered = ADAPTER_REGISTRY[engine].import_back(path)
    for (n_a, p_a), (n_b, p_b) in zip(subject.segments, recovered.segments):
        assert n_a == n_b, f"{engine}: segment name drift"
        assert np.allclose(
            p_a.inertia_tensor, p_b.inertia_tensor, rtol=1e-9, atol=1e-12
        ), f"{engine}: inertia drift"
        assert np.isclose(p_a.mass_kg, p_b.mass_kg, rtol=1e-9)
        assert np.isclose(p_a.length_m, p_b.length_m, rtol=1e-9)
print("all four engines round-tripped losslessly.")
```

The `tests/anthropometrics/test_roundtrip_*.py` suite runs this same
flow against fixture subjects in CI.

---

## F. Reading the validation report

`run_pipeline()` always emits `output_dir/report.html`. Open it in a
browser. Three sections matter for cross-engine work:

### 1. Mass closure

```
Sum of segment masses ÷ subject mass = 1.000000 (target 1.000 ± 1 %).
Status: OK
```

The estimator must allocate every kilogram of the subject's mass
across the segments. Drift > 1 % means the ratio table you picked
does not cover the full body (e.g. some Zatsiorsky variants omit the
neck) — switch estimators or add a residual segment.

### 2. Inertia spectral check

Per-segment table of principal moments `I₁ ≤ I₂ ≤ I₃`, plus two
status columns:

| Column   | Meaning                                                     |
| -------- | ----------------------------------------------------------- |
| **PD**   | All eigenvalues > 0 (positive-definite tensor).             |
| **Triangle** | `I₁ + I₂ ≥ I₃` and the two cyclic permutations.         |

Any **FAIL** here indicates a non-physical segment — the DbC layer
would have raised on construction, so a FAIL in the report means
the validation report itself is being run against a hand-edited
record that bypassed the constructor. Do not export to engines
until every row is **OK**.

### 3. Length closure

```
Sum of axial segment lengths ÷ subject height = 1.04xxx
```

This is informational only — it should be ≈ 1 for an upright body
plan but there is no hard threshold (segments overlap at the joints,
e.g. shoulder vs. neck). Treat values outside `[0.9, 1.15]` as
suspicious and re-check the chosen estimator.

### 4. Per-segment table

Each row shows the segment name, three principal moments in SI
units (`kg · m²`), the PD status, and the triangle status. Sort by
clicking the column header. For cross-engine debugging, compare
this table against the `<inertial>` block in the engine-native file
— the principal moments must match to 1e-9 relative tolerance.

---

## G. Adding a new engine

The `EngineAdapter` Protocol is the only contract:

```python
@runtime_checkable
class EngineAdapter(Protocol):
    engine_name: str
    def export(self, anthropometrics: SubjectAnthropometrics, output_path: Path) -> None: ...
    def import_back(self, input_path: Path) -> SubjectAnthropometrics: ...
```

Drop a new module under `engine_adapters/`, register it in
`engine_adapters/__init__.ADAPTER_REGISTRY`, add an entry to
`pipeline._ENGINE_EXTENSIONS`, and write a paired
`test_roundtrip_<engine>.py`. The new engine becomes available to
both the GUI dialog and the scripting `run_pipeline()` call without
further changes.

---

## See also

- [Quickstart](quickstart.md) — height + mass to URDF in five minutes.
- [Anthropometrics consolidated guide](../anthropometrics.md) —
  estimator cookbook, troubleshooting.
- [ADR-0009](../../adr/0009-anthropometrics-pipeline.md) — canonical
  record and Protocol surface.
- [ADR-0010](../../adr/0010-anthropometrics-pipeline.md) — pipeline
  orchestrator decision record.
- [Developer guide](../../development/anthropometrics_pipeline.md) —
  authoring an `EngineAdapter` end-to-end.
