# Anthropometrics — User Guide

This guide shows how to build a subject-specific biomechanical model from
nothing more than a height and a mass, export it to URDF, and use the
result in any supported physics engine.

- Background and design rationale:
  [ADR-0009](../adr/0009-anthropometrics-pipeline.md)
- Adding a new engine adapter:
  [cross-engine pipeline guide](../development/anthropometrics_pipeline.md)

## Quickstart (5 minutes)

Build a subject from height + mass and write its `<inertial>` blocks to
a URDF:

```python
from pathlib import Path
import xml.etree.ElementTree as ET

from anthropometrics import save_subject
from anthropometrics.estimators import DeLevaEstimator
from anthropometrics.writers import write_urdf_inertial

# 1. Estimate a full subject from two scalars.
estimator = DeLevaEstimator()
subject = estimator.estimate(
    subject_id="demo_subject_01",
    height_m=1.78,
    mass_kg=72.0,
    sex="M",
)

# 2. Persist the canonical record (JSON, schema-versioned).
out_dir = Path("./out")
out_dir.mkdir(exist_ok=True)
save_subject(subject, out_dir / "demo_subject_01.json")

# 3. Emit one URDF <inertial> block per segment.
robot = ET.Element("robot", {"name": subject.subject_id})
for name, props in subject.segments:
    link = ET.SubElement(robot, "link", {"name": name})
    link.append(write_urdf_inertial(props))

ET.ElementTree(robot).write(
    out_dir / "demo_subject_01.urdf",
    encoding="utf-8",
    xml_declaration=True,
)
print(f"Wrote {out_dir / 'demo_subject_01.urdf'}")
```

The output URDF can be loaded directly by Drake, Pinocchio, MuJoCo, or
any URDF-aware engine. Every `<inertial>` block round-trips losslessly
through the matching reader (`anthropometrics.readers.read_urdf_inertial`).

## Cookbook

### Pick the right estimator

| Estimator             | When to use it                                                                       | Source                             |
| --------------------- | ------------------------------------------------------------------------------------ | ---------------------------------- |
| `DeLevaEstimator`     | Default. Sex-specific tables, modern measurement protocol, CoM on longitudinal axis. | de Leva (1996)                     |
| `DempsterEstimator`   | Reproducing classic biomechanics studies that cite Dempster ratios.                  | Dempster (1955)                    |
| `ZatsiorskyEstimator` | When you need raw cadaver-derived inertia tensors (e.g. for sensitivity analyses).   | Zatsiorsky-Seluyanov (1985 / 2002) |

All three implement the `Estimator` Protocol and are interchangeable:

```python
from anthropometrics.estimators import (
    DeLevaEstimator,
    DempsterEstimator,
    ZatsiorskyEstimator,
)

for cls in (DeLevaEstimator, DempsterEstimator, ZatsiorskyEstimator):
    subject = cls().estimate(
        subject_id="cmp",
        height_m=1.78,
        mass_kg=72.0,
        sex="M",
    )
    print(cls.__name__, "->", len(subject.segments), "segments")
```

### Estimate segment lengths from mocap markers

```python
from anthropometrics.estimators import (
    SegmentDef,
    estimate_segment_lengths_from_markers,
)
import numpy as np

# Mean marker positions in metres (one row per frame, or a (3,) mean).
markers = {
    "RSHO": np.array([0.20, 0.00, 1.45]),
    "RELB": np.array([0.25, 0.00, 1.15]),
    "RWRA": np.array([0.27, 0.00, 0.90]),
}
defs = [
    SegmentDef(name="upper_arm_R", proximal="RSHO", distal="RELB"),
    SegmentDef(name="forearm_R",   proximal="RELB", distal="RWRA"),
]
lengths = estimate_segment_lengths_from_markers(markers, defs)
print(lengths)  # {'upper_arm_R': 0.30..., 'forearm_R': 0.25...}
```

Mocap-derived lengths can be combined with regression-derived masses by
constructing `SegmentProperties` directly with
`build_segment_properties_with_inertia`.

### Compute an inertia tensor from primitive geometry

When you need a non-tabulated segment (e.g. a piece of equipment):

```python
from anthropometrics.estimators import (
    inertia_from_cylinder,
    inertia_from_ellipsoid,
    inertia_from_gyration_radii,
)

# Solid cylinder, mass 0.4 kg, radius 0.02 m, length 0.30 m, axis = +Z.
I_cyl = inertia_from_cylinder(mass_kg=0.4, radius_m=0.02, length_m=0.30)
# Solid ellipsoid with semi-axes (a, b, c) in metres.
I_ell = inertia_from_ellipsoid(mass_kg=4.0, semi_axes_m=(0.05, 0.05, 0.20))
# From radii of gyration (kx, ky, kz) about CoM.
I_kg  = inertia_from_gyration_radii(mass_kg=4.0, gyration_m=(0.04, 0.04, 0.10))
```

Each helper returns a 3 × 3 ndarray that already satisfies the
`SegmentProperties` invariants (symmetric, positive-definite, triangle
inequality).

### Persist and reload a subject

```python
from pathlib import Path
from anthropometrics import load_subject, save_subject

save_subject(subject, Path("subject.json"))
restored = load_subject(Path("subject.json"))
assert restored == subject  # frozen dataclasses compare by value
```

The on-disk format is schema-versioned (see `SCHEMA_VERSION`). Reloading
re-runs every DbC invariant — corrupt files fail loudly at `load_subject`
rather than mid-simulation.

### Read C3D subject metadata

Many C3D files carry `SUBJECT_INFO` / `PROCESSING` parameter blocks.
`read_c3d_subject_metadata` extracts the height / mass / sex hints so
you do not have to ask the user twice:

```python
from anthropometrics import read_c3d_subject_metadata
from anthropometrics.estimators import DeLevaEstimator

meta = read_c3d_subject_metadata("data/C3D_TA_Driver.c3d")
subject = DeLevaEstimator().estimate(
    subject_id=meta.subject_id or "ta_driver",
    height_m=meta.height_m or 1.78,
    mass_kg=meta.mass_kg or 72.0,
    sex=meta.sex or "unspecified",
)
```

## Troubleshooting

**`ValueError: inertia_tensor violates triangle inequality on principal moments`**
The principal moments fail `Ix + Iy >= Iz`. This usually means a custom
inertia computation forgot the parallel-axis transform back to the CoM,
or used inconsistent units. Re-derive with `inertia_from_*` helpers.

**`ValueError: com_xyz_m magnitude exceeds 2 * length_m`**
The CoM vector is in the wrong frame (e.g. world coordinates instead of
the segment's local frame). `SegmentProperties.com_xyz_m` is expressed in
the segment-local frame with the proximal joint at the origin.

**`ValueError: segments must be non-empty`**
You constructed `SubjectAnthropometrics` with an empty tuple. Estimators
always produce a populated subject; this only happens when building one
by hand.

**Estimator sex defaults to male.**
de Leva published only male and female tables. `sex="unspecified"` falls
back to the male table to preserve historical behaviour. Pass `sex="F"`
explicitly for female subjects.

**Round-trip URDF mismatch.**
The URDF spec expresses inertia at the link CoM. If your source URDF
expresses inertia at the joint origin, apply the parallel-axis transform
before constructing `SegmentProperties`.

**Numbers disagree between Drake and Pinocchio.**
Both engines should consume the same URDF emitted by `write_urdf_inertial`
and produce identical mass matrices to numerical precision. If they
diverge, the engine adapter is reordering axes or applying its own
scaling — see the
[cross-engine pipeline guide](../development/anthropometrics_pipeline.md)
for the validation harness.

## Where the code lives

- Canonical record: `src/shared/python/anthropometrics/segment_properties.py`
- Subject record: `src/shared/python/anthropometrics/_subject_anthropometrics.py`
- Protocols: `src/shared/python/anthropometrics/contracts.py`
- Estimators: `src/shared/python/anthropometrics/estimators/`
- Readers / writers: `src/shared/python/anthropometrics/readers/`,
  `…/writers/`
- Persistence: `src/shared/python/anthropometrics/persistence.py`
