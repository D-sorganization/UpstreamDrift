# Cross-Engine Anthropometrics Pipeline — Developer Guide

This guide is for contributors adding a new physics engine to the
anthropometrics pipeline, or extending an existing adapter.

- Design rationale: [ADR-0009](../adr/0009-anthropometrics-pipeline.md)
- End-user perspective:
  [user guide](../user_guide/anthropometrics.md)

## Architecture in one picture

```
   ┌──────────────┐    ┌──────────────────────┐    ┌────────────────┐
   │ MoCap C3D    ├───►│  Estimator           ├───►│ SubjectAnthro- │
   │ Height/Mass  │    │ (DeLeva / Dempster / │    │   pometrics    │
   │ User dialog  │    │  Zatsiorsky / mocap) │    │   (canonical)  │
   └──────────────┘    └──────────────────────┘    └───────┬────────┘
                                                            │
                            ┌───────────────────────────────┼────────────────┐
                            ▼                               ▼                ▼
                  ┌────────────────┐             ┌────────────────┐ ┌────────────────┐
                  │ Writer         │             │ Writer         │ │ EngineAdapter  │
                  │ URDF <inertial>│             │ .osim Body     │ │ (Drake, Pin,   │
                  │ MJCF <body>    │             │                │ │  MuJoCo, …)    │
                  └────────┬───────┘             └────────┬───────┘ └────────┬───────┘
                           ▼                              ▼                  ▼
                       Drake / Pinocchio /            OpenSim         Native engine
                       MuJoCo / Simscape                              data structures
```

`SubjectAnthropometrics` is the single source of truth. Engines must
either (a) consume URDF emitted by `write_urdf_inertial`, or (b)
implement an `EngineAdapter` that translates `SegmentProperties`
directly into the engine's native representation.

## The `EngineAdapter` Protocol

Defined in `src/shared/python/anthropometrics/contracts.py`:

```python
from typing import Protocol, runtime_checkable
from anthropometrics import SegmentProperties

@runtime_checkable
class EngineAdapter(Protocol):
    def to_engine_segment(self, props: SegmentProperties) -> object:
        """Return the engine-native representation of *props*."""
        ...
```

The Protocol is intentionally minimal: one method, one segment in, one
opaque engine object out. Multi-segment composition is the caller's
responsibility (engines disagree about whether a "subject" is a kinematic
tree, a flat body list, or a compiled model handle).

## Adding a new engine adapter

Put your adapter under
`src/shared/python/anthropometrics/adapters/<engine_name>.py`.

### 1. Implement `to_engine_segment`

Use the existing Pinocchio adapter as the canonical example. A new
adapter for a hypothetical `MyEngine` that exposes a
`MyEngine.RigidBody(mass, com, I)` constructor would look like:

```python
# src/shared/python/anthropometrics/adapters/myengine.py
from __future__ import annotations

from anthropometrics import SegmentProperties

# Replace with the real engine import:
# import myengine


class MyEngineAdapter:
    """Translate SegmentProperties into MyEngine.RigidBody."""

    def to_engine_segment(self, props: SegmentProperties):
        # SegmentProperties is already SI, CoM in segment-local frame,
        # inertia at CoM, symmetric / PD / triangle-inequality verified.
        return myengine.RigidBody(  # noqa: F821
            mass=props.mass_kg,
            com=tuple(props.com_xyz_m.tolist()),
            inertia=props.inertia_tensor.tolist(),
        )
```

Two non-negotiable rules:

1. **Do not silently mutate physical quantities.** No unit conversions
   inside the adapter — `SegmentProperties` is SI by contract. If the
   engine expects grams or millimetres, convert at the call site and
   document it loudly.
2. **Do not skip the parallel-axis transform** if the engine expresses
   inertia at a frame other than the CoM. URDF, MJCF, and `.osim` all
   match the canonical CoM convention; some bespoke formats do not.

### 2. Register a `runtime_checkable` self-test

```python
from anthropometrics import EngineAdapter

assert isinstance(MyEngineAdapter(), EngineAdapter)
```

This belongs in the adapter's own module-level test file
(`tests/anthropometrics/adapters/test_myengine_adapter.py`) — it is the
cheapest possible regression check that the Protocol is satisfied.

### 3. Validate against published tables

Every adapter PR must include a test that:

1. Builds a subject from a fixed `(height, mass, sex)` triple using
   `DeLevaEstimator`.
2. Runs every segment through the new adapter.
3. Re-extracts mass, CoM, and inertia from the engine-native object.
4. Asserts they match the canonical `SegmentProperties` to documented
   precision (`rtol=1e-9, atol=1e-12` is the bar for in-process
   adapters; lossy formats document and justify a wider tolerance).

Skeleton:

```python
# tests/anthropometrics/adapters/test_myengine_adapter.py
import numpy as np
import pytest

from anthropometrics.estimators import DeLevaEstimator
from anthropometrics.adapters.myengine import MyEngineAdapter


@pytest.mark.unit
def test_myengine_roundtrip_preserves_inertia():
    subject = DeLevaEstimator().estimate(
        subject_id="ref",
        height_m=1.78,
        mass_kg=72.0,
        sex="M",
    )
    adapter = MyEngineAdapter()

    for _, props in subject.segments:
        body = adapter.to_engine_segment(props)
        assert body.mass == pytest.approx(props.mass_kg, rel=1e-12)
        np.testing.assert_allclose(
            body.com, props.com_xyz_m, rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            body.inertia, props.inertia_tensor, rtol=1e-9, atol=1e-12,
        )
```

### 4. Cross-engine parity test

Once two or more adapters exist for the same subject, add a parity test
that asserts identical mass-matrix entries within numerical tolerance.
The Drake ↔ Pinocchio parity test in
`tests/anthropometrics/test_cross_engine_parity.py` is the template.

## Validation against published tables

`tests/anthropometrics/test_validation_published_tables.py` is the
authoritative reference for the precision bar:

| Estimator  | Tolerance                                   | Source                             |
| ---------- | ------------------------------------------- | ---------------------------------- |
| de Leva    | 4 sig figs on CoM ratios and gyration radii | de Leva (1996), Tables 1–4         |
| Dempster   | 3 sig figs on mass / length ratios          | Dempster (1955), WADC TR-55-159    |
| Zatsiorsky | 4 sig figs on principal moments (kg·m²)     | Zatsiorsky-Seluyanov (1985 / 2002) |

When you change a ratio JSON in `estimators/ratios/`, the validation
test will fail until you also update the expected values block — this is
intentional. The change must cite the paper, table, and column being
amended in the PR description.

## Adding a new file format reader/writer

Pair every reader with a writer. The contract is `read(write(x)) == x`
for every valid `SegmentProperties`:

```python
# tests/anthropometrics/test_<format>_roundtrip.py
import numpy as np
from anthropometrics.estimators import DeLevaEstimator
from anthropometrics.readers.<format> import read_<format>
from anthropometrics.writers.<format> import write_<format>


def test_<format>_roundtrip_is_lossless(tmp_path):
    subject = DeLevaEstimator().estimate(
        subject_id="rt", height_m=1.78, mass_kg=72.0, sex="M",
    )
    for _, props in subject.segments:
        elem = write_<format>(props)
        restored = read_<format>(elem)
        assert restored.mass_kg == props.mass_kg
        np.testing.assert_allclose(
            restored.com_xyz_m, props.com_xyz_m, rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            restored.inertia_tensor, props.inertia_tensor,
            rtol=1e-9, atol=1e-12,
        )
```

## Performance budget

From the EPIC ([#4797](https://github.com/D-sorganization/UpstreamDrift/issues/4797)):

- Inertia tensor calc for one segment ≤ 1 ms.
- Bulk pipeline (load C3D → 16 segments → URDF) ≤ 100 ms.

Adapters are well below this bar — `to_engine_segment` is pure data
shuffling. If your adapter needs heavyweight engine setup (e.g. compiling
a MuJoCo XML), do that work once at module level and have
`to_engine_segment` reuse the cached engine handle.

## Checklist for an adapter PR

- [ ] `adapters/<engine>.py` implements `to_engine_segment` and is
      `isinstance(..., EngineAdapter)` true.
- [ ] Roundtrip test against `DeLevaEstimator` reference subject.
- [ ] Cross-engine parity test if a second adapter exists.
- [ ] No unit conversions or axis re-orderings without an explicit
      comment citing the engine's documented convention.
- [ ] Coverage on new code ≥ 85 % line, ≥ 75 % branch.
- [ ] No new `print()` in `src/`; use `logging`.
- [ ] No new TODO/FIXME without a tracked issue.
- [ ] User-facing changes are reflected in
      [`docs/user_guide/anthropometrics.md`](../user_guide/anthropometrics.md).
