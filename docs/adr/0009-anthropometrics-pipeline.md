# ADR-0009: Unified Anthropometrics Pipeline

- Status: Accepted
- Date: 2026-05-08
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: EPIC [#4797](https://github.com/D-sorganization/UpstreamDrift/issues/4797),
  child issues [#4800](https://github.com/D-sorganization/UpstreamDrift/issues/4800),
  [#4815](https://github.com/D-sorganization/UpstreamDrift/issues/4815)–[#4828](https://github.com/D-sorganization/UpstreamDrift/issues/4828),
  this ADR closes [#4827](https://github.com/D-sorganization/UpstreamDrift/issues/4827)

## Context

Before this work, anthropometric scaling lived in three disconnected places:

- `humanoid_character_builder/core/anthropometry.py` — de Leva (1996) ratio
  table used by the character builder GUI only.
- `motion_pipeline/scaling/anthropometric.py` — segment-length estimation
  from mocap markers, used only by the matcher.
- A handful of per-engine `inertia.py` modules that each computed inertia
  tensors with subtly different conventions, axis orderings, and units.

Symptoms:

- The same subject produced different masses, lengths, and inertia tensors
  in Drake, Pinocchio, MuJoCo, and OpenSim runs.
- No GUI exposed the underlying numbers; users could not audit or override.
- No round-trip: a subject calibrated in the matcher could not be exported
  to OpenSim or re-imported from MJCF.
- Triangle-inequality and positive-definiteness violations slipped through
  silently when ratios were applied to extreme heights / masses.

Constraints:

- SI units only, frozen public dataclasses, Design-by-Contract validation.
- No vendor / lab / person names in code identifiers (citations in
  docstrings are fine).
- Per-segment inertia tensor calc ≤ 1 ms; whole-subject pipeline ≤ 100 ms.
- ≥ 85 % line coverage on new code; validation tests must reproduce
  published table values to documented precision.

## Decision

Adopt a single canonical record, a small set of runtime-checkable
Protocols, and URDF as the lingua franca for cross-engine exchange.

**Canonical record.** `src/shared/python/anthropometrics/segment_properties.py`
defines a frozen `SegmentProperties` dataclass with all SI-unit fields
needed by every engine:

```python
from anthropometrics import SegmentProperties

# Every instance is validated on construction:
#   - positive scalar lengths / masses
#   - inertia tensor symmetric, positive-definite,
#     and triangle-inequality on principal moments
#   - |com| <= 2 * length_m
```

A `SubjectAnthropometrics` bundles segments plus subject scalars.

**Protocols** (`anthropometrics.contracts`):

| Protocol        | Purpose                                              |
| --------------- | ---------------------------------------------------- |
| `Estimator`     | `(height, mass, sex, age) -> SubjectAnthropometrics` |
| `Reader`        | `path -> SubjectAnthropometrics`                     |
| `Writer`        | `(SubjectAnthropometrics, path) -> None`             |
| `EngineAdapter` | `SegmentProperties -> engine-native object`          |

All four are `@runtime_checkable` so call sites can fail fast.

**URDF as interchange.** The URDF `<inertial>` reader/writer pair is the
canonical cross-engine bridge. Drake, Pinocchio, MuJoCo, and Simscape all
have first-class URDF importers; OpenSim `.osim` has a paired
reader/writer in the same package.

**Estimators.** Three regression estimators ship in-tree:

| Module                                | Citation                                  |
| ------------------------------------- | ----------------------------------------- |
| `from_de_leva.DeLevaEstimator`        | de Leva, P. (1996), _J. Biomechanics_     |
| `from_dempster.DempsterEstimator`     | Dempster (1955), WADC TR-55-159           |
| `from_zatsiorsky.ZatsiorskyEstimator` | Zatsiorsky-Seluyanov (1985, revised 2002) |

de Leva is the default — its sex-specific tables, modern measurement
conventions, and CoM placement on the longitudinal axis match what every
downstream engine expects without correction. Dempster is retained for
legacy comparability with older biomechanics literature; Zatsiorsky is
retained for studies that need raw cadaver-derived inertia tensors.

## Alternatives Considered

1. **Per-engine ad-hoc scaling.** Status quo. Rejected — the symptoms
   above are inherent: every engine grew its own conventions because
   nothing forced agreement.
2. **OpenSim `.osim` as canonical interchange.** Rejected — `.osim` is
   tightly coupled to the OpenSim wrapped-muscle / Body / Joint model.
   Engines without that machinery (Pinocchio, plain Drake MultibodyPlant)
   would need a lossy translation layer.
3. **Single God-class subject record.** Rejected — would couple I/O,
   estimation, and engine translation. Protocol-driven separation lets
   each subsystem evolve on its own cadence.
4. **Vendor a third-party library** (e.g. `opensim-core`, `biomechanics`).
   Rejected — license incompatibility, install footprint, and none cover
   all four engine targets.

## Consequences

- **Positive**
  - One subject record, identical numbers across all engines.
  - GUIs (matcher calibration dialog, `SegmentPropertiesPanel` in the
    c3d-viewer) read and write the same canonical record.
  - Round-trip URDF / `.osim` / MJCF guaranteed by the read↔write test
    pairs in `tests/anthropometrics/`.
  - DbC catches non-physical inputs at the dataclass boundary, not deep
    inside an engine integrator.
- **Negative**
  - Adding a new engine requires writing an `EngineAdapter`. The
    development guide
    [`anthropometrics_pipeline.md`](../development/anthropometrics_pipeline.md)
    walks through this end-to-end.
  - Estimator ratio tables are now versioned data files
    (`estimators/ratios/*.json`); editing them requires a regression test
    update.
- **Follow-ups**
  - Issue #4828 — first-party adapters for Drake, Pinocchio, MuJoCo,
    MyoSuite, Simscape.
  - Issue #4823 — `SegmentPropertiesPanel` UI component.
  - Issue #4819 — comprehensive validation against published tables.
  - Future: subject-specific MRI / CT scan ingestion as an additional
    `Estimator`.

## Validation

- `tests/anthropometrics/test_validation_published_tables.py` — every
  estimator must reproduce its source paper's tabulated values to
  documented precision (de Leva: 4 sig figs; Dempster: 3; Zatsiorsky: 4).
- `tests/anthropometrics/test_roundtrip_*.py` — `read(write(x)) == x`
  for URDF, `.osim`, and MJCF.
- Property-based tests (`hypothesis`) on `SegmentProperties.__post_init__`
  to confirm DbC invariants reject every class of malformed input.
- CI gates: `ruff check`, `ruff format --check`, file-size budget,
  pytest with the coverage `fail_under` from `pyproject.toml`.

## See also

- User guide: [`docs/user_guide/anthropometrics.md`](../user_guide/anthropometrics.md)
- Cross-engine pipeline guide:
  [`docs/development/anthropometrics_pipeline.md`](../development/anthropometrics_pipeline.md)
