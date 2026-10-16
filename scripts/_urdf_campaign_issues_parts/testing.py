"""URDF Hardening Campaign issue definitions: testing and validation issues."""

from __future__ import annotations

ISSUES: list[tuple[str, list[str], str]] = []

# ============================================================
# TESTING / VALIDATION
# ============================================================
ISSUES.append(
    (
        "[tests] Cross-engine smoke test: load generated URDF in MuJoCo / Drake / Pinocchio / OpenSim",
        ["tests", "integration", "urdf", "cross-engine", "priority:high"],
        """## Goal

For each preset character (athletic, average, heavy, child, tall, short),
verify the generated URDF loads without error in all four physics engines:

- MuJoCo (`mujoco.MjModel.from_xml_string` after URDF→MJCF compile)
- Drake (`MultibodyPlant.AddModelFromFile`)
- Pinocchio (`pinocchio.buildModelFromUrdf`)
- OpenSim (via existing OpenSim URDF import path or document its absence)

## Implementation

New file: `tests/integration/test_urdf_engine_loadability.py`

```python
@pytest.mark.parametrize("preset", ["athletic", "average", "heavy"])
@pytest.mark.parametrize("engine", ["mujoco", "drake", "pinocchio", "opensim"])
def test_generated_urdf_loads_in_engine(preset, engine, tmp_path):
    builder = CharacterBuilder()
    params = CharacterBuilder.create_from_preset(preset)
    result = builder.build(params, mesh_output_dir=tmp_path)
    urdf_path = result.export_urdf(tmp_path)
    _load_in_engine(engine, urdf_path)  # raises on failure
```

Mark slow engines with `@pytest.mark.slow` so the default suite stays
under 60 s.

## Acceptance

- [ ] At least 3 presets × 4 engines = 12 parametrized tests passing
- [ ] Test runtime < 30 s per (preset, engine) pair
- [ ] CI gate added for the matrix

## Depends on

- "[arch] Decide canonical URDF generation subsystem"
- "[bug] CharacterBuildResult missing solver_status"
""",
    )
)

ISSUES.append(
    (
        "[tests] URDF schema validation against URDF spec",
        ["tests", "urdf", "priority:medium"],
        """## Goal

Every URDF the system generates must be **schema-valid XML** that conforms
to the URDF spec.

## Implementation

Add `tests/unit/tools/humanoid_character_builder/test_urdf_schema.py`:

1. Use `lxml` (already a dep) with the URDF XSD if available, OR use
   `urdfdom_py` / `urchecker` to parse and validate.
2. Property-based test: random `BodyParameters` → URDF parses cleanly.
3. Regression fixtures: known-bad URDFs from past bugs (capture them
   under `tests/fixtures/urdf/regressions/`).

## Acceptance

- [ ] Validation runs on every preset
- [ ] Hypothesis property test for arbitrary parameters
- [ ] Failure surface includes the offending element/line number
- [ ] Documented in `docs/testing/`
""",
    )
)

ISSUES.append(
    (
        "[tests] Determinism: same BodyParameters → byte-identical URDF",
        ["tests", "urdf", "determinism", "priority:medium"],
        """## Goal

Calling `builder.generate_urdf(params)` twice with the same `params` must
produce **byte-identical** XML.

This catches:
- Iterator-order non-determinism (dict, set)
- Floating-point formatting drift (different platforms)
- Embedded timestamps / paths / UUIDs

## Implementation

```python
def test_urdf_generation_is_deterministic():
    params = BodyParameters(height_m=1.80, mass_kg=80.0)
    builder = CharacterBuilder()
    a = builder.generate_urdf(params)
    b = builder.generate_urdf(params)
    assert a == b
```

Run for several presets and across 50+ repetitions per preset.

## Acceptance

- [ ] Test passes on Linux + Windows + macOS CI
- [ ] If non-determinism found, fixed (sort dict keys, fix FP formatter)
""",
    )
)

ISSUES.append(
    (
        "[tests] Inertia validation: mass conservation + positive-definite tensors",
        ["tests", "urdf", "physics", "priority:high"],
        """## Goal

Physical sanity checks on every generated character:

1. **Mass conservation** — sum of segment masses ≈ requested `mass_kg`
   (within 0.1%).
2. **Positive-definite inertia** — every segment's 3x3 inertia tensor
   has all eigenvalues > 0.
3. **Triangle inequality** — for each segment, every pair of principal
   moments satisfies `Iₐ + Iᵦ ≥ Iᵧ` (a real rigid body).
4. **Center of mass** — within the segment's bounding volume.

## Implementation

`tests/unit/tools/humanoid_character_builder/test_urdf_inertia_validation.py`

Use `numpy.linalg.eigvalsh` for the positive-definite check.

## Acceptance

- [ ] All four checks pass for every preset
- [ ] Failure message identifies the offending segment + which check
- [ ] Test added for each `InertiaMode`
""",
    )
)

ISSUES.append(
    (
        "[tests] Joint limit anatomical sanity tests",
        ["tests", "urdf", "humanoid", "priority:medium"],
        """## Goal

Verify generated joint limits match human anatomy. Catches typos like
swapped lower/upper bounds, units wrong (degrees vs radians), or limits
that allow knee hyperextension.

## Sanity checks

| Joint                         | Constraint                       |
| ----------------------------- | -------------------------------- |
| Knee (flexion)                | upper ≥ 0; lower ≤ 0; range ≥ 120° |
| Knee (extension past 0°)      | upper ≤ 5°                       |
| Elbow                         | range ≥ 130°                     |
| Hip (sagittal flexion)        | range 100°-150°                  |
| Spine joints                  | range 30°-60° each axis          |
| Lower < upper                 | for every joint                  |
| All limits in radians         | sanity bound: `abs(limit) < 2π`  |

Source the table from a published reference (cite it in the test
docstring).

## Acceptance

- [ ] All joints pass
- [ ] Test failure message names the joint and which sanity rule failed
""",
    )
)

ISSUES.append(
    (
        "[tests] Self-collision pair generation for produced URDFs",
        ["tests", "urdf", "humanoid", "priority:low"],
        """## Goal

Generated URDFs need a self-collision exclusion list (or a `<disable_collisions>`
extension) so adjacent segments don't constantly collide. Today the URDF
likely emits no exclusions and physics simulators must guess.

## Implementation

1. Audit current URDF output: does it emit `<disable_collisions>` extensions?
2. If not, add a generator that walks the joint tree and excludes
   parent-child pairs and adjacent same-side limb pairs.
3. Test that for the default humanoid, the generated exclusion list has
   the expected structure.

## Acceptance

- [ ] Default character emits a collision exclusion list
- [ ] Test verifies no parent-child pair is missing from the exclusion
- [ ] Documented in user guide
""",
    )
)

ISSUES.append(
    (
        "[tests] Cross-engine FK equivalence (5 mm RMSE) for character builder URDFs",
        ["tests", "urdf", "cross-engine", "parity", "priority:high"],
        """## Goal

Mirror the motion-matching cross-engine equivalence gate (5 mm RMSE) for
URDFs produced by the character builder. Same parameters loaded into two
different engines should produce FK end-effector positions within 5 mm
across a sweep of joint configurations.

## Implementation

`tests/integration/test_urdf_cross_engine_fk.py`:

For each engine pair (MuJoCo, Drake, Pinocchio):
1. Build URDF for `preset="average"`.
2. Load in both engines.
3. Sample 100 random joint configurations within limits.
4. Compute FK for the 6 end effectors (hands, feet, head, pelvis).
5. Assert max RMSE ≤ 5 mm across configs.

## Why this matters

If the URDF means different things to different engines, our motion
matching numbers are silently wrong.

## Acceptance

- [ ] All 3 engine pairs within 5 mm RMSE
- [ ] CI gate added (mirror of motion-matching gate)
- [ ] Reference numbers logged to `reports/urdf_cross_engine.json`
""",
    )
)

ISSUES.append(
    (
        "[tests] Mesh inertia tests blocked by torch DLL — torch-free fallback",
        ["tests", "character-builder", "mesh", "priority:low"],
        """## Problem

`tests/unit/tools/humanoid_character_builder/test_mesh_generators.py:225`
is currently skipped with:

```
SKIPPED: torch DLL incompatible with current Python/OS environment
```

This means an entire code path (likely SMPLX-based mesh generation) has
no test coverage on this developer's machine and possibly in CI.

## Action

1. Confirm whether the skip is `pytest.mark.skipif` based on torch import
   or a hard skip.
2. Add a torch-free fallback for the unit test (mock the torch tensor
   ops, or use a fixture mesh on disk).
3. Keep the slow real-torch test under `@pytest.mark.slow` for the
   GPU/Linux CI matrix.

## Acceptance

- [ ] Test runs (and passes) on Windows without torch
- [ ] Real-torch test still runs in the appropriate CI matrix
""",
    )
)
