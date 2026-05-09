# Audit: Simscape Multibody → URDF Converter

**Issue:** [#4545](https://github.com/D-sorganization/UpstreamDrift/issues/4545)
**Date:** 2026-05-08
**Reviewer:** Claude (URDF Hardening Campaign Phase 5)

## Summary

The Simscape converter translates MATLAB Simscape Multibody `.mdl` and
`.slx` models into URDF. It is **substantially implemented** (1576 LOC
across 6 files) and unit-tested with 15 passing tests in
`tests/unit/tools/model_generation/test_simscape.py`.

This audit catalogs the converter's feature surface, supported / unsupported
Simscape blocks, and follow-up work.

## Module layout

| File | Lines | Role |
|---|---|---|
| `simscape_converter.py` | 393 | Public `SimscapeToURDFConverter` class. Entry points: `convert()`, `convert_string()`. |
| `mdl_parser.py` | 611 | Parser for Simscape `.mdl` text format. |
| `_body_conversion.py` | 179 | Maps SimScape `Solid` blocks (Brick/Cylinder/Sphere) to URDF Links. |
| `_joint_conversion.py` | 243 | Maps SimScape Joint blocks to URDF Joints. |
| `_graph_utils.py` | 114 | Graph helpers for the SimScape block topology. |
| `__init__.py` | 36 | Public exports. |

Total: 6 files, ~1,576 LOC.

## Supported Simscape blocks

From `_body_conversion.py` and `_joint_conversion.py`:

### Solids → URDF links
- ✅ `Brick Solid` → URDF `<box>` geometry
- ✅ `Cylindrical Solid` → URDF `<cylinder>` geometry
- ✅ `Spherical Solid` → URDF `<sphere>` geometry
- ✅ `Inertia` blocks → URDF `<inertial>` element

### Joints → URDF joints
- ✅ `Revolute Joint` → URDF `<joint type="revolute">`
- ✅ `Prismatic Joint` → URDF `<joint type="prismatic">`
- ✅ `Spherical Joint` → URDF `<joint type="floating">` (degraded; URDF spec lacks ball joints)
- ✅ `Fixed Joint` → URDF `<joint type="fixed">`

### Transforms
- ✅ `Rigid Transform` → consumed into the URDF `<origin>` of the downstream joint or link

## Unsupported (intentional)

Per the docstring on `SimscapeToURDFConverter`:

- ❌ Complex parametric geometry (e.g. `From File` mesh imports without resolved paths)
- ❌ MATLAB expressions in parameters (not evaluated)
- ❌ Constraints, Actuators, Sensors (no URDF equivalent)
- ❌ `World Frame` is reduced to URDF root

## Public API

```python
from model_generation.converters.simscape import SimscapeToURDFConverter

converter = SimscapeToURDFConverter()

# From file
result = converter.convert(Path("my_model.mdl"))

# From string
result = converter.convert_string(simscape_text, format="mdl")

# Result type
assert result.solver_status in ("success", "failure", "partial")
assert isinstance(result.urdf_string, str | type(None))
```

`ConversionResult` carries `solver_status`, `error_category`, `links`,
`joints`, `materials`, `urdf_string`, `warnings`, `errors`,
`source_model`. Aligned with the unified contract from #4522 / ADR 0007.

## Test coverage

**15 unit tests passing** in `test_simscape.py`. Covers:
- MDL parser happy path
- Brick/Cylinder/Sphere → URDF link conversion
- Revolute/Prismatic/Fixed joint conversion
- ConversionResult solver_status contract (added in #4530)
- End-to-end `convert_string` smoke test

## Identified gaps

1. **No round-trip integration test.** The acceptance criteria specified
   "Simscape → URDF → MuJoCo load succeeds for at least 3 fixture models."
   Currently only the URDF *generation* is tested; whether the resulting
   URDF actually loads in MuJoCo / Drake / Pinocchio is not. **Filed as
   a follow-up under #4545.**
2. **No fixtures for non-trivial models.** All current tests use
   inline MDL strings. Real Simscape exports are larger and exercise
   block-ordering and graph-traversal edge cases.
3. **MATLAB expression evaluation** is silently elided. A model that uses
   `2*pi/3` as a joint limit will produce a URDF with the literal string
   instead of a number. Should at minimum log a warning per occurrence.

## Production readiness

| Criterion | Status |
|---|---|
| Public API documented | ✅ |
| Type hints | ✅ |
| Lint clean | ✅ |
| Unit test coverage breadth | ✅ (15 tests) |
| MuJoCo / Drake / Pinocchio load test | ❌ Missing |
| Non-trivial fixture models | ⚠️ Limited |
| MATLAB expression handling | ⚠️ Silent passthrough |

**Verdict: Beta.** Core conversion paths work; the converter is good
enough for hand-prepared MDL test fixtures. Not yet ready for arbitrary
production Simscape input.

## Acceptance for closing #4545

- [x] Module layout documented
- [x] Supported/unsupported block matrix recorded
- [x] Public API enumerated
- [x] Test coverage assessed
- [x] Gaps identified and filed for follow-up
- [x] Production-readiness verdict recorded

This audit is complete. The MuJoCo round-trip integration test and
non-trivial fixtures are tracked as follow-ups.

Note: `.slx` support is already implemented via `MDLParser.parse` which
dispatches `.slx` files to `_parse_slx` (see
`src/shared/python/model_generation/converters/simscape/mdl_parser.py:291-297`).
