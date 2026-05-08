#!/usr/bin/env python3
"""One-shot script: create the URDF Hardening Campaign issues.

Run from the UpstreamDrift repo root:
    python3 scripts/_urdf_campaign_issues.py

After running, prints a summary and writes issue numbers to
scripts/_urdf_campaign_issues.json.

This script is intentionally a one-off; safe to delete after use.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MILESTONE = "URDF Hardening Campaign"
COMMON_LABELS = ["URDF-hardening"]


def gh_issue_create(title: str, body: str, labels: list[str]) -> int:
    all_labels = sorted(set(COMMON_LABELS + labels))
    cmd = [
        "gh",
        "issue",
        "create",
        "--title",
        title,
        "--body",
        body,
        "--milestone",
        MILESTONE,
    ]
    for lbl in all_labels:
        cmd += ["--label", lbl]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # gh prints the URL on success; extract issue number
    url = out.stdout.strip().splitlines()[-1]
    num = int(url.rsplit("/", 1)[-1])
    print(f"  #{num}  {title}")
    return num


ISSUES: list[tuple[str, list[str], str]] = []

# ============================================================
# EPIC
# ============================================================
ISSUES.append(
    (
        "[Tracking] URDF / Character Builder Hardening Campaign",
        ["epic", "tracking", "urdf", "character-builder", "priority:high"],
        """## Overview

This is the umbrella tracking issue for bringing the URDF generation /
humanoid character builder subsystem to **true production grade**.

The motion-matching pipeline was hardened in Waves 1-6 (see
`reports/PRODUCTION_READINESS_REPORT.md`), but the URDF / character builder
subsystem was **not** part of that campaign. Current status:

- 18 failing tests in `tests/unit/tools/humanoid_character_builder/`
- 23 failing tests + 6 collection errors in `tests/unit/tools/model_generation/`
- Two parallel URDF subsystems with overlapping responsibilities and no
  documented boundary
- `interfaces/api.py` carries a self-declared `ARCHITECTURE_DEBT` comment
- ~15 instances of duplicated `if not (X is not None)` AI-slop validation
- No cross-engine equivalence tests for produced URDFs
- No deterministic-output regression test
- No URDF schema validation in CI

## Scope

Two subsystems, both in scope:

- `src/shared/python/humanoid_character_builder/` — anthropometric
  character creator (BodyParameters → URDF + meshes + inertia)
- `src/shared/python/model_generation/` — generic builder/editor/exporter
  (parametric_builder, urdf_writer, frankenstein_editor, simscape_converter)

## Goals

1. **Unblock** — every existing test passes (zero failures, zero errors).
2. **Decide** — one canonical URDF writer; documented boundary between
   the two subsystems.
3. **Validate** — generated URDFs load in all four physics engines
   (MuJoCo / Drake / Pinocchio / OpenSim) and produce equivalent FK within
   5 mm RMSE.
4. **Harden** — deterministic output, schema-valid XML, conserved mass,
   positive-definite inertia tensors, sane joint limits.
5. **Cover** — advanced features (Frankenstein editor, Simscape converter,
   GUI explorer, CLI, REST API, SMPLX/MakeHuman backends) have at least
   smoke tests.
6. **Document** — per-subsystem readiness report replaces the omnibus one;
   user guide for the character builder.

## Success criteria

- [ ] All sub-issues closed
- [ ] `pytest tests/unit/tools/humanoid_character_builder/ tests/unit/tools/model_generation/` passes with zero failures and zero errors
- [ ] `pytest tests/integration/test_urdf_generation_smoke.py` passes
- [ ] CI gate: subsystem status matrix flips URDF from "alpha" to "production"
- [ ] `docs/status/SUBSYSTEM_STATUS.md` lists URDF as production-ready with current numbers

## Sub-issues

_(populated as issues are created — this comment will be edited after the
campaign issues are filed)_
""",
    )
)

# ============================================================
# ARCHITECTURE DECISIONS
# ============================================================
ISSUES.append(
    (
        "[arch] Decide canonical URDF generation subsystem",
        [
            "architecture",
            "urdf",
            "character-builder",
            "model-generation",
            "priority:high",
            "decoupling",
        ],
        """## Problem

Two parallel URDF generation subsystems exist with overlapping
responsibilities:

| Concern              | `humanoid_character_builder/`            | `model_generation/`                 |
| -------------------- | ---------------------------------------- | ----------------------------------- |
| URDF XML emission    | `generators/urdf_generator.py`, `_urdf_xml_writer.py` | `builders/urdf_writer.py`           |
| Mesh generation      | `generators/mesh_generator*.py`          | `mesh/`                             |
| Inertia computation  | `mesh/inertia_calculator.py`             | `inertia/`                          |
| Builder API          | `interfaces/api.py::CharacterBuilder`    | `builders/parametric_builder.py`    |
| Editor               | _none_                                   | `editor/frankenstein_editor.py`     |
| Library              | `presets/`                               | `library/`                          |
| Converters           | _none_                                   | `converters/simscape/`              |

Neither system is clearly "the" canonical one. Tests reference both. The
public API in `src/shared/python/__init__.py` re-exports both.

## Decision needed

Pick one of three architectures:

**(A) Merge** — fold `humanoid_character_builder` into `model_generation`
as a domain-specific builder (`model_generation/humanoid/`). One URDF
writer, one mesh module, one inertia module. **Largest refactor.**

**(B) Layer** — `model_generation` is the low-level URDF/mesh/inertia
toolkit; `humanoid_character_builder` is the anthropometric-domain layer
on top. Hard rule: character_builder must not contain its own URDF
writer; it composes `model_generation/builders/urdf_writer.py`.
**Medium refactor; clearest separation.**

**(C) Keep both, narrow scope** — character_builder owns the humanoid
domain end-to-end; model_generation is for non-humanoid (clubs, balls,
generic kinematic chains). Document that hard line. **Smallest refactor;
duplication remains.**

## Acceptance criteria

- [ ] ADR written in `docs/adr/` recording the decision and rationale
- [ ] `docs/architecture/URDF_SUBSYSTEM_BOUNDARY.md` describes the
      chosen split with a diagram
- [ ] `src/shared/python/__init__.py` re-exports updated to match
- [ ] All call sites in `tests/`, `apps/`, `src/launchers/` updated
- [ ] No file imports both subsystems (lint rule or grep check in CI)

## Recommendation

Option **(B) Layer**. It removes duplicated URDF emission, gives a clean
testing target, and keeps the anthropometric domain knowledge isolated.
""",
    )
)

ISSUES.append(
    (
        "[arch] Define unified BuildResult contract with solver_status field",
        ["architecture", "urdf", "dbc", "priority:high"],
        """## Problem

Three result classes are missing a `solver_status` attribute that the
test suite expects:

- `humanoid_character_builder.interfaces.api.CharacterBuildResult`
  (6 failing tests in `test_api.py`)
- `model_generation.builders.base_builder.BuildResult`
  (~14 failing tests in `test_integration_roundtrip.py`,
  `test_property_based.py`, `test_simscape.py`)
- `humanoid_character_builder.generators.mesh_generator.GeneratedMeshResult`
  (3 failing tests in `test_mesh_generators.py`)

The motion-matching `FitResult` (`src/shared/python/motion_matching/fit_result.py`)
established `solver_status` as the canonical success/failure signal
during the Wave 2 hardening campaign. The URDF subsystem was not migrated.

## Proposed contract

Define a shared `BuildStatus` enum and a `BuildResultProtocol` (or
mixin / dataclass base) in `src/shared/python/build_result.py`:

```python
class BuildStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # produced output but with warnings

@dataclass
class BuildResultBase:
    solver_status: BuildStatus
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
```

Each result class composes/inherits this base.

## Acceptance criteria

- [ ] `src/shared/python/build_result.py` exists and is unit tested
- [ ] `CharacterBuildResult`, `BuildResult`, `GeneratedMeshResult` all
      expose `solver_status` of type `BuildStatus`
- [ ] All 23 currently failing `solver_status` tests pass
- [ ] `success: bool` field deprecated in favor of `solver_status` (or
      preserved as `@property` for backwards compat for one release)
- [ ] DbC decorators set `solver_status` automatically on raise

## Depends on

- "[arch] Decide canonical URDF generation subsystem" (location of the
  shared base depends on the chosen architecture)
""",
    )
)

ISSUES.append(
    (
        "[arch] Document architectural boundary between humanoid_character_builder and model_generation",
        ["architecture", "documentation", "urdf", "priority:medium"],
        """## Problem

There is no document explaining when a developer should reach for
`humanoid_character_builder` vs `model_generation`. Both subsystems
expose `BodyParameters`-like inputs and produce URDFs.

## Acceptance criteria

- [ ] `docs/architecture/URDF_SUBSYSTEM_BOUNDARY.md` written
- [ ] Includes a decision tree: "If you need X → use Y"
- [ ] Includes a diagram of module dependencies
- [ ] Cross-linked from both subsystems' `__init__.py` docstrings
- [ ] Cross-linked from `AGENTS.md`

## Depends on

- "[arch] Decide canonical URDF generation subsystem"
""",
    )
)

# ============================================================
# CONCRETE TEST FAILURES
# ============================================================
ISSUES.append(
    (
        "[bug] CharacterBuildResult missing solver_status — 6 failing tests",
        ["bug", "tests", "character-builder", "urdf", "priority:high"],
        """## Failing tests (6)

```
tests/unit/tools/humanoid_character_builder/test_api.py::TestCharacterBuilder::test_build_default_params
tests/unit/tools/humanoid_character_builder/test_api.py::TestCharacterBuilder::test_build_custom_params
tests/unit/tools/humanoid_character_builder/test_api.py::TestQuickFunctions::test_quick_build_default
tests/unit/tools/humanoid_character_builder/test_api.py::TestQuickFunctions::test_quick_build_custom
tests/unit/tools/humanoid_character_builder/test_api.py::TestQuickFunctions::test_quick_build_with_preset
tests/unit/tools/humanoid_character_builder/test_api.py::TestQuickFunctions::test_quick_build_with_output
```

## Error

```
AttributeError: 'CharacterBuildResult' object has no attribute 'solver_status'
```

at `tests/unit/tools/humanoid_character_builder/test_api.py:37`, :51,
:218, :225, :231, :238.

## Root cause

`CharacterBuildResult` (in
`src/shared/python/humanoid_character_builder/interfaces/api.py`) exposes
`success: bool` but no `solver_status`. The tests were written to a
contract that was never implemented on this class.

## Fix

Add `solver_status: BuildStatus` per the unified contract issue. Set it
in `CharacterBuilder.build()` based on the try/except branches:

- success path → `BuildStatus.SUCCESS`
- caught `(PermissionError, OSError)` → `BuildStatus.FAILURE`
- validation warnings present → `BuildStatus.PARTIAL`

## Depends on

- "[arch] Define unified BuildResult contract with solver_status field"

## Acceptance

- [ ] All 6 named tests pass
- [ ] `success` retained as `@property` returning `solver_status == SUCCESS`
""",
    )
)

ISSUES.append(
    (
        "[bug] BuildResult missing solver_status — 14 failing tests in model_generation",
        ["bug", "tests", "model-generation", "urdf", "priority:high"],
        """## Failing tests (14 + 6 collection errors)

In `tests/unit/tools/model_generation/test_integration_roundtrip.py`:
- `TestParsedModelOperations::test_copy_is_independent` (ERROR)
- `TestParsedModelOperations::test_to_urdf_produces_valid_xml` (ERROR)
- `TestManualBuilderRoundtrip::test_single_link_roundtrip`
- `TestManualBuilderRoundtrip::test_two_link_chain_roundtrip`
- `TestManualBuilderRoundtrip::test_three_link_branching_roundtrip`
- `TestManualBuilderRoundtrip::test_inertia_values_survive_roundtrip`
- `TestManualBuilderRoundtrip::test_joint_limits_survive_roundtrip`
- `TestManualBuilderRoundtrip::test_material_survives_roundtrip`
- `TestManualBuilderRoundtrip::test_fixed_joint_roundtrip`
- `TestParametricBuilderRoundtrip::test_default_humanoid_roundtrip`
- `TestParametricBuilderRoundtrip::test_parametric_mass_distribution`
- `TestParametricBuilderRoundtrip::test_parametric_height_affects_geometry`
- `TestParametricBuilderRoundtrip::test_parametric_builder_produces_valid_xml`
- `TestParametricBuilderRoundtrip::test_parametric_custom_segment_roundtrip`
- `TestCompositeJointExpansion::test_universal_joint_expands_to_two_revolute`
- `TestCompositeJointExpansion::test_gimbal_joint_expands_to_three_revolute`

In `test_property_based.py`:
- `TestURDFXMLWellFormedness::test_arbitrary_box_body_produces_well_formed_xml`
- `TestURDFXMLWellFormedness::test_arbitrary_cylinder_body_produces_well_formed_xml`
- `TestURDFXMLWellFormedness::test_arbitrary_sphere_body_produces_well_formed_xml`
- `TestLinkJointHierarchyConsistency::test_two_link_chain_hierarchy_is_consistent`
- `TestLinkJointHierarchyConsistency::test_star_topology_all_children_connected`

## Error

```
AttributeError: 'BuildResult' object has no attribute 'solver_status'
```

at `tests/unit/tools/model_generation/test_integration_roundtrip.py:41`.

## Fix

Same as the CharacterBuildResult issue — add `solver_status: BuildStatus`
to `model_generation.builders.base_builder.BuildResult` per the unified
contract.

The 6 collection ERRORs likely stem from a fixture that uses
`solver_status` in setup; once the field exists, they should resolve. If
not, file follow-ups.

## Depends on

- "[arch] Define unified BuildResult contract with solver_status field"

## Acceptance

- [ ] All 14 named tests pass
- [ ] All 6 collection errors resolved or replaced with focused issues
""",
    )
)

ISSUES.append(
    (
        "[bug] GeneratedMeshResult missing solver_status — 3 failing tests",
        ["bug", "tests", "character-builder", "mesh", "priority:medium"],
        """## Failing tests

```
tests/unit/tools/humanoid_character_builder/test_mesh_generators.py::TestSMPLXGenerate::test_generate_handles_exception_gracefully
tests/unit/tools/humanoid_character_builder/test_mesh_generators.py::TestGeneratedMeshResult::test_successful_result
tests/unit/tools/humanoid_character_builder/test_mesh_generators.py::TestGeneratedMeshResult::test_failed_result
```

## Error

```
AttributeError: 'GeneratedMeshResult' object has no attribute 'solver_status'
```

at lines 385, 535, 616, 624.

## Fix

Same pattern: add `solver_status: BuildStatus` to `GeneratedMeshResult`
in `src/shared/python/humanoid_character_builder/generators/mesh_generator.py`.

## Depends on

- "[arch] Define unified BuildResult contract with solver_status field"
""",
    )
)

ISSUES.append(
    (
        "[bug] SMPLXMeshGenerator missing _segment_mesh classmethod",
        ["bug", "tests", "character-builder", "mesh", "priority:medium"],
        """## Failing tests

```
tests/unit/tools/humanoid_character_builder/test_mesh_generators.py::TestSMPLXSegmentMesh::test_segment_extracts_correct_vertices
tests/unit/tools/humanoid_character_builder/test_mesh_generators.py::TestSMPLXSegmentMesh::test_empty_segment
```

## Error

```
AttributeError: type object 'SMPLXMeshGenerator' has no attribute '_segment_mesh'
```

## Investigation needed

The tests call `SMPLXMeshGenerator._segment_mesh(...)` as a classmethod.
The current implementation either:
1. Renamed `_segment_mesh` to something else, or
2. Made it an instance method, or
3. Was never implemented (tests are aspirational).

Check `src/shared/python/humanoid_character_builder/generators/mesh_generator_smplx.py`
and `_mesh_smplx.py` to determine which.

## Fix

Once the actual segmenting logic is located, expose it as a classmethod
matching the test's expected signature, OR update the tests if the
signature has legitimately moved.

## Acceptance

- [ ] Both tests pass
- [ ] If logic moved, comment + ADR explaining the new entry point
""",
    )
)

ISSUES.append(
    (
        "[bug] mesh_generator module missing SMPLX_AVAILABLE / TRIMESH_AVAILABLE attrs — 6 patch failures",
        ["bug", "tests", "character-builder", "mesh", "priority:medium"],
        """## Failing tests (6)

```
TestSMPLXAvailability::test_unavailable_when_smplx_missing
TestSMPLXAvailability::test_unavailable_when_model_dir_missing
TestSMPLXAvailability::test_returns_error_result_when_smplx_missing
TestSMPLXAvailability::test_returns_error_when_trimesh_missing
TestMakeHumanAvailability::test_returns_error_when_unavailable
TestMakeHumanGenerate::test_generate_with_mocked_subprocess
TestMakeHumanGenerate::test_generate_fails_when_script_fails
```

## Error

```
AttributeError: <module 'humanoid_character_builder.generators.mesh_generator'>
    does not have the attribute 'SMPLX_AVAILABLE'
AttributeError: ... does not have the attribute 'TRIMESH_AVAILABLE'
AttributeError: ... does not have the attribute '_trimesh_module'
```

## Root cause

Tests use `unittest.mock.patch("...mesh_generator.SMPLX_AVAILABLE", ...)`
expecting module-level capability flags. The current module appears to
have moved these into nested helpers (`_mesh_smplx.py`, `_mesh_makehuman.py`)
without leaving compatibility re-exports.

## Fix

Two options:

**(A)** Re-export module-level boolean flags from `mesh_generator`:
```python
from humanoid_character_builder.generators._mesh_smplx import SMPLX_AVAILABLE
from humanoid_character_builder.generators._mesh_smplx import TRIMESH_AVAILABLE
from humanoid_character_builder.generators._mesh_smplx import _trimesh_module
```

**(B)** Update tests to patch the new locations.

Recommend (A) — keeps the public surface stable.

## Acceptance

- [ ] All 7 tests pass
- [ ] No regression in headless / CI environments where SMPLX/trimesh
      truly unavailable
""",
    )
)

ISSUES.append(
    (
        "[bug] test_security_fixes — model_library URL restriction + SMPLX vertex validation",
        ["bug", "tests", "model-generation", "security", "priority:high"],
        """## Failing tests (2)

```
tests/unit/tools/model_generation/test_security_fixes.py::TestRepositoryURLRestriction::test_model_library_download_blocks_non_https_source_url
tests/unit/tools/model_generation/test_security_fixes.py::TestSMPLXVertexValidation::test_load_segmentation_logs_warning_on_fallback
```

## Why this is high priority

These are security-regression tests. Failure means either the security
guards have been removed/weakened, or the tests are broken. Either way,
investigate before shipping.

## Action

1. Run tests with `--tb=long` to capture full traceback.
2. Check `git log -p src/shared/python/model_generation/library/`
   for recent changes to URL handling.
3. Check `git log -p src/shared/python/humanoid_character_builder/generators/_mesh_smplx.py`
   for SMPLX segmentation logging changes.
4. Restore the guard or fix the test, with rationale.

## Acceptance

- [ ] Both tests pass
- [ ] If a guard was removed, restored with a comment explaining why
- [ ] If tests were stale, replaced with up-to-date assertions
""",
    )
)

ISSUES.append(
    (
        "[bug] test_simscape — Simscape converter URDF generation failure",
        ["bug", "tests", "model-generation", "priority:medium"],
        """## Failing test

```
tests/unit/tools/model_generation/test_simscape.py::TestSimscapeConverter::test_convert_generates_urdf
```

## Action

1. Capture full traceback with `pytest --tb=long`.
2. Likely a downstream effect of the missing `solver_status` contract;
   confirm and re-run after that lands.
3. If still failing, file a focused fix.

## Depends on

- "[bug] BuildResult missing solver_status — 14 failing tests in model_generation"
""",
    )
)

ISSUES.append(
    (
        "[bug] test_build_humanoid_models — opensim submodule check failure",
        ["bug", "tests", "humanoid", "opensim", "priority:low"],
        """## Failing test

```
tests/test_build_humanoid_models.py::test_opensim_check_handles_missing_submodule
```

## Likely cause

`shared/models/opensim/opensim-models` shows as modified in the working
tree. The test may be reading actual submodule state instead of mocked
state.

## Action

1. Capture full traceback.
2. Decide: should this test be skipped when the submodule is dirty, or
   should it always mock the file system?
3. Fix accordingly.
""",
    )
)

# ============================================================
# CODE QUALITY
# ============================================================
ISSUES.append(
    (
        "[quality] Remove duplicated `if not (X is not None)` guards in api.py",
        ["code-quality", "DRY", "character-builder", "tech-debt", "priority:medium"],
        """## Problem

`src/shared/python/humanoid_character_builder/interfaces/api.py` contains
~15 instances of duplicated nonsense validation guards, each repeated
verbatim twice in a row:

```python
if not (output_dir is not None):
    raise ValueError("output_dir must be provided")
if not (output_dir is not None):
    raise ValueError("output_dir must be provided")
```

Examples at lines 159-162, 243-246, 280-283, 358-361, 473-476, 529-532,
561-564, 620-623, 676-679, 711-714.

These guards are:
1. **Tautological** — `output_dir` is a typed parameter; if the caller
   passes `None` the type checker would catch it.
2. **Duplicated** — every guard appears twice in succession.
3. **Wrong signal** — `if not (x is not None)` is a confusing way to
   write `if x is None`.

This is unreviewed AI-generated boilerplate.

## Fix

Sweep the file. For each duplicated block:
- If the parameter is non-Optional in the signature, **delete** the guard.
- If the parameter genuinely needs runtime validation (e.g. user-facing
  paths from config), keep ONE guard with a useful message.

Apply the same sweep to any other file with `if not (.* is not None):`
duplicates.

## Acceptance

- [ ] `grep -c "if not (.* is not None)"` in `interfaces/api.py` is 0
      (or matches a justified count documented in commit message)
- [ ] No new test failures
- [ ] Ruff passes
- [ ] Net line reduction ~30 lines

## Search command

```bash
grep -rn "if not (.* is not None)" src/shared/python/humanoid_character_builder/
```
""",
    )
)

ISSUES.append(
    (
        "[quality] Resolve ARCHITECTURE_DEBT in interfaces/api.py — split into focused modules",
        [
            "code-quality",
            "refactor",
            "character-builder",
            "tech-debt",
            "architecture",
            "priority:medium",
        ],
        """## Problem

`src/shared/python/humanoid_character_builder/interfaces/api.py` opens
with:

```python
# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates
# excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal
# classes appropriately.
```

The file is 723 lines and contains:
- `SegmentMeshInfo` (data class)
- `ExportOptions` (data class)
- `CharacterBuildResult` (orchestrator with simulate / preview / export)
- `CharacterBuilder` (main builder with build / inertia / preset methods)
- Module-level `quick_build` / `quick_urdf` helpers

## Proposed split

```
interfaces/
  __init__.py        # public re-exports
  api.py             # CharacterBuilder only (orchestration)
  results.py         # CharacterBuildResult, SegmentMeshInfo
  options.py         # ExportOptions
  preview.py         # MuJoCo simulate() / preview() helpers
  quick.py           # quick_build / quick_urdf
```

## Acceptance

- [ ] No file in `interfaces/` exceeds 400 lines
- [ ] `from humanoid_character_builder import CharacterBuilder` still works
- [ ] All existing tests pass without modification
- [ ] `ARCHITECTURE_DEBT` comment removed
- [ ] CLAUDE.md mentions the split if it affects contributor workflow

## Depends on

- "[bug] CharacterBuildResult missing solver_status" (do that first to
  avoid merge churn)
- "[quality] Remove duplicated guards in api.py" (cleanup before split)
""",
    )
)

ISSUES.append(
    (
        "[quality] Replace silent except (PermissionError, OSError) in CharacterBuilder.build()",
        ["code-quality", "error-handling", "character-builder", "priority:medium"],
        """## Problem

`CharacterBuilder.build()` catches only `(PermissionError, OSError)`:

```python
except (PermissionError, OSError) as e:
    logger.error(f"Character build failed: {e}")
    return CharacterBuildResult(success=False, params=params, error_message=str(e))
```

This:
1. **Misses** the most common real-world failures: `ValueError` from
   parameter validation, `KeyError` from missing segments, `RuntimeError`
   from mesh backends, `ImportError` from optional deps.
2. **Conflates** environment errors with build errors (a missing temp dir
   permission ≠ a malformed body parameter).
3. Returns a **misleading** result — the caller can't tell why it failed.

## Fix

After the `solver_status` contract lands, classify failures:

```python
try:
    ...
except ValueError as e:           # bad params
    return _failure(BuildStatus.FAILURE, "validation", e)
except (PermissionError, OSError) as e:   # filesystem
    return _failure(BuildStatus.FAILURE, "io", e)
except ImportError as e:          # missing optional backend
    return _failure(BuildStatus.PARTIAL, "missing_backend", e)
# everything else → propagate (real bug)
```

## Acceptance

- [ ] Build failures distinguishable by error category
- [ ] Tests for each failure class
- [ ] Unexpected exceptions propagate (no bare except)

## Depends on

- "[arch] Define unified BuildResult contract with solver_status field"
""",
    )
)

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

# ============================================================
# ADVANCED FEATURE AUDITS
# ============================================================
ISSUES.append(
    (
        "[audit] Frankenstein editor (text_editor) — feature inventory + test coverage",
        ["audit", "model-generation", "tests", "priority:medium"],
        """## What to audit

`src/shared/python/model_generation/editor/`:
- `frankenstein_editor.py`
- `text_editor.py`, `text_editor_diff_mixin.py`, `text_editor_history_mixin.py`
- `editor_clipboard.py`, `editor_modifications.py`, `editor_types.py`
- `_text_editor_models.py`, `_text_editor_validation.py`

## Deliverables

1. `docs/architecture/FRANKENSTEIN_EDITOR.md` — what it does, what
   features are implemented, what's stubbed.
2. Test coverage report (`coverage run -m pytest tests/unit/tools/model_generation/test_editor.py`).
3. List of missing tests filed as separate issues.
4. A user-facing demo script in `examples/` that exercises the main
   editor flows end-to-end.

## Acceptance

- [ ] Architecture doc written
- [ ] Coverage ≥ 70% on each editor file (or justified exceptions)
- [ ] Demo script runs and produces a valid mutated URDF
""",
    )
)

ISSUES.append(
    (
        "[audit] Simscape converter — feature inventory + integration test",
        ["audit", "model-generation", "tests", "priority:medium"],
        """## What to audit

`src/shared/python/model_generation/converters/simscape/`

## Deliverables

1. Inventory of supported / unsupported Simscape features in
   `docs/converters/SIMSCAPE_CONVERTER.md`.
2. Round-trip integration test: Simscape input → URDF → MuJoCo load
   succeeds for at least 3 fixture models.
3. Known-limitation list filed as separate issues.

## Depends on

- "[bug] test_simscape — Simscape converter URDF generation failure"
""",
    )
)

ISSUES.append(
    (
        "[audit] Character preset library — expand presets, document",
        ["audit", "character-builder", "documentation", "priority:low"],
        """## What to audit

`src/shared/python/humanoid_character_builder/presets/loader.py` and
`presets/` data files.

## Deliverables

1. List currently shipped presets (athletic, average, heavy, ...).
2. Add at least: `child_8yo`, `senior_70yo`, `tall_male`, `petite_female`,
   `pro_golfer_male`, `pro_golfer_female`.
3. Document each preset's anthropometric source (NIH? CDC? sport-specific
   study?) — cite or remove.
4. `docs/user_guide/character_presets.md` listing every preset with
   expected use cases.

## Acceptance

- [ ] Preset count documented and expanded
- [ ] Every preset has a citation or "synthetic, derived from X"
""",
    )
)

ISSUES.append(
    (
        "[audit] Model explorer GUI (src/tools/model_explorer) — manual + automated smoke test",
        ["audit", "gui", "tests", "priority:medium"],
        """## What to audit

`src/tools/model_explorer/`:
- `main_window.py`
- `model_library.py`
- `model_loader_dialog.py`

## Deliverables

1. Manual smoke test: launch the GUI, load each preset, render, exit.
   Document in `docs/testing/MANUAL_SMOKE_TESTS.md`.
2. Automated `pytest-qt` (or equivalent) smoke test that imports the
   window and renders one frame headlessly.
3. Screenshot in user guide.

## Acceptance

- [ ] Manual smoke checklist documented
- [ ] At least one automated test that verifies the GUI imports + creates
      its main window without exception
""",
    )
)

ISSUES.append(
    (
        "[audit] model_generation CLI — smoke test all subcommands",
        ["audit", "cli", "model-generation", "tests", "priority:medium"],
        """## What to audit

`src/shared/python/model_generation/cli/main.py`

## Deliverables

1. Inventory of every subcommand (`--help` walk).
2. For each subcommand, a smoke test in
   `tests/integration/test_model_generation_cli.py` that runs it with
   minimal args and asserts exit 0.
3. Document the CLI in `docs/user_guide/model_generation_cli.md`.

## Acceptance

- [ ] Every subcommand has at least one smoke test
- [ ] User guide page exists
""",
    )
)

ISSUES.append(
    (
        "[audit] model_generation REST API — smoke test endpoints",
        ["audit", "api", "model-generation", "tests", "priority:medium"],
        """## What to audit

`src/shared/python/model_generation/api/rest_api.py` and
`api/generation_handlers.py`.

## Deliverables

1. Inventory every endpoint (method + path + payload + response shape).
2. Smoke tests using `httpx`/`fastapi.testclient` for each endpoint.
3. OpenAPI spec checked into `docs/api/model_generation_openapi.yaml`.

## Acceptance

- [ ] Endpoint table documented
- [ ] Each endpoint has a passing smoke test
- [ ] OpenAPI spec auto-generated and checked in
""",
    )
)

ISSUES.append(
    (
        "[audit] MakeHuman + SMPLX mesh backends — end-to-end with real assets",
        ["audit", "character-builder", "mesh", "priority:medium"],
        """## What to audit

`src/shared/python/humanoid_character_builder/generators/`:
- `mesh_generator_makehuman.py`, `_mesh_makehuman.py`
- `mesh_generator_smplx.py`, `_mesh_smplx.py`

## Deliverables

1. Document the asset/model setup required for each backend
   (where to put SMPLX `.npz` files, how to install MakeHuman).
2. End-to-end test (marked `slow` or `live_simulation`) that:
   - Starts from `BodyParameters`
   - Runs the real backend (not mocked)
   - Produces meshes on disk
   - Validates mesh files are non-empty STL/OBJ
3. CI strategy: a separate optional workflow that runs these on a
   dedicated runner with the assets cached.

## Depends on

- "[bug] mesh_generator module missing SMPLX_AVAILABLE / TRIMESH_AVAILABLE attrs"
- "[tests] Mesh inertia tests blocked by torch DLL"
""",
    )
)

# ============================================================
# DOCS / GOVERNANCE
# ============================================================
ISSUES.append(
    (
        "[docs] URDF subsystem readiness report (replace omnibus PRODUCTION_READINESS_REPORT for URDF)",
        ["documentation", "urdf", "process", "priority:medium"],
        """## Problem

`reports/PRODUCTION_READINESS_REPORT.md` claims the system is
production-ready, but only covers motion-matching. URDF/character builder
is silently lumped under "production" by association.

## Action

1. Create `reports/subsystem_status/URDF_READINESS.md`.
2. Honest current state: failing test count, missing coverage, advanced
   features yet to validate.
3. Tied to this campaign's exit criteria.
4. Update `PRODUCTION_READINESS_REPORT.md` to clarify that it covers
   motion-matching only, with a "see per-subsystem reports" pointer.

## Acceptance

- [ ] URDF readiness report exists
- [ ] Numbers in it are auto-generated from CI (not hand-edited)
- [ ] Updated daily by a CI job (or at least at PR merge)
""",
    )
)

ISSUES.append(
    (
        "[docs] URDF user guide / quickstart",
        ["documentation", "urdf", "character-builder", "priority:medium"],
        """## Goal

A user landing on the repo should be able to generate their first humanoid
URDF in under 5 minutes.

## Deliverable

`docs/user_guide/character_builder_quickstart.md` covering:

1. Install (Python deps, optional SMPLX/MakeHuman)
2. 5-line minimal example using `quick_urdf`
3. Adjusting body parameters (height, mass, build)
4. Choosing presets
5. Loading the URDF in MuJoCo / Drake / Pinocchio
6. Where to go next (frankenstein editor, simscape converter, GUI)

## Acceptance

- [ ] Quickstart written
- [ ] Every code block in it executes (CI doctest or notebook test)
- [ ] Linked from main README
""",
    )
)

ISSUES.append(
    (
        "[ci] Subsystem-status matrix — fail CI if production subsystems have failing tests",
        ["ci", "governance", "process", "priority:medium"],
        """## Goal

Prevent the `PRODUCTION_READINESS_REPORT.md` style drift, where a report
claims production status but tests are red.

## Implementation

1. `docs/status/SUBSYSTEM_STATUS.yaml` — declarative subsystem registry:

   ```yaml
   - name: motion_matching
     status: production
     test_paths: [tests/unit/motion_matching/, tests/motion_matching/]
   - name: urdf
     status: alpha
     test_paths: [tests/unit/tools/humanoid_character_builder/, tests/unit/tools/model_generation/]
   ```

2. CI step `scripts/ci/check_subsystem_status.py`:
   - For each subsystem with status=production, run its tests.
   - Fail the build if any test fails.
   - Allow alpha/beta to fail without blocking merge (but report).

3. PR template asks: "Did you update the subsystem registry?"

## Acceptance

- [ ] Registry checked in
- [ ] CI gate added
- [ ] Documented in `docs/governance/SUBSYSTEM_STATUS.md`
""",
    )
)


SKIP_TITLES_PREFIX = (
    "[Tracking] URDF",
    "[arch] Decide canonical",
    "[arch] Define unified BuildResult",
    "[arch] Document architectural",
    "[bug] CharacterBuildResult",
    "[bug] BuildResult missing",
    "[bug] GeneratedMeshResult",
    "[bug] SMPLXMeshGenerator",
    "[bug] mesh_generator module missing",
    "[bug] test_security_fixes",
    "[bug] test_simscape",
    "[bug] test_build_humanoid_models",
    "[quality] Remove duplicated",
    "[quality] Resolve ARCHITECTURE_DEBT",
    "[quality] Replace silent except",
    "[tests] Cross-engine smoke test:",
    "[tests] URDF schema validation",
    "[tests] Determinism:",
)


def main() -> None:
    remaining = [
        (t, lbls, b) for t, lbls, b in ISSUES if not t.startswith(SKIP_TITLES_PREFIX)
    ]
    print(
        f"Creating {len(remaining)} issues (skipping {len(ISSUES) - len(remaining)} already created) against milestone '{MILESTONE}'...\n"
    )
    created: list[dict[str, object]] = []
    for title, labels, body in remaining:  # noqa
        try:
            num = gh_issue_create(title, body, labels)
            created.append({"number": num, "title": title, "labels": labels})
        except subprocess.CalledProcessError as e:
            print(f"  ERR creating {title!r}: {e.stderr}")
            raise

    out = Path("scripts/_urdf_campaign_issues.json")
    out.write_text(json.dumps(created, indent=2))
    print(f"\nWrote summary to {out}")


if __name__ == "__main__":
    main()
