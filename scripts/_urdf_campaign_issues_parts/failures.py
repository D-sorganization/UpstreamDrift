"""URDF Hardening Campaign issue definitions: concrete test-failure issues."""

from __future__ import annotations

ISSUES: list[tuple[str, list[str], str]] = []

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
tests/scripts/test_build_humanoid_models.py::test_opensim_check_handles_missing_submodule
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
