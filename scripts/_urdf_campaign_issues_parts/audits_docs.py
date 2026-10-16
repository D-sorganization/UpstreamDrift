"""URDF Hardening Campaign issue definitions: advanced feature audit and docs/governance issues."""

from __future__ import annotations

ISSUES: list[tuple[str, list[str], str]] = []

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

`src/shared/python/model_generation/api/rest_api.py` (the public shim) and
`api/rest_api_core.py` with its route mixins `rest_api_generation.py` and
`api/rest_api_assets.py`.

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
