"""URDF Hardening Campaign issue definitions: epic and architecture-decision issues."""

from __future__ import annotations

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
