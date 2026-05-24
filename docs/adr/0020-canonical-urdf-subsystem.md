# ADR 0020: Canonical URDF subsystem (humanoid_character_builder vs model_generation)

- **Status:** Proposed (awaiting decision)
- **Date:** 2026-05-08
- **Tracking issue:** [#4521](https://github.com/D-sorganization/UpstreamDrift/issues/4521)
- **Campaign:** [URDF Hardening Campaign #4520](https://github.com/D-sorganization/UpstreamDrift/issues/4520)

> Renumbered to ADR-0020 on 2026-05-23 to remove duplicate ADR-0006/ADR-0007 slots.

## Context

UpstreamDrift currently has **two parallel URDF generation subsystems** with substantially overlapping responsibilities. Until this campaign, neither was clearly canonical, both shipped tests, and the public API in `src/shared/python/__init__.py` re-exported types from both.

| Concern             | `humanoid_character_builder/`                         | `model_generation/`              |
| ------------------- | ----------------------------------------------------- | -------------------------------- |
| URDF XML emission   | `generators/urdf_generator.py`, `_urdf_xml_writer.py` | `builders/urdf_writer.py`        |
| Mesh generation     | `generators/mesh_generator*.py`                       | `mesh/`                          |
| Inertia computation | `mesh/inertia_calculator.py`                          | `inertia/`                       |
| Builder API         | `interfaces/api.py::CharacterBuilder`                 | `builders/parametric_builder.py` |
| Editor              | _none_                                                | `editor/frankenstein_editor.py`  |
| Library             | `presets/`                                            | `library/`                       |
| Converters          | _none_                                                | `converters/simscape/`           |
| Plugin system       | _none_                                                | `plugins/`                       |
| REST API            | _none_                                                | `api/rest_api.py`                |
| CLI                 | _none_                                                | `cli/main.py`                    |

Phase 1+2 of the URDF Hardening Campaign brought both subsystems' result classes into alignment with the canonical motion-matching `solver_status` contract, but the underlying duplication remains. We need a decision before further structural refactoring.

## Decision options

### Option A — Merge

Fold `humanoid_character_builder` into `model_generation` as a domain-specific builder at `model_generation/humanoid/`.

- One URDF writer, one mesh module, one inertia module
- Largest refactor (~6–8 PRs estimated)
- Maximum code unification; minimum future maintenance
- Risk of churn in motion-matching tests that import from `humanoid_character_builder`

### Option B — Layer (recommended)

`model_generation` is the **low-level URDF / mesh / inertia toolkit**. `humanoid_character_builder` is the **anthropometric domain layer** that composes the toolkit.

- Hard rule: `humanoid_character_builder` MUST NOT contain its own URDF writer; it composes `model_generation/builders/urdf_writer.py`
- Hard rule: `humanoid_character_builder` MUST NOT contain its own inertia primitives; it composes `model_generation/inertia/`
- Mesh generation is allowed to live in either, but with one canonical interface
- Medium refactor (~3–4 PRs estimated)
- Clearest separation of concerns: anthropometry is a discoverable domain, generic URDF tooling is a discoverable toolkit

### Option C — Keep both, narrow scope

`humanoid_character_builder` owns the humanoid domain end-to-end; `model_generation` is for non-humanoid (clubs, balls, generic kinematic chains).

- Smallest refactor (~1 PR for docs + lint rule)
- Duplication remains but is intentional
- Risk: future features have to be implemented twice or the line drifts again

## Recommendation: Option B (Layer)

Reasoning:

1. **The duplication is real, not philosophical.** Both subsystems emit URDF XML, both compute inertia primitives, both have mesh generators. Removing duplication is the higher-order win.
2. **Anthropometry is a domain, not infrastructure.** Body parameters, segment definitions, vertex group mappings, SMPLX/MakeHuman bindings — these are _specific_ to humanoid character creation. Keeping them isolated from the generic toolkit makes both easier to evolve.
3. **The test surface confirms the layering already.** `tests/unit/tools/model_generation/` tests low-level primitives (Link, Joint, Inertia, geometry conversions). `tests/unit/tools/humanoid_character_builder/` tests body-parameter-driven workflows. The test directories already reflect the right boundary.
4. **Option A loses domain clarity.** Folding anthropometry into `model_generation/humanoid/` makes the "generic toolkit" claim weaker — `model_generation` would now contain humanoid-specific code that has nothing to do with URDF emission. Option B keeps the toolkit truly generic.
5. **Option C is the do-nothing-much option.** It documents the current mess as the intended state, leaves duplication, and lets the next refactor be even bigger.

Estimated migration effort: 3–4 PRs over the campaign:

1. Extract `model_generation/inertia/` as the canonical inertia computation module; route `humanoid_character_builder` callers through it.
2. Extract `model_generation/builders/urdf_writer.py` as the canonical XML emitter; route `humanoid_character_builder.generators.urdf_generator` through it.
3. Standardize the mesh-generator interface in one place (likely `humanoid_character_builder.generators` since that's where SMPLX/MakeHuman live), with `model_generation` having a thin adapter for non-humanoid mesh primitives.
4. Update `src/shared/python/__init__.py` re-exports + ADR finalization + boundary doc (#4523).

## Consequences

### Positive

- One source of truth for URDF emission (closes a class of regressions where outputs differ).
- Domain-driven layering — tests, docs, and contributors all benefit.
- Unblocks #4523 (boundary documentation), #4533 (split api.py), and the cross-engine validation work in #4535/#4542.

### Negative

- Migration cost: 3–4 PRs of churn before the new layer feels stable.
- Some integrations may need updating (any caller currently importing from `humanoid_character_builder.mesh.primitive_inertia` would now get a re-export from `model_generation.inertia`).

### Risks

- If Option B is adopted but enforcement is weak, the code can drift back into duplication. Mitigation: add a CI check that `humanoid_character_builder/` does not import its own URDF XML writer (#4523 acceptance criteria).

## Acceptance criteria for closing this ADR

- [ ] User picks A, B, or C (or amends with rationale)
- [ ] If B is picked: migration tracked as 4 sub-issues under #4521
- [ ] `docs/architecture/URDF_SUBSYSTEM_BOUNDARY.md` written (closes #4523)
- [ ] `src/shared/python/__init__.py` re-exports updated to match
- [ ] No file imports from both subsystems (CI check)
