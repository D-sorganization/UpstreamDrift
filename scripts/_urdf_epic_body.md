## Overview

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

- `src/shared/python/humanoid_character_builder/` — anthropometric character creator
- `src/shared/python/model_generation/` — generic builder/editor/exporter

## Goals

1. **Unblock** — every existing test passes (zero failures, zero errors).
2. **Decide** — one canonical URDF writer; documented boundary between the two subsystems.
3. **Validate** — generated URDFs load in MuJoCo / Drake / Pinocchio / OpenSim and produce equivalent FK within 5 mm RMSE.
4. **Harden** — deterministic output, schema-valid XML, conserved mass, positive-definite inertia tensors, sane joint limits.
5. **Cover** — advanced features (Frankenstein editor, Simscape converter, GUI explorer, CLI, REST API, SMPLX/MakeHuman backends) have at least smoke tests.
6. **Document** — per-subsystem readiness report replaces the omnibus one; user guide for the character builder.

## Sub-issues

### Phase 1 — Architecture decisions (do these first)

- [ ] #4521 [arch] Decide canonical URDF generation subsystem
- [ ] #4522 [arch] Define unified BuildResult contract with solver_status field
- [ ] #4523 [arch] Document architectural boundary

### Phase 2 — Unblock failing tests (depends on #4522)

- [ ] #4524 [bug] CharacterBuildResult missing solver_status — 6 tests
- [ ] #4525 [bug] BuildResult missing solver_status — 14 tests
- [ ] #4526 [bug] GeneratedMeshResult missing solver_status — 3 tests
- [ ] #4527 [bug] SMPLXMeshGenerator missing \_segment_mesh classmethod
- [ ] #4528 [bug] mesh_generator missing SMPLX_AVAILABLE / TRIMESH_AVAILABLE attrs — 6 tests
- [ ] #4529 [bug] test_security_fixes — model_library URL + SMPLX vertex validation
- [ ] #4530 [bug] test_simscape — Simscape converter URDF generation failure
- [ ] #4531 [bug] test_build_humanoid_models — opensim submodule check

### Phase 3 — Code quality (depends on Phase 2)

- [ ] #4532 [quality] Remove duplicated `if not (X is not None)` guards
- [ ] #4533 [quality] Resolve ARCHITECTURE_DEBT in interfaces/api.py
- [ ] #4534 [quality] Replace silent except in CharacterBuilder.build()

### Phase 4 — Validation coverage (parallel with Phase 3)

- [ ] #4535 [tests] Cross-engine smoke test — all 4 engines load URDF
- [ ] #4536 [tests] URDF schema validation against URDF spec
- [ ] #4537 [tests] Determinism — same params → byte-identical URDF
- [ ] #4539 [tests] Inertia validation — mass conservation + positive-definite
- [ ] #4540 [tests] Joint limit anatomical sanity tests
- [ ] #4541 [tests] Self-collision pair generation
- [ ] #4542 [tests] Cross-engine FK equivalence (5 mm RMSE)
- [ ] #4543 [tests] Mesh inertia torch-free fallback

### Phase 5 — Advanced feature audits (parallel with Phase 4)

- [ ] #4544 [audit] Frankenstein editor — feature inventory + coverage
- [ ] #4545 [audit] Simscape converter — feature inventory + integration test
- [ ] #4546 [audit] Character preset library — expand + document
- [ ] #4547 [audit] Model explorer GUI — smoke test
- [ ] #4548 [audit] model_generation CLI — smoke all subcommands
- [ ] #4549 [audit] model_generation REST API — smoke endpoints
- [ ] #4550 [audit] MakeHuman + SMPLX mesh backends — end-to-end with real assets

### Phase 6 — Documentation & governance (last)

- [ ] #4551 [docs] URDF subsystem readiness report
- [ ] #4552 [docs] URDF user guide / quickstart
- [ ] #4553 [ci] Subsystem-status matrix CI gate

## Success criteria

- [ ] All sub-issues closed
- [ ] `pytest tests/unit/tools/humanoid_character_builder/ tests/unit/tools/model_generation/` passes with zero failures and zero errors
- [ ] `pytest tests/integration/test_urdf_generation_smoke.py` passes
- [ ] Subsystem status matrix flips URDF from "alpha" to "production"
- [ ] `docs/status/SUBSYSTEM_STATUS.md` lists URDF as production-ready

## Recommended execution order (today's plan)

1. **Now**: #4522 (define `solver_status` contract) — unblocks 23+ tests
2. **Next**: #4524, #4525, #4526 (apply contract to result classes)
3. **Then**: #4521 (architecture decision) — informs everything downstream
4. **Then**: remaining bugs (#4527-#4531)
5. **Then**: code quality (#4532-#4534), in parallel with validation tests (#4535-#4543)
6. **Finally**: feature audits and docs

## Tracking

- Milestone: [URDF Hardening Campaign](../milestone/1)
- Label: `URDF-hardening`
- GitHub Project: _to be created_
