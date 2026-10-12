# UpstreamDrift Agent Handoff

> **Current state only**, capped at 150 lines per `AGENTS.md`. History lives in git and `docs/development/archive/AGENT_HANDOFF_ARCHIVE_2026-10-12.md`.

## Active: SPEC Change-Log Archive (#9415 / PR #12393)

- Branch `docs/9415-spec-archive`; PR #12393. Section 12 rows and trailing per-PR sections moved verbatim to `docs/spec/CHANGELOG_ARCHIVE.md`; live `SPEC.md` stays under the ~300 KB cap (`tests/unit/repo_hygiene/test_spec_archive.py`).
- Gates read the archive for history and `SPEC.md` for the live spec. This PR also re-archives the root handoff under the same 150-line policy.

## Markerless Mocap Program (#9063)

- Tools #4706 owns capture and contract schemas; UpstreamDrift #9069 (folded into #9422) owns app orchestration and makes no physical-lab qualification claim.
- ADR-0041 Amendment 1 (#9630, #9619): consumer-side self-calibration fitters live in `src/motion_capture/reconstruct/`; Tools keeps vendor-neutral reference geometry.
- Real-data acceptance is deferred to `docs/development/planning/DV-9619.md`; the rig soak to `DV-9613.md`.

## Active: Character Builder Docs and Tutorial (CMB-12 #11663)

- User docs under `docs/help/` and `docs/user_guide/`; tutorial numbers guarded by `tests/unit/character_builder/test_cmb12_*.py` (spec hash `4ee2c43d` on the driver document).
- Open: smooth skin / per-segment meshes (CMB-6 #11657), Playwright flow (CMB-7 #11658), lower-limb `mass_kg` scaling (#12199).

## Active: Per-Capture Grip Rotation Stack (#12221)

- Branches `claude/12221-*`; owner decision pending on `--fix-club-markers` default and calibrated grip application via `scripts/set_spec_grip_rotation.py`.

## Do Not

- Copy Tools-owned mocap sources into `src/shared/python` (ADR-0041).
- Treat software-only evidence as physical-lab qualification.

## Quick Gates (PR-Scoped)

```bash
python -m pytest tests/architecture/test_markerless_mocap_authority.py -q
python -m pytest tests/unit/repo_hygiene/test_spec_archive.py -q
python scripts/check_doc_size_budget.py
```
