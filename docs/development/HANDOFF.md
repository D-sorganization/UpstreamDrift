# Implementation Handoff

Keep this file current and concise. Replace instructional placeholders; do not append an unbounded transcript.

## Identity

- Repository: `D-sorganization/UpstreamDrift`
- Working directory: `C:/tmp/UD_w9542`
- Branch: `claude/issue-9542-exit-envelope`
- Baseline commit: `191351bdf84b8b46e2c95dddc073fc8555ce4b12`
- Implementation commit: `SELF` — the commit containing this update; resolve with `git rev-parse HEAD`
- Pull request: not created
- Governing issue/epic: #9542 (parent epic #9541)

## Objective and Status

- Objective: Reject reflected poses are already handled (PR #9574); this increment makes `SandDelivery`'s exit record one consistent owned snapshot (reopen defects 3+4), labels the legacy synthesized launch convention explicitly (defect 2), and delivers the versioned post-impact result envelope with frame transform and verdict (defects 1+5).
- Status: ready for review
- Completed: contradictory `exit_speed_m_s`/`exit_velocity_m_s` pairs refused at construction; list-supplied exit vectors owned/copied and re-validated so post-construction mutation cannot invalidate the record; `ExitVectorProvenance` labels (actual vs modeled convention) on the `to_post_impact_state` boundary; `PostImpactEnvelope` with `ValidityVerdict`, `FidelityTier.F0`, per-group frames, `HEAD_FRAME_TO_FLIGHT_TRANSFORM`, schema version, SHA-256 source digest, JSON round trip, flight-frame expressions; 20 new focused tests (RED first, then GREEN).
- Remaining: issue #9542 completion criteria that require merged protected PRs with rendered/numerical evidence per the issue's closing policy; GUI/workbench adoption of the envelope is intentionally out of scope.

## Files and Decisions

- Files changed: `src/bunkershot3d/ball/splash.py` (exit-record consistency + ownership), `src/bunkershot3d/ball/pipeline.py` (provenance labels, frame transform, envelope), `tests/bunkershot3d/ball/test_splash_transfer.py`, `tests/bunkershot3d/ball/test_pipeline_handoff.py` (RED-first tests), this handoff and `docs/development/DEVELOPMENT_LOG.md`.
- Key decisions: reused `bunkershot3d.solvers.envelope.ValidityVerdict` and `provenance.hashing.canonical_json` instead of new verdict/digest code (DRY); exit speed and exit vector are two representations of one measurement so they must agree within `EXIT_SPEED_AGREEMENT_TOLERANCE` (rel 1e-6, abs 1e-6); provenance determines frame (`ACTUAL_EXIT_STATE` → bunker solver world, `MODELED_CONVENTION` → launch frame) and the envelope enforces the pair; the pi-about-z `HEAD_FRAME_TO_FLIGHT_TRANSFORM` is a proper rotation (det +1), never an axis reflection; envelope `schema_version` 1 with digest-over-payload tamper detection.
- User-owned or unrelated worktree changes: none observed

## Validation

- `python -m pytest tests/bunkershot3d/ball/test_splash_transfer.py tests/bunkershot3d/ball/test_pipeline_handoff.py -q -o addopts=""` — pass, 64 passed (was 44 baseline + 20 new)
- `ruff check <changed files>` — pass; `black --line-length 100 <changed files>` — applied
- Neighbor consumers run: `tests/bunkershot3d/ball/test_launch_credibility.py tests/bunkershot3d/ball/test_lie_dependent_transfer.py tests/bunkershot3d/metrics/test_accelerated_mass_8659.py tests/unit/tools/bunker_shot_gui/test_uncertainty_propagation_9243.py` — 105 passed, 2 failed; the 2 failures (`test_uncertainty_propagation_9243.py::TestPlayabilityWindowCarriesABand::test_the_area_band_is_not_decorative`, `::TestWhatDominates::test_the_mass_interval_swamps_the_budget`) reproduce on clean baseline `191351b` and are unrelated pre-existing failures.

## Blockers and Risks

- Blockers: none
- Risks/assumptions: full bunkershot3d suite and heavy-backend CI not run locally per fleet focused-test policy; CI validates those. The #9574 test `test_preserves_actual_exit_velocity_and_angular_velocity` fixture was corrected to carry a consistent exit pair (its synthetic 12/3.5/-4.2 vector with the default 15 m/s exit speed now contradicts the new consistency contract).

## Next Steps

1. Open protected PR to `main` with `Fixes #9542`, label `agent:claude`, RED/GREEN evidence in body.
2. Follow-up: adopt `post_impact_envelope` in the bunker GUI/workbench boundary so the carry verdict and provenance replace the separate workbench verdict.

## Change Log

- `SELF` — SandDelivery exit-record consistency/ownership, provenance labeling, PostImpactEnvelope with frames/verdict/digest; tests RED→GREEN; handoff and development log created.