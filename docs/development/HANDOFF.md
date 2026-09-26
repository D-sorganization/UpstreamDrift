# Current Handoff — Retire the Review-Comment-to-Issue Converter (RM#1755, Third Attempt)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-10978-v3`
- Branch: `chore/retire-comment-converter-v3` (baseline `b8c27a7d2`)
- Commit: `SELF`
- Pull request: #10992 (draft), supersedes #10978 (which conflicted with main)
- Issue: Repository_Management#1755; development-log entry DL-#1755
- Removed: `.github/workflows/Comment-to-Issue-Converter.yml` and the review-comment processor script; the `.github/WORKFLOWS.md` row, the Nightly Doc Organizer bullet and the `ci_workflow_map.md` row. `tests/ci/test_process_review_comments.py` is now a retirement guard; `tests/ci/test_ci_infrastructure.py` no longer lists the deleted workflow. SPEC's #10931 section is a retirement note.
- Validation: `pytest tests/ci/` 89 passed; `python scripts/check_spec_paths.py` exit 0.
- Workflow change: this PR ships alone.
- Next step: CI green, mark ready, arm via `automerge_guard.py`, close #10978 as superseded.

---

# Implementation Handoff — Consolidated #10960 Batch and Three CLI-Agent Slices 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-consolidated2-20260926`
- Branch: `claude/ud-consolidated2-20260926`
- Baseline commit: `71610a4ff` (origin/main, after #10973 and #10987)
- Implementation commit: `SELF`
- Pull request: not created at commit time (supersedes #10974, #10980, #10981, #10982)
- Governing issues: #10960, #9415, #10965, #9191
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: land the #10960 fabricated-evidence remediation (P0, P1, P2) plus the
  #9415 root allowlist, #10965 coverage-gate authority and #9191 screenshot verifier
  in one PR, under the fleet `pr-queue-consolidation` rule.
- Status: #10974, #10980, #10981 and #10982 merged in order onto the #10973 tree;
  the P1-1..P1-6, P1-11, P2-a and P2-b slices (agy on DeskComputer and OG Laptop,
  reviewed and corrected by Claude) layered on top; squashed onto `71610a4ff`.
- Integration fixes made here:
  - #10974 added a drift-only `open_loop_replay` to the OpenSim full-swing acceptance
    input, which satisfied #10961's evidence-integrity gate while the marker residuals
    are still scored against the capture itself. `MARKER_RESIDUALS_SELF_SCORED` now
    blocks qualification, and #10974's "fully populated" test expects `is_qualified`
    False.
  - P1-4's `VERIFIED_SEED_SOURCES` and P1-5's `synthetic` source unified with
    `SYNTHETIC_SEED_SOURCES` in `seeds.py`; body candidates reject any unverified seed.
  - P1-11 matches the real native-fit lanes exactly (`native`, `drake_native_fit`,
    `crocoddyl_native_fit`) instead of a substring, so no committed receipt changes
    verdict.
  - `reports/matched_swing_ledger.json` regenerated.

## Validation

- `pytest -n 6 tests/unit/motion_matching tests/unit/neural_motion tests/tools/motion_matching tests/opensim tests/config <changed tests>`:
  3034 passed, 21 failed. All failures also fail on main or are host-only: event_alignment x8,
  Rust bench/parity x3, stability_matrix x1, swing_surrogate_training x4, OpenSim binding x2,
  Rajagopal asset x1, artifact_audit x2 (the 10k corpus exists on this host).
- `tests/opensim/test_moco_g1_ladder.py` and the full-swing tests: 27 passed.
- `tests/unit/motion_matching/test_ledger.py`: 11 passed after regeneration.
- `control_replay.py`: explicit None check on the tighter-step residual (pre-push mypy); control_replay tests pass.
- CI round 2: `check_architecture_budget.py` flagged `load_body_target_json` (118 lines) and
  `_run_club_only_match` (114 lines). Split into `_read_body_target_payload`,
  `_build_source_provenance`, `_verified_club_seed` and `_club_only_request`; budget OK, and the
  body-JSON and GUI tests pass (37 passed, 10 skipped).
- CI round 3: `check_file_size_budget.py` flagged `gui.py` at 1205/1200. Moved the verified-seed
  predicate into `seeds.is_verified_seed` + `NO_VERIFIED_SEED_MESSAGE` (tested in
  `test_ui_integration_seed_10960.py`); `gui.py` is 1200 lines; all 29 local repo-structure steps pass.
- CI round 4: `unit-test-gate` 2 failed / 19109 passed. Both were tests pinning pre-#10960
  behaviour: `test_calibration_provenance` asserted the FB4 MuJoCo receipt `PASSED` (P0-9
  honestly relabelled it `REJECTED`, 138 mm RMS > 60 mm), and the tour-viewer compare test ran
  without the now-mandatory seed (P1-4). Tests updated to the new contract; 11 passed locally.
- CI round 5: `unit-test-gate` 1 failed / 19110 passed. `test_acceptance_gate_honesty_10960`
  patched dotted-string targets, which resolve through `src.shared.python` attributes that another
  test rebinds under xdist. It now patches the imported module objects; 9 passed locally.

## Blockers and Risks

- #10965: the coverage-gate `ratchet_on` re-date must land on or after 2026-10-01
  (see #10981's body); five gates measure below their floor on the unit lane.
- `GateStatus` gained `NOT_APPLICABLE` and `DISCLOSED`; `is_verdict_accepted` needs
  at least one PASSED gate.
- UD#9507: `package-standalone-sidekick.yml` non-cone sparse checkout poisons
  self-hosted work dirs (ControlTower-1 repaired 2026-09-26); fix is held.

## Next Steps

1. Open the PR, get `quality-gate` green, arm through `scripts/automerge_guard.py`.
2. After merge, close #10974, #10980, #10981 and #10982 as superseded.
3. Close #10960 only when its checklist is satisfied by merged code.
