# Implementation Handoff — Consolidated UpstreamDrift PR Queue 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-consolidated-20260926`
- Branch: `claude/ud-consolidated-20260926`
- Baseline commit: `fee5b214e` (origin/main)
- Implementation commit: `SELF`
- Pull request: #10973 (draft)
- Governing issues: #9619, #9549, #9688, #9613, #10286, #9411, #10363, #10602, #9703
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: fold the nine open session PRs (#10948, #10947, #10964, #10966, #10967,
  #10969, #10961, #10972, #10949) into one branch under the fleet
  `pr-queue-consolidation` rule, so the strict-protection queue needs one CI cycle.
- Status: merged in dependency order with `--no-ff`; bookkeeping conflicts in
  `SPEC.md`, `AGENT_HANDOFF.md`, `HANDOFF.md` and `DEVELOPMENT_LOG.md` resolved
  keep-both (rows keyed by issue). The #9703 whole-file HANDOFF rewrite was
  resolved take-ours plus its two change-log lines; its narrative lives in DL-#9703.
- 2026-09-26 re-base: rebuilt as one commit on `fee5b214e` (main merged, SPEC keep-both,
  DL union by entry id), plus #9411 quarantine retirement (was `d099b2abf`) and CI
  repairs: ADR README log for ADR-0041/0044, regenerated engine matrix, monolith
  register and divergence inventory, LOD fix in `pre_impact_bundle.py`, and the rig
  CLI split (`rig/_cli_parsers.py`) under the 1200-line budget.
- CI round 2 (`b17ce4a66`): `_measured_package` was 102 lines, so its fixed orientation
  claims and phase coverage became module constants; #10961's evidence-integrity gate moved
  from `acceptance.py` to `motion_matching/evidence_integrity.py` (+5 lines net on main
  instead of +73), and the module-size reason now records 1427 lines. Motion-matching and
  shared-contract tests: 16 failures, all identical on main.
- CI round 3 (`23f7b906c`): the only unit failure was the divergence inventory missing
  the new `evidence_integrity.py`; regenerated with `python -m
scripts.shared_tools.divergence_inventory --write` (`--check` passes).
- Next: CI green on the consolidated PR, arm via `scripts/automerge_guard.py`,
  then close the nine originals as superseded.

## Files and Decisions

- Per-PR file lists and decisions stay in each DL entry: DL-#9619, DL-#9549,
  DL-#9688, DL-#9613, DL-#10286, DL-#9411, DL-#10363, DL-#10602, DL-#9703.
- Decision: bookkeeping conflicts are keep-both (rows keyed by issue); a
  whole-file rewrite of this handoff is take-ours plus the branch's own lines.
- Merge repair: entries whose trailing `Next step` line was shared across a
  keep-both boundary (DL-#9619, #9549, #9688, #9613) had it restored.

## Validation

- `pytest -n 4` over every test file the nine PRs touch plus
  `tests/unit/motion_matching/`, `tests/shared_contracts/`, `tests/motion_capture/rig/`
  — 13 failures, all 13 identical on `origin/main` `2d5830d18` (event_alignment x8,
  Rust parity/bench x3, stability_matrix, tools_session_export); no new failure.
- Heading counts: DEVELOPMENT_LOG 217 -> 224 (7 new DL entries), HANDOFF 7 -> 7,
  SPEC 282 -> 282; no conflict markers in the tree.

## Blockers and Risks

- `main` is red on pre-existing checks (articulated-manufactured, equivalence,
  docker-smoke, Trivy, jaxsim-upgrade-guard, Canonical Core Conformance, Rust+TS).
- #10967's LoD failure cited `ai/knowledge/wizard.py`, which is not on main
  (it is in #10968): a polluted self-hosted workspace, not this diff.

## Next Steps

1. CI green on the consolidated PR, then arm with `scripts/automerge_guard.py --arm --strategy squash`.
2. After merge, close #10948 #10947 #10964 #10966 #10967 #10969 #10961 #10972 #10949 as superseded.
3. Release the leases on the nine governing issues.

## Change Log

- `SELF` — #9411 Q-2..Q-5: retire 43 unit-gate quarantine IDs by repairing mechanical test drift (DL-#9411).
- `SELF` — Consolidate nine session PRs into one branch under `pr-queue-consolidation` (keep-both bookkeeping, restored DL Next-step lines).
- `SELF` — Declare `FloatArray` as an explicit `TypeAlias` so the pre-push mypy hook (no numpy in its env) accepts it (#9703).
- `6c80d5665` — Versioned PreImpactBundle v1 with fail-closed contracts (#9703).
- `SELF` — #10286: ZTCF/ZVCF fail closed (StateError/ValueError) in MyoSuite, pendulum and DTACK; DTACK canonical ZVCF split from control-kept variant (DL-#10286).
- `SELF` — #10286: add unit suite markers to new tests and SPEC change-log row (#10967 CI).
- `SELF` — #10602 C1/C2: club-only matrix scores only complete recorded fit outcomes; no synthetic package or defaulted metrics (DL-#10602).
- `SELF` — Reconcile child-copy convergence, divergence inventory, and agent context views (#10944).
- `SELF` — #9411: unit suite markers on new budget tests and SPEC change-log row (#10969 CI).
- `SELF` — #9411 M-2: mypy exclusion budget fits the 2026-10-01 cap (35/36), tracked-file rule, remaining exclusions re-attested to 2027-01-01 (DL-#9411).
- `SELF` — Retract the fabricated Drake G1 PASS and add fail-closed evidence-integrity gates (#10363, DL-#10363; follow-ups #10960).
- `SELF` — Migrate ActuatorPanel and SimulationToolbar to shared usePolling hook (#8941).
