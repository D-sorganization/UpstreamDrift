# F-UD-035

**Unit-gate quarantine skips security tests (`safe_eval`) and contract tests in the only always-on gate**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | tests    | high       | M      | proposed |

**Locations:** `scripts/config/unit_gate_quarantine.json:8`, `.github/workflows/ci-standard.yml:1240`

## Finding

- Severity: Low - Category: tests
- Confidence: high
- Location: scripts/config/unit_gate_quarantine.json:8-38; .github/workflows/ci-standard.yml:1240
- Evidence: 32 node IDs skipped under `UNIT_GATE_QUARANTINE=1`, including `test_safe_eval_does_not_call_python_eval`,
  `test_integer_blowup_tower_rejected_fast`, `test_failing_invariant_raises_invariant_error`, and the toplevel-package-collision
  hygiene test. "Entries may only be removed, never added" — dated 2026-08-18.
- Failure scenario: the tests that assert `safe_eval` never reaches `eval` are the ones not running; `simpleeval` is a core
  dependency precisely because `eval` was removed.
- Why a newer engineer misses this: quarantine is the right move for flaky tests; these are deterministic failures parked
  for two months.
- Proposed change: fix or delete the 32; the file should be empty.
- Verification: `check_unit_gate_quarantine.py` reports 0 node IDs.
- Effort: M

## Draft Pull Request

**Title:** `test(ci): clear the unit-gate quarantine, safe_eval tests first`

### Why This Matters (The Pitch)

The only always-on gate skips 32 quarantined tests, among them test_safe_eval_does_not_call_python_eval, test_integer_blowup_tower_rejected_fast, test_failing_invariant_raises_invariant_error and the package-collision hygiene test. These are deterministic failures parked since 2026-08-18, not flakes, and simpleeval is a core dependency precisely because eval was removed. Fix or delete the 32 so the quarantine file is empty and the security tests actually run.

### Proposed PR Body

## Problem

`scripts/config/unit_gate_quarantine.json:8-38` lists 32 node IDs skipped under `UNIT_GATE_QUARANTINE=1` (`.github/workflows/ci-standard.yml:1240`); entries may only be removed, never added.

## Change

- [ ] Fix or delete each of the 32 quarantined tests, security tests first
- [ ] Leave `unit_gate_quarantine.json` empty

## Verification

`check_unit_gate_quarantine.py` reports 0 node IDs.

## Risk and Rollback

Medium: fixing may surface real failures. Rollback is re-adding entries.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-035

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
