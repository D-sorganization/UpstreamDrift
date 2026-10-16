# F-UD-040

**The governance apparatus is a second product: 68 check scripts (14.4k LOC) + ~16k LOC of tests of the checks + 57 workflows (13.6k lines)**

| Severity  | Category   | Confidence | Effort | Status   |
| --------- | ---------- | ---------- | ------ | -------- |
| 🟨 Medium | governance | high       | L      | proposed |

**Locations:** `scripts/ci/check_file_size_budget.py:1`, `scripts/check_version_consistency.py:1`, `scripts/ci/check_version_consistency.py:1`, `.github/workflows/ci-standard.yml:1023`, `scripts/config/suite_marker_baseline.json:1`

## Finding

- Severity: Medium - Category: governance
- Confidence: high
- Location: scripts/ci/check_file_size_budget.py:1; scripts/check_version_consistency.py:1; scripts/ci/check_version_consistency.py:1;
  .github/workflows/ci-standard.yml:1023-1026; scripts/config/suite_marker_baseline.json:1
- Evidence: `scripts/` is 158k LOC total; two different `check_version_consistency.py` (548 and 105 lines) both run in CI
  (ci-standard.yml:1023 and :1026); `suite_marker_baseline.json` is 19,500 lines; `ci-standard.yml` is 3,833 lines with ~40
  gate steps; the LoD gate was renamed because two jobs reported the same required context. Of the gates audited in §2, five
  do not measure what they claim (F-UD-004/005/006/010/014) and the DRY/LoD gates grandfather 468 and 5,604 occurrences.
- Failure scenario: the apparatus consumes agent attention (every CLAUDE.md rule is a prompt token) and owner attention (every
  baseline needs a human to tighten it) while the core correctness bugs in this review (F-UD-001/002/003/008) passed every
  gate. That is the definition of a bureaucracy agents satisfy mechanically: it checks the shape of compliance.
- Why a newer engineer misses this: each check was added after a real incident (the comments say so); the sum was never
  reviewed as a system.
- Proposed change: keep ruff/format/pytest/mypy/bandit/size budget/version consistency (one copy) and the ledgers' schema
  tests; retire or merge the rest into a single `scripts/ci/gates.py` with a table of (name, command, baseline, must-shrink);
  budget the apparatus (e.g. ≤ 20 scripts, ≤ 2k LOC) the way files are budgeted.
- Verification: `ls scripts/check_*.py scripts/ci/check_*.py | wc -l` ≤ 20; CI wall time per PR before/after.
- Effort: L

## Draft Pull Request

**Title:** `chore(ci): budget governance scripts and merge duplicate gates`

### Why This Matters (The Pitch)

The governance apparatus has become a second product: 68 check scripts (14.4k LOC), about 16k LOC of tests of the checks, and 57 workflows totalling 13.6k lines. Two different check_version_consistency.py files, 548 and 105 lines, both run in CI. The review found five audited gates that do not measure what they claim, while the core correctness bugs (F-UD-001, 002, 003 and 008) passed every gate. Every CLAUDE.md rule costs agents prompt tokens and every baseline costs the owner attention. Keep ruff, format, pytest, mypy, bandit, the size budget, one version-consistency check and the schema tests; fold the rest into a single gates.py table of name, command, baseline and must-shrink, and budget the apparatus the way files are budgeted. Faster PRs, a smaller CLAUDE.md, and fewer green lights that mean nothing.

### Proposed PR Body

## Problem

`scripts/` is 158k LOC; two `check_version_consistency.py` (`scripts/check_version_consistency.py:1`, 548 lines, and `scripts/ci/check_version_consistency.py:1`, 105 lines) both run at `ci-standard.yml:1023-1026`; `ci-standard.yml` is 3,833 lines with ~40 gate steps and `suite_marker_baseline.json` is 19,500 lines. Five audited gates (F-UD-004/005/006/010/014) do not measure what they claim.

## Change

- [ ] Keep ruff/format/pytest/mypy/bandit/size budget/one version-consistency check and the ledgers' schema tests
- [ ] Retire or merge the rest into a single `scripts/ci/gates.py` with a table of (name, command, baseline, must-shrink)
- [ ] Budget the apparatus (e.g. ≤ 20 scripts, ≤ 2k LOC)

## Verification

`ls scripts/check_*.py scripts/ci/check_*.py | wc -l` ≤ 20; CI wall time per PR before/after.

## Risk and Rollback

Medium: removing a gate removes a guard; do it after the real defects are fixed and measure PR time. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** Follow-ups: F-UD-004, F-UD-005, F-UD-006, F-UD-010, F-UD-014.

Refs: adversarial review 2026-10-16, F-UD-040

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
