# F-UD-013

**`tests/cross_engine/test_cross_engine_equivalence.py` claims to be the CI gate and contains no gate test**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | tests    | high       | S      | proposed |

**Locations:** `tests/cross_engine/test_cross_engine_equivalence.py:1`, `tests/cross_engine/test_cross_engine_equivalence.py:270`, `tests/cross_engine/test_cross_engine_equivalence.py:464`, `.github/workflows/cross-engine-equivalence.yml:261`

## Finding

- Severity: Medium - Category: tests
- Confidence: high
- Location: tests/cross_engine/test_cross_engine_equivalence.py:1-10; tests/cross_engine/test_cross_engine_equivalence.py:270-273;
  tests/cross_engine/test_cross_engine_equivalence.py:464-549; .github/workflows/cross-engine-equivalence.yml:261-267
- Evidence: docstring "This module is the CI gate that enforces the §2.2 acceptance criterion … 5 mm grip-position RMSE";
  `_run_engine` (line 270) and the report builder are never called; the only tests are `test_theta_truth_is_deterministic`,
  `test_quat_angle_zero`, … The workflow runs `tests/motion_matching/test_cross_engine_equivalence.py` instead.
- Failure scenario: 555 lines of dead harness that an agent will "fix" by wiring back up next to the real one, giving two
  5 mm gates with different reference selection.
- Why a newer engineer misses this: the module reads like a complete test; you have to notice that no function starting with
  `test_` calls the harness.
- Proposed change: delete the file (keep the 5 helper smoke tests in `tests/motion_matching/` if wanted).
- Verification: `pytest tests/cross_engine -q --collect-only` no longer lists it; the workflow is unchanged.
- Effort: S

## Draft Pull Request

**Title:** `test(cross-engine): delete the dead tests/cross_engine harness`

### Why This Matters (The Pitch)

tests/cross*engine/test_cross_engine_equivalence.py declares in its docstring that it is the CI gate enforcing the section 2.2 acceptance criterion of 5 mm grip-position RMSE. It is not. The harness functions, including \_run_engine at line 270 and the report builder, are never called by any test* function; the only tests are things like test_theta_truth_is_deterministic and test_quat_angle_zero. The workflow runs tests/motion_matching/test_cross_engine_equivalence.py instead. That leaves 555 lines of dead code that a future agent will helpfully wire back up next to the real gate, producing two 5 mm gates with different reference selection. Deleting the file, and keeping the five helper smoke tests under tests/motion_matching if wanted, means less to clone, lint and read, and one unambiguous gate. Size S and the workflow is untouched.

### Proposed PR Body

## Problem

`tests/cross_engine/test_cross_engine_equivalence.py:1-10` claims to be the CI gate for the 5 mm grip RMSE criterion, but `_run_engine` (`:270-273`) and the report builder (`:464-549`) are never called by any `test_` function. `.github/workflows/cross-engine-equivalence.yml:261-267` runs `tests/motion_matching/test_cross_engine_equivalence.py` instead.

## Change

- [ ] Delete `tests/cross_engine/test_cross_engine_equivalence.py`
- [ ] Optionally keep its 5 helper smoke tests in `tests/motion_matching/`

## Verification

`pytest tests/cross_engine -q --collect-only` no longer lists the file; the workflow is unchanged.

## Risk and Rollback

Low: removes code no gate executes. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-013

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
