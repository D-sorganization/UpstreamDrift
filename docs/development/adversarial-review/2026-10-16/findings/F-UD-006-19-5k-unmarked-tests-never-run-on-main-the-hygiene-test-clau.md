# F-UD-006

**~19.5k unmarked tests never run on `main`; the hygiene test CLAUDE.md names as the enforcer fails today**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| 🟧 High  | tests    | high       | M      | proposed |

**Locations:** `.github/workflows/ci-standard.yml:1244`, `.github/workflows/ci-standard.yml:2139`, `scripts/config/suite_marker_baseline.json:1`, `tests/unit/repo_hygiene/test_no_deprecated_imports.py:37`, `CLAUDE.md:274`

## Finding

- Severity: High - Category: tests
- Confidence: high
- Location: .github/workflows/ci-standard.yml:1244-1247; .github/workflows/ci-standard.yml:2139-2144;
  scripts/config/suite_marker_baseline.json:1-3; tests/unit/repo_hygiene/test_no_deprecated_imports.py:37-47; CLAUDE.md:274-281
- Evidence:
  ```yaml
  -m "unit and not slow and not live_simulation ..."     # the only always-on test gate
  dependency_light_targets=( tests/unit/shared_python tests/unit/sidekick tests/unit/robotics ... )   # "full core lane"
  ```
  `suite_marker_baseline.json` lists 19,495 `unmarked_nodeids` (ratchet: may shrink, never grow). `test_no_deprecated_imports.py`
  has no `pytest.mark.unit`; running it: "Found upstream_drift_tools imports in src/: [43 files]" and 6 in tests/.
- Failure scenario: a regression in `tests/api`, `tests/unit/physics`, `tests/motion_matching`, … is only caught if the PR
  touches that test file (PR-scoped lane) — the suite has ~47k test functions and roughly 40 % of them are dark on trunk. The
  repo's documented seam enforcement is not enforcement.
- Why a newer engineer misses this: marker-based lanes look like a performance optimisation; the baseline file that records
  which tests are excluded is 19,500 lines long and nobody opens it.
- Proposed change: run the unmarked set nightly with `-m "not slow and not live_simulation"` and fail; mark
  `test_no_deprecated_imports.py` `unit` after scoping it to UD-owned files (it cannot pass on the Tools copy); invert the
  default so a new test file without a marker fails collection (`--strict-markers` already exists; add a conftest check).
- Verification: `pytest --collect-only -q -m unit | wc -l` vs `pytest --collect-only -q | wc -l` on main; both numbers in the
  CI summary.
- Effort: M

## Draft Pull Request

**Title:** `ci(tests): run 19.5k unmarked tests nightly; mark the hygiene test`

### Why This Matters (The Pitch)

The only always-on test gate on main is -m unit. The suite_marker_baseline.json file records 19,495 unmarked tests that are excluded, which by the review's estimate is roughly 40 % of about 47k test functions. A regression in tests/api, tests/unit/physics or tests/motion_matching is caught only if a PR happens to touch that test file. Worse, CLAUDE.md names test_no_deprecated_imports.py as the enforcer of the Tools seam, yet it has no unit marker, is never collected by the gate, and fails today on 43 src files. So the documented enforcement does not enforce. A nightly run of the unmarked set with the slow and live-simulation exclusions, a scoped and marked hygiene test, and a conftest check that fails any new unmarked test file would turn a 19,500-line exclusion list into a shrinking number. Anyone merging to main benefits from the safety net, and so does whoever reads the failures a month sooner.

### Proposed PR Body

## Problem

`ci-standard.yml:1244-1247` runs only `-m "unit and not slow and not live_simulation ..."` on main; `scripts/config/suite_marker_baseline.json` lists 19,495 `unmarked_nodeids`. `tests/unit/repo_hygiene/test_no_deprecated_imports.py:37-47` has no `unit` marker and, when run, reports `upstream_drift_tools` imports in 43 `src/` files and 6 in tests/ (CLAUDE.md:274-281 names it as the seam enforcer).

## Change

- [ ] Add a nightly job running the unmarked set with `-m "not slow and not live_simulation"` that fails on error
- [ ] Scope `test_no_deprecated_imports.py` to UD-owned files and mark it `unit`
- [ ] Add a conftest check so a new test file without a marker fails collection (`--strict-markers` already exists)

## Verification

`pytest --collect-only -q -m unit | wc -l` vs `pytest --collect-only -q | wc -l` on main; report both numbers in the CI summary.

## Risk and Rollback

Medium: the nightly job will surface failures that have been dark; expect an initial triage burst. Rollback is disabling the job.

## Handoff

Follow-ups: F-UD-035 (quarantine), F-UD-010 (ratchets).

Refs: adversarial review 2026-10-16, F-UD-006

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
