---
issue: 14948
summary: "Wire engine lane guards into nightly, enforce MuJoCo in CI, and remove dead MATLAB job"
---

## Handoff

- Slices 4, 6 & 7 of #14948 (Closes #14948, Part of #14934):
  - Removed dead `(false)` `matlab-tests` job from `ci-standard.yml` and documented `"not_covered"` status in `engine_test_floor.json`.
  - Added `UD_REQUIRE_ENGINES: "drake,pinocchio,opensim"` and JUnit floor verification to `nightly-cross-engine.yml`.
  - Enforced `UD_REQUIRE_ENGINES: "mujoco"` on `core-tests` in `ci-standard.yml`.
  - Added contract tests in `tests/unit/ci/test_engine_lane_guard_14948.py`.
