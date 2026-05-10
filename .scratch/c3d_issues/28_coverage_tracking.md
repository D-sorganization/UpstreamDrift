# [Tracking] Pristine TDD coverage across the C3D / motion-matching surface

## Why

Several rapid waves landed major C3D + motion-matching features in a short window. Each wave shipped focused tests, but the cumulative production surface has grown faster than the cumulative coverage. The user has asked for **pristine** coverage end-to-end.

## Production vs test surface (today)

| Area | .py files (excl __init__) | test files |
|---|---|---|
| `src/shared/python/motion_matching/` | 74 | 61 |
| `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/` (C3D viewer) | 17 | 8 |
| `src/tools/starting_pose_matcher/` (matcher tool) | 14 | 7 |
| `src/shared/python/upstream_drift_tools/lab/bio/` (C3D reader) | 5 | (mixed in 37) |
| `src/launchers/` | 38 | 36 |

## Plan

Each child issue covers one area. The agent for each:

1. Runs `coverage.py` (`python3 -m pytest <area-tests> --cov=<area-pkg> --cov-report=term-missing -p no:cacheprovider`) to measure baseline.
2. Identifies uncovered lines / branches / files.
3. Adds focused tests until line coverage ≥85% for the area, branch coverage ≥75%.
4. New tests are pure unit-level where possible; integration tests when behaviour can only be observed end-to-end.
5. All new tests pass on `main`; no production code changes (this campaign is pure test-coverage).

## Children

- #__a__ — `motion_matching` coverage to ≥85%
- #__b__ — C3D Viewer (`apps/` engine subtree) coverage to ≥85%
- #__c__ — Starting-pose matcher tool (`src/tools/starting_pose_matcher/`) coverage to ≥85%
- #__d__ — `upstream_drift_tools.lab.bio` (C3D reader internals) coverage to ≥85%
- #__e__ — `launchers/` coverage to ≥85% (excluding GUI-only smoke that needs `pytest-qt`)

## Acceptance

- [ ] Each child issue closed with a coverage delta in its PR description.
- [ ] No production code changes in any child PR (test-only).
- [ ] All new tests pass on `main`.
- [ ] CI green per PR.
- [ ] Generic naming, mypy, ruff, file-size budget all clean.
