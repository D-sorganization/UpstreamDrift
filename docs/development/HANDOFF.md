# Implementation Handoff — Remove Dead Skeleton Extractors Providers

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `/home/dieterolson/staff-worktrees/UpstreamDrift-run-b571f75d9fb0`
- Branch: `staff/issue-remediator-task-2852a7`
- Baseline commit: `447cfada0`
- Implementation commit: `SELF`
- Pull request: #10808 (merged)
- Governing issue/epic: #8866
- Session: `issue-remediator-run-b571f75d9fb0`

## Objective and Status

- Objective: Delete the six per-engine skeleton extractor modules under `src/tools/starting_pose_matcher/skeleton_extractors/` (~1,614 lines) that have no callers in any shipped `src/` code, along with their eight dedicated test modules. Prune stale baseline rows referencing the deleted paths.
- Status: Complete (merged to main in PR #10808)
- Completed: Removed 6 uncalled modules and 8 test suites, pruned stale baseline entries in mypy, suite markers, and LoD baselines.
- Remaining: None (shipped)

## Files and Decisions

- Files changed:
  - Deleted source (6 files, ~1,614 lines):
    - `src/tools/starting_pose_matcher/skeleton_extractors/drake.py`
    - `src/tools/starting_pose_matcher/skeleton_extractors/mediapipe.py`
    - `src/tools/starting_pose_matcher/skeleton_extractors/mujoco.py`
    - `src/tools/starting_pose_matcher/skeleton_extractors/openpose.py`
    - `src/tools/starting_pose_matcher/skeleton_extractors/opensim.py`
    - `src/tools/starting_pose_matcher/skeleton_extractors/pinocchio.py`
  - Deleted tests (8 files):
    - `tests/unit/tools/starting_pose_matcher/test_drake_provider.py`
    - `tests/unit/tools/starting_pose_matcher/test_mujoco_provider.py`
    - `tests/unit/tools/starting_pose_matcher/test_opensim_provider.py`
    - `tests/unit/tools/starting_pose_matcher/test_pinocchio_provider.py`
    - `tests/unit/tools/starting_pose_matcher/test_observed_input_providers.py`
    - `tests/unit/tools/starting_pose_matcher/test_provider_error_paths.py`
    - `tests/tools/starting_pose_matcher/test_observed_extractors.py`
    - `tests/tools/starting_pose_matcher/test_physics_extractors_with_stubs.py`
  - Updated baselines:
    - `scripts/config/full_src_mypy_baseline.json`
    - `scripts/config/suite_marker_baseline.json`
    - `scripts/ci/lod_baseline.txt`
  - Updated docs:
    - `docs/development/opensim_tour_matching/EPIC_GOLF_MODEL.md`
- Key decisions: Pure deletion of 6 uncalled skeleton extractor engines in starting_pose_matcher and 8 dedicated tests; preserved singular `skeleton_extractor.py` for `JsonSkeletonExtractor` in GUI.
- User-owned or unrelated worktree changes: None observed

## Validation

- `ruff check .` — all checks passed (zero violations)
- `ruff format --check .` — no new diffs introduced by this change
- `scripts/ci/check_file_size_budget.py` — OK
- CI Standard: passed 100% green on PR #10808 (run 35973035906)

## Blockers and Risks

- Blockers: None
- Risks/assumptions: None (verified dead code with no remaining callers in shipped code)

## Next Steps

1. Maintain pruned baselines and direct future pose extraction to `pose_interchange`.

## Change Log

- `SELF` — Restore canonical handoff schema (Files and Decisions, Change Log, Governing issue/epic) and update PR state to merged PR #10808.
