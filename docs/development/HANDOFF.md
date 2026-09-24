# Remove Dead skeleton_extractors Providers — #8866

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `/home/dieterolson/staff-worktrees/UpstreamDrift-run-b571f75d9fb0`
- Branch: `staff/issue-remediator-task-2852a7`
- Baseline commit: `447cfada0`
- Implementation commit: `SELF`
- Pull request: pending (draft, to be opened)
- Governing issue: #8866
- Session: `issue-remediator-run-b571f75d9fb0`

## Objective and Status

Delete the six per-engine skeleton extractor modules under
`src/tools/starting_pose_matcher/skeleton_extractors/` (~1,614 lines) that
have no callers in any shipped `src/` code, along with their eight dedicated
test modules. Prune stale baseline rows referencing the deleted paths.

Previous PR #10041 implemented the same deletion but was approved then closed
as "obsolete" by the owner (2026-09-23) due to merge conflicts with
post-refactor baselines (NM-04, ADR 0051, pose_interchange). This PR
re-targets the deletion against current `main`.

## Files Changed

**Deleted source (6 files, ~1,614 lines):**
- `src/tools/starting_pose_matcher/skeleton_extractors/drake.py`
- `src/tools/starting_pose_matcher/skeleton_extractors/mediapipe.py`
- `src/tools/starting_pose_matcher/skeleton_extractors/mujoco.py`
- `src/tools/starting_pose_matcher/skeleton_extractors/openpose.py`
- `src/tools/starting_pose_matcher/skeleton_extractors/opensim.py`
- `src/tools/starting_pose_matcher/skeleton_extractors/pinocchio.py`

**Deleted tests (8 files):**
- `tests/unit/tools/starting_pose_matcher/test_drake_provider.py`
- `tests/unit/tools/starting_pose_matcher/test_mujoco_provider.py`
- `tests/unit/tools/starting_pose_matcher/test_opensim_provider.py`
- `tests/unit/tools/starting_pose_matcher/test_pinocchio_provider.py`
- `tests/unit/tools/starting_pose_matcher/test_observed_input_providers.py`
- `tests/unit/tools/starting_pose_matcher/test_provider_error_paths.py`
- `tests/tools/starting_pose_matcher/test_observed_extractors.py`
- `tests/tools/starting_pose_matcher/test_physics_extractors_with_stubs.py`

**Updated baselines:**
- `scripts/config/full_src_mypy_baseline.json` — removed 9 entries (611 → 602)
- `scripts/config/suite_marker_baseline.json` — removed 89 node IDs (19584 → 19495)
- `scripts/ci/lod_baseline.txt` — removed 3 rows (428 → 425)

**Updated docs:**
- `docs/development/opensim_tour_matching/EPIC_GOLF_MODEL.md` — updated stale
  reference to deleted `skeleton_extractors/opensim.py` noting removal and directing
  future OpenSim pose extraction to `pose_interchange`

**Note:** `skeleton_extractor.py` (singular, 83 lines) is kept — it provides
the `SkeletonExtractor` ABC and `JsonSkeletonExtractor` still used by
`gui_main_widget.py`.

## Validation

- `ruff check .` — all checks passed (zero violations)
- `ruff format --check .` — no new diffs introduced by this change
- `scripts/ci/check_file_size_budget.py` — OK
- Tests: `vendor/ud-tools` submodule not initialized in this worktree (pre-existing
  environment gap per #9733); CI will run the full suite

## Blockers and Risks

- None: this is a pure deletion of verified dead code with no callers in `src/`.
- The `vendor/ud-tools` submodule is not initialized locally, preventing local
  test runs. This pre-dates this change; CI will validate.

## Next Steps

1. CI green → request review
2. Merge (squash)
3. Close issue #8866 via `Fixes #8866` in PR body (included)
