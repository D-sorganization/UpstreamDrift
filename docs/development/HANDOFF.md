# Current Handoff — Retire the Review-Comment-to-Issue Converter (RM#1755)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/retire-comment-converter`
- Branch: `chore/retire-comment-converter-v2`
- Issue: Repository_Management#1755
- What was removed: `.github/workflows/Comment-to-Issue-Converter.yml`, `scripts/ci/process_review_comments.py`, and the processor tests. `tests/ci/test_process_review_comments.py` now holds a retirement guard (the deleted-test gate has no acknowledgement path), and the `.github/WORKFLOWS.md` row was retired with it.
- Also fixed: `tests/ci/test_ci_infrastructure.py::TestCIEnvironmentCompatibility::test_helper_workflows_use_pr_scoped_concurrency` referenced the deleted workflow file directly; dropped it from that test's workflow list so the test does not FileNotFoundError.
- Superseding note: this re-lands #10941, whose branch conflicted with main after a merge-of-main push failed hooks; content is unchanged, branch is fresh off `origin/main`.
- Validation: `py -3.12 <RM>/scripts/campaigns/review_comment_converter_retirement/retire_converter.py --repo . --check` -> exit 0 after `--apply`; `pytest tests/ci/test_ci_infrastructure.py -k helper_workflows_use_pr_scoped_concurrency` passes.
- Next step: open the draft PR for review.

---

# Implementation Handoff - Drift Wizard Sidekick Product Knowledge Pack (#10943)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-10943`
- Branch: `agy/issue-10943`
- Baseline commit: `2d5830d18642c46d915f7639f0f996f995b149ad` (origin/main)
- Implementation commit: `3c4c3940f`
- Pull request: #10968
- Governing issue: #10943
- Lease session: `antigravity-ud-10943`

## Objective and Status

- Objective: Give UpstreamDrift its Wizard ("Drift Wizard"): a product expert in the Sidekick chat that is always current. Add `knowledge/wizard.yml`, `knowledge/pack.yml`, CI pack rebuild workflow, standalone packaging embedding, and comprehensive unit tests.
- Status: Ready for PR (TDD Green verified)
- Completed:
  1. Verified prerequisites Tools K0 and K3a are pinned in `vendor/ud-tools` (`2d5830d18`).
  2. Created worktree `agy-10943` and claimed issue #10943 under lease `antigravity-ud-10943`.
  3. Authored comprehensive test suite `tests/unit/ai/test_drift_wizard.py` covering manifest loading, source glob resolution, fixture retrieval, context rendering, stale banner, and sidekick glue integration.
  4. Executed pytest and confirmed all 7 tests fail cleanly on missing manifest (TDD Red phase, commit `b6af32afb`).
  5. Authored `knowledge/wizard.yml` (`key: upstream_drift`, `name: Drift Wizard`) and `knowledge/pack.yml` (product documentation & reference source catalog).
  6. Updated `sidekick.spec` to bundle `.knowledge/` and `knowledge/` in standalone PyInstaller builds.
  7. Updated `scripts/packaging/build_sidekick_binary.py` to compile knowledge pack before running PyInstaller.
  8. Authored `.github/workflows/wizard-pack.yml` and registered in `.github/WORKFLOWS.md`.
  9. Added `.knowledge/` to `.gitignore`.
  10. Added Change Log row to `SPEC.md`.
  11. Verified all 7 tests in `tests/unit/ai/test_drift_wizard.py` and 30 tests in child copy and divergence test suites pass cleanly.
  12. Verified `scripts/check_workflow_inventory.py`, `scripts/check_spec_paths.py`, and local-only workflow audits pass.
  13. Reverted child-copy `src/shared/python/ai/knowledge/wizard.py` to match `origin/main` to honor child-copy immutability, baselined LOD finding in `scripts/ci/lod_baseline.txt`, and verified repo-wide `check_lod.py` clean scan.
  14. Switched build script packaging to direct Python `build_pack` API to prevent mock interference in PyInstaller tests.

## Next Steps

1. Monitor CI checks on PR #10968 until auto-merge squashes cleanly into main.
2. Release lease on #10943, close issue, and clean up worktree/branch.
