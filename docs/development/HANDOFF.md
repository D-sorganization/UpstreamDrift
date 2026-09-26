# Implementation Handoff - Drift Wizard Sidekick Product Knowledge Pack (#10943)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/agy-10943`
- Branch: `agy/issue-10943`
- Baseline commit: `2d5830d18642c46d915f7639f0f996f995b149ad` (origin/main)
- Implementation commit: `SELF`
- Pull request: not created
- Governing issue: #10943
- Lease session: `antigravity-ud-10943`

## Objective and Status

- Objective: Give UpstreamDrift its Wizard ("Drift Wizard"): a product expert in the Sidekick chat that is always current. Add `knowledge/wizard.yml`, `knowledge/pack.yml`, CI pack rebuild workflow, standalone packaging embedding, and comprehensive unit tests.
- Status: In progress (TDD Red commit authored)
- Completed:
  1. Verified prerequisites Tools K0 and K3a are pinned in `vendor/ud-tools` (`2d5830d18`).
  2. Created worktree `agy-10943` and claimed issue #10943 under lease `antigravity-ud-10943`.
  3. Authored comprehensive test suite `tests/unit/ai/test_drift_wizard.py` covering manifest loading, source glob resolution, fixture retrieval, context rendering, stale banner, and sidekick glue integration.
  4. Executed pytest and confirmed all 7 tests fail cleanly on missing manifest (TDD Red phase).

## Next Steps

1. Commit failing test suite (`test(ai): add failing test suite for K3b Drift Wizard (#10943)`).
2. Author `knowledge/wizard.yml` and `knowledge/pack.yml`.
3. Author `.github/workflows/wizard-pack.yml` and register in `.github/WORKFLOWS.md`.
4. Update `sidekick.spec` to embed knowledge pack when present.
5. Add `.knowledge/` to `.gitignore`.
6. Add Change Log row to `SPEC.md`.
7. Verify all tests pass (TDD Green).
