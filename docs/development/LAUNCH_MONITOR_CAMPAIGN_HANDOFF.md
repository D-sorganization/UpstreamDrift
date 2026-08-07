# Launch Monitor Campaign Handoff

Last verified: 2026-08-06

## Active UpstreamDrift Work

- Repository: `D-sorganization/UpstreamDrift`
- Worktree: `C:\Users\diete\Repositories\UpstreamDrift-worktrees\launch-monitor-showtime`
- Branch: `feat/launch-monitor-showtime-8364`
- Draft PR: [#8369](https://github.com/D-sorganization/UpstreamDrift/pull/8369)
- Verified local head when this handoff was written: `a6702612a`
- Feature commit: `feat(launch-monitor): add flexible traceable analytics (#8364)`

This branch provides the UpstreamDrift Launch Monitor Analytics surface. Keep
its public tab/route and adapter boundaries stable while the reusable contracts
in Tools are reviewed. Do not copy calculator logic between repositories.

## Shared Tools Dependency

The reusable Rate of Closure and launch-monitor work is being integrated in:

- Repository: `D-sorganization/Tools`
- Integration worktree:
  `C:\Users\diete\Repositories\Tools-worktrees\ballflight-campaign-integration`
- Integration branch: `codex/ballflight-campaign-integration`
- Shared analytics PR: [Tools #4212](https://github.com/D-sorganization/Tools/pull/4212)
- Shared convention registry PR:
  [Tools #4203](https://github.com/D-sorganization/Tools/pull/4203)
- Ball-flight release gate:
  [Tools #4201](https://github.com/D-sorganization/Tools/issues/4201)
- Combined Tools integration PR:
  [Tools #4217](https://github.com/D-sorganization/Tools/pull/4217)

The current combined Tools head is
`6578d3fcbe18c68e61002eb6127f73561242e78e`. The Tools analytics hardening
head remains `4b22e79cf`; it preserves the public facades while keeping every
new production module at or below 361 lines. The convention registry hardening
head remains `3d899c8e9`; it compares sign rules explicitly and represents the
unsupported general Foresight Launch Direction sign as `unspecified`.

The combined Tools continuation has also verified the canonical variation
workspace: `120` focused Python tests and `21` focused React tests passed, and a
live 24-trial pendulum study rendered all `36,024` swing vertices with linked
scatter/matrix/arc selection, positional RMS/quiet zones, and typed impact/
landing cohorts. The wedge worked example now pins the 30 mph, -10-degree AoA
decomposition and explicitly identifies the 1,307 deg/s rate as driver-derived,
not wedge-typical. These capabilities remain Tools-owned; UpstreamDrift should
consume their contracts through adapters rather than reproduce their physics.

## Integration Rules

1. Treat Tools as the canonical source for reusable statistics, convention,
   D-plane, wind, target, solver, and playback contracts.
2. Keep UpstreamDrift integration behind its adapter/embedding boundary; do not
   import a Tools GUI widget directly into an unrelated launcher surface.
3. Update the pinned Tools dependency only after the exact Tools integration
   commit passes its combined tests and protected checks.
4. Verify PyQt import without optional GUI dependencies at package-import time.
5. Preserve explicit modeled/derived/measured-comparable status and provenance
   in UI rows and exports.

## Required Verification Before Merge

- Reconcile the UpstreamDrift analytics implementation with the final Tools
  public facade and remove any duplicated calculation logic.
- Run the focused UpstreamDrift analytics, navigation, lazy-import, manifest,
  feature-parity, and adapter tests.
- Run Ruff, Ruff format, mypy, file/module-size gates, and the applicable web
  build/tests on the exact PR head.
- Verify the standalone PyQt widget and the embedded UpstreamDrift tab manually.
- Confirm the Tools dependency pin and installed-package smoke test use the exact
  reviewed Tools commit.
- Observe required protected checks and reviews; do not merge or close the
  release gate based only on local test results.

## Current Blockers

- The Tools ball-flight campaign is still a draft integration stack, not a
  production release.
- New protected checks are running on Tools head `6578d3fcb`; queued work is
  not passing evidence. The prior PR quality-gate failure was traced to two
  Ruff 0.14.10 formatting differences and corrected at `282b1a4d3`.
- Rust/WASM trajectory parity, installed-package verification, worker/thread
  responsiveness, full persistence/export wiring, and independent scientific
  validation remain open in Tools issue #4201.
- Do not claim UpstreamDrift parity with the shared campaign until those gates
  and the dependency pin are verified on the merged commit.
