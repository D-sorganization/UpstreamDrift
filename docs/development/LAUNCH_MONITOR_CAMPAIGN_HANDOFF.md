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

The Tools analytics hardening head was `4b22e79cf` when this handoff was
written. It preserves the public facades while keeping every new production
module at or below 361 lines. The convention registry hardening head was
`3d899c8e9`; it compares sign rules explicitly and represents the unsupported
general Foresight Launch Direction sign as `unspecified`.

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
- Rust/WASM trajectory parity, installed-package verification, worker/thread
  responsiveness, full persistence/export wiring, and independent scientific
  validation remain open in Tools issue #4201.
- Do not claim UpstreamDrift parity with the shared campaign until those gates
  and the dependency pin are verified on the merged commit.
