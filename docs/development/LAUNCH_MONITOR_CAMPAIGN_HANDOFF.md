# Launch Monitor Campaign Handoff

Last verified: 2026-08-09

The first guarded push after the normal `main` merge was correctly blocked by
the repository's Prettier hook. It normalized only indentation and compact
array layout in the newly inherited `e1_sweep.json` and
`results_summary.json`; parsed before/after values are exactly equal. This is
a formatter-only follow-up with no material launch-monitor handoff or contract
change. The hook must be rerun, not bypassed.

## Active UpstreamDrift Work

- Repository: `D-sorganization/UpstreamDrift`
- Exact-head worktree:
  `C:\Users\diete\Repositories\UpstreamDrift-worktrees\launch-monitor-handoff-publish`
- Branch: `feat/launch-monitor-showtime-8364`
- Draft PR: [#8369](https://github.com/D-sorganization/UpstreamDrift/pull/8369)
- Published PR head: `85322ff8a836705a90c51335131ab017c50f3374`
- Recorded merge base: `main@2c37d663b05c49a0c72647ce2e254214953ce2d2`
- Exact live `main` incorporated locally:
  `1bb634b6e45dea4ebc360d6db973236549009e40`
- Local merge parents, in order:
  `85322ff8a836705a90c51335131ab017c50f3374` and
  `1bb634b6e45dea4ebc360d6db973236549009e40`
- Feature commit: `feat(launch-monitor): add flexible traceable analytics (#8364)`

This branch provides the UpstreamDrift Launch Monitor Analytics surface. Keep
its public tab/route and adapter boundaries stable while the reusable contracts
in Tools are reviewed. Do not copy calculator logic between repositories.

The similarly named `launch-monitor-showtime` worktree is stale and divergent
at local head `aee026373`; do not run release validation or make continuation
commits there. PR #8369 has no submitted reviews or unresolved review threads.

The pre-merge release audit found the published branch nine commits ahead and
six commits behind live `main`, with `SPEC.md` as the only content conflict.
The local normal merge preserves the launch-monitor contract work and the
research, security, launcher, and optimization history from `main`.
`SPEC.md` retains both update streams and advances from the feature-side
`1.0.485` and main-side `1.0.484` records to monotonic version `1.0.486`.

## Shared Tools Dependency

The reusable Rate of Closure and launch-monitor work is being integrated in:

- Repository: `D-sorganization/Tools`
- Current carrier worktree:
  `C:\Users\diete\Repositories\Tools-worktrees\toolstrip-workspace`
- Current carrier branch: `feat/4199-wind-workflow`
- Current carrier PR: [Tools #4282](https://github.com/D-sorganization/Tools/pull/4282)
- Shared analytics PR: [Tools #4212](https://github.com/D-sorganization/Tools/pull/4212)
- Shared convention registry PR:
  [Tools #4203](https://github.com/D-sorganization/Tools/pull/4203)
- Ball-flight release gate:
  [Tools #4201](https://github.com/D-sorganization/Tools/issues/4201)
- Combined Tools integration PR:
  [Tools #4217](https://github.com/D-sorganization/Tools/pull/4217)

The exact published Tools carrier head is
`de49580a3c0888b44f66dcc09bba2ab2fa33914a`. Its quality gate passed on
2026-08-09; the rest of the protected suite was still queued when observed.
The checked-in Tools campaign authority explicitly records that no protected
`main` release or immutable UpstreamDrift dependency pin exists. UpstreamDrift's
current `vendor/ud-tools` gitlink is `ff4240217005e1415ca409fd124e50b64ee642d2`,
not the draft carrier head.

The local UpstreamDrift continuation pins a parity fixture to the existing
Tools launch-monitor statistics v1 record fingerprint. The fingerprint excludes
the transient pandas index and hashes ordered record content plus explicit shot,
session, source-row, and monitor identity fields. This is a compatibility
fixture, not an immutable Tools release pin. Rate physics, flight, variation,
wedge, wind, ground, and playback behavior remain Tools-owned.

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
6. Keep unsupported analysis modes, correlation methods, and missing-data
   policies outside the domain by validating them at the API schema boundary.
7. Catch user-correctable selection errors at the Qt signal boundary and show
   accessible inline status; direct domain calls continue to fail closed.

## Required Verification Before Merge

- Reconcile the UpstreamDrift analytics implementation with the final Tools
  public facade and remove any duplicated calculation logic.
- Re-run the cross-contract fingerprint fixture after any Tools v1 contract
  change; do not silently change the fingerprint algorithm under version 1.0.0.
- Run the focused UpstreamDrift analytics, navigation, lazy-import, manifest,
  feature-parity, and adapter tests.
- Run Ruff, Ruff format, mypy, file/module-size gates, and the applicable web
  build/tests on the exact PR head.
- Verify the standalone PyQt widget and the embedded UpstreamDrift tab manually.
- Confirm the Tools dependency pin and installed-package smoke test use the exact
  reviewed Tools commit.
- Observe required protected checks and reviews; do not merge or close the
  release gate based only on local test results.

## Local Continuation Evidence

- **92 passed** across unit launch-monitor, FastAPI route, PyQt/embed, and
  feature-parity tests after merging exact live `main`.
- **39 passed** across the merged proximal-distal research and launcher API
  tests.
- Ruff and Ruff format pass over all 28 Python files in the combined
  feature/main delta. A repository-wide Ruff probe reports 85 unrelated
  baseline findings outside that delta.
- Python 3.11 with mypy 2.1 passes on all three changed production modules
  using the PR-delta `--ignore-missing-imports --follow-imports=skip` boundary.
- Error-handling ratchet, docs governance and its 10 tests, doc catalog, and
  doc-size checks pass.
- Repository-wide file-size scanning is blocked only by the pre-existing
  expired exception for `src/shared/python/chat/_chat_dock_widget_qt.py`
  (1,490 lines); no changed file approaches the 1,200-line limit.

## Current Blockers

- The Tools Rate campaign is still a draft feature stack, not a production
  release. Quality-gate success on `de49580a3` is positive, but queued checks
  are not passing evidence.
- UpstreamDrift PR #8369's previous-head full CI run `31137536924` failed across lanes
  because the self-hosted runners could not reliably fetch the repository,
  PyPI packages, or `postgres:16`. These are infrastructure failures; no code
  fix or repeated rerun storm is justified from those logs.
- The exact published head `85322ff8a8` had zero check runs, commit statuses,
  or Actions workflow runs while the PR was conflicting with live `main`; the
  required `quality-gate` context therefore remains absent. This local merge
  resolves the sole content conflict, but protected checks remain required
  after an authorized normal push.
- React/Vite remains an honest parity gap under issue #8364. The versioned API
  is not itself a native React surface.
- Rust/WASM trajectory parity, installed-package verification, worker/thread
  responsiveness, full persistence/export wiring, and independent scientific
  validation remain open in Tools issue #4201.
- Do not claim UpstreamDrift parity with the shared campaign until those gates
  and the dependency pin are verified on the merged commit.
