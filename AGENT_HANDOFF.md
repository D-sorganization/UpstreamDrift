## Bioptim Simultaneous State and Parameter Estimation (#9762)

Phase 4 of bioptim OCP migration complete. Branch `feat/9762-bioptim-parameter-ocp` implements
`src.shared.python.optimization.ocp.parameter_ocp` (335 LOC) for simultaneous trajectory tracking
and parameter estimation with quadratic priors, identifiability gating, and IPOPT Hessian tuning.
All unit contracts, isolation guards, and quality gates pass. PR pending.

## Capture Rig GUI: Layout Inversion, Responsive Adaptation & Evidence (#9847, #9848)

Epic #9843 complete. Branch `docs/9848-capture-rig-evidence` documents responsive
compact mode in `docs/motion_capture/capture_rig.md`, records before/after metrics
in `docs/motion_capture/evidence/capture_rig_responsive_evidence.md` (minimum width
reduced from 3276 px to 478 px against the <= 900 px budget; preview width 656 px vs
controls 240 px at 1280x800), and provides offscreen screenshot
`capture_rig_responsive_layout.png`. PR #9876 merged (`d34a41a2e`).

## Reference Overlay: Comparison Workspace, Saved Layers & Reproducible Exports (#9866)

Branch `feat/9866-reference-comparison-workspace` implements the synchronized comparison
workspace, saved layer configurations, and reproducible video/sidecar exports for calibrated
reference motions and capture takes. See canonical `docs/development/HANDOFF.md` and `DL-#9866`.
All focused unit and UI tests pass; quality gates verified.

## Calibrated Scene Registration & Event Synchronization (#9865)

Branch `feat/9865-scene-registration` implements calibrated reference scene registration,
event-anchor and offset synchronization, bounded time warping, gap masking, and distortion-aware
camera projection for ReferenceMotion and 2D expert videos. PR #9871 merged to main (`10caddd21`).

## Coaching Reference Work in Progress (#9862)

Isolated branch `feat/9862-coaching-drawings` builds on editing/library PR #9868
(`1e2469296`). Draft PR #9869 contains five saved drawing tools, gesture/keyboard controls, source-frame
visibility and common preview/still/video exports are implemented and passed
focused qualification. Dependency #9868 must merge before this PR is ready. See `docs/development/HANDOFF.md`. External reference epic
#9863 and fleet adoption are still open. No changes to shared clones or vendor
code; the concurrent panel-navigation correction is preserved.

## Segment Force Colors: Epic #9833 in Progress

Branch `feat/segment-force-colors` lives in `_codex_worktrees/segment-force-colors`.
See `docs/development/segment_force_color_epic.md` for current scope and evidence.
Shared Python/Three.js policies, native pendulum and MuJoCo reaction sources,
renderer adapters, desktop controls and WebSocket/Scene3D wiring are implemented.
Epic #9833 and children #9834–#9837 are published. Remote main `ff0effa5a` was merged
into the branch. Git works by clearing the stale `http.https://github.com/.extraheader`
per invocation and using `gh auth git-credential`; global settings are unchanged.
PR #9840 is open. C3D user segments now accept explicitly bound, clock-checked
loads and expose the shared controls. Native MuJoCo raster verification passed
blue/red output and pixel-exact off restoration (local output/force-colors).
Remote main `403292ca3` is merged and the SPEC conflict is resolved with both rows
preserved. PR CI cycle 1 exposed LoD storage access, a plotting import in headless
contracts, render-function size and a redundant websocket cast; fixes and a
headless regression are included. MuJoCo MeshCat now uses shared leaf-object
bindings; its native command test passes with meshcat 0.3.2 installed only under
ignored output/native-meshcat. C3D broad tests have one unrelated loader error-text
expectation mismatch; the new force test passes. Remaining gates: wider interface adapters, protected CI
and merge. Native MuJoCo tests on this Windows host must import mujoco before
pytest/Qt to avoid a loader-order DLL failure. Do not claim universal rollout.
Combined focused regression: 575 passed; web: 68 passed with TypeScript/ESLint.
Cycle 2 fixes add suite markers and merge main `39d944540` (CI's shallow direct
diff had falsely reported its new notebook test deleted). All earlier CI failures
are fixed locally; current-head checks remain required. Wider hosts remain open.

Updated: 2026-09-08 02:55 PDT
Updated: 2026-09-08 03:10 UTC
Updated: 2026-09-09 00:20 UTC (unit-gate PDF identity pins re-synced to the refreshed canonical PDF)
Updated: 2026-09-08 09:30 UTC (PR backlog catch-up sweep)
Updated: 2026-09-08 23:59 UTC (wave-2 PR triage, session UD2PRs)
Updated: 2026-09-08 (wave-2 issue backlog sweep, session UD2IssuesA)

## Capability Atlas #9850

Isolated branch `feat/9850-capability-atlas`, commit `SELF`, PR not created.
See `docs/development/HANDOFF.md` and `DL-#9850` for current validation and
continuation. Generated references consume existing registries and preserve
GUI epic #9843 and optimization-agent file ownership. Product #9849 and
performance review #9851 are separate workstreams. Fleet communication lives
in Repository_Management PR #1580 and has completed a real peer message exchange.

## Wave-2 PR Triage: 2026-09-08 (Agent `claude`, Session UD2PRs)

Repo-wide npm-audit red: advisory GHSA-2883-xcg3-v3hh (js-yaml high,
published 2026-09-08 between the 21:57 main push run and the 22:15 PR runs)
fails `code-quality` (`npm audit --audit-level=high`) on every merge ref
whose lockfile carries js-yaml 4.3.1 - main itself goes red on its next
Standard run. Fix on `bot/claude/npm-audit-jsyaml`: npm `overrides.js-yaml`
= `^4.3.2` in `ui/package.json` (dev-only dep of `@eslint/eslintrc`), audit
drops to 5 moderate, gate passes. Disposition table below is maintained as
PRs settle (in-progress at first commit).

Wave-2 disposition table (REST-verified 2026-09-09 ~00:45 UTC):

| PR                         | Disposition at yield                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #9465                      | **MERGED** (squash `9623a5662`).                                                                                                                                                                                                                                                                                                                                                                                                                     |
| #9827                      | **MERGED** (squash `403292ca3`).                                                                                                                                                                                                                                                                                                                                                                                                                     |
| #9826                      | PDF identity pins in `tests/research/test_publication_quality.py` re-synced to the refreshed artifact (`bf855f79`, 2012367 bytes) on `fix/9825-preserve-reviewed-claims` (commit `30aecd113`); auto-merge armed.                                                                                                                                                                                                                                     |
| #9763-#9767                | Five dependabot /ui bumps; unblocked by this PR's audit fix; auto-merge armed, draining.                                                                                                                                                                                                                                                                                                                                                             |
| #9471                      | Merged main in twice (SPEC row conflicts: dropped the branch-side duplicate #9476 row, kept this PR's #9471 row); repo-structure-gates green; auto-merge armed.                                                                                                                                                                                                                                                                                      |
| #9434                      | Merged main in (WORKFLOW_TRACKING.md conflict: companion entry kept, main's always-on-unit-lane entry kept); auto-merge armed.                                                                                                                                                                                                                                                                                                                       |
| #9440                      | Merged main in: theme modify/delete resolved as PR deletions; shadow ledger 33 -> 17 (stale + this-PR entries dropped); seam rulings = main's retirement narratives + PR cleaned rows for notes/plot_theme/theme; divergence inventory regenerated; SeamRedirectFinder roots synced to merged cleaned rulings (commit `4d3193f81`). Auto-merge armed; the wave-1 #8972/#9037 Tools-palette prerequisite still applies if palette-consumer tests red. |
| #9442                      | Stacked on #9440's branch (base `readiness/p1-9406-delete-tools-canonical-1`, not main); branch merged forward to `b45ec951f` with the seam-root sync; fresh CI running; auto-merge not armable while stacked - retarget to main only after #9440 lands (wave-1 rule).                                                                                                                                                                               |
| #9636, #9633, #9618, #9610 | Conductor drafts, skipped per assignment.                                                                                                                                                                                                                                                                                                                                                                                                            |

## Wave-2 Issue Backlog: 2026-09-08 (Agent `claude`, Session UD2IssuesA)

Second-wave residual-backlog sweep over the older half of the 186 open issues
(#8346–#8930). Sibling PRs from the same sweep (based on the identical main
revision): PR #9828 (#8922, mocap retargeting IK cost) and PR #9831 (#8928,
pendulum result accessor caching).

- **#8842** — `notebooks/bunkershot3d/phase1_mvp.py` was unrunnable: it
  imported a nonexistent top-level `bunkershot3d` package and pointed at a
  nonexistent repo-root `configs/` tree. It now imports via
  `bunkershot3d.*` with a repo-root `sys.path` bootstrap, resolves the
  packaged `src/bunkershot3d/calibration/configs/canonical.yaml`, writes
  artifacts under gitignored `output/bunkershot3d/`, and exits 1 with a
  clear log line when the optional `pychrono` backend is absent (phases 1-2
  still produce their artifacts). Smoke tests:
  `tests/bunkershot3d/test_phase1_mvp_notebook.py`.
- **#8922** (PR #9828) — `MotionRetargeting._solve_frame_ik` called
  `mj_forward` once **per marker per IK iteration** (~40x overcount with the
  golf marker set) and grew the stacked Jacobian with a fresh `np.vstack`
  per marker. The marker→(body, target) pairs are now resolved once per
  frame, `mj_forward` runs exactly once per iteration, and the
  Jacobian/error rows are batched with a single `np.vstack` /
  `np.concatenate`. Regression tests pin the forward-call bound and
  multi-marker error reduction
  (`tests/unit/engines/mujoco/test_motion_capture.py::TestMotionRetargetingIKCost`).
- **#8928** — pendulum simulation result accessors re-integrated the
  trajectory on every call (`all_energies` once per energy key = 3 passes;
  `extract_series` fresh per plot per joint). `TrajectoryResultMixin`
  now computes every `all_*` batch in a single pass and memoizes it on
  the result object, and `energy_at` derives `total` arithmetically
  (E = T + V) instead of re-evaluating both terms. Benchmark (400-step
  golfer result): repeated `all_accelerations` 0.64 s → ~6 µs; repeated
  `all_energies` 0.15 s → ~6 µs.
