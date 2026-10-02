# Active Necromatcher Native Fit Delivery — #11240

Current branch `feat/necromatcher-native-fit-11235` (PR #11240) retargeted to `main` following merge of workspace #11239. It adds source-bound native trajectory fitting with preserved Hermite splines, native video export, research refit controls, ground placement, and effort bindings. All 88 fitting/spline/IK tests and 142 workspace unit tests pass locally.

- Issue: #11235; parent #11232
- PR: #11240 (retargeted to `main`)
- Branch: `feat/necromatcher-native-fit-11235`
- Validation: 88 fitting tests, 142 workspace tests pass; Ruff lint/format clean.
- Latest verified numerical evidence: V13 authored12/6 and V14 exact-six collective geometry-weight trials at `dafab40c107`; all candidates remain rejected/nonconverged. V13 selected dense28.781549px still fails grip/ground; V14 variant dense25.522536px/grip2.198523mm passes four finite checks but ground3.859638mm fails2mm between objective nodes.
- Evidence: [V14 Summary](historical_capture/geometry-weight-v14-summary.json); full hash-checked Desktop numerical copies and actual210-frame V13 overlay verified, with all three stills inspected. No V14 overlay or new report publication; see turnover for immutable receipt hashes and prior edition preservation.

# Active Necromatcher Workspace Delivery — #11239 (Merged)

native adapter and shared source-frame archive reader. Sixteen UI tests, desktop
recall and worker failure tests pass. Web type checking and scoped ESLint pass.
Native import/overlay and actual Hogan/Tiger web review are implemented. URL
navigation now hides stale results, verifies player/swing/capture ownership and
supports retry after frame loading errors. Further form tests, final parity and
fitted-model handoffs remain in progress. CI cycle 2 refreshes canonical launcher
context/atlas views after their freshness checks failed. Library draft PR #11237
depends on capture PR #11231. The capture
PR has an unrelated inherited title-case failure at `.jules/bolt.md:208`; do not
mark it accepted or change unrelated work under this delivery.

Owner priority is the integrated historical-player workspace (#11232), with library #11233, tile/review #11234 and real fitting/downstream qualification #11235. See [Necromatcher Turnover](necromatcher-turnover.md) for contracts, TDD evidence, real capture imports and current PR state. Tiger #11226 and Hogan #11229 remain open.

# Historical Player Capture Handoff

## Active: Tiger 2000 and Ben Hogan

- Repository/worktree: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-historical-capture`.
- Branch: `feat/historical-player-capture-11226`; implementation commit: ab2c869813a2b2be2f14b512ee34647694614377; handoff refresh: SELF.
- PR: https://github.com/D-sorganization/UpstreamDrift/pull/11231; open, CI pending. Epics: #11226 (Tiger), #11229 (Hogan); shared runner: #11230.
- Objective: complete both source-grounded historical reconstructions and make future players repeatable.
- Implemented: bounded PyAV streaming, existing SourceAsset/FrameIdentity contracts,
  existing MediaPipe estimator, normalized image XY/visibility/missingness, exact
  container PTS, lossless decoded PNGs, source/frame/observations/model/code hashes.
- Source authority: `docs/development/historical_capture/source-catalog.json`.
- Procedure: `docs/development/historical-capture-procedure.md`.
- Media/results: `C:/Users/diete/Downloads/historical-capture/` (outside Git).
- Tiger download: requested t_J6Vik3Tss, complete 1080p50 video, separate audio,
  metadata/description, and losslessly remuxed video with audio. Uploader claims
  2000; upload date is 2022-06-10 and does not verify the recording year.
- Final Hogan run: 110–135 presentation seconds, 750 frames, 739 detections,
  MediaPipe 1.0.1; missing detections preserved. Tiger final run: 2000 frames, 1994 detections (110-150 s).
- Earlier MediaPipe 0.10.32 pilot receipts are historical evidence, not current
  runner qualification. Final runs use content-based asset/frame IDs.
- TDD: missing module failed collection, then absent export function failed the
  streaming test; invalid local URI and decimal/Fraction boundary tests failed
  before correction. Last focused suite: 35 passed; latest capture suite: 12 passed.
- Checks: repository-wide Ruff lint passed; format check passed (8263 files);
  tracked file-size budget passed; procedure title check passed; design-manual
  governance verified existing `blocked-inventory-required` release state.
- Commands: `python3 -m pytest tests/unit/shadow_tracker/test_historical_capture.py tests/unit/shadow_tracker/test_footage_workflows.py tests/unit/shadow_tracker/test_fail_closed_fitting.py -q -o addopts=''`; `python3 -m ruff check .`; `python3 -m ruff format --check .`; `python3 scripts/ci/check_file_size_budget.py`.
- Whole Shadow Tracker suite passed: 364 tests in 36.28 s. Scoped mypy passed. Existing import deprecation warnings
  remain. No native-engine, scientific, or website acceptance is claimed.
- Ownership: this worktree started clean from fb1ac44949 on origin/main. Original
  checkout/other worktrees and their user-owned files were preserved. Partial
  failed output directories have no receipt and are not valid capture results.

## Ordered Continuation

1. Final Tiger/Hogan receipts collected; inspect dense source-bound landmark
   overlays and split each window at every cut/identity change. Contact-sheet
   inspection shows foreground body tracking, with errors/low-confidence joints.
2. Higher-resolution Hogan source processed: 899 frames, 892 detections; select clean continuous swings,
   and review P1–P10 checkpoint and impact intervals.
3. Monitor published shared-runner PR #11231 using the
   repository ci-watch-and-fix skill. Keep parent epics open.
4. Resolve recording/event lineage, playback scale and usage permissions; fit
   cameras and subject anthropometry with declared priors/uncertainty.
5. Fit dense constrained kinematics, compare native MuJoCo/Drake/Pinocchio FK,
   then independently qualify uninterrupted forward replay where supported.
6. Evaluate held-out film lineages and integrate eligible comparison artifacts
   into UpstreamDrift/AffineDrift. Do not claim completion until epic evidence exists.

# Historical Capture Continuation

Current authority: [Root Agent Handoff](../../AGENT_HANDOFF.md).
Shared runner #11230, Tiger #11226 and Hogan #11229: streaming extraction is implemented; reconstruction acceptance remains open.

# Current Handoff — Qualify OpenSim Native Dual-Club Dynamics and Replay (#11095)

- Branch: `feat/mmr-10o-opensim-dual-club-11095`
- Pull request: Refs #11095 (partial: fail-closed conversion; native qualification still requires opensim bindings on a pinned host/native CI lane).
- Done: OpenSim qualification schema plus **fail-closed conversion** after review audit (placeholder receipts, gates that always qualified, invented marker metrics):
  - `OpenSimQualificationReceipt` records `missing_evidence` and a resolvable `remedy`; status gated on every recorded check.
  - Unavailable `opensim` runtime ⇒ `UNAVAILABLE` even with a replay payload; unknown native test counts and absent `is_fresh_simulation`/`actuation_applied` flags are missing evidence (never assumed satisfied); missing rollout/marker data and non-finite values reject; derivative mismatch (`dq/dt` vs `v`) rejects.
  - Committed club receipts replaced with honest fail-closed UNAVAILABLE records (empty evidence fields, enumerated `missing_evidence`, remedy names the native lane command).
- Tests: 17 unit tests passed (`tests/unit/engines/opensim/test_opensim_dual_club_qualification.py` incl. 7 new fail-closed tests shown RED against the pre-fix placeholder path, then GREEN); Ruff check and format clean.
- Limitation: no native OpenSim execution exists anywhere in this evidence; real qualification requires the opensim bindings on a pinned host via `scripts/ci/run_native_engine_lane.sh --engine opensim`.

# Current Handoff — Qualify MyoSuite Native Dual-Club Dynamics and Replay (#11096)

- Branch: `feat/mmr-10m-myosuite-dual-club-11096`
- Pull request: Refs #11096 (partial: fail-closed conversion; native qualification still requires myosuite/MuJoCo on a pinned host/native CI lane).
- Done: MyoSuite qualification schema plus **fail-closed conversion** after review audit (placeholder receipts, gates that always qualified, invented marker metrics):
  - `MyoSuiteQualificationReceipt` records `missing_evidence` and a resolvable `remedy`; status gated on every recorded check.
  - Unavailable `myosuite`/MuJoCo runtime ⇒ `UNAVAILABLE` even with a replay payload; unknown native test counts and absent `is_fresh_simulation`/`actuation_applied` flags are missing evidence (never assumed satisfied); missing rollout/marker data, non-finite values, and unnormalized root quaternions reject; derivative mismatch (`dq/dt` vs `v`) rejects.
  - Removed SPEC-claimed but never-enforced tolerances and fabricated values (scaled early/terminal/clubhead RMS, hardcoded `pelvis_yaw_error_pct`); only `whole_rms_m` computed from recorded `markers_m`/`target_m` is emitted.
  - Committed club receipts replaced with honest fail-closed UNAVAILABLE records (empty evidence fields, enumerated `missing_evidence`, remedy names the native lane command); README model hashes demoted to regeneration targets.
- Tests: 19 unit tests passed (`tests/unit/engines/myosuite/test_myosuite_dual_club_qualification.py` incl. 7 new fail-closed tests shown RED against the pre-fix placeholder path, then GREEN); Ruff check and format clean.
- Limitation: no native MyoSuite execution exists anywhere in this evidence; real qualification requires the myosuite/MuJoCo stack on a pinned host via `scripts/ci/run_native_engine_lane.sh --engine myosuite`.

# Current Handoff — Qualify Drake Native Dual-Club Dynamics and Replay (#11094)

- Branch: `feat/mmr-10d-drake-dual-club-11094`
- Pull request: Refs #11094 (partial: fail-closed conversion; native qualification still requires pydrake on a pinned host/native CI lane).
- Done: [MMR-10D] Drake qualification contracts plus **fail-closed conversion** after review audit (placeholder receipts `c0ffee`/`deadbeef`/`cafebabe`, gates that always qualified, invented marker metrics):
  - `DrakeQualificationReceipt` now records `missing_evidence` and a resolvable `remedy`; `DrakeQualificationStatus` (`QUALIFIED`, `REJECTED`, `UNAVAILABLE`) is gated on every recorded check.
  - Unavailable `pydrake` runtime ⇒ `UNAVAILABLE` even when a replay payload is supplied; unknown native test counts are treated as missing evidence (never assumed nonzero); absent `is_fresh_simulation`/`actuation_applied` flags are unverified (fail-closed), not assumed fresh.
  - Missing `native_state`/`time_s` rollout or `markers_m`/`target_m` marker data is recorded as missing evidence and blocks qualification; derivative (`dq/dt` vs `v`) mismatch and non-finite state/energy values reject.
  - Removed synthesized marker metrics (scaled early/terminal/clubhead RMS, hardcoded `pelvis_yaw_error_pct`); only `whole_rms_m` computed from recorded observations is emitted.
  - Committed club receipts replaced with honest fail-closed UNAVAILABLE records (empty evidence fields, enumerated `missing_evidence`, remedy names the native lane command). Nightly lane receipt remains honest `status: fail` (0 executed tests, engine unavailable).
  - Added `"drake"` to `ENGINE_LANES` in `scripts/ci/run_native_engine_lane.py` and updated `scripts/ci/run_native_engine_lane.sh`.
- Tests: 36 focused tests passed (16 `tests/unit/engines/drake/test_drake_dual_club_qualification.py` incl. 7 new fail-closed tests shown RED against the pre-fix placeholder path, then GREEN; 20 lane/freshness tests). Ruff check and format clean on changed files.
- Limitation: no native Drake execution exists anywhere in this evidence; real qualification requires pydrake on a pinned host via `scripts/ci/run_native_engine_lane.sh --engine drake`.
- Next step: merge drivers follow; do not treat UNAVAILABLE receipts as engine qualification.

# Current Handoff — Consolidate Bolt Micro-Optimisation PRs (#11112, #11128, #11129)

# Current Handoff — Restore the High-Severity UI Npm Audit Gate (#11184)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `C:/Users/diete/Repositories/Worktrees/luna-upstream11184-20260930`
- Branch: `fix/main-npm-audit-11184`
- Commit: `SELF` (publication metadata update; implementation commit `d3b3a27bea36070c2db3a06e6e7be727a30e9667`)
- Pull request: [#11187](https://github.com/D-sorganization/UpstreamDrift/pull/11187), draft with `agent:codex` label; branch `fix/main-npm-audit-11184`.
- Governing issue: #11184 — restore the UI `npm audit --audit-level=high` gate using only compatible patched transitive resolutions.
- Done: added `ui/src/test/dependencySecurityContract.test.ts`; moved only `brace-expansion` 5.0.9 → 5.0.12 and `undici` 8.10.0 → 8.11.2 in `ui/package-lock.json`; added one SPEC row and an active development-log entry. No manifest, override, audit-policy, or moderate-advisory changes.
- RED evidence: baseline `npm ci` completed with 2 HIGH and 2 MODERATE advisories; baseline `npm audit --audit-level=high` exited 1. The regression contract failed on old lock versions 5.0.9 and 8.10.0.
- GREEN evidence: final `npm ci` passed; `npm audit --audit-level=high` passed with 2 MODERATE findings left (`@humanfs/node` 0.16.7 and nested `fflate` 0.6.10); `npm ls brace-expansion undici --all` showed only 5.0.12 and 8.11.2 on the affected paths. Contract: 2 passed; lint and type-check passed; all UI tests passed (99 files, 936 tests); build passed. Vitest emitted jsdom `scrollTo` notices; build emitted a large-chunk warning.
- Documentation checks: SPEC changelog validation and fleet hook passed. The repository development-log validator still exits 1 on pre-existing duplicate IDs, portfolio WIP/active-entry ceilings, and file-size ceiling; it reports no DL-#11184 finding.
- Compatibility evidence: registry metadata confirms the published patch releases and parent ranges `minimatch@10.2.5` → `^5.0.5`, `jsdom@30.0.1` → `^8.9.0`. The selected versions stay within those ranges.
- Coordination: fresh Repository_Management inbox was complete with no conflicts or new messages since 2026-09-29. Renewed the existing `codex-luna-upstream11184-20260930` presence, preserving its issue, branch and goal and adding `ui/src/test` and `docs/development`; presence expires at 13:04 UTC. Scoped REST lookup found no pre-existing PR for this branch. Authenticated `git ls-remote` confirmed `origin/main` remained exactly `aeb2edbca47c8b91a504fe199ed77c2e377fa6d5` before publication.
- Publication: root reviewed and accepted the bounded source, lockfile and test diff plus RED→GREEN evidence. Implementation commit `d3b3a27bea36070c2db3a06e6e7be727a30e9667` and metadata commit `5d2f3257ed2342bd2c6064aebed6a99a9564700e` passed normal pre-commit hooks; both branch pushes passed normal pre-push hooks. Draft PR #11187 is open with `Closes #11184` in its body and `agent:codex` label. SPEC uses actual PR key #11187. Remote refs were verified after publication: topic branch at `5d2f3257ed2342bd2c6064aebed6a99a9564700e`, main still at `aeb2edbca47c8b91a504fe199ed77c2e377fa6d5`. Two MODERATE audit findings remain (`@humanfs/node@0.16.7` and nested `fflate@0.6.10`). Root alone decides readiness and merge.
- Worktree state: clean after publication metadata; no merge, release, cleanup, or unrelated changes were performed. The primary checkout’s pre-existing untracked paths and other worktrees remain untouched.
- Next step: root decides whether draft PR #11187 is ready for review/merge. Do not mark ready, merge, release, or clean up as part of this handoff.

---

# Previous Handoff — Consolidate Bolt Micro-Optimisation PRs (#11112, #11128, #11129)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/w-ud-bolt-cons`
- Branch: `claude/ud-bolt-consolidated-0929` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11138; supersedes #11112, #11128 and #11129 (originals left open, closed citing the consolidator's squash SHA once it lands).
- Done: took only the `src/` numeric rewrites from three open Bolt PRs, dropping each PR's own SPEC row (this branch writes one row), and the `.jules/bolt.md` note carried by #11128.
  - #11112 (`bot/bolt-norm-einsum-residual`): row/column `np.linalg.norm(..., axis=N)` becomes `sqrt(np.einsum(...))` across `bunkershot3d/ball/qualification_fit.py` (column norm, `axis=0`) and 8 `motion_matching` sites (`club_calibration`, `club_only/acceptance`, `club_only/control_replay`, `dynamics_filter`, `execution/downswing`, `hip_calibration`, `multi_shooting_fit`, `pipeline/reference`). Carried unchanged, including the new identity test.
  - #11128 (`bolt-opt-pink-tasks-vdot-...`): small fixed-size vector norms in `engines/physics_engines/pinocchio/python/pink_tasks.py` (weld rotation/translation residuals, marker error) become `math.sqrt(np.vdot(x, x))`. Carried the numeric rewrite but dropped the inline `# ⚡ Bolt: ...` rationale comments to match this repo's default no-comments-unless-non-obvious-WHY convention (no prior Bolt PR merged into this repo left such comments in source).
  - #11129 (`bolt-optimize-physics-validation-norm-...`): `physics_validation.py` Jacobian error norm becomes `math.sqrt(np.vdot(diff.ravel(), diff.ravel()))`. Carried the rewrite; fixed the PR's `import math` placement (it was appended after `import numpy as np`, violating isort stdlib-before-third-party grouping) and dropped its inline Bolt comment for the same reason as #11128.
- Verified equivalence: `sqrt(einsum("ij,ij->i"))` / `("ij,ij->j")` / `("...i,...i->...")` and `sqrt(vdot(x,x))` are algebraically identical to `np.linalg.norm` for real arrays at every axis/keepdims/1-D/empty combination used; `tests/unit/motion_matching/test_row_norm_einsum_identity.py` (carried from #11112) checks this directly, including NaN/Inf and empty-array edges. `np.vdot` conjugates its first argument, which is a no-op for the real `float64` arrays at every #11128/#11129 call site.
- Tests: `python -m pytest -q -o addopts="" -p no:cacheprovider --tools-mode vendored --timeout=0` over the test files covering every changed module (qualification_fit, club_calibration, club_only/acceptance+control_replay, dynamics_filter, hip_calibration, multi_shooting_fit, pipeline/reference, pink_tasks, physics_validation, plus the new identity test): 226 passed, 2 pre-existing failures unrelated to this change (`test_club_observation_contracts.py::test_public_input_validation_survives_python_dash_o` and the equivalent in `test_club_plausibility_acceptance.py` — both fail identically on `origin/main` with `ModuleNotFoundError: No module named 'bunkershot3d'` from a `python -O` subprocess whose `sys.path` doesn't include the repo root; unrelated to any …
- Next step: CI green on the refreshed head, then squash-merge this PR and close #11112/#11128/#11129 citing its squash SHA.

---

# Past Handoff — Anti-Phantom-Merge Path Extraction for Scripts, Workflows, and Parentheticals (#11124)

- Repository: D-sorganization/UpstreamDrift
- Branch: `fix/phantom-guard-scripts-paths-11124`
- Commit: `SELF`
- Pull request: closes #11124 (merged as #11131, squash `657d95287e`).
- Done:
  - Extended `ISSUE_PATH_PATTERN` in `scripts/ci/check_phantom_guard_paths.py` with boundary lookbehind `(?<![\w/.-])` and recognized repository prefixes `scripts/` and `.github/workflows/` (and `.github/`).
  - Added trailing punctuation trimming from extracted paths.
  - Implemented `_drop_parenthetical_prose_fragments` to filter comma-separated fragments inside parentheses that lack recognized code/configuration file extensions.
  - Added unit tests for scripts/workflows path extraction and parenthetical prose dropping, plus regression test reproducing issue #10965 / PR #11113.
- Tests: TDD red-to-green workflow followed; 27 unit tests pass in `tests/scripts/test_check_phantom_guard_paths.py`. Ruff, Black, LOD, and Architecture Budget clean.
- Resolution: merged ahead of the consolidator (keep-both rewrite of this section at #11138 refresh); superseded by the merge itself.

# Current Handoff — Govern Screenshot Schema, Capture Metadata, and Qualified Assets (#9191)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/comp-b3-screenshot-governance-9191`
- Issue: #9191
- Commit: `SELF`
- Pull request: Closes #9191.
- Done:
  - Updated screenshot schemas (`docs/api/contracts/upstreamdrift-companion-screenshots-v1.schema.json` and `docs/api/contracts/upstreamdrift-companion-v1.schema.json`) with exact capture metadata (`capture_environment`, `source_commit`, pixels/viewport, and summary screenshot counts).
  - Implemented `scripts/companion_screenshots.py` single authority for screenshot verification, PNG IHDR byte extraction, SHA-256 digest validation, registry loading/validation, and headless deterministic asset generation with Matplotlib Agg backend.
  - Generated 6 representative deterministic PNG visual assets in `docs/screenshots/` (pendulum simulator desktop light/dark + mobile dark, project map, rate of closure, tour matching viewer).
  - Registered all 74 visible programs in `scripts/config/companion_screenshots.v1.json` (6 captured records, 70 pending records with explicit reasons).
  - Wired screenshot registry and inventory into `scripts/companion_catalog.py` and consumer publication bundles.
- Tests:
  - 165 passed across `tests/companion` and `tests/unit/scripts/test_verify_companion_screenshots.py`.
  - Architecture budget OK; DRY duplication gate OK; hardcoded style ratchet OK; ruff check and format clean; prettier clean.
- Next step: Create PR and release lease.

# Current Handoff — Implement Simscape Continuous-Replay Qualification Harness (#11107)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/mmr-07-simscape-continuous-replay-11107`
- Commit: `SELF`
- Pull request: addresses #11107 (bounded implementation).
- Done:
  - Implemented `src/shared/python/motion_matching/simscape_replay_harness.py`:
    - Full-rate, unbroken continuous-replay trajectory contract checking (`validate_continuous_replay_inputs`) with fail-closed enforcement for missing terminal samples, non-finite states, duplicated/non-monotonic timestamps, non-R2025b releases, hidden motion prescription, candidate hash mismatches, and state resets.
    - Added `compute_per_marker_channels` and `compute_phase_channels` generating per-marker and per-phase RMS error distributions over full native coordinates.
    - Integrated canonical metrics into `acceptance.evaluate()` without modifying frozen thresholds (`qualify_simscape_continuous_replay`).
    - Added structured qualification receipt schema (`simscape-continuous-replay-receipt/1`) with lossless serialization (`save_replay_qualification_receipt`, `load_replay_qualification_receipt`).
    - Implemented `load_continuous_replay_trajectory` to parse both explicit `q`/`v` and composite `native_state` arrays.
  - Added `-ContinuousQualification` switch to `scripts/matlab/run_simscape_candidate.ps1`.
  - Review fixes (Codex P1/P2 on this PR): duration gate requires start at t=0 and the elapsed horizon span; acceptance metrics delegate to canonical `compute_replay_five_metrics()` (0.60 s early window, canonical club-cluster labels); pelvis yaw measured from `WaistLeft`/`WaistRight`; unmeasured normal force/penetration/closure disclosed as unavailable instead of fabricated passes; receipt loader preserves null terminal channels; `load_replay_evidence_inputs()` derives the recomputed NPZ digest (fail-closed), candidate `q0`/`qd0`, and receipt-declared control identity; `replay_returned102_r2025b.m` records the actuation/reset identity (run-102 receipt extended additively, measured values unchanged).
  - Evaluated existing native candidate replay evidence (`two_window_fit_9967_102`) producing `simscape_replay_qualification.json`, confirming honest rejection at the terminal phase (~40.3 mm > 35.0 mm G1 ceiling); canonical metrics cross-check the committed native receipt (early 9.995 mm, club 8.391 mm, yaw 0.595 %, terminal 40.30 mm).
  - Implemented 22-test suite in `tests/unit/motion_matching/test_simscape_continuous_replay_harness.py` (11 fail-closed + 11 review-fix regressions, red-first documented in the PR body).
- Validation:
  - 22/22 tests passing in `test_simscape_continuous_replay_harness.py`.
  - Full `tests/unit/motion_matching` run: 1901 passed, 11 failed — all pre-existing on the pristine baseline (`c3d_reader.load_c3d` absent, `bunkershot3d` absent, stability-matrix precondition); not regressions of this change.
  - Lint/format/type checks clean on touched modules: `ruff check`, `ruff format`, `mypy`.
  - Section 12 changelog entry updated in `SPEC.md` and verified with `check_spec_changelog_duplicates.py`.
- Next step: CI green, PR review by owner.

# Current Handoff — Ship the Historical-Video Evidence Review Workflow (MMR-12, #11098)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/mmr-12-historical-video-review-11098` (baseline `origin/main` 7b74fb68c6)
- Pull request: closes #11098.
- Done:
  - Preserved multi-shot frame isolation and revision lineage in `DefaultShadowTrackerService`: frames from multiple distinct shots sharing local `frame_id`s (e.g. `frame-000000`) are stored without collision under scoped `(shot_id, frame_id)` namespace indexing and retrieved deterministically.
  - Manual mask updates invalidate downstream fit results deterministically while tracking revision lineage (`parent_revision_id`, `producer_id`, `correction_note`).
  - Added video file ingestion via `import_video`: container PTS extraction, constant and variable-frame-rate (VFR) containers, affine timing mappings (slow motion), and shot boundary cut definitions.
  - Bounded decode limits and prompt cancellation: passes `DecodeLimits` with active cancellation token callbacks to abort decode loops during ingestion rather than only upon completion.
  - Fail-closed error handling: invalid media and missing codecs raise descriptive errors while leaving active session state intact and recoverable.
  - Installed PyQt review journey: `ShadowTrackerWidget` and `ShadowTrackerReviewModel` support video import toolbar action, keyboard scrubbing (`Key_Left`, `Key_Right`, `Key_Home`, `Key_End`, `Key_W` for worst-frame jump), shortcut bundle persistence (`Ctrl+S`, `Ctrl+O`, `Ctrl+I`), viewport rendering with clock authority metadata, and honest refusal reporting for unverified automated fitting backends.
  - Review fixes (Codex P1s on this PR): `import_video` stages the full decoded batch and validates scope ownership, single source asset, and revision-id uniqueness before any session mutation, so a failing repeat import leaves the reviewed session exactly intact; without an evidenced timing mapping imported frames keep `physical_time_s=None` with the canonical unknown-time reason (PTS authority stays in `timing_mode`/`clock_evidence`, never transplanted into fabricated provenance); the viewport renders unknown physical time as `Physical Time: unknown (<reason>)` via `format_clock_evidence_text` instead of formatting `None` (pre-fix `TypeError` aborted the paint event on GUI-default imports).
- Tests:
  - `tests/unit/shadow_tracker/test_service.py`: 16 passed (incl. repeat-import atomicity against scope collision / foreign asset, unknown-time provenance preserved in export, VFR/slow-motion timing, cancellation during decode, corrupt media recovery, bundle save/reload lineage).
  - `tests/tools/shadow_tracker/test_shadow_tracker_gui.py`: 7 passed (incl. unknown-time viewport rendering and clock-evidence text formatting, installed PyQt review journey, keyboard navigation, worst frame jump, mask correction dirty tracking, honest fit refusal).
  - Entire `tests/unit/shadow_tracker/` and `tests/tools/shadow_tracker/` suites: 330 collected, all passing.
  - Ruff check/format and mypy clean on touched modules; architecture budget and spec changelog duplicate check: PASS.
- Next step: CI green, then ready and arm the PR; release agent lease.

# Current Handoff — Repair Reduced-Model and Club-Only Product Claims (MMR-11 #11097)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/mmr-11-reduced-model-claims-11097`
- Commit: `SELF`
- Pull request: Refs #11097 (partial: fail-closed gates shipped; not claiming "Closes" — see Open below).
- Done (all code-level, unit-test-verified):
  - Disqualified four historical unverified driven-triple pendulum receipts (`driver_amateur`, `driver_elite`, `iron_amateur`, `iron_elite`) at projection time in coverage; driver/iron matrix cells report rejection truthfully. Receipt/evidence files on disk were NOT rewritten.
  - Enforced complete geometry, inertia, control, q0, and v0 hashes and non-positive out-of-plane residual rejection (`out_of_plane_rmse_m <= 0.0`) on promoted baseline packages, failing integrity checks fail-closed. Committed packages were NOT regenerated from raw artifacts in this PR.
  - Early planar floor rejection in `validate_and_project_target` gates the CURRENT target's butt/clubhead distances to the selected plane against a DbC-validated `max_marker_rmse_m` (finite, strictly positive; NaN ceilings fail closed instead of silently disabling the gate), rejecting before expensive optimization.
  - Required fresh continuous replay (`has_continuous_replay`) and labeled inferred body posture (`body_motion_disclaimer`) in club-only UI result views, preventing unqualified views from displaying verified status.
  - Matrix qualification reports cannot claim all-complete while unresolved cells remain (`assert_matrix_not_all_complete_with_unresolved`); the ~80 unresolved cells are still unresolved and tracked.
- Open (MMR-11 acceptance criteria NOT satisfied by this PR): raw-to-package reproduction/regeneration of reported baselines and residuals, Board-selected required club-only cells, and any native qualification run.
- Tests: TDD red-to-green for both Codex findings (current-target planar-floor gating; ceiling DbC validation) plus all fail-closed gate tests. Scoped runs at `46b804bc69`: 18 passed (`tests/unit/motion_matching/test_fit_options_dbc.py`, `tests/unit/engines/physics_engines/pendulum/test_motion_matching_provider.py`) and 86 passed (`tests/unit/tour_baselines/test_coverage_matrix.py`, `test_qualification.py`, `test_baseline_packages.py`, `tests/unit/motion_matching/test_club_matrix_qualification.py`, `test_club_ui_integration.py`). `ruff check` clean on changed files. Full suite, mypy and CI gates were not run by this slice.
- Next step: raw-to-package regeneration and Board cell-selection follow-ups for #11097.

# Current Handoff — Publish a Best-Candidate Viewer With Honest Residuals (#11102)

- Repository: D-sorganization/UpstreamDrift
- Branch: `feat/mmr-16-best-candidate-viewer-11102`
- Issue: #11102
- Pull request: refs #11102 (mergeability and PR body owned by the main lane; desktop-slice scope below)
- Done:
  - Enforced raw observation immutability on `ReplayData` by setting numpy array flags to `writeable = False` on `time_s`, `coordinates`, `model_markers_m`, `target_markers_m`, and `valid_mask`.
  - Added dataclasses `MarkerResidual`, `FrameResidual`, and `ResidualSummary` with `mean_rms_m` property.
  - Implemented `compute_residual_summary` with per-frame RMS, per-marker errors, phase boundaries (Address, Top, Impact, Finish), and 3D residual vectors `(model - target)`.
  - Implemented `export_board_ready_still` generating high-DPI stills with observed dots, model skeleton, residual vector lines, and comprehensive metadata banner (Candidate SHA, Engine, Drive Mode, Frame, RMS, Verdict).
  - Extended `MatchedSwingFilter` with `drive_mode` and `profile`. Implemented `extract_drive_mode` and `rank_candidates(rows, capture, drive_mode)` sorting comparable candidates in strictly ascending RMS error order while preserving disqualified/rejected verdicts honestly.
  - Wired `rank_candidates` into the browser's real list-build path (`_apply_filters`) so the auto-selected first row is the best comparable candidate (ascending `whole_marker_rmse_m`); rejected rows remain visible with their verdicts.
  - Fixed the viewer frame RMS (`_evaluate_single_frame_residual`, `viewer_frame`, `get_per_engine_rms`) to pool per-marker 3D distances (`sqrt(mean(sum(valid_diff**2, axis=-1)))`) matching the canonical `tour_metrics.compute_shared_metrics`, so residual summaries, physics scores and captions agree with the ledger.
  - Frames with zero valid markers claim no worst marker (`FrameResidual.valid_markers`) and are excluded from global-worst selection and `mean_rms_m`, so the Worst Residual jump never lands on unobserved placeholder data.
  - `TourMatchingViewerWidget.load_file` now accepts optional receipt provenance (`candidate_hash`, `engine_name`, `drive_mode`, `is_accepted`, `rejection_reason`); the browser's `_on_open_tour_matching_viewer` forwards the selected `LedgerRow`'s hash, engine, drive mode and verdict so captions match the selected receipt and rejected candidates show their failure banner.
  - Extended `TourMatchingViewerWidget` with properties `current_frame`, `current_rendered_frame_index`, `current_frame_time_s`, `current_rms_error`, `physics_score`, `candidate_hash`, `drive_mode`, `title_caption`, and `capabilities_caption`.
  - Implemented `select_worst_residual()` jumping directly to the exact discrete frame with the highest marker residual error.
  - Implemented camera view presets (`perspective`, `front`, `side`, `top`, `isometric`) and appearance presets (`default`, `high_contrast`, `residual_vectors`, `dots_and_mesh`) preserving physics scores and numerical receipts.
  - Implemented `launch_native_backend` and graceful `_show_recovery_message` handling missing engines without crashes.
- Tests (scoped, `/tmp/ud-venv-11132`, offscreen PyQt6 + mujoco): red outcomes for all four review fixes recorded pre-fix; post-fix `pytest tests/unit/tools/test_matched_swing_browser_best_candidate.py tests/unit/tools/test_tour_matching_viewer_residuals.py` → 21 passed and the targeted viewer suites (core, playback, combo, forces, native_button, adapter) → 29 passed; two combo pins that predated the review fixes updated (old flattened-RMS value and removed `#d9534f` hex literal, see 70762eb7fe). Exit code 0 on every run.
- Not done (remains open on #11102 — the desktop PyQt seam above is fixed, but the issue's acceptance does not stop there):
  - The same journey is not exercised on the supported web/API surfaces (installable-PyQt and web/API parity, accessibility keyboard review and missing-engine recovery on real hosts are unreviewed).
  - Human visual review of captions/stills and acceptance checkboxes in #11102 are unchecked; no claim is made that "all acceptance criteria" pass.
- Next step: main-lane rebase/CI and frontier review of PR #11132.

---

# Past Handoff — Unify Per-Package Coverage Gates on the Exclusion Budget (#10965)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `/tmp/ud-wt-10965` (baseline `origin/main` 64962e8d6)
- Branch: `bot/issue-10965-coverage-gate-authority`
- Commit: `SELF`
- Pull request: closes #10965.
- Done: `scripts/check_coverage_gates.py` reads the 6 gates directly from `scripts/config/mypy_exclusion_budget.json` (single DRY gate authority), matches coverage-report files by repository-relative path prefix using `summary.num_statements`/`covered_lines`, and `--strict` now fails (exit 1) on a gate matching zero files; without `--strict` it warns instead. `check_mypy_exclusion_budget.py` treats an expired gate `ratchet_on` as a non-fatal warning (gates are CI-enforced) and makes `ratchet_to` optional. The `tests` CI job emits `--cov-report=json:coverage.json` and a new step runs `python3 scripts/check_coverage_gates.py --report coverage.json --strict` after the threshold enforcer.
- Measured floors (min_coverage rounded DOWN to 0.1, ratchet_on 2027-01-01, local targeted pytest with coverage, optional-engine backends not installed): api-routes 84.3% (4406/5221), data-io 50.1% (1450/2891), execution-checkpointing 22.4% (567/2521), deployment 48.4% (660/1362), optimization 49.9% (1405/2815), engine-adapters 7.1% (3687/51393).
- Tests: TDD red (2 new unmatched/strict tests failed with old exit-2 semantics) then green; `tests/unit/scripts/test_check_coverage_gates.py` + `test_check_mypy_exclusion_budget.py`: 29 passed. Checker smoke against a real coverage.json fixture: strict exit 1 on unmatched gates, lax exit 0 with WARNING line.
- Next step: CI green, then ready and arm the PR.

---

# Past Handoff — Fleet Critic Pass 2026-09-25

## Identity

- Repository: D-sorganization/UpstreamDrift
- Branch: `staff/fleet-critic-task-653b92-v2`
- Baseline commit: `28b37bd47` (origin/main)
- Implementation commit: SELF
- Pull request: not created (supersedes #10942)
- Governing task: Fleet Critic scheduled pass (bi-weekly, 1st & 3rd Friday)
- Session: `fleet-critic-task-653b92`

## Objective and Status

- Objective: Produce the 2026-09-25 scheduled Fleet Critic review for
  UpstreamDrift, covering the last 30 days of changes (NM-09–NM-12,
  TB-08–TB-10, Bolt optimizations), and commit the artifact under
  `docs/critiques/2026-09-25/`.
- Status: Complete — critique committed, draft PR to be opened. Re-landed as a
  superseding PR because #10942's branch conflicted with main.
- Completed:
  1. Reviewed recent commits; focused on neural-motion matrix builder,
     tour-baseline qualification, benchmark runner, and Bolt journal.
  2. Produced `docs/critiques/README.md` (index), `summary.md`, and
     `weaknesses.md` (6 findings: 3 High, 2 Medium, 1 Low).
  3. Added this handoff section.
- Remaining: Push branch and open draft PR.

## Key Findings

Three High-severity weaknesses in the neural-motion checkpoint pipeline:

1. `matrix/builder.py:_make_evidence()` — synthetic three-seed evidence.
2. `matrix/builder.py:_build_card_for_model()` — model-ID-derived hashes.
3. `benchmark/runner.py:run_model_comparative_benchmark()` — self-referential
   speedup baseline.

Two Medium: hardcoded economics in model cards; refinement sensitivity proxy.
One Low: Bolt speedup claims without benchmark fixtures.

## Files Changed

- `docs/critiques/README.md` — new critique index
- `docs/critiques/2026-09-25/summary.md` — executive summary
- `docs/critiques/2026-09-25/weaknesses.md` — 6-finding weakness catalog
- `docs/development/HANDOFF.md` — this section (inserted, not a replacement)
- `docs/index.md` — catalog row for the new `docs/critiques/` directory (doc-catalog gate)
- `SPEC.md` — one change-log row for this pass

## Validation

- No source code changed; linting/tests not required.
- `python scripts/ci/check_spec_changelog_duplicates.py` — passed.

## Blockers and Risks

- None. Read-only critique pass; no source modifications.

## Next Steps

1. Commit these files.
2. Push `staff/fleet-critic-task-653b92-v2` and open a draft PR targeting `main`.
3. High-severity findings 1–3 should be forwarded to the next Board meeting
   as candidates for the consensus priority list.
4. Author/owner to decide remediation order; suggested tracking via one
   consolidated issue covering all three High findings.

## Change Log

- SELF — Fleet Critic scheduled pass: 6 scientific weaknesses in neural-motion
  checkpoint matrix and benchmark runner.

---

# Past Handoff — Consolidated Bolt Row-Norm Micro-Optimisations (#11073, #11074, #11076)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-bolt-einsum`
- Branch: `claude/ud-bolt-einsum-consolidated` (baseline `origin/main` 0206717f)
- Commit: `SELF`
- Pull request: #11081 (https://github.com/D-sorganization/UpstreamDrift/pull/11081). It supersedes the Bolt PRs #11073, #11074 and #11076, which conflicted on `SPEC.md` and carried `n/a` change-log rows.
- Done: took only the `src/` changes from the three Bolt PRs. Row-wise `np.linalg.norm(..., axis=1)` becomes `np.sqrt(np.einsum("ij,ij->i", d, d))` in `contact_identification`, `multi_shooting_fit` and `tour_baselines/calibration`, and max-norm reductions take one square root after `np.max` in `contact_mode_qualifier` and `calibration`. The unmeasured inline speed claims were removed.
- Measured (NumPy 2.2.6, this workstation): 4.8x on 20000x3 rows and 1.8x on 657x3 rows; results match `np.linalg.norm` to 1e-15.
- Excluded: #11077. On top of the same edits it reverts the `nightly-cross-engine.yml` hardening, deletes cross-engine validation tests and raises the architecture budget.
- Validation: `test_contact_identification`, `test_contact_mode_qualifier_pf04`, `test_multi_shooting_fit` and `test_tour_calibration`: 54 passed. Architecture budget OK. Divergence inventory regenerated (all four files are UD-only).
- Next step: CI green, then ready and arm the PR; close the three Bolt PRs as superseded after merge.

---

# Past Handoff — Run the MJX L-BFGS Arm Toward Convergence (#11071)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11071-lbfgs`
- Branch: `claude/ud-11071-lbfgs-converge` (baseline `origin/main` d4286fbf)
- Commit: `SELF`
- Pull request: #11075 (https://github.com/D-sorganization/UpstreamDrift/pull/11075); closes #11071 (follow-up to #11058, epic #11006).
- Done: re-ran only `mjx-lbfgs` at `--mjx-iterations 60` with the MJX interpreter (Python 3.12.10, MuJoCo 3.13.0, DeskComputer, commit ec249edf, whose `src/` and `scripts/` match main d4286fbf). `none` and `shooting` do not read `--mjx-iterations`, and only `knot_gradient_optimiser.py` changed since their #11058 runs, so their cases hold a `reuse.json` pointer to `../mjx_benchmark` instead of copies; the ledger therefore does not count them twice. `load_row` follows the pointer and rejects a dangling one (`tests/unit/motion_matching/test_benchmark_mjx_knot_solvers_script.py`).
- Budget choice: the #11058 L-BFGS runs took about 1.6-2.8 ks for 10 iterations. 60 iterations took 6217 s (driver) and 5097 s (iron), 70 evaluations each.
- Result (replay RMS through the shared simulator, driver / iron): 40.8 / 42.6 mm, down from 47.2 / 52.9 mm at 10 iterations; shooting stays 84.5 / 88.6 mm. Downswing weight fraction min 0.31 / 0.29, so the iron did **not** lose contact (it rose from 0.17). Both runs stopped on `max_iterations`.
- Trend: the MJX-plant RMS in the optimiser history went 37.6 → 28.2 mm (driver) and 34.5 → 27.3 mm (iron) between evaluations 10 and 69, about 0.05 mm per evaluation at the end. The shared-simulator replay stays 12-15 mm above the MJX plant, so more iterations mostly shrink the plant number, not the scored one.
- Decision: keep `none` as the default (the promotion rule needs a `converged` stop). 60 iterations (about 1.5 h per capture) is the practical budget reached.
- Evidence: `docs/development/full_body_models/evidence/mjx_benchmark_lbfgs60/` (receipts, `provenance.json`, `rows.json`, `REPORT.md`); ledger 133 receipts.
- Next step: none; merged as PR #11075.

---

# Past Handoff — MJX Knot Optimiser Head-to-Head Benchmark and Promotion Decision (#11058)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11058-benchmark`
- Branch: `claude/ud-11058-benchmark` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11072. Closes #11058 (epic #11006 P5).
- Done:
  - `knot_gradient_optimiser.lbfgs_minimise` runs SciPy L-BFGS-B on the same JAX gradient as Adam; `--mjx-method {adam,lbfgs}`.
  - Every MJX reference is rescored through the shared `FullBodySimulator` (`trajectory_optimiser.shared_simulator_replay`, via `pipeline.dynamics.score_reference`). A stage that writes no reference is an error.
  - `weight_fraction_report` gives receipts a per-phase weight fraction.
  - `solver_benchmark.py` scores the receipts and makes the promotion decision; `scripts/benchmark_mjx_knot_solvers.py` runs the benchmark or re-renders it (`--render-only`).
- Evidence: the receipts, `rows.json` and `REPORT.md` are in `docs/development/full_body_models/evidence/mjx_benchmark`. The runs used the MJX interpreter (Python 3.12.10, MuJoCo 3.13.0) on DeskComputer at commit `fee18106`, with shooting at 8 passes and MJX at 10 iterations, one run at a time.
- Result (replay RMS, driver / iron): none 84.5 / 88.6 mm; shooting 84.5 / 88.6 mm (never beats its start); mjx-adam 62.4 / 55.7 mm, but it **loses iron downswing contact** (weight fraction min 0.00); mjx-lbfgs 47.2 / 52.9 mm (weight fraction min 0.33 / 0.17). No arm reaches G1 (25 mm). IPOPT is unavailable: no cyipopt or pydrake binding.
- Decision: keep `none` as the default. Both MJX methods stopped on the iteration budget, not on convergence. Follow-up #11071 runs L-BFGS to convergence.
- Deviation: the issue's `least_squares` FD arm was not run, because one Jacobian means 47 knots × 38 actuated coordinates = 1786 shared replays. L-BFGS-B on the exact gradient replaces it; the report says so.
- Tests: `tests/unit/motion_matching/` (solver benchmark, trajectory optimiser selection, knot gradient optimiser, MJX optimisation, pipeline) pass in Python312 and the MJX environment.
- Next step: none; merged as PR #11072.

---

# Past Handoff — Force-Plate Stitching Tests Canonical Import and Overlay Cleanup (#11034)

- Repository: D-sorganization/UpstreamDrift
- Branch: `fix/force-plate-test-module-identity-11034` (baseline `origin/main`)
- Pull request: closes #11034.
- Problem: In `unit-test-gate` under `pytest -n auto`, `test_force_plate_stitching.py` intermittently failed with `AttributeError: module 'shared.python.sidekick.lab.bio.force_plate_stitching' has no attribute 'CombinedForcePlateProcessor'` due to in-test manipulation of `sys.modules` and overlay reinstallation resolving against stale parent module attributes.
- Fix:
  1. `test_force_plate_stitching.py`: import canonical `CombinedForcePlateProcessor` directly via `from src.shared.python.sidekick.lab.bio.force_plate_stitching import CombinedForcePlateProcessor`.
  2. `sidekick_extension_overlay.py`: when uninstalling loaded extension modules, cleanly `delattr` the attribute from the parent module in `sys.modules` if present to prevent attribute pollution.
- Tests: `pytest tests/unit/sidekick/lab/bio/test_force_plate_stitching.py tests/unit/launcher/test_sidekick_extension_overlay.py` and `pytest -n 2 ...` pass (13 passed).
- Next step: none; merged as PR #11069.

---

# Past Handoff — Shooting Fit No Longer Crashes on an Unimported `fs` (#11059)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11059-shooting`
- Branch: `claude/ud-11059-shooting-fs` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11060 (draft → ready). Closes #11059.
- Problem: `pipeline/dynamics.py` imports `full_body_forward_dynamics as fs` only under `TYPE_CHECKING`; `build_tracking_controller`, `replay` and `zmp_filter` import it locally, but `shooting_fit` did not, so `--shooting-fit N` raised `NameError: name 'fs' is not defined` before its first pass.
- Fix: the same local import in `shooting_fit`.
- Tests: `test_shooting_fit_runs_a_pass` runs one pass with stubbed replay and marker errors; it fails on `main` with the production `NameError` and passes with the fix. `tests/unit/motion_matching/pipeline/`: 77 passed, 10 skipped.
- Acceptance: the driver run `--static-seeds --shooting-fit 8` that crashed on `main` completes (rc 0, 1123 s). Pass 0 replays at 84.5 mm, matching the canonical `anthro_driver_seeds` receipt. Passes 1-8 score 108.6, 121.3, 122.0, 114.5, 118.8, 111.1, 96.3 and 95.9 mm, so the fit keeps pass 0 (`best_iteration` 0) and the final dynamics replay is 84.5 mm.
- Next step: CI green, mark ready and arm the PR.

---

# Past Handoff — Opt-In MJX Knot Trajectory Optimiser Stage in the Matching Pipeline (#11051)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11051-pipeline`
- Branch: `claude/ud-11051-mjx-pipeline` (baseline `origin/main`)
- Commit: merged to `main` as PR #11057
- Pull request: #11057 (https://github.com/D-sorganization/UpstreamDrift/pull/11057). Closes #11051 (epic #11006 package P4). Executed by agy (Gemini 3.8 Flash), reviewed and reworked by the orchestrator.
- Built: `pipeline/trajectory_optimiser.py` (`TRAJECTORY_OPTIMISERS = {none, mjx-knots}`, `run_trajectory_optimiser`) and the CLI flags `--trajectory-optimiser` (default `none`) and `--mjx-iterations` (default 40, `DEFAULT_MJX_ITERATIONS`). With `mjx-knots` the stage exports the MJX package, runs the #11049 knot optimiser, and writes `mjx_optimised_reference.npz` and `mjx_optimisation_receipt.json`; `receipt.json` gains a `trajectory_optimiser` summary. The receipt builder is shared with the #11046 evidence CLI (`optimisation_receipt`).
- Decision: when JAX or `mujoco.mjx` is missing, selecting `mjx-knots` raises `DependencyUnavailableError` naming the module and `--trajectory-optimiser none`. The epic's "falls back cleanly" is read as a clear error, not a silent fallback, because the default stays `none` and a silent skip would hide that the requested stage never ran.
- Tests: the six prototype tests in `test_mjx_optimisation.py` loaded private copies from the evidence script that #11046 replaced; they duplicated `test_knot_gradient_optimiser`, `test_jax_contact` and `test_mjx_tracking_plant`, so they were replaced by a constant-parity test and a toy-package stage test. `test_trajectory_optimiser_selection.py` covers parser defaults and rejections, the `none` no-op, both missing-dependency errors and the receipt key. 29 passed in `~/.venv-mjx` (MuJoCo 3.13); pipeline tests 76 passed, 10 skipped in default Python.
- Acceptance: `--trajectory-optimiser mjx-knots --mjx-iterations 3` on the static-seeds run: port check 0.06516 m, best 0.05029 m (47 knots, 38 actuated coordinates, float32); the evidence CLI on a copy of the same package gives an identical history. Default run on branch vs `main`: identical file set, `receipt.json` differs only in `elapsed_s` and has no `trajectory_optimiser` key.
- Next step: none; merged as PR #11057.

---

# Past Handoff — Every `mj_fullM` Call Routed Through One MuJoCo 3.13-Safe Helper (#11055)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11055-fullm`
- Branch: `claude/ud-11055-fullm-helper` (baseline `origin/main`)
- Commit: merged to `main` as PR #11056
- Pull request: #11056; entry DL-#11055. Closes #11055. Executed by agy (Gemini 3.8 Flash), reviewed by the orchestrator.
- Problem: `pyproject.toml` allows `mujoco>=3.6,<4`, but MuJoCo 3.13 removed `MjData.qM` and changed `mj_fullM(model, dst, data.qM)` to `mj_fullM(model, data, dst)`; measured in `~/.venv-mjx` (3.13.0) the old form raises `AttributeError`, which stopped the matching pipeline there.
- Built: `src/shared/python/engine_core/mujoco_compat.py::full_mass_matrix(mj, model, data, dst=None)` picks the signature at run time and imports no `mujoco` at module level; every `src` call site (mujoco engine, myosuite engine, motion matching, simulation backends, physics validation), including the two existing hand-written dual paths, now calls it. `src/engines/physics_engines/mujoco/docker/` keeps its own pinned MuJoCo and is untouched.
- Orchestrator changes on review: the hinge test checks the analytic inertia (2/5 m r^2 = 0.004 kg m^2) instead of repeating the helper's branch, and the package-level re-export was dropped.
- Placement: the helper lives in `engine_core`, not `simulation_backends` as #11055 proposed; importing anything under `simulation_backends` runs its package `__init__` (measured 8.98 s, 1671 modules), while `engine_core` costs 0.19 s (158 modules, the `src.shared.python` baseline), and the helper is imported by 19 modules.
- DRY ratchet: shortening the four `mj_fullM` call sites in `counterfactuals.py` made the ZTCF/ZVCF blocks hash-identical, so the state-load, forward, mass-matrix and solve sequence is one `_forward_acceleration` helper; ZTCF and ZVCF outputs on a two-link model match `main` exactly.
- Changed-file gates surfaced existing debt in touched files, fixed rather than excepted: `advanced_kinematics.solve_inverse_kinematics` (111 > 100 lines) has its damped-least-squares and nullspace step extracted to `_dls_step`, same arithmetic; `sim_widget.set_axial_color_scale` calls `_render_once()` (what the mixin's `render()` does) because mypy resolves `self.render()` to `QWidget.render`, which needs an argument; the divergence inventory records `mujoco_compat.py` as `ud-only`.
- Tests: `tests/unit/engine_core/test_mujoco_compat.py` (analytic hinge, preallocated `dst`, both branches with fakes, three preconditions) plus `test_native_force_equations.py`: 12 passed on MuJoCo 3.13. The 200 test files that reference a touched module, in default Python (MuJoCo 3.4): 1967 passed; the 31 failures are 19 that fail identically on `main` and 12 that pass when run alone (batch-order pollution).
- Next step: none; merged as PR #11056.

---

# Past Handoff — MJX Knot Optimiser Core Moved Into `src` (#11049)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11049-optimiser`
- Branch: `claude/ud-11049-mjx-optimiser` (baseline `origin/main`)
- Commit: merged to `main` as PR #11054
- Pull request: #11054; entry DL-#11049. Closes #11049. Package P3 of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed and refactored by the orchestrator.
- Built: `src/engines/physics_engines/mujoco/python/motion_matching/mjx_knot_optimiser.py` holds the package loader (`load_mjx_package`, strips equality blocks and floors armature), `KnotOptimisationSettings` (frozen, validated), `optimise_reference` and `diagnose_reference`, sharing one `_prepare` problem builder. The module writes no files and does not change JAX config; the result records the dtype it ran in.
- The evidence CLI `mjx_trajectory_optimisation.py` is now a thin wrapper (362 → 268 lines); every flag, output file and receipt key is kept, and `main()` still runs float32.
- Measured parity on the regenerated `anthro_driver_seeds` package (`~/.venv-mjx`, float32, `--iterations 3`), pristine #11046 CLI vs this CLI: replay marker RMS and total cost identical at every iteration (65.16494, 52.73439, 54.64461, 50.29464 mm; 0.0 relative difference); 47 knots, 38 actuated coordinates.
- Tests: `tests/unit/engines/mujoco/test_mjx_knot_optimiser.py` (17 tests: settings validation, loader, missing file, missing root coordinate, `init_delta` shape, iteration callback, diagnose) and the evidence CLI test share `tests/unit/engines/mujoco/mjx_toy_package.py`; 29 passed in `~/.venv-mjx`, skip without JAX; the CLI test restores `jax_enable_x64` so later x64 tests are unaffected.
- Next step: none; merged as PR #11054.

---

# Past Handoff — MJX Evidence Prototype Rewired Onto the Tested `src` Plant (#11046)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11046-rewire`
- Branch: `claude/ud-11046-mjx-rewire` (baseline `origin/main`)
- Commit: merged to `main` as PR #11048
- Pull request: #11048; entry DL-#11046. Closes #11046. Package 2c of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `evidence/ground_support/mjx_trajectory_optimisation.py` is a thin CLI (648 → 362 lines) over `mjx_tracking_plant` (`TrackingPlantSpec.from_package`, `build_tracking_plant`, `reference_derivatives`, `substep_tables`), `jax_contact.WeldGains` and `knot_gradient_optimiser` (`knot_grid`, `knot_basis`, `horizon_knot_mask`, `AdamSettings`, `adam_minimise`). Every flag, output file and receipt key is kept.
- Orchestrator changes on review: the root vertical coordinate is looked up by name (`TranslationInputZ`) instead of a heuristic, and the new CLI test used a coordinate name the documents do not have.
- Measured parity on the regenerated `anthro_driver_seeds` package (`~/.venv-mjx`, float32), old prototype vs rewired: `--iterations 0` port check 65.16457 vs 65.16494 mm (5.6e-6 relative); `--iterations 3` best replay 50.249 vs 50.295 mm (9.2e-4 relative; float32 reordering differences grow per Adam step: 5.6e-6, 1.6e-5, 7.2e-5, 9.2e-4).
- Tests: `tests/unit/engines/mujoco/test_mjx_evidence_cli.py` (toy package, `--iterations 2`) passes in `~/.venv-mjx` and skips without JAX; ruff and format clean.
- Next step: CI green, mark ready and arm the PR.

---

# Past Handoff — PR-Scoped Tests That All Skip No Longer Fall Back to Whole-`src` Coverage (#11052)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11052-ci`
- Branch: `claude/ud-11052-ci-exit5` (baseline `origin/main`)
- Commit: merged to `main` as PR #11053
- Pull request: #11053; entry DL-#11052. Closes #11052. Workflow-only change, so it ships alone.
- Cause: in `ci-standard.yml` (Run Core Test Suite), a PR-scoped pytest run that collects nothing (exit 5, e.g. a module-level `importorskip("jax")`) fell through to the dependency-light lane with whole-`src` coverage and the 75 % floor. That lane covers about 12 % of `src`, so it failed every such PR (#11048: 2562 passed, "Total coverage: 11.73%").
- Built: when exit 5 happens and no `src`/dependency coverage target changed, the step now reports "Core test suite NOT EXECUTED" (warning plus step summary, the #8771 wording), sets `core_suite_executed=false` and `coverage_generated=false`, and exits 0. With changed source targets, the existing fallback is unchanged.
- Tests: `tests/ci/test_ci_infrastructure.py::test_all_skipped_selection_without_source_changes_is_not_executed` (red before, green after); `tests/ci/` 201 passed, 1 skipped (`--tools-mode vendored`).
- Next step: none; merged as PR #11053.

---

# Past Handoff — Canonical Calibrated Runs Regenerated on Current Code (#11044)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11044-canonical`
- Branch: `claude/ud-11044-canonical` (baseline `origin/main`)
- Commit: merged to `main` as PR #11050
- Pull request: #11050 — merged to `main` as PR #11050
- Finding: the canonical calibrated runs (driver 5.1 / 27.3 / 74.6 mm, 7-iron 4.4 / 28.6 / 144.0 mm) were produced with `BOUND_WIDENING = 2.0`, and both receipts record IK range-of-motion flags (10 and 9 coordinates). Measured bisect on `main` @ `3a11b5c07`, driver `--static-seeds`: current 7.9 / 34.1 / 84.5 mm; pre-HO-11 base spec identical; `BOUND_WIDENING = 2.0` 6.3 / 28.1 / 53.8 mm with 10 flags; hip zero-twist off 6.0 / 36.0 / 82.0 mm; all three reverted 4.9 / 25.7 / 55.1 mm. 7-iron `--static-seeds --zmp-filter`: 6.6 / 31.6 / 88.6 mm, and 4.1 / 23.5 / 53.4 mm with 9 flags at 2.0.
- Built: new canonical receipts `evidence/ground_support/anthro_driver_seeds` and `anthro_iron_seeds_zmp` (receipt plus final scaled spec, added to the provenance-chain test), `bisect_11044_receipt.json` generated from the run receipts, `CANONICAL_RUN.md` §2A/§2C/§3 revised, citations updated in `matched_swing_program/README.md`, `full_body_models/HANDOFF.md` and a correction note in `plans/tour_baselines/final_acceptance_report.md`; ledger and status section regenerated.
- Gate consequence: G1 (whole-swing IK <= 30 mm, 0 RoM violations) is not met by any MuJoCo receipt; the README gate row and the generator's MuJoCo status now say so. `verdicts_2026-09.json` is a dated record and was not edited.
- Next step: none; merged as PR #11050.

---

# Past Handoff — Ball-Flight Parity Fixture Export Is Opt-In (#11008)

---

# Implementation Handoff - Drift Wizard Sidekick Product Knowledge Pack (#10943)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-parity-fixture-optin`
- Branch: `fix/ball-flight-parity-fixture-opt-in` (kept current by merging `origin/main`)
- Implementation commit: `SELF`
- Pull request: #11012 — merged to `main` as PR #11012
- Governing issue: #11008 (DL-#11008); SPEC row keyed by PR #11012

## Objective and Status

- Objective: stop `test_export_reference_vectors` from rewriting the committed golden
  `tests/parity_fixtures/ball_flight/default_trajectory.json` on every Rust-enabled run.
- Done: vector construction is in `build_default_trajectory_vectors()`, the schema is in
  `assert_vector_schema()`, and writing goes through `write_vectors()` (LF, round-trip
  checked). The export test now writes to `tmp_path` and asserts that the committed bytes
  are unchanged. `test_regenerate_committed_fixture` rewrites the committed file only when
  `UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1` (exactly `1`).
- New `TestParityFixtureContract` (not Rust-gated, marked `unit`): checks the opt-in switch,
  the committed fixture's schema, the fixture path pinned in
  `src/config/capability_migration.json`, and that malformed vectors are rejected.
  `pytestmark` became a class-level `@requires_rust` so the contract tests run in every lane.
- The fixture bytes and the path are unchanged, so the sha256 pin stays valid.

## Validation

- `python -m pytest tests/parity/test_ball_flight_parity.py -o addopts=""` (Python 3.12, Rust
  kernel available, main merged through #11045): 19 passed, 1 skipped (opt-in regen).
  `git status` stays clean after the run.
- With `UPSTREAMDRIFT_REGENERATE_PARITY_FIXTURES=1 ... -k regenerate`: the fixture is rewritten (reverted).
- `ruff check` / `ruff format --check`, the suite-marker ratchet and the SPEC changelog check pass.

## Blockers and Risks

- The unit-test gate intermittently errors in `test_force_plate_stitching.py` (#11034, not
  this change); #11042 added diagnostics, the fix is pending.
- The committed golden is stale against the current model: #11004 (merged) did not regenerate
  it, so it still records carry 187 m against 243 m from the current model. No test compares
  values against it. Regenerating it and updating the sha256/size pin is a follow-up.

## Next Steps

1. Keep the branch current with `origin/main` until auto-merge lands it.
2. After merge, file or pick up the fixture regeneration follow-up.

---

# Past Handoff — Replays Start on the Dual-Grip Weld (#11043)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11043-dls`
- Branch: `claude/ud-11043-dls` (baseline `origin/main`)
- Commit: merged to `main` as PR #11047
- Pull request: #11047; entry DL-#11043. Closes #11043.
- Root cause: the KKT solve enforces the weld only at acceleration level (`J a = -J̇ q̇`), so it conserves the relative weld velocity it starts with. `replay` seeded `v0` with the one-sided finite difference of the smoothed reference, which violated the weld by 187 mm/s and 32.1 deg/s on the `--fit-closure` run (17.6 mm/s on `--static-seeds`). The grip opened linearly, gave the weld Jacobian nonzero root columns, and a controller direction crossed `MIN_SINGULAR_VALUE` at t = 0.333 s: 9.5 kN·m on `RScapInputY` in one step.
- Built: `motion_matching/weld_manifold.py::project_onto_weld` (mass-weighted projection, the weld's own inelastic impulse), `FullBodySimulator.consistent_velocity`, with `_mass_and_weld` extracted from `affine_dynamics`. `pipeline/dynamics.replay` and `execution/downswing` now start from the projected velocity.
- Measured with the full pipeline on this branch (driver, DeskComputer, Python 3.12): `--static-seeds --fit-closure` 861.2 → 83.0 mm dynamics RMS, weight fraction max 50.0 → 3.72, peak torque 29 968 → 1 124 N·m, support 0.50 → 0.98, backswing root 493 → 8 mm. `--static-seeds` control 83.0 → 84.5 mm, weight fraction 3.44 → 3.76, peak torque 719 → 912 N·m, support 0.95 → 0.96.
- Tests: `tests/unit/motion_matching/test_weld_manifold.py` (6), plus `test_full_body_simulation.py`, `pipeline/test_dynamics.py` and the downswing tests: all pass. Reverting the `replay` line fails `test_replay_starts_on_the_weld`.
- Not done: the committed evidence receipts were not regenerated; the pipeline still has no fail-closed guard on a diverged replay (the issue's fallback acceptance).
- Pre-push mypy caught an un-annotated `Array` alias in `weld_manifold.py`; it is now `TypeAlias`.
- Next step: none; merged as PR #11047.

---

# Past Handoff — Force-Plate Fixture Reports Module Identity on the Intermittent Failure (#11034)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11034-fp`
- Branch: `claude/ud-11034-force-plate-flake` (baseline `origin/main`)
- Commit: merged to `main` as PR #11042
- Pull request: #11042; entry DL-#11034. Refs #11034; the issue stays open for the root cause.
- Built: `tests/unit/sidekick/lab/bio/test_force_plate_stitching.py` fails with `_module_identity_report` (resolved file and spec, module names, `sys.meta_path`, related `sys.modules` entries) instead of a bare `AttributeError` when the import resolves to a module without `CombinedForcePlateProcessor`.
- Not reproduced locally (Windows, Python 3.12, CI `PYTHONPATH`): the file alone, after `test_sidekick_extension_overlay.py`, and pairwise after every test that rewires `sys.meta_path` or `shared.*`. Two re-runs of the Linux `unit-test-gate` on this PR passed, and `main` has been green since `c2e6fe878`.
- Suspect: the overlay test's synthetic `force_plate_stitching` (canonical name, no class) combined with `SharedImportAliasFinder._find_canonical_spec` re-inserting itself at `meta_path[0]` and `_CanonicalAliasLoader` writing six alias spellings.
- Next step: none; merged as PR #11042.

---

# Past Handoff — MJX Tracking Plant and Differentiable Rollout (#11039)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11039-plant`
- Branch: `claude/ud-11039-mjx-plant` (baseline `origin/main`)
- Commit: merged to `main` as PR #11040
- Pull request: #11040; entry DL-#11039. Package 2b of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/engines/physics_engines/mujoco/python/motion_matching/mjx_tracking_plant.py` (imports JAX and MJX; no `__init__` imports it): `TrackingPlantSpec` (explicit `root_vertical_index`, `validate_against_model` refuses equality constraints and out-of-range ids), computed-torque control over the reference, sphere-ground and grip-weld wrenches from `jax_contact`, substep/frame/rollout, and `initial_state` that seats the model at static penetration under `model.opt.gravity`.
- Orchestrator changes on review: `from_package` now requires `substeps` and `root_vertical_index` (the heuristics were removed) and raises unconditionally when the package declares a grip closure without `weld_gains`; `build_tracking_plant` works on a copy, so the caller's `model.opt.timestep` is untouched; `rollout` reuses `rollout_diagnostic`.
- Location: the engine package, not `src/shared/python/motion_matching/`, because the shared tree may not import `mujoco` (`test_import_graph_no_mujoco_in_shared_motion_matching`); the JAX contact laws it uses stay shared.
- Measured in `~/.venv-mjx` on the toy model: computed-torque residual 7.1e-15, marker RMS 0.28 mm over 20 frames without contact, rollout gradient against central differences 2.2e-9 relative.
- The evidence prototype is still untouched; package 3 benchmarks the plant on an exported `mjx_package.npz` and decides promotion.
- Next step: none; merged as PR #11040.

---

# Past Handoff — Differentiable JAX Contact Law and Grip Weld (#11037)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11037-contact`
- Branch: `claude/ud-11037-jax-contact` (baseline `origin/main`)
- Commit: merged to `main` as PR #11038
- Pull request: #11038; entry DL-#11037. Package 2a of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/shared/python/motion_matching/jax_contact.py` (imports JAX; no `__init__` imports it): `sphere_ground_contact_jax` over the shared `ContactParameters`/`GroundPlane` with an explicit tangential-speed floor, and `weld_wrench_jax` with caller-supplied `WeldGains`.
- Measured parity against `contact_law.sphere_ground_contact` on 2000 seeded states: normal 0.0 N, friction 1.0e-11 N maximum difference (float64).
- The tests run only where JAX is installed: `~/.venv-mjx` from `scripts/setup_mjx_env.ps1` now exists on DeskComputer (jax 0.11.1, mujoco 3.13.0); the default env and CI skip them.
- Next step: none; merged as PR #11038.

---

# Past Handoff — JAX-Free Knot Basis and Adam Driver for the MJX Solver (#11032)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11032-knot`
- Branch: `claude/ud-11032-knot-adam` (baseline `origin/main`)
- Commit: merged to `main` as PR #11035
- Pull request: #11035; entry DL-#11032. Package 1 of epic #11006. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: `src/shared/python/motion_matching/knot_gradient_optimiser.py` (numpy only): `knot_grid` (integer knot count, so no untouched trailing knot as the prototype's float `arange` could give), `knot_basis` (vectorised hat functions, refuses untouched knots), `horizon_knot_mask`, keyword-only `AdamSettings`, and `adam_minimise` over an array namespace `xp` with best-by-objective tracking and non-finite stops.
- Orchestrator rewrite on review: `AdamSettings` was a hand-parsed `*args` initialiser and is now a keyword-only frozen dataclass; `adam_minimise` copied nothing, so freezing `best_x` could freeze the caller's `x0` (now copied, with a test).
- The evidence prototype is untouched; package 2 (MJX plant adapter, needs `scripts/setup_mjx_env`) will reuse this driver.
- Next step: none; merged as PR #11035.

---

# Past Handoff — White-Jerk RTS Kinematic Smoother With Uncertainty (#11029)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11029-rts`
- Branch: `claude/ud-11029-rts-smoother` (baseline `origin/main`)
- Commit: merged to `main` as PR #11031
- Pull request: #11031; entry DL-#11029. Package 5 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/estimation/kinematic_smoother.py` (exact white-jerk discretisation, vectorised Kalman filter plus RTS pass per coordinate, NaN = missing frame, log marginal likelihood, per-coordinate ML noise fit seeded from finite differences) and `smooth_reference_bayesian` in `motion_matching/pipeline/reference.py`. `smooth_reference` is unchanged and pinned bit-for-bit against `butter(4)` + `filtfilt`.
- Orchestrator hardening on review: the initial mean and covariance must be given together and in shape, infinities are refused (only NaN means missing), and result and noise arrays are read-only.
- Validation: see DL-#11029. The 18 local failures in `tests/unit/motion_matching` (event alignment, Rust parity/bench, surrogate training and others) fail identically on clean `origin/main`.
- Next step: none; merged as PR #11031.

---

# Past Handoff — Fitted Sparse Residual for the Physics-Structured Surrogate (#11024)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11024-residual`
- Branch: `claude/ud-11024-sparse-residual` (baseline `origin/main`)
- Commit: merged to `main` as PR #11025
- Pull request: #11025; entry DL-#11024. Package 2 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/neural_motion/surrogates/sparse_residual.py` (numpy-only STLSQ, `CandidateLibrary`, `SparseResidualFit`); `PhysicsStructuredSurrogate` takes an optional fitted residual and refuses without one. The hand-chosen `0.02*tanh(0.5*prior)` residual is deleted, not kept as a fallback.
- Changed existing test: only `test_forward_surrogates_nm07.py::test_physics_structured_surrogate_prior_and_residual`, which asserted the invented constant; it now asserts the refusal and still checks the prior's shape.
- The STLSQ loop is its own helper `_stlsq` so `fit_sparse_residual` stays inside the 100-line architecture budget.
- CI gates fixed on replay: `setflags(write=False)` instead of the three-level `.flags.writeable` chain (lod-quality-gate), and the fail-closed refusal in `residual_correction` names its tracking issue on the same line (stub-introduction guard).
- Validation: see DL-#11024. Known local failures that also fail on clean `origin/main`: `test_artifact_identity_dbc`, `test_nm01_dbc_optimize` (1 each) and `test_swing_surrogate_training` (4).
- Next step: none; merged as PR #11025.

---

# Past Handoff — Parameter Covariance on the Shared Least-Squares Fits (#11021)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11021-covariance`
- Branch: `claude/ud-11021-fit-covariance` (baseline `origin/main`)
- Commit: merged to `main` as PR #11023
- Pull request: #11023; entry DL-#11021. Child of MS-101 #10375 (Board RM#1793). Executed by agy (Gemini 3.8 Flash), reviewed line by line; the orchestrator replaced the duplicated call-site guard with one tested helper (`fitted_uncertainty_or_none`).
- Built: `src/shared/python/estimation/fit_uncertainty.py`; prefix and multiple-shooting fits report parameter uncertainty.
- Validation: see DL-#11021. `residual_regularization.Array` is now a declared `TypeAlias`: the pre-push mypy env has no numpy, and the bare alias failed as `Array?` in any change to `multi_shooting_fit.py`.
- CI `repo-structure-gates` flagged `least_squares_parameter_uncertainty` at 182 lines (budget 100). It is now split into single-purpose helpers (`_validated_jacobian`, `_requested_indices`, `_free_mask`, `_rank_and_condition`, `_free_covariance`, `_marginal_statistics`) with unchanged behaviour: the 81 estimation tests and the wiring tests pass unmodified, and `check_architecture_budget.py` is OK.
- Next step: none; merged as PR #11023.

---

# Past Handoff — Canonical Swing Event Detector (#11014)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11014-events`
- Branch: `claude/ud-11014-swing-events` (baseline `origin/main`)
- Commit: merged to `main` as PR #11020
- Pull request: #11020; entry DL-#11014. Package 1 of epic #11007. Executed by agy (Gemini 3.8 Flash), reviewed line by line and replayed into this worktree.
- Built: `src/shared/python/analysis/swing_events.py` (one detector plus `peak_speed_index`); four call sites delegate to it; parity and contract tests added.
- Validation: see DL-#11014 (374 passed; plus 188 passed across reconstruct, statistical-analysis, advanced-analysis and swing-comparison consumers).
- Next step: none; merged as PR #11020. Package 2 is #11024.

---

# Past Handoff — Retract the NM-09/NM-12 DIAGNOSTIC Receipt Claims (#10960)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-10960-receipts`
- Branch: `claude/ud-10960-diagnostic-receipts` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11017 (merged); entry DL-#10960. Executed by agy (Gemini 3.8 Flash), reviewed line by line.
- Built: the committed NM-09 checkpoint-matrix receipt lists no `qualified_native` models (the three move to `unqualified`) and its limitation says native ODE replay is not verified; the NM-12 turnover receipt promotes no models (the three move to `unmeasured_models`), `end_to_end_verification.status` is `not_verified` and `all_issues_completed` is false. New `tests/unit/neural_motion/test_diagnostic_receipts_10960.py` fails on the old receipts: a DIAGNOSTIC receipt may not qualify or promote anything, and no nested status except `validation.outcome` may read `passed`.
- Validation: `pytest tests/unit/neural_motion/ --deselect test_artifact_audit.py` -> 155 passed (the artifact audit times out locally on a 9.1 GB parquet on disk; CI runs it).
- Next step: CI green, mark ready, arm; then close #10960 with evidence (all code slices already on main).

---

# Past Handoff — Rust Kernel in the Linux Unit-Test Gate (#9411)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-gate-rust`
- Branch: `claude/ud-unit-gate-rust-kernel` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #10997 (draft), workflow change shipping alone; entry DL-#9411.
- Built: `ci-standard.yml` `unit-test-gate` installs the pinned Rust toolchain, caches the Cargo registry, maturin-builds only `rust_core/upstream-physics` into the unit-gate venv and runs the fail-closed `import_built_rust_wheels.py upstream_physics` probe before pytest; timeout 25 -> 35 min. The 12 BunkerShot workbench/GUI IDs leave `unit_gate_quarantine.json` (67 -> 55 on main after #11010) because `ball_simulator` raises without the kernel.
- Validation: `pytest tests/ci/test_unit_gate_rust_kernel.py tests/ci/test_ci_infrastructure.py tests/unit/repo_hygiene/test_hygiene_guards_run_in_ci.py` -> 111 passed, 1 skipped (new file fails 3/3 against the old workflow); `tests/ci/test_unit_gate_quarantine_contract.py` -> 11 passed; the 12 retired tests pass locally with the kernel installed. Linux proof is this PR's unit-test-gate run.
- First Linux run (0f01454cf): the kernel built and imported, and none of the 12 retired IDs failed. Six tests that the kernel un-skipped failed. Root `Cargo.lock` clutter came from this workflow and is fixed (the untracked lock is removed after the build). Rust vs enhanced carry and TrackMan windows = #11000, fixed in #11004. The degrees-regression test is updated in #11004. The two #9243 uncertainty claims = #11003 (tier:strong).
- Unblocked: #11004 and #11010 (#11003) merged; replayed onto main d716c698f.
- Next step: unit-test-gate green on the replay, then mark ready and arm via `automerge_guard.py`.

---

# Past Handoff — Restate the #9243 BunkerShot Uncertainty Claims (#11003)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-11003`
- Branch: `claude/ud-9243-claims-restated` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11010 (draft); entry DL-#11003. Unblocks UD#10997 (Rust kernel in the unit gate).
- Decision (tier:strong, owner delegated the judgement 2026-09-26): both failures are stale claims, not model defects. (1) The window-band fixture aimed at a fixed 2 m target; the corrected F0 model's grid carries are 0.05-0.66 m (nominal 0.690 m), so the window was empty on all three grids. The fixture now targets the nominal shot's own carry and asserts a non-empty window; the band measures (0.0, 1.141, 2.282) and the not-decorative claim holds. (2) The accelerated-mass share is 0.676 < `DOMINANCE_SHARE` 0.75: it dominates but does not swamp. The test now states that; the threshold is unchanged.
- Validation: `tests/unit/tools/bunker_shot_gui/test_uncertainty_propagation_9243.py` 30 passed with upstream-physics 2.1.3 built from source, and 30 passed with the user-site 2.1.0 wheel.
- Known: an absolute nominal carry of 0.69 m is short for a greenside splash shot; it sits inside the named uncalibrated transfer-efficiency gap (TestHonestyBoundary), not this PR.
- Next step: CI green, mark ready, arm, then replay #10997.

# Past Handoff — Rust Ball-Flight Kernel Base Cd (#11000)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-rust-drag`
- Branch: `claude/ud-rust-drag-base-cd` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #11004 (draft); entry DL-#11000. Found by UD#10997 (Rust kernel in the unit gate), which is blocked on this PR and on #11003.
- Built: `ball_simulator.py` passes `GOLF_BALL_DRAG_COEFFICIENT` (0.25) instead of `ball.cd0` (0.21) as the kernel's Reynolds-curve base Cd. New kernel-independent boundary test with a fake `upstream_physics`. `test_raw_degrees_would_collapse_carry` became `test_raw_degrees_are_refused_before_they_collapse_carry` (the launch contract now raises). The exported parity fixture `default_trajectory.json` is left untouched: `src/config/capability_migration.json` pins its byte hash (`test_green_fixture_byte_hashes_unchanged`).
- Validation: see DL-#11000. Local kernel = upstream-physics 2.1.3 built with maturin into a scratch venv; the user-site 2.1.0 wheel on this box is stale and gives different numbers.
- Known: `tests/parity/test_ball_flight_parity.py::test_export_reference_vectors` rewrites the committed, hash-pinned fixture on every run where the kernel is present (pre-existing; follow-up task filed). Restore it with `git checkout -- tests/parity_fixtures` after local runs.
- Next step: CI green, mark ready, arm, then rebase #10997 and fix its untracked root `Cargo.lock` (root-clutter gate).

# Past Handoff — #9411 Unit-Gate Quarantine Burn-Down (Slice 2) 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-9411-probe`
- Branch: `claude/ud-9411-unquarantine-passing`
- Baseline commit: `b8c27a7d2` (origin/main, after #10988)
- Implementation commit: `SELF`
- Pull request: #10990 (draft)
- Governing issue: #9411 (DL-#9411)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: retire unit-gate quarantine IDs whose tests pass for the right reason.
- Status: 54 IDs retired (155 → 101). The second Linux run re-quarantined the order-dependent
  `test_install_prompt` worker test and replaced a module `reload` in `test_synthesize_target` (it
  broke the pinocchio_golf facade identity test) with an AST check. The first push retired 90, but Linux CI failed 24 of
  them (GL-less mujoco, missing trimesh, bunker GUI carry, force-plate ownership guard) and the
  child-copy contract flagged the `ai/adapters` edits. `ai/adapters` and the gear-effect sign flip
  were reverted to main (the flip contradicted `test_impact_physics_value_assertions`: toe impact
  gives draw spin, so the two `test_impact_model` gear tests need a convention decision, not a
  sign change); their 11 dependent IDs and the 24 Linux failures are re-quarantined.
- Slices (agy on DeskComputer and OG Laptop, reviewed and trimmed by Claude):
  - validation-contracts: shoulder FK origin (the gear-effect sign flip was reverted).
  - signal-hygiene: launcher test drift.
  - ai-gui-adapters: lazy API routers; theme router prefix (the `ai` adapter shadow edits were
    reverted: Tools child copies may only be deleted, never edited).
  - data-fitting split: `data_fitting.py` 1081 → 479 LOC by removing classes duplicated in
    `_data_fitting_models.py` / `_data_fitting_solvers.py`. The agy `theta_optimal` alias was
    dropped; `test_init_dempster` asserts `coefficients` again (a #4273 blind rename).
  - mujoco-viewer split: coordinator duplicates removed. The agy `sys.modules` mock lookup was
    replaced by patching `_mujoco_viewer_backend` in `test_model_explorer_temp_file.py`.
  - sidekick-drift subset: Simscape C3D embed adapter clears its widget on cleanup; test-only
    retargets. Shadow deletions, the `ai` panel growth and the bootstrap path flip were held
    back for a dedicated #9406 slice.
- Reverted: edits to Tools-owned shadow files (`signal_toolkit` guards, `model_generation`
  gravity and positive-mass checks, `sidekick/data_processing`, `ai/gui/session_manager` and
  `assistant_panel`, `humanoid_character_builder`). Each grew the file's drift from vendor; the
  fixes belong in Tools. 11 IDs that only passed because of them are quarantined again.
- Rejected: deleting `tests/unit/test_safe_eval.py`. Its five failures are real (Tools#5360,
  fixed in Tools#5361); the IDs stay quarantined until the vendor pin moves.

## Validation

- Quarantine rerun after the revert: `pytest -n 6 -o addopts="" --tools-mode vendored <101
candidate IDs>` → 90 passed, 11 failed (re-quarantined).
- `scripts/ci/check_unit_gate_quarantine.py`: contract passed, 65 IDs in 10 clusters;
  `tests/ci/test_unit_gate_quarantine_contract.py`: 11 passed.
- Data-fitting tests (4 files): 72 passed. Viewer, embed-adapter, launcher and sidekick tests:
  234 passed, 4 failed (all 4 still quarantined).
- `ruff check` / `ruff format --check` on changed files: clean. 29 local repo-structure steps pass.

## Blockers and Risks

- Remaining 65 IDs: inertia API (#6995 design, 10), `safe_eval` (Tools#5360, 5), sidekick
  shadow retirement (#9406), signal_toolkit limits/core, `test_level` (`-O` contract level
  design), size budgets for Tools-owned files, Simscape `ezc3d`.
- Order-dependent failures in the remaining set come from `sys.modules` pollution, not the DbC
  level: `tests/unit/sidekick/standalone/test_session_store.py` makes later tests load the Tools
  copies (gravity 9.81, no guards). The #9406 shadow retirement removes that test's target.
- `check_architecture_budget.py` fails on unmodified main (`validate_inertia_tensor`, 101 lines).
- Rebase onto main after #10988: HANDOFF conflicts (take this file), DL/SPEC keep both rows.

## Next Steps

1. After #10988 merges, rebase, open the draft PR, add the SPEC row keyed by its number.
2. Get `quality-gate` green; mark ready; arm through `scripts/automerge_guard.py`.
3. Comment on #9411 with the remaining clusters and owners.

---

# Past Handoff — Bump `vendor/ud-tools` to Tools Main `3678409fc` (#9411)

- **Branch:** `claude/ud-9411-vendor-bump-safe-eval`; PR #10995, pairs with Tools#5364 (`UD-PAIR`).
- **Change:** gitlink 95ed6b478 → 3678409fc; `Cargo.toml`, `requirements-tools.txt`, `src/config/impact_acceptance.json`, `reconciliation.py` and its test aligned; converged child copies in `src/shared/python/` on canonical Tools; retired 10 quarantined tests in `scripts/config/unit_gate_quarantine.json` (ratchet 155 -> 145 node IDs); divergence inventory and agent context regenerated.
- **Validation:** companion (143 passed), reconciliation (3 passed), 10 un-quarantined tests passed, child-copy contract (20 passed), quarantine ratchet (145 IDs in 10 clusters), divergence inventory and `agent_context check` clean.
- **Next:** verify pre-push checks, push, run CI Standard, verify quality gate, and arm auto-merge.

---

# Past Handoff — Consolidated Bolt Norm Micro-Optimizations (#10983, #10984)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-bolt-consol`
- Branch: `claude/ud-bolt-consolidated-20260926` (baseline `origin/main`)
- Commit: `SELF`
- Pull request: #10993 (draft); supersedes #10983 and #10984 (both conflicted on their SPEC rows)
- Issue: none (Bolt performance PRs). No material development-log change — two behaviour-preserving single-expression rewrites.
- Built: `fit_pipeline._quality` computes landmark distances as `sqrt(einsum)` (#10984); `shot._rotation_increment` computes the angular-velocity norm once with `math.sqrt(np.vdot)` and reuses it for the axis (#10983). Inline comments shortened to the 88-column limit; the Bolt journal entry is kept with its date corrected to 2026-09-26.
- Validation: `pytest tests/motion_capture/test_reference_fit_pipeline.py tests/motion_capture/test_reference_fit_preview.py tests/bunkershot3d/solvers/test_shot.py` -> 45 passed; ruff check/format clean.
- Next step: CI green, mark ready, arm via `automerge_guard.py`, then close #10983 and #10984 as superseded.

---

# Past Handoff — Main Red on Bandit B314 in the Coverage Gate Checker (#10989)

- Repository: D-sorganization/UpstreamDrift
- Worktree: `UpstreamDrift-worktrees/claude-ud-10989`
- Branch: `fix/10989-coverage-gates-defusedxml` (baseline `84ac37579`)
- Commit: `SELF`
- Pull request: #10994 (draft)
- Issue: #10989 (fleet-main-health: CI Standard red on main); development-log entry DL-#10965
- Cause: #10988 landed `scripts/check_coverage_gates.py` parsing Cobertura XML with `xml.etree.ElementTree.parse`; the push-lane full-tree `bandit -ll -ii` flags B314 and fails `security-scans`, which fails `quality-gate`.
- Fix: parse with `defusedxml.ElementTree` (a core dependency, the convention in `scripts/config/coverage_enforcer.py`); `Element` imported under `TYPE_CHECKING` for the annotation. New test `test_xml_entity_expansion_is_rejected` was red on stdlib ET and is green now.
- Validation: `pytest tests/unit/scripts/test_check_coverage_gates.py` -> 13 passed (new test carries `@pytest.mark.unit` for the suite-marker ratchet); `bandit -ll -ii scripts/check_coverage_gates.py` clean; ruff and mypy clean.
- Next step: CI green, mark ready, arm via `automerge_guard.py`; #10989 closes itself when main's next CI Standard run succeeds.

---

# Implementation Handoff — #9406 Sidekick Shadow Retirement 2026-09-26

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/claude-ud-9406-shadow`
- Branch: `claude/ud-9406-sidekick-shadow-retire`
- Baseline commit: `b8c27a7d2` (origin/main, after #10988)
- Implementation commit: `SELF`
- Pull request: #10991 (draft)
- Governing issue: #9406 (DL-#9406)
- Lease session: `claude-deskcomputer-20260925-ud`

## Objective and Status

- Objective: stop UpstreamDrift shipping stale copies of Tools-owned `sidekick` modules, so the
  quarantined tests that demand their absence pass.
- Status: agy (Gemini 3.8 Flash) slice on DeskComputer, reviewed and trimmed by Claude.
  - Deleted, now resolved from `vendor/ud-tools/src/shared/python/sidekick/`: `standalone/`
    (runner, window, session_store, preferences, onboarding), `persistence/`, `__main__.py`,
    `ui/tools_sidebar/default_tabs.py`.
  - `embedded_tool_bootstrap._bootstrap_python_paths`: vendored or explicit Tools paths now
    precede UpstreamDrift's own, as the bootstrap tests require.
  - `sidekick.spec`: Windows icon is the pinned `vendor/ud-tools/assets/tools_icon_hq.ico`.
  - Tests retargeted to the vendor API: pickle is no longer auto-detected (`test_data_io`), C3D
    export needs `C3D_ALLOW_ANY_EXPORT_PATH` (`test_c3d_reader`), and `test_cli` patches the
    launcher factory under every import prefix.
  - 14 quarantine IDs retired (155 → 141).
- Dropped from the agy output: an `architecture_budget.json` exception (the violation predates
  this branch) and an `ai/gui/assistant_panel.py` vendor copy (the backward-compat identity tests
  still fail because vendor imports through the `shared.python` prefix).

## Validation

- `pytest -n 6 -o addopts="" --tools-mode vendored <33 sidekick/packaging/bootstrap IDs>`:
  14 passed (all retired); the rest stay quarantined.
- `UNIT_GATE_QUARANTINE=1 pytest -n 6 --tools-mode vendored tests/unit/sidekick tests/unit/launcher
tests/launchers tests/unit/packaging tests/unit/repo_hygiene tests/integration/sidekick
tests/ui/c3d_viewer`: 67 failed / 2686 passed here vs 65 / 2687 on `b8c27a7d2`; the only new
  ID (`test_embed_adapter::test_create_main_widget_returns_qwidget`) passes alone twice
  (xdist order). The #9406 slice also fixed one main failure.
- `scripts/ci/check_unit_gate_quarantine.py`: contract passed, 141 IDs.

## Blockers and Risks

- The same launcher/launchers failures appear on main in this Windows environment.
- #9411 (quarantine burn-down) edits the same ledger; whichever lands second takes the union
  of removals.

## Next Steps

1. Open the draft PR, add the SPEC row keyed by its number, get `quality-gate` green.
2. Mark ready and arm through `scripts/automerge_guard.py`.

## Motion Matching Review Packet

Owner-requested review on `docs/motion-matching-board-review`, PR #11083. See
[Board Packet](2026-09-28-motion-matching-board-review.md): 18 draft issue bodies,
main and GS3DX branch evidence, metrics and licensing budgets, and Shadow Tracker
integration/qualification gaps. No implementation issues claimed or closed.
375 focused tests and full Ruff lint/format passed. Board approval and native
qualification are separate next steps; preserve the active #10979 work.

## Current Coordination Limits

Issue leases succeeded. The presence inbox reported incomplete board evidence
(page limit and malformed comments); absence of messages is not evidence that
the repository is unoccupied. This owned isolated worktree preserves all others.
The original root handoff already exceeded its 150-line guideline; unrelated
active sections were preserved. No full-repository pytest/coverage run was made;
364 scoped tests and all configured pre-push checks (including mypy, Bandit and
core/DbC/utils tests) passed. A guessed SPEC test path was absent; no SPEC-test
pass is claimed. Required commit and design-manual governance hooks passed.

## Publication Refresh

Merged origin/main 51a0c1bfa4 into the owned branch without conflicts, retaining
both SPEC rows. Post-merge focused Shadow Tracker tests are being verified;
source/model receipts remain unchanged. Pre-push checks must pass on the merge.

## Additional Hogan Source

Owner-requested DJDYMjmvFwg was downloaded with yt-dlp, including audio and
metadata: 10 Minutes of Ben Hogan (Every Angle Ever Recorded), Sonic Titan Golf,
10:23, 1920x1080 at 60 presentation fps. SHA-256 is recorded in source-catalog.json.
Original archive cadence, individual recording dates and film overlap are unknown;
this compilation must not be treated as synchronized multiview or independent
held-out footage. Local media: Downloads/historical-capture/ben_hogan/.

The new Hogan compilation window 253-267 presentation seconds was processed:
839 frames, 769 detections and 70 explicit missing detections. Receipt and hashes
are committed; source-bound frames and observations remain outside Git. This
window is unreviewed and may cross cuts; 60 presentation fps does not establish
original film timing or independent multiview.

The first post-merge push was stopped because documentation changed while the
security hook was running (no security issues were identified). Finish the
current documentation commit and retry from a clean worktree.
