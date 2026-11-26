# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-12 by claude
- **Current metadata review:** Last verified commit identifiers below bind path inventory only to e68baeabde8ff639f846a167ca7703ace43bd0a9; no tests or native runs were performed. Full original row text before narrative condensation is retained in [the pre-compression snapshot](evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_PRE_COMPRESSION_20261011.md.snapshot). Prior verification wording and per-path Git blob identities are preserved in [the source metadata audit](evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json).

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#15833 · Add Source-Attributed Legend Appearance Candidates

- **State:** in_progress
- **Owner:** Codex
- **Issue:** #15833
- **Branch:** codex/15833-legends-profiles
- **PR:** pending
- **Paths:** player appearance catalog, profile accessor and contract tests; morphology reference
- **Started:** 2026-11-26
- **Summary:** Add five source-attributed, stylized golfer profiles; preserve measurement conflicts and visual limits.
- **Next step:** Publish core candidates; final delivery #15836 adds Snead's straw_hat after #15838 lands.

### DL-#15115 · Local API App Applies the Shared Allowed-Host Policy

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15115
- **Branch:** claude-og/local-server-host-check
- **PR:** #15116
- **Paths:** see #15116
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`795caaad`; collated from changes/15115-local-api-app-applies-the-shared-allowed.md)
- **Summary:** Local API app applies the shared allowed-host policy
- **Next step:** Merge the PR.

### DL-#15112 · Fail Closed on Per-Crate Maturin Build Errors in Dockerfile and Enforce in check_dockerfile_contracts.py.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15112
- **Branch:** gemini-og/15112-dockerfile-maturin-fail-closed
- **PR:** #15114
- **Paths:** see #15114
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`7699de0c`; collated from changes/15112-dockerfile-maturin-fail-closed.md)
- **Summary:** Fail closed on per-crate maturin build errors in Dockerfile and enforce in check_dockerfile_contracts.py.
- **Next step:** Merge PR for #15112.

### DL-#15107 · Fix Broken Repo Paths in Curated Testing, Architecture, and Manual Docs and Enforce in check_spec_paths.py.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #15107
- **Branch:** gemini-og/15107-docs-broken-repo-paths
- **PR:** #15113
- **Paths:** see #15113
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`6b454ac0`; collated from changes/15107-docs-broken-repo-paths.md)
- **Summary:** Fix broken repo paths in curated testing, architecture, and manual docs and enforce in check_spec_paths.py.
- **Next step:** Merge PR #15113.

### DL-#13888 · Add HILL_MTU Drive Mode to 3DOF FTLE Study With Matched Nominal Excitation and Four-Mode Receipt.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13888
- **Branch:** cursor/hill-mtu-drive-mode-3dof-ftle-f296
- **PR:** #15076
- **Paths:** see #15076
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`1b4ce670`; collated from changes/13888-hill-mtu-drive-mode-3dof-ftle-study.md)
- **Summary:** Add HILL_MTU drive mode to 3DOF FTLE study with matched nominal excitation and four-mode receipt.
- **Next step:** Merge after CI; full t_end=0.2 Hill receipt rows remain NOT_QUALIFIED until MTU equilibrium converges on full swing.

### DL-#13782 · ML-WS-5 MuJoCo Corpus Run Receipt, Manifest-Verified Loader, and CI Evidence

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13782
- **Branch:** cursor/ml-ws5-corpus-receipt-loader-791f
- **PR:** #15075
- **Paths:** see #15075
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`0adc2797`; collated from changes/13782-ml-ws-5-mujoco-corpus-run-receipt-manife.md)
- **Summary:** ML-WS-5 MuJoCo corpus run receipt, manifest-verified loader, and CI evidence
- **Next step:** Barb review / CI

### DL-#13761 · Bump Dnspython to 2.9.2 and Tornado to 6.6.1; Remove Stale Tornado Pip-Audit Waivers.

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13761
- **Branch:** cursor/pip-audit-dnspython-tornado-54d2
- **PR:** #15086
- **Paths:** see #15086
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`b6abbfbf`; collated from changes/13761-bump-dnspython-to-2-9-2-and-tornado-to-6.md)
- **Summary:** Bump dnspython to 2.9.2 and tornado to 6.6.1; remove stale tornado pip-audit waivers.
- **Next step:** Merge after CI green.

### DL-#13427 · Regression-Test Address Reserve Share; Wire Study Allow_Implausible and OS-9 Asset Gate

- **State:** in_review
- **Owner:** unassigned
- **Issue:** #13427
- **Branch:** cursor/spn-d9-address-reserve-share-ab3e
- **PR:** #15077
- **Paths:** see #15077
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`d56bdccf`; collated from changes/13427-regression-test-address-reserve-share-wi.md)
- **Summary:** Regression-test address reserve share; wire study allow_implausible and OS-9 asset gate
- **Next step:** Owner decides address trunk bound; merge qualified OS-9 native receipt when measured

### DL-#15045 · Hill F_V Zero Beyond Vmax; Exponential Activation Substeps for Mocap Dt

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15045
- **Branch:** cursor/hill-activation-sci-15045-1ec6
- **PR:** #15054
- **Paths:** see #15054
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`59cb1fcb`; collated from changes/15045-hill-f-v-zero-beyond-vmax-exponential-ac.md)
- **Summary:** Hill f_v zero beyond Vmax; exponential activation substeps for mocap dt

### DL-#15044 · SO(3) Quat_Inverse_Distance (Relative-Quaternion Angle), Interpolated Ball-Flight Ground Landing

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15044
- **Branch:** cursor/quat-landing-sci-15044-e727
- **PR:** #15056
- **Paths:** see #15056
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`67fef889`; collated from changes/15044-so-3-quat-inverse-distance-and-interpola.md)
- **Summary:** SO(3) quat_inverse_distance (relative-quaternion angle), interpolated ball-flight ground landing

### DL-#15043 · Align Rigid-Body Off-Center COR Impulse With Validator Budget via Club Torque; Solver Integration Test.

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15043
- **Branch:** cursor/validate-energy-balance-15043-4fad
- **PR:** #15051
- **Paths:** see #15051
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`37a0e84f`; collated from changes/15043-energy-validator-uses-effective-club-mas.md)
- **Summary:** Align rigid-body off-center COR impulse with validator budget via club torque; solver integration test.

### DL-#15040 · Fix Turf Contact Impact: Joule Energy Loss, Receding Guard, Coupled Friction-Spin; Release-Safe Mass/Radius Validation

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15040
- **Branch:** cursor/turf-contact-impact-15040-eeaa
- **PR:** #15048
- **Paths:** see #15048
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`b2212bc8`; collated from changes/15040-fix-turf-contact-impact-joule-energy-los.md)
- **Summary:** Fix turf contact impact: joule energy loss, receding guard, coupled friction-spin; release-safe mass/radius validation

### DL-#15022 · Agent API V1 Conformance CI and FastMCP Migration for Ud-Sim-Mcp

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15022
- **Branch:** cursor/agent-api-v1-fastmcp-15022-1251
- **PR:** #15073
- **Paths:** see #15073
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`808174d4`; collated from changes/15022-agent-api-v1-fastmcp.md)
- **Summary:** Agent API v1 conformance CI and FastMCP migration for ud-sim-mcp

### DL-#15015 · Fail-Closed Motion-Matching Gates; Archive Run-102 Evidence Script With Restore Ref.

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #15015
- **Branch:** cursor/motion-matching-gate-honesty-15015-2352
- **PR:** #15037
- **Paths:** see #15037
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`1d015b47`; collated from changes/15015-fail-closed-motion-matching-gates-archiv.md)
- **Summary:** Fail-closed motion-matching gates; archive run-102 evidence script with restore ref.

### DL-#15014 · Fix Dempster/Zatsiorsky Segment Mass Expansion and C3D Export Placeholder (Part A)

- **State:** shipped
- **Shipped:** 2026-11-14
- **Owner:** unassigned
- **Issue:** #15014
- **Branch:** cursor/segment-mass-c3d-fix-e135
- **PR:** #15036, #15035
- **Paths:** see #15036
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`01c67c54`; collated from changes/15014-urdf-mjcf-converter-preserves-joint-and.md)
- **Summary:** Fix Dempster/Zatsiorsky segment mass expansion and C3D export placeholder (part a)

### DL-#13887 · Add Hill MTU Joint Actuator (Agonist/Antagonist, Compliant Tendon, Linearisation API).

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #13887
- **Branch:** cursor/hill-mtu-joint-actuator-cce3
- **PR:** #15062
- **Paths:** see #15062
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`1b0ef44a`; collated from changes/13887-add-hill-mtu-joint-actuator-agonist-anta.md)
- **Summary:** Add Hill MTU joint actuator (agonist/antagonist, compliant tendon, linearisation API).

### DL-#13781 · ML-WS-4 Native MuJoCo Episode Corpus Generator, Manifest Hashes, and CI Fixture

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #13781
- **Branch:** cursor/ml-ws4-mujoco-episode-corpus-b024
- **PR:** #15059
- **Paths:** see #15059
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`aa6cfb89`; collated from changes/13781-ml-ws-4-native-mujoco-episode-corpus-gen.md)
- **Summary:** ML-WS-4 native MuJoCo episode corpus generator, manifest hashes, and CI fixture

### DL-#12723 · NSX-94: Commit Cross-Engine Co-Contraction Trade-Off Evidence JSON and Validation Harness

- **State:** shipped
- **Shipped:** 2026-11-13
- **Owner:** unassigned
- **Issue:** #12723
- **Branch:** cursor/nsx94-tradeoff-report-3f62
- **PR:** #15067
- **Paths:** see #15067
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`90bfee5b`; collated from changes/12723-nsx-94-commit-cross-engine-co-contractio.md)
- **Summary:** NSX-94: commit cross-engine co-contraction trade-off evidence JSON and validation harness

### DL-#12690 · R9c: Catalog MATLAB R2025b Appearance Acceptance as Deferred; Extend Headless Export and MATLAB Consumer Contract Tests.

- **State:** shipped
- **Shipped:** 2026-11-12
- **Owner:** unassigned
- **Issue:** #12690
- **Branch:** cursor/simscape-appearance-r9c-c25e
- **PR:** #15065
- **Paths:** see #15065
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`bbaf03c0`; collated from changes/12690-r9c-catalog-matlab-r2025b-appearance-acc.md)
- **Summary:** R9c: catalog MATLAB R2025b appearance acceptance as deferred; extend headless export and MATLAB consumer contract tests.

### DL-#11161 — Capture Registry, Capture Export and Swing Comparison (Owner Swing vs Tour Average)

- **Next step:** Coordinate the separately tracked engine deliveries in #11165–#11169 and cross-engine comparison #11170; keep this parent active until its scoped acceptance is met.
- **PR:** not recorded in source
- **State:** in_review
- **Owner:** claude
- **Issue:** #11161
- **Branch:** `feat/capture-registry-11161`
- **Paths:** `data/capture_registry.json`, `src/motion_capture/capture_registry.py`, `src/motion_capture/capture_export.py`, `src/shared/python/swing_comparison/`, `src/shared/python/motion_matching/leaderboard.py`, `src/shared/python/motion_matching/pipeline/`
- **Started:** 2026-09-29
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Capture-O parent coordinates a neutral capture registry/export and engine-independent swing comparison. Engine slices #11165 through #11169 and cross-engine comparison #11170 remain separately tracked.

### DL-#11421 · Dynamics-Informed Mocap Matching With ZTCF Prediction and Continuous Forward Replay

- **State:** in_progress
- **Owner:** antigravity
- **Issue:** #11421 (children #11422–#11430, #11434, #11435, #11437 closed; #11436 active)
- **PR:** #11436
- **Branch:** `feat/dime-11436-learned-initializers`
- **Paths:** `src/shared/python/estimation/dime_learned_initializers.py`, `src/shared/python/estimation/__init__.py`, `tests/unit/estimation/test_dime_learned_initializers.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Epic #11421: ZTCF drift prediction, kinematics-informed mocap matching, calibrated uncertainty weighting, uninterrupted forward simulation replay without frame-to-frame resetting.
- **Next step:** Create PR, verify CI Standard, squash-merge into origin/main.

### DL-#11313 · Necromatcher Video Export Force/Torque Layer

- **State:** in_review
- **Owner:** claude (Sonnet 5 CLI delegate)
- **Issue:** #11313; parent #11285
- **PR:** draft from `claude/issue-11313` (number on the PR; `Fixes #11313`)
- **Branch:** `claude/issue-11313`
- **Paths:** `src/shared/python/workspace/necromatcher_video_forces.py`, `src/shared/python/workspace/necromatcher_video.py`, `src/shared/python/workspace/necromatcher_video_worker.py`, `src/shared/python/workspace/necromatcher_video_jobs.py`, `src/api/routes/necromatcher.py`, `src/tools/necromatcher/video_dialog.py`, `ui/src/components/necromatcher/VideoExportControls.tsx`, `tests/unit/workspace/test_necromatcher_video_forces.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Off-by-default `force_layer {enabled, kinds, scale, segment_shading}` drawn through the shared FTO-8 renderer with the fit's camera; analytic v/a from the preserved spline (kinematic-only fits report the layer unavailable); legend states research-fit, not measured forces; manifest and per-frame receipts are deterministic.
- **Next step:** Frontier review of the draft PR, then rebase against paused draft #11359 (also edits `necromatcher_video.py`) before merge.

### DL-#11278 · COV-10 Model-Candidate Video Comparison

- **State:** in_progress
- **Owner:** claude
- **Issue:** #11278; parent #11268; blocked (real data) on #11165
- **PR:** not created (Refs #11278)
- **Branch:** `claude/issue-11278`
- **Paths:** `src/motion_capture/reference/model_candidate_source.py`, `tests/unit/motion_capture/test_cov10_model_candidate_source.py`, `docs/development/capture-o-video/procedure.md`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Python-only engine-generic path projecting any model candidate into video views and comparing it through the COV-7 L2 receipt; real Simscape evidence needs a stored `capture-O` candidate from #11165 and an R2025b host.
- **Next step:** After #11165 stores a candidate, run `compare_candidate_2d` on an R2025b host for every paired swing and record aggregate results on #11278.

### DL-#11268 · Capture-O Video Companion

- **State:** in_progress
- **Owner:** local
- **Issue:** #11268; children #11269–#11279; parent program #11161
- **PR:** #11502 (Closes #11271, Refs #11268)
- **Branch:** `feat/cov-3-protocol-decision-11271`
- **Paths:** `docs/development/capture-o-video/comparison-protocol.md`, `src/config/cov_comparison_profile.v1.json`, `src/config/cov_landmark_correspondence.json`, `tests/unit/motion_capture/test_cov3_protocol_and_correspondence.py`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Ratify comparison protocol L0-L3, frozen comparison profile, and landmark correspondence table.
- **Next step:** Merge PR for COV-3, complete Epic #11268.

- **PR:** #11499 (Closes #11272, #11274, #11275, #11276, #11277, #11279, Refs #11268)
- **Branch:** `feat/cov-benchmark-suite-11268`
- **Paths:** `src/motion_capture/reference/**`, `docs/development/capture-o-video/**`, `tests/unit/motion_capture/test_cov*`
- **Started:** 2026-10-04
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Complete implementation and verification of Capture-O Video benchmark suite spanning levels L0–L3, virtual camera projection, pairing confidence, marker-anchored subject scaling, 2D/3D residuals, and error budget reporting.
- **Next step:** Merge consolidated PR, close superseded individual draft PRs, and close Epic #11268.

- **PR:** #11416 (Closes #11270, Refs #11268)
- **Branch:** `feat/cov-2-register-sources-11270`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-3 comparison protocol specification and level definitions (#11271).

- **PR:** #11413 (Closes #11269, Refs #11268)
- **Branch:** `feat/cov-1-acquisition-11269`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-2 (#11270) video source registration and swing window grading.

- **PR:** #11417 (Closes #11273, Refs #11268)
- **Branch:** `feat/cov-5-runner-matrix-11273`
- **Started:** 2026-10-02
- **Last verified:** 2026-10-11 at e68baeabde8f
- **Summary:** Bring the owner's capture-session video album into the markerless, Necromatcher and engine-matching pipelines, compare against the `capture-O` marker capture at graded levels L0–L3, and publish a single-view reconstruction error budget. [Procedure](capture-o-video/procedure.md).
- **Next step:** Advance to COV-6 validation harness and error budget report (#11274).

### DL-#11095 — Qualify OpenSim Native Dual-Club Dynamics and Replay

- **Next step:** Native qualification remains unachieved. Run only on the corresponding pinned native lane with complete source-bound evidence; this metadata review did not run tests or native code.
- **Started:** not recorded in source
- **PR:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #11095
- **Branch:** `feat/mmr-10o-opensim-dual-club-11095`
- **Paths:** `src/engines/physics_engines/opensim/python/native_qualification.py`, `tests/unit/engines/opensim/test_opensim_dual_club_qualification.py`, `docs/development/matched_swing_program/evidence/opensim/`
- **Last verified:** 2026-09-29 at d5dd606ffa6dafc11408614676a2576764a5e092 — 17 unit tests passed; red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native OpenSim qualification NOT achieved: opensim bindings unavailable on all reachable hosts.
- **Summary:** OpenSim receipts now fail closed on missing runtime/evidence; fabricated marker metrics, unenforced tolerances and placeholder hashes were removed. Native qualification remains blocked without the pinned bindings.

### DL-#11096 — Qualify MyoSuite Native Dual-Club Dynamics and Replay

- **Next step:** Native qualification remains unachieved. Run only on the corresponding pinned native lane with complete source-bound evidence; this metadata review did not run tests or native code.
- **Started:** not recorded in source
- **PR:** not recorded in source
- **Owner:** not recorded in source
- **State:** in_review
- **Issue:** #11096
- **Branch:** `feat/mmr-10m-myosuite-dual-club-11096`
- **Paths:** `src/engines/physics_engines/myosuite/python/native_qualification.py`, `tests/unit/engines/myosuite/test_myosuite_dual_club_qualification.py`, `docs/development/matched_swing_program/evidence/myosuite/`
- **Last verified:** 2026-09-29 at 613b0e28ed88b5af4719ecccfe876d5d9db90419 — 19 unit tests passed; red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native MyoSuite qualification NOT achieved: myosuite/MuJoCo unavailable on all reachable hosts.
- **Summary:** MyoSuite receipts fail closed on unavailable runtime and missing/invalid rollout, marker, derivative or quaternion evidence; fabricated metrics and unenforced tolerances were removed. Native qualification remains unavailable.

### DL-#11094 — Qualify Drake Native Dual-Club Dynamics and Replay

- **Started:** not recorded in source
- **PR:** not recorded in source
- **State:** in_review
- **Owner:** UDFixTrio10x
- **Issue:** #11094
- **Branch:** `feat/mmr-10d-drake-dual-club-11094`
- **Paths:** `src/engines/physics_engines/drake/python/native_qualification.py`, `tests/unit/engines/drake/test_drake_dual_club_qualification.py`, `scripts/ci/run_native_engine_lane.py`, `scripts/ci/run_native_engine_lane.sh`, `tests/scripts/test_run_native_engine_lane.py`, `docs/development/matched_swing_program/evidence/nightly/drake_receipt.json`, `docs/development/matched_swing_program/evidence/drake/`
- **Last verified:** 2026-09-29 at 3b23bd41502a7b7d028f291564997b6460f14cbc — 36 focused tests passed (test_drake_dual_club_qualification.py 16, test_run_native_engine_lane.py + test_native_lane_freshness.py 20); red evidence recorded for the 7 new fail-closed tests against the pre-fix placeholder path; ruff check and format clean on changed files. Native Drake qualification NOT achieved: pydrake unavailable on all reachable hosts.
- **Summary:** Drake receipts now fail closed on missing runtime/evidence; fabricated marker metrics and placeholder hashes were removed, and Drake entered the nightly lane. Native qualification remains blocked without pydrake.
- **Next step:** Merge drivers follow; native qualification remains blocked on engine availability (recorded in receipts, not silently).

### DL-#10286 · Native Swing ZTCF/ZVCF Fail-Closed Dynamics and DTACK Semantics

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #10286
- **PR:** draft PR from `agy/ud-10286-fail-closed`
- **Branch:** `agy/ud-10286-fail-closed`
- **Paths:** `src/engines/physics_engines/myosuite/python/_drift_control.py`, `src/engines/physics_engines/pendulum/python/golf_swing_physics_engine.py`, `src/engines/physics_engines/pinocchio/python/dtack/gui/main_window.py`, `src/engines/physics_engines/pinocchio/python/dtack/sim/dynamics.py`, `tests/engines/physics_engines/test_golf_swing_pendulum.py`, `tests/engines/physics_engines/test_myosuite_engine.py`, `tests/unit/engines/pinocchio/dtack/sim/test_dynamics.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Remove success-shaped empty/zero array returns from uninitialized ZTCF/ZVCF counterfactual methods (raising StateError and ValueError for invalid dimensions), update dtack compute_zvcf to canonical (v=0, tau=0) semantics, add compute_zero_velocity_controlled for control-preserved dynamics, and update GUI caller and button label to "Zero-velocity (control kept)".
- **Next step:** CI green, review, merge.

### DL-#9619 · Amend ADR-0041 to Record Consumer-Side Fitter Decision

- **PR:** not recorded in source
- **State:** in_review
- **Owner:** local (session `claude-deskcomputer-20260925-ud`)
- **Issue:** #9619 (companion #9630)
- **Branch:** `agy/ud-9619-adr0041`
- **Paths:** `docs/adr/0041-markerless-mocap-consumer-authority.md`, `tests/architecture/test_markerless_mocap_authority.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Amended ADR-0041 to record that consumer-side self-calibration fitters live in UpstreamDrift (`src/motion_capture/reconstruct/`) while Tools maintains vendor-neutral reference geometry (#9630, #9619).
- **Next step:** CI green, review, merge.

### DL-#9688 · Bunker P1: Decide and Benchmark a Genuine 3D Sand-Motion Tier

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9688
- **PR:** draft PR from `agy/ud-9688-capability-register`
- **Branch:** `agy/ud-9688-capability-register`
- **Paths:** `src/bunkershot3d/solvers/capability.py`, `src/bunkershot3d/solvers/exceptions.py`, `src/bunkershot3d/solvers/__init__.py`, `docs/adr/0044-out-of-plane-fidelity-for-bunkershot3d.md`, `tests/bunkershot3d/solvers/test_capability_9688.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added fail-closed sand-motion capability register (SandMotionKind, SandMotionCapability, register mapping, and require\_\* guards) in bunkershot3d.solvers to prevent presenting F0/F1/proxy/tracer outputs as genuine 3-D individual-grain trajectories or spherical ball spin.
- **Next step:** CI green, review, merge.

### DL-#9613 · Rig Soak: Multi-Camera Repeat Record Mode

- **State:** in_review
- **Owner:** antigravity
- **Issue:** #9613
- **PR:** draft PR from `agy/ud-9613-record-repeat`
- **Branch:** `agy/ud-9613-record-repeat`
- **Paths:** `src/motion_capture/rig/__main__.py`, `src/motion_capture/rig/soak.py`, `tests/motion_capture/rig/test_record_repeat.py`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added `record --repeat N [--pause S]` soak mode to camera rig CLI, supporting N back-to-back takes in take subdirectories, machine-readable soak summary, and worst-take exit code.
- **Next step:** CI green, review, merge.

### DL-#9411 · Keep MyPy Exclusion Budget and Unit-Gate Quarantine Ratchets Green

- **State:** in_review
- **Owner:** claude (agy executor)
- **Issue:** #9411
- **PR:** draft PR from `claude/ud-9411-unquarantine-passing`
- **Branch:** `claude/ud-9411-unquarantine-passing`
- **Paths:** `scripts/config/unit_gate_quarantine.json`, quarantined unit test files, `src/shared/python/validation_pkg/data_fitting.py`, `src/tools/model_explorer/mujoco_viewer.py`, `src/shared/python/ai/adapters/`, `src/api/routes/__init__.py`, `src/shared/python/physics/impact_model/utils.py`, `src/shared/python/pendulum_simulator/physics.py`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-26 — #10997 Linux run: kernel built and imported, none of the 12 retired BunkerShot IDs failed; the tests the kernel un-skipped exposed #11000 (fixed in #11004) and #11003 (fixed in #11010), and an untracked root `Cargo.lock` (fixed); replayed onto main d716c698f, ledger contract passes at 55 IDs.
- **Summary:** Retire unit-gate quarantine IDs whose tests pass on the Linux CI lane after fixing real defects (shoulder FK origin), deduplicating the `data_fitting` and `mujoco_viewer` coordinators onto their existing helper modules, and retargeting stale tests. Tools-owned `ai` adapter changes and the gear-effect sign question are out of scope here (Tools#5362 / a physics-convention decision). Earlier slice (#10973) landed the mypy budget and 43 retirements. Rust slice (#10997): unit-test-gate maturin-builds upstream-physics into its venv (ball_simulator enforces strict Rust parity), retiring the 12 BunkerShot workbench/GUI IDs. Consolidated delivered subscopes from the preserved source rows: Claude #10995 (claude/ud-9411-vendor-bump-safe-eval) advanced the Tools pin and retired 10 quarantined tests; Antigravity #10999 (feat/ud-9411-quarantine-burndown-batch1) retired 6; #11005 (feat/ud-9411-ai-adapters-convergence) converged Gemini/BitNet adapters and retired 5; #11013 (feat/ud-9411-quarantine-batch2) retired 1 and narrowed optional-dependency mock scope. The source snapshot retains original owners, branches, paths, verification text and next steps. Broader parent mypy/quarantine readiness remains in review; no full ratchet completion is claimed.
- **Next step:** Complete parent-level mypy-exclusion and quarantine acceptance against a source-bound Linux unit-gate run, then reconcile the remaining eligible IDs. The delivered child tranches do not establish full parent readiness; the original child next steps are preserved in the source snapshot.

### DL-#9703 · Versioned Pre-Impact Bundle for Tools Impact Kernels

- **State:** in_review
- **Owner:** claude
- **Issue:** #9703 (IA-U2; parent #9700)
- **PR:** draft (see branch)
- **Branch:** `feat/9703-pre-impact-bundle`
- **Paths:** `src/shared/python/physics/pre_impact_bundle.py`, `src/shared/python/physics/_pre_impact_contracts.py`, `src/shared/python/physics/_pre_impact_frames.py`, `tests/shared_contracts/test_pre_impact_bundle.py`, `docs/development/impact_acoustics_program.md`
- **Started:** 2026-09-25
- **Last verified:** 2026-09-25 — 48/48 tests pass against vendor pin a9ed0e7c5 (incl. python -O); pre-push mypy and unit hooks pass (Rust wheel hidden, #10946)
- **Summary:** Consumer-side PreImpactBundle v1 composing Tools conventions: provenance, bounded timebase, world/head/grip poses, head COM inertia and contact, ball, reduced shaft modal state and prestress field, per-hand wrench/impedance, per-field origin; absent fields raise; power-invariant transforms; energy-reporting modal projection. Engine adapters and installed-wheel fixtures remain.
- **Next step:** Review the draft PR, then implement the first engine adapter against the bundle under #9703.

### DL-#10342 · OpenSim/MyoSuite Native Nightly Lane Receipts

- **PR:** not recorded in source
- **State:** in_review
- **Owner:** local
- **Issue:** #10342 (MS-43, epic #10363)
- **Branch:** fix/issue-10342-ms-43-native-lane-local
- **Paths:** scripts/ci/run_native_engine_lane.py; scripts/ci/run_native_engine_lane.sh; docs/development/matched_swing_program/evidence/nightly/; tests/docs/test_native_lane_freshness.py; tests/scripts/test_run_native_engine_lane.py
- **Started:** 2026-09-21
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Added idempotent native-engine lane runner emitting hashed nightly receipts for OpenSim and MyoSuite (`requires_opensim` / `requires_myosuite` pytest markers), freshness gate warning at seven days and failing at thirty days, and fleet native-host verification docs without editing `.github/workflows`.
- **Next step:** Refresh receipts on the fleet native host opensim-10003 venv; merge PR closing #10342.
- **Evidence:** docs/development/matched_swing_program/evidence/nightly/opensim_receipt.json; docs/development/matched_swing_program/evidence/nightly/myosuite_receipt.json; tests/docs/test_native_lane_freshness.py.

### DL-#9162 · Local Branch Triage

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** claude
- **Issue:** #9162 (rollout Repository_Management#1460; no-seed bootstrap #9161)
- **Branch:** conductor/issue-9162
- **Paths:** docs/development/branch_triage_9162.md
- **Started:** 2026-09-15
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Disposition ledger for every local branch in the primary checkout: 3 protected, 27 deferred to the worktree pass, 34 live (merged sweep then DL entry), 178 stale and 144 abandoned scratch branches (bundle snapshot then delete). The runbook in the ledger executes the deletions and the coordination notice for agent-owned branches.
- **Next step:** Execute the runbook in docs/development/branch_triage_9162.md from the primary checkout and fill in its Outcome section.

### DL-#9410 · Adversarial Product Review Remediation Epic

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** claude
- **Issue:** #9410 (program Repository_Management#1505; children #8820–#8943, #8360, #8641, #8843, #8846, #8853, #8861–#8870, #8874–#8876, #8894)
- **Branch:** conductor/issue-9410
- **Paths:** docs/development/adversarial_review_remediation_9410.md
- **Started:** 2026-09-16
- **Last verified:** 2026-09-16 (SELF; every child re-checked against `db4fe88c4` by first-parent history search plus source reads of each residual)
- **Summary:** Reconciliation ledger for the 2026-08-21 adversarial review: 34 of 61 children landed on `main` (SHAs recorded), 27 residual. Three residual fixes already exist on unmerged branches (`readiness/p0-9412-one-tile-registry`, `conductor/issue-8865`, `conductor/issue-8866`). Epic acceptance (one registry, one API factory, one C3D reader, one pose type, no GUI-thread simulation) is not met on `main`.
- **Next step:** Rebase and merge `readiness/p0-9412-one-tile-registry`, then re-verify the Cluster B rows in the ledger.

### DL-#10602 · Club-Only Motion Matching Plan

- **State:** in_review
- **Owner:** codex
- **Issue:** #10602
- **Branch:** `agy/ud-10602-real-matrix`
- **PR:** draft PR from `agy/ud-10602-real-matrix`
- **Paths:** docs/plans/club_neural_review/; docs/plans/club_only_matching/; docs/plans/neural_motion_matching/; src/shared/python/motion_matching/club_only/
- **Started:** 2026-09-20
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Published bounded implementation issues with TDD/DbC/LoD/DRY prompts, dependency ordering, native validation gates and shared technical review. Planning artifacts do not qualify physical results or speedup.
- **Next step:** CI green, review, merge.
- **Evidence:** docs/plans/club_neural_review/REVIEW.md; docs/plans/club_neural_review/excel_audit.json.

### DL-#10363 · Matched Swing Continuation Review & Drake G1 Retraction

- **State:** in_review
- **Owner:** claude-deskcomputer-20260925-ud
- **Issue:** #10363
- **Branch:** agy/ud-10363-drake-retraction
- **PR:** draft PR from `agy/ud-10363-drake-retraction`
- **Paths:** docs/development/matched_swing_program; src/shared/python/motion_matching/acceptance.py; src/engines/physics_engines/drake/python/full_body_fit.py; evidence/matched/driver_g1_drake/reevaluation.json; docs/development/full_body_models/evidence/acceptance/verdicts_2026-09.json; reports/matched_swing_ledger.json
- **Started:** 2026-09-18
- **Last verified:** 2026-09-25 at ff9fbee62 (retracted Drake G1 fabricated PASS via reevaluation.json; closed synthesis paths in full_body_fit.py; added fail-closed integrity gates in acceptance.py; 0 PASSED rows in ledger; motion_matching/tour_baselines/opensim ladder/browser suites green except 13 pre-existing local failures)
- **Summary:** 2026-09-18 continuation review (codex) preserved checkpoint/source recovery evidence and bounded the cheaper-agent continuation, with no physical acceptance claimed. Then: retracted the fabricated Drake G1 receipt (PR #10506) under MS-100; added fail-closed evidence integrity gates (\_evaluate_evidence_integrity) rejecting placeholder hashes and zero residuals without replay evidence; removed physical_audit numeric literals, np.zeros target fallback, and silent warm-start fallback in Drake full_body_fit.py; regenerated ledger and status matrices confirming 0 PASSED rows.
- **Next step:** CI green, review, merge.
- **Evidence:** docs/development/matched_swing_program/evidence/continuation_20260918/matching-handoff-snapshot.json; evidence/matched/driver_g1_drake/reevaluation.json; reports/matched_swing_ledger.json; tests/unit/motion_matching/test_acceptance.py; tests/unit/motion_matching/test_drake_full_body_fit.py.

### DL-#10362 · Tools Dependency Gate for Matched Swing Program

- **State:** in_progress
- **Owner:** local
- **Issue:** #10362 (epic #10363)
- **Branch:** feat/ms95-tools-dependency-gate-10362
- **PR:** open
- **Paths:** vendor/ud-tools; Cargo.toml; requirements-tools.txt; docs/shared_tools/divergence_inventory.md; docs/shared_tools/divergence_inventory.v1.json; docs/agent_context/README.md; docs/agent_context/index.html
- **Started:** 2026-09-17
- **Last verified:** 2026-09-17 at 62e8cdbf9 (SELF; Tools main green, Tools #4494 and #4262 closed, Tools #5227 landed; four-way pin bumped to 62e8cdbf9; divergence inventory regenerated; agent context verified; test_no_shadow_of_tools_shared and run_checks pass)
- **Summary:** UpstreamDrift ownership of humanoid_character_builder and model_generation ruled per Tools #4494; Tools #4262 and #4494 closed; vendor/ud-tools, requirements-tools.txt, Cargo.toml, and divergence inventory repinned to Tools main 62e8cdbf9.
- **Next step:** Open PR referencing Closes #10362, enable auto-merge, release lease.
- **Evidence:** docs/shared_tools/divergence_inventory.v1.json; tests/unit/repo_hygiene/test_no_shadow_of_tools_shared.py; tests/fixtures/reference_calibration/run_checks.py.

### DL-#9967 · Native Simscape Tour Matching

- **PR:** not recorded in source
- **State:** in_progress
- **Owner:** codex (turnover review; execution ownership by next lease)
- **Issue:** #9967 (parent #9921)
- **Branch:** feat/9967-native-simscape-pinocchio
- **Paths:** src/shared/python/motion_matching; docs/development/simscape_tour_matching
- **Started:** 2026-09-09
- **Last verified:** 2026-10-11 at e68baeabde8f (metadata/path inventory only; no tests or native runs; see evidence/devlog_reconciliation_11935_20261011/DEVELOPMENT_LOG_CURRENT_SOURCE_METADATA_AUDIT_e68_20261011.json)
- **Summary:** Run101 improves yaw and has measured R2025b–Pinocchio prefix agreement of 0.0605 mm maximum. Terminal RMS 40.31 mm fails the 35 mm gate; full 1.814 s capture is incomplete. No optimizer launched by this review.
- **Next step:** Follow COMPLETION_HANDOFF_20260916.md: restore clean-runtime native providers, coordinate #10260 finite-weld derivative qualification, produce articulated feasibility evidence, fit0.90 s and deliver repeatable reports. Run102 remains rejected; no new compute launched by this review.

## Shipped (Last 90 Days)

### DL-#15117 · Desktop UI Sends the Launcher Capability Token on Backend API Requests Through One Shared Fetch Helper

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15117
- **Branch:** merged via #15118
- **PR:** #15118
- **Paths:** see #15118
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`800ac553`; collated from changes/15117-desktop-ui-sends-the-launcher-capability.md)
- **Summary:** Desktop UI sends the launcher capability token on backend API requests through one shared fetch helper
- **Next step:** Shipped in PR #15118.

### DL-#15108 · Scope Tests/Unit/conftest.py Engine MagicMocks to Tests/Unit and Guard OpenSim Dynamics Parity (#15108)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15108
- **Branch:** merged via #15109
- **PR:** #15109
- **Paths:** see #15109
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`daf6ce2a`; collated from changes/15108-scope-unit-conftest-engine-mocks.md)
- **Summary:** Scope tests/unit/conftest.py engine MagicMocks to tests/unit and guard OpenSim dynamics parity (#15108)
- **Next step:** Shipped in PR #15109.

### DL-#15101 · Join or Retain Retiring BinderThread and CameraWorker Instances Until Qt Finishes Thread Teardown so Capture Rig Preview Never Aborts in ~QThread.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15101
- **Branch:** merged via #15102
- **PR:** #15102
- **Paths:** see #15102
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`8123908e`; collated from changes/15101-preview-qthread-lifetime.md)
- **Summary:** Join or retain retiring BinderThread and CameraWorker instances until Qt finishes thread teardown so Capture Rig preview never aborts in ~QThread.
- **Next step:** Shipped in PR #15102.

### DL-#15097 · Isolate Sidekick Sys.Modules and Sys.Meta_Path State Across Overlay Tests (#15097)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15097
- **Branch:** merged via #15099
- **PR:** #15099
- **Paths:** see #15099
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`eb2d9b7e`; collated from changes/15097-sidekick-overlay-module-isolation.md)
- **Summary:** Isolate sidekick sys.modules and sys.meta_path state across overlay tests (#15097)
- **Next step:** Shipped in PR #15099.

### DL-#15091 · Drain Child Output Streams Before Transitioning ManagedProcess Exit State and Exposing Exit_Code

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15091
- **Branch:** merged via #15092
- **PR:** #15092
- **Paths:** see #15092
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`257654e0`; collated from changes/15091-stderr-drain-before-exit.md)
- **Summary:** Drain child output streams before transitioning ManagedProcess exit state and exposing exit_code
- **Next step:** Shipped in PR #15092.

### DL-#15013 · Restore UdAttachmentManifestMixin so Rejected File_Upload Clears Dedup SHA and Retry Sends WebSocket Again; Re-Add Reject-Then-Retry Regression Tests.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15013
- **Branch:** merged via #15072
- **PR:** #15072
- **Paths:** see #15072
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`ae164ab5`; collated from changes/15013-reconcile-sg-putting-logistic-with-regis.md)
- **Summary:** Restore UdAttachmentManifestMixin so rejected file_upload clears dedup SHA and retry sends WebSocket again; re-add reject-then-retry regression tests.
- **Next step:** Shipped in PR #15072.

### DL-#14850 · Redact Host/Home Literals in Batch 5 Docs and Add Shell Placeholder Regression Test

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14850
- **Branch:** merged via #14850
- **PR:** #14850
- **Paths:** see #14850
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`9a0dbab7`; collated from changes/14850-redact-host-home-literals-in-batch-5-doc.md)
- **Summary:** Redact host/home literals in batch 5 docs and add shell placeholder regression test
- **Next step:** Shipped in PR #14850.

### DL-#14849 · Parameterize Remote Evidence Runners and Finish Batch 3 Host Redaction

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14849
- **Branch:** merged via #14849
- **PR:** #14849
- **Paths:** see #14849
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`c60c445c`; collated from changes/14849-parameterize-remote-evidence-runners-and.md)
- **Summary:** Parameterize remote evidence runners and finish batch 3 host redaction
- **Next step:** Shipped in PR #14849.

### DL-#14310 · Headless Qt/Matplotlib Teardown After Each Test and MuJoCo Offscreen Renderer Release (#14310)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14310
- **Branch:** merged via #15098
- **PR:** #15098
- **Paths:** see #15098
- **Started:** 2026-11-15
- **Last verified:** 2026-11-15 (`180394fd`; collated from changes/14310-headless-gui-teardown.md)
- **Summary:** Headless Qt/matplotlib teardown after each test and MuJoCo offscreen renderer release (#14310)
- **Next step:** Shipped in PR #15098.

### DL-#15095 · Fix SwingMetricsMixin Test Harness After Velocity Delegation and Add Unit Marker (#15095)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15095
- **Branch:** merged via #15096
- **PR:** #15096
- **Paths:** see #15096
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`c0d0f6c9`; collated from changes/15095-dbc-swing-phase-unit-marker.md)
- **Summary:** Fix SwingMetricsMixin test harness after velocity delegation and add unit marker (#15095)
- **Next step:** Shipped in PR #15096.

### DL-#15093 · Guard NP-05 Lexical Test Against Collated Changelog Fragment Removal (#15093)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15093
- **Branch:** merged via #15094
- **PR:** #15094
- **Paths:** see #15094
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`a011577c`; collated from changes/15093-np05-collated-changes-guard.md)
- **Summary:** Guard NP-05 lexical test against collated changelog fragment removal (#15093)
- **Next step:** Shipped in PR #15094.

### DL-#15046 · SCI-10/12/13/14: Shaft FEM Scope Docs, Pendulum Inertia Param, Static-Opt Cost, Spatial Ordering

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15046
- **Branch:** merged via #15052
- **PR:** #15052
- **Paths:** see #15052
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`e25c7b5b`; collated from changes/15046-sci-10-12-13-14-shaft-fem-scope-docs-pen.md)
- **Summary:** SCI-10/12/13/14: shaft FEM scope docs, pendulum inertia param, static-opt cost, spatial ordering
- **Next step:** Shipped in PR #15052.

### DL-#15042 · Use Transverse Spin in Magnus/Lift Magnitude; Document Frame and Parity Fixtures (#15042).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15042
- **Branch:** merged via #15053
- **PR:** #15053
- **Paths:** see #15053
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`9c8f8834`; collated from changes/15042-use-transverse-spin-in-magnus-lift-magni.md)
- **Summary:** Use transverse spin in Magnus/lift magnitude; document frame and parity fixtures (#15042).
- **Next step:** Shipped in PR #15053.

### DL-#15041 · Evaluate Constrained Per-Actuator Columns at Measured (Q, V) With ZTCF Baseline

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15041
- **Branch:** merged via #15049
- **PR:** #15049
- **Paths:** see #15049
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`c6acff9d`; collated from changes/15041-evaluate-constrained-per-actuator-column.md)
- **Summary:** Evaluate constrained per-actuator columns at measured (q, v) with ZTCF baseline
- **Next step:** Shipped in PR #15049.

### DL-#15030 · Quantize Spatial Model Canonical Hashes for Cross-Runtime Manufactured Authority Parity, Regenerate Publication Evidence, and Re-Pick Optional-Stack Shooting Defect Floor After #14227 Gravity Alignment.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15030
- **Branch:** merged via #15069
- **PR:** #15069
- **Paths:** see #15069
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`3e7c72e8`; collated from changes/15030-quantize-spatial-model-canonical-hashes.md)
- **Summary:** Quantize spatial model canonical hashes for cross-runtime manufactured authority parity, regenerate publication evidence, and re-pick optional-stack shooting defect floor after #14227 gravity alignment.
- **Next step:** Shipped in PR #15069.

### DL-#15028 · Wave 0.1: Independent Triage of 67 Budget Exceptions That Expired 2026-11-30 (Measured Sizes, Regression Tests).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15028
- **Branch:** merged via #15074
- **PR:** #15074
- **Paths:** see #15074
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`175e7bc6`; collated from changes/15028-wave0-architecture-budget-triage.md)
- **Summary:** Wave 0.1: independent triage of 67 budget exceptions that expired 2026-11-30 (measured sizes, regression tests).
- **Next step:** Shipped in PR #15074.

### DL-#15021 · Expose ZTCF and Golf-Simulator Golden Capabilities via Sim_Run With REST/MCP Parity and 64 KiB Artifact Spill.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15021
- **Branch:** merged via #15061
- **PR:** #15061
- **Paths:** see #15061
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`b1bc35ea`; collated from changes/15021-expose-ztcf-and-golf-simulator-golden-ca.md)
- **Summary:** Expose ZTCF and golf-simulator golden capabilities via sim_run with REST/MCP parity and 64 KiB artifact spill.
- **Next step:** Shipped in PR #15061.

### DL-#15016 · Link Tile Help to AffineDrift Theory; Register Six Provisional Core Calculations and Dedupe Registry Blocker Ids

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15016
- **Branch:** merged via #15063
- **PR:** #15063, #15080, #15079, #15090
- **Paths:** see #15063
- **Started:** 2026-11-14
- **Last verified:** 2026-11-15 (`d54c0c57`; collated from changes/15016-spec-input-map-manifest-honesty.md)
- **Summary:** Link tile help to AffineDrift theory; register six provisional core calculations and dedupe registry blocker ids
- **Next step:** Shipped in PR #15063.

### DL-#15012 · Fix WBC Zero Torques With Contact and Hierarchical Task Enforcement

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15012
- **Branch:** merged via #15025
- **PR:** #15025, #15083
- **Paths:** see #15025
- **Started:** 2026-11-14
- **Last verified:** 2026-11-15 (`339be5a5`; collated from changes/15012-friction-cone-rl-honesty.md)
- **Summary:** Fix WBC zero torques with contact and hierarchical task enforcement
- **Next step:** Shipped in PR #15025.

### DL-#15010 · Correct Putting Slope Rolling (5/7 G), Unify Stimp Friction, Impact Club Spin, Flight Spin Sign Docs, Rust Air Temperature K

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15010
- **Branch:** merged via #15026
- **PR:** #15026
- **Paths:** see #15026
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`49d3fc78`; collated from changes/15010-correct-putting-slope-rolling-5-7-g-unif.md)
- **Summary:** Correct putting slope rolling (5/7 g), unify stimp friction, impact club spin, flight spin sign docs, Rust air temperature K
- **Next step:** Shipped in PR #15026.

### DL-#14949 · Managed Local Child-Process Lifecycle; Concurrent Stdout/Stderr Drain Avoids Pipe Deadlock (Refs #14949).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14949
- **Branch:** merged via #14956
- **PR:** #14956, #15081
- **Paths:** see #14956
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`450becd6`; collated from changes/14949-wsl-matlab-docker-lifecycle.md)
- **Summary:** Managed local child-process lifecycle; concurrent stdout/stderr drain avoids pipe deadlock (Refs #14949).
- **Next step:** Shipped in PR #14956.

### DL-#14942 · EngineManager Reloads LOADED Engines; Cleanup Resets Status and Closes Active Engine (#14942).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14942
- **Branch:** merged via #15017
- **PR:** #15017, #14961
- **Paths:** see #15017
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`38c09978`; collated from changes/14942-enginemanager-reloads-loaded-engines-cle.md)
- **Summary:** EngineManager reloads LOADED engines; cleanup resets status and closes active engine (#14942).
- **Next step:** Shipped in PR #15017.

### DL-#14936 · JaxSim Canonical-V2 Round-Trip; Rename Velocity Constant to SPATIAL_WRENCH_REPRESENTATION

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14936
- **Branch:** merged via #15038
- **PR:** #15038, #14966
- **Paths:** see #15038
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`1f3202e7`; collated from changes/14936-wire-canonical-v2-get-set-on-basephysics.md)
- **Summary:** JaxSim canonical-v2 round-trip; rename velocity constant to SPATIAL_WRENCH_REPRESENTATION
- **Next step:** Shipped in PR #15038.

### DL-#14935 · Committed Evidence Scan Calls Receipt Taxonomy Validation; Missing Fields and Conflicts Are Tested (#14935).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14935
- **Branch:** merged via #14960
- **PR:** #14960, #15120, #15082
- **Paths:** see #14960
- **Started:** 2026-11-14
- **Last verified:** 2026-11-15 (`318242c0`; collated from changes/14935-evidence-recompute-and-ratchets.md)
- **Summary:** Committed evidence scan calls receipt taxonomy validation; missing fields and conflicts are tested (#14935).
- **Next step:** Shipped in PR #14960.

### DL-#14502 · NP-09 Policy Input-Output Sensitivity at Stated Operating Points (#14502)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14502
- **Branch:** merged via #14825
- **PR:** #14825, #15100
- **Paths:** see #14825
- **Started:** 2026-11-14
- **Last verified:** 2026-11-15 (`30f937bf`; collated from changes/14502-np09-lexical-fragment-guard.md)
- **Summary:** NP-09 policy input-output sensitivity at stated operating points (#14502)
- **Next step:** Shipped in PR #14825.

### DL-#14427 · Event-Relative Time Normalisation (101-Point Resampling) Between Named Swing Events

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14427
- **Branch:** merged via #14834
- **PR:** #14834
- **Paths:** see #14834
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`e64ae5e7`; collated from changes/14427-gm4-event-time-normalisation.md)
- **Summary:** Event-relative time normalisation (101-point resampling) between named swing events
- **Next step:** Shipped in PR #14834.

### DL-#14426 · GM-3 Cheetham X-Factor Stretch and Kinematic Sequence (#14426)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14426
- **Branch:** merged via #14836
- **PR:** #14836
- **Paths:** see #14836
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`8722411f`; collated from changes/14426-gm3-xfactor-kinematic-seq.md)
- **Summary:** GM-3 Cheetham X-factor stretch and kinematic sequence (#14426)
- **Next step:** Shipped in PR #14836.

### DL-#14425 · GRF and COP Observability, Event-Anchored Peaks, and Normalisation Contracts (GM-2).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14425
- **Branch:** merged via #14837
- **PR:** #14837
- **Paths:** see #14837
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`120e4803`; collated from changes/14425-gm2-grf-cop-observability.md)
- **Summary:** GRF and COP observability, event-anchored peaks, and normalisation contracts (GM-2).
- **Next step:** Shipped in PR #14837.

### DL-#12727 · NSX-98: Document A(t)/B(t) Assembly in Derivative Conformance Harness

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12727
- **Branch:** merged via #15066
- **PR:** #15066
- **Paths:** see #15066
- **Started:** 2026-11-14
- **Last verified:** 2026-11-14 (`f7edaa35`; collated from changes/12727-nsx-98-document-a-t-b-t-assembly-in-deri.md)
- **Summary:** NSX-98: document A(t)/B(t) assembly in derivative conformance harness
- **Next step:** Shipped in PR #15066.

### DL-#12046 · Interim 30-Day Realtime Shadow Ledger Sunset (#12046) so Hygiene Tests Pass Pending Tools Migration

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12046
- **Branch:** merged via #14929
- **PR:** #14929, #15085
- **Paths:** see #14929
- **Started:** 2026-11-14
- **Last verified:** 2026-11-15 (`e47a24aa`; collated from changes/12046-realtime-adr-split-ruling.md)
- **Summary:** Interim 30-day realtime shadow ledger sunset (#12046) so hygiene tests pass pending Tools migration
- **Next step:** Shipped in PR #14929.

### DL-#15047 · README Onboarding: Venv-First Install, Toolchains, Ud-Sim, Engines, Layout Tree.

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15047
- **Branch:** cursor/readme-onboarding-15047-1324
- **PR:** #15057
- **Paths:** see #15057
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`95058d94`; collated from changes/15047-readme-onboarding-venv-first-install-too.md)
- **Summary:** README onboarding: venv-first install, toolchains, ud-sim, engines, layout tree.
- **Next step:** Shipped in PR #15057.

### DL-#15029 · Renew or Delete 67 Budget Exceptions Expiring 2026-11-30 (Wave 0 #15028).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15029
- **Branch:** merged via #15031
- **PR:** #15031
- **Paths:** see #15031
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`28da654d`; collated from changes/15029-renew-or-delete-67-budget-exceptions-exp.md)
- **Summary:** Renew or delete 67 budget exceptions expiring 2026-11-30 (Wave 0 #15028).
- **Next step:** Shipped in PR #15031.

### DL-#15009 · Fix OpenSim IAA Gravity Sign and Moment-Arm Mapping; MuJoCo Constraint-Consistent Control Acceleration; Minor Docs/Dead Code

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #15009
- **Branch:** merged via #15024
- **PR:** #15024
- **Paths:** see #15024
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`625a16e6`; collated from changes/15009-fix-opensim-iaa-gravity-sign-and-moment.md)
- **Summary:** Fix OpenSim IAA gravity sign and moment-arm mapping; MuJoCo constraint-consistent control acceleration; minor docs/dead code
- **Next step:** Shipped in PR #15024.

### DL-#14939 · Agent Control Stubs Return Honest Not_Implemented Results (#14939).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14939
- **Branch:** merged via #14957
- **PR:** #14957
- **Paths:** see #14957
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`2383b013`; collated from changes/14939-agent-control-stubs-return-honest-not-im.md)
- **Summary:** Agent control stubs return honest not_implemented results (#14939).
- **Next step:** Shipped in PR #14957.

### DL-#14480 · NP-05 Benchmark Localized Edit vs Constrained Fine-Tune (#14480)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14480
- **Branch:** merged via #14828
- **PR:** #14828
- **Paths:** see #14828
- **Started:** 2026-11-12
- **Last verified:** 2026-11-12 (`ecc9b1f0`; collated from changes/14480-np05-edit-vs-finetune-benchmark.md)
- **Summary:** NP-05 benchmark localized edit vs constrained fine-tune (#14480)
- **Next step:** Shipped in PR #14828.

### DL-#14940 · AIP JSON-RPC Stubs Return Not_Implemented; Capabilities Self-Check (#14940).

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14940
- **Branch:** merged via #15000
- **PR:** #15000, #14959
- **Paths:** see #15000
- **Started:** 2026-11-10
- **Last verified:** 2026-11-14 (`d91cd296`; collated from changes/14940-aip-json-rpc-stubs-return-not-implemente.md)
- **Summary:** AIP JSON-RPC stubs return not_implemented; capabilities self-check (#14940).
- **Next step:** Shipped in PR #15000.

### DL-#14943 · CounterfactualKind/Result, Simscape/Putting Defaults, Conformance (#14943 Slices 1–4)

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14943
- **Branch:** merged via #14998
- **PR:** #14998, #14965, #15005, #14973, #15007
- **Paths:** see #14998
- **Started:** 2026-11-10
- **Last verified:** 2026-11-14 (`96af1f50`; collated from changes/14943-mujoco-tau-generalized-force.md)
- **Summary:** CounterfactualKind/Result, simscape/putting defaults, conformance (#14943 slices 1–4)
- **Next step:** Shipped in PR #14998.

### DL-#14948 · Enforce Required Physics Engine Execution, JUnit Floor Checks, and JaxSim Heavy Test Lane

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14948
- **Branch:** merged via #14977
- **PR:** #14977, #14993, #15004
- **Paths:** see #14977
- **Started:** 2026-11-10
- **Last verified:** 2026-11-14 (`7376c4ab`; collated from changes/14948-conftest-makereport-and-floor.md)
- **Summary:** Enforce required physics engine execution, JUnit floor checks, and JaxSim heavy test lane
- **Next step:** Shipped in PR #14977.

### DL-#14227 · Document CI Optional Stack Slice 1 (Pinocchio Gravity) and Add Runbook Contract Tests

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #14227
- **Branch:** merged via #14923
- **PR:** #14923, #15089
- **Paths:** see #14923
- **Started:** 2026-11-10
- **Last verified:** 2026-11-15 (`20669b6c`; collated from changes/14227-optional-stack-launcher-processes.md)
- **Summary:** Document CI Optional Stack slice 1 (Pinocchio gravity) and add runbook contract tests
- **Next step:** Shipped in PR #14923.

### DL-#12724 · S6 Catalogue Reason Now Names the Missing Arm on the NSX-97 UCM Split

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12724
- **Branch:** merged via #14761
- **PR:** #14761, #15068
- **Paths:** see #14761
- **Started:** 2026-11-06
- **Last verified:** 2026-11-14 (`36f6dbbf`; collated from changes/12724-nsx-95-campaign-catalogue-json-strategy.md)
- **Summary:** S6 catalogue reason now names the missing arm on the NSX-97 UCM split
- **Next step:** Shipped in PR #14761.

### DL-#12553 · NSX-04 Slice 1: Verified Subtalar Inversion/Eversion and MTP Flexion Torque Capacities (da Fonseca 2025, Saeki 2021); Hip and MTP-Extension Slots Keep ASSUMED With Stated Reasons

- **State:** shipped
- **Owner:** unassigned
- **Issue:** #12553
- **Branch:** merged via #14684
- **PR:** #14684, #14690, #14696, #14704, #14713, #14725, #14787, #14925, #15064
- **Paths:** see #14684
- **Started:** 2026-11-05
- **Last verified:** 2026-11-14 (`65360d12`; collated from changes/12553-verify-hip-adduction-and-rotation-torque.md)
- **Summary:** NSX-04 slice 1: verified subtalar inversion/eversion and MTP flexion torque capacities (da Fonseca 2025, Saeki 2021); hip and MTP-extension slots keep ASSUMED with stated reasons
- **Next step:** Shipped in PR #14684.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

## Field Reference

| Field           | Required                   | Notes                                                          |
| --------------- | -------------------------- | -------------------------------------------------------------- |
| `State`         | Always                     | One of the six states above                                    |
| `Owner`         | Always                     | Agent id from the fleet roster, or `unassigned`                |
| `Issue`         | While live                 | Governing GitHub issue; enforces the entry/issue join          |
| `Branch`        | `in_progress`, `in_review` | Enforces the entry/branch join                                 |
| `PR`            | Always                     | Number and state, or `not created`                             |
| `Paths`         | Always                     | Globs; drives silent-entry detection                           |
| `Started`       | Always                     | Drives cycle time                                              |
| `Last verified` | Always                     | Date plus SHA — the liveness signal                            |
| `Summary`       | Always                     | One or two sentences                                           |
| `Next step`     | While live                 | Exactly one action; if it needs two sentences, split the entry |
| `Parked`        | When `parked`              | Date plus reason                                               |

Never place credentials, tokens, or customer data in a development log.
No material development-log change — Bolt `np.linalg.norm` → `einsum` consolidation (#11073, #11074, #11076) is a behaviour-preserving micro-optimisation with no feature entry.
No material development-log change — Bolt `np.linalg.norm` → `sqrt(einsum)`/`math.sqrt(np.vdot)` consolidation (#11112, #11128, #11129) is a behaviour-preserving micro-optimisation with no feature entry.
No material development-log change — PyJWT floor/lock bump to 2.14.0 for OSV GHSA-w6j9-cwv2-h6wq (#11153) is a dependency-only change with no feature entry.
No material development-log change — urllib3 (2.8.0, CVE-2026-97687) and PyJWT (2.15.0, CVE-2026-101918) floor/lock bumps (#11191) are dependency-only changes with no feature entry.
No material development-log change — Bolt np.linalg.norm → sqrt(einsum) in rigidity.py (#11189) is a behaviour-preserving micro-optimisation with no feature entry.

## DL-#11357 — Pure Raw Shaft Fragment Derivative Diagnostics

- **Change:** Canonical immutable options/check/result contracts and public
  raw fragment analytic/central FD diagnostics; canonical camera/marker reuse.
- **Validation:** Repository fail-first scaffold regression followed by focused
  native-free synthetic tests, curated exports and fingerprint inclusion.
- **Limits:** No historical native/Library/render/fit operation, no objective
  or evidence-role change, and no clinical/scientific qualification.
- **Next Step:** Review typed core, then separately design authenticated
  workspace consumers using existing job/store/renderer boundaries.

## Authored Replay Impact Application Checkpoint — #11464

The normal React journey now previews impact from a verified authored replay,
with exact geometry/sample/flight declarations, bounded native execution,
cancellation, persisted run-ID recall and authenticated four-file download.
Saved results display their own assumptions and metrics. Canonical receipt and
trajectory owners remain unchanged; source/runtime/parent and artifact checks
fail closed. Research acceptance remains rejected, with authored seconds and
false physical/scientific qualification.

Root final validation: 125 focused Python cases pass with one Windows symlink
capability skip; one additional opt-in native case passes with actual MuJoCo,
clean process transport, impact solver and Rust flight (no substitutions).
The React cohort passes 104 cases; configured six-owner mypy, scoped lint and
format checks pass. Earlier failures and immutable agent freezes are retained
in the application review. Four-page LaTeX Application Supplement 1 is compiled
without final layout warnings and all pages visually reviewed; editable source,
PDF and the explicitly synthetic native bundle are saved on the local Desktop.
Earlier Methods V8 and the five-page extraction report remain unchanged.

See [Application Procedure](necromatcher-replay-impact.md) and its linked review/report. Tiger stays
within original frames 0–190 ([0,191)); Hogan stays [0,750). No historical
Library mutation, qualified match or new fleet-native-host delivery is established.
Remaining goal work includes improved historical matching, authored dynamics,
simulation golf integration and desktop parity. The full goal remains active.

## Native Desktop Replay Impact Checkpoint — #11487

The native Necromatcher tile now imports registered authored replays, runs the
shared bounded replay-impact session off the GUI thread, cancels owned jobs,
recalls retained run IDs after restart and exports an authenticated four-file
ZIP exclusively. Saved assumptions remain separate from editable declarations.
The public Shot Tracer handoff retains exact detached positions and provenance.

Root validation passes 146 focused Qt/viewer/parity cases without skips and
configured mypy on three production owners. One additional opt-in actual
MuJoCo/HDF5/clean-worker/impact/Rust native Qt case passes with Python exit 0.
The three final synthetic screenshots are readable and visually reviewed;
retained positions are verified, while OpenGL pixel accuracy is unverified.
Test-only font bootstrap uses existing Windows Segoe UI when the offscreen
font database is empty. The 120-second whole-test cap does not change the
60-second native budget; earlier timeout evidence is preserved.

The three-page Native Desktop Supplement 1 and editable LaTeX, screenshots and
intact synthetic bundle are saved on the local Desktop. See [Procedure](necromatcher-replay-impact.md)
and [Frozen Review](historical_capture/native-desktop-replay-impact-review.json). Tiger remains [0,191), Hogan [0,750).
Historical fits, scientific qualification, simulation golf and final overall
desktop parity acceptance remain open. No new fleet-native-host delivery or remote
CI-green claim is made. The full goal remains active.

## Local Research Admission Race Follow-Up — #11493

A concurrent request regression reproduced reused shot-ID admission after another
request prepared and cancelled during authentication. The API now applies the
same identity guard before and immediately after authentication. All 10 API
cases pass; a fresh additional actual MuJoCo/clean-worker/Rust API and Qt
acceptance passes in 24.56 s, with seven exact retained samples and no
re-simulation on recall. Configured mypy passes for the changed API owner.
The earlier 209-case cohort and 106 React cases remain separate checkpoint
evidence. The follow-up preserves unverified qualification and all historical
limits; the full goal remains active.

See [Follow-Up Review](historical_capture/local-research-golf-admission-race-review.json). Source inventory covers 6,697 src/test
files, with equal before/after maps; this is not a whole-repository baseline.

## User-Requested Handoff and Stop Checkpoint

The user requested a committed checkpoint and instructed this agent to stop
pursuing the goal after turnover. See [Next-Agent Turnover](necromatcher-next-agent-turnover.md) for the
exact implementation state at f0e14944, scoped work packets, public owners,
validation commands, historical evidence boundaries and remaining acceptance.
No proposed optimized video export or new fit was started. This documentation
checkpoint follows the published local research integration; the full goal is
unfinished and is to be paused after normal commit/push and handoff publication.

## Capture Registry Parent Integration - #11162 / #11172

Accepted-main synchronization preserves capture-registry, export and swing-comparison source changes. Record conflicts retain distinct entries from both branches. All 133 ordered matched-swing ledger rows and non-timestamp metadata are identical; retain accepted-main timestamp. Regenerate the monolith register from merged source. Scientific matching/dynamics acceptance remains separate.

Record validation: all five monolith-register tests pass; no duplicate primary headings or conflict markers; all 133 ledger rows and non-timestamp metadata preserved exactly; registry/export/comparison source unchanged. Broader capture tests are pending pinned Tools initialization; first collection attempt stopped on the missing submodule. Protected parent CI and full scoped rerun remain required before merge.

## Capture Comparison DRY Gate Remediation - #11162 / #11172

CI identified duplicated club endpoint selection in angular-speed and wrist metrics. A shared private helper preserves explicit endpoint precedence, measured-marker aliases, array identity and missing-endpoint/shaft-axis behavior. AGY Gemini Flash supplied the bounded extraction; root reviewed the exact diff. Existing focused regression suite passes 190 tests with 15 unavailable-private-workbook skips, zero errors/failures, natural exit 0. Ruff format/check and full DRY duplication gate pass without changing baselines. Regenerate the monolith register for the revised file length. Current-head protected CI remains required.

## Capture Leaderboard Suite Classification - #11162 / #11172

Protected CI verified the DRY extraction, then identified 14 net-new unmarked tests in TestPerCaptureLeaderboard. Mark this pure-Python class as unit; all 14 execute and pass under unit selection with no skips/errors. Full suite-marker ratchet passes without baseline changes. Broader focused capture/comparison qualification remains 190 passed and 15 unavailable-private-workbook skips. Protected current-head CI remains required before merge.

## Capture Comparison Inventory Integration - #11162 / #11172

CI at bd1ee9a8e passes repository structure and 20,546 unit cases; its only unit failure identifies five new swing_comparison paths missing from the shared-tools divergence inventory. Regenerate JSON and Markdown with the authoritative generator against the pinned Tools tree. All five paths classify as UD-only; existing overlap totals remain unchanged. All ten inventory tests pass with no errors/skips, including the committed-tree freshness test, and generator --check succeeds. Protected current-head CI remains required before merging.

### Selected Club Continuity Update (2026-10-03)

PR #11256 merged at `7db86ff173653a235ea63bb4af347af0a8a5ab83`. The eight verified Human ellipsoid MP4s remain on both Desktops in `Simscape_Matches_20261003_Club_Refined_1080p`; these are IK reconstructions. The source reference now presents this current selection before historical experiments.

The selected club fits were tested at all anchors and midpoints using independent native position, velocity and fresh 91-state checks. Sequential seeding accepted 1,306/1,307 tour states and 732/733 owner states. A targeted diagnostic recovered the owner's rejected original anchor from its saved closed-arm seed; the previous accepted pose reproduced an alternate branch. A complete anchor-seeded rerun exited naturally at 20:31:23 UTC and accepted all 733 owner states (367 anchors and 366 midpoints), while the tour still rejected midpoint 998 at 1.3847222 s. All original anchors matched within 1.73e-15 physical residual. The peak accepted dependent angular rates remain approximately 1,639 and 4,118 deg/s; sampled-state acceptance does not qualify between-sample branch continuity, dependent acceleration, anatomy, contact or forward dynamics.

The private production integration candidate passed 17 native pre-model contract cases (parent RED, candidate GREEN, 51 recorded checks) at 20:20:58 UTC. Numerical whole-output parity and positive-fit validation remain separate pending gates. Original failed runner receipts are preserved. A custom vector measurement block built successfully, but the eight-axis instrumented simulation hit the Home license's 1,000-nonvirtual-block limit. No all-35 measured torque or moving feedback-off replay claim follows from these tests. Original physical model files were not saved.

See the separate editable research reference and the extended aggregate `tangent_c2_checkpoint_20261003.json` for scope and provenance. The built-in LaTeX compiler still fails with `Unable to find standard directories for platform`; PDF compilation/page review are unverified. Canonical calculation inventory and governed manual release remain blocked; this update grants neither a release exemption nor scientific approval.

## F05d Native Manifold BoxFDDP Candidate (#11932; 2026-10-09)

A test-first MuJoCo 3.8/Crocoddyl 3.2.1 child now solves the same
contact-free floating-root/two-hinge physical state with a 16-dimensional
configuration/velocity tangent, exact native Euler action derivative,
bounded direct motor input and nonlinear native fallback admission. The
paired source-hashed receipt measures BoxFDDP against SciPy native shooting
under identical reference, truth, input box, horizon, iteration and
cooperative wall budget, alternates solver order and charges a common
Tools T01 bootstrap to both. Every applied input independently replays from
complete native integration state. This is synthetic controller evidence,
not a hard deadline, measured capture fit or full-body qualification;
the F05 parent and canonical manual publication remain blocked. Read
chapter 26 and `F05D_NATIVE_MANIFOLD_BOX_FDDP_TURNOVER.md`.

## Bounded Nonuniform Moco Refinement Plan (#12244; 2026-10-10)

The 20-cell synthetic Moco parent inputs are retained byte-for-byte and the
offline activation report reproduces from those committed source, request,
solution, control and replay-audit artifacts. A pure source-bound planner marks
the observed full Hermite-Simpson cells, caps one child pass at four marked
cells and 24 intervals, and rejects evidence/hash or grid mismatches. The worker
uses the normalized custom mesh and parent state seed, then linearly samples
the persisted controls at every child endpoint and midpoint without control
trajectory resampling. Pure planner/interpolation regressions pass. No native
child solve or replay was run; solver API/time-grid acceptance and all physical
qualification remain open. The context catalog could not be refreshed here:
its `vendor/ud-tools` junction escapes this checkout to the frozen
`feedback-drake-native-11838` donor, and the context CLI correctly rejects that
path. See the #12244 change fragment, chapter 13 and F07 ladder turnover.

The follow-up recomputes and compares the full plan before solver setup and
builds the post-solve mesh audit from the seeded grid in the child request.
Pure JSON-reload and tampering regressions pass; there is still no native child
solve or replay evidence.

The next correction uses the OpenSim 4.6-compatible states-table insertion
signature and restores exact declared initial values after interpolation; a
strict fake-SWIG test covers the behavior without running the native solver.

An installed OpenSim 4.6 API smoke also passed for the committed parent STOs:
the states table inserted into a native 43-row `MocoTrajectory`, controls were
PWL-seeded, and cold initial values matched exactly. It ran under the 30-second
owned-tree supervisor with 140 MB sampled peak, and called no model init, solve,
or replay. A first malformed empty-trajectory smoke crashed in a SWIG setter;
the corrected data-bearing constructor passed. Actual solver acceptance remains
unverified.
