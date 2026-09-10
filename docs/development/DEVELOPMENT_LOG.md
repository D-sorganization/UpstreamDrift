# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-08 by claude

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#9899 · Calibration Revision Status

- **State:** in_review
- **Owner:** codex
- **Issue:** #9899
- **Branch:** feat/9899-calibration-revision-status
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/9959
- **Paths:** reconstruct; rig command; capture_rig result evidence; tests.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (26 controls; six refactor checks; budgets pass)
- **Summary:** Calibration and reconstruction fingerprints invalidate stale outputs; preserve results.
- **Next step:** See capture_product_turnover.md; resolve #9954/#9959, then ruler UI.

### DL-#9952 · Native Camera Setup

- **State:** in_review
- **Owner:** codex
- **Issue:** #9952; parent #9906
- **Branch:** feat/9952-camera-setup
- **PR:** #9954
- **Paths:** src/tools/capture_rig/camera_setup\*.py; wizard/header; capability registry; tests and camera setup guide.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (563 Capture Rig tests, 16 atlas/workflow and 15 goal tests, six-module mypy and Ruff pass)
- **Summary:** Background discovery, stable named bindings, immutable plan revisions and optional wizard entry reuse the rig pipeline.
- **Next step:** Installed package passes; complete #9950/#9953 dependency and protected PR delivery.

### DL-#9926 · Unified Model and Video Analysis

- **State:** shipped
- **Owner:** codex
- **Issue:** #9926; children #9929, #9930, #9932, #9942
- **Branch:** feat/9942-metric-simulation-analysis; feat/9932-comparison-drawings
- **PR:** #9933, #9943 and #9945 merged; final documentation closure references #9926
- **Paths:** src/motion_capture/coaching; src/tools/capture_rig; src/tools/pose_studio
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (SELF completion record; #9945 merged2d41aba41; CI34440058289 succeeded;88 combined integration tests pass; preserved remote f58ee5b3/main300d96a1; reviewed contracts4824f370a; catalog20 runtime audit pass; Traceb6325dfe8 persistence17 pass; PR trackingf0fe99bd9; isolated push mypy14/Bandit/unit tests pass; map5a2e8d2c1; combined600 had only stale atlas hash, regenerated and56 focused tests pass; main d40956742; native785269090; 513 Capture Rig/geometry regressions, ten focused readout tests, four-source mypy, 126 reference/import regressions, five-source Trace mypy and 17 topology/import tests; native30 tests and four-source mypy, visual plane/editor inspected)
- **Summary:** Shared geometry and model-only coaching; see [ledger](unified_analysis_9926.md).
- **Evidence:** Comparison launch/save, PNG parity, snapshot video, cancellation and camera-evidence tests pass; Ruff/format, budgets and LoD pass; Driver comparison drawing UI and PNG inspected.
- **Next step:** Merge the documentation closure record and verify epic #9926 closure; runtime delivery is complete.

### DL-#9934 · Cross-Model Biomechanics Analysis

- **State:** shipped
- **Owner:** codex
- **Issue:** #9934
- **Branch:** feat/9934-biomechanical-analysis
- **PR:** #9941
- **Paths:** src/shared/python/biomechanics, src/api/routes/biomechanics.py, src/shared/python/analysis/biomechanics_display.py, src/shared/python/dashboard, ui/src/components/analysis
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (merged in #9941)
- **Summary:** Shared calibrated conventions and golf metrics across model inputs with explicit availability and configurable displays.
- **Evidence:** Focused 63-test suite passes; API compute/convert/display and web plot tests pass.
- **Next step:** Complete.

### DL-#9907 · Guided Capture Outcomes

- **State:** shipped
- **Owner:** codex
- **Issue:** #9907; #9908; epic #9906
- **Branch:** feat/9907-capture-goal-wizard
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/9931
- **Paths:** `src/tools/capture_rig`; capability graph/generator; matching tests and guide.
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (a74d5ebd2; 600 integrated regressions, normal push hooks, scoped mypy and Qt/browser review pass)
- **Summary:** Standard Qt outcome wizard shares typed map metadata and existing editors/readiness; capture-owned resume, optional My Clubs, background status and safe map-plan import.
- **Next step:** Shipped in #9931; retain advanced-route limits and calibration/fleet follow-up.

### DL-#9912 · Impact Shaft Provider Integration

- **State:** in_review
- **Owner:** codex
- **Issue:** #9912; parent #9703
- **Branch:** feat/9912-impact-provider-pin
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/9916
- **Paths:** vendor/ud-tools, tests/shared_contracts, docs/development/impact-acoustics
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (Tools 4dabe900c; main 5ada5e6a6; inventory repair SELF)
- **Summary:** Qualify the exact Tools shaft/theme provider; see PROVIDER_PIN_RESULTS.json.
- **Evidence:** 271 controls pass after preserving concurrent eacd69858 and restoring the pinned inventory. Existing inventory RED is retained. Isolated 32a8b36ec wheel and Qt construction pass with gui-tools; web assets omitted. Earlier evidence retained.
- **Next step:** Advance to the reviewed Tools #5143 descendant and qualify that exact pin; retain #8942 review exception and physical gates.

### DL-#9914 · C3D Reference Fitting

- **State:** shipped
- **Owner:** codex
- **Issue:** #9914
- **PR:** #9918 (merged)
- **Branch:** feat/c3d-reference-overlay-9914
- **Paths:** src/motion_capture
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`6f2d63325` merge; protected CI, 20 club assets)
- **Summary:** Fits, club/volume/handedness display; see [evidence](reference_fitting_epic.md).
- **Next step:** Retain fitting and display regressions.

### DL-#9905 · Player Bag and Capture Equipment

- **State:** in_progress
- **Owner:** codex
- **Issue:** #9905; epic #9902
- **Branch:** feat/9905-player-club-bag
- **PR:** not created
- **Paths:** club_data/player_clubs.py; rig/capture_notes.py and equipment.py; Capture Rig bag/editor/library; model/session.py; matching tests, guide and generated maps.
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (working tree;652 broad regressions and69 focused/map/parity checks pass; visual QA and3020-file LoD pass)
- **Summary:** My Clubs supports catalog/custom entries, partial measurements, notes, archive and capture assignment. Captures preserve club snapshots and editable-copy lineage; fit provenance retains evidence without unsupported club constraints. Dialog visual review passed with Segoe UI.
- **Next step:** Finish broad qualification and publish; connect wizard entry under #9906 before closing #9905.

### DL-#9915 · Verified Agent Context

- **State:** in_review
- **Owner:** codex
- **Issue:** #9915
- **PR:** #9920
- **Branch:** feat/issue-9915-agent-context
- **Paths:** `docs/agent_context`, `.github/workflows/ci-standard.yml`, `scripts/check_doc_size_budget.py`, `tests/ci`, `tests/scripts`
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10; final e83bd2e4 pins pass107 provider/seam, six context,31 atlas and12 navigation controls. The frontend-inclusive installed wheel passes launcher,31 calibration tests and pip check. Concurrent SPEC dedup4ded98ad6 is integrated; checker passes. Handoff records revisions and hashes.
- **Summary:** Twelve components and five reviewed integrations reuse the atlas and capture goals. Main276998030 is integrated; a required regression rejects divergent pip/source/Rust providers.
- **Next step:** Qualify final published Tools pins and required CI on PR #9920.

### DL-#9913 · Capture Journey Feedback and Detachable Views

- **State:** shipped
- **Owner:** codex
- **Issue:** #9913; epic #9906
- **Branch:** feat/9913-capture-journey
- **PR:** #9917 (merged)
- **Paths:** capture_rig source/tests, guide and parity registry
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`8fce9f238` merge;445 local tests and protected CI pass)
- **Summary:** Identity/history, linked help/provenance and retained detachable Qt views.
- **Next step:** Retain the capture journey regressions.

### DL-#9904 · Offline Club Source Catalog

- **State:** in_review
- **Owner:** codex
- **Issue:** #9904; epic #9902
- **Branch:** feat/9903-player-club-catalog
- **PR:** #9919
- **Paths:** club_data/catalog_sources.py, public_clubs.json, scripts/review_club_catalog.py and tests
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (SELF;69 tests, offline validator and scoped mypy pass)
- **Summary:** Three sourced builds, review diffs and preserved player overrides.
- **Next step:** Qualify protected checks on #9919.

### DL-#9903 · Attributed Club Catalog

- **State:** in_review
- **Owner:** codex
- **Issue:** #9903; epic #9902
- **Branch:** feat/9903-player-club-catalog
- **PR:** #9919
- **Paths:** club_data/, test_club_catalog.py and club_catalog.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (SELF; 63 catalog/legacy tests and scoped mypy pass)
- **Summary:** Attributed optional properties, explicit inference gates and lossless exchange.
- **Next step:** Qualify protected checks on #9919.

### DL-#9911 · Preview Discovery Failure Recovery

- **State:** shipped
- **Owner:** codex
- **Issue:** #9911
- **Branch:** feat/capture-guided-setup
- **PR:** #9910 (merged)
- **Paths:** src/tools/capture_rig/preview.py, tests/tools/capture_rig/test_preview.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`18c8f922e` merge; recovery tests pass)
- **Summary:** Report discovery imports/timeouts through preview status.
- **Next step:** Retain the preview recovery regressions.

### DL-#9898 · Common Reference Calibration

- **State:** in_progress
- **Owner:** codex
- **Issue:** #9898/#9900/#9909
- **Branch:** feat/9898-common-reference-sessions
- **PR:** #9946
- **Paths:** reference_calibration, wizard, lens adapter
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (732553479;28 pin/context checks pass)
- **Summary:** Reviewed paper/ruler calibration and guided recovery. [Evidence and limits](common_reference_calibration.md).
- **Next step:** Merge #9949 package repairs; #9946 merged126158943 before these fixes. Installed startup/wizard,29 calibration tests and pip check pass;660 source integration tests pass. Scientific release remains separate.

### DL-#9881 · Reference Timing and Camera Evidence

- **State:** in_review
- **Owner:** codex
- **Issue:** #9881 (advanced reference epic #9863)
- **Branch:** fix/9881-reference-timing
- **PR:** #9885
- **Paths:** src/motion_capture/reference, src/motion_capture/reconstruct/overlay3d.py, src/tools/capture_rig/reference_comparison.py, src/tools/capture_rig/reference_export.py, related tests and benchmark
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`SELF`; local qualification passed; protected checks pending)
- **Summary:** Immutable bounded event anchors, binary-search gap-aware sampling, actual camera/clock snapshots and stale-registration checks replace unsupported calibration assumptions.
- **Evidence:** 12 adverse regressions failed before repair; 46 combined tests and 14-module mypy pass. Sampling medians: 0.157/0.093/0.304 ms for120/1200/12000 frames. CI typing/budget corrections pass16 tests.
- **Next step:** Finish protected checks, then continue #9882/#9883.

### DL-#9879 · Comparison State and Export Lifetime

- **State:** in_review
- **Owner:** codex
- **Issue:** #9879 (advanced reference epic #9863)
- **Branch:** fix/reference-comparison-qualification
- **PR:** #9884
- **Paths:** src/motion_capture/reference/comparison.py, src/tools/capture_rig/reference_comparison.py, src/tools/capture_rig/swing_export_actions.py, tests/tools/capture_rig/test_reference_comparison_state.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`SELF`; local validation complete)
- **Summary:** Preserve exact unrelated layer/registration fields, reject bad saved records, and reuse the existing export controller for safe thread ownership and deferred close.
- **Evidence:** Nine adverse regressions preceded repair;31 comparison/cancellation/swing/coaching tests and three-module mypy pass.
- **Next step:** Architecture/DRY/LoD pass; await protected checks. Continue #9881, #9882 and #9883 before closing #9863.

### DL-#9865 · Reference Scene Registration & Synchronization

- **State:** in_review
- **Owner:** codex
- **Issue:** #9865 (advanced reference epic #9863)
- **Branch:** feat/9865-scene-registration
- **PR:** #9871
- **Paths:** src/motion_capture/reference/registration.py, src/motion_capture/reconstruct/overlay3d.py, related tests/docs
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`SELF`; local qualification complete)
- **Summary:** Calibrated scene registration, event-anchor and offset time synchronization, missing-joint gap mask preservation across bounded interpolation, distortion-aware camera projection, and 2D expert video homography without 3D claims.
- **Evidence:** 7 focused registration tests pass in tests/motion_capture/test_reference_registration.py. Strict round-trip serialization/deserialization validated. Projection tested with both pinhole and Brown-Conrady distortion. Ruff checks pass cleanly.
- **Next step:** Await CI Standard completion and automated squash merge of PR #9871.

### DL-#9864 · Expert Reference Asset Imports

- **State:** shipped
- **Owner:** codex
- **Issue:** #9864 (advanced reference epic #9863)
- **Branch:** feat/9864-reference-assets
- **PR:** #9870 (draft)
- **Paths:** src/motion_capture/reference, src/tools/capture_rig/reference_import.py, src/tools/capture_rig/reference_library_dialog.py, src/tools/capture_rig/library_dialog.py, src/shared/python/motion_pipeline/sources/c3d_adapter.py and related tests/docs/maps
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`90b147e10`; local qualification complete)
- **Summary:** Versioned portable reference assets retain explicit mapping, source hashes, timestamps and missing points; native library adds imports, notes/archive and background I/O. Expert videos remain linked 2D assets.
- **Evidence:** 30 integration tests pass, including real C3D and fresh-process isolation. Four native UI tests, eight-module mypy and architecture checks pass after layout/helper corrections.
- **Next step:** Integrate the drawing theme correction, then verify protected CI on #9870.

### DL-#9862 · Saved Coaching References

- **State:** shipped
- **Owner:** codex
- **Issue:** #9862 (product #9849)
- **PR:** #9869
- **Branch:** feat/9862-coaching-drawings
- **Paths:** src/motion_capture/coaching, src/tools/capture_rig/coaching_canvas.py, coaching_dialog.py, coaching_export.py and related integration/tests/docs
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`77e6bca88`; protected PR merged)
- **Summary:** Saved source-coordinate shapes; draw/edit/style/frame visibility, undo/redo, library/editor access and cancellable PNG/video export.
- **Evidence:** 49 registry/atlas and19 drawing/export tests;12-module mypy;3012-file LoD clean. Visual minimums:496px references,465px editor.
- **Next step:** Retain regression coverage.

### DL-#9860 · Capture Editing and Library

- **State:** shipped
- **Owner:** codex
- **Issue:** #9860, #9861 (product #9849)
- **PR:** #9868
- **Branch:** feat/9860-swing-editing
- **Paths:** src/motion_capture/rig/edits.py, ingest.py, src/tools/capture_rig/swing_editor.py, related tests and docs/development/capture_editing_integration.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`4c89892d7`; protected PR merged)
- **Summary:** Source-preserving trim/crop, capture notes/library, archive/storage/rename rollback, editable copies, cancellable export and timeline guards.
- **Evidence:** 300 integrated, 12 library/UI and 5 editor tests; eight-module mypy. Visual QA: 850x650, minimum492px.
- **Next step:** Retain regression coverage.

### DL-#9851 · Capture Responsiveness and Recovery

- **State:** shipped
- **Owner:** codex
- **Issue:** #9851, #9857 (epic #9849)
- **Branch:** `perf/9851-capture-responsiveness`
- **PR:** #9859
- **Paths:** src/tools/capture_rig/player.py, process_runner.py, benchmark_capture_responsiveness.py and cache/process tests; docs/development/capture_product_review.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`eaf8503ce`)
- **Acceptance:** Repeated frames decode once with isolated pixels; failed starts restore lifecycle/retry; benchmark limits documented.
- **Evidence:** 241 camera tests after cache; six focused tests after recovery; duplicate median 196.788 to 12.613 ms.
- **Next step:** Retain regression coverage.

### DL-#9850 · Generated Capability Atlas

- **State:** shipped
- **Owner:** codex
- **Issue:** #9850 (children #9852, #9853; product #9849)
- **Branch:** `feat/9850-capability-atlas`
- **PR:** #9856
- **Paths:** `scripts/capability_atlas/`, `scripts/generate_capability_atlas.py`,
  `src/config/capability_connections.json`, `ui/public/capability-atlas/`,
  `ui/src/components/simulation/LauncherDashboard.tsx`, `docs/architecture/`,
  `tests/scripts/test_capability_atlas.py`
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`9d6e6a872`)
- **Summary:** Generated C4-style context, workflow/artifact maps, searchable capabilities and Mermaid from existing registries.
- **Next step:** Retain regression coverage.

### DL-#9830 · Independent Shooting Accuracy

- **State:** shipped
- **Owner:** codex
- **Issue:** #9830
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/9841
- **Branch:** fix/9830-independent-shooting-convergence
- **Paths:** src/shared/python/optimization; docs/development/shooting_convergence_9830_turnover.md
- **Started:** 2026-09-08
- **Last verified:** 2026-09-09 (merged 28d9bf79e)
- **Summary:** Adaptive reference defects; 21 native Bioptim/Casadi 3.6.7 passes. Casadi 3.8 failure and physical limits remain in the linked turnover.
- **Next step:** Preserve recorded runtime/physical limits.

### DL-#9825 · Preserve Reviewed Manufactured Claims in Actual Registration

- **State:** shipped
- **Owner:** codex
- **Issue:** #9825
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/9826
- **Branch:** fix/9825-preserve-reviewed-claims
- **Paths:** docs/development/claim_preservation_9825_turnover.md; manufactured registration and evidence
- **Started:** 2026-09-08
- **Last verified:** 2026-09-09 (merged a410ae705)
- **Summary:** Preserves 328 reviewed outcomes; 128 strict contracts and 11 publication controls pass. Linked turnover retains full provenance and physical limits.
- **Next step:** Preserve reviewed claim boundaries.

### DL-#9787 · Manufactured Authority Runtime and Provenance

- **State:** shipped
- **Owner:** codex
- **Issue:** #9787
- **PR:** https://github.com/D-sorganization/UpstreamDrift/pull/9804
- **Branch:** fix/9787-manufactured-authority
- **Paths:** authority runtime pins, native provenance/CI contracts and manufactured_authority_9787_turnover.md.
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (PR merged as 736ec2189 from b8da0c024)
- **Summary:** Compatible native pins and runtime support merged. The merged revision differs from locally validated 6235789dc; its actual registration bypass and stale evidence require follow-up #9825. Historical test results do not certify differing merged bytes.
- **Next step:** Follow DL-#9825 for current preservation and native/publication qualification; retain the old branch as historical evidence.

### DL-#9478 · Launcher Registry Truth: `tools://` Provenance Scheme and Ready/Beta Maturity Gate

- **State:** in_review
- **Owner:** claude
- **Issue:** #9478
- **Branch:** `claude/issue-9478-registry-truth`
- **PR:** #9729 (open)
- **Paths:** `src/config/models.yaml`, `src/config/launcher_manifest.json`,
  `src/shared/python/config/tile_target_resolution.py`,
  `tests/config/test_tile_paths_resolve.py`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (Windows, Python 3.13.3, pytest 9.0.3)
- **Summary:** `provider: tools` entries in `src/config/models.yaml` and
  `src/config/launcher_manifest.json` now declare vendor provenance in the path
  itself (`tools://src/<tool>/...`). `ToolsVendorModelSourceProvider` strips the
  scheme at launch time; the local-repo and sibling providers reject it, so a
  vendor path can no longer masquerade as a repo-relative one. The registry gate
  (`tests/config/test_tile_paths_resolve.py`) gained a `ready`/`beta` maturity
  gate (`ready_maturity_gate` in
  `src/shared/python/config/tile_target_resolution.py`): a tile claiming
  launchability whose entry point does not resolve fails the gate, except a
  pinned-vendor target in a checkout that has not materialised the
  `vendor/ud-tools` gitlink (skip with a reason, never faked as success);
  the three vendor-materialised registry tests now carry a
  `requires_vendor_gitlink` skipif so a submodule-less CI checkout (the
  `unit-test-gate` job checks out no submodule) skips them through the
  repo's #9501 seam-skip convention instead of failing on the authority's
  fail-closed reason.
  Maturity corrections: the four `*_models_shared` sibling-folder tiles and
  `movement_optimizer` downgraded `ready` → `experimental` with sibling-folder
  caveats; `motion_capture` downgraded `beta` → `experimental` (argparse CLI,
  exits on a usage error as a tile); `myosim_suite`, `biomech_gait`,
  `biomech_sit_to_stand`, `chat_assistant`, and `tools_calculator_hub`
  descriptions now state what the tiles actually do. Focused verification:
  `python -m pytest tests/config/test_tile_paths_resolve.py` — 85 passed,
  5 skips, 0 failures.
- **Next step:** build the dedicated calculator surface for
  `tools_calculator_hub` (it still opens the shared Data Processor window).

### DL-#9612 · Video Upload Suffix Derived From Filename Allow-List

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9612`
- **PR:** #9720
- **Paths:** `src/api/routes/video.py`, `tests/unit/api/test_routes_video.py`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (`SELF`)
- **Summary:** Video analysis uploads no longer default temp files to `.mp4`;
  the container suffix is derived from the upload filename against the
  `SUPPORTED_VIDEO_SUFFIXES` allow-list in one place and unknown or missing
  extensions fail closed with a 400 before any side effect.
- **Next step:** Merge the protected PR for `#9612` and confirm CI runs green.

### DL-#9762 · `bioptim` Optimal-Control Backend and the Swing-Dynamics Fixes

- **State:** in_review
- **Owner:** claude
- **Issue:** #9762 (epic); prerequisites #9755, #9756, #9757, #9758, #9759, #9760, #9761
- **Branch:** `claude/fixes-epic-implementation-x2bu36`
- **PR:** [#9768](https://github.com/D-sorganization/UpstreamDrift/pull/9768) (open)
- **Paths:** `src/shared/python/optimization/ocp/`,
  `src/shared/python/optimization/casadi_backend.py`,
  `src/shared/python/optimization/model_provider.py`,
  `src/shared/python/optimization/backend_registry.py`,
  `src/shared/python/motion_pipeline/model_bridge.py`,
  `src/shared/python/estimation/`, `benchmarks/bioptim_parity.py`,
  `docs/adr/0050-optimizer-backend-registry-and-bioptim.md`,
  `docs/estimation/bioptim_parity.md`, `docs/issues/EPIC_BIOPTIM_OCP_INTEGRATION.md`,
  `.github/workflows/ci-optional-stack.yml`, `scripts/config/architecture_budget.json`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** Adopts `pyomeca/bioptim` as an opt-in optimal-control layer
  driven by UpstreamDrift's own CasADi dynamics through bioptim's custom-model
  protocol (no biorbd, no conda), and fixes the defects that made the existing
  dynamic backends unphysical. The URDF bridge and model provider now emit
  anthropometric link inertials so torque limits mean something (#9755); the
  CasADi backend gains mass-matrix, forward-dynamics and RK4 kernels, a
  `dynamics_defect` diagnostic and a real multiple-shooting transcription, and
  its finite-difference path is deprecated (#9756); the MAP estimators count and
  can refuse non-finite residuals (#9757) and gate free parameters on
  identifiability (#9758); the optional-stack lane gained CasADi, Crocoddyl and
  bioptim legs (#9759); `backend_registry` plus ADR-0050 assign each of six
  backends a problem class (#9760). Phases 0-3 of the epic are implemented and
  tested: compat shims, `SymbolicSwingModel` validated against Pinocchio,
  `SwingBioModel`, the clubhead-speed OCP with a parity benchmark, and the
  keypoint-tracking OCP. Two structural findings are recorded rather than
  hidden: maximising terminal speed is concave and does not converge in any
  backend once the dynamics are enforced (so the OCP defaults to a convex
  target-speed objective), and the six-marker set cannot observe the full
  seven-DOF chain (hip and trunk rotation are an exact null direction), so every
  tracking solve reports what it could not see.
- **Next step:** Confirm the `tests` lanes on PR #9768, then open the phase-4
  parameter-block entry.

### DL-#9783 · Reviewed Renderer Provider Compatibility

- **State:** in_progress
- **Owner:** codex
- **Issue:** #9783
- **Branch:** `fix/9783-reviewed-renderer-reference`
- **Paths:** `tests/shared_contracts/test_tools_provider_contracts.py`, `docs/development/renderer_reference_9783_turnover.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (15 provider tests pass against both pinned and candidate Tools; four strict pair refusals)
- **Summary:** Reproduced the candidate's exact old-hash failure before accepting the two reviewed source/hash pairs. Tolerances, immutable provider origin and the current vendor pin remain strict.
- **Next step:** Complete normal protected delivery, then verify the Tools downstream consumer lane against merged UpstreamDrift.

### DL-#9482 · Launcher Tile Logo Families and Registry Gate

- **State:** in_review
- **Owner:** claude
- **Issue:** #9482
- **Branch:** `claude/issue-9482-icon-families`
- **PR:** #9725 (open; `Fixes #9482`)
- **Paths:** `src/config/launcher_manifest.json`, `assets/logos/**`,
  `scripts/check_launcher_logo_families.py`,
  `tests/config/launcher_manifest/test_logo_families.py`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (`191351bdf`)
- **Summary:** Broke the launcher grid's worst logo reuse (data_explorer x9,
  golf_logo x7) by reassigning 16 tiles to distinct existing SVG assets
  (including verbatim copies of the Sidekick and movement-optimizer icons into
  `assets/logos/`), declared a data-derived family rule — `engine:<engine_type>`
  else `category:<category>` — and added `scripts/check_launcher_logo_families.py`
  plus a focused pytest gate that fails any logo shared outside one family or
  shared without a documented declaration.
- **Next step:** Address review feedback on PR #9725 and merge when approved.

## Shipped (Last 90 Days)

### DL-#9953 · Scalar Parameter Bounds

- **State:** shipped
- **Owner:** codex
- **Issue:** #9953
- **Branch:** fix/9953-scalar-bounds
- **PR:** #9955
- **Paths:** src/shared/python/optimization/ocp/parameter_ocp.py; parameter OCP tests; calculation inventory.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (30 OCP tests pass; two optional Pinocchio checks skipped)
- **Summary:** Explicit constant interpolation gives each shared parameter a single-column bound, preserving limits and locked values.
- **Evidence:** New actual-SDK regression failed with (1,3) before the fix; all five parameter tests pass after it, including unchanged numerical recovery tests. The optimization unit suite also passes after its bounds stub adopted the SDK add API. CasADi 3.6.7 and pinned Bioptim fdafe4d9; no tolerance changes.
- **Evidence:** All CI passed, including Linux Bioptim OCP and both manufactured checks. Merged as32410babfd1e4741fa0c53bf05dd8403a51bf233.
- **Next step:** None for #9953; capture integration continues in #9950/#9954.

### DL-#9882 · Comparison Rendering and Export Qualification

- **State:** shipped
- **Owner:** codex
- **Issue:** #9882
- **Branch:** fix/9882-comparison-rendering
- **PR:** #9896 (integrates #9889)
- **Paths:** src/tools/capture_rig/reference_rendering.py, reference_export.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Shared compositor, coverage-aware expert homography and staged exports delivered with qualified product #9896.

### DL-#9894 · Scoped Ubuntu CI Dependencies

- **State:** shipped
- **Owner:** codex
- **Issue:** #9894
- **PR:** #9896 (merged; prior proposals preserved)
- **Paths:** .github/workflows/ci-standard.yml, scripts/ci/, tests/scripts/
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Signed per-job APT sources preserve shared-runner configuration; six Bash regressions and standard CI pass.

### DL-#9892 · Fleet Guide Compatibility

- **State:** shipped
- **Owner:** codex
- **Issue:** #9892
- **PR:** #9896 (merged; prior proposals preserved)
- **Paths:** scripts/check_agent_docs_consistency.py, tests/architecture/test_check_agent_docs_consistency.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Central managed guidance and legitimate external/optional paths pass without hiding real missing-file failures.

### DL-#9883 · Instructor Reference Alignment Workspace

- **State:** shipped
- **Owner:** codex
- **Issue:** #9883
- **PR:** #9896 (merged; prior proposals preserved)
- **Paths:** src/tools/capture*rig/reference*\*.py, styling.py, tests/tools/capture_rig/
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Responsive placement/timing/notes controls, event alignment, revision checks, preview/export parity and native layout evidence are delivered.
- **Evidence:** Qualified candidate equals merged tree; standard unit gate passed 14,821 tests.

Entries stay here for 90 days after merge, then move to the archive.

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

### DL-#9648 · RTMPose ONNX Pose Estimator Behind the Registry

- **State:** in_review
- **Owner:** claude
- **Issue:** #9648
- **Branch:** `claude/issue-9648-pose-backends`
- **PR:** #9739 (open; `Fixes #9648`)
- **Paths:** `src/shared/python/pose_estimation/rtmpose_onnx_estimator.py`, `src/shared/python/pose_estimation/rtmpose_models.py`, `src/shared/python/pose_estimation/registry.py`, `src/motion_capture/rig/ingest.py`, `src/motion_capture/reconstruct/layouts.py`, `pyproject.toml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`197e0a942`)
- **Summary:** Registers `rtmpose_onnx` (SimCC decode via onnxruntime, COCO-17/Halpe-26, whole-frame letterbox) with `capture_source=False`; adds the optional `pose-onnx` extra; pins the official OpenMMLab ONNX model URLs/sizes with digests PENDING OWNER APPROVAL; teaches `RegisteredFrameEstimator` to honour instance-level `LANDMARK_MAP`/`LAYOUT_NAME`; extends `layouts.py` with the Halpe-26 `hip`→`mid_hip` alias.
- **Next step:** owner approves and verifies the pinned RTMPose model download (run the `rtmpose_models` command once, pin both SHA-256 digests), then qualify `rig compare` against MediaPipe on take 2.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

### DL-#9499 · Spec Check Reminder Fail-Safe Extraction

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9499`
- **PR:** #9719 (open; `Fixes #9499`)
- **Paths:** `.github/workflows/spec-check.yml`, `scripts/post_spec_reminder.py`, `tests/ci/test_spec_check_workflow.py`
- **Summary:** The `Verify SPEC.md freshness` job posts its SPEC reminder
  through a fail-safe script instead of an inline `github-script` heredoc, so a
  reporting failure prints the diagnostic into the job log while the
  `always()`-guarded staleness step still fails the run.
- **Next step:** Merge the PR filed from `claude/issue-9499-spec-freshness`
  after required checks pass.

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

### DL-#9542 · Bunker Exit State Consistency, Provenance, and Result Envelope

- **State:** in_review
- **Owner:** claude
- **Issue:** #9542
- **Branch:** `claude/issue-9542-exit-envelope`
- **PR:** #9728 (open)
- **Paths:** `src/bunkershot3d/ball/**`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (`SELF`)
- **Summary:** `SandDelivery` now refuses contradictory exit speed/vector pairs and owns copies of list-supplied exit vectors so post-construction mutation cannot invalidate the frozen record; the `to_post_impact_state` boundary carries explicit `ExitVectorProvenance` labels, and `PostImpactEnvelope` wraps the flight handoff with the validity verdict, F0 tier, per-group frames, the proper `HEAD_FRAME_TO_FLIGHT_TRANSFORM`, a schema version, and a SHA-256 source digest with JSON round trip. Reflection rejection itself was already delivered by PR #9574 and is not redone.
- **Next step:** Open the protected PR to `main` with `Fixes #9542`, label `agent:claude`, and RED/GREEN evidence in the body.

### DL-#9409 · Always-On Quality Gate Lane and Conftest Src-Pivot Guard

- **State:** in_review
- **Owner:** `claude`
- **Issue:** [#9409](https://github.com/D-sorganization/UpstreamDrift/issues/9409)
- **PR:** opened from `claude/issue-9409-always-on-gate` immediately after the `SELF` commit (body starts `Fixes #9409`)
- **Paths:** `.github/workflows/ci-standard.yml`, `tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py`, `docs/workflows/WORKFLOW_TRACKING.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** CI Standard gains an always-on, ≤10-minute `always-on-unit-lane` (verify_installation import smoke over the shared Tools alias roots, top-level smoke tests, contract tests) that `quality-gate` requires `success` on every PR including docs-only ones; a repo-hygiene guard forbids any conftest from pivoting `sys.modules["src"]` directly (must use `EngineSrcPivot`). Deferred on #9409: main-branch cancel exemption (RM campaign) and nightly cross-engine dedupe (#8725/#9002).
- **Next step:** Verify the first CI run of the PR executes `always-on-unit-lane` to `success` within its 10-minute budget.

### DL-#9607 · Authority Runtime Native Library Bootstrap for Cmeel Pinocchio Wheels

- **State:** in_review
- **Owner:** claude
- **Issue:** #9607
- **PR:** #9726 (open; `Fixes #9607`)
- **Branch:** `claude/issue-9607-pinocchio-abi`
- **Paths:** `scripts/research/proximal_distal_energy/articulated_native_runtime.py`, `scripts/research/proximal_distal_energy/run_articulated_manufactured_solution.py`, `tests/research/test_articulated_native_runtime.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** Resolves the authority lane's `liburdfdom_sensor.so.4.0` import failure by resolving `cmeel.prefix/lib` from the live venv, verifying the locked sonames with an explicit DbC diagnostic, and re-execing the authority profile with `LD_LIBRARY_PATH` prepended before `import pinocchio`.
- **Next step:** Verify the `articulated-manufactured-authority` job in `ci-optional-stack.yml` imports pinocchio, then regenerate the committed publication authority record from a byte-identical locked Linux run.

### DL-#9533 · Test-Only Extras Reachable From the Dev Lock

- **State:** in_review
- **Owner:** claude
- **PR:** #9716 (open; `Fixes #9533`)
- **Paths:** `pyproject.toml`, `.github/workflows/lock-refresh.yml`,
  `requirements*.lock`, `environment.yml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** `openpyxl` and `imageio` were declared only in the `gui-tools`
  and `pose` extras, so the dev-compiled `requirements-dev.lock` never installed
  them and ~24 CI tests failed on import. Both now resolve through the `dev`
  extra; lock regeneration is delegated to a dispatch-only `lock-refresh.yml`
  workflow that runs `make sync-deps` on ubuntu + Python 3.12 and opens a PR,
  since Windows/WSL cannot regenerate correctly (#9533).
- **Next step:** Land the regenerated locks. PR #9768 carries them already
  (regenerated with `make sync-deps` on Python 3.12, the exact interpreter
  the `dependency-consistency` gate uses), because that gate is red on
  `main` and blocks every open PR until the locks catch up; `lock-refresh.yml`
  stays the standing mechanism for the next drift.

### DL-#9091 · Phantom-Guard Rule-3 False Positive on Shallow Base Fetch

- **State:** in_review
- **Owner:** claude
- **PR:** #9717 (open; `Fixes #9091`)
- **Paths:** `.github/workflows/anti-phantom-merge.yml`, `scripts/ci/check_phantom_guard_paths.py`, `tests/scripts/test_check_phantom_guard_paths.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** The phantom-guard rule-3 closes-issue path-membership check
  produced a false positive when a shallow base fetch defeated `git merge-base`;
  the check is now extracted into `scripts/ci/check_phantom_guard_paths.py`,
  which resolves changed files from local git diff first and falls back to the
  GitHub API changed-file list (`PR_CHANGED_FILES` / `gh pr view --json files`)
  when merge-base fails, failing closed with a diagnostic when neither source
  is available. Workflow rule-3 block invokes the script; rules 1/2/4 unchanged.
- **Next step:** Observe the first post-merge `phantom-guard` run on a PR
  whose base fetch is too shallow for merge-base to confirm rule 3 defers to
  the API list.

### DL-#9484 · Impact Explorer Web Route Has a CI Bundle Producer

- **State:** in_review
- **Owner:** W4_9484 (agent claude)
- **Issue:** #9484
- **PR:** #9724 (open)
- **Paths:** `.github/workflows/ci-standard.yml`, `scripts/check_declared_route_producers.py`, `tests/scripts/test_declared_route_producers.py`, `docs/workflows/WORKFLOW_TRACKING.md`
- **Branch:** `claude/issue-9484-impact-web-build`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** The `rate_of_closure` tile declared `web.mode: route` for `/tools/impact-explorer` but no pipeline built `vendor/ud-tools/src/rate_of_closure/web/dist`, so a clean checkout served the honest fallback. CI Standard now builds the bundle from the pinned Tools tree with `npm run build -- --base=/impact-explorer-app/`, and `scripts/check_declared_route_producers.py` fails any declared route that no pipeline produces. Shipping the bundle inside the wheel/image remains an open maintainer decision (#9417).
- **Next step:** Merge PR for #9484, then decide the bundle distribution channel (wheel/image vs fetched Tools release artifact, #9417).

## Shipped (Last 90 Days)

Entries stay here for 90 days after merge, then move to the archive.

## Archive

Older entries live in `DEVELOPMENT_LOG_ARCHIVE_<year>.md`.

### DL-#9249 · UI: Pin @vitejs/Plugin-React to ^5 Until Vite 8

- **State:** in_review
- **Owner:** claude
- **PR:** #9718 (open; `Fixes #9249`)
- **Paths:** `.github/dependabot.yml`, `ui/README.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`7cdbb0a3d`)
- **Summary:** Dependabot ignores `@vitejs/plugin-react` major updates
  because 6.x needs Vite 8 (Vite 7 exports no `./internal`); the pairing
  constraint is documented in `ui/README.md`.
- **Next step:** Merge the guard PR; revisit the paired vite@8 +
  plugin-react@6 upgrade once `vitest`/`@react-three/*` are Vite-8 ready.

### DL-#9470 · Launch-Monitor Analysis Handlers Onto the Async_Action Worker

- **State:** in_review
- **Owner:** claude
- **Issue:** #9470
- **PR:** #9742
- **Paths:** `src/tools/launch_monitor_analytics/gui.py`, `src/tools/launch_monitor_analytics/_embed_adapter.py`, `tests/ui/tools/launch_monitor/test_async_actions.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF` on this branch: `tests/ui/tools/launch_monitor` + `tests/tools/test_async_action.py` all passing, PyQt6 6.11.0 offscreen)
- **Summary:** All seven analysis handlers (`treatment`, `relationship`, `multivariate`, `model`, `comparison`, `dispersion`, `trend`) now run their compute on the #8880 `async_action` worker via one shared `AsyncActionBar`; synchronous `present(compute())` paths kept; embed adapter `cleanup()` cancels and joins the worker. First slice of the #9470 tool checklist; the remaining tools are follow-ups.
- **Next step:** Merge PR #9472 (the #8880 mechanism) before this branch — it is stacked on `readiness/p2-8880-async-action-worker`.

### DL-#9387 · Unit-Gate Worker Corruption: `src`-Identity Sentinel and Leak Fixes

- **State:** shipped
- **Owner:** claude
- **PR:** #9741 (merged; `Fixes #9387`)
- **Paths:** `tests/unit/repo_hygiene/test_src_identity_sentinel.py`,
  `tests/imports/test_gui_import_boundaries.py`,
  `tests/integration/test_golf_launcher_integration.py`,
  `tests/unit/engines/pinocchio/test_tasks.py`,
  `tests/unit/test_ux_enhancements.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`c70d5ddae`)
- **Summary:** Static audit of `tests/` (conftests excluded) found 83
  `sys.modules['src*']` mutation sites in 23 files with 0 unambiguous
  leakers; the judgment-call leaks (ux-enhancements fixture, pinocchio
  tasks fixture, GUI import-boundary drops, golf-launcher pops) are now
  explicitly snapshot/restored, and a runtime sentinel runs each of the
  four documented victim files in a serial subprocess asserting
  `sys.modules['src']` identity and the `src.*` namespace are unchanged.
- **Next step:** Watch this PR's `quality-gate` run once after opening;
  on green, protected squash merge closes #9387, then rerun the flaky
  `unit-test-gate` histories of #9384/#9374/#9404 to confirm no fresh
  worker-corruption victims appear.

### DL-#9494 · Resolve the CLAUDE.md `--no-verify` Contradiction by Fixing the Windows Hook Environment

- **State:** shipped
- **Owner:** claude
- **Issue:** #9494
- **Branch:** `claude/issue-9494-precommit-env`
- **PR:** #9744
- **Paths:** `CLAUDE.md`, `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`, `SPEC.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`dbc6727aa`)
- **Summary:** CLAUDE.md forbade `git commit --no-verify` while agents on
  Windows reported every pre-commit invocation failing (hook virtualenvs
  targeting Python 3.11, absent from the workstation). Investigation found no
  interpreter pin left in `.pre-commit-config.yaml` (`default_language_version:
python: python`, 3.11 pin removed by #1792/#2720), and on Python 3.13.3 every
  commit-stage hook plus pre-push `mypy`/`bandit` passes after a from-scratch
  environment build. Option (a) of the issue is therefore satisfied; CLAUDE.md
  now documents the resolved environment and states that the `--no-verify`
  prohibition stands on Windows with no blanket exception.
- **Next step:** Record CI on PR #9744; on merge, confirm the protected-main
  sync lands the resolved hook environment note.

### DL-#9476 · Re-Vendor the Corrected Spec Merge Driver and Pin Drift

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9476`
- **Branch:** `claude/issue-9476-driver-wiring`
- **PR:** #9736 (open, in_review; re-vendor leg #9734)
- **Paths:** `scripts/install_spec_merge_driver.py`,
  `shared_scripts/spec_changelog.py`,
  `tests/unit/scripts/test_spec_merge_driver_vendor_drift.py`,
  `scripts/setup_hooks.py`,
  `tests/unit/scripts/test_setup_hooks_wires_spec_merge_driver.py`, `SPEC.md`,
  `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`875cd501e`)
- **Summary:** The vendored installer still stamped the withdrawn
  merge-abort claim into `$GIT_COMMON_DIR/info/attributes` and both vendored
  copies had no drift detection. Re-vendored both files byte-identical from
  Repository_Management#1521's corrected copies (verified: the false
  `ATTRIBUTE_BLOCK` text is gone; the repo-wide scan's only surviving match
  in UD-owned files is the correction narrative quoting the wrong claim to
  refute it, plus true statements about the half-configured state), and added
  `tests/unit/scripts/test_spec_merge_driver_vendor_drift.py` pinning SHA-256
  digests against the upstream reference so future divergence fails loudly.
  The behaviour change the issue requests followed on this branch:
  `scripts/setup_hooks.py` (the documented local-automation entry point)
  now calls the vendored installer, so the documented setup registers the
  `spec-rows` driver, and the installer docstring names this repository's
  entry point instead of Repository_Management's.
- **Next step:** Land PR #9734, then the wiring PR.

### DL-#9533 · Test-Only Extras Reachable From the Dev Lock

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9533`
- **Branch:** `claude/issue-9533-test-extras`
- **PR:** #9716 (open, in_review)
- **Paths:** `pyproject.toml`, `.github/workflows/lock-refresh.yml`, `requirements*.lock`, `environment.yml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`289b3aa`)
- **Summary:** `openpyxl` and `imageio` were declared only in the `gui-tools` and `pose` extras, so the dev-compiled `requirements-dev.lock` never installed them and ~24 CI tests failed on import. Both now resolve through the `dev` extra; lock regeneration is delegated to a dispatch-only `lock-refresh.yml` workflow that runs `make sync-deps` on ubuntu + Python 3.12 and opens a PR, since Windows/WSL cannot regenerate correctly (#9533).
- **Next step:** Dispatch `.github/workflows/lock-refresh.yml` from `main` once this PR merges, then confirm the `ci-standard.yml` dependency-consistency freshness gate and the 24 previously failing tests go green.

### DL-#9733 · Fail Fast on the Uninitialized Vendored Tools Fallback

- **State:** in_review
- **Owner:** claude
- **Issue:** #9733
- **Branch:** `claude/issue-9733-fail-fast`
- **PR:** #9743 (open; `Fixes #9733`)
- **Paths:** `src/__init__.py`, `tests/unit/repo_hygiene/test_src_fallback_fail_fast_9733.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`160f9b759`)
- **Summary:** When `vendor/ud-tools` is uninitialized, `src/__init__.py`'s
  fallback registration mistook UpstreamDrift's own aliased `shared.python` copy
  for an installed Tools distribution, installed `_VendoredToolsFallbackFinder`,
  and livelocked pytest collection in meta-path `find_spec` recursion. The
  registration probe now raises an actionable ImportError naming the remediation
  command before any finder is installed; the initialized path is unchanged.
- **Next step:** Record CI on PR #9743; on green, protected squash merge
  closes #9733.

### DL-#8943 · Cache API CPU Work Off the Event Loop

- **State:** in_review
- **Owner:** W3_8943 (agent `claude`)
- **Issue:** #8943
- **Branch:** `claude/issue-8943-api-cache`
- **PR:** #9727 (open)
- **Paths:** `src/api/routes/analysis_plots.py`, `src/api/routes/model_explorer.py`,
  `src/api/routes/models.py`, `src/api/routes/launch_monitor_analytics.py`,
  `src/api/routes/_route_utils.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`SELF`)
- **Summary:** `GET /analysis/plot-data/{plot_type}` now builds the
  `AnalysisOrchestrator` once per recorder identity (LRU invalidated when the
  recorder is replaced) and serves per-plot-type results from an LRU, computed
  under `anyio.to_thread.run_sync`. Model-explorer and models URDF handlers read
  and parse through `functools.lru_cache` helpers keyed by
  `(resolved_path, st_mtime_ns, st_size)` (shared `urdf_file_key` helper), also
  run in a worker thread. `launch_monitor_analytics` defers `import pandas` into
  its seven handlers so API boot no longer pays the pandas import.
- **Next step:** Merge the PR and confirm CI route/lazy-import gates pass on
  `main`.

### DL-#9631 · Vendor Pin Carries the Tools#5048 Alias-Predicate Fix

- **State:** in_review
- **Owner:** claude
- **Issue:** `#9631`
- **Branch:** `claude/issue-9631-vendor-pin`
- **PR:** #9722 (open; `Fixes #9631`)
- **Paths:** `vendor/ud-tools`, `tests/unit/repo_hygiene/test_pinned_import_alias_contract.py`
- **Started:** `2026-09-08`
- **Last verified:** `2026-09-08` (`e4c47751f`)
- **Summary:** The `vendor/ud-tools` pin `eab74a901a` already carries the Tools#5049
  flattened-install fix (`f8b94bfe` is an ancestor), so the v2.1.2 wheel defect is fixed at
  the pin; this entry lands the repository's own TDD contract test asserting the pinned
  predicate in both layouts and records that the pin must not be rewound.
- **Next step:** maintainer re-cuts the 2.1.3 release via tag/workflow dispatch after the PR merges.

## Shipped (Last 90 Days)
