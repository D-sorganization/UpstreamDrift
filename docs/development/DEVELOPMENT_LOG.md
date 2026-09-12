# Development Log — UpstreamDrift

State table for every feature in flight in this repository. Update entries
**in place**; never append dated sections. One entry per feature, from proposal
to ship. See the `development-logs` section of `AGENTS.md` for the binding rules
and `shared_scripts/development_log.py` for the validator.

- **Portfolio:** golf
- **WIP limit:** 8
- **Last audited:** 2026-09-12 by claude

## States

`proposed` → `in_progress` → `in_review` → `shipped`, with `parked` reachable
from any live state and `abandoned` from `parked`. `shipped` never returns to
`in_progress`; open a new entry instead.

## Active

### DL-#8766 · Burn Down AI Assistant & Chat Session Quarantine Debt

- **State:** in_progress
- **Owner:** claude
- **Issue:** #8766
- **Branch:** fix/ai-chat-session-tz-and-submodules
- **PR:** pending
- **Paths:** src/shared/python/ai/gui/session_manager.py, src/shared/python/ai/gui/assistant_panel.py, scripts/config/unit_gate_quarantine.json, SPEC.md, docs/development/DEVELOPMENT_LOG.md
- **Started:** 2026-09-12
- **Last verified:** 2026-09-12 (`c578ca942`)
- **Summary:** Normalize timestamps in ChatSessionManager.list_sessions() for mixed naive/aware datetimes, re-export decomposed AI assistant submodules in assistant_panel.py, and burn down 5 quarantined test node IDs in unit_gate_quarantine.json.
- **Next step:** Open PR, pass CI, and merge.

## Shipped (Last 90 Days)

### DL-#8695 · DRY Duplication Quarantine Tightening

- **State:** shipped
- **Owner:** claude
- **Issue:** #8695
- **PR:** #10005 (merged)
- **Paths:** scripts/config/dry_duplication_quarantine.json, SPEC.md, docs/development/DEVELOPMENT_LOG.md
- **Started:** 2026-09-11
- **Last verified:** 2026-09-12 (`c578ca942`)
- **Summary:** Pruned 72 dead quarantined fingerprints across supported scanner runtimes (666 -> 594); no entry raised, baseline not regenerated.
- **Evidence:** All CI passed including phantom-guard, doc-governance, code-quality, unit-test-gate, optional-stack-check; merged to main at c578ca942.

### DL-#9747 · Signed Release Tag Enforcement and Verification

- **State:** shipped
- **Owner:** claude
- **Issue:** #9747
- **PR:** #10008 (merged)
- **Paths:** .github/workflows/release.yml, docs/operations/release-runbook.md, tests/ci/test_ci_infrastructure.py, SPEC.md, docs/development/DEVELOPMENT_LOG.md
- **Started:** 2026-09-12
- **Last verified:** 2026-09-12 (`c893a73ad`)
- **Summary:** Enforce cryptographic signature verification on production release tags in release.yml and update release runbook.
- **Evidence:** All CI passed including quality-gate and unit-test-gate; merged to main at c893a73ad.

### DL-#9953 · Scalar Parameter Bounds

- **State:** shipped
- **Owner:** codex
- **Issue:** #9953
- **PR:** #9955 (merged)
- **Paths:** src/shared/python/optimization/ocp/parameter_ocp.py; parameter OCP tests; calculation inventory.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`32410babfd`)
- **Summary:** Explicit constant interpolation gives each shared parameter a single-column bound, preserving limits and locked values.
- **Evidence:** All CI passed, including Linux Bioptim OCP and both manufactured checks. Merged as32410babfd1e4741fa0c53bf05dd8403a51bf233.

### DL-#9952 · Native Camera Setup

- **State:** shipped
- **Owner:** codex
- **Issue:** #9952; parent #9906
- **PR:** #9954 (merged)
- **Paths:** src/tools/capture_rig/camera_setup\*.py; wizard/header; capability registry; tests and camera setup guide.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`4b305df1d1`)
- **Summary:** Background discovery, stable named bindings, immutable plan revisions and optional wizard entry reuse the rig pipeline.

### DL-#9934 · Cross-Model Biomechanics Analysis

- **State:** shipped
- **Owner:** codex
- **Issue:** #9934
- **PR:** #9941 (merged)
- **Paths:** src/shared/python/biomechanics, src/api/routes/biomechanics.py, src/shared/python/analysis/biomechanics_display.py, src/shared/python/dashboard, ui/src/components/analysis
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (`300d96a1b2`)
- **Summary:** Shared calibrated conventions and golf metrics across model inputs with explicit availability and configurable displays.
- **Evidence:** Focused 63-test suite passes; API compute/convert/display and web plot tests pass.

### DL-#9926 · Unified Model and Video Analysis

- **State:** shipped
- **Owner:** codex
- **Issue:** #9926; children #9929, #9930, #9932, #9942
- **PR:** #9933 (merged)
- **Paths:** src/motion_capture/coaching; src/tools/capture_rig; src/tools/pose_studio
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (`f04aa1a570`)
- **Summary:** Shared geometry and model-only coaching; see [ledger](unified_analysis_9926.md).
- **Evidence:** Comparison launch/save, PNG parity, snapshot video, cancellation and camera-evidence tests pass; Ruff/format, budgets and LoD pass; Driver comparison drawing UI and PNG inspected.

### DL-#9921 · Native Simscape Tour-Average Matching

- **State:** shipped
- **Owner:** codex
- **Issue:** #9921; implementation #9924, #9925, #9927
- **PR:** #9948 (merged)
- **Paths:** Simscape MATLAB motion_matching/shared, model initialization, shared Python prefix_fit, tests and simscape_tour_matching docs
- **Started:** 2026-09-09
- **Last verified:** 2026-09-11 (`156fcf2fa5`)
- **Summary:** Reproducible forward-dynamics matching with fixed geometry and continuous polynomial torques; native starting pose verified, full swing fit outstanding.

### DL-#9915 · Verified Agent Context

- **State:** shipped
- **Owner:** codex
- **Issue:** #9915
- **PR:** #9920 (merged)
- **Paths:** `docs/agent_context`, `.github/workflows/ci-standard.yml`, `scripts/check_doc_size_budget.py`, `tests/ci`, `tests/scripts`
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (final e83bd2e4 pins pass provider/seam, context, atlas and navigation controls; wheel passes launcher and calibration tests).
- **Summary:** Twelve components and five reviewed integrations reuse the atlas and capture goals. Main276998030 is integrated; a required regression rejects divergent pip/source/Rust providers.

### DL-#9914 · C3D Reference Fitting

- **State:** shipped
- **Owner:** codex
- **Issue:** #9914
- **PR:** #9918 (merged)
- **Paths:** src/motion_capture
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`6f2d63325` merge; protected CI, 20 club assets)
- **Summary:** Fits, club/volume/handedness display; see [evidence](reference_fitting_epic.md).

### DL-#9913 · Capture Journey Feedback and Detachable Views

- **State:** shipped
- **Owner:** codex
- **Issue:** #9913; epic #9906
- **PR:** #9917 (merged)
- **Paths:** capture_rig source/tests, guide and parity registry
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`8fce9f238` merge;445 local tests and protected CI pass)
- **Summary:** Identity/history, linked help/provenance and retained detachable Qt views.

### DL-#9912 · Impact Shaft Provider Integration

- **State:** shipped
- **Owner:** codex
- **Issue:** #9912; parent #9703
- **PR:** #9916 (merged)
- **Paths:** vendor/ud-tools, tests/shared_contracts, docs/development/impact-acoustics
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (PR #9916 and #9920 merged). Main pin e83bd2e4a7a29a2dcd8145ef2d1efa07123324f0 consistent across pins.
- **Summary:** Qualify Tools shaft/theme provider; see PROVIDER_PIN_RESULTS.json.

### DL-#9911 · Preview Discovery Failure Recovery

- **State:** shipped
- **Owner:** codex
- **Issue:** #9911
- **PR:** #9910 (merged)
- **Paths:** src/tools/capture_rig/preview.py, tests/tools/capture_rig/test_preview.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`18c8f922e` merge; recovery tests pass)
- **Summary:** Report discovery imports/timeouts through preview status.

### DL-#9907 · Guided Capture Outcomes

- **State:** shipped
- **Owner:** codex
- **Issue:** #9907; #9908; epic #9906
- **PR:** #9931 (merged)
- **Paths:** `src/tools/capture_rig`; capability graph/generator; matching tests and guide.
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (a74d5ebd2; 600 integrated regressions, normal push hooks, scoped mypy and Qt/browser review pass)
- **Summary:** Standard Qt outcome wizard shares typed map metadata and existing editors/readiness; capture-owned resume, optional My Clubs, background status and safe map-plan import.

### DL-#9905 · Player Bag and Capture Equipment

- **State:** shipped
- **Owner:** codex
- **Issue:** #9905; epic #9902
- **PR:** #9923 (merged)
- **Paths:** club_data/player_clubs.py; rig/capture_notes.py and equipment.py; Capture Rig bag/editor/library; model/session.py; matching tests, guide and generated maps.
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (`5ada5e6a68`)
- **Summary:** My Clubs supports catalog/custom entries, partial measurements, notes, archive and capture assignment. Captures preserve club snapshots and editable-copy lineage; fit provenance retains evidence.

### DL-#9904 · Offline Club Source Catalog

- **State:** shipped
- **Owner:** codex
- **Issue:** #9904; epic #9902
- **PR:** #9919 (merged)
- **Paths:** club_data/catalog_sources.py, public_clubs.json, scripts/review_club_catalog.py and tests
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`01831aa4c5`)
- **Summary:** Three sourced builds, review diffs and preserved player overrides.

### DL-#9903 · Attributed Club Catalog

- **State:** shipped
- **Owner:** codex
- **Issue:** #9903; epic #9902
- **PR:** #9919 (merged)
- **Paths:** club_data/, test_club_catalog.py and club_catalog.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`01831aa4c5`)
- **Summary:** Attributed optional properties, explicit inference gates and lossless exchange.

### DL-#9899 · Calibration Revision Status

- **State:** shipped
- **Owner:** codex
- **Issue:** #9899
- **PR:** #9959 (merged)
- **Paths:** reconstruct; rig command; capture_rig result evidence; tests.
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`fff95d5780`)
- **Summary:** Calibration and reconstruction fingerprints invalidate stale outputs; preserve results.

### DL-#9898 · Common Reference Calibration

- **State:** shipped
- **Owner:** codex
- **Issue:** #9898/#9900/#9909
- **PR:** #9946 (merged)
- **Paths:** reference_calibration, wizard, lens adapter
- **Started:** 2026-09-09
- **Last verified:** 2026-09-10 (732553479;28 pin/context checks pass)
- **Summary:** Reviewed paper/ruler calibration and guided recovery. [Evidence and limits](common_reference_calibration.md).

### DL-#9894 · Scoped Ubuntu CI Dependencies

- **State:** shipped
- **Owner:** codex
- **Issue:** #9894
- **PR:** #9896 (merged)
- **Paths:** .github/workflows/ci-standard.yml, scripts/ci/, tests/scripts/
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Signed per-job APT sources preserve shared-runner configuration; six Bash regressions and standard CI pass.

### DL-#9892 · Fleet Guide Compatibility

- **State:** shipped
- **Owner:** codex
- **Issue:** #9892
- **PR:** #9896 (merged)
- **Paths:** scripts/check_agent_docs_consistency.py, tests/architecture/test_check_agent_docs_consistency.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Central managed guidance and legitimate external/optional paths pass without hiding real missing-file failures.

### DL-#9883 · Instructor Reference Alignment Workspace

- **State:** shipped
- **Owner:** codex
- **Issue:** #9883
- **PR:** #9896 (merged)
- **Paths:** src/tools/capture*rig/reference*\*.py, styling.py, tests/tools/capture_rig/
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Responsive placement/timing/notes controls, event alignment, revision checks, preview/export parity and native layout evidence are delivered.
- **Evidence:** Qualified candidate equals merged tree; standard unit gate passed 14,821 tests.

### DL-#9882 · Comparison Rendering and Export Qualification

- **State:** shipped
- **Owner:** codex
- **Issue:** #9882
- **PR:** #9896 (merged)
- **Paths:** src/tools/capture_rig/reference_rendering.py, reference_export.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`5ef615e6fe22a046aec0bac4f80a1c241d03fef3`)
- **Summary:** Shared compositor, coverage-aware expert homography and staged exports delivered with qualified product #9896.

### DL-#9881 · Reference Timing and Camera Evidence

- **State:** shipped
- **Owner:** codex
- **Issue:** #9881 (advanced reference epic #9863)
- **PR:** #9885 (merged)
- **Paths:** src/motion_capture/reference, src/motion_capture/reconstruct/overlay3d.py, src/tools/capture_rig/reference_comparison.py, src/tools/capture_rig/reference_export.py, related tests and benchmark
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`8901f5804f`)
- **Summary:** Immutable bounded event anchors, binary-search gap-aware sampling, actual camera/clock snapshots and stale-registration checks replace unsupported calibration assumptions.
- **Evidence:** 12 adverse regressions failed before repair; 46 combined tests and 14-module mypy pass. Sampling medians: 0.157/0.093/0.304 ms for120/1200/12000 frames. CI typing/budget corrections pass16 tests.

### DL-#9879 · Comparison State and Export Lifetime

- **State:** shipped
- **Owner:** codex
- **Issue:** #9879 (advanced reference epic #9863)
- **PR:** #9884 (merged)
- **Paths:** src/motion_capture/reference/comparison.py, src/tools/capture_rig/reference_comparison.py, src/tools/capture_rig/swing_export_actions.py, tests/tools/capture_rig/test_reference_comparison_state.py
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`2c99bc83ea`)
- **Summary:** Preserve exact unrelated layer/registration fields, reject bad saved records, and reuse the existing export controller for safe thread ownership and deferred close.
- **Evidence:** Nine adverse regressions preceded repair;31 comparison/cancellation/swing/coaching tests and three-module mypy pass.

### DL-#9865 · Reference Scene Registration & Synchronization

- **State:** shipped
- **Owner:** codex
- **Issue:** #9865 (advanced reference epic #9863)
- **PR:** #9871 (merged)
- **Paths:** src/motion_capture/reference/registration.py, src/motion_capture/reconstruct/overlay3d.py, related tests/docs
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`10caddd219`)
- **Summary:** Calibrated scene registration, event-anchor and offset time synchronization, missing-joint gap mask preservation across bounded interpolation, distortion-aware camera projection, and 2D expert video homography without 3D claims.
- **Evidence:** 7 focused registration tests pass in tests/motion_capture/test_reference_registration.py. Strict round-trip serialization/deserialization validated. Projection tested with both pinhole and Brown-Conrady distortion. Ruff checks pass cleanly.

### DL-#9864 · Expert Reference Asset Imports

- **State:** shipped
- **Owner:** codex
- **Issue:** #9864 (advanced reference epic #9863)
- **PR:** #9870 (merged)
- **Paths:** src/motion_capture/reference, src/tools/capture_rig/reference_import.py, src/tools/capture_rig/reference_library_dialog.py, src/tools/capture_rig/library_dialog.py, src/shared/python/motion_pipeline/sources/c3d_adapter.py and related tests/docs/maps
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`90b147e10`; local qualification complete)
- **Summary:** Versioned portable reference assets retain explicit mapping, source hashes, timestamps and missing points; native library adds imports, notes/archive and background I/O. Expert videos remain linked 2D assets.
- **Evidence:** 30 integration tests pass, including real C3D and fresh-process isolation. Four native UI tests, eight-module mypy and architecture checks pass after layout/helper corrections.

### DL-#9862 · Saved Coaching References

- **State:** shipped
- **Owner:** codex
- **Issue:** #9862 (product #9849)
- **PR:** #9869 (merged)
- **Paths:** src/motion_capture/coaching, src/tools/capture_rig/coaching_canvas.py, coaching_dialog.py, coaching_export.py and related integration/tests/docs
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`77e6bca88`; protected PR merged)
- **Summary:** Saved source-coordinate shapes; draw/edit/style/frame visibility, undo/redo, library/editor access and cancellable PNG/video export.
- **Evidence:** 49 registry/atlas and19 drawing/export tests;12-module mypy;3012-file LoD clean. Visual minimums:496px references,465px editor.

### DL-#9860 · Capture Editing and Library

- **State:** shipped
- **Owner:** codex
- **Issue:** #9860, #9861 (product #9849)
- **PR:** #9868 (merged)
- **Paths:** src/motion_capture/rig/edits.py, ingest.py, src/tools/capture_rig/swing_editor.py, related tests and docs/development/capture_editing_integration.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`4c89892d7`; protected PR merged)
- **Summary:** Source-preserving trim/crop, capture notes/library, archive/storage/rename rollback, editable copies, cancellable export and timeline guards.
- **Evidence:** 300 integrated, 12 library/UI and 5 editor tests; eight-module mypy. Visual QA: 850x650, minimum492px.

### DL-#9851 · Capture Responsiveness and Recovery

- **State:** shipped
- **Owner:** codex
- **Issue:** #9851, #9857 (epic #9849)
- **PR:** #9859 (merged)
- **Paths:** src/tools/capture_rig/player.py, process_runner.py, benchmark_capture_responsiveness.py and cache/process tests; docs/development/capture_product_review.md
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`eaf8503ce`)
- **Summary:** Capture Responsiveness and Recovery
- **Acceptance:** Repeated frames decode once with isolated pixels; failed starts restore lifecycle/retry; benchmark limits documented.
- **Evidence:** 241 camera tests after cache; six focused tests after recovery; duplicate median 196.788 to 12.613 ms.

### DL-#9850 · Generated Capability Atlas

- **State:** shipped
- **Owner:** codex
- **Issue:** #9850 (children #9852, #9853; product #9849)
- **PR:** #9856 (merged)
- **Paths:** `scripts/capability_atlas/`, `scripts/generate_capability_atlas.py`,
- **Started:** 2026-09-09
- **Last verified:** 2026-09-09 (`9d6e6a872`)
- **Summary:** Generated C4-style context, workflow/artifact maps, searchable capabilities and Mermaid from existing registries.

### DL-#9830 · Independent Shooting Accuracy

- **State:** shipped
- **Owner:** codex
- **Issue:** #9830
- **PR:** #9841 (merged)
- **Paths:** src/shared/python/optimization; docs/development/shooting_convergence_9830_turnover.md
- **Started:** 2026-09-08
- **Last verified:** 2026-09-09 (merged 28d9bf79e)
- **Summary:** Adaptive reference defects; 21 native Bioptim/Casadi 3.6.7 passes. Casadi 3.8 failure and physical limits remain in the linked turnover.

### DL-#9825 · Preserve Reviewed Manufactured Claims in Actual Registration

- **State:** shipped
- **Owner:** codex
- **Issue:** #9825
- **PR:** #9826 (merged)
- **Paths:** docs/development/claim_preservation_9825_turnover.md; manufactured registration and evidence
- **Started:** 2026-09-08
- **Last verified:** 2026-09-09 (merged a410ae705)
- **Summary:** Preserves 328 reviewed outcomes; 128 strict contracts and 11 publication controls pass. Linked turnover retains full provenance and physical limits.

### DL-#9787 · Manufactured Authority Runtime and Provenance

- **State:** shipped
- **Owner:** codex
- **Issue:** #9787
- **PR:** #9804 (merged)
- **Paths:** authority runtime pins, native provenance/CI contracts and manufactured_authority_9787_turnover.md.
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (PR merged as 736ec2189 from b8da0c024)
- **Summary:** Compatible native pins and runtime support merged. The merged revision differs from locally validated 6235789dc; its actual registration bypass and stale evidence require follow-up #9825. Historical test results do not certify differing merged bytes.

### DL-#9783 · Reviewed Renderer Provider Compatibility

- **State:** shipped
- **Owner:** codex
- **Issue:** #9783
- **PR:** #9784 (merged)
- **Paths:** `tests/shared_contracts/test_tools_provider_contracts.py`, `docs/development/renderer_reference_9783_turnover.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`9aa26e4f83`)
- **Summary:** Reproduced the candidate's exact old-hash failure before accepting the two reviewed source/hash pairs. Tolerances, immutable provider origin and the current vendor pin remain strict.

### DL-#9762 · `bioptim` Optimal-Control Backend and the Swing-Dynamics Fixes

- **State:** shipped
- **Owner:** claude
- **Issue:** #9762 (epic); prerequisites #9755, #9756, #9757, #9758, #9759, #9760, #9761
- **PR:** #9768 (merged)
- **Paths:** `src/shared/python/optimization/ocp/`,
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`af6923bc70`)
- **Summary:** Adopts `pyomeca/bioptim` as an opt-in optimal-control layer

### DL-#9733 · Fail Fast on the Uninitialized Vendored Tools Fallback

- **State:** shipped
- **Owner:** claude
- **Issue:** #9733
- **PR:** #9743 (merged)
- **Paths:** `src/__init__.py`, `tests/unit/repo_hygiene/test_src_fallback_fail_fast_9733.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`930543b24d`)
- **Summary:** Uninitialized `vendor/ud-tools` raises actionable ImportError naming remediation command before finder installation.

### DL-#9648 · RTMPose ONNX Pose Estimator Behind the Registry

- **State:** shipped
- **Owner:** claude
- **Issue:** #9648
- **PR:** #9739 (merged)
- **Paths:** `src/shared/python/pose_estimation/rtmpose_onnx_estimator.py`, `src/shared/python/pose_estimation/rtmpose_models.py`, `src/shared/python/pose_estimation/registry.py`, `src/motion_capture/rig/ingest.py`, `src/motion_capture/reconstruct/layouts.py`, `pyproject.toml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`197e0a942`)
- **Summary:** Registers `rtmpose_onnx` (SimCC decode via onnxruntime, COCO-17/Halpe-26, whole-frame letterbox) with `capture_source=False`; adds the optional `pose-onnx` extra; pins the official OpenMMLab ONNX model URLs/sizes with digests PENDING OWNER APPROVAL; teaches `RegisteredFrameEstimator` to honour instance-level `LANDMARK_MAP`/`LAYOUT_NAME`; extends `layouts.py` with the Halpe-26 `hip`→`mid_hip` alias.

### DL-#9631 · Vendor Pin Carries the Tools#5048 Alias-Predicate Fix

- **State:** shipped
- **Owner:** claude
- **Issue:** #9631
- **PR:** #9722 (merged)
- **Paths:** `vendor/ud-tools`, `tests/unit/repo_hygiene/test_pinned_import_alias_contract.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`648d4c9213`)
- **Summary:** Pinned `vendor/ud-tools` carries Tools#5049 flattened-install fix; added TDD contract test asserting pinned predicate in both layouts.

### DL-#9612 · Video Upload Suffix Derived From Filename Allow-List

- **State:** shipped
- **Owner:** claude
- **Issue:** `#9612`
- **PR:** #9720 (merged)
- **Paths:** `src/api/routes/video.py`, `tests/unit/api/test_routes_video.py`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (`4f705f508c`)
- **Summary:** Video analysis uploads no longer default temp files to `.mp4`;

### DL-#9607 · Authority Runtime Native Library Bootstrap for Cmeel Pinocchio Wheels

- **State:** shipped
- **Owner:** claude
- **Issue:** #9607
- **PR:** #9726 (merged)
- **Paths:** `scripts/research/proximal_distal_energy/articulated_native_runtime.py`, `scripts/research/proximal_distal_energy/run_articulated_manufactured_solution.py`, `tests/research/test_articulated_native_runtime.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`474ea6bf56`)
- **Summary:** Resolves the authority lane's `liburdfdom_sensor.so.4.0` import failure by resolving `cmeel.prefix/lib` from the live venv, verifying the locked sonames with an explicit DbC diagnostic, and re-execing the authority profile with `LD_LIBRARY_PATH` prepended before `import pinocchio`.

### DL-#9542 · Bunker Exit State Consistency, Provenance, and Result Envelope

- **State:** shipped
- **Owner:** claude
- **Issue:** #9542
- **PR:** #9728 (merged)
- **Paths:** `src/bunkershot3d/ball/**`
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (`ba6e6e2a30`)
- **Summary:** `SandDelivery` now refuses contradictory exit speed/vector pairs and owns copies of list-supplied exit vectors so post-construction mutation cannot invalidate the frozen record; the `to_post_impact_state` boundary carries explicit `ExitVectorProvenance` labels, and `PostImpactEnvelope` wraps the flight handoff with the validity verdict, F0 tier, per-group frames, the proper `HEAD_FRAME_TO_FLIGHT_TRANSFORM`, a schema version, and a SHA-256 source digest with JSON round trip. Reflection rejection itself was already delivered by PR #9574 and is not redone.

### DL-#9533 · Test-Only Extras Reachable From the Dev Lock

- **State:** shipped
- **Owner:** claude
- **Issue:** `#9533`
- **PR:** #9716 (merged)
- **Paths:** `pyproject.toml`, `.github/workflows/lock-refresh.yml`, `requirements*.lock`, `environment.yml`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`bae85ae4c4`)
- **Summary:** `openpyxl` and `imageio` resolve through `dev` extra; lock regeneration delegated to `lock-refresh.yml`.

### DL-#9499 · Spec Check Reminder Fail-Safe Extraction

- **State:** shipped
- **Owner:** claude
- **Issue:** `#9499`
- **PR:** #9719 (merged)
- **Paths:** `.github/workflows/spec-check.yml`, `scripts/post_spec_reminder.py`, `tests/ci/test_spec_check_workflow.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`e214fae7bd`)
- **Summary:** The `Verify SPEC.md freshness` job posts its SPEC reminder

### DL-#9494 · Resolve the CLAUDE.md `--no-verify` Contradiction by Fixing the Windows Hook Environment

- **State:** shipped
- **Owner:** claude
- **Issue:** #9494
- **PR:** #9744 (merged)
- **Paths:** `CLAUDE.md`, `AGENT_HANDOFF.md`, `docs/development/DEVELOPMENT_LOG.md`, `SPEC.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`dbc6727aa`)
- **Summary:** CLAUDE.md forbade `git commit --no-verify` while agents on

### DL-#9484 · Impact Explorer Web Route Has a CI Bundle Producer

- **State:** shipped
- **Owner:** W4_9484 (agent claude)
- **Issue:** #9484
- **PR:** #9724 (merged)
- **Paths:** `.github/workflows/ci-standard.yml`, `scripts/check_declared_route_producers.py`, `tests/scripts/test_declared_route_producers.py`, `docs/workflows/WORKFLOW_TRACKING.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`ba841ae524`)
- **Summary:** The `rate_of_closure` tile declared `web.mode: route` for `/tools/impact-explorer` but no pipeline built `vendor/ud-tools/src/rate_of_closure/web/dist`, so a clean checkout served the honest fallback. CI Standard now builds the bundle from the pinned Tools tree with `npm run build -- --base=/impact-explorer-app/`, and `scripts/check_declared_route_producers.py` fails any declared route that no pipeline produces. Shipping the bundle inside the wheel/image remains an open maintainer decision (#9417).

### DL-#9482 · Launcher Tile Logo Families and Registry Gate

- **State:** shipped
- **Owner:** claude
- **Issue:** #9482
- **PR:** #9725 (merged)
- **Paths:** `src/config/launcher_manifest.json`, `assets/logos/**`,
- **Started:** 2026-09-07
- **Last verified:** 2026-09-07 (`191351bdf`)
- **Summary:** Broke the launcher grid's worst logo reuse (data_explorer x9,

### DL-#9478 · Launcher Registry Truth: `tools://` Provenance Scheme and Ready/Beta Maturity Gate

- **State:** shipped
- **Owner:** claude
- **Issue:** #9478
- **PR:** #9729 (merged)
- **Paths:** `src/config/models.yaml`, `src/config/launcher_manifest.json`,
- **Started:** 2026-09-07
- **Last verified:** 2026-09-08 (`8c2c6fa4a1`)
- **Summary:** `provider: tools` entries in `src/config/models.yaml` and

### DL-#9476 · Re-Vendor the Corrected Spec Merge Driver and Pin Drift

- **State:** shipped
- **Owner:** claude
- **Issue:** `#9476`
- **PR:** #9736 (merged)
- **Paths:** `scripts/install_spec_merge_driver.py`,
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`875cd501e`)
- **Summary:** The vendored installer still stamped the withdrawn

### DL-#9470 · Launch-Monitor Analysis Handlers Onto the Async_Action Worker

- **State:** shipped
- **Owner:** claude
- **Issue:** #9470
- **PR:** #9742 (merged)
- **Paths:** `src/tools/launch_monitor_analytics/gui.py`, `src/tools/launch_monitor_analytics/_embed_adapter.py`, `tests/ui/tools/launch_monitor/test_async_actions.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`af0aed1d57`)
- **Summary:** All seven analysis handlers (`treatment`, `relationship`, `multivariate`, `model`, `comparison`, `dispersion`, `trend`) now run their compute on the #8880 `async_action` worker via one shared `AsyncActionBar`; synchronous `present(compute())` paths kept; embed adapter `cleanup()` cancels and joins the worker. First slice of the #9470 tool checklist; the remaining tools are follow-ups.

### DL-#9409 · Always-On Quality Gate Lane and Conftest Src-Pivot Guard

- **State:** shipped
- **Owner:** `claude`
- **Issue:** [#9409](https://github.com/D-sorganization/UpstreamDrift/issues/9409)
- **PR:** #9723 (merged)
- **Paths:** `.github/workflows/ci-standard.yml`, `tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py`, `docs/workflows/WORKFLOW_TRACKING.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`6428062284`)
- **Summary:** CI Standard gains an always-on, ≤10-minute `always-on-unit-lane` (verify_installation import smoke over the shared Tools alias roots, top-level smoke tests, contract tests) that `quality-gate` requires `success` on every PR including docs-only ones; a repo-hygiene guard forbids any conftest from pivoting `sys.modules["src"]` directly (must use `EngineSrcPivot`). Deferred on #9409: main-branch cancel exemption (RM campaign) and nightly cross-engine dedupe (#8725/#9002).

### DL-#9387 · Unit-Gate Worker Corruption: `src`-Identity Sentinel and Leak Fixes

- **State:** shipped
- **Owner:** claude
- **Issue:** #9387
- **PR:** #9741 (merged)
- **Paths:** `tests/unit/repo_hygiene/test_src_identity_sentinel.py`,
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`c70d5ddae`)
- **Summary:** Static audit of `tests/` (conftests excluded) found 83

### DL-#9249 · UI: Pin @vitejs/Plugin-React to ^5 Until Vite 8

- **State:** shipped
- **Owner:** claude
- **Issue:** #9249
- **PR:** #9718 (merged)
- **Paths:** `.github/dependabot.yml`, `ui/README.md`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`7cdbb0a3d`)
- **Summary:** Dependabot ignores `@vitejs/plugin-react` major updates

### DL-#9091 · Phantom-Guard Rule-3 False Positive on Shallow Base Fetch

- **State:** shipped
- **Owner:** claude
- **Issue:** #9091
- **PR:** #9717 (merged)
- **Paths:** `.github/workflows/anti-phantom-merge.yml`, `scripts/ci/check_phantom_guard_paths.py`, `tests/scripts/test_check_phantom_guard_paths.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`e55c4f3e19`)
- **Summary:** The phantom-guard rule-3 closes-issue path-membership check

### DL-#8943 · Cache API CPU Work Off the Event Loop

- **State:** shipped
- **Owner:** claude
- **Issue:** #8943
- **PR:** #9727 (merged)
- **Paths:** `src/api/routes/analysis_plots.py`, `src/api/routes/model_explorer.py`, `src/api/routes/models.py`, `src/api/routes/launch_monitor_analytics.py`, `src/api/routes/_route_utils.py`
- **Started:** 2026-09-08
- **Last verified:** 2026-09-08 (`17beca5537`)
- **Summary:** `GET /analysis/plot-data/{plot_type}` builds orchestrator once per recorder identity and serves results from an LRU off the event loop. Model explorer and models URDF handlers use LRU cache in worker threads.

### DL-#8901 · Accessible Model Card Actions and Grid Navigation

- **State:** shipped
- **Owner:** claude
- **Issue:** #8901
- **PR:** #10006 (merged)
- **Paths:** src/launchers/model_card.py, tests/launchers/test_model_card_accessibility.py
- **Started:** 2026-09-12
- **Last verified:** 2026-09-12 (`eb4a042cbb`)
- **Summary:** Harden model card touch and keyboard accessibility, ensure WCAG target sizes, and add arrow-key grid navigation in launcher.

### DL-#8360 · Bounded Launcher Splash and Optional-Provider Degradation

- **State:** shipped
- **Owner:** claude
- **Issue:** #8360 (related #8339, #8358, #8359)
- **PR:** #9951 (merged)
- **Paths:** src/launchers/startup.py, src/launchers/startup_phases.py, src/launchers/startup_session.py, src/launchers/startup_failure_dialog.py, src/launchers/upstream_drift_launcher_main.py, src/launchers/launcher_orchestrator.py
- **Started:** 2026-09-10
- **Last verified:** 2026-09-11 (`35c69c2894`)
- **Summary:** Bounded, timestamped startup phases with a StartupSession watchdog; optional Tools/Rate provider degrades the shell instead of stalling the splash; Retry / Continue / Copy diagnostics / Close dialog replaces quit-on-error.
- **Evidence:** Deterministic tests inject successful, missing, exception-raising and never-completing providers and prove bounded splash lifetime, degraded shell startup, stale-generation isolation and deleted-widget guards.

### DL-#1616 · Mermaid C4 Architecture Maps

- **State:** shipped
- **Owner:** local
- **Issue:** #1616
- **PR:** #9963 (merged)
- **Paths:** docs/architecture/C4.md, scripts/architecture_map_contract.py
- **Started:** 2026-09-10
- **Last verified:** 2026-09-10 (`09f6d22da3`)
- **Summary:** Baseline adoption of Mermaid C4 architecture maps in UpstreamDrift.

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
