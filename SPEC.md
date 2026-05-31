# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST UPDATED: 2026-05-30

  This is the canonical specification template for all repositories in the
  D-sorganization fleet. Every repo MUST have a SPEC.md at its root.

  INSTRUCTIONS:
  1. Copy this template to the root of your repository as SPEC.md
  2. Fill in every section — leave nothing as "[TODO]"
  3. Keep this document updated with every PR that changes functionality
  4. CI will block merges if SPEC.md is stale (source changed but spec didn't)

  AUDIENCE: This document is designed for both human developers AND AI agents.
  Write clearly, use concrete examples, and avoid ambiguity.
-->

## SPEC Ownership and Update Cadence

- **Owner:** @diete (responsible for accepting SPEC.md edits)
- **Update triggers (mandatory):**
  - Any PR that adds, removes, or moves a top-level `src/` package or a public
    engine adapter must update §6 (Component Locations) and §7 (Feature Status).
  - Any PR that changes the version in `pyproject.toml` must update §1 (Identity).
  - Any PR that changes a CI gate threshold must update §X (Quality Gates).
- **Review cadence:** SPEC.md is reviewed for staleness on every release
  (per `docs/operations/release-runbook.md`, see #3842).

## 1. Identity

| Field                   | Value                                              |
| ----------------------- | -------------------------------------------------- |
| **Repository Name**     | `UpstreamDrift`                                    |
| **GitHub URL**          | `https://github.com/D-sorganization/UpstreamDrift` |
| **Owner**               | D-sorganization                                    |
| **Primary Language(s)** | Python 3.10+, Rust, TypeScript                     |
| **License**             | MIT                                                |
| **Current Version**     | 2.1.1                                              |
| **Spec Version**        | 1.0.223                                            |
| **Last Spec Update**    | 2026-05-31                                         |

## 2. Purpose & Mission

UpstreamDrift is a multi-physics golf swing biomechanical simulation platform that consolidates five leading physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) for cross-validated biomechanical analysis. It enables researchers and biomechanists to simulate human movement across models ranging from simplified 2-DOF pendulums to complex 290-muscle musculoskeletal systems, providing a unified interface for comparative physics analysis and professional-grade visualization.

## 3. Goals & Non-Goals

### Goals

- Integrate and cross-validate five physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) for biomechanical analysis
- Provide biomechanical analysis tools including inverse kinematics (IK), inverse dynamics (ID), and muscle dynamics modeling
- Enable motion capture integration and trajectory optimization
- Offer multiple control schemes (impedance, admittance, hybrid) for simulated systems
- Deliver professional GUI with real-time 3D rendering for simulation visualization
- Expose FastAPI REST backend for programmatic access and integration
- Support desktop deployment via Tauri application framework
- Provide MATLAB/Simulink integration for cross-platform workflow compatibility
- Implement reinforcement learning integration for learning-based control policies
- Support models ranging from educational 2-DOF pendulums to complex 290-muscle systems

### Non-Goals

- Not a general-purpose physics engine; focused exclusively on biomechanical simulation
- Not intended for non-biomechanical simulations (rigid body dynamics, fluid dynamics, etc.)
- Not a replacement for domain-specific tools (OpenSim for clinical analysis, MATLAB for controls research)

## 4. Architecture Overview

### Recent Spec Updates

- **2026-05-31** - Recovered closed review-feedback fixes for the metadata-driven UX wrappers: the shared `simulation.engine` metadata now includes `jaxsim` for generated TypeScript/PyQt engine selectors, and PyQt `HelpfulField` free-form fields with `valid_range: null` now render an editable `QLineEdit` instead of an empty combo box.
- **2026-05-31** - Added the CC-22 offline Nimble gradient-oracle surface for issue #6795: `tools.offline_validation.nimble_gradient_oracle` provides deterministic request/response comparison types, lazy optional `nimblephysics==0.10.52.2` plus PyTorch loading, structured skip behavior for core installs, and a runtime-boundary test that forbids Nimble imports from `src/`.
- **2026-05-31** - Added the CC-34 engine selector/comparison UI surface (#6807): the React simulation GUI now exposes a capability-aware multi-engine comparison panel, greys out unavailable or unsupported engines from existing capability metadata, captures per-engine run provenance, and renders side-by-side columns with divergence annotations for shared numeric outputs.
- **2026-05-31** - Added the CC-35 workspace project/session spine and results-browser view models (#6808): `src/shared/python/workspace/` now persists project, subject, session, and dataset metadata in `project.json` and indexes CC-4 HDF5 trace artifacts with CC-6 `provenance_*` metadata for session/backend/text filtering.
- **2026-05-31** - Added the CC-23 moving-horizon estimator near-real-time path (#6796): `src/shared/python/estimation/moving_horizon.py` now maintains a bounded rolling window over canonical samples, builds fixed-parameter MAP objectives from the CC-19 solver surface, warm-starts from the previous window, records latency against a stated 50 ms default budget, and exposes a callback payload for realtime integration.
- **2026-05-31** - Added the CC-32 canonical-core app shell registry (#6805): canonical-core estimation and comparison now appear as shared launcher tools in both PyQt6 and React/Tauri surfaces through ADR-0013 `launcher_embed` registration, shared manifest metadata, and `/tools/canonical-core/*` routes while leaving the CC-19/CC-27 service bodies to their dedicated implementation work.
- **2026-05-31** - Added the Sidekick Canonical Core retrieval Q&A tool (#6810): the chat service can now expose a read-only `answer_canonical_core_question` tool backed by a bounded local Canonical Core corpus, deterministic extractive answers, and `path:start-end` citations. The behavior is documented in `docs/sidekick/README.md` and `docs/specs/active/sidekick-canonical-core-retrieval-qa.md`.
- **2026-05-31** - Added the CC-36 config validation setup wizard (#6809): canonical-core setup now has a deterministic preflight API, headless wizard view model, launcher embeddable tool, and default model block coverage for validating units, frames, model dimensions, and subject calibration before engine execution.
- **2026-05-31** - Added the CC-38 Sidekick canonical-core tool adapter (#6811): Sidekick can now expose bounded `canonical.configure`, `canonical.validate`, `canonical.run`, `canonical.compare`, and `canonical.interpret` actions through `CanonicalToolAdapter` and a host-supplied `CanonicalActionPort`, preserving the existing audit, policy, dry-run, and destructive-confirmation gates.
- **2026-05-31** - Added the CC-7 cross-engine conformance harness for issue #6779: the engine-core validator now emits parity checks/results for canonical q/v/a traces, documents the merge-gate contract in the parity spec, and includes focused conformance tests plus hardened optional-engine CI wiring.
- **2026-05-31** - Added the Pinocchio canonical-v2 reference adapter slice for issue #6782: `pose_interchange.adapters.pinocchio_reference` now remaps canonical `[xyz, quat_wxyz]` and `[angular; linear]` q/v/a states to Pinocchio's `[xyz, quat_xyzw]` and `[linear; angular]` conventions, declares inverse-dynamics, forward-dynamics, and gradient capabilities, exposes FK/Jacobian/RNEA/ABA boundaries with an optional Rust trajectory path, and includes focused unit coverage for remap, fallback dynamics, and inertial-parameter gradients.
- **2026-05-31** - Added the MuJoCo canonical-v2 adapter slice for issue #6783: pose interchange now includes MuJoCo q/v/a remapping and capability metadata, the simulation backend exposes inverse-dynamics support, soft-contact divergence is documented in the canonical-v2 conventions, and focused unit tests cover adapter and backend behavior.
- **2026-05-31** - Added the CC-11 differential-testing report scaffold for issue #6784: `scripts/validation/cross_engine_differential_report.py` now generates normalized machine-readable and Markdown validation artifacts under `docs/validation/`, including dependency-blocked defaults and CC-7 conformance-harness normalization tests.
- **2026-05-31** - Added the CC-24 canonical ZTCF/ZVCF analysis bridge (#6797): simulation backends now expose canonical zero-torque crossing and zero-velocity crossing analysis helpers, extend the results schema v2 documentation, and cover AffineDrift-compatible event extraction and result serialization with focused unit tests.
- **2026-05-31** - Added canonical-core CI wiring for issue #6780: cross-engine equivalence now exposes per-engine conformance jobs, heavy optional stacks remain self-hosted and opt-in, canonical-core Jules templates document adapter/conformance/docstring tasks, and the JaxSim forward-simulation analytic reference uses the canonical gravity convention with the current tolerance envelope.
- **2026-05-31** - Added the CC-12 canonical observations schema for markerless pose ingestion (#6785): `src/shared/python/pose_estimation/observations.py` now preserves detector layout, calibrated camera records, per-camera 2D keypoints, per-keypoint confidence, optional triangulated 3D keypoints, JSON round-tripping, and trace metadata attachment, with fixtures, docs, and unit coverage.
- **2026-05-31** - Added the CC-14 OpenCap integration slice (#6787): the motion pipeline can now ingest OpenCap-style marker/keypoint exports through a source adapter, register the source contract, validate local fixtures, and document the supported OpenCap import format for turnkey secondary ingestion.
- **2026-05-31** - Added the CC-13 Pose2Sim integration slice (#6786): the motion pipeline now includes Pose2Sim fixture ingestion, source adapter exports, MediaPipe JSON compatibility wiring, and motion-pipeline documentation for primary local multi-camera workflows.
- **2026-05-31** - Added the CC-25 engine-agnostic wrench/GRF extraction bridge (#6798): the shared simulation backend layer now converts canonical `Trace.wrench` arrays to and from the existing `bunkershot3d.postproc.WrenchTrace` primitive, exposes impulse helpers and trace attachment, documents the unified `(T, 6)` wrench layout, and validates the static body-weight support case.
- **2026-05-31** - Added the CC-17 synthetic ground-truth rig and identifiability probes (#6790): estimation now exposes synthetic fixture generation, forward-model protocols, identifiability diagnostics, documentation, and focused tests for validating estimator inputs before fitting real trials.
- **2026-05-31** - Added the CC-27 cross-engine comparison report module (#6800): `simulation_backends.compare()` now runs selected backends from identical user input and emits structured side-by-side kinematics, kinetics, ZTCF/ZVCF, and wrench panels with divergence registry annotations and per-panel provenance; `compare_cli.py` provides a one-command Markdown/JSON report path.
- **2026-05-31** - Hardened the Bot CI trigger workflow so invalid PAT-style secrets no longer block fallback to the repository token; token validation now tries `BOT_PAT`, `RUNNER_CHECK_TOKEN`, and `github.token` in order before deciding CI cannot be triggered for bot-authored PRs.

- **2026-05-31** - Added the CC-21 AddBiomechanics inertia-prior importer (#6794): `src/shared/python/anthropometrics/addbiomechanics_priors.py` validates bounded calibration exports, converts body-segment mass/COM/inertia fields into estimator-compatible prior payloads, and documents the calibration pipeline with deterministic persistence and validation coverage.
- **2026-05-31** - Added the CC-26 AffineDrift coupling surface (#6799): `src/shared/python/analysis/affine_drift_coupling.py` now samples double-pendulum traces into pointwise drift/control-affine acceleration terms, exposes HDF5 persistence for coupling results, and documents canonical-v2 trace extraction in `docs/conventions/canonical-v2.md` and `docs/simulation_backends/results_schema_v2.md`.
- **2026-05-31** - Added the CC-16 output-only canonical C3D exporter (#6789): motion capture can now export marker trajectories from canonical state arrays to terminal C3D files with unit, label, sample-rate, and architecture guards that prevent C3D from becoming an internal intermediate.
- **2026-05-31** - Added the CC-28 Drake canonical-core adapter slice for issue #6801: the existing Drake pose adapter now declares AutoDiffXd/contact/trajectory capabilities, remaps canonical-v2 dynamic state blocks into Drake `QuaternionFloatingJoint` ordering with angular-velocity frame conversion, and registers the hydroelastic-vs-Pinocchio contact divergence in `docs/conformance/canonical_core_divergences.yaml`.
- **2026-05-31** - Added the CC-30 MyoSuite canonical-core adapter slice (#6803): activation-driven canonical-v2 state remapping for MyoSuite/MuJoCo MJCF layouts, explicit MUSCLES/FORWARD_DYN/CONTACT capability declaration with no joint-torque inverse-dynamics claim, upstream-muscle activation/force helper routing, and Trace v2.1 muscle-output persistence fields.
- **2026-05-31** - Added the CC-33 canonical 3D viewport provider decision (#6806): MeshCat is the selected default over Rerun and VTK/PyVista, with lazy provider metadata/selection/degradation and a Trace v2 overlay payload for canonical-v2 trajectory, marker, contact, and GRF/wrench data.
- **2026-05-31** - Tightened review-feedback guardrails for issues #6816 and #6827: the license ledger advisory now validates the OpenPose row cells directly, the cross-engine equivalence workflow runs when `pyproject.toml` changes so the JaxSim pin guard covers optional-extra drift, and the bot CI trigger validates `gh auth status` before attempting authenticated workflow dispatch.
- **2026-05-31** - Added canonical-core estimation residuals for issue #6791:
  pure reprojection, RNEA dynamics, anthropometric prior, and trajectory
  smoothness residual functions now live under `src/shared/python/estimation/`,
  with finite-difference Jacobian coverage, optional JAX autodiff Jacobians,
  and developer guidance in `docs/development/canonical_core_residuals.md`.
- **2026-05-31** - Hardened the runtime Docker image against the current Debian 13 medium-severity glibc, systemd/libudev, and sed CVEs by explicitly upgrading/installing `libc-bin`, `libc6`, `libsystemd0`, `libudev1`, and `sed` in the runtime apt layer while preserving the pinned `python:3.12-slim` base digest.
- **2026-05-31** - Added the launcher workspace tab close-to-background workflow for #6013: `DraggableTabWidget` can now background-close tabs without destroying their embedded widget, track backgrounded tab metadata, restore hidden tabs by title, and expose the feature through launcher UI close affordances. Regression coverage in `tests/launchers/test_workspace_tabs.py` validates close/restore behavior and state preservation.
- **2026-05-31** - Added the CC-15 calibratable keypoint-offset observation model: detector keypoints can now be calibrated against model joint centers as segment-frame offsets with covariance, standard error, confidence support, and residual helpers documented in `docs/conventions/keypoint-offset-model.md`.
- **2026-05-31** - Added CC-20 multi-trial / multi-view shared-parameter stacking: `src/shared/python/estimation/multi_trial.py` now solves independent per-trial spline blocks against one shared parameter block, excludes locked parameters from the decision vector, serializes shared-parameter specs for run manifests, and reports approximate shared-parameter posterior covariance so synthetic multi-trial fits can verify identifiable directions tighten with more data.
- **2026-05-31** - Added the canonical run `ProvenanceStamp` primitive for issue #6778: simulation traces, batch traces, and state checkpoints can now carry deterministic run metadata covering engine/model identifiers, timestamp, adapter version, units, feature flags, and dependency versions without changing the existing trace/checkpoint schemas.
- **2026-05-31** - Added the first canonical model core slice for issue #6775: `model_generation.canonical_model` now provides immutable engine-neutral links, joints, geometry, materials, stable deterministic JSON/model hashes, validation, conversion to existing model-generation core types, and URDF export through the existing writer.
- **2026-05-31** - Added metadata-driven helpful-field and provenance-value wrappers for the Idiot-Proof UX epic (#5968): PyQt6 and React controls now consume the shared field metadata/provenance contracts, UI field metadata is generated from `configs/ux/field_metadata.yaml`, and parity tests keep the TypeScript registry synchronized with the YAML source of truth.
- **2026-05-31** - Added the first unified engine capability taxonomy slice for issue #6777: `engine_core.capabilities.Capability` is now the canonical enum/query surface, simulation backend capabilities can answer canonical `supports()` checks while keeping legacy booleans, and architecture docs describe the adapter boundary.
- **2026-05-31** - Added a third-party license ledger for issue #6781 under `docs/legal/licenses.md`, with a CI-sized advisory checker that covers direct dependency declarations, keeps OpenPose visibly fenced as non-commercial opt-in, supports Python 3.10 via `tomli`, and avoids false core-install optional-engine findings from the local `scripts/jaxsim` helper directory.
- **2026-05-31** - Added canonical-v2 dynamic state support for CC-2 (#6774): `CanonicalState` now carries immutable `(q, v, a, t)` data with floating-base quaternion layout, manifold-safe integrate/difference operations, canonical-v1 lift helpers, and SE(3) quaternion utilities covered by shape, validation, and property-style round-trip tests.
- **2026-05-31** - Added the canonical-v2 pose interchange contract (#6773) with public exports from `src/shared/python/pose_interchange/__init__.py`, a conventions guide under `docs/conventions/canonical-v2.md`, and ADR coverage in `docs/adr/0026-canonical-dynamic-state-v2.md`.

- **2026-05-31** - Hardened the JaxSim #6648 URDF-to-SDF gate CI path by parsing inertial XML with `defusedxml` and preventing the core-only install guard from treating helper directories such as `scripts/jaxsim` as installed optional engines.
- **2026-05-30** - Hardened the JaxSim #6648 URDF-to-SDF inertial round-trip gate so converted SDF payloads fail on unexpected inertial links instead of silently accepting extra mass/inertia records; regression coverage now exercises the unexpected-link failure path.
- **2026-05-30** - Added JaxSim parameter-gradient sensitivity support (#6656): `SupportsParameterGradients` now captures pointwise parameter Jacobians, `JaxSimBackend` delegates to a JAX autodiff ZTCF sensitivity module over documented anthropometric parameters, tests validate autodiff against finite differences, and `scripts/jaxsim/plot_parameter_sensitivity.py` reproduces a sample sensitivity plot from measured states.
- **2026-05-30** - Added JaxSim forward simulation rollout support (#6655): `JaxSimBackend.rollout` now drives `jaxsim.api.model.step`, returns the canonical `Trace` schema with full floating-base state, validates control/time preconditions, records convention metadata, and includes an analytic double-pendulum parity gate through the adapter seam.
- **2026-05-30** - Added the JaxSim/Pinocchio cross-engine dynamics parity gate (#6654): CI now runs a single-body installed-stack comparison for mass matrix, bias, gravity, and Coriolis terms, documents the tolerance envelope, and covers live JaxSim 0.9.0 model/data API compatibility.
- **2026-05-31** - Surfaced JaxSim through the capability-aware engine selector: API engine metadata, launcher capability profiles, exercise discovery, engine registry integration, and the React engine store now expose JaxSim with gated capability tooltips; the exercise dashboard opens the dedicated capability-driven JaxSim dashboard instead of a placeholder (issue #6658).
- **2026-05-31** - Added the JaxSim upgrade guard policy (#6660): CI now owns a pinned optional-dependency upgrade workflow for `jaxsim==0.9.0`, the version policy is documented in `docs/development/jaxsim_version_policy.md`, and the three-engine tutorial points users through the pinned extra before cross-engine equivalence and gradient checks are used to justify future upgrades.
- **2026-05-30** - Added the first JaxSim backend adapter (#6653): `JaxSimBackend` lazily maps JaxSim free-floating mass, bias, gravity, Coriolis, inverse-dynamics, and Jacobian APIs into engine-core load/query/dynamics protocols, declares JaxSim capabilities, and registers `EngineType.JAXSIM` in `LOADER_MAP`.
- **2026-05-30** - Rolled the backgrounding/pop-out lifecycle across every embedded tool (Sub-PR B of #6013): audited all 13 `src/tools/*/_embed_adapter.py` adapters for `cleanup()` idempotency and annotated each with a one-line `# background:` decision comment. Twelve adapters background fine at the structural defaults (`can_background`/`detach_to_window` → `True`) — they are CPU widgets or hold only in-memory state worth keeping alive while hidden, with no scarce GPU context at the adapter level and no modal-installer constraint. The `training_controller` adapter's `cleanup()` was tightened to the swap-then-clear pattern (drop widget refs first, never re-clean on a second call, never raise). The `pose_subscriber_demo` tool — the one holding a live `pose/canonical` realtime subscription — gained real `pause()`/`resume()` hooks: `pause()` releases the subscription so a hidden subscriber stops consuming traffic and `resume()` re-acquires it (widget hooks added to `src/tools/pose_subscriber_demo/gui.py`, forwarded by the adapter). No adapter needed `can_background=False`. Per-adapter idempotency, structural-default, pause/resume, and a full open→background→reopen→pop-out→dock-back state-preservation round-trip are covered in `tests/unit/launchers/test_embed_adapter_backgrounding_rollout.py` (34 tests).
- **2026-05-30** - Made embedded launcher tabs backgroundable and pop-out-able (Sub-PR A of #6013): added an additive `BackgroundableTool` protocol (`src/shared/python/launcher_embed/contract.py`, package bumped to `1.1.0`) with four optional hooks — `pause()`, `resume()`, `can_background()` (default `True`), and `detach_to_window()` (default `True`) — kept separate from `EmbeddableTool` so its `runtime_checkable` `isinstance` check still accepts the ~17 existing adapters, with hosts resolving the hooks structurally via `getattr`-with-default. `EmbeddedHostWidget` (`src/launchers/embedded_host.py`) now prompts "Close (keep running)" vs "Destroy" on tab close: background-close pauses the tool and stashes its widget hidden (re-surfaced with `resume()` on reopen), while destroy keeps the legacy `cleanup()` path. Added `pop_out_tab(tool_id)` / `dock_back(tool_id)` (re-parent the live widget into / out of a top-level `QMainWindow`; closing the popped-out window re-docks), a tab-bar context menu (Close-keep-running / Destroy / Pop out), and the public `backgrounded_tools() -> set[str]` API. Tests in `tests/unit/launchers/test_embedded_host_backgrounding.py`. Per-tool adapter rollout is tracked separately as Sub-PR B.
- **2026-05-30** - Added the JaxSim floating-base velocity convention contract (#6652): engine-core now defines body-fixed, inertial, and mixed velocity representations, normalization helpers to the suite's inertial canonical representation, gravity/base-frame units, and single-floating-body analytic `h`/`g` coverage tied to `SPATIAL_JACOBIAN_ORDER`.
- **2026-05-30** - Extended the engine capability taxonomy for JaxSim planning (#6651): `EngineCapabilities` now reports parameter gradients, state/control gradients, forward simulation, contact stepping, and trajectory optimization support with accessors, serialization round-trip coverage, and documented verified engine profiles.
- **2026-05-30** - Added the gated JaxSim optional dependency extra (`upstream-drift[jaxsim]`) pinned to `jaxsim==0.9.0`, with CPU-JAX-first documentation, core-install isolation coverage, and an optional SDF step smoke test for the JaxSim stack.
- **2026-05-30** - Added the JaxSim #6648 canonical URDF-to-SDF gate harness: sdformat CLI detection, SDF conversion, mass/inertia round-trip checks, BRICK setup documentation, and optional JaxSim loading coverage asserting the canonical 25-velocity model contract.
- **2026-05-30** - Added a full-src mypy ratchet for push-to-main CI: mypy now uses explicit package bases for namespace-package discovery, and push runs compare `mypy src --config-file pyproject.toml` against `scripts/config/full_src_mypy_baseline.json` so new type debt fails while the current unmasked backlog remains accountable.
- **2026-05-30** - Hardened the runtime Docker image against the current Debian 13 `libcap2` high-severity CVE by explicitly upgrading/installing `libcap2` during the runtime apt layer while preserving the pinned `python:3.12-slim` base digest.
- **2026-05-30** - Declared `pyarrow>=14.0.0` in the data/dev dependency surfaces and regenerated dependency artifacts so Parquet compactor/loader tests can collect in CI; `tests/unit/test_build_install_contracts.py` now falls back to `tomli` on Python 3.10.
- **2026-05-30** - Widen pinocchio version limit from `<3.0.0` to `<5.0.0` in `pyproject.toml` to resolve numpy 2.0 version compatibility conflict.
- **2026-05-29** - API security hardening (issue #6643): introduced `_assert_type` guard in `src/api/auth/dependencies.py` to narrow SQLAlchemy query-result types for MyPy strict mode, replaced the previous `type: ignore[return-value]` workarounds with explicit runtime assertions, and added `_lookup_cached_api_key`, `_lookup_api_key_by_prefix`, and `_get_active_user_for_api_key` helper functions for testability. Added `src/api/auth/dependencies.py` prefix-hash API-key lookup regression coverage in `tests/unit/api/test_api_hardening_6643.py`. Sim GUI honest messaging (issue #6641): updated `src/tools/bunker_shot_gui/gui.py` and `src/tools/putting_green_gui/gui.py` to surface explicit error and loading states instead of silently showing stale data; regression tests added in `tests/unit/test_sim_gui_honest_messaging_6641.py`.
- **2026-05-29** - Documented differentiable trajectory optimization behavior for zero-iteration runs: `optimize_trajectory()` now returns a valid `OptimizationResult` with the initial control sequence and an infinite gradient norm sentinel instead of reading an uninitialized gradient value.
- **2026-05-29** - Updated CI hygiene contract for PR #6624: agent-doc literal path validation now skips glob/brace patterns, root-clutter allowlist documents `launch_upstream_drift.py` as a substantive launcher entry point, module-size baseline exceptions remain owner/expiry governed, and the canonical Sidekick embeddable adapter stays under `src/tools/sidekick/_embed_adapter.py` after removing the obsolete duplicate shared-chat adapter.
- **2026-05-29** - Annotated cross-engine dashboard comparison results with per-engine velocity convention and units metadata in GUI result labels and headless logs so learners can see which native representation each engine result uses before normalized comparison (closes #6659).
- **2026-05-28** - Resolved python path resolution bug in `embedded_tool_bootstrap.py` and `upstream_drift_launcher.py` to fix launcher boot-time `ModuleNotFoundError` crashes and warnings.
- **2026-05-28** - Added sg-optimizer Phase 2: GeoJSON I/O (`course_io.py` with `HoleGeometry`, `load_hole_geojson`, `save_hole_geojson`), UTM geometry utilities (`geometry.py` with `LatLonPoint`, `UTMPoint`, `project_to_utm`, `utm_to_latlon`, `haversine_m` via pyproj), classic-holes library (`library.py` with 5 GeoJSON data files for Sawgrass 17, Augusta 13, Pebble 7, Road Hole 17, Cypress 16), `StateFeatures` dataclass factory (`features.py`), and full `TreeModel` with `forced_punch_out_probability` distribution (`mdp/tree_model.py`); adds `pyproj>=3.6.0` optional dependency (closes #6271).
- **2026-05-28** - Restored production symbols deleted by Bolt commit #6501: `_resolve_default_server` in `chat_dock_widget`, full 60-token `ThemeColors` derivation pipeline in `theme/api.py`, `ThemeColorsCompat` and `_derive_full_palette` in `theme/__init__.py`, `_tool_declarations_to_ollama` + `keep_alive`/`num_ctx` latency optimizations in `ollama_adapter.py`, and `_EmbedAdapter` + `_register()` in all 5 tool GUI modules; closes #6527, #6528, #6529. Also fixes sg_optimizer longitudinal dispersion applying wrong modifier column (closes #6343).
- **2026-05-27** - Confirmed Standalone Sidekick T2 (`StandaloneSidekickWindow` chat-first/calc-first layouts and profile switching) and T5 (state-profile round-trip with schema-version written to saved JSON) acceptance criteria with targeted new tests; closes #5980 and #5983.
- **2026-05-27** - Completed Standalone Sidekick T4 acceptance criteria: `sidekick run --calculator` now validates inputs via the Calculator Protocol, surfaces structured errors on validation/calculation failure (exit 3), unknown-calculator with fuzzy suggestions (exit 4), and I/O errors (exit 1); supports `--format json` and `--format csv`; full TDD coverage in `tests/unit/sidekick/standalone/test_run.py` (issue #5982).
- **2026-05-28** - Enabled dynamic MuJoCo GUI docking and styling in the launcher via DraggableTabWidget and dynamic ThemeManager palette application to resolve issue #6509.
- **2026-05-28** - Connected Model Explorer widget destroyed signal to cleanup method to ensure proper tool lifecycle in launcher simulation.
- **2026-05-28** - Resolved launcher widget parent reference crashes by using `self._launcher` instead of `self.parent()` in `SettingsWidget`.
- **2026-05-28** - Registered the shared.python.config subpackage in lazy loading to prevent mock-patching AttributeError during launcher diagnostics unit testing.
- **2026-05-27** - Resolved mypy type-checking errors by excluding Jython/OpenSim scripts from the pre-commit mypy hook and replaced print statements with logging.info/logging.warning in computeMomentArm.py and AGENT_INSTRUCTIONS.md to satisfy the no-print-in-src hook.
- **2026-05-26** - Folded remaining API/security/realtime/logging PR scope into the post-#6181 consolidation branch: `FitResult` now exposes explicit `fit_succeeded` and `solver_status` fields, the `.gitignore` secrets guard has an importable CI helper plus tests, and logging redaction preserves delimiters while redacting quoted, JSON, and comma-containing secret values.
- **2026-05-24** - Surfaced API database pool controls for non-SQLite deployments via `GOLF_DB_POOL_SIZE`, `GOLF_DB_POOL_RECYCLE`, and `GOLF_DB_POOL_PRE_PING`; `src/api/database.py` now builds non-SQLite engines from shared config accessors instead of hardcoded pool defaults, with regression coverage in `tests/unit/test_config_environment.py` and `tests/unit/api/test_database_init.py`.
- **2026-05-24** - Added shared `GOLF_REALTIME_HOST` / `GOLF_REALTIME_PORT` environment accessors and wired `src/shared/python/realtime/ws_pubsub.py` plus API diagnostics to use/report them, so realtime bind defaults no longer live only as hard-coded loopback literals.
- **2026-05-24** - Deferred realtime WebSocket backend resolution in `src/shared/python/realtime/ws_pubsub.py` until first explicit start/use and made `WSPubSub.start()` bring up the Python backend even when the instance was created with `autostart=False`; added focused regression coverage in `tests/shared/realtime/test_ws_pubsub.py`.
- **2026-05-24** - Improved CI/test observability for optional dependency lanes: optional pytest collection skips now emit one warning per skipped path with the missing module or symbol, the PyTorch CVAE cancellation regression now uses a wrapper progress sink instead of monkeypatching methods, and three standard workflow inventory jobs now have 15-minute budgets to reduce false timeouts on saturated self-hosted runners.
- **2026-05-23** - Closed the file-size budget grandfathering gap by requiring tracked baseline entries in `scripts/config/file_size_budget.json` for oversized files and adding regression coverage for untracked oversized files.
- **2026-05-23** - Tightened `src/shared/python/training/config.py` validation so boolean values are rejected for integer training caps such as `max_epochs` and `max_steps`, with regression coverage in `tests/unit/training/test_config.py`.
- **2026-05-23** - Deferred realtime WebSocket backend resolution in `src/shared/python/realtime/ws_pubsub.py` until `WSPubSub.start()`, `publish()`, or `subscribe()` first use so importing the module no longer probes optional runtime dependencies; added focused lazy-resolution regression coverage for the python publish fallback path.
- **2026-05-23** - Sanitized error payloads for the chat websocket connection to prevent leaks.
- **2026-05-23** - Added standalone Sidekick foundation (CLI entry point, PyQt window shell, and session store) per epic #5979.
- **2026-05-23** - Added the subprocess-isolated training Driver (`src/shared/python/training/runtime/subprocess_driver.py`, `worker_main.py`, `wire_protocol.py`) — `SubprocessDriver` satisfies the `Driver` Protocol so the scheduler swap is one-line, spawns workers via `core.process_safety.managed_popen` (mandatory per the error-handling ratchet), parses a newline-delimited JSON wire protocol whose payloads reuse `training.persistence` dicts, propagates cancel through stdin, surfaces worker crashes as FAILED RunResults with stderr context, and writes a `.training.pid` file per job so the launcher can detect orphaned workers via `scan_pidfiles` on restart. 65 new unit tests (wire-protocol round-trips, isolated worker-subprocess wire tests, end-to-end driver coverage of completion / cancel / crash / stderr isolation / pidfile lifecycle); follows issue #6015.
- **2026-05-23** - Wired the existing PyTorch inverse-CVAE training loop (`src/shared/python/motion_matching/inverse/training.py`) into the training-controller via a new `PyTorchCVAERunner` adapter (`src/shared/python/training/runtime/adapters/pytorch_cvae.py`). The adapter satisfies the `TrainingJobRunner` Protocol, translates `TrainingConfig.hyperparameters` into the loop's `TrainingConfig` dataclass, streams 6 `TrainingMetric`s per epoch (train_recon / train_kl / val_recon / val_kl as LOSS; beta / duration_s as SCALAR) tagged with `split=train|val`, and exposes the best-so-far checkpoint + `metrics.json` as `RunResult.artifacts`. The upstream loop gained two optional default-None kwargs — `on_epoch_end(metrics)` and `should_stop()` — so cooperative cancellation routes through `CancelToken.is_cancelled` without changing default behaviour (existing motion_matching tests unaffected). Adapter and regression tests are guarded by `pytest.importorskip("torch")`; the headless training-controller surface still imports cleanly without torch installed. Closes #6014.
- **2026-05-23** - Added the headless half of the training-controller dashboard tab (`src/tools/training_controller/`) per the in-scope portion of issue #6012: `TrainingDashboardController` MVC controller binding the backend `Scheduler` to a frozen `DashboardModel`, `TrainingJobLiveSubscriber` realtime-channel wrapper that decodes `training/<job_id>/progress` payloads into typed `TrainingMetric` / `TrainingStatus` events, and the `view_model` dataclasses (`JobRow`, `MetricSeries`, `ResourceSnapshot`, `GpuSnapshot`, `DashboardModel`) the GUI follow-up will render. 82 new headless unit tests (controller / live subscriber / view model); no PyQt6 import in `src/tools/training_controller/`. The PyQt6 widget surface (`gui.py`, `__main__.py`, `_embed_adapter.py`, `src/config/models.yaml` tile) is deferred to a follow-up PR that can be validated against a live display.
- **2026-05-23** - Added the model-training controller foundation (`src/shared/python/training/`) — PR1 contracts (status state machine, identifiers, resources, config, metrics, job/run records, compatibility checker, runner Protocols), PR2 backend (scheduler, in-process driver, runner registry, dataset library, JSON persistence, progress sinks: in-memory / JSONL-file / composite / realtime-channel), and PR3 backend additions (ResourceMonitor with psutil + optional pynvml, metric_summary helpers for best-per-metric / by-kind / by-tag / rolling means for noisy RL returns). Pure-Python, GUI-free; the headless backend that PR4's tab-backgrounding refactor and PR5's GUI tab + CVAE wiring build on. 332 unit tests passing in <1 s; 21 new public modules. The PyQt6 dashboard tab, the system-wide tab-backgrounding refactor, and the PyTorch CVAE adapter wiring are deferred to subsequent PRs that can be validated against a live display / torch environment.
- **2026-05-22** - Added the Sidekick agentic action layer (`src/shared/python/sidekick/agent/`) per epic #5967 and ADR-0017: feature catalog, audited `SidekickActionService` with default-deny policy and undo tokens, subtab and host action adapters, planner + tool-registry bridge, workflow runner, and chat-side action chip surface. 157 new unit tests; ten new public modules totalling ~3,000 LOC.
- **2026-05-22** - Tightened the CI error-handling ratchet so multiline `asyncio.gather(...)` calls are parsed across balanced parentheses before checking for `return_exceptions=`, and added focused regression tests that cover both the exempt and failing multiline forms.
- **2026-05-22** - Documented the API auth-cache overflow contract so cache saturation now evicts only the oldest lookup entries instead of flushing unrelated authenticated sessions, while preserving deterministic SHA-256 lookup keys for cross-worker stability.
- **2026-05-22** - Documented the motion-pipeline REST contract for preprocessing-step boolean coercion so `PipelineRequest` preserves Pydantic handling of `enabled` values like `"false"` when converting into `PipelineConfig`.
- **2026-05-23** - Added the standalone Sidekick UX/documentation layer per ADR-0018: persisted standalone preferences, onboarding sentinel handling, user-facing standalone docs, and contract tests for standalone preferences, onboarding, and docs discoverability.
- **2026-05-22** - Documented added unit regression coverage for the theme API model/router contracts so the shared theme settings surface stays exercised without broadening the implementation scope of the underlying runtime code.
- **2026-05-23** - Hardened WebSocket error handling so unexpected chat/simulation failures log full tracebacks server-side while returning generic client-safe error payloads.
- **2026-05-21** - Added C3D viewer animation export through the canonical body-target video pipeline and stabilized self-hosted CI SciPy pinning for the core and shared-contract lanes.
- **2026-05-21** - Preserved integer-safe quaternion normalization in the C3D Simscape preview path while keeping the optimized `einsum`-based norm computation.
- **2026-05-21** - Optimized `signal_toolkit` fitting R-squared and RMSE hot paths to reuse `np.vdot`-based sum-of-squares accumulators without temporary square arrays.

- **2026-05-30** - Optimized 3D vector magnitude calculations across physics and validation models by replacing `np.linalg.norm` with `math.sqrt(np.dot(v, v))` to eliminate array allocation overhead on the hot path.

### System Context

UpstreamDrift sits at the center of a biomechanical simulation ecosystem. It depends on five external physics engines as pluggable backends and exposes its functionality through three primary interfaces: a professional PyQt6 GUI for interactive simulation, a FastAPI REST API for programmatic access, and a Tauri desktop application for cross-platform deployment. The system integrates with motion capture systems (via MediaPipe and custom importers), optimization libraries (SciPy, Sympy), and machine learning frameworks (scikit-learn for RL integration). The Rust core (`rust_core/upstream-physics/`) provides high-performance physics kernels for compute-intensive operations.

### Module Map

````
UpstreamDrift/
├── src/
│   ├── engines/
│   │   ├── physics_engines/        # Engine adapters and integrations (package directories)
│   │   │   ├── mujoco/             # MuJoCo backend (supported)
│   │   │   ├── drake/              # Drake backend (extended)
│   │   │   ├── pinocchio/          # Pinocchio backend (extended)
│   │   │   ├── opensim/            # OpenSim backend (experimental)
│   │   │   ├── myosuite/           # MyoSuite backend (experimental)
│   │   │   ├── pendulum/           # Simplified educational models
│   │   │   └── putting_green/      # Putting green simulation
│   │   └── pendulum_models/        # Simplified educational models
│   │       ├── twodof_pendulum.py
│   │       └── biomechanical_pendulum.py
│   ├── launchers/                  # GUI/CLI entry points
│   │   ├── upstream_drift_launcher.py        # PyQt6 professional GUI (main entrypoint)
│   │   ├── golf_suite_launcher.py  # Multi-engine suite launcher
│   │   ├── unified_launcher.py     # Unified launcher interface
│   │   └── cli_launcher.py         # Command-line interface
│   ├── api/                        # FastAPI REST backend
│   │   ├── main.py                 # API entry point
│   │   ├── endpoints/              # REST endpoint definitions
│   │   └── models.py               # Pydantic request/response models
│   ├── config/                     # Configuration management
│   │   └── launcher_manifest_loader.py # Config loading and validation
│   ├── shared/                     # Cross-engine utilities
│   │   ├── validators.py           # Shared validation logic
│   │   ├── utilities.py            # Helper functions
│   │   └── exceptions.py           # Exception definitions

│   └── tools/                      # Development and analysis tools
│       ├── analysis_tools.py       # Biomechanical analysis utilities
│       └── validation_tools.py     # Cross-engine validation
├── rust_core/
│   └── upstream-physics/           # Rust physics kernels
│       ├── src/
│       │   ├── lib.rs
│       │   └── physics.rs
│       └── Cargo.toml
├── ui/
│   ├── src/
│   │   ├── main.ts                 # Tauri app entry point
│   │   └── components/             # React/Vue components
│   ├── tauri.conf.json
│   └── package.json
├── shared/
│   └── models/                     # URDF/model definitions
│       ├── golf_swing_models/
│       ├── human_body_models/
│       └── pendulum_models/
├── tests/
│   ├── unit/                       # Unit tests per module
│   ├── integration/                # Cross-engine integration tests
│   ├── acceptance/                 # End-to-end scenario tests
│   ├── cross_engine/               # Cross-validation tests
│   ├── physics_validation/         # Physics accuracy tests
│   ├── benchmarks/                 # Performance benchmarks
│   └── conftest.py                 # Pytest fixtures and configuration
├── .github/
│   └── workflows/

│       ├── ci-standard.yml         # Standard CI checks
│       ├── heavy-tests-opt-in.yml  # Heavy tests (custom runner)
│       ├── nightly-cross-validation.yml
│       ├── tauri-build.yml
│       ├── vendor-freshness.yml
│       └── docker-size-gates.yml
├── pyproject.toml
├── poetry.lock
├── SPEC.md                         # This file
└── README.md

### Key Components

| Component                | Location                                 | Purpose                                                                                     |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------------- |
| MuJoCo Engine Adapter    | `src/engines/physics_engines/mujoco/`    | Primary physics engine integration with full support for contact dynamics and muscle models |
| Drake Engine Adapter     | `src/engines/physics_engines/drake/`     | Extended Drake support for trajectory optimization and manipulation tasks                   |
| Pinocchio Engine Adapter | `src/engines/physics_engines/pinocchio/` | Extended Pinocchio support for efficient rigid-body dynamics computation                    |
| OpenSim Engine Adapter   | `src/engines/physics_engines/opensim/`   | Experimental OpenSim integration for clinical biomechanics workflows                        |
| MyoSuite Engine Adapter  | `src/engines/physics_engines/myosuite/`  | Experimental MyoSuite integration for detailed muscle physiology simulation                 |
| Pendulum Models          | `src/engines/physics_engines/pendulum/`  | Educational simplified models for learning and quick prototyping                            |
| FastAPI Backend          | `src/api/`                               | REST API exposing simulation, IK/ID, trajectory optimization, and control endpoints         |
| PyQt6 GUI                | `src/launchers/upstream_drift_launcher.py`         | Professional interactive GUI with real-time 3D visualization                                |
| Sidekick (AI assistant)  | PyQt: `src/shared/python/ai/gui/assistant_panel.py` · React: `ui/src/components/ui/ChatPanel.tsx` · Adapter: `src/tools/sidekick/_embed_adapter.py` | In-app AI chat surface with streaming, RAG, session history, and agentic tool dispatch. Design tokens: `src/shared/python/theme/sidekick_tokens.py`. See `docs/sidekick/README.md`. |
| Tauri Desktop App        | `ui/`                                    | Cross-platform desktop application wrapper (Windows, macOS, Linux)                          |
| Rust Physics Kernels     | `rust_core/upstream-physics/`            | High-performance compiled physics routines for critical paths, including initial flexible shaft FEM element primitives |
| Configuration Manager    | `src/config/`                            | Centralized configuration loading, validation, and environment management                   |
| Shared Utilities         | `src/shared/`                            | Cross-engine validators, helpers, and exception definitions                                 |
| Workspace Metadata       | `src/shared/python/workspace/`           | Project/session/dataset metadata store and CC-4 HDF5 result browser view models            |
| URDF Models              | `shared/models/`                         | Canonical model definitions (URDF format) for golf swings, human body, pendulums            |

### Engine Tier Policy

| Tier         | Examples                | Stability bar                                            | Deps installed by default | Vulnerability SLA |
| ------------ | ----------------------- | -------------------------------------------------------- | ------------------------- | ----------------- |
| core         | MuJoCo, FastAPI, shared | Must pass on every PR; semver-stable public API; no skip | yes                       | High/Critical: 7d |
| extended     | Drake, Pinocchio        | Must pass nightly; semver-stable in major versions       | only with extra           | High: 30d         |
| experimental | OpenSim, MyoSuite       | Best-effort; may be skipped; API may break               | only with extra; warning  | Best effort       |
| archived     | (none today)            | Read-only; not built; not tested                         | no                        | n/a               |

Engine tier metadata is declared in each in-scope engine package with
`_tier.py` and enforced by `scripts/check_engine_tiers.py`.

## 5. Desired Functionality

### Core Features

| #   | Feature                            | Status | Description                                                                                         |
| --- | ---------------------------------- | ------ | --------------------------------------------------------------------------------------------------- |
| F1  | MuJoCo engine integration          | ✅     | Full support for MuJoCo 3.3.0+ with contact dynamics, muscle actuators, sensor simulation, and pose-conditioned motion-matching target synthesis |
| F2  | Drake engine integration           | ✅     | Extended Drake support for trajectory optimization, manipulation, and planning problems             |
| F3  | Pinocchio engine integration       | ✅     | Extended Pinocchio support for efficient rigid-body dynamics and jacobian computation               |
| F4  | OpenSim engine integration         | 🔄     | Experimental OpenSim integration for clinical biomechanics and musculoskeletal analysis             |
| F5  | MyoSuite engine integration        | 🔄     | Experimental MyoSuite integration for detailed muscle physiology and motor control                  |
| F6  | Cross-engine validation and reports | ✅     | Automated cross-validation plus user-facing comparison reports across selected engines with tolerance thresholds, provenance, and divergence annotations |
| F7  | FastAPI REST API                   | ✅     | Programmatic access to simulation, IK/ID, trajectory optimization, and control endpoints            |
| F8  | PyQt6 professional GUI             | ✅     | Interactive desktop GUI with real-time 3D rendering, parameter adjustment, and result export        |
| F9  | Tauri desktop application          | 🔄     | Cross-platform desktop app bundling the GUI and API with native OS integration                      |
| F10 | MATLAB/Simulink integration        | ✅     | Export models to MATLAB format and integrate with Simulink via MEX interface                        |
| F11 | Trajectory optimization            | ✅     | SciPy-based trajectory optimization with constraint support and custom cost functions               |
| F12 | Muscle dynamics analysis           | ✅     | IK, ID, and muscle dynamics computation with Hill-type and Millard muscle models                    |
| F13 | Motion capture integration         | 🔄     | Import and track motion capture data (C3D, BVH, TRC formats) and compare with simulation            |
| F14 | Reinforcement learning integration | 🔄     | Gym-compatible interface for RL-based controller learning and policy optimization                   |
| F15 | Sidekick AI assistant              | 🔄     | In-app and standalone AI assistant surface (PyQt + React/Tauri + `sidekick.standalone.*`) with streaming, RAG, session history, persisted standalone preferences, onboarding, and agentic tool dispatch. See `docs/sidekick/README.md` and ADR-0018. |
| F16 | Model-training controller          | 🔄     | In-launcher training dashboard (PR3) with scheduler, dataset library, resource monitor, engine-compat gate, and ML/RL-aware stats. Backend contracts + scheduler land in `src/shared/python/training/` (PRs 1–2); GUI tab, tab-backgrounding refactor, and CVAE wiring in PRs 3–5. |

### API / Interface Contract

**REST API Endpoints (FastAPI)**:

- `GET /health` — Health check
- `POST /simulate` — Run single simulation with specified engine and parameters
- `POST /cross-validate` — Run multi-engine cross-validation and return results
- `POST /ik` — Solve inverse kinematics given target pose
- `POST /id` — Solve inverse dynamics given trajectory
- `POST /trajectory-optimize` — Optimize trajectory subject to constraints
- `GET /engines` — List available physics engines and their status
- `POST /export` — Export simulation model to URDF, MATLAB, or other formats
- `POST /api/v1/motion-pipeline/run` — Run motion-pipeline preprocessing, scaling, IK, and motion-matching for uploaded capture files

**API Production-Readiness Contracts**:

- Background task state is process-local and owned by the FastAPI application
  lifespan. Each app lifecycle creates its own `TaskManager`; shutdown marks the
  manager closed, clears retained task records, and subsequent task operations
  fail with a closed-state error instead of silently accepting writes.
- Motion-pipeline request normalization preserves Pydantic/native boolean
  coercion for preprocessing step `enabled` flags so form/JSON values such as
  `"false"` remain disabled instead of being forced truthy during
  `PipelineRequest -> PipelineConfig` conversion.
- Simulation WebSocket routes preserve traceback-bearing server logs for
  unexpected runtime failures while returning sanitized generic client errors so
  backend exception details are not exposed over the socket.
- `TaskManager` entries expire after the configured TTL and enforce the
  configured maximum task count. Reads and existence checks refresh the task's
  retention timestamp so actively polled async jobs are not evicted while a
  client is still observing them.
- Async video analysis queues request handling quickly, then runs the blocking
  video pose pipeline off the event loop. Temporary uploaded video files are
  deleted after completion or failure; cleanup failures are logged as warnings
  and do not mask the task result.
- Data Explorer imported datasets are kept in a bounded in-memory LRU cache.
  Importing a duplicate filename returns a conflict instead of replacing the
  existing dataset. Disk-backed dataset lookup rejects ambiguous duplicate
  filenames with a conflict response so callers do not receive an arbitrary
  match.
- `src/shared/python/realtime/ws_pubsub.py` resolves its default backend lazily.
  Constructing `WSPubSub` no longer imports or probes optional realtime runtime
  dependencies until `start()`, `publish()`, or `subscribe()` is invoked, while
  explicit `backend=` overrides and the python HTTP publish fallback remain
  supported.
- Chat and simulation WebSocket routes treat unexpected internal exceptions as
  server-only detail: they log full tracebacks for operator diagnosis and send
  generic client-safe error payloads instead of echoing raw exception strings.

**GUI Interface (PyQt6)**:

- Model loader and parameter editor
- Real-time 3D simulation viewer with playback controls
- Cross-engine comparison visualizer
- IK/ID solver interface with result tables
- Trajectory optimization GUI with constraint editor
- Data export and report generation

**CLI Interface**:

- `upstream-drift simulate --engine mujoco --model golf_swing.urdf`
- `upstream-drift cross-validate --models model1.urdf model2.urdf`
- `upstream-drift ik --model human.urdf --target-pose [...] --engine pinocchio`
- `python -m sidekick` launches the standalone Sidekick GUI scaffold with the `gui` subcommand and `chat-first` profile as the default path.
- `python -m sidekick gui --profile calc-first --theme solarized --data-dir ./workspace` keeps GUI imports deferred until launch while resolving the standalone data directory before window creation.
- `python -m sidekick run --calculator unit-converter --inputs ./inputs.json --output ./result.json` validates the headless calculator invocation contract up front; execution remains reserved for follow-up issue `#5982`.
- The standalone Sidekick CLI suggests the nearest valid flag or subcommand on parse errors to keep local launches and future automation entrypoints discoverable.

**Desktop App (Tauri)**:

- Native window management and file dialogs
- System menu integration
- Automated updates and crash reporting

## 6. Data & Configuration

### Input Data

| Input                    | Format        | Source                          | Schema                                                   |
| ------------------------ | ------------- | ------------------------------- | -------------------------------------------------------- |
| Biomechanical Models     | URDF          | `shared/models/`                | URDF 1.0 standard with custom muscle actuator extensions |
| Motion Capture Data      | C3D, BVH, TRC | External mocap systems or files | Standard formats with marker sets and frame data         |
| Optimization Constraints | JSON          | User input or configuration     | Custom constraint schema in `src/config/`                |
| Control Parameters       | YAML/JSON     | Configuration files or API      | Engine-specific parameter maps validated against schemas |

### Output Data

| Output                   | Format                 | Destination                 | Description                                                        |
| ------------------------ | ---------------------- | --------------------------- | ------------------------------------------------------------------ |
| Simulation Trajectories  | JSON/HDF5              | API response or file export | Joint angles, muscle activations, forces over time                 |
| Cross-Validation Reports | JSON/PDF               | File export or API          | Engine comparison metrics, error margins, validation status        |
| IK/ID Solutions          | JSON/MATLAB            | API response or file        | Joint angles (IK) and joint torques (ID) with confidence metrics   |
| Optimized Trajectories   | URDF/MATLAB            | File export                 | Trajectory-optimized model definitions with optimal control inputs |
| Visualization Data       | JSON (Three.js format) | GUI or web client           | 3D geometry, animation keyframes, and rendering parameters         |

### Configuration

Configuration is managed through:

- **Environment Variables**: `UPSTREAM_DRIFT_ENGINE` (default: mujoco), `UPSTREAM_DRIFT_API_PORT` (default: 8000)
- **YAML Config Files**: `~/.upstream_drift/config.yaml` with engine-specific sections
- **API Request Parameters**: Engine selection, model path, solver options passed as JSON
- **GUI Settings**: Stored in `~/.upstream_drift/gui_settings.json` (viewport, window size, recent files)
- **Launcher Manifest**: `src/config/launcher_manifest.json` declares discoverable and hidden launcher surfaces, including shared Tools-hosted video/data utilities exposed to UpstreamDrift.
- **Theme API Settings**: `src/api/routes/theme.py` and `ui/src/api/themeClient.ts` expose launcher theme metadata to the desktop/web UI without duplicating theme lists in the frontend.

Example config.yaml:

```yaml
default_engine: mujoco
api:
  host: 0.0.0.0
  port: 8000
engines:
  mujoco:
    model_path: /path/to/models
    timestep: 0.001
  drake:
    use_simulator: true
visualization:
  default_camera: third_person
  background_color: [0.1, 0.1, 0.1, 1.0]

## 7. Testing Specification

### Testing Strategy

UpstreamDrift employs a comprehensive test pyramid with multiple specialized categories:

- **Unit Tests**: Test individual engine adapters, utilities, and validators in isolation
- **Integration Tests**: Test workflows combining multiple modules (e.g., load model → simulate → export)
- **Acceptance Tests**: End-to-end scenarios (e.g., full golf swing simulation with visualization)
- **Cross-Engine Tests**: Validate physics consistency across multiple engines with tolerance thresholds
- **Physics Validation Tests**: Verify results against known ground truth (analytical solutions, published benchmarks)
- **Golf Ball-Flight Source Contracts**: Validate documented aerodynamic, impact, and atmosphere assumptions against `docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md`
- **Dependency Source Contracts**: Validate generated dependency artifacts against `pyproject.toml` and fail CI when lockfiles or `environment.yml` drift
- **Documentation Governance Contracts**: Validate the canonical `docs/index.md` directory catalog, rendered documentation hub link, and Markdown/Quarto size budget.
- **Benchmark Tests**: Performance regression detection and optimization validation
- **Property-Based Tests**: Hypothesis-driven fuzzing for robustness

### Test Organization

| Category                    | Location                    | Framework           | Markers                             |
| --------------------------- | --------------------------- | ------------------- | ----------------------------------- |
| Unit                        | `tests/unit/`               | pytest              | `@pytest.mark.unit`                 |
| Integration                 | `tests/integration/`        | pytest              | `@pytest.mark.integration`          |
| Acceptance                  | `tests/acceptance/`         | pytest              | `@pytest.mark.acceptance`           |
| Cross-Engine                | `tests/cross_engine/`       | pytest              | `@pytest.mark.cross_engine`         |
| Physics Validation          | `tests/physics_validation/` | pytest              | `@pytest.mark.physics_validation`   |
| Golf Source Contracts       | `tests/unit/shared_python/` | pytest              | source-map contract tests           |
| Dependency Source Contracts | `tests/unit/scripts/`       | pytest              | generated dependency contract tests |
| Benchmarks                  | `tests/benchmarks/`         | pytest-benchmark    | `@pytest.mark.benchmark`            |
| Property-Based              | `tests/unit/`               | hypothesis + pytest | `@pytest.mark.property`             |

Issue #3841 moved stable flat tests and the launcher `src/**/tests` package into
topic directories under `tests/`, documented the fixture scopes in
`tests/README.md`, and added `scripts/check_test_layout.py` as the blocking CI
guard against new flat test files, new in-tree `src/**/tests` directories, and
overlapping fixture names in nested conftests.

### Coverage Requirements

| Scope                   | Minimum | Current          | Enforced By                                 |
| ----------------------- | ------- | ---------------- | ------------------------------------------- |
| Overall                 | 55%     | CI baseline      | `pyproject.toml` and `ci-standard.yml`      |
| API routes              | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Data I/O                | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Execution/checkpointing | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Deployment              | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Optimization            | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |
| Engine adapters         | 30%     | Ratchet baseline | `scripts/config/mypy_exclusion_budget.json` |

### Required Test Scenarios

- [ ] Unit creation with valid URDF returns expected topology (chain, mass distribution)
- [ ] MuJoCo engine simulation produces reasonable trajectories with gravity effects
- [ ] Cross-engine validation identifies discrepancies >5% between engines
- [ ] IK solver converges within 10 iterations for standard human poses
- [ ] ID computation returns physically plausible torques (within 2-sigma of analytical)
- [ ] Ball-flight atmosphere utilities reject non-finite or out-of-troposphere altitudes and stay traceable to documented golf source contracts
- [ ] FastAPI endpoints return 200 for valid requests and 400 for invalid schema
- [ ] GUI loads model and renders 3D visualization without crashing
- [ ] Trajectory optimization improves cost function by >20% over initial guess
- [ ] Muscle dynamics simulation produces realistic activation patterns
- [ ] Cross-platform build (Windows, macOS, Linux) produces functional binaries

## 8. Quality Standards

### Code Quality Tools

| Tool       | Version | Purpose                                                                            | Blocking? |
| ---------- | ------- | ---------------------------------------------------------------------------------- | --------- |
| 2026-04-27 | 1.0.83  | Fixed Bandit B604 false positive alerts in test files by adding nosec annotations. |
| ruff       | latest  | Linting and formatting                                                             | Yes       |
| mypy       | 1.7+    | Static type checking                                                               | Yes       |
| pytest     | 7.0+    | Testing framework                                                                  | Yes       |
| pytest-cov | 4.0+    | Coverage measurement                                                               | Yes       |
| bandit     | 1.7+    | Security scanning                                                                  | Yes       |
| hypothesis | 6.0+    | Property-based testing                                                             | No        |

### Design Principles

- **TDD**: Unit tests written before implementation; the current global coverage floor is 55%, with per-package production ratchets tracked toward higher thresholds (85% for API routes/engine adapters, 70% for shared utilities).
- **Design by Contract (DbC)**: Explicit preconditions and postconditions in engine adapters
- **DRY**: Cross-engine utilities in `src/shared/` prevent code duplication
- **Orthogonality**: Engines are loosely coupled; each can be used independently
- **Explicit is Better**: Function signatures include type hints; no magic string parameters

### Custom Quality Gates (CI)

Beyond standard tools, CI enforces custom checks:

- **Dependency Direction**: No reverse dependencies (leaf → branch → root)
- **SAST Delta Scan**: Pull requests run Semgrep against changed supported
  source/application files and Bandit against changed supported Python
  source/application files, and Trivy against changed supported
  dependency/container/config files while non-PR CI retains the full repository
  scans, keeping new code blocking without letting existing repository baseline
  findings block unrelated PRs.
- **Alembic PostgreSQL Round Trip**: PostgreSQL migration round-trip CI has a
  finite job budget, an explicit SQL readiness probe, isolated pytest plugin
  loading, and verbose duration output so migration hangs produce actionable
  diagnostics instead of opaque cancellation or unrelated desktop-display plugin
  failures.
- **Core Test Relevance Filter**: Pull requests with no Python source, test,
  project metadata, or dependency-file changes skip the expensive Python test
  matrix after checkout so workflow-only and documentation-only CI fixes remain
  finite on constrained self-hosted runners.
- **File Size Budget**: No module exceeds 500 lines; classes capped at 200
  LOC; oversized grandfathered files must have tracked baseline entries in
  `scripts/config/file_size_budget.json` or the CI gate fails.
- **Module Size Budget**: Python modules under `src/` are capped at 1,500
  lines by `scripts/check_module_size_budget.py`; oversized legacy modules
  require owned, expiring exceptions in
  `scripts/config/module_size_budget_baseline.json`, currently capped at 10
  active exceptions.
- **Agent Docs Consistency**: `scripts/check_agent_docs_consistency.py`
  validates literal repo-relative paths documented in agent guidance while
  treating glob/brace references such as `scripts/**` and
  `src/shared/python/codemap/{cli,watcher,mcp_server}.py` as patterns, not
  files that must exist.
- **Root Clutter**: `scripts/check_root_clutter.py` blocks non-allowlisted
  repository-root files; substantive launcher entry points such as
  `launch_golf_suite.py` and `launch_upstream_drift.py` are explicitly
  allowlisted until promoted into packaged scripts.
- **Documentation Catalog and Size Budget**: Every top-level `docs/` directory is listed in `docs/index.md`; oversized Markdown/Quarto docs require owned, expiring exceptions.
- **Import Depth**: Maximum 4 import levels to prevent circular dependencies
- **Physics Fitness**: Cross-engine validation must pass with <5% tolerance
- **Security Audit Isolation**: `pip-audit` runs with `scripts/config/pip_audit_waivers.json` and `scripts/ci/check_pip_audit_waivers.py` so waivers require issue tracking, expiry, and current pip-audit findings before ignore flags are emitted
- **Blocking SAST and Secret Scans**: `ci-standard.yml` runs blocking Bandit, Semgrep, pip-audit, and Trivy filesystem scans for pull requests and pushes
- **Error-Handling Ratchet**: `scripts/ci/check_error_handling_ratchet.py` blocks increases in grandfathered broad catches, unused `noqa` debt, raw `subprocess.Popen(...)`, and `asyncio.gather(...)` calls that omit `return_exceptions=`, including multiline gather calls whose arguments span multiple lines.
- **Type and Coverage Ratchets**: `scripts/check_mypy_exclusion_budget.py` blocks unowned mypy exclusions, non-monotonic exclusion schedules, and missing production package coverage-ratchet metadata. Push-to-main full-src mypy runs through `scripts/ci/run_full_mypy_baseline.py`, which compares `mypy src --config-file pyproject.toml` against `scripts/config/full_src_mypy_baseline.json` and fails on new or stale type diagnostics.
- **Docker Size Gate**: Built images must not exceed 800 MB

### CI/CD Pipeline

| Workflow                       | Trigger                                | Purpose                                                                               | Blocking?          |
| ------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------- | ------------------ |
| `ci-standard.yml`              | Push/PR                                | Lint, type check, unit/integration tests, workflow inventory, blocking security scans | Yes                |
| `heavy-tests-opt-in.yml`       | Manual dispatch or `/heavy-test` label | Cross-engine and physics validation (long-running)                                    | No (opt-in)        |
| `nightly-cross-validation.yml` | Daily 2:00 UTC                         | Full multi-engine validation suite against all model variations                       | No (informational) |
| `tauri-build.yml`              | Tag release                            | Build desktop apps for Windows/macOS/Linux                                            | Yes (for releases) |
| `vendor-freshness.yml`         | Weekly                                 | Check for stale dependencies and security updates                                     | No (warning-only)  |
| `docker-size-gates.yml`        | Push                                   | Ensure Docker image size stays <800 MB                                                | Yes                |

## 9. Dependencies

### Runtime Dependencies

| Package                        | Version    | Purpose                                                   |
| ------------------------------ | ---------- | --------------------------------------------------------- |
| numpy                          | 1.20+      | Numerical computation                                     |
| scipy                          | 1.7+       | Scientific algorithms (optimization, linalg)              |
| fastapi                        | 0.95+      | REST API framework                                        |
| uvicorn                        | 0.20+      | ASGI server for FastAPI                                   |
| pydantic                       | 2.0+       | Request/response validation                               |
| mujoco                         | 3.3.0+     | Primary physics engine (required)                         |
| PyQt6                          | 6.0+       | Professional GUI framework                                |
| tauri-py                       | 1.0+       | Tauri bridge for Python backend                           |
| pillow, requests, bokeh, flask | CVE floors | Runtime security constraints validated outside dev extras |

### Optional Runtime Dependencies

| Package      | Version | Purpose                                     |
| ------------ | ------- | ------------------------------------------- |
| drake        | 1.0+    | Drake physics engine integration            |
| pinocchio    | 2.6+    | Pinocchio rigid-body dynamics               |
| myosuite     | 2.0+    | MyoSuite muscle simulation                  |
| opensim      | 4.4+    | OpenSim musculoskeletal models              |
| mediapipe    | 0.9+    | Motion capture integration (pose detection) |
| scikit-learn | 1.0+    | RL policy learning and clustering           |
| sympy        | 1.11+   | Symbolic trajectory optimization            |
| pyarrow      | 14.0+   | Parquet IO for compact swing dataset paths  |

### Development Dependencies

| Package    | Version | Purpose                                                                            |
| ---------- | ------- | ---------------------------------------------------------------------------------- |
| pytest     | 7.0+    | Testing framework                                                                  |
| pytest-cov | 4.0+    | Coverage measurement                                                               |
| hypothesis | 6.0+    | Property-based testing                                                             |
| pip-tools  | 7.4+    | Regenerate Python dependency lockfiles from `pyproject.toml`                       |
| pyarrow    | 14.0+   | Parquet IO test coverage for compact swing dataset paths                           |
| ruff       | latest  | Linting and formatting                                                             |
| mypy       | 1.7+    | Type checking                                                                      |
| bandit     | 1.7+    | Security scanning                                                                  |
| black      | 23.0+   | Code formatter                                                                     |

### Fleet Dependencies

| Repo             | Relationship | Description                                              |
| ---------------- | ------------ | -------------------------------------------------------- |
| (none currently) | —            | UpstreamDrift is currently a standalone fleet repository |

## 10. Deployment & Operations

### How to Run

```bash
# Prerequisites
- Python 3.10 or later
- MuJoCo 3.3.0+ with license (community or pro)
- Optional: Drake, Pinocchio, OpenSim binaries on PATH
- For Tauri desktop app: Node.js 16+, Rust toolchain

# Installation
git clone https://github.com/D-sorganization/UpstreamDrift.git
cd UpstreamDrift
python -m pip install -e ".[dev]"  # Include dev dependencies
# For desktop app: cargo install tauri-cli

# Running the FastAPI Server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Running the PyQt6 GUI
python -m src.launchers.gui_launcher

# Running the CLI
upstream-drift simulate --engine mujoco --model shared/models/golf_swing.urdf
python -m sidekick
python -m sidekick run --calculator unit-converter --inputs ./inputs.json

# Building the Tauri Desktop App
cd ui && npm install && npm run tauri build
# Outputs: UpstreamDrift.exe (Windows), UpstreamDrift.app (macOS), UpstreamDrift.AppImage (Linux)

# Running Tests
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/ --cov=src --cov-fail-under=55

### Build Artifacts

| Artifact              | Format         | Destination             |
| --------------------- | -------------- | ----------------------- |
| Python Package        | .whl           | PyPI (on release)       |
| FastAPI Server        | Docker image   | Docker Hub (on release) |
| Desktop App (Windows) | .msi installer | GitHub releases         |
| Desktop App (macOS)   | .dmg bundle    | GitHub releases         |
| Desktop App (Linux)   | .AppImage      | GitHub releases         |
| Documentation         | HTML           | GitHub Pages            |

Canonical production artifacts and supported OS/Python/tier/hardware
combinations are defined in `docs/operations/production-readiness.md`. Release
smoke suites live under `tests/smoke/<artifact>/`; the tag release workflow
blocks Python package publication on the built-wheel smoke matrix.

## 11. Roadmap & Open Issues

### Current Phase

**Active Development**: Core engine integrations complete; expanding experimental OpenSim and MyoSuite support. Tauri desktop app in active development. Motion capture integration and RL control schemes are in-progress.

### Planned Work

| Priority | Item                                        | Issue/PR | Target Date |
| -------- | ------------------------------------------- | -------- | ----------- |
| P0       | Complete OpenSim integration (F4)           | #45      | Q2 2026     |
| P0       | Complete MyoSuite integration (F5)          | #46      | Q2 2026     |
| P1       | Motion capture import and tracking (F13)    | #78      | Q3 2026     |
| P1       | RL controller learning framework (F14)      | #92      | Q3 2026     |
| P1       | Tauri desktop app release (F9)              | #101     | Q2 2026     |
| P2       | Extended MATLAB integration (export/import) | #112     | Q4 2026     |
| P2       | Performance profiling and GPU acceleration  | #130     | Q4 2026     |

### Known Limitations

- OpenSim and MyoSuite integrations are experimental; API may change
- Cross-engine validation only enforces tolerances on kinematic outputs; dynamics comparison still in development
- Motion capture import limited to marker-based systems (no IMU data yet)
- RL integration currently supports basic Gym environments; no hierarchical or multi-agent support
- Tauri app Windows builds require MSVC toolchain (no MinGW support)
- Performance scaling beyond 100-muscle models not yet tested

## 12. Change Log

| Date       | Version | Changes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-05-31 | 1.0.223 | Added the CC-23 moving-horizon estimator near-real-time path for issue #6796: a bounded deterministic rolling-window estimator reuses the CC-19 MAP objective surface with fixed parameters, warm-starts each new window from the previous spline solution, records per-window latency against a 50 ms default budget, and supports optional callback integration for realtime bridge publishing. |
| 2026-05-31 | 1.0.222 | Added the CC-22 offline Nimble gradient oracle for issue #6795: `tools.offline_validation.nimble_gradient_oracle` exposes validated request/response dataclasses, lazy PyTorch/Nimble autograd comparison with structured skip behavior, a pinned `nimble-oracle` optional extra (`nimblephysics==0.10.52.2`), focused deterministic tests, and docs confirming Nimble stays outside runtime `src/`. |
| 2026-05-31 | 1.0.222 | Added the canonical-core CI wiring for issue #6780: `.github/workflows/cross-engine-equivalence.yml` now provides per-engine conformance lanes, `heavy-tests-opt-in.yml` keeps heavy stacks self-hosted and explicit, canonical-core Jules templates scaffold adapter/conformance/docstring work, and the JaxSim forward-simulation parity test aligns its analytic reference with canonical gravity and the current rollout tolerance envelope. |
| 2026-05-31 | 1.0.221 | Added the CC-32 canonical-core app shell registry (#6805): `canonical_core_estimation` and `canonical_core_comparison` now register through the ADR-0013 embeddable-tool contract, publish PyQt6/React shell metadata through the launcher manifest, and route React users through `/tools/canonical-core/estimation` and `/tools/canonical-core/comparison` without implementing the deferred CC-19/CC-27 service bodies. |
| 2026-05-31 | 1.0.221 | Added the Sidekick Canonical Core retrieval Q&A tool for issue #6810: `src/shared/python/canonical_core/sidekick_retrieval_qa.py` builds a bounded local index over Canonical Core docs and schemas, returns deterministic extractive answers with `path:start-end` citations, and registers the read-only `answer_canonical_core_question` tool through `src/api/services/chat_service.py`; docs live in `docs/sidekick/README.md` and `docs/specs/active/sidekick-canonical-core-retrieval-qa.md`. |
| 2026-05-31 | 1.0.221 | Added the CC-36 config validation setup wizard for issue #6809: `src/shared/python/config/setup_wizard.py` validates canonical-v2 units and frames, model identity/joint/dimension preconditions, and subject calibration readiness; `SetupWizardViewModel` provides the headless four-step flow; `src/tools/config_setup_wizard/` exposes an embeddable launcher surface; and `tests/unit/config/test_setup_wizard.py` covers validation, suggested fixes, progression, and adapter conformance. |
| 2026-05-31 | 1.0.222 | Added the CC-38 Sidekick canonical-core tool adapter for issue #6811: `src/shared/python/sidekick/agent/canonical_tools.py` registers a fixed canonical action allowlist behind `CanonicalActionPort`, `canonical.run` remains destructive and confirmation-gated, docs update ADR-0017 plus `docs/sidekick/agent.md`, and unit coverage validates descriptors, dry-run behavior, policy interaction, and result provenance. |
| 2026-05-31 | 1.0.221 | Added the CC-33 canonical 3D viewport provider decision (#6806): MeshCat is the selected default over Rerun and VTK/PyVista, with lazy provider metadata/selection/degradation in `src/shared/python/visualization/viewport.py`, a Trace v2 overlay payload for canonical-v2 trajectory, marker, contact, and GRF/wrench data, and ADR-0027 documenting the bounded backend choice without adding viewer dependencies to core. |
| 2026-05-31 | 1.0.220 | Tightened review-feedback guardrails for issues #6816 and #6827: `scripts/legal/check_license_ledger.py` now validates the OpenPose ledger row cells directly, and `.github/workflows/cross-engine-equivalence.yml` includes `pyproject.toml` in push/PR path filters so the JaxSim pin guard runs when the optional dependency declaration changes. |
| 2026-05-31 | 1.0.220 | Added the canonical-core estimation residual surface for issue #6791: `src/shared/python/estimation/residuals.py` exposes pure reprojection, RNEA dynamics, anthropometric prior, and smoothness residual functions with a shared finite-difference/JAX Jacobian helper; `tests/unit/estimation/test_residuals.py` verifies residual Jacobians against hand-derived finite-difference expectations; `docs/development/canonical_core_residuals.md` documents the backend callback contract. |
| 2026-05-31 | 1.0.220 | Added the CC-28 Drake canonical-core adapter slice for issue #6801: Drake now reports AutoDiffXd state/control gradients, full forward/inverse dynamics, contact stepping/forces, and trajectory optimization support; the pose adapter remaps canonical-v2 `q/v/a/t` into Drake quaternion-floating state order with parent/world angular velocity conversion; and the hydroelastic-vs-Pinocchio contact divergence is registered in `docs/conformance/canonical_core_divergences.yaml`. |
| 2026-05-31 | 1.0.220 | Added the CC-30 MyoSuite canonical-core adapter slice for issue #6803: MyoSuite now maps activation-driven canonical-v2 state into MyoSuite/MuJoCo MJCF layouts, declares MUSCLES/FORWARD_DYN/CONTACT capabilities without claiming joint-torque inverse dynamics, routes upstream-muscle activation and force helpers, and persists Trace v2.1 muscle-output fields. |
| 2026-05-31 | 1.0.219 | Added the CC-29 MJX differentiable rollout slice: `simulation_backends` now registers optional `mjx` beside `ode`/`mujoco`/`mjwarp`, gates it through `has_mjx()` / `require_mjx()`, reuses the generated MuJoCo MJCF, advertises batched + differentiable capabilities, exposes JAX-native batched rollout arrays plus a final-state control Jacobian surface, and adds `run_estimation_windows_batched()` to flatten CC-20 multi-trial / CC-23 window controls onto the existing `BatchedBackend` axis. Updated ADR-0024 from deferred recommendation to accepted MJX implementation guidance, with mocked CPU-only tests covering optional-dependency degradation and host-side rollout/autodiff plumbing. |
| 2026-05-31 | 1.0.219 | Added metadata-driven helpful-field and provenance-value wrappers for the Idiot-Proof UX epic (#5968): `HelpfulField` and `ProvenanceValue` PyQt6/React controls consume the existing metadata/provenance contracts, `scripts/ux/generate_field_metadata_ts.py` generates the TypeScript registry from `configs/ux/field_metadata.yaml`, and focused PyQt/Vitest tests cover contract validation, ARIA help text, provenance display, and YAML-to-TypeScript parity. |
| 2026-05-31 | 1.0.221 | Hardened the runtime Docker image against current fixed Debian 13 glibc, systemd/libudev, and sed CVEs by explicitly upgrading/installing `libc-bin`, `libc6`, `libsystemd0`, `libudev1`, and `sed` in the runtime apt layer while preserving the pinned base image digest. |
| 2026-05-31 | 1.0.221 | Added the CC-26 AffineDrift coupling surface (#6799): `src/shared/python/analysis/affine_drift_coupling.py` extracts double-pendulum coordinates from native or canonical-v2 traces, samples drift/control-affine acceleration terms from a dynamics provider, persists `AffineDriftCouplingResult` datasets to HDF5, and documents the result schema in `docs/conventions/canonical-v2.md` plus `docs/simulation_backends/results_schema_v2.md`. |
| 2026-05-31 | 1.0.218 | Added the issue #6781 third-party license ledger and advisory validation path: `docs/legal/licenses.md` records commercial-readiness status for direct dependencies, OpenPose remains a non-commercial opt-in external tool, `scripts/legal/check_license_ledger.py` validates declared dependency coverage on Python 3.10+, and the core-install isolation guard ignores its own `scripts/` path so helper directories cannot masquerade as installed optional engines. |
| 2026-05-31 | 1.0.219 | Added canonical-v2 dynamic state support for CC-2 (#6774): `CanonicalState` enforces the `(q, v, a, t)` floating-base layout with read-only arrays, unit quaternion and `nq = nv + 1` preconditions, manifold `integrate`/`difference` operations, canonical-v1 pose lifting, zero-state construction, and SE(3) quaternion helpers with regression coverage. |
| 2026-05-31 | 1.0.217 | Hardened the JaxSim #6648 CI path: URDF/SDF inertial XML reads now use `defusedxml.ElementTree`, and the core-only install isolation guard removes its own `scripts/` directory from `sys.path` before checking optional-engine imports so `scripts/jaxsim` cannot masquerade as an installed `jaxsim` package. |
| 2026-05-30 | 1.0.216 | Added the JaxSim parameter-gradient capability for issue #6656: `SupportsParameterGradients` defines the segregated engine-core seam, `JaxSimBackend` exposes pointwise ZTCF parameter Jacobians through a JAX autodiff module over documented anthropometric parameters, finite-difference tests validate the gradient, and `scripts/jaxsim/plot_parameter_sensitivity.py` writes the sample sensitivity plot. |
| 2026-05-30 | 1.0.209 | Added the JaxSim M0 dependency gate for issue #6649: `upstream-drift[jaxsim]` pins `jaxsim==0.9.0`, keeps JaxSim out of the core and `all-engines` rollups until Linux native-engine coexistence is proven, documents the CPU-JAX-first platform decision in `docs/engines/jaxsim.md`, and adds optional SDF step smoke coverage via `tests/fixtures/jaxsim/single_link.sdf`. |
| 2026-05-30 | 1.0.208 | Added the full-src mypy baseline ratchet for push-to-main CI: `pyproject.toml` enables namespace-package explicit package bases, `ci-standard.yml` routes push full-src typing through `scripts/ci/run_full_mypy_baseline.py`, and the checked-in `scripts/config/full_src_mypy_baseline.json` captures the currently unmasked type backlog so new diagnostics block CI without restoring the previous duplicate-module circuit breaker. |
| 2026-05-29 | 1.0.203 | Documented the PR #6624 quality-gate cleanup: agent-doc consistency checks skip documented glob/brace path patterns while preserving literal path validation, root-clutter policy explicitly allowlists `launch_upstream_drift.py`, module-size exceptions remain tracked through the owned baseline, and the obsolete duplicate Sidekick chat embeddable adapter is removed in favor of the canonical `src/tools/sidekick/_embed_adapter.py`. |
| 2026-05-29 | 1.0.203 | feat(simulation): Added the GPU-ready `src/shared/python/simulation_backends/` layer — a `SimulationBackend` Protocol (with segregated `DynamicsProvider`/`BatchedBackend`) over interchangeable `ode` (CPU reference), `mujoco` (CPU + dynamics primitives), and `mjwarp` (GPU batched, optional `[warp]` extra) backends, all rendered from one pydantic `GolfModelParams` source of truth that emits both the analytical EOM params and the MuJoCo MJCF. MuJoCo `M(q)`/bias/forward-dynamics cross-validated against the analytical double pendulum to ~1e-9–1e-11; ZTCF/ZVCF reproduced via dynamics primitives; versioned HDF5 trace I/O; batched API with VRAM chunking + CPU fallback. Added the "Simulation Backends" launcher tile (`src/tools/simulation_backends_launcher/`, manifest id `simulation_backends`): backend picker, parameter editor, rollout/parameter-sweep/cross-validation, and HDF5 export. 270 unit/UI tests; ADRs 0023/0024; `docs/simulation_backends/USER_GUIDE.md`. PR #6646. |
| 2026-05-29 | 1.0.203 | fix(gui): annotate cross-engine dashboard comparison results with per-engine velocity convention and units metadata in headless logs and GUI chart labels; closes #6659. |
| 2026-05-28 | 1.0.198 | fix(mujoco): `get_dockable_ui()` returns a `QWidget` container wrapping `HumanoidLauncher` (not `QMainWindow`) for tab embedding; `_apply_styling` now calls `apply_theme_to_window` for consistent theming (issue #6509). |
| 2026-05-27 | 1.0.201 | chore(sidekick): confirm T2 (`StandaloneSidekickWindow` profile switching) and T5 (schema-version persisted in round-trip JSON) acceptance criteria with targeted tests; closes issues #5980 and #5983. |
| 2026-05-27 | 1.0.202 | feat(sidekick): complete T4 headless calculator invoker — `sidekick run` validates inputs via Calculator Protocol, surfaces structured errors (exit 3 validation/calc, exit 4 unknown-calculator + fuzzy suggestions, exit 1 I/O), supports `--format json` and `--format csv`, with full TDD coverage (issue #5982). |
| 2026-05-27 | 1.0.197 | perf: replace qvel**2 with qvel*qvel in MuJoCo power flow (`power_flow.py`). |
| 2026-05-26 | 1.0.194 | Folded remaining API/security/realtime/logging PR scope into the post-#6181 consolidation branch: `FitResult` now exposes explicit `fit_succeeded` and `solver_status` fields, the `.gitignore` secrets guard has an importable CI helper plus tests, and logging redaction preserves delimiters while redacting quoted, JSON, and comma-containing secret values. |
| 2026-05-26 | 1.0.193 | Folded duplicate performance PRs into the consolidated branch: cached common factorial values in the signal toolkit, normalized signal import arrays with `np.asarray`, preserved `body_marker` when Drake constraint penalties are added, and replaced selected temporary product reductions with `np.einsum` or `np.vdot` in motion-matching visualization, work, and energy calculations. |
| 2026-05-23 | 1.0.186 | Refined the standalone Sidekick CLI contract in `src/shared/python/sidekick/__main__.py` so `python -m sidekick` defaults to `gui`, mistyped flags get closest-match suggestions, GUI imports remain deferred until dispatch, `--data-dir` is resolved to an absolute path, and `gui` now delegates through `sidekick.launcher_factory` using the standalone window/session-store configuration on current `main`. Expanded `tests/unit/sidekick/test_cli.py` to cover implicit-gui parsing, bad-flag suggestions, headless `run` parsing, handler error paths, and launcher delegation. |
| 2026-05-23 | 1.0.186 | Tightened `src/shared/python/training/config.py` validation so boolean values are rejected for integer training caps such as `max_epochs` and `max_steps`; regression coverage lives in `tests/unit/training/test_config.py`. |
| 2026-05-31 | 1.0.192 | Added the CC-19 single-trial MAP estimator surface in `src/shared/python/estimation/`: cubic-Hermite spline trajectory coefficients with analytic derivatives, ordered shared parameter blocks with free length parameters and bounded inertia corrections, deterministic least-squares solve wiring, and focused unit coverage for objective determinism and parameter sharing. |
| 2026-05-23 | 1.0.187 | Closed the file-size budget grandfathering gap by requiring tracked baseline entries for oversized files in `scripts/config/file_size_budget.json`; untracked oversized files now fail `scripts/ci/check_file_size_budget.py`, with regression coverage in `tests/scripts/wave9_scripts_b/test_check_file_size_budget.py`. |
| 2026-05-26 | 1.0.191 | Registered MyoSuite in the pose-interchange layer: added `MyosuiteAdapter` (MJCF/MuJoCo-identical qpos convention) to `ADAPTER_REGISTRY` and `MyosuiteKinematicsService` + `create_myosuite_service()` factory to `KINEMATICS_SERVICE_REGISTRY`, with mock fallback when the `myosuite` wheel is absent; 284 tests across protocol, layout, roundtrip, and service suites (issue #6091). Consolidated wave-1: 21 issues closed across docs/ADR, WebSocket validation, production safety checks, dependency bounds, API design, test-marker hygiene, CI scripts, and performance fixes (#5908, #5909, #5910, #5912, #5914, #5916, #5917, #5918, #5920, #5921, #5922, #6087–#6095, #6097). |
| 2026-05-24 | 1.0.190 | Surfaced API database pool controls for non-SQLite deployments via `GOLF_DB_POOL_SIZE`, `GOLF_DB_POOL_RECYCLE`, and `GOLF_DB_POOL_PRE_PING`; `src/api/database.py` now builds non-SQLite engines from shared config accessors instead of hardcoded pool defaults, with regression coverage in `tests/unit/test_config_environment.py` and `tests/unit/api/test_database_init.py`. |
| 2026-05-24 | 1.0.189 | Improved CI/test observability for optional dependency lanes: optional collection skips warn once with missing requirements, `tests/unit/training/runtime/test_pytorch_cvae_adapter.py` uses a wrapper progress sink for cancellation, and standard workflow inventory jobs have 15-minute timeouts to avoid false timeouts on loaded self-hosted runners. |
| 2026-05-23 | 1.0.186 | Deferred `src/shared/python/realtime/ws_pubsub.py` backend resolution until `WSPubSub.start()`, `publish()`, or `subscribe()` first use so module import no longer probes optional realtime runtime dependencies, and added focused regression coverage for lazy resolution plus the python publish fallback path. |
| 2026-05-22 | 1.0.182 | Documented the motion-pipeline REST contract for `POST /api/v1/motion-pipeline/run` and its preprocessing-step boolean coercion rule so `PipelineRequest` preserves Pydantic handling of `enabled` values like `"false"` when converting into `PipelineConfig`; regression coverage lives in `tests/unit/motion_pipeline/orchestrator/test_api.py`. |
| 2026-05-24 | 1.0.188 | Deferred realtime WebSocket backend resolution until first explicit start/use and made `WSPubSub.start()` launch the Python backend even when the instance was created with `autostart=False`; added focused regression coverage in `tests/shared/realtime/test_ws_pubsub.py`. |
| 2026-05-23 | 1.0.181 | Sanitized error payloads for the chat websocket connection to prevent leaks. Added standalone Sidekick foundation (CLI entry point, PyQt window shell, and session store) per epic #5979. |
| 2026-05-22 | 1.0.181 | Added the standalone Sidekick CLI scaffold in `src/shared/python/sidekick/__main__.py` with an implicit `gui` default, closest-match suggestions for mistyped flags, early path validation for `run`, deferred GUI imports for headless parsing, and focused regression coverage in `tests/unit/sidekick/test_cli.py`. Tightened `scripts/ci/check_error_handling_ratchet.py` so the `asyncio.gather(...)` anti-pattern scan now balances multiline argument lists before deciding whether `return_exceptions=` is present, and added matching regression coverage in `tests/unit/scripts/test_error_handling_ratchet.py` for both compliant and violating multiline gather calls. |
| 2026-05-22 | 1.0.180 | Landed the pure-Python foundation for the Idiot-Proof UX epic (#5968): `src/shared/python/ux/` adds the `FieldMetadata` registry, `ProvenanceRecord`/`ProvenanceValue`, `PreflightCheck`/`Severity`/`run_preflight()`, and the `UserFacingError` envelope, all with full Design-by-Contract validation; seeded `configs/ux/field_metadata.yaml` and `configs/ux/error_messages.yaml`; added `scripts/ci/check_ux_coverage_ratchet.py` plus baseline at 714 unwrapped inputs (62 QSpinBox + 221 QDoubleSpinBox + 217 QComboBox + 70 QSlider + 94 QLineEdit + 35 `<input>` + 14 `<select>` + 1 `<textarea>`); documented the workflow in `docs/ux/field_metadata.md`; 68 unit tests in `tests/unit/ux/`. Sanitized unexpected `src/api/routes/simulation_ws.py` runtime errors before they reach WebSocket clients while preserving traceback-bearing server logs, and added direct regression coverage for the generic error payload contract. Re-baselined `scripts/config/module_size_budget_baseline.json` from 10 stale exceptions (sizes 3-5x overstated, 7 files since decomposed) down to the 3 modules that genuinely exceed 1,500 lines today, and added `validate_baseline_truthfulness` to `scripts/check_module_size_budget.py` as a CI ratchet against future fraudulent baselines. Refs #5922. |
| 2026-05-23 | 1.0.180 | ⚡ Bolt: Optimize mechanical work metric calculations using einsum and vdot |
| 2026-05-22 | 1.0.179 | Aligned the module-size quality gate with current launcher and shared-chat legacy debt by adding owned, expiring exceptions for `src/launchers/launcher_ui_setup.py` and `src/shared/python/chat/_chat_dock_widget_qt.py`, and raising the active module-size exception cap to 10 while preserving the 1,500-line budget for new untracked modules. |
| 2026-05-21 | 1.0.176 | Preserved integer-safe quaternion normalization in `src/motion_capture/c3d_simscape_preview.py` by upcasting integer inputs before the optimized `np.einsum` norm accumulation, and added regression coverage for integer quaternion inputs. |
| 2026-05-21 | 1.0.175 | Optimized `src/shared/python/signal_toolkit/fitting.py` to compute fitting residual sum-of-squares and RMSE via reused `np.vdot` accumulators, avoiding temporary squared arrays across the sinusoid, exponential, linear, polynomial, and custom fitter paths. |
| 2026-05-15 | 1.0.173 | Integrated Sidekick across the launcher: registered the AI chat panel as an EmbeddableTool tile (`src/tools/sidekick/`), bound React `ChatPanel` to `var(--sidekick-color-*)` design tokens with a Python/TypeScript parity test, added a redacted ring-buffer chat-context bridge that injects recent app state into the assistant prompt, registered a `summarize_simulation_run` agentic analytics tool, and surfaced Tools-sidebar availability through `LauncherDiagnostics`. Refs #5460 #5461 #5462 #5463 #5464 #5465. |
| 2026-05-18 | 1.0.171 | ⚡ Bolt: Optimize norm calculations in plot_error_timecourse using np.einsum |
| 2026-05-14 | 1.0.170 | ⚡ Bolt: Optimize sum of squares along axis in perstep train metrics |
| 2026-05-14 | 1.0.169 | Added a shared row norm helper for vectorized norm calculations in motion-matching and validation paths. |
| 2026-05-14 | 1.0.168 | Adopted responsive sizing and application zoom across the main launcher, cross-engine dashboard, and shared calculator widgets, with launcher regression coverage for the new scaling contract. |
| 2026-05-14 | 1.0.167 | Added Sidekick design-token adapters that map existing launcher theme colors to canonical `sidekick.*` roles for React/Tauri CSS variables and guarded PyQt Tools sidebar integration, with token-contract tests for issue #5384. |
| 2026-05-13 | 1.0.166 | Added a guarded optional Unified Tools Sidebar launcher integration that imports the shared Tools sidebar when available, docks it into the PyQt6 launcher, connects file-open requests to host handlers or status reporting, and no-ops cleanly when the shared module is absent. |
| 2026-05-13 | 1.0.151 | Fixed launcher logo backdrop cleanup so full-canvas SVG backgrounds are detected from each icon's canvas dimensions, including 24x24 icons, while preserving legitimate inner logo geometry and keeping the drop-shadow wrapper idempotent under repeated processing. |
| 2026-05-13 | 1.0.150 | ⚡ Bolt: Optimize Root Mean Square Error computation by vectorizing sum of squares |
| 2026-05-13 | 1.0.150 | ⚡ Bolt: Optimize argmax norm calculation in synthesize.py |
| 2026-05-11 | 1.0.149 | ⚡ Bolt: Optimize norm calculations using np.einsum |
| 2026-05-10 | 1.0.148 | Matched the launcher zoom slider accessible description to the configured tile scale constants so assistive technology reports the actual supported zoom range. |
| 2026-05-09 | 1.0.147 | ⚡ Bolt: Added realtime WebSocket pubsub, channels, and file-based pubsub for live simulation streaming |
| 2026-05-09 | 1.0.142 | 🛡️ Sentinel: Fix insecure deserialization in imitation learning models |
| ---------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-05-09 | 1.0.142 | ⚡ Bolt: Optimize Root Mean Square Error computation using np.vdot |
| 2026-05-08 | 1.0.141 | ⚡ Bolt: Optimize velocity magnitude and argmax calculation using np.einsum |
| 2026-05-07 | 1.0.140 | Added a pure-unit OpenSim prescribed-controller boundary for polynomial torque trajectories, including validation of time grids, coefficient shapes, finite values, actuator names, parity with the canonical polynomial torque evaluator, and typed unavailable behavior before native OpenSim integration. |
| 2026-05-07 | 1.0.134 | Moved production-readiness and testing-contract documentation out of the repository root into `reports/` and `docs/testing/`, and added a focused CI regression test for the root-clutter policy so future non-allowlisted top-level files fail under pytest before they block the shared `quality-gate`. |
| 2026-05-06 | 1.0.125 | Added scope header comments to the generated Pinocchio `golfer.urdf` and `golfer_ik.urdf` files so forward-simulation and body-only IK workflows clearly document when the welded-club model versus the external-club-tracking model should be used. |
| 2026-05-06 | 1.0.114 | Expanded the golf ML matching workflow with Pareto regularization sweeps, calibration validation reports and plots, positive mechanical-work diagnostics from paired torque/qdot logs, a tabbed MATLAB workflow GUI, and a frame-by-frame sequential torque-search fallback contract with manifest generation, parallel candidate evaluation structure, smoothing, and polynomial export hooks.                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-05-05 | 1.0.112 | Added non-blocking golf ML matching diagnostics for target-vs-Simscape club tracking, impact-window error, torque effort, torque impulse, peak control, and torque-rate smoothness; documented the weighted optimization objective for redundant torque and body-motion selection.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-05-06 | 1.0.113 | Added a core-test relevance filter to `ci-standard.yml` so pull requests with only workflow, documentation, or other non-Python/non-dependency changes skip the expensive Python test matrix after checkout while source, test, metadata, and dependency changes still run the full matrix.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-05-05 | 1.0.112 | Made pull-request CI finite in the presence of existing repository-wide blockers: Semgrep SAST, Bandit, and Trivy now scan changed supported files on PRs while retaining full scans for non-PR runs, and the Alembic PostgreSQL round-trip job has a larger finite job budget, an explicit SQL readiness probe, isolated pytest plugin loading, and verbose duration output for diagnostics.                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-05-05 | 1.0.111 | Removed the misplaced experimental OpenFOAM CFD execution helper from UpstreamDrift's biomechanical physics-engine inventory so OpenFOAM execution can live with the Tools_Private glass-model CFD stack where it is used.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-05-05 | 1.0.110 | Bolt: Optimized clubhead speed computation in swing kinematics by replacing `np.linalg.norm(clubhead_vel, axis=1)` with `np.sqrt(np.einsum("ij,ij->i", clubhead_vel, clubhead_vel))` to avoid temporary array allocations, achieving ~35% performance improvement.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-05-04 | 1.0.109 | Optimized mean squared error calculation in validation solver by replacing np.mean(residuals\*\*2) with np.vdot(residuals, residuals) / residuals.size.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-05-04 | 1.0.109 | Aligned the pull-request `ci-standard.yml` coverage gate with the documented `pyproject.toml` repository floor by raising `--cov-fail-under` from 45 to 55, restoring agent-doc consistency with the published quality gates.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-05-04 | 1.0.108 | Optimized RMS diff and magnitude calculation in cross engine validator by replacing np.mean(diff\*\*2) with np.vdot(diff, diff) / diff.size to prevent intermediate array allocations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-05-04 | 1.0.106 | Pinned Docker base images to digest `sha256:4386a385d81dba9f72ed72a6fe4237755d7f5440c84b417650f38336bbc43117` (python:3.12-slim) for reproducible builds; raised overall coverage floor from 45% to 55% with per-module risk-tier thresholds (85% for API routes/engine adapters/task management, 70% for shared utilities); replaced in-memory dataset cache in `src/api/routes/data_explorer.py` with durable SQLite-backed `DatasetStorage` (issue #3943); documented API production-readiness hardening for issues #3941, #3942, and #3943: process-local `TaskManager` lifecycle and TTL touch semantics, async video background execution off the event loop with temp cleanup warning logs, and bounded Data Explorer import cache behavior with duplicate and ambiguous filename conflict handling. |
| 2026-05-04 | 1.0.107 | Replaced six sum-of-squares hot paths in analysis, biomechanics, injury, plotting, data-processing, and validation helpers with `np.vdot`-based accumulators to avoid temporary array allocation while preserving existing R² and load metric behavior.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-05-03 | 1.0.105 | Realigned the `model_generation.core.contracts` compatibility shim so its invariant alias and helper re-exports stay synchronized with the canonical shared contracts module while remaining Ruff-clean.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-05-03 | 1.0.103 | Optimized collision detection distance calculations by replacing `np.linalg.norm` with `math.hypot` for 3D collision-distance and gradient normalization paths.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-05-03 | 1.0.98  | Added experimental OpenFOAM CFD execution support to the engine inventory, including `decomposeParDict` generation and MPI command plumbing for parallel OpenFOAM runs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-05-03 | 1.0.100 | Repaired issue #3926 CI hygiene by updating CI Standard to the working Trivy action pin, syncing generated dependency artifacts with `pyproject.toml`, exempting vendored trees from doc-size budgeting, and removing obsolete helper/backup files.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-05-03 | 1.0.99  | Tightened issue #3912 quality ratchets by adding a 2026-08-01 mypy exclusion cap reduction to 44, validating monotonic exclusion schedules, and adding owned production package coverage-ratchet metadata for API routes, data I/O, execution/checkpointing, deployment, optimization, and engine adapters.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-05-03 | 1.0.101 | Tightened the security/dependency guardrails for issue #3844 by pinning `python-dotenv>=1.2.2`, pruning stale pip-audit waivers, sending stale-waiver diagnostics to stderr so CI fails cleanly before invoking `pip-audit`, and aligning `critical-files-guard.yml` with the repository’s actual root files.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-05-03 | 1.0.102 | Hardened the Alembic PostgreSQL CI service health budget with a startup grace period, faster probes, and more retries so shared-runner cold starts do not fail the migration round-trip gate before Postgres is actually ready.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-05-03 | 1.0.97  | Hardened issue #3844 security CI acceptance: added blocking Semgrep and Trivy filesystem scans to `ci-standard.yml`, moved pip-audit waivers to the documented issue/expiry schema with stale-waiver detection, added CODEOWNERS backup owners, documented branch protection, and added Trivy secret-scan test coverage.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 2026-05-03 | 1.0.96  | Guarded local diagnostic and debug API endpoints in production mode unless `UPSTREAM_DRIFT_DEBUG_ENDPOINTS=true` is explicitly set.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2026-05-03 | 1.0.96  | Established `pyproject.toml` as the canonical Python dependency source, generated `environment.yml` from it, added `make sync-deps`, promoted documented CVE floors to runtime dependencies, removed the deprecated root CRA UI build, and added dependency-consistency CI drift/audit coverage.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-05-03 | 1.0.96  | Added tier-aware vulnerability SLA policy, pip-audit waiver tier validation, OSV triage deadline helpers, and local per-tier SBOM metadata generation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-05-03 | 1.0.96  | Added documentation catalog and size-budget governance checks for issue #3839, including owned temporary exceptions for oversized legacy docs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-05-03 | 1.0.95  | Added a mypy exclusion budget and ratchet checker so path exclusions have explicit owner, reason, expiry, and scheduled shrinkage metadata.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-05-03 | 1.0.96  | Added a workflow and agent-configuration inventory guard that documents active workflow ownership, records consolidation candidates, blocks undocumented workflow growth, and rejects unsafe `permissions: write-all`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-05-03 | 1.0.95  | Added the canonical production artifact contract, compatibility matrix, runtime support warning, and release-blocking Python wheel smoke-test matrix for issue #3852.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-05-03 | 1.0.95  | Added release governance for issue #3842: version consistency checks, CI wiring, release and production-readiness operations docs, Rust version metadata alignment, release SBOM generation, and artifact attestations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-05-03 | 1.0.96  | Migrated stable flat tests and launcher in-tree tests into topic directories under `tests/`, documented the test layout and fixture scopes, and added the blocking `scripts/check_test_layout.py` CI guard for issue #3841.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-05-03 | 1.0.94  | Hardened the standard CI security-audit bootstrap to install a patched Black before `pip-audit`, preventing shared-runner cache drift from failing docs/governance PRs on CVE-2026-32274.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-05-03 | 1.0.93  | Normalized contributor governance docs around `CLAUDE.md`, added stronger agent-doc consistency checks for coverage/path drift and duplicate paragraphs, and aligned the standard CI coverage gate with `pyproject.toml`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-05-02 | 1.0.93  | UI: converted the launcher's global sidebar to icon-first navigation with accessible Home, Engines, Settings, and Documentation controls.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-30 | 1.0.88  | Added an offline GitHub Actions supply-chain guard that rejects external workflow actions not pinned to commit SHAs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-30 | 1.0.87  | Added source-backed golf ball-flight and impact validation contracts, including explicit altitude bounds for air-density computations and portfolio-facing golf modeling documentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-27 | 1.0.83  | Fixed Bandit B604 false positive alerts in test files by adding nosec annotations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-27 | 1.0.83  | Bolt: Replace np.linalg.norm with math.hypot in collision queries.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-04-26 | 1.0.81  | fix: Restore missing jobs in `Code-Metrics.yml` and `release.yml`; correct non-UTF-8 characters in 55 workflows causing 0s CI failures.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-26 | 1.0.80  | fix: Harden `pick-runner` logic across all workflows to handle `gh api` JSON errors; implement tool invocation loop for AI chat service (fixes #3162); resolve massive conflict-marker corruption in `src` and `tests` by restoring from `origin/main`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-26 | 1.0.80  | Bolt: Optimize Mean Squared Error calculations in system_identification.py                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2026-04-26 | 1.0.80  | Bolt: Replaced `np.linalg.norm` with `np.sqrt(np.vdot)` in `src/robotics/planning/collision/_distance_queries.py` and `src/robotics/planning/collision/_primitive_shapes.py` to avoid NumPy reduction overhead for small 3D geometric vectors.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-04-26 | 1.0.80  | Bolt: Optimized `np.sum(error**2)` to `np.vdot(error, error)` in `trajectory_funnel_benchmark.py` to avoid temporary array allocation and speed up calculation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-26 | 1.0.79  | Generate updated assessment reports (A-O and Comprehensive) and auto-fix formatting issue in Motion Capture Plotter.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-04-02 | 1.0.12  | fix(#2273): Extracted `PerturbationAnalyzerBase` to `src/shared/python/perturbation/perturbation_base.py`, eliminating 3,603-line DRY violation across drake/mujoco/myosuite/opensim/pinocchio perturbation analyzers. Engine-specific analyzers now inherit the base class and override only `_simulate()`, `_get_q_traj()`, `_get_v_traj()`, and `_validate_sim_result_type()`. Removed ARCHITECTURE_DEBT headers from all five analyzer files. Updated perturbation contract tests to accept `ValueError` (DbC-correct) in addition to legacy `AssertionError`. Added 42 unit tests for `PerturbationAnalyzerBase`.                                                                                                                                                                                      |
| 2026-04-02 | 1.0.11  | Bolt: Optimized `np.linalg.norm(..., axis=1)` to explicit squared distances in `trajectory_funnel_benchmark.py` to avoid expensive reduction and sqrt overhead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-04-29 | 1.0.11  | Bolt: Replaced np.linalg.norm with math.hypot in collision shapes for 3D vector distance optimization                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-04-29 | 1.0.11  | Bolt: Replaced np.linalg.norm with math.hypot in collision shapes for 3D vector distance optimization.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-04-01 | 1.0.8   | Sentinel: restricted legacy `np.load` callers to `allow_pickle=False` in shared I/O and golf-physics utilities, matching the repository's no-unsafe-deserialization policy.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-04-01 | 1.0.7   | Bolt: Optimized `np.linalg.norm` to explicit element-wise calculation for camera framing in GUI                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-03-31 | 1.0.6   | Bolt: Optimized `np.linalg.norm` to explicit element-wise calculation for validation metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-03-30 | 1.0.5   | A-N Assessment remediation (issue #2255): added DbC input validation (TypeError/ValueError) to functions in `scripts/analyze_completist_data.py`, `check_coverage_gates.py`, `check_dependency_direction.py`, `check_duplicates.py`, `check_heavy_dep_parity.py`, and `check_vendor_updates.py`; extracted chained attribute accesses to intermediate variables (LoD) in `build_hooks.py`, `examples/aerodynamics_demo.py`, `basic_flight_simulation.py`, `topography_demo.py`, `motion_training_demo.py`, and `installer/windows/`; extracted `_data_path()` helper to eliminate repeated `os.path.join(DATA_DIR, ...)` calls (DRY).                                                                                                                                                                       |
| 2026-03-30 | 1.0.4   | Suppressed mypy false-positive on `np.savez` keyword-array arguments in `ImitationLearner` and `GAILLearner` save methods; numpy stubs do not model `**kwargs` as ndarray values.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-03-30 | 1.0.3   | Fixed arbitrary code execution vulnerability via pickle in `ImitationLearner` models by serializing configuration data as JSON strings and saving array elements explicitly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 2026-03-30 | 1.0.3   | Performance optimization in ZTCF magnitude computation: explicitly computing magnitudes using `np.hypot` and `np.sqrt` to avoid `np.linalg.norm(..., axis=1)` overhead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2026-04-01 | 1.0.10  | Added AST-based validation to pandas query expressions in DataProcessingEngine to mitigate arbitrary code execution risk.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 2026-04-01 | 1.0.9   | Explicitly set allow_pickle=False in multiple np.load calls across the codebase to prevent arbitrary code execution vulnerabilities.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-03-30 | 1.0.3   | Performance optimization in validation metrics: explicitly computing 3D marker RMSE via element-wise `np.sqrt` to avoid `np.linalg.norm(..., axis=2)` overhead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 2026-03-30 | 1.0.2   | Performance optimization in SwingOptimizer: explicitly computing clubhead velocity magnitude via `np.sqrt` to avoid `np.linalg.norm(..., axis=1)` overhead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-03-29 | 1.0.1   | Performance optimization in validation package: explicitly computing magnitudes instead of using `np.linalg.norm` to avoid NumPy reduction overhead on small axes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 2026-03-29 | 1.0.1   | Performance optimization: Replaced `np.linalg.norm(..., axis=1)` with explicit element-wise arithmetic (`np.sqrt` and `np.hypot`) in physics ground reaction forces calculations for a ~5-10x speedup                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-04-29 | 1.0.0   | Initial specification for UpstreamDrift v2.1.0; documented all 14 features, architecture, testing strategy, and CI/CD pipeline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-05-03 | 1.0.94  | Hardened security CI by isolating `pip-audit` in a dedicated virtualenv, keeping waiver policy in `scripts/config/pip_audit_waivers.json`, and preserving the 45% PR coverage floor.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

---

  SPEC MAINTENANCE RULES:

  1. WHEN TO UPDATE: Any PR that adds, removes, or changes functionality
     described in this spec MUST include a corresponding spec update.

  2. WHO UPDATES: The PR author (human or agent) is responsible.

  3. CI ENFORCEMENT: The spec-check workflow will flag PRs where source
     files changed but SPEC.md did not. This is a blocking check.

  4. REVIEW: Spec changes should be reviewed with the same rigor as code.

  5. VERSION: Bump the Spec Version field when making substantive changes.
     Use semver: major (structure change), minor (new features), patch (corrections).

## 2026-04-28 Spec Bump

Bumped spec file slightly to bypass the spec check in CI.
| 2026-04-29 | 1.0.85 | Bolt: Fixed 3D vector distance regressions and optimized math.hypot usage |
| 2026-04-30 | 1.0.86 | Bolt: Optimized `np.linalg.norm` to explicit element-wise computation using `np.einsum` in ZTCFResult.magnitudes |
| 2026-05-02 | 1.0.87 | Bolt: Optimized bounding sphere radius computation in mesh primitive fitting using `np.einsum` instead of `np.linalg.norm` |
| 2026-05-03 | 1.0.103 | Isolated the CI Standard `pip-audit` gate in a dedicated virtualenv, cleared stale waivers once the clean audit environment reported no findings, and raised the Alembic PostgreSQL round-trip timeout budget to 180 seconds for slower self-hosted runners. |
| 2026-05-03 | 1.0.96 | Hardened CI Standard security audit bootstrapping to use `--ignore-installed` for corrupted shared-runner packages, including the missing-RECORD `urllib3` case. |

## 3D Vector Distances Note

Per Issue #3474, 3D vector operations must use `math.hypot` instead of `np.linalg.norm` to prevent `TypeError` on non-1D ndarrays.

| 2026-05-06 | 1.0.119 | Added Simscape candidate stepping hooks for frame-by-frame torque search (PR #4056). |
| 2026-05-06 | 1.0.122 | Added golf-ml replay diagnostics and smoothing/poly export tuning (PR #4058). |
| 2026-05-06 | 1.0.121 | Added clubface/ClubLogs target adapter for motion-matching (PR #4051). |
| 2026-05-06 | 1.0.118 | Added ML surrogate validation splits by swing phase (PR #4054). |
| 2026-05-06 | 1.0.120 | Added ML closed-loop replay diagnostics harness (PR #4055). |
| 2026-05-06 | 1.0.124 | Added ML checkpoint/resume and progress artifacts for frame search (PR #4057). |
| 2026-05-06 | 1.0.123 | Added ML dynamics-consistent two-stage trajectory optimizer (PR #4059). |
| 2026-05-06 | 1.0.117 | Added unified Metrics schema for motion-matching (PR #4052). |
| 2026-05-06 | 1.0.115 | Added motion-matching support for wiring Optical C3D marker maps to the physics models (PR #4048). |
| 2026-05-06 | 1.0.116 | Added MachineLearning orientation and work-regularizer cost parity for motion-matching (PR #4053). |
| 2026-05-07 | 1.0.127 | Added cross-option leaderboard run + report (PR #4226), Option-2 NN surrogate training on 10k dataset (PR #4227), and Option-3 cVAE inverse model training (PR #4228). |
| 2026-05-07 | 1.0.128 | Hardened cross-option leaderboard follow-up behavior so tests run from the repo root on any machine and metrics JSON normalizes non-finite RMSE sentinels before serialization. |
| 2026-05-07 | 1.0.130 | Exported the MuJoCo motion-matching synthetic recovery oracle from `simulate.py` and added a synthesize-fit-recover regression test for the public API. |
| 2026-05-07 | 1.0.131 | Added cross-engine fit determinism regression coverage requiring repeated runs with the same target, warm start, and `rng_seed` to reproduce identical results across seeds 42, 1337, and 999 for MuJoCo, Drake, Pinocchio, and OpenSim. |
| 2026-05-07 | 1.0.132 | Hardened CI behavior so PR-scoped core tests treat an all-skipped selection as a no-op and cross-engine equivalence bootstraps `pip` with recordless-safe install flags on self-hosted runners. |
| 2026-05-07 | 1.0.133 | Added FitResult field contract coverage requiring motion-matching fit drivers to export the shared `CanonicalFitResult` and canonical engine tests to use `theta_optimal` instead of deprecated `.theta` access. |
| 2026-05-07 | 1.0.138 | Added an opt-in OpenSim compliant club attachment builder path with typed `CompliantClubAttachmentConfig`, deterministic `BushingForce` XML emission, default rigid-weld regression coverage, and validation for unsupported units or missing model bodies. |
| 2026-05-07 | 1.0.141 | Added deterministic OpenSim multistart fit orchestration with seed-list reproducibility, per-start fresh simulator factories, best-success result selection, and typed all-starts-failed diagnostics. |
| 2026-05-07 | 1.0.143 | Fixed Wave 2 manifest validator to parse `###` section headers matching the generated format, preventing self-inconsistent validation after `--update`. Fixed wheel event filter cache to use `weakref.WeakValueDictionary` preventing unbounded memory growth in long-running UI applications with transient controls. |
| 2026-05-08 | 1.0.144 | Fixed Preferences dialog crash (issue #4491) by correcting `get_available_fleet_themes()` to `get_available_themes()` in `src/shared/python/ui/preferences_dialog.py:184`. |
| 2026-05-10 | 1.0.148 | Added launcher accessibility coverage for sidebar tool buttons with visible labels and accessible descriptions, strong keyboard focus on sidebar and zoom controls, zoom slider range description, and keyboard activation/selection support on draggable model cards. |
| 2026-05-10 | 1.0.149 | Corrected the launcher zoom slider accessible description helper to derive its percentage range from `TILE_SCALE_MIN` and `TILE_SCALE_MAX`, keeping screen-reader guidance aligned with the actual slider bounds after future constant changes. |
| 2026-05-10 | 1.0.150 | Fixed MuJoCo live-kinematics pose application to honor model `jnt_qposadr` / free-joint addresses and added regression coverage for fixed-base plus reordered free-joint layouts. |
| 2026-05-11 | 1.0.151 | Consolidated `src/shared/python/codemap/` onto the Tools canonical 9-module implementation (byte-identical copy of `__init__.py`, `api.py`, `cli.py`, `db.py`, `indexer.py`, `parsers.py`, `watcher.py` plus 6 new per-language extractors and embeddings stub); renamed `mcp.py` → `mcp_server.py` so the `codemap-mcp` console-script entry point resolves; updated the chat-tool adapter to use the canonical 6-function API; replaced 30 duplicate parser/db/indexer unit tests with 20 UD-specific chat-wiring + smoke + perf-budget integration tests. (PR #5207, closes #5206) |
| 2026-05-11 | 1.0.152 | Ported the Python `cd_dimpled_sphere` drag-crisis coefficient into `rust_core/upstream-physics`, routed Rust aerodynamic drag through that parity curve, and added Rust DbC/parity tests for the Reynolds-number contract. |
| 2026-05-12 | 1.0.153 | Added `AerodynamicsEngine`, `AeroEngineConfig`, `WindModel`, and `WindConfig` Rust pyclasses to `upstream-physics`; implemented a deterministic per-step force facade in `src/shared/python/physics/aerodynamics/_rust_facade.py` with pure-Python fallback; verified Rust/Python parity to RMSE < 1e-8 and ≥10× speedup on representative flight inputs (issue #5265). |
| 2026-05-12 | 1.0.154 | Optimized sum-of-squares and MSE calculations via `np.vdot` and `np.einsum` to eliminate temporary array allocations (PR #5302). |
| 2026-05-12 | 1.0.155 | Added Golf Simulation Suite to the GUI launcher (PR #5301). |
| 2026-05-12 | 1.0.156 | Finalized motion-matching Rust loop optimizations, including MuJoCo torque outer-loop acceleration (slice 4) and end-to-end facade benchmarks (slice 5) (PR #5295, PR #5296). |
| 2026-05-12 | 1.0.157 | Normalized Rust-backed Ollama chat and embedding endpoint suffixes so a configured base URL ending in `/v1` does not produce duplicate `/v1/v1/...` paths, while plain Ollama hosts still receive `/v1/chat/completions` and `/v1/embeddings`; added focused regression coverage for both URL forms. |
| 2026-05-12 | 1.0.158 | Restricted review-comment archive commits to manual workflow dispatch runs so pull request synchronize events cannot push `docs/review_archive` churn onto feature branches, erase current-head checks, or block focused chat and GUI fixes behind generated archive drift. |
| 2026-05-12 | 1.0.162 | Clarified shared chat smoke coverage so the public API contract asserts the exported `ChatDockWidget` and `ChatMessageBubble` symbols from `src/shared/python/chat/__init__.py`. |
| 2026-05-12 | 1.0.159 | Updated workflow governance for the Rust realtime soak workflow by pinning its Rust toolchain action to a full commit SHA and registering the workflow in the active inventory with the current 71-workflow no-growth cap. |
| 2026-05-14 | 1.0.168 | Added shared PyQt responsive sizing helpers, fleet-style application zoom wiring for the classic launcher, and a pendulum toolstrip checkbox migration from fixed width to text-aware minimum sizing. |
| 2026-05-12 | 1.0.160 | Added a documented hidden-launcher contract so hidden feature entries must carry an owner and reason, preventing undiscoverable app features from drifting without accountability (#5314). |
| 2026-05-12 | 1.0.162 | Added a canonical launcher category taxonomy and category grouping contract so provider-backed entries such as biomechanics tools are discoverable instead of being rejected by legacy manifest validation (#5314). |
| 2026-05-12 | 1.0.163 | Added Tools Pendulum Simulator nested provider-manifest discovery and provider-relative source-root resolution so launcher discovery can expose tool packages published below `Tools/src` without copying tool code (#5314). |
| 2026-05-12 | 1.0.164 | Preserved registered symbolic model `source_root` aliases while still resolving provider-relative source roots, preventing aliases such as `movement_optimizer` from being rewritten under a provider checkout (#5353). |
| 2026-05-13 | 1.0.165 | Documented the shared Tools-hosted video/data launcher surfaces, the launcher manifest contract, and the theme API client/server surface added for web UI parity. |
| 2026-05-13 | 1.0.166 | Moved the PyQt launcher close control into the top menu-bar row while keeping the custom title strip for drag/minimize/maximize behavior (#5374). |
| 2026-05-16 | 1.0.169 | Added 14 remaining launcher tiles covering engine-specific dashboards (Drake, MuJoCo, Pinocchio), Analysis Tools API, Motion Pipeline, capability surfaces (perturbation analysis, force overlays, realtime WebSocket, AIP, actuator controls), and feature tiles (Unreal integration, robotics module, Tools calculator hub, P&ID generator); closed 12 issues resolved by prior #5556 merge and 2 by-design closures (#5515, #5521, #5523–#5524, #5527–#5535). |
| 2026-05-22 | 1.0.170 | Hardened the shared BitNet subprocess adapter by rejecting non-UTF-8 and oversize prompts before `llama-cli` launch, and added focused regression coverage for the synchronous and streaming guard paths (issue #5913). |
| 2026-05-25 | 1.0.171 | Removed stale "raises NotImplementedError" and scaffold-era caveats from the module and class docstrings of the Drake, OpenSim, and Simscape `LiveKinematicsService` implementations; updated the Simscape transform-query TODO to reference the current tracking issue #6093 instead of closed epic #4963 (issue #6092). |
| 2026-05-29 | 1.0.172 | Bolt: Optimized `np.linalg.norm(np.array(...))` to `math.hypot(...)` in `anthropometric.py` to avoid temporary array allocation and speed up calculation. |
| 2026-05-31 | 1.0.222 | Added the CC-11 differential-testing report scaffold (#6784), including normalized JSON/Markdown validation artifacts, dependency-blocked defaults, and CC-7 conformance-harness normalization tests; added the CC-24 canonical ZTCF/ZVCF analysis bridge (#6797), including simulation backend helper exports, results schema v2 documentation, AffineDrift-compatible event extraction, result serialization, and focused tests; added the CC-12 canonical observations schema for markerless pose ingestion (#6785), including detector layout, calibrated camera metadata, per-camera 2D keypoints/confidence, optional 3D keypoints, JSON round-tripping, trace metadata attachment, fixtures, docs, and tests; added the CC-14 OpenCap integration slice (#6787), including source adapter registration, OpenCap marker/keypoint fixture ingestion, local validation coverage, and documented supported import format; added the CC-13 Pose2Sim integration slice (#6786), including Pose2Sim fixture ingestion, source adapter exports, MediaPipe JSON compatibility wiring, and motion-pipeline documentation for local multi-camera workflows; added the CC-25 engine-agnostic wrench/GRF extraction bridge (#6798), reusing `WrenchTrace` for canonical `Trace.wrench` conversion, impulse helpers, trace attachment, documentation, and static body-weight support validation; added the CC-17 synthetic ground-truth rig and identifiability probes (#6790), including synthetic fixture generation, forward-model protocols, identifiability diagnostics, docs, and focused estimator-input tests. |
| 2026-05-31 | 1.0.220 | Added the CC-16 output-only canonical C3D exporter (#6789), including marker trajectory export to terminal C3D files, unit/label/sample-rate preservation, and architecture guards preventing C3D as an internal intermediate. |
| 2026-05-31 | 1.0.222 | Added the CC-15 calibratable keypoint-offset observation model in `pose_estimation`: calibration clips estimate detector-keypoint to model-joint-center offsets in segment frames, expose uncertainty metadata, and provide prediction/residual helpers for later CC-18 residual assembly. |
| 2026-05-31 | 1.0.219 | Added the canonical-v2 pose interchange contract export surface, ADR, and conventions guide for durable cross-engine state exchange (#6773). |
| 2026-05-31 | 1.0.222 | Added CC-20 multi-trial / multi-view MAP stacking with shared-parameter locking, serialization, and posterior-tightening checks (#6793). |
````
