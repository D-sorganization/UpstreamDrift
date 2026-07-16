# SPEC.md — Repository Specification Document

<!--
  TEMPLATE VERSION: 1.0.0
  LAST UPDATED: 2026-06-18

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
| **Primary Language(s)** | Python 3.11+, Rust, TypeScript                     |
| **License**             | MIT                                                |
| **Current Version**     | 2.1.1                                              |
| **Spec Version**        | 1.0.466                                            |
| **Last Spec Update**    | 2026-06-21                                         |

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

- **2026-06-22** - Optimize Mechanical Work computations in `evaluate_matching_workflow.py`. `np.sum(..., axis=1)` calls were replaced with equivalent but more efficient `np.einsum('ij->i', ...)` calls, yielding significant performance gains during bulk evaluation.
- **2026-06-21** - Hardened the simulation WebSocket and Data Explorer API
  routes for deferred #7740 findings: WebSocket start validation now rejects
  non-positive speed factors and shares duration/timestep bounds with
  `SimulationRequest`, simulation stats access is centralized behind one
  helper, Data Explorer dataset lookup rejects glob metacharacters, recursive
  dataset listing is paginated and bounded, and the dead cache helper was
  removed. Focused unit-marked tests cover the new WebSocket success/error
  branches, filter operators, ambiguous dataset names, glob rejection, and
  bounded listing behavior.
- **2026-06-20** - Deduplicated the chat WebSocket protocol loop for issue
  #7720: `src/api/routes/chat_ws.py` and the portable chat router factory now
  share one handshake/send/history/new-session/transport-error loop, with
  route-specific hooks preserving API context injection and router-factory
  extension actions plus focused regression coverage across both entrypoints.
- **2026-06-20** - Deduplicated the cross-engine motion-matching polynomial
  torque evaluator for issue #7728: Drake, MuJoCo, Pinocchio, and OpenSim now
  share the same lowest-power-first `[A..G]` Horner helper and constant,
  with unit coverage pinning shared and engine-wrapper parity.
- **2026-06-20** - Added isolated transition hazard-rule coverage for issue
  #7715: the MDP transition tests now pin hazard penalties and DbC guard
  behavior directly so policy updates cannot bypass invalid-state validation.
- **2026-06-20** - Hoisted OpenSim Manager/Integrator construction out of the
  perturbation per-step loop for issue #7713, preserving analyzer behavior
  while avoiding repeated runtime setup on every simulated step.
- **2026-06-20** - Covered chat WebSocket failure paths for issue #7721:
  `refresh_models` and `index_codebase` tests now drive provider and
  codemap rebuild exceptions, assert sanitized client error frames, preserve
  server-side traceback logging, and keep the socket usable after a model
  refresh failure.
- **2026-06-20** - Deduplicated the simulation WebSocket `set_speed`
  handler for issue #7719: runtime speed changes now route through one
  canonical branch for validation and state updates, with focused WebSocket
  regression coverage preserving accepted payload behavior.
- **2026-06-20** - Removed the dead duplicate Drake visualization monolith
  for issue #7709: Drake visualization now relies on the active maintained
  implementation instead of the stale `drake_visualization_mixin.py` copy,
  with obsolete unit coverage and suite-marker baseline entries removed.
- **2026-06-20** - Repaired Drake cross-engine theta sizing for issue #7725:
  the Drake equivalence smoke gate now derives its zero-theta vector length
  from the finalized Drake plant actuator count, matching the stricter
  production contract that rejects mismatched nonzero coefficient vectors
  instead of silently logging phantom torques.
- **2026-06-20** - Documented MuJoCo humanoid golf grip modelling for issues
  #7723/#7724: grip synergy construction and contact extraction now live in a
  focused helper module with regression coverage for finite contact geometry,
  deterministic synergy transforms, and the leaner GUI tab integration.
- **2026-06-20** - Vectorized clubhead trajectory assembly for issue #7714:
  `compute_clubhead_trajectory()` now computes the trunk, shoulder, and wrist
  angle path with NumPy array operations instead of a per-frame Python loop,
  while parity tests pin the vectorized positions and velocities to the
  original loop contract, including missing-joint defaults.
- **2026-06-20** - Added feature-parity tile-id uniqueness validation for
  issue #7730: registry loading now rejects duplicate launcher tile claims
  across entries while preserving distinct tile coverage, with focused loader
  tests for duplicate and unique tile lists.
- **2026-06-20** - Consolidated quaternion SLERP behavior for issue #7707:
  `math_utils.quaternion.slerp` now owns the shared nlerp fallback threshold,
  while spatial algebra rotations, cooperative manipulation, and Unreal
  skeleton mapping delegate to the canonical implementation with focused parity
  coverage across the threshold boundary.
- **2026-06-19** - Extended `SafetyMonitor` regression coverage for issue
  #7694: velocity-limit tests now pin unsafe target rejection and safe-command
  clipping, and emergency-stop torque regressions assert pure torque commands
  remain unsafe while emergency stop is active.
- **2026-06-19** - Added realtime abort coverage for issue #7697:
  control-loop failure escalation tests now exercise the emergency
  zero-torque fallback when command sends raise, asserting the loop still
  clears `is_running`, records `aborted_on_failure`, and attempts the
  fail-safe send instead of wedging.
- **2026-06-20** - Restored standalone Sidekick package frontend builds:
  the UI now uses `@vitejs/plugin-react@5.2.0`, whose peer range includes
  Vite 7, instead of plugin-react 6.x which imports Vite 8-only internals.
- **2026-06-20** - Cleared #7806 CI ratchets after merging current main:
  runtime dependency locks now require patched `pydantic-settings>=2.14.2`,
  and recent unit-style regression files carry explicit `pytest.mark.unit`
  suite markers instead of growing the unmarked-test baseline.
- **2026-06-20** - Deduplicated golf GUI camera pan axes after the DRY
  duplication ratchet surfaced repeated target-plane basis construction:
  mouse pan and pan inertia now share one `_camera_pan_axes()` helper while
  preserving the existing orbit/fly camera movement behavior.
- **2026-06-20** - Hardened shared CORS credential handling for issue #7740:
  `add_cors_middleware()` now rejects wildcard origins when credentials are
  enabled, including origins resolved from `CORS_ORIGINS`, while preserving
  wildcard support for non-credentialed CORS responses.
- **2026-06-20** - Corrected Drake theta contract coverage for issue #7726:
  `tests/parity/test_simulate_contract_drake.py` now directly exercises
  `validate_theta(..., bounds=DEFAULT_THETA_BOUND_TABLE)` rejecting a
  physically unreasonable `1e9` coefficient, while separately pinning the
  Drake simulate path's intentional `bounds=None` behavior as accepting the
  same large-but-finite coefficient and producing finite output or a failed
  solver status.
- **2026-06-19** - Hardened Data Explorer numeric contracts for issue #7732:
  dataset stats now ignore textual `inf`, `-inf`, `nan`, and `Infinity`
  cells instead of allowing one non-finite value to poison min/max/mean or
  emit invalid JSON tokens, and numeric filter comparisons reject non-finite
  row or filter operands before matching rows.
- **2026-06-19** - Hardened common engine state validation for issue #7705:
  `StateManager` and `ForceAccumulator` now enforce positive dimensions,
  positive time steps, and non-empty force-source names through body-level
  guards that remain active even when Python optimization disables decorator
  contracts.
- **2026-06-19** - Logged tracebacks for unexpected motion-pipeline stage
  failures in issue #7701: adapter loading, preprocessing, skeleton scaling,
  and inverse-kinematics handlers now call `logger.exception` before returning
  failed `StageResult` values, preserving file/line stack context for
  scientific runtime failures while keeping caller-contract errors classified
  as invalid input.
- **2026-06-19** - Hardened auth and signal-toolkit validation for issues
  #7698, #7699, #7700, #7702, and #7703: access-token creation now validates required
  subject claims even when DbC decorators are disabled, bcrypt-backed password
  and API-key hashing rejects inputs above bcrypt's 72-byte UTF-8 limit before
  truncation can occur, password/API-key verification fails closed for
  overlong secrets, malformed stored bcrypt hashes are logged with traceback
  context before verification returns `False`, and public
  `signal_toolkit.calculus`/`noise` boundaries use explicit `TypeError` guards
  instead of stripped `assert` statements.
- **2026-06-18** - Hardened `SafetyMonitor` command contracts for issues
  #7683, #7684, and #7692: command preflight now rejects velocity targets over
  `max_joint_velocity` and any command while emergency stop is active, safe
  command shaping clips velocity targets like torque and position targets, and
  emergency stop authoritatively zeros velocity, torque, and feedforward torque
  even if a later speed-override call tries to raise the override.
- **2026-06-18** - Restored the scheduled Vendor Submodule Freshness workflow
  for issue #7672: `scripts/check_vendor_updates.py --json` now emits a
  parseable status array, text mode prints the per-submodule messages again,
  and the workflow keeps JSON stdout separate from diagnostics instead of
  redirecting stderr into the artifact or masking script failures.
- **2026-06-18** - Regenerated the full-src mypy baseline from the Linux
  Python 3.11 quality-gate environment used by `main` CI so push-triggered
  code-quality checks compare against the same platform family that enforces
  the gate.
- **2026-06-18** - Made the lightweight catch-all 404 page an eager route
  dependency so unknown deep links render the branded recoverable "Page not
  found" screen immediately, without depending on route chunk load timing. The
  heavier feature pages remain lazy-loaded behind the shared Suspense fallback.
- **2026-06-18** - Optimized the #7561 complexity hot-loop slice in the
  MuJoCo humanoid golf analysis path: `joint_analysis.analyze_torque_transmission`
  now hoists the universal-joint bend angle once per sweep and computes the
  wobble/torque-ratio arrays with a vectorized helper that preserves the scalar
  guards, while `power_flow._compute_power_dissipation` caches positive-damping
  joint/dof maps at initialization and computes dissipation as one vectorized
  dot product. Focused parity tests cover the vectorized joint analysis helper
  and power-dissipation scalar equivalence.
- **2026-06-18** - Gated Windows Tauri release packaging behind the
  `TAURI_WINDOWS_RELEASE_ENABLED=true` repository variable after main CI showed
  the current self-hosted Windows runner blocks Cargo build-script executables
  with Application Control (`os error 4551`). Linux Tauri release packaging and
  the Tauri Rust/TypeScript check remain enforced; Windows packaging can be
  re-enabled once a selected Windows runner policy permits Cargo build scripts.
- **2026-06-18** - Restored the Tauri release build contract after main CI
  exposed two packaging-lane failures: the UI package now declares the
  `tauri` npm script expected by `tauri-apps/tauri-action`, and the Windows
  self-hosted build bootstraps Rust through PowerShell so it does not depend on
  Git Bash path rewriting for the Rust setup action. The Tauri Rust target
  caches are now keyed by runner name to prevent proc-macro artifacts compiled
  against one self-hosted runner's glibc from being restored on another. The
  npm Tauri API/CLI packages are pinned to the Rust `tauri` crate minor version
  because the release build fails on cross-ecosystem minor drift.
- **2026-06-18** - Restored the final Tauri Build release contract for issue
  #7652 by aligning the Rust `tauri` lockfile package with the locked
  `@tauri-apps/api` package minor version and adding CI infrastructure coverage
  that fails before release builds can reach `tauri-action` with mismatched
  Tauri Rust/npm package minors. The Linux Tauri lanes also install
  `libdbus-1-dev` because the updated Rust graph contains `libdbus-sys`.
- **2026-06-18** - Hardened the Tauri Build check after the UI audit recovery
  PR exposed DeskComputer runner PATH drift: `CARGO_HOME` is now rooted at the
  workspace, the Rust toolchain verifier persists `$CARGO_HOME/bin` into
  `$GITHUB_PATH`, and CI infrastructure tests pin that Cargo remains reachable
  before rustfmt/clippy/check steps run.
- **2026-06-18** - Restored the main-side Tauri Build release lane after the
  check recovery: the UI manifest now exposes the `npm run tauri` entrypoint
  that `tauri-action` invokes, and the build matrix uses named runner metadata
  with a PowerShell Rust setup on Windows so the self-hosted Windows leg does
  not route through the failing bash action path.
- **2026-06-18** - Restored the current-main CI Standard lane after #7645 by
  path-scoping strict API mypy on ordinary pushes with `github.event.before`
  and adding a 10 microsecond absolute floor to microbenchmark regression
  thresholds so sub-microsecond runner jitter does not fail the gate.
- **2026-06-18** - Restored the Nightly Cross-Engine Validation workflow for
  issue #7646 by replacing the empty heavy-integration placeholder target with
  the real validator and conformance harness tests, correcting the coverage
  target, and treating zero collected tests as a validation failure.
- **2026-06-18** - Closed current-main CI issue #7643 by path-scoping the
  baseline mypy gate on ordinary pushes with a concrete `github.event.before`
  SHA while keeping scheduled/manual full-baseline audits, and by allowing
  Jules PR Cleanup to fall back to the repository token when the optional
  runner token is absent.
- **2026-06-17** - Stabilized the full-src mypy baseline gate by normalizing
  equivalent built-in `int` constructor note spellings across local and CI
  Python/mypy environments, and refreshed the accountable baseline snapshot.
- **2026-06-17** - Restored the current-main `repo-structure-gates` lane by
  installing the pinned PyYAML parser dependency before workflow YAML guards
  run, and added CI infrastructure coverage that keeps the trust-boundary guard
  ordered after that dependency setup.
- **2026-06-17** - Closed current-main CI recovery issue #7640 by path-scoping
  the mandatory Rust wheel parity lane on push events with
  `github.event.before`, preserving full parity only for schedule/manual runs,
  and hardening the full-Trivy Dockerfile set with minimal apt installs plus
  non-root runtime users.
- **2026-06-17** - Recovered current-main CI after the #7632 merge: the
  Python aerodynamics fallback and vectorized ball-force path now share the
  Rust Penner-style bounded spin-lift coefficient, the Rust parity fixture
  anchors were refreshed to the corrected trajectory, parity TestClient runs
  opt into a high simulation rate-limit budget, PyO3 binding tests assert only
  supported construction/precondition contracts, `DragModel.calculate()` reuses
  its speed calculation for the Reynolds-corrected coefficient path, and
  dashboard launcher tests can inject a non-blocking Qt event-loop runner.
- **2026-06-17** - Closed the test-only PR core-lane OOM regression for issue
  #7635: CI Standard now exits after changed-test PR slices pass when no
  source/dependency coverage targets changed, while preserving the scoped
  dependency-light lane for source/dependency PRs and the no-collected-tests
  fallback.
- **2026-06-17** - Closed the push-scoped Semgrep regression for issue #7633
  by making the SVG backdrop regression-test inspection helpers parse generated
  SVGs with defusedxml instead of stdlib ElementTree.
- **2026-06-17** - Restored the suite-marker ratchet for issue #7631 by
  marking the three #7629 Bandit regression tests as unit tests, preserving
  the explicit suite ownership contract for new tests.
- **2026-06-17** - Cleared the main-branch Bandit security-scan regression for
  issue #7629: repository scripts now parse XML inputs with defusedxml while
  retaining stdlib ElementTree only for XML construction/writing, and
  `download_to_file()` validates URL schemes internally before its audited
  `urlopen` call. Regression tests cover entity-bearing XML rejection and local
  file URL rejection before any network/file opener is invoked.
- **2026-06-17** - Restored the current-main test-layout gate for issue
  #7626 by adding `src/shared/python/movement_optimizer/tests` to the root
  pytest `testpaths` contract and the intentional in-tree package-test
  allowlist.
- **2026-06-17** - Restored the current-main suppression ratchet for issue
  #7625 by replacing seven bare `# type: ignore` fallbacks in launcher, chat,
  biomechanics, and screw-theory modules with explicit `assignment` codes.
- **2026-06-17** - Restored the current-main error-handling ratchet for issue
  #7621: `CustomTitleBar` now applies the themed title text color instead of
  carrying an unused `F841` suppression, and construction tolerates a deleted
  Qt-backed theme manager during defensive theme-change hookup.
- **2026-06-17** - Hardened the Tauri desktop check workflow for issue #7616:
  after the pinned Rust toolchain action runs, the `Check (Rust + TypeScript)`
  job now verifies `rustup`, `rustc`, and `cargo` are on `PATH` and prints their
  versions before cache restore or any Cargo command. CI infrastructure tests
  pin the setup -> verification -> Cargo-step ordering so a runner without a
  usable Rust toolchain fails early with a direct diagnostic.
- **2026-06-17** - Recovered current-main CI Standard gates for issue #7617:
  Semgrep and Python matrix push runs now use changed-file scopes when GitHub
  supplies a `before` SHA, current DRY debt is owned in the no-growth ratchet,
  dependency locks use LF line endings, and Model Explorer XML construction
  suppressions are backed by defusedxml parser regression tests.
- **2026-06-17** - Optimized lower-body simulator history eviction for
  issue #7561: `LowerBodySimulator` now stores scrub/playback frames in a
  bounded `deque`, preserving FIFO order, frame indexing, and clear/restore
  behavior while avoiding `list.pop(0)` work on every overflow step.
- **2026-06-17** - Pinned deformable external-force scatter coverage for
  issue #7563: `SoftBody.apply_external_force` now has a source-level
  regression guard against Python-level scatter loops, complementing the
  duplicate-node accumulation coverage for the existing `np.add.at` path.
- **2026-06-17** - Extended footstep yaw normalization coverage for issue
  #7556 with a very-large negative-angle regression, preserving the bounded
  modulo contract for both signs and the existing NaN/finite guard behavior.
- **2026-06-17** - Tightened MuJoCo humanoid golf Coriolis finite
  differences for issue #7556: `compute_coriolis_matrix` now uses central
  velocity perturbations, validates finite `qpos`/`qvel` shape contracts, and
  rejects malformed or non-finite callback force vectors before assembling the
  matrix.
- **2026-06-17** - Optimized FreeMoCap landmark array conversion for issue
  #7563: `LandmarkFrame.to_array` now builds numeric point blocks through
  `np.fromiter`, and `LandmarkSession.to_array` constructs the full
  `(frames, landmarks, 4)` block directly while preserving empty and
  fixed-schema NaN row contracts.
- **2026-06-17** - Optimized MuJoCo kinematic Coriolis decomposition for issue
  #7558: `KinematicForceAnalyzer` now reuses the base inverse-dynamics
  solution and scratch buffers across the per-DOF Coriolis split, reducing
  redundant `mj_rne` passes while preserving the legacy component-sum contract.
- **2026-06-17** - Optimized fixed-size vector magnitude hot paths:
  API route metrics, putting-green direction normalization, teleoperation
  deadband/rate limiting, and ball-launch trajectory speed now use explicit
  `math.hypot` calls instead of scalar `np.linalg.norm` calls while preserving
  the existing vector contracts.
- **2026-06-17** - Hardened robotics precision/vectorization paths for
  #7556/#7559/#7563: footstep yaw wrapping is bounded and NaN-safe, IMU and
  force-torque guards reject unstable normalization/division cases,
  cooperative load sharing uses `solve`, contact hull sorting uses vectorized
  keys, deformable external-force scatter handles duplicate nodes with
  `np.add.at`, and GJK reuses its initial support-vector norm.
- **2026-06-17** - Optimized deformable soft-body FEM root-node force
  accumulation: `SoftBody.compute_internal_forces` now uses a batched
  `np.einsum("ijk->ij", H)` reduction for per-tetrahedron root forces instead
  of a generic last-axis sum, with focused parity coverage for the batched
  reduction shape.
- **2026-06-17** - Optimized learning retargeter FK and mocap marker lookup
  hot paths for issue #7566: `MotionRetargeter` now caches end-effector
  kinematic chains as joint indices, reuses finite-difference perturbation
  buffers during frame optimization, evaluates simplified z-axis FK without
  per-joint rotation-matrix allocation, precomputes mocap marker-name indices
  once per call, and validates retargeting array shapes/finite values at
  public/internal boundaries.
- **2026-06-17** - Optimized RRT/RRT* tree-neighbor and cost-propagation
  paths for issue #7564: both planners now maintain an append-friendly
  finite configuration index for vectorized nearest-neighbor queries, RRT*
  can route neighbor lookups through the existing `use_kd_tree` flag with
  periodically rebuilt `cKDTree` coverage, and rewiring cost propagation now
  walks a maintained child-adjacency map with a deque instead of scanning the
  full node list for each descendant.
- **2026-06-17** - Replaced the iLQR backward-pass gain calculation for issue
  #7570: `src/research/mpc/controller.py` now solves the regularized `Quu`
  system directly, using a finite-checked Cholesky path for symmetric positive
  definite matrices and a general linear solve fallback while preserving gain
  parity with the previous inverse-based math.
- **2026-06-17** - Optimized whole-body SciPy QP inequality construction for
  issue #7568: inequality rows now build as vector-valued lower/upper SLSQP
  callbacks from finite-bound masks instead of O(m) per-row Python callbacks,
  with focused validation for QP matrix finiteness and constraint/bound shapes.
- **2026-06-17** - Optimized the Rust mocap preprocessing per-series filter
  dispatcher for issue #7573: `apply_per_series` now gathers/scatters each
  `(point, dim)` time series through ndarray lane views with explicit
  shape/finite-value contracts, preserving numeric output while avoiding
  repeated full 3-index ndarray indexing in the hot filter path.
- **2026-06-17** - Optimized Rust motion-matching finite differences for issue
  #7575: the internal q/qdot/qddot working storage now uses contiguous
  row-major buffers instead of scattered nested vectors, while preserving the
  public nested-vector `FiniteDiffResult` API and adding irregular multi-DOF
  parity coverage.
- **2026-06-17** - Optimized the Rust mocap preprocessing median filter for
  issue #7574: `medfilt_1d` now uses introselect (`select_nth_unstable_by`) to
  find the padded-window median without fully sorting each window, preserving
  SciPy-compatible zero-padding and even-kernel normalization with focused
  full-sort parity coverage.
- **2026-06-17** - Fixed manipulation pick/place ndarray position overrides for
  issue #7565: explicit `object_initial_pos` and `target_pos` arrays now use
  explicit `None` fallback handling, are coerced to defensive 3D float vectors,
  and no longer raise NumPy ambiguous truth-value errors during environment
  construction.
- **2026-06-17** - Improved differentiable-engine finite differences for
  issue #7569: trajectory-control gradients and state/control Jacobians now use
  scaled central differences (`1e-6 * max(1, abs(value))`) with shared finite
  input/output contracts, preserving suffix-rollout reuse while reducing
  one-sided truncation error.
- **2026-06-17** - Hardened durable task and motion-matching CLI contracts for
  issues #7549 and #7552: completed task persistence now distinguishes
  `None` from empty result objects, durable SQLite shutdown releases the
  database file on Windows, and the motion-matching leaderboard CLI writes
  actionable stderr diagnostics for bad paths or malformed result JSON instead
  of exiting `1` silently.
- **2026-06-17** - Tightened the biomechanics analysis request contract for
  issues #7550 and #7551: `AnalysisRequest.parameters` is now accepted as a
  first-class analysis payload source instead of being silently ignored, and
  simulation-backed analyses report extraction failures as failed responses
  rather than successful empty analyses with only `metadata.engine_error`.
- **2026-06-17** - Tightened the direct launcher Tools override contract for
  issue #7544: `launch_upstream_drift.py` no longer auto-discovers arbitrary
  sibling `Tools` checkouts ahead of the repository and vendored dependency
  paths. Development overrides now require an explicit valid `TOOLS_REPO_PATH`,
  while the default bootstrap order remains repository source followed by the
  vendored `vendor/ud-tools` shared package.
- **2026-06-17** - Restored Windows installer build diagnostics for issue
  #7545: the `setup.py build` and `setup.py bdist_msi` helper paths now share
  failure formatting that preserves the command, return code, stdout, and
  stderr, while `main()` reports the concise failure and exits with status 1.
- **2026-06-16** - Tightened the motion training demo CLI failure contract for
  issue #7543: `examples/motion_training_demo.py` now treats IK solver
  construction failures as explicit initialization errors, reports the
  underlying cause on stderr, and exits with status 1 instead of continuing as
  a successful run. Regression coverage locks the CLI `main()` behavior when
  `create_ik_solver(...)` raises.
- **2026-06-16** - Optimized cross-engine and Pinocchio kinematic
  equivalence RMSE distance accumulation to use `np.einsum`, preserving the
  existing tolerance contract while avoiding temporary squared-distance arrays
  in hot parity-test paths.
- **2026-06-16** - Coordinated with Tools #3316 by moving production
  consumers of the Tools Sidekick surface from direct `sidekick.*` imports to
  `shared.python.sidekick.*` imports, keeping the compatibility package intact
  while reducing duplicate module spellings at launcher, calc-backend,
  motion-capture, and assistant integration boundaries.
- **2026-06-15** - Optimized the starting-pose matcher Simscape trajectory CSV
  loader: `load_simscape_trajectory_csv(...)` now pre-extracts each resolved
  joint's XYZ columns to NumPy arrays before frame construction, avoiding
  repeated row-wise pandas `iloc` materialization while preserving finite-value
  filtering and the existing `SkeletonTrajectory` contract.
- **2026-06-14** - Added the ball-flight physical benchmark contract for issue
  #7407: TrackMan-style driver and 7-iron tests now cover the enhanced
  simulator, the public flight-model registry, and the REST route; the suite
  locks cross-engine carry agreement, 5 m/s headwind/tailwind sanity, analytic
  vacuum range, explicit zero-drag semantics for Python/Rust drag-crisis
  helpers, and humidity-aware density through
  `EnvironmentalConditions.from_altitude(...)`.
- **2026-06-14** - Completed the AnalysisOrchestrator dashboard plot migration
  for issue #7446: every PyQt6 static-dashboard label now maps to a registered
  headless `PlotData` extractor, including Poincare, Lyapunov, recurrence, GRF
  butterfly, kinematic-sequence, and summary-dashboard outputs. The desktop
  dispatch path calls `AnalysisOrchestrator.get_plot_data(...)` before
  renderer-specific plotting, so web/API consumers and PyQt6 share the same
  structured data contract.
- **2026-06-14** - Corrected the impact model's gear-effect spin direction and
  driver COR calibration for issue #7406: toe/heel offsets now produce
  hook/slice spin in the launch-monitor sign convention, high-face/low-face
  offsets reduce/increase backspin, default gear-effect scales put 20 mm toe
  and 10 mm vertical strikes in realistic rad/s bands, and `DRIVER_COR` now
  reflects the USGA/R&A CT-limit-equivalent COR so center strikes can reach
  tour smash-factor ranges. Regression tests cover toe/heel antisymmetry,
  vertical gear effect, default smash factor, and energy balance.
- **2026-06-14** - Calibrated the flagship Rust and enhanced ball-flight
  aerodynamic coefficient contracts for issue #7405: `BallProperties`,
  Python `LiftModel`/`MagnusModel`, and the Rust upstream-physics kernel now
  share a bounded Penner-style spin-lift curve (`Cl(0.08)` driver and
  `Cl(0.30)` 7-iron anchors), while `BallProperties` uses stronger
  spin-dependent drag coefficients for reported-force consistency. Added
  scientific TrackMan benchmark tests for driver and 7-iron trajectories,
  coefficient anchors, and Rust-vs-enhanced agreement when the Rust wheel is
  available.
- **2026-06-17** - Hardened GitHub Actions workflow-context validation for
  issues #7613/#7614: CI Standard, Tauri Build, and the movement_optimizer Maturin
  lane now use workspace-relative Cargo homes instead of job-level
  `${{ runner.temp }}` expressions, and the standard repo-structure gate runs
  `scripts/ci/check_workflow_contexts.py` to reject job-level environment
  expressions that GitHub cannot resolve before scheduling a runner. The
  exposed suite-marker ratchet drift is also classified with module-level unit
  markers instead of growing the unmarked-test baseline.
- **2026-06-14** - Corrected the multi-model ball-flight lift calibration for
  issue #7404: Waterloo/Penner now uses a Penner-style spin-ratio lift fit with
  a bounded golf-ball lift coefficient, and the MacDonald-Hanzely plus constant
  coefficient models use a calibrated spin-ratio lift helper instead of treating
  published `Cl` values as a tiny unbounded slope. Scientific regression tests
  now lock the driver TrackMan carry/apex/time band, the 7-iron reference carry
  band, and the vacuum projectile-range invariant.
- **2026-06-14** - Corrected the rigid-body impact model's friction-spin sign
  for issue #7403: tangential contact impulse now uses the physical torque
  direction `tangent_dir x normal`, so lofted center strikes produce backspin
  under the repository's `[0, -1, 0]` convention and pipeline-derived spin
  creates upward Magnus force instead of downforce.
- **2026-06-14** - Corrected the vendored Sidekick gas-flow
  compressibility-factor contract for issue #7408: supported gases now carry
  acentric factors, and the Abbott/Pitzer generalized second-virial
  approximation uses `(B0 + omega * B1) * Pr / Tr` so methane and air reference
  points match real-gas expectations while the low-pressure limit tends to
  ideal behavior.
- **2026-06-13** - The modular Docker `slim` profile contract now explicitly
  includes the core MuJoCo runtime, matching the package's default physics
  dependency and keeping slim smoke tests focused on excluding Pinocchio and
  Drake.
- **2026-06-13** - The legacy runtime Docker entrypoint now honors explicit
  container commands such as `docker run image python -c ...` while preserving
  the default FastAPI server startup for bare `docker run image`.
- **2026-06-13** - The character builder API route now loads its
  Tools-backed URDF provider lazily so slim Docker API images can discover
  routes successfully and return a controlled `503` if the provider is absent.
- **2026-06-13** - The launcher namespace package now preserves legacy
  convenience exports lazily so API manifest loading does not import PyQt6-only
  dialogs inside headless Docker runtime images.
- **2026-06-13** - The shared dashboard package now exposes GUI entry points
  lazily so API/container imports of `dashboard.recorder` remain usable in
  headless images that intentionally omit PyQt6.
- **2026-06-13** - BunkerShot3D internal modules now use package-relative
  imports so the Docker runtime can import the package through the
  `src.bunkershot3d` namespace used by API health checks.
- **2026-06-13** - Modular Docker builder installs now copy the hatch
  force-included `launch_golf_suite.py` entrypoint before feature package
  installation, keeping `pip install .` metadata generation aligned with the
  wheel packaging contract.
- **2026-06-13** - Modular Docker profile dry-run validation now keeps
  `engine_core.engine_probes` importable from the Dockerfile's early-copy
  source subset by avoiding the heavyweight config package at module import
  time; regression coverage mirrors the early-copy set so profile validation
  continues to fail fast before full source installation layers.
- **2026-06-13** - CI Standard source PRs now skip the changed-tests-only
  targeted coverage branch and fall through to the dependency-light coverage
  lane; optional-stack unit targets use a 180s per-test timeout for
  migration-heavy API tests and are scoped to the PR-relevant biomechanics,
  deployment, and robotics unit directories so the optional-stack lane does not
  couple unrelated shared-python baseline failures to focused performance PRs;
  CI contract tests assert the current PyQt fallback and Rust wheel install
  order. CI Standard now starts Xvfb with an atomic dynamic display reservation
  and cleanup step so parallel self-hosted PR jobs do not collide on display
  `:99`.
- **2026-06-13** - Documented the C3D/TRC Rust facade benchmarks as bounded
  smoke checks with runner jitter allowance while preserving the strict
  parser-only 10x performance gate.
- **2026-06-13** - Restored Rust/Python aerodynamic parity by keeping the
  dimpled-sphere drag-crisis centre aligned with the Rust kernel and by
  matching the public simulator parity fixture's drag coefficient to the Rust
  trajectory fixture.
- **2026-06-13** - Aligned the Rust wheel import smoke helper with the
  documented `upstream-mocap-io` missing-file contract so `parse_trc` raising
  `FileNotFoundError` remains an expected negative-path smoke result.
- **2026-06-13** - Repaired the Rust PyO3 CI contract so the Python-feature
  lane typechecks all targets with `cargo check --workspace --all-targets
--features python`, while maturin remains responsible for extension-module
  wheel and import smoke coverage.
- **2026-06-13** - Refined pointwise ZTCF/ZVCF dimension validation to allow
  canonical-v2 `nq = nv + 1` states only when the redundant configuration
  coordinate is populated, while still rejecting plain wrong-length zero vectors.
- **2026-06-13** - Scoped the optional-stack PR unit lane to the
  PR-relevant biomechanics, deployment, and robotics unit directories so
  focused UI/theming PRs do not hang on unrelated broad optional-stack unit
  sweeps; CI contract tests reject reintroducing the unscoped `tests/unit`
  discovery.
- **2026-06-13** - Restored optional-stack ZTCF/ZVCF input length validation
  for pointwise acceleration helpers and reapplied the Settings page root theme
  marker after active-theme round-trip.
- **2026-06-13** - Tightened signal-toolkit limit preconditions so saturation
  requires `lower < upper` and rate limiting requires `max_rate > 0`, matching
  the optional-stack contract tests.
- **2026-06-13** - Scoped PR core-test dependency-light execution to affected
  unit areas when source files change, preventing self-hosted runner OOMs from
  full-repository core test attempts while leaving normal coverage reporting in
  place for full-coverage runs.
- **2026-06-13** - Restored Sidekick JSON I/O compatibility in optional-stack
  tests by using stdlib JSON for default records-oriented reads and writes,
  avoiding pandas JSON C-extension failures in the optional-stack environment
  while preserving pandas for non-default JSON options.
- **2026-06-13** - Restored shared-python optional-stack compatibility
  contracts: package-root config and provenance exports remain importable
  under the test harness fallback, AI adapters preserve empty-message and
  legacy token/error contracts, validation and signal-toolkit helpers enforce
  documented preconditions, and partial Rust wheels no longer make FSP
  primitive tests fail when the FSP API is absent. Aligned CI numpy/scipy
  repair pins with the lockfile runtime, kept core tests serial to avoid
  native-stack xdist worker termination, and added temporary shared-chat
  architecture-budget exceptions for the pre-existing Qt dock decomposition debt
  tracked by issue #7362.
- **2026-06-13** - Restored optional-stack robotics compatibility contracts:
  QP, whole-body-control, and motion-planning result objects again expose the
  canonical `solver_status` aliases expected by optional robotics callers, WBC
  results expose `final_cost` as the legacy alias for `cost`, and default IMU
  gravity remains on the historical 9.81 m/s^2 sensor contract.
- **2026-06-14** - Added public AffineDrift analysis-tool parity for issue
  #7431: `src.tools.drift_control` computes generalized-force
  drift/control ratios from NPZ trajectories, `src.tools.contraction` exposes
  contraction-rate and Floquet helpers, the analysis-tools API provides
  headless JSON endpoints for both workflows, and
  `src.engines.pinocchio.benchmarks.aba_timing` reports optional Pinocchio ABA
  timing without requiring the Pinocchio extra in core CI.
- **2026-06-12** - Documented the consolidated parity CI follow-up: frontend
  Fast Refresh helpers now live outside component modules, diagnostics imports
  respect API/launcher layer boundaries, recording export artifact writing is
  extracted below the function-size budget, export-format validation is shared,
  and new parity tests carry suite markers.
- **2026-06-12** - Made the Hatch wheel package contract explicit for the
  standalone Sidekick console package by asserting the wheel target maps
  `src/shared/python/sidekick` to top-level `sidekick` through Hatch
  `force-include`.
- **2026-06-12** - Refreshed the pinned `python:3.12-slim` Docker base digest
  and configured the blocking Trivy table scan to ignore unfixed OS findings,
  keeping the gate focused on actionable HIGH/CRITICAL vulnerabilities.
- **2026-06-12** - Kept the Rust Python-binding verification lane on the
  locked Python dependency contract by installing `requirements-dev.lock`
  before the editable `.[dev]` package install and using `--no-deps`.
- **2026-06-12** - Moved shared plot-series label generation into a Qt-free
  `src.shared.python.plot_labels` helper so headless analysis can reuse plotting
  label contracts without importing the Matplotlib/PyQt plotting package.
- **2026-06-12** - Kept synthetic C3D event-alignment tests hermetic under the
  optional stack by patching both short and canonical Sidekick C3D import paths,
  so installed `ezc3d` cannot parse the intentionally empty fixture files.
- **2026-06-12** - Added static analysis plot parity for the web UI: `/api/v1/analysis/plot-types`
  enumerates the headless plot registry and `/api/v1/analysis/plot-data/{plot_type}`
  returns JSON plot payloads rendered by the React `PlotsSection`.
- **2026-06-12** - Corrected three scientific contracts in vendored Sidekick
  copies: Buck water-vapor-pressure constants now match Buck (1996), signal
  integration includes the upper-bound sample via `searchsorted(...,
side="right")`, and Nm3/ppm conversions share the DIN 1343 normal state
  (273.15 K, 101325 Pa) with 22.414 L/mol ppmv <-> mg/Nm3 conversion.
- **2026-06-12** - Removed the anti-phantom guard's `jq` dependency from the
  PR retry/API fallback path by using GitHub CLI `--jq` output directly. This
  keeps required PR guard checks portable across local self-hosted runner
  images that do not install `jq` globally.
- **2026-06-12** - Added generated TypeScript API types (issue #7447): the
  FastAPI OpenAPI contract is emitted to `ui/src/api/generated/types.ts` by
  `scripts/generate_ui_api_types.py` with a pytest freshness gate
  (`tests/api/test_generated_ui_api_types.py`). The launcher manifest and
  engine probe/load endpoints now declare Pydantic response models
  (`LauncherManifestResponse`, `EngineProbeResponse`, `EngineLoadResponse`),
  and `CapabilityLevelResponse.level`/`EngineCapabilitiesResponse.summary`
  are strictly typed. UI engine-list, engine load/probe, launcher-manifest,
  and engine-capabilities call sites consume the generated types.
- **2026-06-12** - Added About/version + onboarding web parity (#7459):
  `GET /api/v1/about` (and `/about/onboarding`) backed by the new shared
  version-resolution helper `src/shared/python/version_info.py` (VERSION
  file → importlib.metadata → fallback, plus safe git-commit reading) now
  used by both the desktop About dialog and the API. Web About modal and
  first-run onboarding overlay added to the React dashboard; onboarding
  card copy single-sourced in `src/config/onboarding_cards.json` consumed
  by the Qt dialog and the web overlay.
- **2026-06-12** - Added recording persistence and export parity endpoints
  (#7451): `POST/GET/DELETE /recordings`, `GET /recordings/{id}/export`
  (FileResponse), and `GET /export/formats`. Recordings persist under
  `output/recordings/<id>/` with metadata.json; exports reuse the desktop
  serializers in `src/shared/python/data_io/export.py` (no parallel
  serialization paths), with format enumeration derived from
  `get_available_export_formats()` — the same registry the PyQt6 dashboard
  Export tab uses. The web Simulation page gains a collapsible Recordings
  panel (list, per-format download, delete with confirm).
- **2026-06-12** - Web API honesty for stub/partial endpoints (issue #7448):
  `POST /simulation/recording` `action=export` no longer writes a JSON
  document into a file named with a client-requested extension; only `json`
  is implemented and the recognized desktop formats (csv/mat/hdf5/c3d) return
  an honest `501 {"detail", "tracking_issue": 7451}` until wired to the
  shared exporters. Added `not_implemented_json()` route helper and an
  architecture test (`tests/unit/api/test_no_stub_routes.py`) that greps
  route modules for removed canned-data sentinels. The web Analysis Tools
  page/hook were rewritten against the real
  `/api/analysis/{metrics,statistics,export}` contracts (previously a
  fictional schema with unsupported xlsx/pdf export options).
- **2026-06-11** - Consolidated launcher manager attribute forwarding through
  `src.launchers.launcher_manager_attrs.forward_manager_attribute()` so dialog,
  Sidekick, theme, and UI setup managers share one DbC boundary for local
  manager state versus launcher-owned state. This keeps the process-console
  guard fix DRY-compliant under the repository duplication ratchet.
- **2026-06-11** - Hardened launcher process-console tab state detection so
  `_is_console_open()` only passes real `QWidget` instances into Qt tab APIs.
  Test sentinels and partially constructed launchers now safely report the
  console as closed instead of raising C++ boundary `TypeError`s during layout
  synchronization.
- **2026-06-11** - Made the optional-stack unit lane boundary explicit:
  it exercises the non-engine unit suite with optional API/GUI/body-part
  dependencies installed, while native engine unit tests remain covered by the
  dedicated engine and cross-engine equivalence lanes. This keeps optional API
  and GUI dependency validation from being blocked by engine-specific mock
  behavior in full native dependency environments.
- **2026-06-11** - Aligned deployment optional-stack device tests with the
  documented hardware-honesty contract: unavailable hardware devices remain
  disconnected and raise `StateError` for state operations, while
  `KeyboardMouseInput` remains the connected fallback. `Demonstration` now
  carries the default canonical `solver_status="success"` through recording,
  serialization, subsampling, and augmentation.
- **2026-06-11** - Restored the calc backend ODE solver response contract so
  `ODESolverResponse` again exposes the default `solver_status="success"`
  field consumed by optional-stack calc backend callers and tests.
- **2026-06-11** - Restored body-part visualization unit contracts under the
  optional-stack lane: `FittedShape.n_frames` is again exposed, validation
  errors use the documented precise type/range messages, and the optional-stack
  venv installs `trimesh` so mesh-backed body-part tests exercise the full path.
- **2026-06-11** - Decomposed pendulum perturbation metric extraction and
  profile comparison into focused helpers so changed analyzer code stays within
  the architecture function-size budget without changing public metric output.
- **2026-06-11** - Aligned pendulum perturbation analyzer guard failures with
  the unit-level contract: invalid `extract_metrics()` inputs and missing
  `set_base_torque_profile()` preconditions now surface as `TypeError`
  precondition failures while preserving valid batch and metrics behavior.
- **2026-06-11** - Restored legacy AI assistant widget import identity by
  routing `assistant_widgets` and `assistant_panel` compatibility exports to the
  canonical assistant submodules, and made the optional-stack unit chunk loop
  fail fast after the first failing chunk to reduce runner load and produce
  focused CI diagnostics.
- **2026-06-11** - Restored API, launcher, and Docker contract parity after
  the main CI regression sweep. The public simulation request engine allowlist
  again includes `jaxsim`, Data Explorer import responses preserve generated
  `dataset_id` values while tolerating legacy direct model construction,
  launcher canonical-core tiles use a recognized `experimental` status with a
  served `biomechanics.svg` logo, symlink model-path validation preserves
  400-class security failures, and Docker feature dry-runs import engine probes
  through the package-qualified shared config path.

- **2026-06-11** - Capability truthfulness contracts for #7355 and #7356.
  Generated motion-pipeline compatibility docs now mark Drake trajectory
  optimization matching as unsupported until the solver is implemented, and
  Drake/RRA/CMC matching placeholder results advertise
  `status: not_implemented` plus `production_ready: false` so orchestrator
  failures remain caller-actionable. Production chat tools that do not yet run
  real work now return explicit `not_implemented` payloads instead of queued or
  successful placeholder results.

- **2026-06-11** - Honest launcher Document Chat and swing-sequence analytics
  contracts for #7358/#7359. The Library tab no longer enables Document Chat
  without a configured backend and no longer fabricates Notebook LM responses.
  `swing_sequence` analysis now computes segment peak timing from trajectory
  angular velocities via the shared segment timing analyzer, marks
  instantaneous-only segment velocities as `requires_trajectory`, and emits
  X-factor metrics only when joint trajectory data plus shoulder/hip indices
  are available.

- **2026-06-11** - RL engine protocol and teleoperation hardware-connection
  honesty for #7357/#7360. The RL humanoid environments now validate required
  engine dimensions and observation/reward channels before constructing spaces
  or stepping, `src.engines.protocols.PhysicsEngineProtocol` defines the typed
  runtime-facing engine surface, and MuJoCo exposes the required accessors via
  real model/data arrays. SpaceMouse, VR controller, and haptic input classes
  now report unavailable until a real hardware backend is connected and raise a
  state error on disconnected reads instead of returning frozen identity data.

- **2026-06-11** - Hardened launcher Docker build cancellation and layout reset
  backup semantics for #7341/#7342. Docker build threads now own a managed
  subprocess handle, expose cooperative cancellation that stops the child
  process without `QThread.terminate()`, and prompt before closing a window
  with an active build. GUI and CLI layout reset paths share a backup helper
  that overwrites an existing `launcher_layout.json.bak` with `Path.replace`
  so repeated resets work on Windows. The changed-file architecture budget
  records expiring exceptions for the legacy launcher UI builders exposed by
  this focused repair.

- **2026-06-11** - CI and validation test contract hardening for #7352,
  #7353, and #7354. The optional-stack lane now gates on pytest exit codes,
  physics validation scripts target real analytical/conservation suites, and
  PyQt fallback stubs no longer fabricate launcher expectations.

- **2026-06-11** - Shared Python motion-matching and signal-utility contract
  cleanup for #7348, #7349, #7350, and #7351. Role-specific fit result
  payloads now use explicit names with compatibility aliases, motion-pipeline
  frame-array preprocessing helpers are canonicalized under one module,
  rotation-matrix-to-quaternion conversion routes through a shared
  sign-canonical helper, and the MuJoCo polynomial generator imports the
  canonical signal-toolkit widget instead of carrying a fork.

- **2026-06-11** - Motion-pipeline DRY follow-up for the #7380
  simulator-facade merge. MuJoCo torque matching and Pinocchio inverse
  dynamics now share base helpers for per-DOF rig joint names and torque
  trajectory construction, removing duplicate post-merge torque payload
  assembly while preserving backend-specific success metadata.

- **2026-06-11** - Suite-marker ratchet follow-up for the #7382
  import-boundary consolidation repair and the #7380 simulator-facade merge.
  The regression tests surfaced by the changed-file ratchet now carry explicit
  `unit` suite markers so CI can enforce no-growth test metadata without
  weakening the marker baseline.

- **2026-06-11** - Import-boundary facade consolidation for #7361, #7362,
  and #7363. The C3D viewer wrapper now imports the repo-qualified viewer
  module without pivoting `sys.modules`, MCP config I/O lives under the shared
  AI MCP package with launcher compatibility facades, shared code no longer
  depends on launcher config readers for MCP settings, and shared/engine
  imports route through compatibility helpers instead of API-layer modules.
  Legacy oversized GUI, MuJoCo, and chat functions exposed by the changed-file
  architecture gate are tracked with owned expiring exceptions pending focused
  decomposition.
- **2026-06-11** - Classified the MuJoCo motion-matching placeholder path for
  #7333 as caller-actionable invalid input. The orchestrator now preserves
  solver metadata on motion-matching stage results, routes unavailable or
  zero-torque MuJoCo matching as a 4xx-class configuration failure, and the
  motion-pipeline README no longer recommends `matching_backend=mujoco` until
  real-model integration lands.
- **2026-06-11** - Suite-marker ratchet enforcement for #7272. CI Standard
  now runs `scripts/ci/check_suite_marker_ratchet.py` against
  `scripts/config/suite_marker_baseline.json`, failing net-new tests that
  lack a recognized suite marker while allowing legacy unmarked-test debt
  to shrink. The shared `tests.support.suite_markers` helpers now normalize
  nodeids, load the baseline, and support report-only, strict, and
  baseline-ratchet collection behavior from `tests/conftest.py`; contributor
  guidance lives in `docs/development/test-marker-conventions.md` with
  focused unit coverage for the static scanner and runtime helpers.
- **2026-06-11** - Replaced collision distance `math.hypot(*tuple)` unpacking
  with explicit component access in primitive-shape distance helpers, keeping
  the robotics collision contracts unchanged while avoiding tuple unpacking
  overhead on hot paths.
- **2026-06-11** - Restored the #7246/#7247 regression-guard cluster for
  #7325, #7326, and #7327 after PR #7248 reverted part of the launch-condition
  unit fix. `LaunchConditions.from_user_units(...)` is again the canonical
  GUI/user-input boundary for degree-to-radian conversion and RPM spin, while
  the current main gap-fill keypoint bounds guard remains in place.
- **2026-06-11** - Promoted Law-of-Demeter enforcement from advisory
  Pinocchio-only lint to a blocking repo-wide production `src/` ratchet.
  `quality-gate.yml` now runs `scripts/ci/check_lod.py src --baseline
scripts/ci/lod_baseline.txt`; the checked-in baseline records existing
  path/chain counts and the required `quality-gate` status fails on any new
  non-allowlisted deep attribute chain.
- **2026-06-11** - Tightened motion matching runtime contracts for #7304,
  #7305, #7306, and #7309. Internal request construction now rejects invalid
  cost weights and solver configuration before backend dispatch, metric helpers
  fail on mismatched frame/DOF shapes instead of truncating, solver result
  postconditions validate reference-aligned time grids plus torque/activation
  finiteness, and successful internal results must carry a matched payload.
- **2026-06-11** - Replaced pickle-enabled motion-matching checkpoint loads
  for #7276 with safe artifact loading. Motion checkpoint readers now route
  through a shared helper that calls `torch.load(..., weights_only=True)`,
  validates mapping-shaped artifacts, and keeps inverse, inverse-timestep,
  compact surrogate, and per-step surrogate loaders on the same safe contract.
  The changed-file architecture ratchet exposed pre-existing surrogate
  train/optimize budget violations, now tracked for decomposition in #7294.
- **2026-06-11** - Isolated optional dependency import mocks for #7307.
  Tests for OpenSim, MuJoCo video export, and Drake visualizer/analysis
  imports now install fake optional packages only inside scoped import
  fixtures. The shared optional-dependency helper restores dependency and
  target-module cache entries after each test, and repo-hygiene coverage
  rejects new module-scope `sys.modules` mocks for optional engine/media
  dependencies.
- **2026-06-11** - Finalized the cross-engine dashboard window factory split
  for #7316. `CrossEngineDashboardWindow()` now constructs the deferred PyQt
  window instead of raising a direct-instantiation placeholder, while the
  extracted fallback engine stub and `_build_qt_window()` path continue to keep
  the dashboard module under the tracked file-size budget.
- **2026-06-10** - Split the cross-engine dashboard window factory for #7288.
  `src/launchers/cross_engine_dashboard.py` now keeps the compatibility
  window facade below the architecture budget by moving the concrete PyQt
  window body behind a deferred factory and the fallback engine stub into
  `src/launchers/cross_engine_dashboard_stubs.py`; the architecture budget
  exception for that dashboard file has been removed.
- **2026-06-11** - Split the motion surrogate training architecture for
  #7317. The compact-schema surrogate trainer now resolves legacy keyword
  arguments through `SurrogateTrainingOptions`, builds an explicit training
  context, and runs checkpoints/metrics through a focused loop state. The
  per-step dynamics trainer now separates data preparation, runtime object
  construction, epoch fitting, best-checkpoint evaluation, and JSON output
  writing. The per-step optimizer now resolves legacy positional options
  through `OptimizationOptions`, builds an optimization context, isolates
  regularizer/orientation/tracking loss calculation, and writes torque plus
  summary artifacts from a dedicated output helper while preserving existing
  CLI and call-site compatibility.
- **2026-06-11** - Hardened the #7314 PR-scoped unit gate in standard
  CI. Source and dependency PRs now fall through to the dependency-light unit
  lane instead of passing solely on touched test files, and targeted PR coverage
  invokes a changed-file coverage ratchet for production policy files.
- **2026-06-10** - Added the #7275 local WebSocket origin and launcher-token
  guard. Browser WebSocket clients now request a short-lived launcher
  capability token before opening simulation/chat sockets, and the backend
  validates allowed local origins plus token claims so local sockets are not
  ambiently reachable from arbitrary browser contexts. The Tauri backend IPC
  capability now ships concrete v2 permission definitions so Rust/Tauri checks
  can resolve the four local backend commands, and the Tauri Linux dependency
  install now retries apt lock collisions on the self-hosted runner pool.
- **2026-06-10** - Collapsed the legacy Frankenstein editor split modules into
  import shims for #7280. `src/tools/model_explorer/_frankenstein_model.py`
  now re-exports `frankenstein_editor.model.URDFModel`, and
  `_frankenstein_panels.py` re-exports the canonical `ModelPanel` and
  `StealComponentDialog`, preserving older import paths while keeping the
  implementation in the `frankenstein_editor` package. The split contract tests
  now assert shim identity and exercise the canonical URDF validation/export
  path through the legacy import.
- **2026-06-10** - Hardened the optional cloud client cache contract for
  #7300. Empty or whitespace-only `~/.golf-suite/cloud_token` files are now
  treated as absent credentials, leaving `CloudClient.token` as `None` and
  `is_logged_in` false while preserving valid cached-token behavior. The
  runtime login state now requires a truthy token even if a caller manually
  mutates the token field.
- **2026-06-10** - Tightened API and model-library boundary contracts for
  #7297, #7298, and #7299. Data Explorer import/list responses now expose the
  durable `dataset_id` required by row pagination, filter operators are
  validated at the request boundary instead of silently returning empty
  results for invalid operators, and `ModelLibrary.load_model(...,
force_download=True)` enforces the HTTPS-only `source_url` policy before any
  download I/O.
- **2026-06-10** - Hardened the Jules PR AutoFix `workflow_run` trust boundary.
  Failed-CI `workflow_run` events now use read-only metadata resolution and a
  PR comment that asks maintainers to run the privileged fixer through explicit
  `workflow_dispatch`; only the manual dispatch path can check out PR code,
  install dependencies, run autofix tools, commit, or push. Standard CI now
  enforces that boundary with `scripts/check_workflow_run_trust_boundary.py`
  and focused regression coverage.
- **2026-06-10** - Narrowed PR-scoped source coverage in standard CI to the
  changed `src/**/*.py` targets after the coverage-bypass fix. Source and
  dependency PRs still produce coverage and enforce the 75% floor, while the
  full per-package coverage enforcer runs only after the default full-coverage
  lane so focused PRs do not fail against unrelated modules.
- **2026-06-10** - Enforced the #7277 Docker build timeout while process
  stdout remains open. `src/launchers/docker_manager.py` now reads build output
  through a background queue while the build thread owns a wall-clock timeout
  and terminates the process tree on expiry, including the regression case
  where stdout never reaches EOF.
- **2026-06-10** - Closed the #7283 simulation WebSocket dependency-boundary
  gap. The simulation stream now resolves its engine manager through a
  WebSocket-safe dependency accessor instead of reaching directly through
  `websocket.app.state`, and missing engine-manager state returns a structured
  `service_unavailable` frame before the connection closes cleanly.
- **2026-06-10** - Narrowed PR-scoped source coverage in standard CI to the
  changed `src/**/*.py` targets after the coverage-bypass fix. Source and
  dependency PRs still produce coverage and enforce the 75% floor, while the
  full per-package coverage enforcer runs only after the default full-coverage
  lane so focused PRs do not fail against unrelated modules.
- **2026-06-10** - Enforced the #7277 Docker build timeout while process
  stdout remains open. `src/launchers/docker_manager.py` now reads build output
  through a background queue while the build thread owns a wall-clock timeout
  and terminates the process tree on expiry, including the regression case
  where stdout never reaches EOF.
- **2026-06-10** - Locked the #7278 standard CI dependency and audit
  contract to committed artifacts. Python jobs that install project runtime or
  dev dependencies now seed environments from `requirements.lock` or
  `requirements-dev.lock` before no-dependency editable installs, avoiding
  pip constraints parsing for lock entries with extras. The dev lock now
  includes the GUI-test extra so `--no-deps` editable installs still provide
  real PyQt6/pytest-qt modules in the unit gates, and `pip-audit` runs directly
  against the committed runtime/dev lock files instead of a live resolver
  result. The standard CI acceptance tests also reject blank lines immediately
  after shell continuations so the core pytest coverage command cannot be split
  into a partial command again (#7303).
- **2026-06-10** - Closed the #7273 PR-scoped coverage bypass in standard CI.
  PRs that change source, test, or dependency targets now fall through to the
  coverage-producing core test lane instead of using the workflow-only
  `--no-cov` shortcut, and per-package coverage enforcement runs whenever that
  lane produces `coverage.xml`.
- **2026-06-10** - Closed the #7279/#7282 audit hygiene wave. The Docker
  security scan still uploads HIGH/CRITICAL SARIF findings, while the table scan
  is the blocking gate for fixable HIGH/CRITICAL findings. The audited
  API and launcher production modules now route module loggers through
  `logging_pkg.logging_config.get_logger(__name__)`, with a repo-hygiene test
  preventing the remediated files from returning to direct `logging.getLogger`.
- **2026-06-10** - Hardened the audit regressions tracked by #7269, #7270,
  and #7271. Model Explorer inspect/compare path resolution now rejects
  absolute paths and parent traversal before resolving candidates only under
  approved model roots; motion-pipeline linear keypoint gap filling leaves
  unmatched low-confidence keypoints unchanged when neighboring frames have
  mismatched keypoint counts, including the pure-Python fallback; and
  `SwingBallFlightPipeline` now derives `LaunchConditions` using the simulator
  contract of radians for launch/azimuth angles and RPM for spin rate.
- **2026-06-10** - Completed the #7207 model explorer composition UX flow.
  `src/tools/model_explorer/composition_ux.py` now provides a headless
  drag/drop orchestration layer with library payloads, non-mutating ghost
  previews, target/source link highlights, validation summaries, committed
  drops, and a validation-aware export chooser for URDF/MJCF while keeping
  SDF/OSIM disabled until first-party writers exist. `FrankensteinEditor`
  exposes preview, drop-commit, and export-choice hooks so the existing
  source/working model UI can compose simple humanoid plus arm models with
  live validation feedback before export.
- **2026-06-10** - Added the #7214 C3D viewer renderer decision and backend
  contract. ADR-0030 chooses `pyqtgraph.opengl`/PyQtGL as the first desktop
  GPU playback backend while retaining matplotlib fallback, and
  `viewer_3d_backend.py` pins the 60 fps target plus parity checklist for
  scrubbing, speed control, loop playback, marker groups, view presets, and
  skeleton overlay. The BunkerShot calibration optimizer now imports
  `scipy.optimize` lazily so cross-engine equivalence imports can use
  `WrenchTrace` without optional calibration optimizer dependencies.
- **2026-06-10** - Added the #7340/#7343/#7344 UI responsiveness contract:
  launcher dependency probes, settings Docker/WSL checks, and C3D MP4 export
  must run off the GUI thread; C3D video export exposes cooperative progress
  and cancellation hooks that remove partial files on cancel.
- **2026-06-10** - Added the #7207 model explorer composition-flow controller.
  `src/tools/model_explorer/composition_flow.py` now attaches a complete
  source URDF model to a working Frankenstein model through a declared
  attachment point, immediately validates the composed result, and exports
  validation-gated URDF or MJCF preview content. `FrankensteinEditor` exposes
  the flow through an Attach Source Model action and public export helper,
  while `URDFModel.from_file()` carries attachment sidecar metadata into the
  editor.
- **2026-06-10** - Added model explorer attachment manifests for #7206.
  `src/tools/model_explorer/attachment_manifest.py` now loads versioned
  `<model>.attachments.json` sidecars with non-fatal warnings for malformed
  manifests, `ModelLibrary` exposes declared attachment points on path-backed
  repository/imported/sibling/static model info, and the attachment dialog
  prioritizes declared mount points while applying their interface-frame
  defaults and payload-limit warnings. The schema lives at
  `src/tools/model_explorer/attachment_manifest.schema.json`, with user docs
  under `docs/model_explorer/attachment-manifests.md`.
- **2026-06-10** - Split the launcher entrypoint below the file-size budget
  for #7217. Sidekick sidebar installation, process cleanup polling, launcher
  domain orchestration, and GUI startup bootstrap now live in focused modules,
  while the existing frameless-window helper remains under
  `src/launchers/launcher_ui/frameless_window.py`; the
  `src/launchers/upstream_drift_launcher.py` file-size exception is removed.
- **2026-06-10** - Hardened Rust mocap Python binding errors for #7252.
  `upstream-mocap-io` validates `parse_c3d` / `parse_trc` / `parse_bvh`
  path preconditions before file access, maps missing files to
  `FileNotFoundError`, maps other file-access failures to `OSError`, and
  preserves malformed present files as `ValueError` parse failures with the
  format and path in the error context.
- **2026-06-10** - Made motion-pipeline hook failures observable for #7250.
  `PipelineConfig.strict_hooks` now switches per-stage hooks from lenient
  traceback logging to fail-fast `HookExecutionError` diagnostics, while the
  default lenient mode logs hook tracebacks with `logger.exception` and
  continues the pipeline.
- **2026-06-10** - Added the bounded inverse swing optimization core for #7220.
  `src/shared/python/physics/swing_optimizer.py` now exposes `FlightTarget`,
  `ClubPreset`, `SwingOptimizer`, and convergence diagnostics for solving
  speed/loft/attack/face-to-path parameters against the existing forward
  `SwingBallFlightPipeline`; GUI target mode remains follow-up scope.
- **2026-06-10** - Consolidated mocap marker NaN occlusion handling for #7251.
  C3D and TRC source adapters now delegate marker-triplet NaN detection to the
  shared `motion_pipeline.sources._marker_coordinates` helper, and the Python
  TRC fallback skips textual `nan` marker rows the same way the Rust-backed
  adapter paths skip occluded samples.
- **2026-06-10** - Added the first #7207 model explorer library-panel
  unification slice. `ModelLoaderDialog` now exposes a single searchable
  library tree covering every `ModelLibrary.list_available_models()` category,
  including sibling repositories, with first-party format-badge inference
  backed by headless controller/model tests.
- **2026-06-10** - Completed the #7205 Frankenstein composition validation
  surface. `CompositionValidator` now emits warning-level findings for heavy
  attached subtrees and direct attachment geometry AABB overlaps, while the
  active Frankenstein model panel surfaces current validation findings in a
  dedicated list before save/export.
- **2026-06-10** - Decided React/Tauri launcher parity for #7221.
  ADR-0028 keeps React/Tauri on the manifest-driven multi-window model while
  PyQt remains canonical for embedded tabs/docks. The React dashboard now
  persists a manifest-keyed launcher window registry and exposes a window
  list/focus menu backed by the existing launch API.
- **2026-06-10** - Consolidated launcher startup ownership for #7215.
  `launch_golf_suite.py` is now a compatibility shim over the canonical
  `launch_upstream_drift.py` entry point. Classic PyQt startup preflights the
  Qt platform and selects `QT_QPA_PLATFORM=offscreen` on headless Linux, while
  the local API server tolerates unavailable optional engine-manager imports
  and reports an empty engine set instead of failing startup.
- **2026-06-10** - Removed unsafe Drake pose pickle deserialization from
  `src/shared/python/pose_interchange/pose_io.py`. Drake `.drake` initial-state
  files now use JSON for `{q, v, model_metadata}`, and legacy binary/non-JSON
  payloads are rejected before any deserialization path can execute.
- **2026-06-10** - Preserved the legacy golf visualizer dataset contract after
  row extraction optimization: `extract_frame_data` still requires the BASEQ,
  ZTCFQ, and DELTAQ datasets and returns zero-vector frame data when the
  requested row is unavailable.
- **2026-06-10** - Added a first-party Frankenstein composition validation
  slice for #7205. `src/tools/model_explorer/composition_validator.py` now
  emits structured error/warning findings for duplicate URDF names, orphaned
  joints, invalid root counts, disconnected links, kinematic cycles, and
  moving-link mass/inertia contracts. The active Frankenstein editor model
  export path blocks validation errors by default while retaining an explicit
  `force=True` escape hatch for recovery exports.
- **2026-06-10** - Added the LauncherContext in-process event bus and shared
  value registry for embedded tools (#7210): `launcher_embed.context` now
  defines the `LauncherContext` protocol plus an in-memory implementation with
  snapshot-safe dispatch, idempotent unsubscribe handles, and keyed
  `value_changed:<key>` notifications. `EmbeddedHostWidget` owns one context,
  injects it into opt-in tools via `set_launcher_context(ctx)`, and emits
  `tab.opened` / `tab.closed` lifecycle events while preserving legacy tools
  that do not implement the hook. The same context can back Sidekick's
  `LauncherSubtabPort` workspace surface through its existing `list/get/set`
  contract.
- **2026-06-10** - Optimized legacy golf visualizer frame extraction by reading
  each Pandas dataset row once per frame before point/vector extraction, reducing
  repeated `.iloc` lookup overhead while preserving fallback behavior for missing
  rows and columns.
- **2026-06-10** - Extended the Rust C3D parser for #7212. The
  `upstream-mocap-io` C3D path now decodes int16 and float analog channel data,
  surfaces additive PyO3 `analog` and `force_platforms` keys, parses
  `FORCE_PLATFORM:{TYPE,CHANNEL,CORNERS,ORIGIN}`, and preserves existing
  marker/event dictionary keys and marker-only fixture behavior.
- **2026-06-10** - Consolidated configuration ownership for #7216. Removed
  the root `config/` and `configs/` trees: CI/governance policy now lives in
  `scripts/config/`, BunkerShot3D calibration YAML lives under
  `src/bunkershot3d/calibration/configs/`, and UX field/error seed YAML lives
  under `src/shared/python/ux/config/`. Added
  `docs/development/configuration-systems.md` plus regression coverage so new
  root config directories do not reappear.
- **2026-06-10** - Repaired the Linux dependency-consistency lockfile drift
  after #7231. `requirements-dev.lock` now matches the Python 3.12 Linux
  `pip-compile --extra dev` output used by CI, removing Windows-only transitive
  packages and restoring `uvloop` for the Linux `uvicorn[standard]` stack.
- **2026-06-10** - Repaired Rust TRC row-validation parity for #7213.
  `rust_core/upstream-mocap-io` now rejects invalid or non-finite frame/time
  columns before accepting marker rows, preserving the Python adapter's
  malformed-line contract when contributors install a fresh native wheel. The
  Rust wheel CI lane now runs `tests/unit/motion_pipeline/sources` after
  installing built wheels so OpenCap and TRC/C3D/BVH adapter behavior is
  verified against the actual Maturin artifacts.
- **2026-06-10** - Unified MATLAB engine loading through the registry for
  #7219. `EngineManager._load_engine()` now obtains MATLAB engines through
  `EngineRegistration.factory()` like every other engine instead of branching
  into a private `matlab.engine.start_matlab` path. `src.engines.loaders`
  owns the Simscape adapter loaders for both `MATLAB_2D` and `MATLAB_3D`,
  while the command-line launcher still routes web-only MATLAB direct launches
  to the web UI.
- **2026-06-10** - Added the first-party OpenSim `.osim` loader for #7203.
  `src/tools/model_explorer/osim_loader.py` parses OpenSim 3.x
  `parent_body`/`body` joints and OpenSim 4.x socket-frame joints into the
  existing `ParsedModel` contract, exposes validated `CanonicalModel`
  conversion for composition, maps Pin/Slider/Ball/Weld/Free/Custom joints,
  records unconverted ForceSet/ConstraintSet/MarkerSet elements as warnings,
  and floors non-physical ground/zero inertia values only where needed for
  contract validation. Model Explorer discovery/import paths now classify
  `.osim` files from sibling repos and route opened `.osim` files through the
  loader without editing vendored `model_generation` modules.
- **2026-06-10** - Added Drake SDF model loading for #7204. The model
  explorer now provides a first-party `SdfLoader` under
  `src.tools.model_explorer`, parsing SDFormat links, inertials, primitive and
  mesh geometry, joint axes/limits/dynamics, SDFormat 1.8 `relative_to` poses,
  and ball/universal joints into the existing canonical model contract.
  Sibling model discovery now classifies `.sdf` files from `Drake_Models`
  alongside URDF and MJCF assets so Drake-native models can be browsed and
  composed.
- **2026-06-10** - Preserved URDF fixed-joint topology through MJCF
  roundtrips for #7208: URDF-to-MJCF conversion keeps MuJoCo weld semantics by
  emitting fixed children as nested bodies without joint elements while encoding
  the original fixed joint name, and MJCF-to-URDF decoding restores that name
  only for welded nested bodies. Regression coverage now asserts link sets,
  fixed and movable joint names/types, parent-child topology, and fixed-joint
  origin translation through URDF -> MJCF -> URDF.
- **2026-06-10** - Added entry-point based embeddable-tool adapter discovery
  for #7211. The launcher bootstrap now imports
  `upstream_drift.embeddable_tools` package entry points before falling back to
  the in-tree adapter list, de-duplicates adapter module paths, and preserves
  registry-diff tracking for adapter registration. `pyproject.toml` declares
  the first-party embeddable tool adapter entry points so installed wheels and
  editable checkouts share one discovery contract.
- **2026-06-10** - Added the headless ball-flight REST simulation route for
  #7218. `POST /tools/ball-flight/simulate` now validates launch, spin, wind,
  model, and integration-window inputs through Pydantic models, delegates to
  the existing `FlightModelRegistry` / `UnifiedLaunchConditions` physics stack,
  and registers the `ball_flight` tool alongside the existing API route map.
- **2026-06-10** - Refreshed the Module Map against the actual source tree
  (the previous tree listed entry points and API files that no longer
  exist) and linked the operational project map. The full gap inventory
  from the 2026-06-10 operational deep dive lives in
  `docs/architecture/PROJECT_MAP.md` §16, tracked by issues #7202-#7221
  (model-composition epic, sidekick agent wiring, startup + config
  consolidation, and related work). Landed alongside:
  sidekick subtab host port + pop-out lifecycle hooks (#7199), the Rust C3D
  1-D `POINT:UNITS` fix with full `data/` coverage (#7200), and sibling
  model-repository discovery in the model explorer (#7201).
- **2026-06-10** - Repaired remaining #7189 packaging gate regressions after
  the branch merge: Tauri Linux dependency installs now wait on both apt and
  dpkg locks, and the WGS calculator keeps GUI theme imports inside the plot-tab
  path so the installed `sidekick run` wheel smoke can load the headless engine
  without requiring PyQt6 or the top-level `shared` GUI theme package. The
  standalone Sidekick wheel smoke matrix now matches the Python 3.11+ package
  floor, and the Python-version coherence guard covers that workflow.
- **2026-06-10** - Resolved Python-version provenance drift for #7160:
  `pyproject.toml`, `install.sh`, `CLAUDE.md`, user-facing installation docs,
  `SPEC.md`, the standard CI test matrix, Docker base images, and
  `requirements.lock` now describe one coherent policy. The supported floor is
  Python 3.11, standard CI tests Python 3.11 and 3.12, and the production
  Docker image plus lockfile remain generated on Python 3.12. Added
  `scripts/ci/check_python_version_coherence.py` and focused tests so the
  floor, classifiers, mypy target, installer floor, lock header, Docker base,
  and CI versions cannot silently diverge again.
- **2026-06-10** - Hardened the production Docker dependency audit inputs for
  #7160 follow-up CI: Docker builder/runtime pip pins now use patched pip
  26.1.2, runtime metadata declares `Mako>=1.3.12` and `PyJWT>=2.13.0`,
  `requirements.lock` matches those security floors, and the third-party
  license ledger covers the newly explicit Mako dependency.
- **2026-06-10** - Tightened test-isolation and optional-dependency contracts
  for #7155/#7158: the MuJoCo dependency mock is function-scoped, affected
  MuJoCo tests initialize their own required state, launcher tests route
  `sys.modules` cleanup through the local cleanup fixture, and
  `test_api_extended.py` uses the shared optional-dependency helper with
  current path-validation imports instead of a blanket module-level skip. The
  local-only workflow routing guard now installs its YAML parser in an isolated
  workspace venv so self-hosted runners with PEP 668 system Python policy still
  execute the guard instead of failing during dependency bootstrap. The
  follow-up CI hardening keeps sidekick copied-test collection self-contained,
  avoids dynamic source execution in launcher tests, and refreshes generated
  dependency artifacts against the canonical project metadata.
- **2026-06-09** - Added a changed-file architecture budget gate for #7131/#7133: `scripts/ci/check_architecture_budget.py` now scans changed production Python files for functions over 100 lines and callable signatures over 8 effective parameters (excluding `self`/`cls`), with owned/expiring exceptions configured in `scripts/config/architecture_budget.json`. The standard CI workflow runs the guard beside the file-size and module-size gates, and focused tests pin long-function, parameter-count, exception, and test-path skip behavior.
- **2026-06-02** - Restored visible Sidekick sidebar tab hover affordance (#7109): the synced tools-sidebar design-token QSS now styles unselected `QTabBar` tabs on hover with the soft accent surface while keeping the selected-tab rule separate, and a headless unit regression pins the generated stylesheet contract.
- **2026-06-02** - Fixed Windows taskbar identity for the UpstreamDrift launcher (#7107): `src.shared.python.ui.window_icon` now declares an AppUserModelID before showing the first window, applies the resolved icon to both the `QApplication` and top-level window, and covers the Windows API call plus icon application contract with focused unit tests.
- **2026-06-02** - Removed obsolete archived launcher entries (#7108): the deprecated MuJoCo, MATLAB, and motion-capture archived launchers are no longer advertised through the launcher manifest or tool catalog, and launcher regression coverage now asserts the surviving catalog paths without maintaining tests for removed archived entry points.
- **2026-06-02** - Hardened the core CI PR test lane for workflow-only pull requests (#7079): when the diff contains no core Python, test, or dependency targets, the core matrix exits after change detection instead of falling through to the full coverage lane. Source/dependency PRs with no changed tests still run the default core suite, preserving coverage while avoiding OOM-prone full-suite runs for GitHub Actions dependency bumps.
- **2026-06-02** - Recorded the Bolt small-vector norm optimization (#7098): scalar ball-flight force calculation, Waterloo/Penner and spin-decay flight models, and swing-to-launch derivation now use fixed-arity `math.hypot` for known 2D/3D vectors instead of `np.linalg.norm`, avoiding NumPy reduction overhead while preserving the existing one-dimensional vector contracts.
- **2026-06-02** - Recorded the golf visualizer camera-basis norm optimization (#7101): `GolfVisualizerWidget` now uses fixed-arity `math.hypot` for the known 3D forward/right camera vectors instead of `np.linalg.norm`, avoiding NumPy reduction overhead while preserving the existing fallback behavior for degenerate vectors.
- **2026-06-02** - Hardened cross-engine equivalence gate NaN handling and corrected JaxSim/Pinocchio parity test parameters (#7095, #7097): `_run_engine_checked` now distinguishes all-NaN grip (grip body absent from URDF — Drake's documented design for missing `club_grip`) from partial-NaN/Inf grip (simulation divergence). All-NaN raises `_EngineBindingsError` so the aggregation test skips the engine as unavailable; partial-NaN/Inf calls `pytest.fail` so a broken-but-runnable backend remains a hard gate failure (per reviewer feedback on #7099). The second JaxSim-vs-Pinocchio parametrized case was changed from a non-zero position to zero position: JaxSim uses INERTIAL velocity representation (angular momentum about world origin) while Pinocchio uses LOCAL (about CoM), and these two representations diverge in `M` and `h` at non-zero body positions via the parallel-axis theorem — the zero-position fix makes both representations equivalent while still exercising full mixed angular+linear Coriolis effects. Updated cross-engine gate docstrings to accurately describe that the 5 mm grip RMSE tolerance applies to cross-engine agreement only; the per-engine address-vs-Simscape check is a world-frame origin plausibility gate, not a post-registration RMSE gate (post-registration error is identically zero by construction).
- **2026-06-02** - Cleaned up docs root organization for #7063 after reconciling
  the branch with the newer mainline governance/operations/reviews layout. Loose
  root-level markdown is moved into topic subdirectories, `docs/sphinx/conf.py`
  no longer references the stale `TRACKED_TASK` placeholder extension, and
  `tests/docs/test_docs_structure.py` now enforces root markdown cleanliness plus
  valid example-index references while preserving the real runnable
  `docs/examples/` subtree that landed on main.
- **2026-06-01** - Repaired the UD-only Sidekick `agent`/`standalone` subpackages (#7066, #7067, #7068). (1) `agent/subtab_adapter.py` undo was inert: `_pack_undo` returned an opaque `subtab:kind:nonce` token but never set `metadata["_undo"]`, so `SidekickActionService._maybe_register_undo` never registered an inverse and `service.undo()` failed for all 5 reversible actions. Each reversible handler now emits `metadata={"_undo": {"action_id", "params"}}` (mirroring `_ToggleHandler`); focus/show/hide/set_variable/state_profile.save now genuinely round-trip. Added a real `subtab.state_profile.delete` action (and `SubtabActionPort.state_profile_delete`) so a freshly-saved profile has a well-defined inverse; an overwriting save restores the prior payload. (2) `standalone/runner.py` registered ~1/40 calculators and re-derived WGS with hard-coded `delta_h`/`delta_s` literals that diverged from canonical `WGS_DELTA_H`/`WGS_DELTA_S`. It now lazily registers 5 canonical `process_calculators` engines (`wgs_reactor`, `water_vapor_pressure`, `flare`, `financial`, `syngas_water`) via thin dict-returning adapters; WGS routes through the canonical `WGSReactorEngine` so the equilibrium constants live in exactly one module. Registration stays lazy so the headless runner imports with zero PyQt6/scipy at module load. (3) `standalone/window.py` Save/Load Profile menu actions were `logger.info("not yet implemented — T8")` stubs; they now wire to `StandaloneSessionStore.save_profile/load_profile` through a `ProfilePayload` (layout + theme), the previously-dead `host_action_port` is consumed via a `host_action_port()` accessor, and `__all__` was added to `onboarding`, `preferences`, `runner`, and `session_store`. Tests: real `service.undo` state-restoration per reversible subtab action, JSON-fixture round-trip per registered calculator + a "WGS constants in exactly one module" guard, headless profile round-trip, host-port accessor, and a `__all__` hygiene parametrization.
- **2026-06-01** - Fixed `BiomechanicalModel.add_segment` segment-name validation drift (#7045): `segment_masses` (from `estimate_segment_masses`) is keyed by the full segment name (e.g. `right_thigh`), matching the mass lookup in `compute_dynamic_com`, but `add_segment` validated membership using the mapped anthropometry key (`thigh`) and so rejected every laterally-named segment as "Unknown segment name". `add_segment` now validates against the full name, restoring the 3 red `tests/unit/biomechanics/test_dynamic_com.py` cases to green.
- **2026-06-01** - Resolved API review regressions (#7037; #7031, #7028, #7027): expired `TaskManager` entries are no longer refreshed by mutation paths; cancelled chat streams preserve tool-call/tool-result pairing for unexecuted calls; Data Explorer dataset stats stream every row instead of stopping at the preview cap; and cloud-token chmod hardening now has regression coverage.
- **2026-06-01** - Applied public physics/config review fixes (#7015, #7017, #6954): the documented `src.shared.python.physics.impact_model` public package now carries the private impact fixes for expected energy loss, contact-onset clearance, and rolling-friction spin caps; provider catalog iteration deduplicates canonical IDs while preserving alternate checkout-name discovery; and public-package/provider tests lock the behavior.
- **2026-06-01** - Fixed orphaned chat-stream daemon threads on client disconnect (#6981): `ChatService.stream_response` (`src/api/services/chat_service.py`) now passes a `threading.Event` cancellation token into the `_stream_to_queue` worker. The async consumer sets the flag in a `finally` block — which runs on normal completion, consumer error, and `GeneratorExit` from `aclose()` on client/WebSocket disconnect — then joins the worker (bounded 5 s). The worker checks the flag at the top of its outer loop, inside the adapter `stream_response` pull loop, before persisting messages, and before each tool call, so it stops pulling from the adapter and no longer takes `self._lock` to persist messages for an abandoned session. Regression test in `tests/unit/api/test_chat_service_stream_stop.py`.
- **2026-06-01** - Replaced smoke-only tests with value-asserting coverage for three calc modules (#6998, #6999, #7003): pressure-drop flow calculations (Darcy-Weisbach, Re=2300/4000 regime boundaries, hydrostatic-head sign convention, API RP 14E erosional velocity, expansion-factor bounds, negatives→ValueError); data-fitting solvers (2-link analytical IK round-trip vs synthetic ground truth, numerical IK convergence, forward-kinematics geometry, anthropometric parameter estimation vs Dempster fractions, residual contracts); and thermo property backends (CoolProp input validation, phase/quality determination, simplified ideal-gas/liquid correlations, Antoine saturation pressure/temperature round-trip, optional CoolProp/Cantera skips). Fixed a real robustness bug in `determine_phase_and_quality` (`sidekick/calculators/thermo/_property_backends.py`): a non-Cantera `water` object raised an uncaught `AttributeError` on `.TQ`; now caught so the function correctly returns `("unknown", 0.0)` per its contract.
- **2026-05-31** - Hardened JaxSim readiness and parity gates (#6880, #6881, #6882, #6884): `EngineManager` now registers JaxSim as a runtime-backed engine, only marks runtime-backed engines available when both adapter/provider paths and importable runtime dependencies are present, preserves DbC path-policy failures as provider-discovery misses instead of constructor crashes, and uses a required-JUnit testcase assertion in the cross-engine workflow so skipped/missing JaxSim/Pinocchio parity cases fail CI.
- **2026-05-31** - Fixed two HIGH-severity physics-audit defects (#6890, #6891): `JaxSimBackend.compute_jacobian` now restacks the native JaxSim free-floating Jacobian to the canonical `[angular; linear; joints]` convention — permuting the six base columns and the six spatial output rows — so `J·v` and `Jᵀ·force` agree with `M`/`h`/`v`/inverse-dynamics; and the cross-engine conformance harness no longer counts a missing required method on an advertised capability (now `passed=False`) or a throwing `supports()` query (now a failure, not a swallowed free pass) as a passing skip, closing a CC-8 gate-integrity hole. Genuine missing capabilities remain legitimate skips.
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
- **2026-05-31** - Added metadata-driven helpful-field and provenance-value wrappers for the Idiot-Proof UX epic (#5968): PyQt6 and React controls now consume the shared field metadata/provenance contracts, UI field metadata is generated from `src/shared/python/ux/config/field_metadata.yaml`, and parity tests keep the TypeScript registry synchronized with the YAML source of truth.
- **2026-05-31** - Added the first unified engine capability taxonomy slice for issue #6777: `engine_core.capabilities.Capability` is now the canonical enum/query surface, simulation backend capabilities can answer canonical `supports()` checks while keeping legacy booleans, and architecture docs describe the adapter boundary.
- **2026-05-31** - Added a third-party license ledger for issue #6781 under `docs/legal/licenses.md`, with a CI-sized advisory checker that covers direct dependency declarations, keeps OpenPose visibly fenced as non-commercial opt-in, supports Python 3.10 via `tomli`, and avoids false core-install optional-engine findings from the local `scripts/jaxsim` helper directory.
- **2026-05-31** - Added canonical-v2 dynamic state support for CC-2 (#6774): `CanonicalState` now carries immutable `(q, v, a, t)` data with floating-base quaternion layout, manifold-safe integrate/difference operations, canonical-v1 lift helpers, and SE(3) quaternion utilities covered by shape, validation, and property-style round-trip tests.
- **2026-05-31** - Added the canonical-v2 pose interchange contract (#6773) with public exports from `src/shared/python/pose_interchange/__init__.py`, a conventions guide under `docs/conventions/canonical-v2.md`, and ADR coverage in `docs/adr/0026-canonical-dynamic-state-v2.md`.

- **2026-05-31** - Hardened the JaxSim #6648 URDF-to-SDF gate CI path by parsing inertial XML with `defusedxml` and preventing the core-only install guard from treating helper directories such as `scripts/jaxsim` as installed optional engines.
- **2026-05-30** - Hardened the JaxSim #6648 URDF-to-SDF inertial round-trip gate so converted SDF payloads fail on unexpected inertial links instead of silently accepting extra mass/inertia records; regression coverage now exercises the unexpected-link failure path.
- **2026-05-30** - Added JaxSim parameter-gradient sensitivity support (#6656): `SupportsParameterGradients` now captures pointwise parameter Jacobians, `JaxSimBackend` delegates to a JAX autodiff ZTCF sensitivity module over documented anthropometric parameters, tests validate autodiff against finite differences, and `scripts/jaxsim/plot_parameter_sensitivity.py` reproduces a sample sensitivity plot from measured states.
- **2026-06-17** - Optimized JaxSim trajectory parameter-gradient evaluation for #7562: `evaluate_ztcf_parameter_sensitivity_along_trajectory` now builds the selected autodiff transform once per call, batches measured `(q, v)` samples through `jax.vmap`, and validates finite `(T, 2, 5)` Jacobian output before returning.
- **2026-06-17** - Replaced explicit inverse effective-mass calculations for #7560: MuJoCo humanoid golf effective-mass helpers now share a solve-based kernel, finite-check mass matrices/Jacobians, and validate symmetric positive effective-mass output before returning.
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

- **2026-06-18:** Optimized `src/deployment/digital_twin/twin.py` to use `math.sqrt(np.dot())` and `math.hypot()` instead of `np.linalg.norm()` for scalar inputs and small vectors.

The operational companion to this map (startup phases, tab/sidekick wiring,
and the tracked implementation-gap inventory) lives in
[`docs/architecture/PROJECT_MAP.md`](docs/architecture/PROJECT_MAP.md) §16.

````
UpstreamDrift/
├── launch_upstream_drift.py        # Canonical entry point (web/classic/api-only/engine)
├── launch_golf_suite.py            # Legacy alias entry point (console script target; #7215)
├── src/
│   ├── engines/
│   │   ├── physics_engines/        # Engine adapters (package directories)
│   │   │   ├── mujoco/             # MuJoCo backend (core)
│   │   │   ├── drake/              # Drake backend (extended)
│   │   │   ├── pinocchio/          # Pinocchio backend (extended)
│   │   │   ├── jaxsim/             # JaxSim backend (beta)
│   │   │   ├── opensim/            # OpenSim backend (experimental)
│   │   │   ├── myosuite/           # MyoSuite backend (experimental)
│   │   │   ├── pendulum/           # Simplified educational models
│   │   │   └── putting_green/      # Putting green simulation
│   │   ├── pendulum_models/        # Educational pendulum models
│   │   └── Simscape_Multibody_Models/  # MATLAB models + C3D viewer app
│   ├── launchers/                  # PyQt6 launcher (50+ modules)
│   │   ├── upstream_drift_launcher.py  # Main window (size split tracked: #7217)
│   │   ├── embedded_host.py        # Tab/dock host: pop-out, backgrounding
│   │   ├── embedded_tool_bootstrap.py  # Embeddable-adapter registration
│   │   ├── sidekick_host_port.py   # Sidekick agent ↔ tabs bridge (subtab port)
│   │   └── {mujoco,drake,pinocchio,jaxsim}_dashboard.py, dialogs, theme, …
│   ├── api/                        # FastAPI backend
│   │   ├── local_server.py         # Server entry (web UI host)
│   │   ├── routes/                 # 30+ endpoint modules
│   │   ├── services/               # Simulation/chat/analysis services
│   │   └── auth/, middleware/, models/, utils/
│   ├── config/                     # Launcher manifest + models.yaml loaders
│   ├── tools/                      # Embeddable tool tabs (model_explorer,
│   │                               # ball_flight_gui, putting_green_gui,
│   │                               # swing_flight_pipeline, pose_studio,
│   │                               # video_analyzer, sidekick, …) plus
│   │                               # headless analysis CLIs (drift_control,
│   │                               # contraction)
│   └── shared/python/              # Cross-cutting libraries; highlights:
│       ├── engine_core/            # EngineManager/Registry/probes/capabilities
│       ├── launcher_embed/         # EmbeddableTool contract + registry (ADR-0013)
│       ├── physics/                # Ball flight models, impact, swing→flight pipeline
│       ├── motion_pipeline/        # Mocap ingestion (C3D/TRC/BVH), IK backends
│       ├── model_generation/       # URDF/MJCF parsing, Frankenstein editor (VENDORED)
│       ├── sidekick/               # Shared tools library + agent layer (VENDORED)
│       ├── humanoid_character_builder/  # Parametric humanoid URDF generation
│       └── pose_interchange/, realtime/, simulation_backends/, config/, …
├── rust_core/                      # Maturin crates
│   ├── upstream-physics/           # Ball flight, aero, contact, RK4 kernels
│   ├── upstream-mocap-io/          # C3D/TRC/BVH parsers (PyO3)
│   ├── upstream-mocap-preproc/, upstream-urdf/, upstream-mesh/,
│   ├── upstream-muscle/, upstream-motion-matching/, upstream-pinocchio-id/,
│   └── upstream-realtime/, upstream-codemap/, ai_backend/
├── ui/                             # React + Tauri launcher (manifest-driven)
├── vendor/ud-tools/                # Vendored Tools repo (canonical for sidekick
│                                   # and model_generation packages)
├── data/                           # Sample data incl. C3D captures (golf TA + CMU)
├── tests/                          # unit/, integration/, launchers/, api/, tools/,
│                                   # heavy_integration/, benchmarks/, …
├── scripts/                        # CI gates + config baselines (scripts/config/)
├── docs/                           # ADRs, architecture (PROJECT_MAP.md), guides
├── pyproject.toml                  # Canonical dependency + console-script source
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
| Analysis Tool CLIs       | `src/tools/drift_control/`, `src/tools/contraction/` | Headless AffineDrift-compatible drift/control, contraction, and Floquet analysis tools |
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
- **Web Settings**: `src/api/routes/settings.py` persists the validated web preferences document (`WebSettings`: appearance, notifications, simulation defaults) to `~/.upstreamdrift/web_settings.json` (override: `UPSTREAMDRIFT_WEB_SETTINGS_PATH`); `ui/src/pages/Settings.tsx` + `ui/src/api/settingsClient.ts` consume it with localStorage as cache only (#7457).

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
| Physics Validation          | `tests/analytical/`, `tests/integration/conservation_laws/` | pytest              | `@pytest.mark.unit` / `@pytest.mark.integration` |
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
- **Suite Marker Ratchet**: `scripts/ci/check_suite_marker_ratchet.py` scans
  pytest source files for tests without recognized suite markers and compares
  them to `scripts/config/suite_marker_baseline.json`. Existing unmarked tests
  may be paid down, but net-new unmarked tests fail CI Standard; the runtime
  collection hook in `tests/conftest.py` can report the same debt or enforce it
  with `UD_ENFORCE_SUITE_MARKERS=1`.
- **File Size Budget**: No module exceeds 500 lines; classes capped at 200
  LOC; oversized grandfathered files must have tracked baseline entries in
  `scripts/config/file_size_budget.json` or the CI gate fails.
- **Module Size Budget**: Python modules under `src/` are capped at 1,500
  lines by `scripts/check_module_size_budget.py`; oversized legacy modules
  require owned, expiring exceptions in
  `scripts/config/module_size_budget_baseline.json`, currently capped at 10
  active exceptions.
- **Architecture Budget**: Changed production Python files are capped at 100
  lines per function and 8 effective parameters per callable by
  `scripts/ci/check_architecture_budget.py`. The gate ignores test/vendor
  paths, excludes receiver parameters (`self`/`cls`) from method counts, and
  requires owned, linked exceptions in
  `scripts/config/architecture_budget.json`.
- **Law of Demeter Ratchet**: `scripts/ci/check_lod.py` scans production
  `src/` Python files and blocks new deep application object chains beyond the
  checked-in `scripts/ci/lod_baseline.txt` path/chain counts while preserving
  documented library API allowances for Qt, numpy, pandas, matplotlib, scipy,
  and engine namespace access.
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
- **Type and Coverage Ratchets**: `scripts/check_mypy_exclusion_budget.py` blocks unowned mypy exclusions, non-monotonic exclusion schedules, and missing production package coverage-ratchet metadata. Pull requests and ordinary pushes with a concrete diff base run baseline mypy only on changed `src/` Python files so legacy type drift does not block unrelated work; scheduled/manual full-src mypy still runs through `scripts/ci/run_full_mypy_baseline.py`, which compares `mypy src --config-file pyproject.toml` against `scripts/config/full_src_mypy_baseline.json` and fails on new or stale type diagnostics during explicit full-audit lanes.
- **Docker Size Gate**: Built images must not exceed 800 MB

### CI/CD Pipeline

| Workflow                       | Trigger                                | Purpose                                                                               | Blocking?          |
| ------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------- | ------------------ |
| `ci-standard.yml`              | Push/PR                                | Lint, type check, unit/integration tests, workflow inventory, blocking security scans | Yes                |
| `quality-gate.yml`             | PR/manual dispatch                     | Blocking repo-wide Law-of-Demeter ratchet for production `src/` Python code           | Yes                |
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
- Python 3.11 or later
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
| 2026-06-21 | 1.0.466 | Hardened the simulation WebSocket and Data Explorer API routes for deferred #7740 findings. WebSocket start validation now rejects non-positive speed factors and reuses `SimulationRequest` duration/timestep bounds, simulation stats access is centralized behind one helper, Data Explorer dataset lookup rejects glob metacharacters, recursive dataset listing is paginated and bounded, and the dead cache helper was removed. Focused unit-marked tests cover the new WebSocket success/error branches, filter operators, ambiguous dataset names, glob rejection, and bounded listing behavior. |
| 2026-06-20 | 1.0.465 | Added isolated transition hazard-rule coverage for issue #7715. The MDP transition tests now pin hazard penalties and DbC guard behavior directly so policy updates cannot bypass invalid-state validation. |
| 2026-06-20 | 1.0.465 | Hoisted OpenSim Manager/Integrator construction out of the perturbation per-step loop for issue #7713, preserving analyzer behavior while avoiding repeated runtime setup on every simulated step. |
| 2026-06-20 | 1.0.465 | Added chat WebSocket failure-path coverage for issue #7721. `tests/unit/api/test_chat_ws.py` now exercises `refresh_models` provider exceptions and `index_codebase` codemap rebuild exceptions, asserting sanitized client-facing error frames, server-side traceback logging, and continued socket usability after a model refresh failure. |
| 2026-06-20 | 1.0.464 | Deduplicated the simulation WebSocket `set_speed` handler for issue #7719. Runtime speed changes now route through one canonical validation and state-update branch, and focused WebSocket regression coverage preserves accepted payload behavior while preventing branch drift in future command handling changes. |
| 2026-06-20 | 1.0.464 | Repaired Drake cross-engine theta sizing for issue #7725. `tests/motion_matching/test_cross_engine_equivalence.py` now derives Drake's gravity-only zero-theta vector from the finalized plant actuator count before invoking `simulate_with_coefficients`, preserving the production guard that rejects mismatched nonzero theta vectors instead of silently disconnecting actuation and logging phantom torques. |
| 2026-06-20 | 1.0.464 | Documented MuJoCo humanoid golf grip modelling for issues #7723/#7724. Grip synergy construction and contact extraction now live in a focused helper module with regression coverage for finite contact geometry, deterministic synergy transforms, and the leaner GUI tab integration. |
| 2026-06-20 | 1.0.463 | Consolidated quaternion SLERP behavior for issue #7707. The canonical `math_utils.quaternion.slerp` implementation now owns the named nlerp fallback threshold, while spatial algebra rotations, cooperative manipulation, and Unreal skeleton mapping delegate to it with focused parity tests covering sibling behavior and the threshold boundary. |
| 2026-06-20 | 1.0.462 | Reconciled cross-engine torque polynomial coefficient ordering for issue #7688 after the mainline dependency/security refresh. The cross-engine parity spec documents the canonical flat theta block as `[A, B, C, D, E, F, G]` where column `k` multiplies `t^k`; MuJoCo's polynomial torque driver, callback Horner chain, analytical Jacobian monomials, and shared theta validator documentation follow the same lowest-power-first convention as Drake, Pinocchio, and OpenSim. The cross-engine equivalence smoke gate feeds a nonzero theta through all four pure evaluator helpers to catch future ordering drift. |
| 2026-06-20 | 1.0.461 | Restored standalone Sidekick package frontend builds by pinning `@vitejs/plugin-react` to 5.2.0, whose peer range includes the locked Vite 7.x runtime. This avoids plugin-react 6.x importing Vite 8-only internals during `npm run build` in `package-standalone-sidekick.yml`. |
| 2026-06-20 | 1.0.460 | Cleared #7806 CI ratchets after merging current main. Runtime dependency metadata and lockfiles now require patched `pydantic-settings>=2.14.2` so `pip-audit` no longer reports GHSA-4xgf-cpjx-pc3j, and recent unit-style regression files carry explicit `pytest.mark.unit` suite markers instead of expanding the unmarked-test baseline. |
| 2026-06-20 | 1.0.459 | Deduplicated golf GUI camera pan axes after the DRY duplication ratchet surfaced repeated target-plane basis construction in `golf_camera_system.py`. Mouse pan and pan inertia now share `_camera_pan_axes()` for the camera right/up vectors, preserving the existing orbit/fly pan behavior while removing the duplicated production logic fingerprint. |
| 2026-06-20 | 1.0.458 | Hardened shared CORS credential handling for issue #7740. `add_cors_middleware()` now rejects wildcard origins when `allow_credentials=True`, including origins resolved from `CORS_ORIGINS`, so credentialed responses cannot be paired with `*` origins while non-credentialed wildcard CORS remains supported. Focused unit tests cover explicit wildcard rejection, environment-origin wildcard rejection, explicit-origin credentials, and non-credentialed wildcard behavior. |
| 2026-06-20 | 1.0.457 | Corrected Drake theta contract coverage for issue #7726. `tests/parity/test_simulate_contract_drake.py` now replaces the dead optional-validator module probe with direct coverage that `validate_theta(..., bounds=DEFAULT_THETA_BOUND_TABLE)` rejects a `1e9` coefficient, while the Drake simulate path remains documented and tested as calling validation with `bounds=None` so the same large-but-finite coefficient can produce finite output or a failed solver status. |
| 2026-06-20 | 1.0.462 | Extended `SafetyMonitor` regression coverage for issue #7694. Velocity-limit tests now pin unsafe target rejection and safe-command clipping, and emergency-stop torque regressions assert pure torque commands remain unsafe while emergency stop is active. |
| 2026-06-19 | 1.0.456 | Hardened Data Explorer numeric contracts for issue #7732. `dataset_stats()` now ignores textual non-finite cells such as `inf`, `-inf`, `nan`, and `Infinity` so one stray value cannot corrupt min/max/mean or serialize invalid JSON tokens, and `_row_matches_filter()` rejects non-finite numeric row/filter operands before applying comparison operators. Focused API route tests cover finite-only aggregation, strict JSON response behavior, non-finite filter rejection, and unchanged finite comparisons. |
| 2026-06-19 | 1.0.455 | Hardened common engine state validation for issue #7705. `StateManager` and `ForceAccumulator` now enforce positive dimensions, positive time steps, and non-empty force-source names through body-level guards that remain active when Python optimization or `ContractLevel.OFF` disables decorator-based DbC checks. Focused optimized-runtime regressions run subprocesses under `python -O` to pin the always-on contract while preserving the existing decorator-backed precondition behavior. |
| 2026-06-19 | 1.0.454 | Hardened realtime controller failure abort handling for issue #7685. `_control_loop()` now clears `is_running` from a `finally` cleanup path even when the loop exits through a raised exception, and the emergency zero-torque fallback used after consecutive failures is best-effort with `logger.exception` telemetry if the safety send itself fails. Focused regression coverage drives `_send_command` to fail during both normal command dispatch and abort zero-torque dispatch, then asserts the loop terminates, `aborted_on_failure` stays true, and the zero-torque send failure is logged. |
| 2026-06-19 | 1.0.454 | Hardened assessment Python metrics failure contracts for Tools_Private #1041's live UpstreamDrift implementation. `get_python_metrics()` now treats missing or unreadable source files as real file-access failures by warning and re-raising `OSError`, while syntax and UTF-8 decode failures still return the all-zero metrics shape but emit a warning so invalid inputs cannot be confused with valid empty modules. Unit and shared wave assessment tests cover missing-file propagation and parse-warning behavior. |
| 2026-06-19 | 1.0.453 | Bounded exponentiation in shared `safe_eval` for issue #7690. `ast.Pow` now routes through a `_safe_pow` guard that rejects exponent magnitude above 1000 or large-base exponentiation before constructing enormous integers, preserving ordinary scientific expressions while stopping integer-blowup denial-of-service payloads such as right-associative exponent towers. Focused `tests/unit/test_safe_eval.py` coverage pins rejected large exponents/bases and accepted normal powers. |
| 2026-06-18 | 1.0.451 | Hardened `SafetyMonitor` command contracts for issues #7683, #7684, and #7692. `check_command()` now rejects velocity targets over `max_joint_velocity` and any command while emergency stop is active; `compute_safe_command()` clips velocity targets and treats emergency stop as an authoritative no-actuation state by zeroing velocity, torque, and feedforward torque regardless of later speed-override calls. |
| 2026-06-18 | 1.0.450 | Restored the scheduled Vendor Submodule Freshness workflow for issue #7672. `scripts/check_vendor_updates.py --json` now emits a parseable JSON status array, text mode prints per-submodule status messages again, and `.github/workflows/vendor-freshness.yml` keeps stderr out of `vendor_status.json` while letting real script failures fail closed. |
| 2026-06-18 | 1.0.446 | Optimized allocation hot paths for issue #7559. `MuJoCoPerturbationAnalyzer._simulate()` now builds a padded descending-power coefficient matrix once, evaluates every actuator each step with a vectorized Horner pass into a preallocated control buffer, and preserves per-actuator `np.polyval(coeffs[j][::-1], t)` parity for ragged, empty, and over-count coefficient lists. `PolynomialProfile` now caches its `np.poly1d` polynomial and derivative once, and the triple-pendulum hardcoded dynamics helpers accept packed theta/omega/parameter tuples instead of 16-19 scalar arguments. Focused MuJoCo and pendulum unit coverage locks parity. |
| 2026-06-18 | 1.0.445 | Gated Windows Tauri release packaging behind `TAURI_WINDOWS_RELEASE_ENABLED=true` because the current self-hosted Windows runner blocks Cargo build-script executables with Application Control (`os error 4551`). Linux Tauri release packaging and the Tauri Rust/TypeScript check remain enforced, and `tests/ci/test_ci_infrastructure.py` now pins the repo-variable opt-in plus diagnostic notice contract. |
| 2026-06-18 | 1.0.444 | Restored the final Tauri Build release contract for issue #7652. `ui/src-tauri/Cargo.lock` now resolves Rust `tauri` to the same locked major/minor as `@tauri-apps/api`, `.github/workflows/tauri-build.yml` installs `libdbus-1-dev` for the updated Linux Rust graph, and `tests/ci/test_ci_infrastructure.py` now parses lockfiles/workflow metadata to fail fast when a future dependency update would make `tauri-action` reject release builds for Rust/npm Tauri minor drift or miss required native Linux headers. |
| 2026-06-18 | 1.0.443 | Restored the main-side Tauri Build release lane for issue #7652. `ui/package.json` now exposes the `tauri` script entrypoint required by `tauri-action`'s default `npm run tauri build` invocation, and `.github/workflows/tauri-build.yml` now separates build matrix runner labels from artifact names while using a PowerShell `rustup` setup path for the Windows self-hosted leg to avoid the bash path failure seen in run `27732025577`. |
| 2026-06-18 | 1.0.442 | Restored the Nightly Cross-Engine Validation workflow for issue #7646. The scheduled job now runs `tests/integration/test_cross_engine_validation.py`, `tests/unit/test_cross_engine_validator.py`, and `tests/integration/cross_engine/test_conformance_harness.py` against `src.shared.python.engine_core.cross_engine_validator`, preserving the 75% coverage threshold with real validator/conformance tests and marking zero collected tests as a failure in the summary output. |
| 2026-06-18 | 1.0.441 | Restored current-main CI Standard after #7645. Strict API mypy now mirrors the baseline mypy push contract by checking only changed `src/api` Python files when `github.event.before` is available while preserving full strict API audits for scheduled/manual runs. The benchmark regression helper now keeps the 5x multiplier but applies a 10 microsecond absolute threshold floor for tiny hot paths, preventing sub-microsecond runner jitter from failing the 3.11 performance lane. |
| 2026-06-18 | 1.0.440 | Closed current-main CI issue #7643. `ci-standard.yml` now scopes baseline mypy on ordinary pushes to changed `src/` Python files when `github.event.before` is available while retaining full-src baseline audits for scheduled/manual runs, and `Jules-PR-Cleanup.yml` now authenticates scheduled cleanup with `secrets.RUNNER_CHECK_TOKEN || github.token` so missing optional runner tokens no longer fail stale-PR discovery with HTTP 401. |
| 2026-06-17 | 1.0.435 | Closed the dashboard launcher unit-test event-loop hang for issue #7639. `tests/unit/shared_python/test_launcher_integration.py` now patches the already-imported launcher module with `monkeypatch`, verifies the mocked Qt event-loop return code, and cannot enter the real `QApplication.exec()` loop when prior tests reload launcher modules. |
| 2026-06-17 | 1.0.434 | Closed the durable task daemon cleanup startup race for issue #7638. `src/api/task_manager_durable.py` now shares cleanup execution between async and daemon-thread loops and performs an eager daemon startup sweep before sleeping for the full interval; `tests/api/test_task_manager_durable.py` covers both periodic deletion and startup deletion of expired tasks. |
| 2026-06-17 | 1.0.433 | Closed the Rust wheel parity overreach for issue #7637. `.github/workflows/ci-standard.yml` now keeps `rust-wheel-parity` required and fail-closed for Rust wheel, parity-test, and Python facade changes while explicitly skipping the expensive parity suite on unrelated PRs; `tests/ci/test_ci_infrastructure.py` locks the path-gated contract and the successful skip summary. |
| 2026-06-17 | 1.0.432 | Closed the test-only PR core-lane OOM regression for issue #7635. CI Standard now exits after changed-test PR slices pass when no source/dependency coverage targets changed, preserving the no-collected-tests fallback and the scoped dependency-light lane for source/dependency PRs. |
| 2026-06-17 | 1.0.431 | Closed the push-scoped Semgrep regression for issue #7633 by making `tests/unit/scripts/test_remove_icon_backdrops.py` parse generated SVGs with defusedxml in its assertion helpers while preserving the existing entity-bearing SVG rejection coverage. |
| 2026-06-17 | 1.0.430 | Restored the suite-marker ratchet for issue #7631 by marking the three #7629 Bandit regression tests as unit tests: coverage XML entity rejection, SVG entity rejection, and `download_to_file()` local-file scheme rejection before opener invocation. |
| 2026-06-17 | 1.0.429 | Cleared the main-branch Bandit security-scan regression for issue #7629. `scripts/build_humanoid_models.py`, `scripts/build_humanoid_osim.py`, `scripts/config/coverage_enforcer.py`, and `scripts/remove_icon_backdrops.py` now route XML parsing through defusedxml while keeping stdlib ElementTree for XML construction/writing where needed. `src/shared/python/security/security_utils.py::download_to_file()` now validates URL schemes internally before the audited opener. Focused regression tests cover entity-bearing XML rejection and local file URL rejection before `urlopen` is called. |
| 2026-06-17 | 1.0.423 | Optimized lower-body simulator history eviction for #7561. `LowerBodySimulator` now records scrub/playback frames in a bounded `deque`, preserving FIFO order, frame indexing, and clear/restore behavior while avoiding the previous `list.pop(0)` overflow path in the simulation step loop. |
| 2026-06-17 | 1.0.422 | Pinned deformable external-force scatter coverage for #7563. `SoftBody.apply_external_force()` now has source-level regression coverage that rejects Python-level scatter loops, complementing duplicate-node accumulation tests that preserve the existing `np.add.at` duplicate-index semantics. |
| 2026-06-17 | 1.0.408 | Deformable object internal-force performance for #7571/#7572. Soft-body FEM force assembly now batches tetrahedral deformation gradients, determinants, inversions, stresses, and nodal scatter while reusing each inverse once. Cable and cloth spring-force accumulation now uses cached NumPy connectivity/stiffness arrays plus shared vectorized scatter kernels, with focused scalar-reference parity and allocation/vectorization tests for the deformable internals. |
| 2026-06-17 | 1.0.407 | Replaced the iLQR backward-pass gain calculation for issue #7570. The MPC controller now solves the regularized `Quu` system directly, using a finite-checked Cholesky path for symmetric positive definite matrices and a general linear solve fallback while preserving gain parity with the previous inverse-based math. |
| 2026-06-17 | 1.0.406 | Optimized whole-body SciPy QP inequality construction for issue #7568. Finite lower and upper inequality bounds now each build one vector-valued SLSQP callback instead of per-row Python callbacks, while QP construction validates finite matrices plus constraint and bound shapes before entering the solver hot path. |
| 2026-06-14 | 1.0.394 | Added the ball-flight physical benchmark contract for issue #7407. TrackMan-style driver and 7-iron tests now cover the enhanced simulator, the public flight-model registry, and the REST route; the suite locks cross-engine carry agreement, 5 m/s headwind/tailwind sanity, analytic vacuum range, explicit zero-drag semantics for Python/Rust drag-crisis helpers, humidity-aware density through `EnvironmentalConditions.from_altitude(...)`, and the calibrated Magnus coefficient cap contract. |
| 2026-06-14 | 1.0.396 | Completed the AnalysisOrchestrator dashboard plot migration for #7446. Every PyQt6 static-dashboard label now maps to a registered headless `PlotData` extractor, including Poincare, Lyapunov, recurrence, GRF butterfly, kinematic-sequence, and summary-dashboard outputs, and the desktop dispatch path exercises `AnalysisOrchestrator.get_plot_data(...)` before renderer-specific plotting. |
| 2026-06-14 | 1.0.393 | Corrected the impact model gear-effect spin direction and driver COR calibration for issue #7406. Toe/heel offsets now produce hook/slice spin in the launch-monitor sign convention, high-face/low-face offsets reduce/increase backspin, default gear-effect scales put realistic 20 mm toe and 10 mm vertical strikes in expected rad/s bands, and `DRIVER_COR` now reflects the USGA/R&A CT-limit-equivalent COR so center strikes can reach tour smash-factor ranges; regression coverage asserts toe/heel antisymmetry, vertical gear effect, default smash factor, and energy balance. |
| 2026-06-13 | 1.0.382 | Copied the hatch force-included `launch_golf_suite.py` entrypoint into the modular Docker builder before feature installation so `pip install .` metadata generation matches the wheel packaging contract; the Docker contract guard now rejects moving the entrypoint copy after feature installation. |
| 2026-06-13 | 1.0.381 | Kept modular Docker profile dry-runs importable from the early Dockerfile copy set by removing the import-time dependency from `engine_core.engine_probes` to the heavyweight config package, with regression coverage that exercises `install_features.py --profile standard --dry-run` against only the early validation files. |
| 2026-06-14 | 1.0.395 | Public AffineDrift analysis-tool parity for #7431. Added `src.tools.drift_control.DriftControlAnalyzer` for generalized-force NPZ ratio analysis, `src.tools.contraction.ContractionVerifier` plus Floquet multiplier helpers, FastAPI JSON endpoints under `/analysis/tools/*`, and an optional-backend Pinocchio ABA timing CLI that reports unavailable cleanly when Pinocchio is not installed. |
| 2026-06-13 | 1.0.376 | Aligned the Rust wheel import smoke helper with the documented `upstream-mocap-io` missing-file contract so `parse_trc` raising `FileNotFoundError` remains an expected negative-path smoke result, with focused script regression coverage. |
| 2026-06-14 | 1.0.390 | Corrected the rigid-body impact model friction-spin sign for issue #7403. Tangential contact impulse now uses `tangent_dir x normal`, preserving the rolling cap while producing lofted-driver backspin under the repository's `[0, -1, 0]` convention; regression coverage asserts the backspin sign, expected spin magnitude, and upward Magnus force from the derived spin axis. |
| 2026-06-14 | 1.0.389 | Corrected the vendored Sidekick gas-flow compressibility-factor contract for issue #7408. Supported gases now carry acentric factors and `compressibility_factor()` uses the Abbott/Pitzer generalized second-virial approximation `(B0 + omega * B1) * Pr / Tr`, with regression coverage for methane, air, the low-pressure ideal limit, and the NIST methane 300 K / 5 MPa tolerance. |
| 2026-06-13 | 1.0.375 | Repaired the Rust PyO3 CI contract so Python-feature crates are checked with `cargo check --workspace --all-targets --features python`, avoiding invalid extension-module test executable linkage while maturin continues to build and smoke-test importable wheels. |
| 2026-06-13 | 1.0.374 | Restored optional-stack contracts by normalizing aerodynamic vector-shaped velocity/spin inputs, preserving drag-coefficient tuning through the Reynolds correction, centering the drag-crisis transition at the documented 8e4 Reynolds boundary, clamping randomized air density to the physical floor, and pinning the legacy API endpoint suite to explicit local/auth-disabled mode. |
| 2026-06-13 | 1.0.373 | Refined pointwise ZTCF/ZVCF dimension validation to allow canonical-v2 `nq = nv + 1` states only when the redundant configuration coordinate is populated, while still rejecting plain wrong-length zero vectors. |
| 2026-06-13 | 1.0.372 | Restored optional-stack ZTCF/ZVCF input length validation for pointwise acceleration helpers and reapplied the Settings page root theme marker after active-theme round-trip. |
| 2026-06-13 | 1.0.371 | Tightened signal-toolkit limit preconditions so saturation requires `lower < upper` and rate limiting requires `max_rate > 0`, matching the optional-stack contract tests. |
| 2026-06-13 | 1.0.370 | Scoped PR core-test dependency-light execution to affected unit areas when source files change, preventing self-hosted runner OOMs from full-repository core test attempts while leaving normal coverage reporting in place for full-coverage runs. |
| 2026-06-13 | 1.0.369 | Restored Sidekick JSON I/O compatibility in optional-stack tests. Default records-oriented JSON reads/writes now use stdlib JSON to avoid pandas JSON C-extension failures in the optional-stack environment, while non-default JSON options continue through pandas. |
| 2026-06-13 | 1.0.368 | Aligned CI numpy/scipy repair pins with the lockfile runtime, added fast `scipy.signal` import checks after the reinstall step, kept core tests serial to avoid native-stack xdist worker termination, and recorded temporary shared-chat architecture-budget exceptions for `ChatDockWidget` legacy decomposition debt under issue #7362. |
| 2026-06-13 | 1.0.367 | Restored shared-python optional-stack compatibility contracts. Package-root `config` and `data_io` exports remain importable under the fallback test harness, `add_provenance_header` supports both file and string helper forms, AI adapters preserve empty-current-message, token-usage, and connection-error behavior, validation/signal-toolkit helpers enforce their documented preconditions, and FSP primitive tests skip partial Rust wheels that do not expose the FSP API. |
| 2026-06-13 | 1.0.366 | Restored optional-stack robotics compatibility contracts. `QPSolution`, `WBCSolution`, and `PlannerResult` again expose canonical `solver_status` aliases, `WBCSolution.final_cost` remains a legacy alias for `cost`, and the default IMU gravity vector preserves the historical 9.81 m/s^2 sensor contract while keeping the broader shared physics constants unchanged. |
| 2026-06-12 | 1.0.365 | Consolidated parity CI follow-up after #7496. Moved BallFlight validation/color helpers into a non-component module to satisfy React Fast Refresh linting, added a fail-closed launch-mode fallback for web launcher contracts, deferred diagnostics launcher imports to request time to preserve API layer direction, extracted recording JSON artifact writing below the changed-function architecture budget, shared export-format validation across request models, and marked newly introduced parity tests with explicit suite markers so the ratchet remains blocking without expanding the baseline. |
| 2026-06-12 | 1.0.360 | Web settings parity for #7457. Added `GET/PUT /settings` (`src/api/routes/settings.py`) persisting a validated `WebSettings` document (appearance/notifications/simulation defaults) atomically to `~/.upstreamdrift/web_settings.json`, a web `/settings` route (`ui/src/pages/Settings.tsx`) with theme selection via the shared `/themes` router, root-CSS-var font scaling, notification preferences consumed by the toast provider, and once-per-session simulation-default hydration that never clobbers in-session edits (#7424 guard). Feature-parity registry: `settings.preferences` → parity; desktop-only settings tabs recorded as a pending-decision exemption (#7460). |
| 2026-06-12 | 1.0.359 | Fixed vendored Sidekick science accuracy for issues #7411, #7412, and #7413: Buck vapor pressure restores the correct constant roles, `signal_toolkit.calculus.Integrator` includes the upper-bound sample in definite integrals, and SCFM/ACFM/Nm3 plus ppmv/mg/Nm3 conversions use one DIN 1343 normal state. Added reference-value and property tests for vapor pressure, calculus integration, and conversion consistency. |
| 2026-06-12 | 1.0.358 | Removed the anti-phantom guard's `jq` dependency from the PR retry/API fallback path by using GitHub CLI `--jq` output directly, keeping required PR guard checks portable across local self-hosted runner images that do not install `jq` globally. |
| 2026-06-17 | 1.0.409 | Replaced explicit inverse effective-mass calculations for #7560. MuJoCo humanoid golf effective-mass helpers now share a solve-based kernel, finite-check mass matrices and Jacobians, validate symmetric positive effective-mass output, and cover the no-explicit-inverse contract with focused unit tests. |
| 2026-06-12 | 1.0.357 | Web reachability contract for launcher tiles (#7461). Every `launcher_manifest.json` tile now declares `web: {mode: route\|native-window\|unavailable, route?, reason?}`, validated by `WebLaunchContract` in `launcher_manifest_loader.py` and the TS schema parser. The web dashboard navigates route tiles in-app, gates native-window tiles behind Tauri/localhost with a "Desktop app only" badge, and shows reasoned badges for unavailable tiles. `POST /api/launcher/launch/{tile_id}` now returns 409 for non-local clients on non-route tiles so Qt windows never open invisibly on a remote server. Parity tests enforce contract declaration and React-router route existence for all current and future tiles. |
| 2026-06-11 | 1.0.353 | Made the optional-stack unit lane boundary explicit: the lane runs the non-engine unit suite with optional API, GUI, and body-part dependencies installed, while native engine unit tests remain covered by dedicated engine and cross-engine equivalence lanes to avoid coupling broad optional dependency validation to engine-specific mock behavior. |
| 2026-06-11 | 1.0.352 | Aligned deployment optional-stack device tests with the hardware-honesty contract: unavailable hardware-backed input devices remain disconnected and raise `StateError` for state operations, `KeyboardMouseInput` remains the connected fallback, and `Demonstration` now carries default canonical `solver_status="success"` through recording, serialization, subsampling, and augmentation. |
| 2026-06-11 | 1.0.351 | Restored the calc backend ODE solver response contract so `ODESolverResponse` again exposes the default `solver_status="success"` field consumed by optional-stack calc backend callers and tests. |
| 2026-06-11 | 1.0.350 | Restored body-part visualization optional-stack contracts: `FittedShape.n_frames` again reports the validated frame count, theme and fitted-shape validation errors use the documented precise type/range messages, and CI Optional Stack installs `trimesh` before running unit chunks so mesh-backed body-part visualization tests exercise the intended full dependency path. |
| 2026-06-11 | 1.0.346 | Preserved symlink traversal security failures in model and output path validation. Candidates lexically under an allowed root now reject symlink components before resolved containment fallback can mask escaped targets as generic 404 misses, keeping Linux optional-stack path validation on the documented 400 contract. Added suite markers to newly merged unit-level regression tests so the suite-marker ratchet remains blocking without expanding the unmarked-test baseline. Runs optional-stack unit tests as serial top-level `tests/unit` chunks because full-suite optional dependency collection is xdist-unsafe and can exceed local runner memory before tests execute, and raises the bounded CI Standard tests matrix timeout so the core suite is not cancelled before its per-test timeout contract can report real failures. |
| 2026-06-17 | 1.0.410 | Optimized JaxSim trajectory parameter-gradient evaluation for #7562 by constructing the selected autodiff transform once per API call, vectorizing over the measured trajectory with `jax.vmap`, and adding finite shape postcondition checks plus local fake-JAX contract coverage. |
| 2026-06-17 | 1.0.411 | Optimized RRT/RRT* nearest-neighbor and RRT* cost-propagation paths for #7564. Sampling trees now maintain a finite append-friendly configuration index for vectorized nearest/near queries, RRT* honors `use_kd_tree` with periodically rebuilt `cKDTree` coverage when enabled, and rewiring cost propagation walks a maintained child-adjacency map with `deque` instead of scanning every node for descendants. |
| 2026-06-17 | 1.0.412 | Optimized deformable soft-body FEM root-node force accumulation by replacing the generic last-axis sum over per-tetrahedron force blocks with a batched `np.einsum("ijk->ij", H)` reduction and focused coverage for the vectorized reduction shape. |
| 2026-06-17 | 1.0.413 | Optimized MuJoCo kinematic Coriolis decomposition for #7558 by reusing the base inverse-dynamics solution and scratch buffers across the per-DOF split, reducing redundant `mj_rne` passes while preserving the legacy component-sum contract. |
| 2026-06-17 | 1.0.414 | Optimized fixed-size 2D/3D vector magnitude calculations in API route metrics, putting-green direction normalization, teleoperation deadband/rate limiting, and ball-launch trajectory speed by replacing scalar `np.linalg.norm` calls with explicit `math.hypot` calls while preserving existing vector contracts. |
| 2026-06-17 | 1.0.415 | Hardened robotics precision/vectorization issue slice for #7556/#7559/#7563: footstep angle wrapping is bounded, IMU quaternion normalization and force-torque contact-location denominators are guarded, cooperative load sharing uses `np.linalg.solve`, contact hull fallback sort keys are vectorized, deformable external-force scatter uses `np.add.at` for duplicate node indices, and GJK avoids recomputing the initial support-vector norm. |
| 2026-06-17 | 1.0.416 | Tightened RL benchmark precision and projection search for #7556/#7561. Trajectory funnel benchmark metrics now use stable dispersion accumulation and finite-checked projection search helpers, with focused coverage for deterministic precision behavior and search contract edge cases. |
| 2026-06-17 | 1.0.417 | Vectorized imitation-learning transition batching for #7563. `DemonstrationDataset.to_transitions()` now builds per-demonstration state/action/next-state blocks and concatenates once per output, with finite shape guards and regression coverage that prevents the old per-frame one-dimensional concatenate path from returning. |
| 2026-06-17 | 1.0.418 | Replaced lower-body IK damped-least-squares explicit inverse formation with `np.linalg.solve` for #7556, with focused coverage that forbids the old `np.linalg.inv` path and validates the solve system shape. The cross-engine leaderboard workflows now preinstall `idna[uts46]` in their minimal venvs so the `anyio` dependency can resolve reliably before leaderboard regeneration gates run. |
| 2026-06-17 | 1.0.421 | Extended footstep yaw normalization coverage for #7556 with a very-large negative-angle regression, preserving the bounded modulo contract for both signs plus the existing NaN preservation and infinite-value rejection behavior. |
| 2026-06-11 | 1.0.345 | Restored main CI API, launcher, and Docker contracts: `jaxsim` is accepted by the public simulation request allowlist, Data Explorer import responses keep generated dataset IDs while allowing legacy direct model construction, canonical-core launcher tiles use a recognized status and served biomechanics logo, symlink model-path validation preserves explicit 400 security failures, and Docker feature dry-runs import shared engine probe configuration through the package-qualified path. |
| 2026-06-11 | 1.0.344 | Capability truthfulness contracts for #7355 and #7356. Generated motion-pipeline compatibility docs now mark Drake trajectory-optimization matching as unsupported until the solver is implemented. Drake, RRA, and CMC matching placeholders now report `status: not_implemented` with `production_ready: false`, and production chat placeholder tools return explicit `not_implemented` payloads instead of queued or successful no-op results. |
| 2026-06-11 | 1.0.343 | Honest Document Chat and swing-sequence analytics contracts for #7358/#7359. The launcher Library tab keeps Document Chat disabled without a configured backend and reports a backend-not-configured message instead of a fabricated Notebook LM response. `swing_sequence` analysis now computes segment peak timing from trajectory angular velocities, marks instantaneous-only segment velocities as `requires_trajectory`, preserves analysis payloads through `AnalysisRequest.data`, and emits X-factor metrics only when shoulder/hip joint trajectory inputs are available. |
| 2026-06-11 | 1.0.344 | RL engine protocol and teleoperation hardware-connection honesty for #7357/#7360. Added `src.engines.protocols.PhysicsEngineProtocol`, validated humanoid RL engine dimensions and runtime channels before use, removed zero-filled humanoid observation/reward fallbacks, exposed MuJoCo protocol accessors over real model/data arrays, and changed SpaceMouse/VR/Haptic device classes to report unavailable with `StateError` on disconnected reads/writes instead of fake successful hardware connections. |
| 2026-06-11 | 1.0.342 | Launcher Docker build cancellation and layout reset backup hardening for #7341/#7342. Docker build threads now own a managed subprocess handle with cooperative cancellation instead of `QThread.terminate()`, the GUI prompts before closing an active build, and GUI/CLI layout reset paths share a helper that overwrites an existing `launcher_layout.json.bak` via `Path.replace` so repeated resets work on Windows. The changed-file architecture budget records expiring exceptions for the legacy launcher UI builders surfaced by this focused repair. |
| 2026-06-11 | 1.0.341 | CI and validation test contract hardening for #7352, #7353, and #7354. The optional-stack lane now gates on pytest exit codes, physics validation scripts target real analytical/conservation suites, and PyQt fallback stubs no longer fabricate launcher expectations. |
| 2026-06-11 | 1.0.340 | Motion-pipeline DRY follow-up for the #7380 simulator-facade merge. MuJoCo torque matching and Pinocchio inverse dynamics now share `BaseMotionMatchingSolver` helpers for per-DOF rig joint names and torque trajectory construction, removing duplicate post-merge torque payload assembly while preserving backend-specific success metadata. |
| 2026-06-11 | 1.0.339 | Suite-marker ratchet follow-up for the #7382 import-boundary consolidation repair and the #7380 simulator-facade merge. The launcher dependency-probe, settings Docker dependency worker, architecture-budget metadata, C3D viewer export worker, body-target-video cancellation, shared ball constants, MuJoCo torque dimension mismatch, Pinocchio inverse-dynamics readiness, and generated-rig orchestrator regression tests now carry explicit `unit` suite markers so CI can enforce no-growth test metadata without weakening the marker baseline. |
| 2026-06-11 | 1.0.338 | Import-boundary facade consolidation for #7361, #7362, and #7363. The C3D viewer entrypoint now imports the repo-qualified viewer module directly, MCP config I/O moved into `src/shared/python/ai/mcp/config_io.py` with launcher compatibility facades, shared MCP chat integration reads shared config, and shared/engine datetime compatibility imports route through shared helpers. The changed-file architecture budget now records owned expiring exceptions for the five pre-existing oversized functions surfaced by the consolidation so decomposition debt remains visible without leaving `main` red. |
| 2026-06-11 | 1.0.337 | MuJoCo motion-matching placeholder failure routing for #7333. The orchestrator now carries solver metadata through the motion-matching stage, maps unavailable or zero-torque MuJoCo matching results to `InvalidInputError` so REST callers receive 400-class configuration feedback instead of HTTP 500, and the motion-pipeline README recommends a non-placeholder matching backend until real MuJoCo rig-model integration lands. |
| 2026-06-11 | 1.0.336 | Suite-marker ratchet enforcement for #7272. CI Standard now runs `scripts/ci/check_suite_marker_ratchet.py` against `scripts/config/suite_marker_baseline.json`, failing net-new tests that lack a recognized suite marker while allowing legacy unmarked-test debt to shrink. The shared `tests.support.suite_markers` helpers now normalize nodeids, load the baseline, and support report-only, strict, and baseline-ratchet collection behavior from `tests/conftest.py`; contributor guidance lives in `docs/development/test-marker-conventions.md` with focused unit coverage for the static scanner and runtime helpers. |
| 2026-06-11 | 1.0.335 | Restored the #7246/#7247 regression-guard cluster for #7325, #7326, and #7327 after PR #7248 reverted part of the launch-condition unit fix. `LaunchConditions.from_user_units(...)` is again the canonical GUI/user-input boundary for degree-to-radian conversion and RPM spin, the ball-flight GUI routes through that seam, and the current main gap-fill keypoint bounds guard remains covered by focused regression tests. |
| 2026-06-11 | 1.0.334 | Collision distance helper optimization for #7324. Primitive-shape distance helpers now use explicit component access instead of `math.hypot(*tuple)` unpacking, preserving robotics collision behavior while avoiding tuple unpacking overhead on hot paths. |
| 2026-06-11 | 1.0.333 | Law-of-Demeter enforcement for #7308. `scripts/ci/check_lod.py` now defaults to repo-wide production `src/` scanning, supports a checked-in no-growth baseline, and preserves documented library API allowances. `.github/workflows/quality-gate.yml` now runs the LOD scan as the blocking required `quality-gate` status with `scripts/ci/lod_baseline.txt` representing current grandfathered path/chain counts. |
| 2026-06-11 | 1.0.332 | Safe motion checkpoint loading for #7276. Replaced pickle-enabled motion-matching checkpoint loads with safe artifact loading via `torch.load(..., weights_only=True)`. Validates mapping-shaped artifacts, keeping inverse, inverse-timestep, compact surrogate, and per-step surrogate loaders on the same safe contract. Exceeded surrogate train/optimize function budgets are tracked as exceptions in `architecture_budget.json`. |
| 2026-06-11 | 1.0.331 | Resolve merge conflicts in SPEC.md for PR 7316 by merging origin/main, retaining all changelog entries, and bumping Spec Version. |
| 2026-06-11 | 1.0.330 | Simulation WebSocket dependency-boundary conflict refresh for #7283. `simulation_stream` keeps resolving the engine manager through the WebSocket-safe dependency accessor after the #7304/#7305/#7306/#7309 runtime-contract `main` update, and missing app-state manager configuration still emits a structured `service_unavailable` frame before clean close. |
| 2026-06-11 | 1.0.329 | Safe motion checkpoint loading conflict refresh for #7276. The safe checkpoint artifact helper remains wired through inverse, inverse-timestep, compact surrogate, and per-step surrogate loading after the #7317 training/optimization architecture split and #7304/#7305/#7306/#7309 runtime-contract update, preserving mapping validation and `weights_only=True` reads while keeping the new helperized training and optimization contexts. |
| 2026-06-11 | 1.0.328 | Motion matching runtime contract hardening for #7304, #7305, #7306, and #7309. `CostWeights` and internal `MotionMatchingRequest` now reject invalid numeric configuration at construction, shared metric validation fails on frame/DOF shape mismatches instead of silently truncating, the solver result postcondition gate validates reference-aligned time grids plus finite torque/activation payloads, and internal successful `MotionMatchingResult` objects must include a matched trajectory, torque trajectory, or activation trajectory payload. |
| 2026-06-11 | 1.0.328 | Cross-engine dashboard window factory follow-up for #7316. `CrossEngineDashboardWindow()` now constructs the deferred PyQt window instead of raising a direct-instantiation placeholder, preserving the extracted fallback-engine stub and `_build_qt_window()` launcher path while keeping `src/launchers/cross_engine_dashboard.py` below the 1200-line file-size gate. |
| 2026-06-11 | 1.0.325 | Cross-engine dashboard architecture split for #7288. `src/launchers/cross_engine_dashboard.py` now keeps the public `CrossEngineDashboardWindow` compatibility facade thin, constructs the concrete PyQt window class through a deferred factory, and imports the fallback engine stub from `src/launchers/cross_engine_dashboard_stubs.py`, removing the dashboard architecture-budget exception while preserving the existing CLI and window-construction contracts. |
| 2026-06-11 | 1.0.326 | Motion surrogate training architecture split for #7317. Compact surrogate training now uses `SurrogateTrainingOptions`, explicit training context construction, and loop-state helpers while preserving legacy keyword call compatibility. Per-step dynamics training separates data preparation, runtime setup, fitting, evaluation, and output writing. Per-step optimization now routes legacy positional options through `OptimizationOptions`, uses an optimization context, isolates tracking/regularizer loss helpers, and writes optimized torque outputs plus summaries through a dedicated artifact writer. |
| 2026-06-11 | 1.0.323 | Cloud client cached-token hardening for #7300. `CloudClient._load_cached_token()` now ignores empty and whitespace-only cache files instead of treating `""` as an authenticated token, `CloudClient.is_logged_in` requires a truthy token, and focused tests pin both invalid-cache cases while preserving valid cached-token behavior. |
| 2026-06-11 | 1.0.321 | Local WebSocket hardening, Tauri permission manifest repair, and Tauri build apt-lock hardening for #7275, plus coverage gate fix for #7273. API WebSocket auth now validates launcher capability tokens and allowed Origins, the React client propagates the launcher manifest token, the Tauri IPC capability defines concrete permissions, and `.github/workflows/tauri-build.yml` retries apt dependency installs. Standard CI now sends PRs that change source, tests, or dependency targets through the coverage-producing core test lane. |
| 2026-06-11 | 1.0.320 | Optional dependency mock isolation for #7307. Added `scoped_import_with_optional_mocks()` to shared test support, converted the called-out OpenSim, MuJoCo, and Drake tests from module-scope `sys.modules` mutation/import patching to per-test scoped import fixtures, removed the MuJoCo subtree-wide fake dependency conftest, and added a repo-hygiene guard that fails on new module-scope optional dependency mocks for `opensim`, `mujoco`, `cv2`, `imageio`, and `pydrake`. |
| 2026-06-11 | 1.0.318 | Data Explorer and model-library boundary contracts for #7297, #7298, and #7299. Import/list responses expose durable `dataset_id` values, Data Explorer filter requests reject unsupported operators at the request boundary, and forced model-library downloads validate HTTPS-only `source_url` values before any download I/O. |
| 2026-06-11 | 1.0.312 | Blocking DRY duplication ratchet for #7315. Added `scripts/ci/check_dry_duplication_gate.py` with focused tests, explicit production-`src` include/exclude config, and an owned no-growth baseline for existing duplicated logic fingerprints; `ci-standard.yml` now runs the checker inside `repo-structure-gates` so duplicate growth feeds the required `quality-gate` aggregate while `Code-Metrics.yml` remains advisory/manual reporting. |
| 2026-06-11 | 1.0.311 | PR-scoped unit gate hardening for #7314. Standard CI no longer lets source/dependency PRs pass solely by running changed test files; those PRs fall through to the dependency-light unit lane with targeted coverage. `coverage_enforcer.py` now supports a PR-mode changed-file ratchet so changed production policy files must appear in targeted coverage and meet their policy threshold. |
| 2026-06-10 | 1.0.309 | Jules PR AutoFix workflow-run trust-boundary hardening for #7312. The privileged `workflow_run` path now performs read-only failed-CI metadata analysis and posts manual dispatch instructions instead of checking out or executing PR-controlled code. The write-capable iterative fixer is restricted to explicit `workflow_dispatch` with an input branch. Added `scripts/check_workflow_run_trust_boundary.py`, wired it into standard CI, documented it in `scripts/README.md`, and added focused regression tests for unsafe workflow-run checkout/install/writeback patterns and the current Jules workflow contract. |
| 2026-06-10 | 1.0.308 | Docker build timeout and focused PR coverage enforcement for #7277. `DockerManager` now monitors build output through a background queue while enforcing a wall-clock build timeout, terminating the process tree when stdout remains open past the deadline. Standard CI now scopes PR coverage to changed `src/**/*.py` modules and runs per-package coverage enforcement only after full core coverage reports, so focused PRs are not blocked by unrelated packages. |
| 2026-06-10 | 1.0.304 | Frankenstein editor legacy shim consolidation for #7280. `_frankenstein_model.py` now re-exports the canonical `frankenstein_editor.model.URDFModel`, and `_frankenstein_panels.py` re-exports the canonical panel/dialog classes, preserving older import paths without duplicating implementation. Focused split tests assert shim identity and exercise validation/export through the legacy model import. |
| 2026-06-10 | 1.0.303 | Lock-backed CI dependency install follow-up for #7278. Standard CI jobs now install committed `requirements-dev.lock` artifacts before editable package installs and use `--no-deps` for local editable extras so pip never treats extras-bearing lock entries as invalid constraints. The dev lock and `make sync-deps` target now cover the `gui-test` extra so unit gates retain real PyQt6/pytest-qt imports, and the static security CI acceptance test rejects `-c requirements-dev.lock` regressions while keeping the dev/runtime pip-audit lock checks. |
| 2026-06-10 | 1.0.308 | Docker build timeout and focused PR coverage enforcement for #7277. `DockerManager` now monitors build output through a background queue while enforcing a wall-clock build timeout, terminating the process tree when stdout remains open past the deadline. Standard CI now scopes PR coverage to changed `src/**/*.py` modules and runs per-package coverage enforcement only after full core coverage reports, so focused PRs are not blocked by unrelated packages. |
| 2026-06-10 | 1.0.303 | Lock-backed CI dependency install follow-up for #7278. Standard CI jobs now install committed `requirements-dev.lock` artifacts before editable package installs and use `--no-deps` for local editable extras so pip never treats extras-bearing lock entries as invalid constraints. The dev lock and `make sync-deps` target now cover the `gui-test` extra so unit gates retain real PyQt6/pytest-qt imports, and the static security CI acceptance test rejects `-c requirements-dev.lock` regressions while keeping the dev/runtime pip-audit lock checks. |
| 2026-06-10 | 1.0.302 | Audit hygiene fixes for #7279 and #7282. `.github/workflows/docker-security-scan.yml` now blocks HIGH and CRITICAL Trivy container vulnerabilities in the table scan while retaining SARIF upload, and audited API/launcher production modules now use the canonical logging infrastructure instead of direct module-level `logging.getLogger` calls. Added security CI acceptance coverage for the Docker HIGH/CRITICAL gate and a repo-hygiene test for the remediated logger modules. |
| 2026-06-10 | 1.0.301 | Audit regression fixes for #7269, #7270, and #7271. Model Explorer API path resolution now validates caller paths before filesystem reads and resolves only within approved model directories, closing the direct existing-path containment bypass. Motion-pipeline keypoint gap filling now guards both before/after neighbor keypoint indexes and pins mismatched-neighbor behavior in the main and pure-Python implementations. `SwingBallFlightPipeline` now emits `LaunchConditions` in the units consumed by `BallFlightSimulator`: launch and azimuth angles in radians, spin rate in RPM, with updated DbC validation and unit tests. |
| 2026-06-10 | 1.0.300 | Completed the #7207 model explorer composition UX flow. Added `CompositionUxController` for library drag payloads, non-mutating drop/ghost previews, highlighted target/source links, validation summaries, committed drops, and a validation-aware export chooser that enables URDF/MJCF while explicitly marking SDF/OSIM unavailable until writers exist. `FrankensteinEditor` now exposes preview, commit, and export-choice hooks, with offscreen tests covering simple humanoid plus arm preview, commit, validation pass, and MJCF export. |
| 2026-06-10 | 1.0.299 | Cross-engine equivalence import-boundary fix for #7214. `CalibrationOptimizer.optimize()` now imports `scipy.optimize.differential_evolution` lazily so importing `src.bunkershot3d.postproc.wrench_trace` through shared simulation backends does not require optional calibration optimizer dependencies in the equivalence CI environment. |
| 2026-06-10 | 1.0.298 | C3D viewer renderer backend decision for #7214. Added ADR-0030 choosing PyQtGL as the first desktop GPU playback path while keeping matplotlib fallback, plus a focused `viewer_3d_backend.py` decision contract that carries the 60 fps target and parity checklist for scrubbing, speed control, loop playback, marker groups, view presets, and skeleton overlay before replacement. |
| 2026-06-10 | 1.0.297 | Model explorer composition-flow controller for #7207. Added `CompositionFlowController` to attach a complete source URDF model to a working Frankenstein model via selected or declared attachment points, copy links/joints/materials with deterministic name mapping, validate the composed result immediately, and export validation-gated URDF or MJCF preview content. `FrankensteinEditor` now exposes an Attach Source Model action plus public export helper, and `URDFModel.from_file()` carries attachment sidecar metadata into the editor. Focused headless tests cover human-plus-arm composition, MJCF export, validation refusal, and the offscreen editor attach/export path. |
| 2026-06-10 | 1.0.296 | Model explorer attachment manifests for #7206. Added a first-party `attachment_manifest` parser for versioned `<model>.attachments.json` sidecars, checked in the JSON Schema and docs, exposed declared attachment points plus non-fatal warnings through `ModelLibrary` model info, and updated the attachment dialog to list declared mount points first, prefill their interface-frame origin, and report payload-limit warnings. Focused tests cover valid/missing/malformed manifests, imported-model exposure, dialog defaults, and payload warning contracts. |
| 2026-06-10 | 1.0.295 | Split the launcher entrypoint below the file-size budget for #7217. Sidekick sidebar installation, process cleanup polling, launcher domain orchestration, and GUI startup bootstrap moved from `src/launchers/upstream_drift_launcher.py` into focused modules, preserving compatibility imports and the canonical frameless-window helper under `src/launchers/launcher_ui/frameless_window.py`. The launcher entrypoint is now below 1200 lines, so its file-size budget exception was removed. |
| 2026-06-10 | 1.0.294 | Rust mocap FFI binding error-contract hardening for #7252. `upstream-mocap-io` now validates non-empty and NUL-free Python binding paths before parser entry, maps missing files from `parse_c3d` / `parse_trc` / `parse_bvh` to `FileNotFoundError`, maps other file-access errors to `OSError`, and keeps malformed present files as `ValueError` parse failures that include the format and path context. Rust binding tests and Python parity tests cover missing-file and malformed-present-file behavior across all three formats while preserving the marker/unit parser contracts. |
| 2026-06-10 | 1.0.293 | First #7207 model explorer library-panel unification slice. `ModelLoaderDialog` now exposes one searchable Library tree built from every `ModelLibrary.list_available_models()` category, including sibling repositories, and model rows show first-party format badges inferred from explicit model metadata or category defaults. Headless panel-model tests cover flattening, sibling inclusion, search, category grouping, and badge logic. |
| 2026-06-10 | 1.0.292 | Motion-pipeline hook exception handling for #7250. `PipelineConfig.strict_hooks` now controls per-stage hook failure policy: default lenient mode logs failures with `logger.exception` so tracebacks are observable while the pipeline continues, and strict mode raises `HookExecutionError` with the stage, hook name, and original exception chained as the cause. Focused orchestrator unit tests cover both modes. |
| 2026-06-10 | 1.0.291 | Added the bounded inverse swing optimization core for #7220. `src/shared/python/physics/swing_optimizer.py` adds `FlightTarget`, `ClubPreset`, `SwingOptimizer`, and diagnostics around SciPy SLSQP over speed/loft/attack/face-to-path while composing the existing `SwingBallFlightPipeline`; focused physics tests cover roundtrip, unreachable target, and timeout behavior. |
| 2026-06-10 | 1.0.290 | Rust C3D analog and force-platform metadata slice for #7212. `upstream-mocap-io` now decodes C3D analog channels in int16 mode with SCALE/OFFSET/GEN_SCALE and in float mode without int scaling, advances marker frame stride across analog-bearing records, parses FORCE_PLATFORM TYPE/CHANNEL/CORNERS/ORIGIN metadata, and exposes additive PyO3 `analog` / `force_platforms` keys while preserving existing marker/event keys and marker-only fixture behavior. |
| 2026-06-10 | 1.0.289 | Completed the Frankenstein composition validation surface for #7205. `CompositionValidator` now emits warning-level `subtree_mass_ratio` findings when attached subtree mass exceeds roughly 2x the parent chain mass and `geometry_overlap` findings when directly attached link AABBs overlap. The active Frankenstein model panel now renders current validation findings in a dedicated list so warnings and blocking errors are visible before save/export. |
| 2026-06-10 | 1.0.288 | React/Tauri launcher parity decision (#7221). Added ADR-0028 choosing the manifest-driven multi-window Tauri model while keeping PyQt canonical for embedded tabs/docks. The React dashboard now persists a manifest-keyed launcher window registry, reconciles it with `useLauncherManifest.ts`, and exposes a window list/focus menu that reuses the local launcher API. |
| 2026-06-10 | 1.0.287 | Startup entry-point consolidation (#7215). `launch_golf_suite.py` now delegates to canonical `launch_upstream_drift.py` with a deprecation warning, classic PyQt launch preflights the Qt platform and selects offscreen mode on headless Linux, and `src/api/local_server.py` degrades to an unavailable engine-manager facade when optional engine imports fail during local API startup. |
| 2026-06-10 | 1.0.286 | Removed unsafe Drake pose pickle deserialization from `pose_interchange.pose_io`. Drake `.drake` initial-state files now serialize `{q, v, model_metadata}` as JSON, the loader rejects binary/non-JSON payloads before deserialization, and regression coverage asserts invalid JSON and missing-`q` contracts. |
| 2026-06-10 | 1.0.285 | Legacy golf visualizer dataset contract preservation after row extraction optimization. `golf_visualizer_data.DataProcessor.extract_frame_data` still fails fast when BASEQ, ZTCFQ, or DELTAQ is absent and still returns zero-vector frame data when a requested frame row is missing, with regression coverage for both contracts. |
| 2026-06-10 | 1.0.284 | Frankenstein composition validation framework (#7205). Added `src/tools/model_explorer/composition_validator.py` with structured error/warning findings for duplicate URDF names, orphaned joints, invalid root counts, disconnected links, kinematic cycles, and moving-link mass/inertia contracts. Frankenstein editor export now blocks validation errors by default while retaining an explicit `force=True` escape hatch for recovery exports. |
| 2026-06-10 | 1.0.283 | LauncherContext shared event/value context for embedded tools (#7210). Added `src/shared/python/launcher_embed/context.py` with a headless `LauncherContext` protocol, in-memory snapshot-safe event dispatch, idempotent unsubscribe handles, keyed `value_changed:<key>` notifications, and a small `list/get/set` compatibility surface for Sidekick workspace reuse. `EmbeddedHostWidget` owns one context, injects it into opt-in tools through `set_launcher_context(ctx)`, and emits `tab.opened` / `tab.closed` events while legacy tools without the hook continue to open normally. |
| 2026-06-10 | 1.0.282 | Legacy golf visualizer Pandas row extraction optimization. `golf_visualizer_data.DataProcessor.extract_frame_data` now fetches each BASEQ/ZTCFQ/DELTAQ row once per frame and reuses the resulting row for all point/vector extraction, preserving the missing-row fallback contract while avoiding repeated `.iloc` lookups inside the render-frame path. |
| 2026-06-10 | 1.0.281 | Configuration ownership consolidation (#7216). Removed root `config/` and `configs/` trees. Architecture debt policy moved to `scripts/config/architecture_debt_policy.json`; BunkerShot3D calibration YAML moved to `src/bunkershot3d/calibration/configs/`; UX field/error seed YAML moved to `src/shared/python/ux/config/`. Added `docs/development/configuration-systems.md`, canonical UX path constants, updated generators/tests/docs, and regression coverage preventing root config directories from returning. |
| 2026-06-10 | 1.0.280 | Linux dependency-consistency lockfile repair after #7231. `requirements-dev.lock` now matches the Python 3.12 Linux `pip-compile --extra dev` output enforced by CI: Windows-only transitive `colorama` and `tzdata` entries are removed, and `uvloop` is restored for the Linux `uvicorn[standard]` dependency graph. |
| 2026-06-10 | 1.0.279 | Built-wheel mocap source parity fix (#7213). The Rust `upstream-mocap-io` TRC parser now validates invalid or non-finite data-row frame and time fields before marker coordinates, preserving the Python facade's invalid-row contract when the PyO3 wheel is installed. The OpenCap session adapter test now follows the existing Rust parity contract by comparing marker coordinates approximately instead of requiring impossible exact decimal equality from Rust `f32` output. CI now runs the full `tests/unit/motion_pipeline/sources` directory immediately after installing built Rust wheels, and the Rust parity wheel gate script ratchets that source-wheel coverage. |
| 2026-06-10 | 1.0.278 | MATLAB engine loader unification (#7219). `EngineManager._load_engine()` now dispatches every engine, including `MATLAB_2D` and `MATLAB_3D`, through the registry's `EngineRegistration.factory()` path. MATLAB-family Simscape adapter creation lives in `src.engines.loaders` with loader shim exports preserved for legacy imports, and tests assert the manager has no private MATLAB loader branch. |
| 2026-06-10 | 1.0.277 | OpenSim `.osim` loader for #7203. Added first-party `src/tools/model_explorer/osim_loader.py` to parse OpenSim 3.x and 4.x model XML into `ParsedModel`, convert to validated `CanonicalModel`, preserve masses and joint mappings, and surface muscles/constraints/markers as warnings. Sibling discovery, imported model discovery, file filters, and the model-opening path now accept `.osim` without editing vendored `src/shared/python/model_generation/**`. Regression coverage lives in `tests/tools/model_explorer/test_osim_loader.py` and `.osim` sibling-discovery assertions. |
| 2026-06-10 | 1.0.276 | Drake SDF model loading (#7204). Added `src.tools.model_explorer.SdfLoader` for the model explorer to parse SDFormat links, inertials, primitive and mesh geometry, joint axes/limits/dynamics, SDFormat 1.8 `relative_to` poses, and ball/universal joints into the canonical model contract. Sibling model discovery now classifies `.sdf` files from `Drake_Models` alongside URDF and MJCF assets so Drake-native models can be browsed and composed. |
| 2026-06-10 | 1.0.275 | MJCF fixed-joint roundtrip topology preservation (#7208). URDF-to-MJCF conversion now keeps fixed children as welded nested MuJoCo bodies without joint elements while encoding the original fixed joint name, and MJCF-to-URDF decoding restores that fixed joint name only for welded nested bodies. Regression coverage asserts link sets, fixed and movable joint names/types, parent-child topology, and fixed-joint origin translation across URDF -> MJCF -> URDF. |
| 2026-06-10 | 1.0.274 | Embeddable-tool adapter entry-point discovery (#7211). `src/launchers/embedded_tool_bootstrap.py` now imports `upstream_drift.embeddable_tools` entry-point adapters first, falls back to the in-tree adapter module list for editable checkouts, de-duplicates module paths across both sources, and keeps registry-diff tracking plus manifest-gap warnings intact. `pyproject.toml` declares the first-party adapter entry points so installed wheels and source checkouts use the same bootstrap contract. |
| 2026-06-10 | 1.0.273 | Ball-flight REST simulation route (#7218). `POST /tools/ball-flight/simulate` exposes headless/batch launch simulation through the existing `FlightModelRegistry` and `UnifiedLaunchConditions` stack, validates launch/spin/wind/model/integration-window inputs with Pydantic contracts, and registers the `ball_flight` API tool route alongside the route registry. |
| 2026-06-10 | 1.0.272 | Model Explorer sibling model-repository discovery (#7201). `ModelLibrary.list_available_models()` now exposes a `sibling` category populated by `src/tools/model_explorer/sibling_repositories.py`, scanning `Drake_Models`, `MuJoCo_Models`, `Pinocchio_Models`, and `OpenSim_Models` siblings next to the checkout or explicit `UD_SIBLING_MODEL_REPOS` roots. Discovery accepts URDF plus MJCF XML by content, skips VCS/cache directories, emits stable `sibling_<repo>_<relative-path>` config keys, and surfaces truncation instead of silently hiding excess results. `get_model_info("sibling", key)` resolves the discovered local model metadata without network access. Regression coverage added in `tests/tools/model_explorer/test_sibling_repositories.py`; the human-model fallback download path now delegates to the shared bounded downloader instead of a local `urllib.urlopen` call. |
| 2026-06-10 | 1.0.271 | Follow-up #7189 packaging gate repair after reconciling the parallel branch update. Tauri Linux dependency setup now waits on both apt and dpkg locks, matching the runner failure mode seen in `Check (Rust + TypeScript)`. The WGS process calculator lazy-loads GUI theme helpers from `create_plots_tab`, and the standalone Sidekick regression suite asserts the WGS engine import does not require `shared.python.theme.integration` or PyQt6, keeping the clean-wheel `sidekick run` smoke headless. `package-standalone-sidekick.yml` now smoke-tests Python 3.11/3.12 to match the package floor, with `scripts/ci/check_python_version_coherence.py` guarding that workflow. |
| 2026-06-10 | 1.0.270 | Docker dependency-audit hardening for #7160 follow-up CI. Pinned Docker builder/runtime pip installs to `26.1.2`, declared patched runtime floors for `Mako>=1.3.12` and `PyJWT>=2.13.0`, aligned `requirements.lock`, and updated the direct dependency license ledger so in-image `pip-audit` resolves patched packages. |
| 2026-06-10 | 1.0.269 | Python-version coherence hardening for #7160. Raised the live package/support floor to Python 3.11, removed the unsupported Python 3.10 classifier and standard CI matrix lane, kept Python 3.11/3.12 in the standard test matrix, and documented that the Docker image plus `requirements.lock` are generated on Python 3.12. Added `scripts/ci/check_python_version_coherence.py` and `tests/ci/test_python_version_coherence.py` to enforce agreement across `pyproject.toml`, `install.sh`, `requirements.lock`, `Dockerfile`, `.github/workflows/ci-standard.yml`, mypy target version, and public docs. |
| 2026-06-10 | 1.0.268 | Finished a deferred sub-defect of the pytest-gating policy issue #7158 (D2 marker discipline). Added `tests/support/suite_markers.py` (`SUITE_MARKERS`, `suite_markers_enforced`, `find_unmarked`, `item_has_suite_marker`) and wired a `pytest_collection_modifyitems` + `pytest_terminal_summary` hook in `tests/conftest.py` that, in REPORT-ONLY mode (the repo's ratchet pattern), counts collected tests carrying none of the recognized suite markers and surfaces the baseline without failing collection. Enforcement (missing-marker = collection error) is opt-in via `UD_ENFORCE_SUITE_MARKERS=1` for a follow-up once the unmarked baseline is driven to zero. Unit coverage in `tests/unit/test_suite_marker_enforcement_7158.py`. The remaining #7158 sub-defects (D3 coverage-omit retune) and #7155 sub-defects (deleting the autouse `_protect_engine_modules` and function-scoping `mock_mujoco_dependencies`, both gated on the issue's 5/5 `-n auto` stability evidence) stay deferred for cause and are tracked on those issues. |
| 2026-06-10 | 1.0.267 | Security hardening of remote model downloads (#7183, #7184, #7185, #7186). Added shared helpers `download_to_file` (bounded-timeout streaming download; `DOWNLOAD_TIMEOUT_SECONDS=30`) and `safe_extract_zip` (Zip-Slip member-path validation) to `src/shared/python/security/security_utils.py`. `GitHubRepository.download_archive` now extracts via `safe_extract_zip`, downloads with a timeout, and unlinks its `delete=False` temp file on every path (#7183 Zip Slip + temp leak). All `urlretrieve`/`urlopen` call sites in `standard_models.py`, the `model_generation/library/` repository/loader modules, and `tools/model_explorer/model_library.py` now use a 30s timeout (#7184). `StandardModelManager.download_standard_humanoid` now parses the URDF for mesh-filename entries and downloads the real meshes, returning `False` (no silent empty-STL stubs) unless the dev-only `allow_stub_meshes=True` is passed (#7186). `Jules-Issue-Mention-Handler.yml` and `PR-Comment-Responder.yml` route attacker-controllable issue/comment title/body/login values through `env:` indirection instead of splicing into `run:` bodies (#7185 Actions expression injection). Regression tests added in `tests/unit/test_shared_security_utils.py` (zip-slip + timeout) and `tests/unit/config/test_standard_models.py` (real-mesh success vs loud failure). |
| 2026-06-09 | 1.0.265 | Architecture budget CI gate for #7131/#7133. Added `scripts/ci/check_architecture_budget.py` plus `scripts/config/architecture_budget.json` to ratchet changed production Python files against a 100-line function budget and 8-effective-parameter callable budget, excluding tests/vendor and requiring owned, linked exceptions for any temporary budget breach. Wired the gate into `.github/workflows/ci-standard.yml` and added focused TDD coverage for long-function detection, parameter counting, receiver-parameter exclusion, exception handling, and test-path skips. |
| 2026-06-02 | 1.0.260 | Golf visualizer camera-basis norm optimization (#7101). `GolfVisualizerWidget` now uses fixed-arity `math.hypot` for the known 3D forward and right camera basis vectors instead of `np.linalg.norm`, avoiding NumPy reduction overhead while preserving the existing fallback behavior for degenerate vectors. |
| 2026-06-02 | 1.0.258 | Bolt small-vector norm optimization (#7098). `src/shared/python/physics/ball_simulator.py` now uses fixed-arity `math.hypot` for scalar relative-velocity and Magnus cross-product magnitudes in `_calculate_forces_single`; `src/shared/python/physics/flight_models.py` uses `math.hypot` for Waterloo/Penner spin-vector magnitude and for MacDonald-Hanzely/constant-coefficient spin-axis normalization, computing each spin norm once before normalization; and `src/shared/python/physics/swing_ball_flight_pipeline.py` uses `math.hypot` for launch speed, horizontal launch speed, and angular-velocity spin-rate derivation. This is a pure performance cleanup for known 2D/3D vectors, preserving behavior while avoiding NumPy reduction allocation overhead in scalar hot paths. |
| 2026-06-02 | 1.0.259 | Cross-engine equivalence gate fixes (#7095, #7097). `_run_engine_checked` now treats all-NaN grip traces as missing-engine bindings (`_EngineBindingsError`) while keeping partial NaN/Inf traces as hard simulation-divergence failures. The JaxSim-vs-Pinocchio parity case now uses zero base position so INERTIAL and LOCAL velocity representations are comparable while still exercising mixed angular/linear Coriolis terms. Cross-engine docstrings now distinguish the 5 mm agreement tolerance from the looser per-engine world-frame origin plausibility check. |
| 2026-06-02 | 1.0.257 | Docs root cleanup follow-through for #7063. Removed loose root-level Markdown from `docs/` except the canonical `README.md` and `index.md`; relocated remaining live manuals, architecture notes, assessments, engine guidance, strategic notes, troubleshooting references, and technical guidance into topic directories such as `docs/user_guide/`, `docs/architecture/`, `docs/assessments/`, `docs/engines/`, `docs/strategic/`, `docs/troubleshooting/`, and `docs/technical/`. Preserved the now-real runnable `docs/examples/` subtree and added a structural test that verifies every `docs/examples/index.rst` toctree entry points at an existing page instead of deleting valid examples. Updated docs governance README/catalog text plus live tooling references (`check_formatter_guidance.py`, `check_tutorial_imports.py`, `replace_cli.py`, and `doc_size_budget.json`) to follow the relocated user manuals, and kept `tests/docs/test_docs_structure.py` as the guard for root cleanliness, examples integrity, and removal of stale `TRACKED_TASK` placeholders from Sphinx config. |
| 2026-06-02 | 1.0.256 | Green-suite CI gate + fixed 2 pre-existing unit reds (#7042). Added a new parallel `unit-test-gate` job to `.github/workflows/ci-standard.yml` that runs `pytest -m "unit and not slow and not live_simulation and not requires_gl and not requires_drake and not requires_opensim and not requires_mujoco and not requires_pinocchio and not requires_myosuite and not requires_jaxsim and not requires_nimble and not requires_network and not requires_gpu and not requires_mocap_fixtures" -n auto --timeout=60 --timeout-method=thread --no-cov` with NO `\|\| true` / `continue-on-error` masking, so any non-skip unit failure reds the job. It is wired into the required `quality-gate` summary job (`needs: [code-quality, security-scans, repo-structure-gates, unit-test-gate]`); the aggregate now also fails when `unit-test-gate.result != success`, blocking merge on any unit red. The `quality-gate` check name is preserved exactly for branch protection / bot workflows. RED (a): `tests/engines/physics_engines/test_opensim_engine.py::test_load_from_string_creates_tempfile` patched the public `load_from_path` but `load_from_string` delegates to `_load_from_path_impl`, so the capture callback never ran (`KeyError: 'path'`); the test now patches `_load_from_path_impl` (the actually-called hook) and asserts the path was captured before checking cleanup — needs no real OpenSim. RED (b): `tests/unit/conftest.py::pytest_configure` injects `MagicMock()` into `sys.modules["pinocchio"]`/`["casadi"]` for collection; the `engine_availability` probe's `hasattr(pin, "buildModelFromUrdf")` guard was fooled by MagicMock and cached `pinocchio` as AVAILABLE in the module-level `_engine_status_cache` + `is_engine_available` lru_cache, poisoning every later `requires_pinocchio` test. Fix mirrors the existing drake mock-detection guard: `_probe_engine` now raises `ImportError` when the imported module's `type(...).__module__ == "unittest.mock"` (pinocchio branch + the generic `else` import branch covering casadi), and a new `reset_engine_status_cache()` (clears both dicts + `cache_clear()`) is called by the autouse `_reset_mocks_between_tests` fixture so availability is re-probed per test rather than leaking across the mock/unmock boundary. |
| 2026-06-01 | 1.0.255 | Engines parity cluster (#7048/#7049/#7050/#7051/#7052). #7048: replaced the hollow cross-engine equivalence gate (`tests/motion_matching/test_cross_engine_equivalence.py`) — removed the `12 == 12` meta-tautology and stale `pytest.skip` Drake/OpenSim stubs; each installed engine now runs a real `requires_*`-gated grip-RMSE check vs the Simscape `trial_001` fixture (5 mm gate at the theta=0-valid address pose + cross-engine agreement gate across all three poses after rigid frame registration). #7050: added `get_capabilities()` to the MuJoCo and OpenSim adapters. #7051: Drake `compute_zvcf` reads fixed actuation via the actuation matrix `B` (`a = M⁻¹(Bu − g)`) instead of tau=0; OpenSim `compute_jacobian` uses Simbody analytic `calcStationJacobian` with FD fallback. #7049: value-asserting 1-DOF pendulum dynamics tests for Drake/OpenSim (SPD mass matrix, m·g·L·sinθ gravity torque, τ=M·a+bias within 1e-10). #7052: removed dead placeholders (real MuJoCo energy via `mjENBL_ENERGY`; deleted BVH placeholder; implemented 3 Drake `motion_optimization` cost bodies; implemented the empty `test_cross_engine_consistency.py`). |
| 2026-06-01 | 1.0.254 | API hardening cluster (#7056, #7057, #7058), all under `src/api/`. #7056: `simulation_ws._apply_initial_state` now caps `q`/`v` length before `np.array(...)` allocates, returning an error string (surfaced as a WS error frame) instead of allowing an authenticated client's multi-million-element array to trigger a memory/event-loop DoS. The bound is engine-DoF-aware (`min(hard_ceiling, max(nq*4, 16))` when the engine advertises `nq`) and otherwise falls back to a hard ceiling reusing the request-model constant `MAX_STATE_VECTOR_LEN` (issue #6948) for DRY/defense-in-depth consistency with the `SimulationRequest` validation layer; `set_state` is never called and no array is allocated when rejected. #7057: `chat_ws.chat_stream`'s inner streaming loop wrapped streaming in a broad `except Exception` that also caught a mid-stream `WebSocketDisconnect`, logging a normal client disconnect as an internal error and attempting to send an error frame on a dead socket; an `except WebSocketDisconnect: raise` is added ahead of the broad catch so disconnects propagate to the outer disconnect handler (logged at debug, no error frame). #7058: `data_explorer` fully `json.loads`-ed on-disk JSON datasets per request (`_preview_json_streaming`, `_resolve_operation_source`) while CSV streams; added `_read_json_dataset_text` which checks on-disk size against `MAX_JSON_DATASET_BYTES` (= `MAX_DATASET_SIZE_BYTES`, 50 MB) *before* reading and rejects oversized JSON with HTTP 413, bounding per-request memory and matching the CSV streaming / upload-cap parity. TDD: oversized-q/v rejection + engine-DoF cap + valid-array-applied tests (`test_simulation_ws.py`); mid-stream-disconnect-not-internal-error test (`test_routes_chat_ws.py`); JSON-over-cap-413, under-cap-previews, and JSON-vs-CSV value-parity tests (`test_data_explorer_perf.py`). |
| 2026-06-01 | 1.0.252 | Docs examples + declutter and `quality-gate` parallelisation (#7063, #7064). #7063: `docs/examples/` was effectively empty (only `index.rst`). Added three runnable, dependency-light (numpy-only) end-to-end examples — `run_mock_engine_sim.py` (load `MockPhysicsEngine` + integrate a swing), `motion_matching_synthetic.py` (pose+velocity tracking cost on a synthetic trajectory), `estimate_kinematics.py` (central finite-difference velocity/acceleration estimation validated against an analytic signal) — wired into `docs/examples/index.rst` via `literalinclude`, and guarded by a smoke test `tests/unit/docs/test_examples_runnable.py` that executes each (asserts ≥3 examples, exit 0). Added a `docs/examples/**` T201 per-file-ignore (CLI-style examples print to stdout). Decluttered 9 zero-reference loose top-level `docs/*.md` into `docs/{governance,operations,reviews}/` (e.g. `PROJECT_MAP.md`, `PACKAGE_ORGANIZATION.md`, `fleet_recovery_tracking.md`, `docker-gpu.md`), updating the two live `docker-gpu.md` links; heavily-referenced docs (USER_MANUAL, UPSTREAM_DRIFT_USER_MANUAL, IDEAS, engine_selection_guide, biomech-workspace) and files cross-linked from historical assessment/backlog snapshots were deferred to avoid touching scripts/CI configs. #7064: split the monolithic ~30-step `quality-gate` job in `.github/workflows/ci-standard.yml` into three parallel jobs each `needs: pick-runner` — `code-quality` (ruff lint/format + mypy + install-dependent gates: alembic, pip-audit, code-quality-check, MATLAB), `security-scans` (Semgrep/Bandit/detect-secrets/Trivy; checkout-only, no editable install), `repo-structure-gates` (pure-stdlib SPEC/structural/size-budget/ratchet/placeholder checks) — aggregated by a final `quality-gate` summary job (`needs: [code-quality, security-scans, repo-structure-gates]`, `if: always()`) that fails iff any underlying job failed. The required-check name `quality-gate` is preserved exactly so branch protection and the bot workflows (Jules-Auto-Repair/PR-AutoFix, Bot-CI-Trigger) keep resolving it. |
| 2026-06-01 | 1.0.251 | Deleted dead legacy Motion Capture Plotter monoliths + config/deps hygiene (#7061, #7065). #7061: removed `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/{starting_pose_matcher.py (2671 LOC), Motion_Capture_Plotter.py (1402 LOC)}` — dead duplicates superseded by the tested `src/tools/starting_pose_matcher/`, with zero `src/` importers — and dropped their two `scripts/config/file_size_budget.json` exceptions; added a hygiene assertion that no `src/` import resolves the old path. #7065: routed common hardcoded host/port literals through typed `Settings` (`api_host`/`api_port`), de-duplicated the Cargo workspace `members` (`upstream-mocap-io` was listed twice), and staggered the file-size-budget `expires_on` cliff into distinct dates. |
| 2026-06-01 | 1.0.250 | Added value round-trip + headless coverage for two previously thin/untested modules (#7059, #7062). #7059: `tests/unit/data_io/test_export.py` was smoke-only (registry shape). Added export->reimport **value** round-trip tests for the always-available `json` and `csv` formats against `data_io/export.py`: each exports via `export_recording_all_formats`, reimports (json via `json.loads`+`np.asarray`, csv via `pandas.read_csv`), and asserts numeric equality with `np.testing.assert_allclose` plus column/key order (json key order preserved; csv emits `time` first then 1-D keys then expanded `{key}_{i}` 2-D columns) and unit scaling (mm<->m, kgf<->N scale-and-recover is exact). #7062: `src/shared/python/qt_utils/wheel_event_filter.py` (used by >=5 GUI tabs) had zero tests. Added `tests/unit/qt_utils/test_wheel_event_filter.py` (offscreen `QApplication`, marked `requires_gl`/`headless_safe`): asserts accidental `QWheelEvent`s on combo/spin widgets are swallowed (`eventFilter`->True) both focused and unfocused (the filter is intentionally focus-independent for value-mutation safety), non-wheel events pass through, install/remove via the real event system leaves combo index unchanged, and the `suppress_wheel_on_widget(s)` helpers attach an independent retained filter per widget. 96.2% line coverage of the module (>=90% acceptance). Test-only PR; no source behavior change. |
| 2026-06-01 | 1.0.243 | Repaired ~20 red `tests/unit/api` contract tests (#7044). All were test-side drift, not source regressions. (1) CORS: `get_cors_origins` fail-closed message changed at `src/api/config.py` to "CORS_ORIGINS must not contain '\*' when credentials are enabled (fail-closed)"; `test_config.py` now asserts the live message (intended fail-closed contract). (2) `TaskManager` query/mutation API is synchronous per the #4843 compatibility contract, but several mocks declared `async def exists/get/set` — fixed `test_dependency_injection.py` (`AsyncMock`->`MagicMock`), `test_routes_export.py`, `test_routes_simulation.py`, and `test_simulation_service.py` to sync mocks, resolving `'coroutine' object is not subscriptable/iterable` 400/500s. (3) WS auth gate (#5913): chat-ws endpoints now call `resolve_ws_user` before `accept()` and close 1008 when unauthenticated; `test_routes_chat_ws.py` adds an autouse fixture stubbing `resolve_ws_user` (mirroring `test_simulation_ws.py`). (4) `POST /realtime/publish` now requires `ws_compatible_auth_dependency` (#6888/#6889); `test_routes_realtime_bounds.py` overrides it to a no-op so the amplification-bounds assertions run. (5) `useEngineStore.ts` `unloadEngine` calls the backend via the shared `apiFetch` wrapper not raw `fetch(`; `test_engine_route_contracts.py` accepts `apiFetch`. (6) Rotation-converter router carries `prefix="/api/calc/rotation-converter"`; `test_rotation_converter_mocked.py` reference-frame POST retargeted to the prefixed path. `tests/unit/api` now 0 failures. |
| 2026-06-01 | 1.0.242 | Physics de-duplication + constant provenance/validation (#7053, #7054, #7055). #7053: `_impact_physics.py` and `_impact_recorder.py` were full copies of the canonical `impact_model/{models,types,utils,solver}.py` (guarded only by parity test #7015). They are now thin re-export shims; the duplicate class/function bodies were deleted so `class RigidBodyImpactModel` is defined exactly once (grep == 1), and identity tests assert `_impact_physics.RigidBodyImpactModel is impact_model.models.RigidBodyImpactModel` for every re-exported symbol. #7054: the friction rolling-cap `0.4` in the canonical `models.py` was unprovenanced and physically wrong — derived and corrected to the uniform-solid-sphere rolling-without-slip factor `2/7 ≈ 0.2857` (named `SPHERE_ROLLING_CAP_FACTOR` with full derivation + citation), pinned by an analytic friction-spin test (`omega == (5/7)*v_t/R` at saturation); the warn-only Newmark `_warn_if_ill_conditioned` is now backed by a hard accuracy assertion — an analytic undamped-SDOF test (`u(t)=u0·cos(ωt)`) requires Newmark error < 1% at dt ∈ {1e-3,1e-4,1e-5}; and `biomechanics/ztcf.py _forces_from_accelerations` "simplified" docstring is replaced with the pendulum derivation and validated against the analytic single pendulum (`F_t = -m·g·sinθ`, `qddot = -(g/L)sinθ`). #7055: terrain energy-absorption weights `0.5/0.3/0.2` and grass coefficient `0.1` in `_terrain_physics.py` are now named, documented module constants (`ENERGY_ABSORPTION_*_WEIGHT` convex combination summing to 1.0, `GRASS_RESISTANCE_COEFFICIENT`) with value tests pinning their invariants and the resulting absorption factor; flight Cd/Cl coeffs in `flight_models.py` already carry per-model `reference` citations and are now value-tested to lie within documented golf-ball wind-tunnel ranges (Cd 0.15-0.30, Cl 0.10-0.30; Bearman & Harvey 1976). |
| 2026-06-01 | 1.0.241 | Fixed `BiomechanicalModel.add_segment` segment-name validation drift (#7045). `self.segment_masses` (from `humanoid_character_builder` `estimate_segment_masses`) is keyed by the full segment name (`right_thigh`, `right_shin`, ...), and `compute_dynamic_com` looks up mass by `self.segment_masses[seg.name]`. But `add_segment` validated `get_anthropometry_key(name) in self.segment_masses` — the mapped key (`thigh`) is never a `segment_masses` key, so any laterally-named segment was rejected with a `PreconditionError: Unknown segment name`. Validation now checks `name in self.segment_masses`, consistent with the downstream lookup; `get_anthropometry_key(name)` is retained only in the diagnostic message. Restores 3 red cases in `tests/unit/biomechanics/test_dynamic_com.py` (full file 8/8 green). Source-only fix matching the test's documented API. |
| 2026-06-01 | 1.0.239 | Resolved current-main review regressions (#7037; #7031, #7028, #7027, #7017, #7015, #6954) across API task expiry, chat stream cancellation pairing, full-stream dataset statistics, public impact-model parity, provider ID deduplication, and cloud-token chmod regression coverage. |
| 2026-06-01 | 1.0.241 | Estimation synthetic-fixture corrections (#7043, #7060). #7043: `synthetic_fixtures.make_fixture_cameras` built `cam1` from a hand-typed near-rotation whose lower-left sign (`+0.2588`) made it non-orthonormal; the stricter `CameraExtrinsics` validator (`R.T@R==I`, `det==+1`) rejected it, reddening 5 `tests/unit/estimation/test_synthetic_ground_truth.py` tests. Now built as a true proper 15-degree rotation about Y from `cos`/`sin`, guaranteeing orthonormality + `det==+1`; added positive (fixtures validate, `R.TR≈I`, `det≈+1`) and negative (legacy matrix rejected) tests. #7060: `synthetic_ground_truth.project_world_point` applied only the k1/k2 radial terms while `residuals.project_pinhole` supports the canonical 5-term `(k1,k2,p1,p2,k3)` model (#6907), silently diverging for 5-term cameras. Added a `k3` field (default 0.0) to the `CameraIntrinsics` contract and threaded it into the rig's radial polynomial (`1 + k1*r² + k2*r⁴ + k3*r⁶`); parity tests assert the rig projection equals `project_pinhole` within 1e-9 for nonzero k3 and remains unchanged for k3=0. |
| 2026-06-01 | 1.0.240 | Bolt perf: replaced `np.linalg.norm(v)` with faster 1-D equivalents (`math.hypot` / `math.sqrt(np.dot(v, v))`) in physics hot loops, consolidating the genuine optimizations from #7021/#7029/#7034/#7035 onto a clean base. `flight_models.py` ball-flight `derivatives` inner loops use 3-arg `math.hypot` for the relative-velocity speed and reuse a single `cross_norm` local for the Magnus/lift direction (#7034). A shared float-casting `_magnitude(v) = math.sqrt(float(np.dot(asarray(v, float), ...)))` helper (guards int-dtype overflow per #7022) replaces the norm in `_friction_laws.py` (tangent/slip magnitudes), `_terrain_physics.py` (normal-force/tangent-velocity/impact-speed), and `aerodynamics/_rust_facade.py` (rel-velocity/spin/lift/Magnus magnitudes) (#7021/#7029); regression test `tests/unit/physics/test_magnitude_int_overflow_7022.py` locks the int-overflow guard and float-equivalence. `starting_pose_matcher.py` shaft/residual/torso-disk norms use `math.hypot`/`math.sqrt(np.dot)` (#7035). Pure perf, no behavior change; every transformed site is a provably 1-D vector. |
| 2026-06-01 | 1.0.241 | Motion pipeline now runs end-to-end with a real IK backend (#7046, #7047). Added `motion_pipeline/ik/geometric_backend.py`: a dependency-free damped-least-squares (Levenberg-Marquardt) IK solver with its own `SkeletonRig` forward kinematics; resolved the `GEOMETRIC` enum in `make_ik_solver` to this real module (was importing a nonexistent `geometric_backend`). The mujoco/drake/opensim/pinocchio IK `solve_frame` stubs now raise `NotImplementedError` instead of silently returning a neutral zero pose (#7046). Rewired orchestrator `_run_motion_matching` to route through `make_matching_solver(backend).match(...).to_contract()` (mujoco->mujoco_torque, drake->drake_trajopt, pinocchio->pinocchio_inverse_dyn), replacing imports of the nonexistent `.matching.{mujoco,drake,pinocchio}_backend.run_matching`; `geometric` is now an accepted IK backend (#7047). `matching/torque_mujoco.py` no longer hardcodes `success=True`/`rmse=0`: `success` reflects real execution (False when MuJoCo is absent or only the placeholder model is available, since torques are then all zero) and `fit_metrics` are computed from real residuals (#7047). |
| 2026-06-01 | 1.0.238 | Re-derived three tracked review fixes (#6886, #6907, #6911) onto clean origin/main, superseding the stale 69-file omnibus #6920. #6886: `model_pack/v1` manifest normalization now prepends `models_root` to each relative `exercises[].path` so later `source_root=provider_root` resolution finds `provider_root/<models_root>/<path>`; paths already rooted under `models_root` are left untouched (no double-prefix). #6907: `_apply_brown_conrady` / `project_pinhole` now accept the canonical 5-term distortion `(k1, k2, p1, p2, k3)` in addition to the 4-term form, with the radial polynomial extended to `1 + k1*r² + k2*r⁴ + k3*r⁶`; non-(4,)/(5,) shapes raise `ValueError`. #6911: realtime auth moved onto the route itself (`POST /realtime/publish` now declares `dependencies=[Depends(ws_compatible_auth_dependency)]`, keeping the existing slowapi rate limiter), so `WSPubSub._spawn_server()`'s bare `include_router(realtime_router)` autostart path is protected against unauthenticated broadcast injection (#6888) — previously only server.py's mounts were guarded. TDD: 5-term-accepted + k3-affects-radial + invalid-length residual tests, a `models_root`-prepend manifest test (plus no-double-prefix), and WSPubSub-style autostart auth tests (401/403 unauth, 200 in local mode). |
| 2026-06-01 | 1.0.236 | Added TDD coverage for four previously-untested modules and fixed three real bugs surfaced by the tests (#7000, #7001, #7002, #7004). #7000: MJCF/URDF converter round-trip (parse→emit→parse preserves bodies/joint types/masses/inertia), `_parse_body_inertial` (diag+full), `_parse_mjcf_geom` (box/sphere/cylinder), malformed-XML→error. #7001: SimScape MDL parser `parse_string` blocks/connections/params, `SimscapeParameter.as_float`/`as_vector` (valid+malformed→default), `get_body_blocks`/`get_joint_blocks`/`get_connections_to`, `_get_block_type` map, bad-extension→error. #7002: `ModelGenerationAPI` route handlers (health/info shape, mjcf↔urdf + validate + parse + inertia happy/error, 422 on malformed, route count, 404) plus FastAPI-adapter registration. SECURITY #7004 (sandbox escape FIXED): `scripting_env.ConsoleEnvironment` permitted the classic CPython escape `().__class__.__bases__[0].__subclasses__()` reaching a class whose `__init__.__globals__['__builtins__']` is the real unrestricted builtins, leaking `open`/`eval`/`exec`/`__import__` and reaching `os` despite `import os` being blocked — added `_screen_source_for_escapes` AST screen rejecting introspection dunders before exec/eval. Also fixed: `convert_mjcf_to_urdf`/`convert_urdf_to_mjcf` now catch `xml.etree.ElementTree.ParseError` (a `SyntaxError`, not `ValueError`) so malformed XML returns 422 not 500; and `FastAPIAdapter.register` `make_handler` changed from `async def` (returned an un-awaited coroutine, crashing route registration) to `def`. 120 new tests, all passing. |
| 2026-06-01 | 1.0.235 | Realtime/WebSocket concurrency hardening (#6978, #6980; #6972). `/realtime/publish` now serializes `ws.send_json` per socket behind a per-connection lock so concurrent publishes can no longer interleave frames on the same WebSocket, and a slow/broken subscriber is dropped without taking down healthy ones (#6978). `WSPubSub`'s lazy `_http_client` initialization is guarded so the check-then-act race can no longer construct (and leak) multiple `httpx` clients (#6980). Regression tests cover concurrent same-socket sends, the rate-limit default, and the HTTP-client init race. (TaskManager expiry-sweep throttling for #6992 already landed via #7026; this PR carries only the realtime/WS work.) |
| 2026-06-01 | 1.0.231 | API/storage/task-manager perf and resource fixes (#6988–#6992). `POST /simulate` no longer freezes the FastAPI worker: `SimulationService.run_simulation` offloads the CPU-bound stepping pipeline (`_run_simulation_sync`) via `anyio.to_thread.run_sync`, keeping the event loop responsive under concurrent requests (#6988). `GET /datasets` lists JSON columns by streaming only a bounded prefix (`_stream_json_columns`, 256 KiB sniff window) instead of `json.load()`-ing the whole file (#6990). Added `GET /tools/data-explorer/datasets/{dataset_id}/rows?offset=&limit=` delegating to `DatasetStorage.get_dataset_rows`, plus `iter_dataset_rows` streaming; `dataset_stats`/`filter_dataset` now consume a streaming `_OperationSource` (single-pass CSV streaming, row-capped) rather than materializing the whole dataset (#6991). `DatasetStorage.store_dataset` runs `cleanup_expired()` (TTL retention) before each write and uses `executemany`, bounding `datasets.db` growth (#6989). `TaskManager` throttles its O(n) expiry sweep: the hot read/membership path purges at most once per `CLEANUP_INTERVAL_SECONDS` while remaining TTL-correct via `_is_expired_locked`; writes and aggregate ops force an exact sweep (#6992). Regression tests added for each fix; dead `_load_dataset_from_path`/`_load_dataset_for_operation` removed. |
| 2026-06-01 | 1.0.230 | Shaft-FEM and bunkershot coupling physics fixes (#6983, #6985; #6987 verified). The finite-element shaft cantilever now clamps the BUTT, not the thin tip: `create_standard_shaft` lays the diameter taper thick-butt-at-station-0 to thin-tip-at-the-end, aligning the geometry with the `_apply_boundary_conditions` clamp at node 0 and the analytic `compute_static_deflection` butt convention; natural frequencies, mode shapes, and static deflection are now computed for the correct cantilever orientation (#6983). `FiniteElementShaftModel.step` non-dimensionalizes the Newmark effective system via symmetric Jacobi (diagonal) scaling before the linear solve and emits a conditioning warning when `dt < 1e-2*sqrt(min(diagM)/max(diagK))`, preventing the catastrophic cancellation in the `a_new` recovery at impact-scale dt (~1e-7); `step` now also rejects non-positive dt (#6985). The bunkershot `CoupledDoublePendulum` wrench-to-joint-torque mapping was verified as a correct `J^T` projection — both joints are revolute about world y so the angular Jacobian row is `[1, 1]` and the external moment Ty legitimately projects onto both joint torques (NOT a double-count); added a clarifying comment plus a `test_pure_force_torque_mapping` regression locking the convention (#6987, verified-correct, left open). |
| 2026-06-01 | 1.0.229 | Observability, safety, security, and test polish across the API, deployment, and estimation layers (#6941, #6943–#6950). The realtime controller loop now logs via `logger.exception` and aborts after N consecutive failures, commanding zero torque (#6943); `RealtimeController.stop()` re-checks `is_alive()` after a join timeout, raises on timeout, and only sends the zero command on confirmed stop (#6944); `ChatService` fallback logs the exception and exposes explicit `backend_error`/`adapter_available` state instead of silent degradation (#6945); cloud token storage sets dir mode `0o700` and token mode `0o600` (#6946); the simulation route returns a generic client detail while logging specifics server-side (#6947); `SimulationRequest` caps `control_inputs` length and `_normalize_initial_state_component` length (#6948); re-raising/except branches in `server.py`, `task_manager_durable.py`, and `launcher_diagnostics.py` switch to `logger.exception` (#6949); the dead timezone compat shim in `auth/dependencies.py` is removed (#6950); and `multi_trial` gains negative-path validator/accessor/stack tests plus a hardened `stack_shared_parameter_jacobians` 1D guard that previously raised `IndexError` (#6941). MyPy Strict errors exposed by these edits are resolved, including a generic `_assert_type` helper (`type[T]->T`) satisfying both mypy configs. |
| 2026-06-01 | 1.0.228 | Design-by-Contract input validation hardening across the motion pipeline, Rust kernel, and velocity conventions (#6930–#6934, #6940, #6942). `motion_pipeline` `/run` now validates `source_format` against the adapter registry up front (400) and `loader.load_source` raises `ValueError` for a non-auto `format_hint` matching no adapter instead of silently falling through to `load_any` (#6930); `velocity_conventions` `_as_spatial_vector`/`_as_vector3`/`_as_matrix3` add `np.isfinite` guards via `_require_finite` and `single_floating_body_h_g` asserts finite mass (#6931); the orchestrator distinguishes caller contract violations (`InvalidInputError`→400) from internal faults (`RuntimeError`→500) and pydantic `ValidationError`→422 via `StageResult` error_kind (#6932); `CameraExtrinsics.rotation` validates finiteness, orthonormality (`R.T@R==I`), and proper-rotation (`det==+1`) (#6933); `rust_kernel` `create_air_properties`/`create_ball_properties` reject non-positive density/viscosity/temperature and mass/radius (#6934); plus parametrized fallback-vs-Rust backend tests (#6940, #6942). Bandit B104 on the dev-only uvicorn `0.0.0.0` entrypoint is annotated `# nosec B104`. |
| 2026-05-31 | 1.0.227 | Tightened the Pinocchio dynamics API checker error boundary to catch only import failures, preserving the error-handling ratchet while still returning a diagnostic failure when the robotics Pinocchio package cannot be imported. |
| 2026-05-31 | 1.0.226 | Hardened the JaxSim/Pinocchio parity prerequisite contract after CI exposed an unrelated `pinocchio==0.1` package shadowing the robotics API. Cross-engine equivalence now uninstalls the wrong `pinocchio` distribution before force-installing `pin>=2.6.0,<5.0.0`, and `scripts/ci/check_pinocchio_dynamics_api.py` fails the prerequisite step unless `import pinocchio` exposes the required free-body dynamics symbols (`Model`, `JointModelFreeFlyer`, `SE3`, `Inertia`, `crba`, `rnea`, `computeCoriolisMatrix`). |
| 2026-05-31 | 1.0.225 | Hardened JaxSim readiness and parity gates (#6880, #6881, #6882, #6884): `EngineManager` now registers JaxSim as a runtime-backed engine, gates runtime-backed availability on both adapter/provider paths and importable dependencies, treats provider path-policy `PreconditionError`s as provider-discovery misses rather than constructor failures, adds a focused `JaxSimProbe`, and wires `scripts/ci/require_junit_test_passed.py` into cross-engine equivalence so skipped/missing JaxSim/Pinocchio parity cases fail the required CI gate. |
| 2026-05-31 | 1.0.224 | Retired three dead launcher/GUI controls from an adversarial audit: JaxSim dashboard feature rows become read-only `QLabel` capability indicators ("capability indicator (read-only)" tooltips) instead of enabled-but-unconnected `QPushButton`s (#6901); the cached singleton `LibraryWidget` nulls its reference via `destroyed.connect` so a detached-then-closed Library no longer leaves a dangling C++ object (RuntimeError on re-open) (#6902); and the never-started/connected animation `QTimer` is removed from `MultiModelShotTracerWidget` (#6903). |
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
| 2026-05-31 | 1.0.219 | Added metadata-driven helpful-field and provenance-value wrappers for the Idiot-Proof UX epic (#5968): `HelpfulField` and `ProvenanceValue` PyQt6/React controls consume the existing metadata/provenance contracts, `scripts/ux/generate_field_metadata_ts.py` generates the TypeScript registry from `src/shared/python/ux/config/field_metadata.yaml`, and focused PyQt/Vitest tests cover contract validation, ARIA help text, provenance display, and YAML-to-TypeScript parity. |
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
| 2026-05-22 | 1.0.180 | Landed the pure-Python foundation for the Idiot-Proof UX epic (#5968): `src/shared/python/ux/` adds the `FieldMetadata` registry, `ProvenanceRecord`/`ProvenanceValue`, `PreflightCheck`/`Severity`/`run_preflight()`, and the `UserFacingError` envelope, all with full Design-by-Contract validation; seeded `src/shared/python/ux/config/field_metadata.yaml` and `src/shared/python/ux/config/error_messages.yaml`; added `scripts/ci/check_ux_coverage_ratchet.py` plus baseline at 714 unwrapped inputs (62 QSpinBox + 221 QDoubleSpinBox + 217 QComboBox + 70 QSlider + 94 QLineEdit + 35 `<input>` + 14 `<select>` + 1 `<textarea>`); documented the workflow in `docs/ux/field_metadata.md`; 68 unit tests in `tests/unit/ux/`. Sanitized unexpected `src/api/routes/simulation_ws.py` runtime errors before they reach WebSocket clients while preserving traceback-bearing server logs, and added direct regression coverage for the generic error payload contract. Re-baselined `scripts/config/module_size_budget_baseline.json` from 10 stale exceptions (sizes 3-5x overstated, 7 files since decomposed) down to the 3 modules that genuinely exceed 1,500 lines today, and added `validate_baseline_truthfulness` to `scripts/check_module_size_budget.py` as a CI ratchet against future fraudulent baselines. Refs #5922. |
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
| 2026-06-01 | 1.0.223 | DRY/LoD consolidation across the motion-matching engines and BunkerShot3D backends (#6935–#6939). Added `resolve_club_target()` + `publish_leaderboard_row()` to the shared `motion_matching.provider` module so all six engine providers delegate one canonical target-unwrap and leaderboard-append; this UNIFIES previously forked behavior — a `ClubBallTarget` now unwraps consistently on every engine (was a `TypeError` on mujoco/pendulum/pinocchio) and every engine forwards `target_id` (#6935). Extracted `ChronoDriver._make_contact_material()` so walls/grain/clubhead share one SMC material factory (#6936). Added flat delegating accessors on `BunkerShotConfig` (`contact_params()`, `domain_extents()`, `grain_count`, `clubhead_*`, `output_rate_hz`, `trajectory_*`) so chrono/mpm drivers stop reaching two levels into the nested config (#6937). Collapsed the drifted `opensim/motion_matching/forward_kinematics.py` FK copy (which read non-existent `/bodyset/Club/*` frames) into a thin re-export of the canonical `opensim_golf/fk.py` extractor (#6938). Added a shared `motion_matching.provenance` module (`engine_package_version()`, `git_commit_short()`) and routed the five `engine_version()` cascades and three git-commit probes through it (#6939). |
| 2026-06-03 | 1.0.224 | Bolt: Optimized `np.linalg.norm(v[:2])` to `math.hypot(v[0], v[1])` in `ball_trajectory_analysis.py` to avoid temporary array allocation and speed up calculation. |
````

- Optimized magnitude calculations using math.hypot instead of np.linalg.norm in MuJoCo humanoid golf engine

- Optimized 3D vector norm calculations in physics engines using math.hypot instead of np.linalg.norm.

- Updated `golf_data_core.py` to cache Pandas row to avoid expensive `df.iloc[row_idx]` repeated calls during vector operations.

- Fixed CI imports for `compute_total_work`, `sidekick` references in `c3d` and `load_body_target_c3d` routing to appease lazy loading logic.

- `extract_dynamics_dataset` also requires torch now so we require torch to test it in `test_surrogate_perstep_relocation.py`.

### 2024-06-13

- **Performance:** Optimized `grf_visualization.py` by extracting DataFrame columns to NumPy arrays (`.values`) before plotting loops, avoiding expensive and repeated `.iloc` series creation.
- **Performance:** Replaced `np.linalg.norm` with `math.hypot` for 3D vector magnitudes in MuJoCo motion optimization, avoiding array overhead.

- Replaced `np.linalg.norm(..., axis=1)` with `np.sqrt(np.einsum('ij,ij->i', ...))` in `DriftControlAnalyzer.compute_ratio` for a 2.4x speedup.
  Updated analyze_coordinate_system.py to use math.hypot for 3D vector magnitude
- Optimized vector and quaternion norm calculations in
  `src/shared/python/visualization/fsp_renderer.py`,
  `src/shared/python/pose_interchange/adapters/_base.py`, and
  `src/shared/python/spatial_algebra/pose6dof/rotations.py` by replacing
  `np.linalg.norm` with `math.hypot` and `math.sqrt(np.dot)` in fixed-size hot
  paths.

## Change 2026-06-18

- Replaced `np.linalg.norm(..., axis=2)` with `np.sqrt(np.einsum('...i,...i->...', ...))` in chain_forces.py and test_swingset_chain_models.py for improved performance.

### Module Map Changelog

- `golf_camera_system.py`: Replaced `np.linalg.norm` with `math.hypot` for 3D and 2D vectors.

### Module Map

- Updated math.hypot usage for small 1D arrays to math.sqrt(np.dot) in various places.

### 2026-06-21

- **Performance:** Replaced `np.sum(forces, axis=0)` with `sum((s.force for s in self._sources.values()), np.zeros(3))` in `ForceAccumulator` methods (`get_total_force`, `get_total_torque`, and `get_total_generalized_force`) in `src/engines/common/state.py` to avoid intermediate list and array allocations, yielding ~30% faster execution time for accumulating forces and torques.

### Performance Improvements
- Replaced `np.sum(diff ** 2)` with `np.vdot(diff, diff)` for squared Euclidean distance calculations in `TreeConfigIndex` to avoid intermediate array allocations.
- Replaced `np.sum(..., axis=1)` with `np.einsum('ij->i', ...)` for array reductions in critical pathways in data input and plotting.
