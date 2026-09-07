# Changelog

All notable changes to UpstreamDrift will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.2] - 2026-09-03

This release carries every entry that was staged for 2.1.1. The `v2.1.1` tag was
pushed on 2026-09-02 but published no artifacts: its `release.yml` run failed in
the `build` job because `build_hooks.py` requires `ui/dist` while the workflow
set `SKIP_UI_BUILD` and never compiled the frontend (#9449). Every downstream
job was skipped, so no wheel, sdist, CycloneDX SBOM, `SHA256SUMS.txt`, PyPI
distribution, or GitHub release exists for 2.1.1.

The `v2.1.1` tag is retained exactly where it is. A pushed tag is already
fetchable by consumers, mirrors, forks, and CI caches; deleting and re-pushing
it would make one name resolve to two different commits for different
observers. A tag with no release attached is itself the accurate record that
nothing shipped. 2.1.1 is superseded by 2.1.2, which ships the same content on
top of the release-workflow fix in #9450.

### Added

- A two-excitation typed-slack dynamic audit for the momentum-transfer program,
  with separate mechanical and control boundaries, work-energy closure,
  passivity checks, local scaled-sensitivity ranks, cross-class output
  separation, and explicit non-identification and human-evidence limits.
- New `simulation_backends` package: the golf double-pendulum model is described
  once (`GolfModelParams`) and run through interchangeable `ode` (CPU analytical
  reference), `mujoco` (CPU), and `mjwarp` (MuJoCo Warp GPU) backends behind one
  `SimulationBackend` Protocol, cross-validated CPU↔CPU to ~`1e-9`. Includes the
  **Simulation Backends** launcher tile (id `simulation_backends`, simulation
  category) for editing the model, running rollouts and parameter sweeps,
  cross-validating dynamics, and exporting HDF5 traces; a batched `rollout_batch`
  API for GPU sweeps; ZTCF/ZVCF acceleration decomposition; HDF5 trace I/O; and
  user docs in `docs/simulation_backends/`. See ADR-0023 and ADR-0024.
- Sidekick design-token adapters for React/Tauri CSS variables and guarded PyQt
  Tools sidebar theme handoff (#5384).
- Release governance guard `scripts/check_version_consistency.py` with CI wiring,
  release runbook, production-readiness contract, SBOM generation, and release
  artifact attestations for issue #3842.
- `validate_baseline_truthfulness` ratchet in
  `scripts/check_module_size_budget.py`: fails CI when a module-size baseline
  exception references an under-budget file or quotes an "N lines" figure that
  diverges from reality by more than 10% (#5922).

### Changed

- Optimized whole-body SciPy QP inequality construction for #7568 by replacing
  per-row SLSQP callbacks with vector-valued lower/upper bound callbacks and
  tightening QP matrix/bound validation at construction.
- Hardened the standard CI PR-scoped unit gate for #7314: source/dependency PRs
  now use the dependency-light unit lane instead of passing solely on touched
  test files, and targeted PR coverage runs a changed-file policy ratchet.
- Tightened internal motion matching runtime contracts (#7304, #7305, #7306,
  #7309): invalid request/weight configuration fails at construction, metric
  helpers reject mismatched trajectory frame/DOF shapes instead of truncating,
  solver result validation checks reference-aligned time grids plus finite
  torque/activation payloads, and successful internal results require a matched
  payload before reaching orchestration.
- Re-baselined `scripts/config/module_size_budget_baseline.json` to drop 7 stale
  exceptions for files that have since been decomposed back under the 1,500-line
  cap, and updated the remaining 3 entries with current truthful line counts
  (#5922).
- Recorded ADR-0021 and `docker/README.md` to make the root container policy
  explicit: `Dockerfile` is the default release/runtime path,
  `Dockerfile.heavy_test` stays heavy-test parity only, and
  `Dockerfile.modular` remains the opt-in profile build surface (#6097).

### Fixed

- Tightened the feature-parity registry loader and reconciled stale issue refs
  (deferred P2 findings from #7740): `FeatureParityEntry.from_dict` now rejects a
  non-string `notes` field and rejects a truthy `pending_decision` on any
  non-`exempt` entry (it is exemption-scoped). The duplicated positive-integer
  issue-number predicate is extracted into a shared `_is_valid_issue` helper used
  by both the gap and non-gap branches. The two parity rows that still carried
  open-issue links — `analysis.static_plots` (#7449) and
  `simulation.shot_tracer` (#7456), both since closed completed — drop the `issue`
  field (provenance preserved in `notes`); the generated parity matrix doc is
  regenerated to match.
- Cleared a batch of scattered deferred P2 findings from #7740: unified the
  cross-engine output time grid behind a shared `build_output_grid`
  (Drake/MuJoCo now align endpoints for non-integer-divisible `(T, rate)`;
  bit-identical for divisible cases); extracted a shared `rk4_step` integrator
  for the double/triple pendulum models (bit-identical); vectorized the
  analysis `_section_crossings` zero-crossing scan; replaced O(N^2) grain-XML
  string concatenation in the MPM driver with a single `"".join`; replaced
  `assert`-based input guards in `signal_toolkit.series` with `require(...)`
  (survives `-O`); de-duplicated `ForceVector3D` construction via a
  `_make_force_vector` helper; added `Scheduler.has()` / `Scheduler.list_jobs()`
  facade passthroughs and a `LabelledControl.set_value_silent` so callers stop
  reaching through to private internals (Law of Demeter); dropped the dead
  `"ok"` solver-status literal from the MuJoCo synthesizer; guarded the
  best-effort recovery `send_json` in the chat WebSocket index and
  router-factory condense handlers with `contextlib.suppress`; and seeded the
  imitation-learning test RNG plus replaced the flaky wall-clock latency budget
  in the pose-studio realtime test with a readiness handshake.
- Hardened the simulation WebSocket and data-explorer API routes (deferred P2
  findings from #7740): the WS start guard now rejects a non-positive
  `speed_factor` instead of silently clamping it to 1.0, and its duration /
  timestep bounds reuse the Pydantic `SimulationRequest` constants
  (`MAX_SIMULATION_DURATION` / `MAX_TIMESTEP` / `MIN_TIMESTEP`) so a value
  between the old looser WS caps and the model caps no longer passes the WS
  layer only to fail validation with a generic error. The repeated
  `app.state.simulation_service.stats` reach-through chain was collapsed into a
  single `_resolve_sim_stats` accessor, and the previously-untested
  `_run_simulation_loop` success paths (frame emission/throttle, pause/resume,
  real-time pacing, live-analysis) now have direct async coverage. In the data
  explorer, `_find_dataset_path` rejects glob metacharacters and matches by
  exact filename (closing a wildcard existence oracle), `GET /datasets` bounds
  its recursive scan with `os.scandir` + offset/limit and a hard cap instead of
  sorting the whole tree synchronously, and the dead contract-violating
  `_store_cached_dataset` helper was removed. Added filter operator/edge-case
  and ambiguous-name 409 tests.
- Hardened the deployment safety/realtime layer and the Pinocchio club-target
  adapter (#7740): `RealTimeController._is_running` is now read/written under a
  shared lock and cleared in a `try/finally` so a stale `True` cannot survive
  the abort-raises path; `SafetyMonitor.check_state` raises an approaching-limit
  WARNING symmetrically on both joint bounds (margin hoisted to
  `NEAR_LIMIT_MARGIN_RAD`); `SafetyMonitor.get_stopping_distance` dropped its
  unused `body` parameter and documents that it returns a joint-space braking
  angle (rad), not a Cartesian distance; the club-target adapter shares the
  canonical validation tolerance constants instead of re-declaring them and
  drops a dead `butt_rotmats` re-slice with a corrected comment; added coverage
  for the resampled-shape-mismatch raw fallback in `load_robneal_target`.
- Hardened the MuJoCo grip-modelling tab contact pipeline (#7740): compute real
  relative contact velocity via `mj_objectVelocity` instead of feeding a
  fabricated `np.zeros(3)` (which left velocity-based slip detection dead);
  attribute each hand contact to the hand-side body explicitly instead of
  always using `body1_name` (previously mislabelled ~half of contacts with the
  club/object body when the hand was `geom2`); drop the redundant
  `mj_forward`/`render` calls after `set_state_and_forward` in `_update_joint`
  and `_update_joints` (they doubled the solve + GL render per slider tick);
  replace `-O`-stripped `assert` model/data preconditions in `_update_joints`
  with an early-return guard matching `_update_joint`; and name the club-weight
  static-equilibrium load as `CLUB_WEIGHT_N` instead of an unexplained `3.0`.
- Rendered the catch-all 404 route synchronously so unknown deep links show the
  recoverable branded "Page not found" screen immediately instead of racing the
  route-level lazy-loading fallback (#7430).
- Restored the Tauri release-build dependency contract for #7652 by aligning
  the Rust `tauri` lockfile package with the locked `@tauri-apps/api` minor
  version, installing the `libdbus-1-dev` Linux header package required by the
  updated Rust graph, and adding CI infrastructure regression tests that fail
  before `tauri-action` can reject release builds for Rust/npm Tauri minor drift
  or missing native Linux headers.
- Gated Windows Tauri release packaging behind the
  `TAURI_WINDOWS_RELEASE_ENABLED=true` repository variable because the current
  self-hosted Windows runner blocks Cargo build-script executables with
  Application Control (`os error 4551`). Linux release packaging and the Tauri
  Rust/TypeScript check remain enforced.
- Vectorized deformable object internal-force hot paths for #7571/#7572:
  FEM assembly now batches tetrahedral deformation gradients, inversions, and
  nodal force scatter while reusing each inverse once; cable and cloth spring
  forces now use cached vector connectivity with scalar-reference parity tests.
- Replaced the iLQR backward-pass explicit `np.linalg.inv` gain solve with a
  finite-checked Cholesky solve path plus general-solve fallback for #7570.
- Replaced MuJoCo humanoid golf effective-mass explicit inverse calculations
  with a shared solve-based kernel and finite/symmetry postconditions for
  #7560.
- Optimized JaxSim trajectory parameter-gradient evaluation for #7562 by
  constructing the selected autodiff transform once per API call, batching
  samples with `jax.vmap`, and validating finite Jacobian output shapes.

### Refactor

- Renamed source-revealing identifiers and directories in motion-matching code
  to generic names (#4480). Affected files include
  `motion_matching/loaders/_gears.py` -> `_marker_clusters.py`
  (with `GearsClubPose` -> `ClusterClubPose`,
  `is_gears_schema` -> `has_marker_clusters`,
  `extract_gears_pose` -> `extract_cluster_club_pose`),
  `pinocchio/python/dtack/utils/gears_parser.py` -> `mat_dataset_parser.py`
  (with `GearsParser` -> `MatDatasetParser`),
  `pinocchio/python/dtack/viz/rob_neal_viewer.py` -> `swing_dataset_viewer.py`
  (with `RobNealDataViewer` -> `SwingDatasetViewer`), MATLAB
  `gears_marker_map.m` -> `cluster_marker_map.m`, data dirs
  `pinocchio/data/rob_neal/` -> `club_swing_dataset/`,
  `pinocchio/data/gears_tour_average/` -> `tour_average_mocap/`, and
  `Simscape .../Data/Gears C3D Files/` -> `Mocap C3D Files/`. Backwards-compat
  shims live at the old Python module paths and emit `DeprecationWarning`;
  they will be removed in a future release. Filenames of `.c3d`/`.mat`/`.xlsx`
  data files are preserved as-is.

### April 2026

#### Added

- GitHub Actions pinning to commit SHAs for supply chain security (#3210)
- Comprehensive pytest markers added to 937+ test files for better test organization
- Aerodynamics engine integration with BallFlightSimulator for consistent wind/ball settings (#3204)

#### Fixed

- CI/CD gaps: added proper gating on build success before release artifacts (#3210)
- MyPy exclusion list reduced from 65+ to targeted disable_error_code directives (#3075, #3207)
- Pip-audit CVE ignore list cleaned up with proper dependency upgrades (#3208)
- Removed sys.path hacks from examples for clean package install (#3049, #3209)
- Pinocchio zero-force fallback for unsupported contact-force queries (#3201, #3211)
- verify_installation.py expanded with comprehensive environment checks (#3172, #3203)
- basic_flight_simulation.py example now produces output instead of silent execution (#3171)
- HelpPanel populated with substantive content and contextual bindings (#3170)
- Tutorial imports fixed: broken URLs and incorrect module paths corrected (#3169)
- UI README replaced Vite template with UpstreamDrift developer guide (#3176)
- Silent failure paths replaced with actionable error messages (#3175)
- Aerodynamics and impact models wired into physics engine step() (#3167)
- Four React Tool pages (DataExplorer, MotionCapture, VideoAnalyzer, PuttingGreen) backend implementation (#3166)
- RAG store wired on startup and glossary endpoint added (#3164, #3165)
- Tool calling enabled in chat service streaming (#3162, #3163)
- ChatPanel component shipped in React/Tauri web UI (#3161)
- AI assistant core extracted for dashboard-help integrations (#3145)
- Model editor with duplicate, parameter presets, and post-sim summary (#3174, #3190)
- Orphan root-level scripts cleaned up and reorganized (#3070)
- Test pollution eliminated: sys.modules mocking replaced with proper patching (#3212)

#### Documentation

- Updated SPEC.md for accuracy with current implementation (#3071)
- Removed orphan root scripts and tracked junk files (#3070)
- Added docs hub for navigable documentation tree (#3213)

### February-March 2026

- Initial assessment framework (Adversarial Review A-O)
- Cross-engine parity testing infrastructure
- Physics correctness validation framework
- Refactored launcher consolidation
- Humanoid builder consolidation

### Security (CRITICAL - January 13, 2026)

Critical security fixes for authentication, data exposure, and dependency vulnerabilities.

#### Added

- `SECURITY.md`: Comprehensive security policy with reporting procedures, best practices, and compliance standards
- `docs/SECURITY_UPGRADE_GUIDE.md`: Step-by-step migration guide for API key regeneration
- `scripts/migrate_api_keys.py`: Automated migration script from SHA256 to bcrypt hashing
- `tests/unit/test_api_security.py`: Comprehensive security test suite (200+ lines)
- `engines/pendulum_models/archive/README_SECURITY_WARNING.md`: Security warnings for legacy code
- `.gitattributes`: Exclude archive code from language statistics

#### Fixed (CRITICAL)

- **API Key Security**: Upgraded from SHA256 (fast hash, brute-force vulnerable) to bcrypt (slow hash, industry standard)
  - **BREAKING CHANGE**: All API keys must be regenerated - old keys will NOT work
  - Constant-time comparison prevents timing attacks
  - Files: `api/auth/dependencies.py`
- **JWT Token Generation**: Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)`
  - Python 3.12+ compatible
  - Explicit timezone handling for distributed systems
  - Files: `api/auth/security.py`, `api/auth/dependencies.py`
- **Password Logging**: Removed plaintext password from logs
  - Admin password no longer logged on startup
  - Recovery instructions provided instead
  - Files: `api/database.py`
- **Archive Code Isolation**: Added security warnings for unsafe eval() usage
  - Legacy code with code injection vulnerabilities clearly marked
  - Excluded from GitHub language statistics
  - Files: `.gitattributes`, archive README
- **Security Audit**: Made `pip-audit` blocking in CI
  - Vulnerabilities now fail CI instead of warning
  - Automated dependency security enforcement
  - Files: `.github/workflows/ci-standard.yml`

#### Documentation

- `CRITICAL_PROJECT_REVIEW.md`: 557-line adversarial security review
- `SECURITY_FIXES_SUMMARY.md`: Technical details of all security fixes
- `PR_SUMMARY.md`: Comprehensive PR summary with migration checklist

#### Compliance Achieved

- ✅ OWASP Top 10 (Authentication, Sensitive Data Exposure)
- ✅ CWE-327 (Broken Cryptography)
- ✅ CWE-532 (Sensitive Info in Logs)
- ✅ CWE-94 (Code Injection - archived/warned)
- ✅ Python Security Best Practices
- ✅ FastAPI Security Guidelines

**Production Ready**: Previously unsuitable for production → Now production-ready ✅

---

### Added

- Comprehensive assessment framework (A-O) with 15 quality categories
- MyoSuite integration for musculoskeletal modeling
- OpenSim tutorials and example scripts
- AGENTS.md restored to root with Golf-specific guidelines
- Critical files protection CI workflow (prevents accidental deletion)
- Expanded pre-commit hooks (trailing whitespace, YAML validation, large file detection)

### Changed

- Updated README status from BETA to STABLE
- Removed broken GolfingRobot.png reference
- Cleaned 30+ debris files from root directory
- Updated .gitignore to prevent future accumulation
- Moved utility scripts from root to scripts/ directory
- Archived pre-Jan11-2026 assessment documents

### Fixed

- Mypy errors in plotting module
- Type annotations across physics engines

## [2.1.1] - 2026-09-02

Tagged but never published. The release workflow's `build` job failed (#9449),
so no release, wheel, sdist, SBOM, checksum file, or PyPI distribution was ever
produced for this version. The tag is retained as the record that nothing
shipped from it and is superseded by 2.1.2. See
`docs/operations/release-runbook.md`, "Failed Release Recovery -- Fix Forward,
Never Move a Tag".

## [1.0.0] - 2026-01-10

### Added

- 5 Physics Engines: MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite
- 1,563+ unit tests for comprehensive validation
- Professional PyQt6 GUI launcher
- Multi-engine comparison capabilities
- URDF generator with bundled assets

### Features

- Manipulability ellipsoid visualization
- Flexible shaft dynamics modeling
- Grip contact force analysis
- Ground reaction force processing

### Infrastructure

- Cross-engine validation framework
- Scientific plotting architecture
- Energy monitoring system
