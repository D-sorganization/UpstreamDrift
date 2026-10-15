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
| **Current Version**     | 2.1.3                                              |
| **Spec Version**        | 1.0.720                                            |
| **Last Spec Update**    | 2026-09-10                                         |

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

- **2026-09-03** - Made the tag-driven release able to produce a wheel at all
  (issue #9449). `build_hooks.py` refuses to package once `CI` is set and
  `ui/dist` is absent, but `release.yml`'s `build` job never compiled the
  frontend, so the PEP 517 backend failed and tag `v2.1.1` published no
  release, wheel, SBOM, or PyPI distribution. The `build` job now installs
  Node 24 with an `ui/package-lock.json`-keyed npm cache, runs `npm ci` and
  `npm run build` in `ui/`, and asserts `ui/dist/index.html` before
  `python -m build`. Because `_register_ui_bundle` only warns when the bundle
  is missing, the release-blocking wheel smoke suite now asserts that the
  published wheel carries the `ui/dist` payload including a compiled
  `ui/dist/assets/*.js` bundle, and that the wheel version equals the
  `pyproject.toml` version and, on a tag run, the tag itself. A failed tag is
  recovered by fixing forward to the next patch version rather than moving a
  published tag; `docs/operations/release-runbook.md` carries the procedure.

- **2026-08-22** - Made the modular Docker build boundary independently
  verifiable for issue #8996. Reusable pinned-Tools setup emits the exact
  `vendor/ud-tools` gitlink plus a deterministic SHA-256 over the minimal
  Tools-owned package roots. The modular build copies only those roots and
  verifies the digest without relying on Git metadata. Missing roots,
  symlinks, malformed attestations, incomplete source sets, and content drift
  fail closed. The Hatch hook loads its adjacent canonical helper by file path
  so isolated PEP 517 builds do not depend on the repository root being on
  `sys.path`. Constructor-compatible Hatchling test doubles and explicit suite
  isolation make package-boundary tests independent of collection order. All
  supported image definitions reassert scanner-fixed `msgpack` 1.2.1 and
  `setuptools` 83.0.0 after their final dependency layer, while both runtime
  Dockerfiles retain pip 26.2.1. This closes GHSA-6v7p-g79w-8964,
  CVE-2025-47273, and PYSEC-2026-3447 in the resolved container rather than
  suppressing Trivy or pip-audit. Production runtime targets then remove pip
  after their dependency environments are finalized, eliminating pip's
  embedded third-party SBOM and vendored build surface. The training stage
  explicitly restores the audited builder environment because it is the only
  downstream stage authorized to install packages. The modular builder also
  copies both force-included launcher modules before feature installation, so
  isolated package metadata generation matches the declared wheel surface.

- **2026-08-21** - Resolved the dual aero-coefficient-set discrepancy (issue
  #8978, epic #8965 WS2): the multi-model ball-flight framework's
  Waterloo/Penner set is now a named, provenance-documented
  `AeroCoefficientSet` (`WATERLOO_PENNER_COEFFICIENTS`) that shares the Penner
  lift shape with `ball_properties.py` (single source) while keeping its
  deliberate constant-spin calibration (cd1 = 0.05, lift cap 0.155) distinct
  from the core simulator authority (cd1 = 0.25, cap 0.26). Every
  `FlightResult` and REST `/tools/ball-flight/simulate` response now carries
  the model's coefficient dictionary for attribution, the calc sheet documents
  both families with parity-gated markers, and
  `tests/unit/physics/test_aero_coefficient_authority.py` pins each public
  entry path to its declared set.

- **2026-08-21** - Pinned the proximal--distal companion consumer to protected
  Tools merge `1664d806df8a2c7b184d2d3fbcea93b714caaee5`. The launcher authority
  and gitlink now agree on one immutable provider, and the pendulum-simulator
  launcher is included in the Tools-source contract. Cross-repository
  qualification verifies the complete ordered 18-run rotating-base catalog,
  its 13 valid cases and adverse indices 6/7/8/15/16, the exact qualified-study
  and catalog digests, and every run's nonanatomical-coordinate,
  unavailable-human-validation, and unsupported-coaching boundaries. No solver
  or catalog is copied into UpstreamDrift, and this transport qualification does
  not promote the model to anatomical, empirical, or coaching evidence. The
  standalone package build now retains a 30-minute execution budget so a cold
  pinned-Tools checkout, frontend build, wheel-content gate, and artifact upload
  can complete without the previous 15-minute timeout canceling a valid upload.

- **2026-08-20** - Added attested session-unit longitudinal analysis under
  contract `launch-monitor-longitudinal-session/1.0.0`. Trusted, distinct
  player/session/order evidence is mandatory. Shots collapse into equal-weight
  player/session/stratum cells before per-player descriptive slopes and pooled
  player-fixed-effects OLS with player-clustered finite-sample uncertainty.
  Strata, confounders, missingness, unavailable states, units, lineage, and
  source-linked backing remain explicit; shot-level inference, causal
  improvement, and automatic interpretation of metric direction are false.
- **2026-08-20** - Added canonical player covariation and population synthesis
  under contract `launch-monitor-player-covariation/1.0.0`. Selected-pair
  results distinguish pooled, player-centered, between-player, and per-player
  associations; fixed- and DerSimonian-Laird random-effects Fisher-z summaries
  report Q, tau-squared, and I-squared heterogeneity. Trusted explicit player
  identity is mandatory and remains separate from session/order evidence.
  Missing rows, ineligible groups, constant variables, insufficient population
  evidence, units, vendor/model provenance, and source-joinable backing hashes
  remain explicit. The bounded deterministic pair scan is exploratory, retains
  unavailable pairs, and warns about multiplicity, aggregation reversal,
  causality, and population-generalization limits.
- **2026-08-20** - Added immutable, aggregate-only launch-monitor dataset jobs
  under contract `launch-monitor-dataset-job/1.0.0`. A request identifies a
  server-authorized opaque root and exact repository, commit, corpus-manifest
  SHA-256, deterministic Parquet-content SHA-256, and expected row count; it
  cannot carry rows, paths, URLs, SQL, or arbitrary query text. Verification
  fails closed on checkout, lineage, content, or count mismatch. The initial
  source-summary, metric-summary, and correlation allowlist suppresses groups
  below ten rows and bounds result/page sizes. Production route registration
  requires the existing bearer authentication dependency; source outputs omit
  filenames, raw URLs, server paths, and observations. FastAPI lifespan joins
  bounded worker threads, and capacity exhaustion is a retryable HTTP 429.
- **2026-08-20** - Added the canonical source-backed strokes-gained authority
  under `src/tools/launch_monitor_model/`. The contract requires complete
  start and finish course states and a versioned, licensed, HTTP(S)-sourced,
  SHA-256-verified expected-strokes baseline. It interpolates only within an
  exact lie/context/target stratum, fails closed on extrapolation, retains row
  and dataset hashes plus backing benchmark values and exclusions, and reports
  uncertainty as unavailable when the source supplies no standard errors.
  Trusted identifiers and numeric ordering are required for grouped or
  longitudinal summaries. A separate outcome-proxy contract reports
  target-relative radial error but is structurally forbidden from claiming
  strokes gained. FastAPI publishes both contracts and their generated OpenAPI
  types for React consumers; these descriptive results make no causal,
  device-emulation, device-certification, or independent endorsement claim.
- **2026-08-20** - Recovered BunkerShot3D cross-tier plumbing and MPM code verification
  inside CI architecture and file budgets (issue #8743, #8741). Brings `solvers/mpm/verification.py`
  and `bunker_shot_gui/bridge.py` inside budget gates, verifying 177 tests across conservation,
  analytic elastic column, GCI mesh convergence, and F0 cross-check cases without regressions.
- **2026-08-19** - Corrected the AI provider-adapter tests that had not
  followed deliberate implementation changes (issue #8771). 19 tests under
  `tests/unit/shared_python/ai/adapters/` fail on `main`; ten of them are
  test-side and are fixed here. The base-adapter test double never gained the
  `list_models` / `thinking_capabilities` methods #5635 made abstract for the
  chat-dock dropdowns. Ollama and OpenAI usage assertions still named
  provider-specific token keys that #2763 replaced with a canonical
  input/output/total triple. Gemini error tests still expected transport
  failures rendered as model content, which #3179 deliberately replaced with
  typed raises, and its streaming test predated #2763's guarantee that every
  stream ends with exactly one `is_final=True` chunk. Each is updated to
  assert the contract by name rather than having its number loosened.
  Separately, `test_gemini_adapter.py` stubbed `google.generativeai` with a
  bare `MagicMock`, which auto-creates any attribute including `Client`, so
  the adapter's `HAS_GEMINI_CLIENT` probe came out true under test and nine
  tests exercised the per-instance Client branch while asserting against
  `configure` / `GenerativeModel` mocks nothing ever touched; the legacy
  branch is pinned explicitly and the modern branch gets its own test. The
  remaining nine failures need canonical Tools source and are fixed in
  Tools PR #4574, arriving here through a `vendor/ud-tools` bump - the
  adapters are Tools-owned child copies and
  `test_tools_child_copy_contract.py` correctly refuses direct edits.

- **2026-08-19** - Closed the false-green hole in CI Standard's `tests` lane
  (issue #8771). `main` @ `6b68f94` reported `tests (3.11)` / `tests (3.12)`
  as successful without running the unit suite. The lane derives its scope
  from `git diff <base> HEAD`, but read that diff through
  `mapfile -t x < <(git diff ...)`, which discards git's exit code. The push
  diff base was not present in the shallow checkout, so all four diffs failed
  with `fatal: bad object 59ecef7d...`, every `changed_*` array came back
  empty, and the "no core Python/test/dependency changes detected" branch
  exited 0. The green badge covered 14 tests, not ~2,500. Three changes: the
  diffs are captured through files so the step's `-e` makes a failed diff
  fatal; an unresolvable diff base is now an explicit `::error::` and exit 1
  rather than an inferred "nothing changed"; and pushes to the default branch
  always run the full lane unscoped, because path-scoping is a pull-request
  latency optimisation and on trunk it lets a commit that touches no Python
  vouch for a suite it never ran. A genuine "nothing relevant changed" skip
  now emits a `::warning::` and a job-summary block stating that the suite was
  not executed, so a passing check can be told apart from a green suite. The
  same silent-failure shape in the `Check for core test relevant changes`
  pre-step is hardened identically, and `tests/ci/test_ci_infrastructure.py`
  gains two guards asserting the new contract - a failed diff cannot be
  read as "nothing changed", and trunk pushes are never path-scoped.
  pre-step is hardened identically.

- **2026-08-19** - Repaired all five `profile-size-matrix` Docker builds
  (issue #8771). `Dockerfile.modular`'s builder stage copies a deliberate
  minimal slice - `src/shared/python/__init__.py`, `engine_core/` and
  `feature_registry/` - so resolving a profile does not invalidate the layer
  cache on every source change. `scripts/docker/install_features.py` then did
  `from src.shared.python.feature_registry.features import get_feature`, and
  importing a submodule executes every parent package `__init__` first: once
  `src/shared/python/__init__.py` grew an eager `from . import ai`, and `ai/`
  is not in the slice, every modular image build died at `ImportError: cannot
import name 'ai' from partially initialized module`. The script now loads
  `features.py` from its path with no parent package, which is what its own
  comment always claimed ("read the features module in isolation") and what
  makes the builder slice self-sufficient regardless of what any package
  `__init__` later imports. `features.py` depends only on `dataclasses` and
  `typing`. The module is registered in `sys.modules` before execution
  because `@dataclass` resolves `cls.__module__` through it while processing
  the class body. `src/shared/python/__init__.py` is a Tools-owned child copy
  and is deliberately left untouched.

- **2026-08-19** - Git-ignored the test-run artefacts that have twice been
  committed at the repository root (issue #8771). The autouse
  `_prevent_repo_root_io` fixture chdirs every test into `tmp_path` so they are
  not produced (#7935), but nothing stopped them being staged once they
  existed: #8322 committed all nine (`base.csv`, `base.json`, `base.mat`, two
  `.provenance.json` files, three `pytest_report*.txt` and a one-off patch
  script) and #8747 had to delete them again while repairing the root-clutter
  gate. The names are now ignored, anchored to the root so the tracked
  `docs/research/.../base.csv` fixture is unaffected.
- **2026-08-19** - Restored `optional-stack-check (3.11)` (issue #8771). The
  lane failed on exactly three tests in `tests/unit/deployment/test_devices.py`,
  each asserting `not dev.connect()` for the SpaceMouse, VR-controller and
  haptic stubs, which raise `NotImplementedError` instead. #8322 made that
  change and kept the tests. It also deleted the reasoning: before #8322 each
  method returned `False` under a docstring explaining that "there is no
  hardware driver behind this class, so the honest answer is always 'not
  connected' (#7360)", and that the body it replaced had probed for an
  optional backend and returned `False` on both arms, making the stub look
  conditionally functional. Returning `False` is the contract the rest of the
  subsystem keeps: `BaseInputDevice.connect` returns `False`, the ROS2 and UDP
  controller stubs return `False` with tests asserting it, and
  `test_base_input_device` asserts it of the base class directly - so the
  three overrides were inconsistent with their own parent. #8322 additionally
  rewrote two of the three test files that cover these stubs
  (`tests/deployment/test_teleoperation.py` and
  `tests/deployment/wave5_deployment/test_teleoperation.py`, six assertions)
  from `assert not d.connect()` to `pytest.raises(NotImplementedError)`, and
  missed the third - which is the whole reason `optional-stack-check` went red
  while `unit-test-gate` did not. Before #8322 all three files asserted
  `False`. The implementation bodies, their rationale and the six rewritten
  assertions are all restored to their pre-#8322 text, so one contract holds
  across the subsystem again; the `#8058` tracking reference is kept and no
  test was skipped or deleted.

- **2026-08-16** - Scoped BunkerShot3D's camber-clamped flag so it cannot
  understate substitution (`src/bunkershot3d/geometry/lofting.py`,
  `src/tools/bunker_shot_gui/`, issue #8698, epic #8699). The observability
  work below left one boolean answering a narrower question than its name
  implied. `LoftedWedge.camber_was_clamped` compared the declared camber area
  against the band the **declared** sole width admits. Heel and toe relief
  narrows the sole toward the ends, and a narrower sole admits a narrower
  camber band, so the relieved stations are refitted to their own bands while
  the declaration itself is honoured — and the flag read `False` beside a
  non-empty `clamped_stations`. That is #8698's own failure mode, silent
  substitution invisible to a caller, reappearing through the simpler check.
  Measured at shipped resolution, three of the six shipped presets hit it:
  `sm9_58_m` declares 42.00 mm² inside its (38.70, 42.44) mm² band and refits
  3 of 17 stations; `acushnet_example_1` refits 5 of 17; `tour_shaved_heel_lob`
  refits 13 of 17. All three reported "not clamped".

  Resolved by renaming rather than redefining. Redefining a published boolean
  in place would have silently changed every existing caller's answer, which
  is the same failure mode one level up; the rename breaks loudly instead.
  `camber_was_clamped` becomes `aggregate_camber_was_clamped` (unambiguously
  the declared-versus-effective question), and a new `any_camber_was_clamped`
  is true when the declaration **or** any station was substituted, so a caller
  checking one boolean gets the honest answer and the flag cannot read `False`
  while `clamped_stations` holds anything. `camber_substitution_m2` becomes
  `aggregate_camber_substitution_m2`, which also ends its name collision with
  `StationCamber`'s per-station property. Both meanings are genuinely used, so
  a single flag could not serve: the workbench's camber-area line needs the
  aggregate because it prints the declared number, and the contact-patch
  caveat needs any-station. The camber-area line now also names the refitted
  station count, so an in-band declaration over refitted stations no longer
  renders as a clean number. `PATCH_CONFOUND_CAVEAT` is gated on the per-
  station account rather than the aggregate flag and is restated in terms of
  stations — gated on the aggregate it was silent on the default design, which
  is exactly when the caveat is needed.

- **2026-08-16** - Made BunkerShot3D's sole-camber substitution observable
  (`src/bunkershot3d/geometry/`, issue #8698, epic #8699). A wedge sole can
  only realise camber areas inside a band set by its width and bounce, so
  `build_wedge_mesh` fitted a declared `sole_camber_area_m2` that fell outside
  it to the nearest constructible value. That fit is physically correct and is
  retained — a narrow sole geometrically cannot host an arbitrarily large
  camber, and emitting an inconstructible section would be worse — but it was
  **unobservable**: `constructible_camber_range_m2` was not re-exported from
  `bunkershot3d.geometry` or the top level, and no result object carried the
  effective value back, so a caller who declared 48 mm² received a different
  sole and had no way to find out. Measured on a 77-point demo sweep, the
  clamp fired on 40 points and moved the effective camber over 24.5–61.6 mm²
  against a constant declared 48.0 mm².

  This matters most in `bunkershot3d.study`: a `MorrisDesign`,
  `SaltelliDesign` or `SobolIndices` run over sole width or bounce would
  attribute variance to a camber the user believes is pinned, and no
  diagnostic in the artifact would say so.

  Three complementary changes. (1) `loft_wedge()` returns a new `LoftedWedge`
  carrying `effective_camber_area_m2`, `constructible_camber_range_m2`,
  `aggregate_camber_was_clamped`, `any_camber_was_clamped`,
  `aggregate_camber_substitution_m2` and a per-station
  `StationCamber` account; `build_wedge_mesh()` is now a thin wrapper that
  returns only its mesh. (2) `constructible_camber_range_m2` is re-exported
  from `bunkershot3d.geometry` and the top-level package, and
  `DesignSpace.check_wedge_camber(geometry)` screens the corners of a design
  box against the band before a sweep spends solver time inside it — the wedge
  knowledge lives in `geometry/design_bounds.py` and is imported on call, so
  the study layer keeps its independence from the geometry package. (3)
  `CamberFit` makes silence opt-in: the default `CamberFit.STRICT` raises
  `InconstructibleCamberError` when a **declared** camber is outside the band,
  and `CamberFit.NEAREST` is the explicit opt-in to nearest-constructible
  behaviour. Relief-scaled stations are always fitted rather than refused,
  because that request is derived by the lofter rather than declared by the
  caller — but every substitution, declared or derived, is recorded.

  `STRICT` is the default because it is what the rest of the package already
  does: `build_sole_profile` raises for exactly this condition one layer down,
  `DesignSpace.sample` raises rather than silently lose Sobol' balance, and
  `WedgeGeometry.__post_init__` rejects inadmissible combinations. The lofter
  was the single place that downgraded a loud failure to a silent one.
  `InconstructibleCamberError` subclasses `BunkerShot3DValueError`, so
  existing `except ValueError` sites keep working. Per CLAUDE.md every guard
  `raise`s and none `assert`s, because `python -O` strips assertions.

  Two consequences fell out of making the check loud. Three shipped grind
  presets (`acushnet_example_2`, `acushnet_example_3`, `tour_shaved_heel_lob`)
  declared a camber area their own sole cannot carry, so their meshes had
  never matched their declarations; their `sole_camber_area_m2` — an
  `ESTIMATED` field in every preset — is corrected to a constructible value,
  and the patent-example helper now takes it per example because the band
  climbs steeply with bounce. And the workbench GUI
  (`src/tools/bunker_shot_gui/`) opts into `CamberFit.NEAREST` explicitly,
  because a designer dragging a bounce slider must keep getting a head to look
  at; having opted in, its evaluation report now states the camber area the
  head actually carries alongside the declared one.

- **2026-08-15** - Restored the two repository-hygiene guards
  (`tests/unit/repo_hygiene/test_vendor_submodule_clean.py` and
  `test_no_shadow_of_tools_shared.py`) to actually enforce. Both were
  introduced 2026-05-16 (#5623 / PR #5625) and reduced to stubs on 2026-08-01
  by consolidation commit 0575fb4b8 (#8322), which also emptied the shadow
  ledger and added four new shadows in the same commit. Three independent
  mechanisms kept them vacuous: neither file carried a suite marker, so
  `unit-test-gate`'s `-m "unit and ..."` selector never collected them; both
  called `pytest.skip` when `vendor/ud-tools` was absent, so they passed
  vacuously even when collected; and push-to-main runs are cancelled by
  `cancel-in-progress` concurrency (83 of the last 85), so the only lane that
  would collect them effectively never completed.

  A missing vendor tree now raises `AssertionError` when `$CI` is set and only
  skips on a developer machine, matching `test_tools_child_copy_contract.py`.
  The shadow ledger in `scripts/config/shadow_modules.yaml` is re-established
  as a no-growth ratchet: 32 grandfathered entries (28 from the original #5623
  baseline plus the 4 that #8322 added while the guard was off), each carrying
  a `tracking_issue` and a `sunset_date`. Bare name lists are rejected, expired
  sunset dates fail, and stale entries must be pruned, so the ledger can only
  shrink.

  The vendor-clean guard also carried a latent detection bug: it filtered
  `git status` lines for the prefix `vendor/ud-tools/` (trailing slash), but
  git collapses all submodule-internal state onto the gitlink entry whose path
  is exactly `vendor/ud-tools`, with flags `S<c><m><u>`. The predicate could
  never match. Detection now parses the sub-status field, so modified (`S.M.`)
  and untracked (`S..U`) content fail while a deliberate pointer bump (`SC..`)
  passes. A new `test_hygiene_guards_run_in_ci.py` asserts that every workflow
  job running this package materialises `vendor/ud-tools`, and that
  `unit-test-gate` does so before invoking pytest.

- **2026-08-15** - Restored input validation on the symbolic solver router
  (`src/shared/python/calc_backend/routers/symbolic_solver.py`, issue #8675).
  This file is a shadow copy of a module owned by the Tools repository. Tools
  hardened its copy on 2026-07-22 by guarding every `parse_expr` call with
  `validate_expression`; the shadow was forked on 2026-05-20 and never received
  that change, so it carried six fewer guards than upstream while keeping all
  four parse sites. The removal was not deliberate — `validate_expression` has
  no history in this file at all — it is drift by omission.

  The exposure was not theoretical. `sympy.parse_expr` is an _evaluating_
  parser, and all three affected endpoints (`/api/calc/symbolic/solve`,
  `/derivative`, `/simplify`) take their expression from the request body, so
  the input is remote by construction. Against the unguarded module,
  `__import__("os").getcwd()` executed and returned the server's working
  directory in the response error string, and `9**9**9**9` hung the worker
  in an uninterruptible bignum computation that the thread-based pytest
  timeout could not kill.

  The fix re-adds `from src.shared.python.safe_eval import validate_expression`
  (UD's own `src.`-prefixed convention, matching the sibling `ode_solver`
  router) and the four upstream call sites: both sides of an `lhs = rhs`
  equation, the bare-expression branch of `/solve`, and the single expression
  on `/derivative` and `/simplify`. Note that a `vendor/ud-tools` submodule
  bump cannot fix this: the submodule is unpopulated, and
  `src/shared/python` precedes it in resolution, so the shadow is what is
  actually imported.

  Regression coverage pins the guard to behaviour rather than to its presence:
  each endpoint must reject attribute-based calls, attribute access, and
  exponentiation bombs; legitimate work (`x**2 - 4 = 0`, `x^2` via
  `convert_xor`, `sin(x)**2 + cos(x)**2`) must still solve; the payload's
  observable side effect must never appear in a response; and an
  instrumented-ordering test asserts `parse_expr` never runs ahead of
  `validate_expression` on any path.

- **2026-08-14** - Rebuilt BunkerShot3D as a multi-fidelity wedge-design tool
  under epic #8607 (ADR-0032). The package is now organised around the design
  question — given two sole geometries, which performs better, in what
  conditions, and how confident are we — rather than around granular backends.
  Resolved DEM was demoted from the primary path on arithmetic, not preference:
  the previous canonical configuration held 50,000 grains at 0.4 mm in a
  0.4x0.3x0.1 m domain, a solid fraction of 1.4e-4 and a settled bed 0.023 mm
  deep, while a real 100 mm USGA base needs 2.1e8 grains, and the Chrono driver
  integrated at ~11,900x the Rayleigh stability limit by using the output
  sampling rate as its timestep. The default tier (F0) is now an analytic
  dynamic Resistive Force Theory solver at ~14 ms/shot, with DEM retained and
  explicitly labelled non-viable at true grain scale.

  New public surface: `bunkershot3d.{geometry,sand,domain,solvers,ball,metrics,
study,vandv,provenance,units}`, with subpackages re-exported by name and a
  curated flat set; `study` is lazily imported so `postproc` does not require
  the optimisation extras. The result schema is versioned (contiguous arrays
  replacing one HDF5 group per timestep, reader accepting v1 and v2) and every
  run carries a manifest with config and physics hashes, seeds, and a validity
  verdict.

  **Credibility, stated plainly: verification is real, validation is zero.**
  Observed orders are 1.004 (energy) and 2.001 (surface quadrature) against a
  closed-form integral, and an angular-momentum check catches axis swaps that
  leave the resultant force bit-identical. But no published data exists for ball
  launch, spin, head deceleration in sand, energy split or ejecta mass, so
  `ValidationComparison` refuses to construct such a comparison at all. The
  solver is used ~60x outside RFT's stated Fr < 0.4 envelope and ~20x beyond any
  published validation; `delta_h` and `lambda` are uncalibrated for a wedge, and
  every F0 coefficient is borrowed rather than measured. Out-of-envelope queries
  refuse rather than return a plausible number. See
  `docs/bunkershot3d/credibility.md`; `docs/bunkershot3d/comparison.md` was
  rewritten after seven of its eight claims were found to contradict the code.

- **2026-08-14** - Completed the release-level claim-review authority for epic
  #8557. All 36 release claims now link to supporting atomic claims, evidence,
  negative controls, falsifiers, uncertainty boundaries, a scientific
  disposition, and a remaining gate. Open review bookkeeping is zero while
  human self-stabilization, physical bilateral sensing, and other scientific
  gates remain explicitly untested or conditional.

- `src/shared/python/club_data/loader.py`: Replaced `.iterrows()` with `df.to_dict('records')` (spec-exempt: micro-optimization)

- **2026-08-14** - Reconciled all nine photographed momentum-transfer source
  points with inspectable evidence artifacts and a generated readiness audit.
  Eight retain bounded model answers, partial answers, or a negative
  general-rule result. MTQ-06, whether passive or drift-mediated transfer
  reduces timing precision beyond the adverse planar comparison and in people,
  remains unresolved. Candidate-census completion is now reported separately
  from 10 pending or in-progress release reviews.

- **2026-08-14** - Added MT-E08 subject-scaled spatial contact-closure audit.
  Six deterministic de Leva engineering profiles, three grip spans, and 61
  prescribed states per case expose hand-to-grip miss distances of
  0.171--0.616 m, with no sample meeting the registered 5 mm tolerance. Every
  local bilateral contact Jacobian still has rank six, demonstrating that local
  correction rank does not establish geometric closure. Closed-contact inverse
  kinematics with joint-limit and collision checks is now a precondition for
  compliant forward-contact, anatomical, passive-timing, slack, or human-
  strategy claims.

- **2026-08-14** - Qualified the MT-E07 bilateral point-force estimator over
  deterministic synthetic trajectories with normalized noise and cross-talk,
  cross-talk calibration residual, contact-center migration, and tracked-contact
  controls. Net-wrench-only inversion retains its manufactured axial allocation
  error despite numerical resultant closure. This is not calibration of a
  physical bilateral six-axis device and does not support anatomical or human
  strategy claims.

- **2026-08-14** - Added MT-E07 bilateral-wrench structural identifiability.
  Two separated three-axis point forces map to one net club wrench with rank 5
  and one equal-and-opposite axial null mode; adding one independent axial
  scalar closes that point-force rank gap. Two full six-axis hand wrenches map
  to net wrench with rank 6 and nullity 6, so net club wrench cannot recover
  individual bilateral allocation. Grip-span and proper-rotation controls,
  publication evidence, tests, claim adjudication, and the sensor-qualification
  plan are release artifacts. Practical noise/cross-talk, contact migration,
  muscle or scapular action, and governed human validation remain open.

- **2026-08-14** - Added the #8557 handwritten momentum-transfer agenda
  readiness contract. Nine independently testable points retain their present
  answer state, decisive next test, falsifier, data gate, model plan, and
  participant-held-out human stage in a generated fail-closed audit. Casting
  is definition-dependent and partly answered; broader timing precision remains
  unresolved beyond the adverse planar comparison. Issue #8556 remains the
  human-data blocker.

- **2026-08-13** - Adjudicated the transmission robustness chapter for #8557.
  All four registered programs remain nondominated in every held-out
  leave-one-case-out recomputation. The task map retains algebraic rank three
  but exposes scale-dependent practical rank and material held-out linearization
  error, preventing promotion of a local null space to a neural synergy.

- **2026-08-13** - Reconciled the forward distributed modal-shaft chapter
  under epic #8557. The regenerated hash-bound authority now records the
  preregistered 5% tip-deflection model-use screen and its observed 13.48%
  failure. The run remains evidence for numerical modal coupling, force-couple
  geometry controls, and timestep behavior, but quantitative linear-shaft,
  equipment, human, and coaching inferences are rejected for this baseline.

- **2026-08-13** - Adjudicated both counterfactual chapters under epic #8557.
  The legacy MATLAB killswitch is now classified as a rerun-based pointwise
  sampler, and its `DELTA` force-work columns are explicitly rejected as an
  additive actuator or muscle-work partition. The forward matched-state bundle
  now samples the command at every state time including the terminal endpoint,
  records affine closure and interpolation contracts, hashes its complete
  declared source/data closure, separates 1 ms and 2 ms refinement from the
  0.5 ms reference, labels gravity/damping studies as nonadditive whole-model
  variants, and proves byte-deterministic evidence. All 45 chapter candidates
  map into PD-CLAIM-093 through PD-CLAIM-103; one paragraph already reviewed in
  an earlier chapter retains its stable identity, yielding 288 unique reviewed
  candidates overall. Human, bilateral-hand, anatomical, and universal-strategy
  conclusions remain untested.

- **2026-08-13** - Corrected and hardened the proximal-link velocity evidence
  under epic #8557. The paper and machine contract now use the actual relative
  coordinate convention (`q2` relative, `q1 + q2` absolute), include the
  achieved state in every reference-centered sweep, report finite-range and
  local slopes beside linear-fit adequacy, preserve all grid-selection limits,
  expose regression rank and conditioning plus Pareto coverage, distinguish a
  net planar wrist-interface force from either hand's anatomical load, refine
  the best fixed program at half timestep, hash the computational dependency
  closure, and prove byte-deterministic JSON/NPZ evidence. All 30 chapter
  candidates are mapped to PD-CLAIM-081 through PD-CLAIM-092; anatomical
  shoulder, thorax, bilateral-hand, and coaching conclusions remain untested.

- `src/shared/python/motion_matching/loaders/c3d.py`: Optimized `np.linalg.norm` to `math.hypot` inside `_shaft_quaternions` for better performance on small 3D vectors (spec-exempt: micro-optimization).
- **2026-08-12** - Added the interactive proximal--distal dynamics companion
  under epic #8511. UpstreamDrift now resolves the canonical sibling Tools
  provider and its dockable adapter instead of maintaining launcher-specific
  physics. The PyQt6 and React/Tauri clients share a validated experiment and
  glossary catalog covering double, triple, bilateral, counterfactual, and
  robustness studies. Interactive output is explicitly exploratory; scripted,
  frozen analyses remain the publication authority.

- **2026-08-12** - Added epic #8507's adversarial transmission and task-
  robustness tier: paired clock/state-trigger programs on common training and
  held-out perturbations, lower-tail speed and variability/load/effort
  objectives, explicit pathway and contact-power closure, a local task-null
  variance analysis, a machine-readable 12-item gap register, and four vector
  figures. All programs remain Pareto-nondominated; state triggering improves
  selected speed and planar face/path metrics while increasing peak hand force.
  Human self-stabilization, causal physiology, and coaching prescriptions remain
  untested or unsupported.

- **2026-08-12** - Expanded the proximal--distal resource with explicit
  reference-frame and point-transport equations, numerical power and virtual-
  work invariance checks, a reduced Hill-type redundancy and activation-history
  bridge, five canonical-pose adapter round trips, a question-to-engine ladder,
  five vector figure families, a normative terminology contract, a reviewer
  index, and an advanced falsification roadmap under epic #8505. Pose adapter
  closure is representation evidence only; subject-scaled anatomy, five-engine
  dynamics parity, and human validation remain open.

- **2026-08-11** - Added the continuous preparation-history falsifier for the
  arm--wrist transmission study. Both channels now start relaxed, load for 180
  ms, and retain their exact deflection and transmitted torque across the
  command transition; persistent directions remain engaged while complete role
  reversal crosses the declared dead zones. This is an abstract transmission
  history, not an anatomical backswing or muscle-activation model.

- **2026-08-11** - Adjudicated the independent proximal-to-distal technical
  review in epic #8499. Corrected the phase-integration boundary and replaced
  sampled COM differentiation with analytic acceleration, retained every
  geometric impact candidate with a reason-coded status, added 20--50 ms
  command-rise sensitivity, and narrowed finite-grid, actuator-work,
  pointwise-counterfactual, physiological, and coaching claims.

- **2026-08-11** - Replaced the obsolete fixed 1 MiB PDF ceiling with an
  artifact-integrity and repository-boundary policy. Changed PDFs must carry a
  valid `%PDF-` signature, files above GitHub's 50,000,000-byte recommendation
  are reported, and files beyond its 100,000,000-byte hard boundary fail. The
  manuscript optimizer still proves page, URI-link, and outline preservation;
  an explicit release-specific ceiling remains available when justified.

- **2026-08-11** - Added a coupled moving-base, two-hand, three-mode shaft
  experiment and a separate matched-task arm--wrist allocation/preload study.
  The latter holds an 8 N m club task exact while varying actuator subspace,
  quantifies internal load tradeoffs, and tests persistent-direction versus
  role-reversal commands over a declared dead-zone/time-constant grid. Both
  remain synthetic mechanism tiers; muscle, scapular, tissue-slack, equipment,
  human-performance, and universal-technique claims remain unsupported.

- **2026-08-18** - Added a first-class corpus entry point to the Launch
  Monitor Analytics workbench. A "Load Private Corpus" button on the Sessions
  tab and a matching File-menu action load every source in the authorized
  private corpus as one session per source (261,666 shots across 27 sources in
  about 3 seconds), through `MainWidget.load_private_corpus_sessions()`. The
  action is repeatable — already-loaded sources are skipped rather than
  raising — and fails closed with a dialog, not a traceback, when no
  authorized checkout or Parquet reader is present. The Trends and Dispersion
  tabs remain inert for corpus data because the corpus carries no capture
  timestamp or lateral carry; both gaps are tracked as data-authority issues
  #18 and #19 and are recoverable from retained native fields.
- **2026-08-19** - Bound the Launch Monitor Analytics Dispersion and Trends
  tabs to corpus data. The data authority now extracts lateral carry, flight
  time, and a capture date, and `launch_monitor.corpus` maps them onto the
  canonical `lateral_carry` (m), `flight_time` (s), and `captured_at` fields.
  Both tabs were previously inert against the corpus for want of a lateral
  coordinate and a time column; they now run over 20,099 and 8,488 shots
  respectively. Column selection is filtered against the dataset schema, so a
  corpus pinned before those columns exist still loads.
- **2026-08-19** - Published Launch Monitor Analysis Contract v2 as the
  canonical cross-repository evidence envelope. The v2 API carries explicit
  unit authority, exact commit/source/transformation/backing-record lineage,
  missingness and exclusions, uncertainty, vendor/model provenance, and typed
  available/partial/unavailable states. Player grouping requires an explicitly
  trusted identifier and evidence; session, club, filename, source layout, and
  row order are never identities. Contract v1 remains available through its
  unchanged endpoint and a bounded compatibility adapter.
- **2026-08-18** - Connected Launch Monitor Analytics to the private shot
  corpus. `launch_monitor.corpus.load_private_corpus()` reads the data
  authority's source-partitioned Parquet corpus (261,666 shots across 27
  sources at current head) into the canonical schema via the importer's unit
  tables, with source/metric pushdown, lazy `pyarrow`, the established
  fail-closed `LAUNCH_MONITOR_DATA_ROOT` convention, and `apex_native`
  excluded as unit-ambiguous. Also repaired the bare `flight_models` import
  inside `kaggle_validation.compare_all_models_to_dataset()`, which only
  resolved under pytest's `pythonpath` and broke installed consumers.
- **2026-08-11** - Added the first UpstreamDrift consumer boundary for the
  canonical Tools ground-model contracts (Tools #4276). The headless gateway
  validates the exact flight-to-ground request/result and reference-execution
  v1 schemas before binding Tools parsers or execution, stays import-safe when
  the optional authority is absent, and preserves returned records and
  provenance unchanged. This is not a dependency pin or UI parity claim: the
  exact `vendor/ud-tools`/Cargo repin, FastAPI, PyQt, React, clean-install, and
  protected-release gates remain blocked on the reviewed Tools ground merge.
- **2026-08-13** - Ratified the cross-repository control-affine terminology
  contract for #8586. Canonical ZVCF now means an instantaneous fixed-
  configuration evaluation with velocity and declared applied control both
  zero; analysis schema 3.0.0 preserves the former control-retaining result as
  `zero_velocity_control_preserved_acceleration`. The proximal-distal evidence,
  constrained-reaction decomposition, engine protocol, paper, and regression
  tests distinguish the two quantities and link to AffineDrift's normative
  notation authority.
- **2026-08-11** - Added #8493 ground-reaction drift attribution: a
  frame-explicit constrained-contact reaction solver decomposes support
  reactions into configuration, velocity, control, and retained-external
  components; deterministic double-pendulum evidence verifies total, ZTCF, and
  ZVCF closure; and the scientific article defines measurement, identifiability,
  and human-data falsification boundaries without treating overlapping
  counterfactuals as additive effort fractions.
- **2026-08-11** - Prepared the local launcher UI-setup decomposition tracked by
  [#8490](https://github.com/D-sorganization/UpstreamDrift/issues/8490) from
  exact PR #8489 head `2f664d2beaddf7444b12f90080ae9897aea24fcc`.
  Navigation/sidebar/menu construction now lives in the private
  `_launcher_navigation_ui.py` mixin, while status/search/runtime/view/zoom
  construction and the historical top-bar widgets live in the private
  `_launcher_top_bar_ui.py` mixin. `UISetupManager` retains every historical
  method through inheritance or its compatibility facade, including dynamic
  manager-to-launcher rebinding and monkeypatch-sensitive exports.
- **2026-08-11** - Decomposed the launcher settings surface from exact draft
  PR #8486 head `624043537a5ab10aa7ef56dc61685a004b872c0c` without changing
  its public dialog/widget contract.
- **2026-08-11** - Decomposed the Simscape C3D viewer for #8485 without
  changing its public UI contract.
- **2026-08-10** - Added #8458 hand-path attribution: canonical pointwise
  ZTCF/control/ZVCF definitions; exact double-pendulum, one-arm, and closed-loop
  two-arm adapters; deterministic force-vector, impulse, power, work,
  joint/time-window, common/differential-mode, sensitivity, and closure
  evidence; and a bounded residual-couple preview hypothesis with explicit
  physiological limits. The final 106-page PDF uses contract-checked lossless
  object-stream compaction and preserves 110 URI links plus 122 outline entries.
- **2026-08-10** - Added #8446 two-hand passive-couple reproduction: a
  frame-explicit and tested wrench API, hash-traceable BASE/ZTCF/DELTA table
  exports, all-sample resultant/couple/power reconstruction, reversal and
  resampling analysis, grip-separation/orientation counterfactuals, eight
  publication figures, and a neutral chapter separating contact-force moment,
  free torque, resultant force, and power.
- **2026-08-10** - Added #8445 matched-state counterfactual persistence:
  deterministic commanded/zero-torque futures, 96 cut-time/horizon/timestep
  cases, state/force/power/work/speed divergence, torque-switch bracketing,
  gravity and damping ablations, a WSCG BASE-minus-counterfactual convention
  check, four publication figures, and a visually verified 58-page article.
- **2026-08-10** - Added the first interaction-force mechanisms slice for
  epic #8443 / issue #8444: exact double-pendulum wrist-force and force-power
  decomposition, tested Newton and moment balances, a matched-state
  torque-killswitch distinct from pointwise ZTCF, hash-registered WSCG source
  decks and chart extraction, seven publication figures, machine-readable
  evidence, and an expanded scientific chapter with explicit nonclaims and
  falsification tests.
- **2026-08-09** - Stabilized the Launch Monitor Analytics v1 consumer
  contract against the current Tools record contract. Dataset fingerprints
  now hash canonical ordered record content and ignore transient pandas index
  labels; the shared domain owns the single `1.0.0` version constant; FastAPI
  validates analysis mode, correlation method, and missing-data policy as
  closed enums; and the PyQt signal boundary converts user-correctable
  selection failures into accessible inline status. The parity fixture pins
  the observed Tools v1 fingerprint but does not claim an immutable Tools
  release or dependency pin.
- **2026-08-06** - Extended Launch Monitor Analytics with a UI-neutral,
  versioned flexible-analysis contract and matched PyQt/FastAPI surfaces.
  Users may select arbitrary numeric outcomes and predictors, Pearson,
  Spearman, or Kendall association, pairwise/listwise/fail missingness,
  Benjamini-Hochberg correction, grouped analysis, and OLS regression with
  confidence intervals, residual diagnostics, and deterministic dataset
  lineage. Aggregate observations remain excluded from regression; explicit
  aggregate correlations are labeled descriptive with an ecological-bias
  warning. Vendor-specific fields cannot be pooled across monitor vendors, and
  association or predictive fit is never presented as causal evidence.

- **2026-08-09** - Removed the real 832-shot launch-monitor trajectory CSV
  from the public repository. The historical validation loader now resolves its
  default dataset only from the authenticated private data authority named by
  `LAUNCH_MONITOR_DATA_ROOT`, and fails closed when that authority or its pinned
  source snapshot is unavailable. Explicit caller-provided paths remain
  supported for tests and user-owned data.

- **2026-08-09** - Refined the proximal-to-distal energy-transfer research
  package as a neutral, reproducible open resource. The report now distinguishes
  empirical evidence, model-derived results, and hypotheses; documents its
  applicability limits; and includes a deterministic 13-case, one-at-a-time
  parameter-sensitivity analysis over segment geometry, mass, plane inclination,
  and damping. Unit tests enforce the analysis contract and verify that the
  reported strategy ordering is reproduced for the defined cases. These
  simplified-model results are not a universal effect estimate or a coaching
  prescription.
- **2026-08-11** - Added the first bounded #4262 immutable Tools-provider
  source contract. Launcher entries with `provider: tools` now resolve only
  from the repository-pinned `vendor/ud-tools` gitlink, before generic sibling
  and installed-package providers; the five Tools launchers no longer serialize
  `../Tools` source roots. Authority requires the already-declared exact
  `ff4240217005e1415ca409fd124e50b64ee642d2` gitlink, matching clean checkout
  HEAD, current-superproject attachment, and a non-reparse vendor directory.
  Every resolved artifact, working directory, fallback, and Python path must
  remain under that authority after canonicalization. Missing, uninitialized,
  dirty, replaced, mismatched, or escaping sources report
  `provider_unavailable` and never fall back to mutable sibling/package input.
  Generic sibling-provider resolution is unchanged. This slice does not change
  the gitlink pin, accept an unvalidated `TOOLS_REPO_PATH`, or complete #4262.
- (spec-exempt: security fix) Fixed Command Injection in `pandas.DataFrame.query()` inside `rust_engine.py` (both `data_processor` and `data_processor_io`) by explicitly validating user expressions using an AST-based validator (`validate_pandas_formula`). This eliminates an arbitrary code execution vulnerability.

- **2026-08-05** - Retargeted #8345 P1's 3D putting workflow to `main`
  after the headless dynamics foundation merged. The FastAPI route executes the
  canonical Python collision/surface solver and returns a complete playback DTO;
  the React client uses TanStack Query, Zustand, and an R3F green/ball/putter
  scene with orbit controls, signed-spin marker, collision slowdown, adjustable
  hosel controls, responsive playback, and theme tokens.
- **2026-08-04** - Added #8345 P1's generated-contract 3D putting workflow.
  The FastAPI route executes the canonical Python collision/surface solver and
  returns a complete playback DTO; the React client uses TanStack Query,
  Zustand, and an R3F green/ball/putter scene with orbit controls, signed-spin
  marker, collision slowdown, adjustable hosel controls, responsive playback,
  and theme tokens. Migrated the UI stylesheet entry point to the required
  Tailwind v4 import/config syntax so responsive and theme utilities compile.
- **2026-08-05** - Hardened classic PyQt6 startup from nested worktrees.
  Tools-source discovery now walks workspace ancestors after honoring explicit
  and initialized vendored sources, so a worktree can locate the canonical
  sibling `Tools/src` checkout. An unavailable implicit Sidekick runtime
  disables only the optional sidebar; an explicitly configured incomplete
  `TOOLS_REPO_PATH` remains fail-closed. Focused regressions pin nested
  discovery and both fallback contracts.
- **2026-08-04** - Added the headless putting-dynamics foundation for #8345
  P2/P3/P4. `src/shared/python/putting_dynamics/` provides DbC-validated
  heterogeneous height/friction fields, seeded bumpiness, signed skid/overspin
  transition, rolling/rest modes, full-chord hole capture, and finite-mass
  putter-ball collision outputs including dynamic loft, slowdown, adjustable
  hosel position, and an attachment-point impulse wrench. The public-data
  review records verified sources, explicit assumptions, discrepancies, model
  defaults, and validation bands. Seventy focused unit tests pin analytic
  limits, determinism, symmetry, conservation, and the public façade.
- **2026-08-04** - Added the vendor-neutral Launch Monitor Analytics workbench
  (#8342). A canonical, unit-normalized shot schema and provenance-preserving
  import pipeline aggregate common TrackMan, Foresight, FlightScope, Garmin,
  SkyTrak, Uneekor, Full Swing, Rapsodo, and GSPro/Open Connect exports, with a
  generic mapping fallback. The PyQt6 workbench provides auditable treatment,
  correlation and partial-correlation mapping, regularized regression and an
  optional shallow neural network, matched-monitor agreement, dispersion, and
  actual-time longitudinal trend analysis. Associations and predictions are
  explicitly non-causal; unmatched monitor comparisons are descriptive only.
- **2026-07-27** - Corrected classic PyQt6 Diagnostics provider-manifest
  validation (#8121). The parent `models.yaml` check now validates only its 46
  directly declared tiles, while the separate runtime registry check retains
  all 75 parent and provider models. Computer control confirmed the repaired
  Diagnostics screen reports `Status: HEALTHY` with zero failed checks instead
  of falsely classifying 29 provider-only models as missing.
- **2026-07-27** - Extended the classic PyQt6 Sidekick acceptance contract for
  the selected Tools authority. The background API child now inherits the same
  validated Tools checkout as the sidebar and places its package roots before
  UpstreamDrift's partial shared packages, preventing
  `chat.websocket_protocol` import failures (#8120). The matrix
  also records the Tools #3950 warnings-as-errors Units regression and its
  verified `100 °C` to `212 °F` retest. A complete API-tree fault injection
  confirmed dynamic-port recovery, Chat reconnection, and close-time descendant
  cleanup without disturbing an unrelated port-8000 process.
- **2026-07-25** - Hardened classic PyQt6 Sidekick startup for #8102. The
  launcher selects an isolated loopback port when the historical default is
  occupied, exports one ephemeral launcher/API capability, verifies the child
  instance through `/readyz`, installs local Sidekick tools independently of
  Chat readiness, performs bounded child restarts, and loads the pinned Tools
  direct-package source before legacy aliases, copied sources, or mutable
  sibling sources. An explicit `TOOLS_REPO_PATH` is authoritative and fails
  closed when invalid rather than silently mixing with vendored or sibling
  sources. Each automatic restart revalidates a dynamically selected API port
  and selects a new free loopback port when necessary while preserving the
  launcher's capability and instance identity; explicitly configured ports
  remain unchanged. Host close also delegates to the canonical sidebar
  shutdown contract so Terminal PTY, shell, bridge, and API child processes
  cannot survive the launcher (#3938). A PR-diff hygiene contract now rejects
  any non-deletion edit to a Python path owned by the exact pinned Tools
  gitlink, regardless of warning header or shadow allow-list, while retaining
  the original base/current warning-header checks. The protected unit gate
  fetches the pull request base and sparse-checks out only the pinned Tools
  `src/shared/python` inventory before running the fail-closed comparison.
  Sidekick/chat paths without a same-relative Tools source require file-level
  owner state, rationale, tracking issue, and unexpired review date in
  `scripts/config/shared_python_ownership_exceptions.yaml`.
  Release-wheel assembly verifies the exact Tools gitlink and installs the
  parent-owned shared package graph plus its Chat/Sidekick compatibility
  aliases, utilities, and DbC contracts. The canonical alias finder coalesces
  direct, `shared.python`, and legacy `src.shared.python` spellings to prevent
  installed applications from executing stale nested child copies; only
  non-conflicting Upstream-owned extensions supplement the parent graph. Tagged
  releases initialize the exact Tools submodule and build a wheel directly
  from that verified checkout rather than rebuilding an unverifiable wheel
  from an unpacked source archive. Clean installed-wheel probes verify module
  identity, dependency closure, and byte-level source fidelity.
  The acceptance and source-ownership audit matrix is in
  `docs/testing/sidekick-pyqt6-startup-matrix.md`.
- **2026-06-22** - Optimize Mechanical Work computations in `evaluate_matching_workflow.py`. `np.sum(..., axis=1)` calls were replaced with equivalent but more efficient `np.einsum('ij->i', ...)` calls, yielding significant performance gains during bulk evaluation.
- **2026-07-15** - Hardened the simulation WebSocket and Data Explorer API
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
- **2026-07-29** - Recorded the Bolt IK vector norm optimization: The `pinocchio_golf.diff_ik` module now uses `math.sqrt(np.vdot(err, err))` inside its `differential_ik` and `solve_dual_frame_ik` loop conditions instead of `np.linalg.norm(err)`. This eliminates array allocation overhead on the hot path for ~2x performance speedup without changing the mathematical behavior.
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

```
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
│   │                               # launch_monitor_analytics,
│   │                               # video_analyzer, sidekick, …) plus
│   │                               # headless analysis CLIs (drift_control,
│   │                               # contraction)
│   └── shared/python/              # Cross-cutting libraries; highlights:
│       ├── engine_core/            # EngineManager/Registry/probes/capabilities
│       ├── ground_model/            # Fail-closed consumer of Tools ground v1
│       ├── launcher_embed/         # EmbeddableTool contract + registry (ADR-0013)
│       ├── physics/                # Ball flight models, impact, swing→flight pipeline
│       ├── putting_dynamics/       # Surface-aware putt collision and roll physics
│       ├── launch_monitor/         # Canonical shot import, treatment, and analytics
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
```

### Key Components

| Component                | Location                                                                                                                                            | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| MuJoCo Engine Adapter    | `src/engines/physics_engines/mujoco/`                                                                                                               | Primary physics engine integration with full support for contact dynamics and muscle models                                                                                                                                                                                                                                                                                                                                                                                                      |
| Drake Engine Adapter     | `src/engines/physics_engines/drake/`                                                                                                                | Extended Drake support for trajectory optimization and manipulation tasks                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Pinocchio Engine Adapter | `src/engines/physics_engines/pinocchio/`                                                                                                            | Extended Pinocchio support for efficient rigid-body dynamics computation                                                                                                                                                                                                                                                                                                                                                                                                                         |
| OpenSim Engine Adapter   | `src/engines/physics_engines/opensim/`                                                                                                              | Experimental OpenSim integration for clinical biomechanics workflows                                                                                                                                                                                                                                                                                                                                                                                                                             |
| MyoSuite Engine Adapter  | `src/engines/physics_engines/myosuite/`                                                                                                             | Experimental MyoSuite integration for detailed muscle physiology simulation                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Pendulum Models          | `src/engines/physics_engines/pendulum/`                                                                                                             | Educational simplified models for learning and quick prototyping                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| FastAPI Backend          | `src/api/`                                                                                                                                          | REST API exposing simulation, IK/ID, trajectory optimization, and control endpoints                                                                                                                                                                                                                                                                                                                                                                                                              |
| PyQt6 GUI                | `src/launchers/upstream_drift_launcher.py`                                                                                                          | Professional interactive GUI with real-time 3D visualization                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Sidekick (AI assistant)  | PyQt: `src/shared/python/ai/gui/assistant_panel.py` · React: `ui/src/components/ui/ChatPanel.tsx` · Adapter: `src/tools/sidekick/_embed_adapter.py` | In-app AI chat surface with streaming, RAG, session history, and agentic tool dispatch. Design tokens: `src/shared/python/theme/sidekick_tokens.py`. See `docs/sidekick/README.md`.                                                                                                                                                                                                                                                                                                              |
| Tauri Desktop App        | `ui/`                                                                                                                                               | Cross-platform desktop application wrapper (Windows, macOS, Linux)                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Rust Physics Kernels     | `rust_core/upstream-physics/`                                                                                                                       | High-performance compiled physics routines for critical paths, including initial flexible shaft FEM element primitives                                                                                                                                                                                                                                                                                                                                                                           |
| Configuration Manager    | `src/config/`                                                                                                                                       | Centralized configuration loading, validation, and environment management                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Analysis Tool CLIs       | `src/tools/drift_control/`, `src/tools/contraction/`                                                                                                | Headless AffineDrift-compatible drift/control, contraction, and Floquet analysis tools                                                                                                                                                                                                                                                                                                                                                                                                           |
| Launch Monitor Analytics | `src/tools/launch_monitor_analytics/`, `src/tools/launch_monitor_model/`, `src/api/services/launch_monitor_dataset_jobs.py`                         | PyQt6, FastAPI, and headless vendor-neutral analysis; contract v2 adds traceable arbitrary-field analysis, source-backed SG verifies benchmark provenance, immutable aggregate-only dataset jobs analyze hash-pinned private authorities without transferring rows or client paths, canonical player covariation separates pooled/within/between effects with fixed/random population synthesis, and attested longitudinal analysis aggregates by session before descriptive clustered inference |
| Tools Ground Consumer    | `src/shared/python/ground_model/`                                                                                                                   | Headless exact-schema gateway to Tools flight-to-ground v1 records and reference execution; UI and final dependency pins remain tracked                                                                                                                                                                                                                                                                                                                                                          |
| Putting Dynamics         | `src/shared/python/putting_dynamics/`                                                                                                               | Headless heterogeneous-green, collision, loft, hosel-wrench, skid/roll/rest, and hole-capture physics for #8345                                                                                                                                                                                                                                                                                                                                                                                  |
| 3D Putting UI            | `src/api/routes/putting_green.py`, `ui/src/pages/PuttingGreen.tsx`, `ui/src/components/visualization/PuttingScene3D.tsx`                            | Generated-contract R3F playback of the canonical putting model with collision, spin, hosel, surface, camera, and video controls for #8345 P1                                                                                                                                                                                                                                                                                                                                                     |
| BunkerShot3D Metrics     | `src/bunkershot3d/metrics/`                                                                                                                         | Designer-facing metrics for bunker shot analysis: trajectory (dig/skid, depth trace), energy partition (club KE, sand/ball transfer), force/deceleration, head twist (shaft/CG moments), and forgiveness sensitivity gradients. Computed from HDF5 result artifacts for tier-agnostic (F0–F3) analysis per #8614.                                                                                                                                                                                |
| Shared Utilities         | `src/shared/`                                                                                                                                       | Cross-engine validators, helpers, and exception definitions                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Workspace Metadata       | `src/shared/python/workspace/`                                                                                                                      | Project/session/dataset metadata store and CC-4 HDF5 result browser view models                                                                                                                                                                                                                                                                                                                                                                                                                  |
| URDF Models              | `shared/models/`                                                                                                                                    | Canonical model definitions (URDF format) for golf swings, human body, pendulums                                                                                                                                                                                                                                                                                                                                                                                                                 |

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

| #   | Feature                                     | Status | Description                                                                                                                                                                                                                                                                                                                    |
| --- | ------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| F1  | MuJoCo engine integration                   | ✅     | Full support for MuJoCo 3.3.0+ with contact dynamics, muscle actuators, sensor simulation, and pose-conditioned motion-matching target synthesis                                                                                                                                                                               |
| F2  | Drake engine integration                    | ✅     | Extended Drake support for trajectory optimization, manipulation, and planning problems                                                                                                                                                                                                                                        |
| F3  | Pinocchio engine integration                | ✅     | Extended Pinocchio support for efficient rigid-body dynamics and jacobian computation                                                                                                                                                                                                                                          |
| F4  | OpenSim engine integration                  | 🔄     | Experimental OpenSim integration for clinical biomechanics and musculoskeletal analysis                                                                                                                                                                                                                                        |
| F5  | MyoSuite engine integration                 | 🔄     | Experimental MyoSuite integration for detailed muscle physiology and motor control                                                                                                                                                                                                                                             |
| F6  | Cross-engine validation and reports         | ✅     | Automated cross-validation plus user-facing comparison reports across selected engines with tolerance thresholds, provenance, and divergence annotations                                                                                                                                                                       |
| F7  | FastAPI REST API                            | ✅     | Programmatic access to simulation, IK/ID, trajectory optimization, and control endpoints                                                                                                                                                                                                                                       |
| F8  | PyQt6 professional GUI                      | ✅     | Interactive desktop GUI with real-time 3D rendering, parameter adjustment, and result export                                                                                                                                                                                                                                   |
| F9  | Tauri desktop application                   | 🔄     | Cross-platform desktop app bundling the GUI and API with native OS integration                                                                                                                                                                                                                                                 |
| F10 | MATLAB/Simulink integration                 | ✅     | Export models to MATLAB format and integrate with Simulink via MEX interface                                                                                                                                                                                                                                                   |
| F11 | Trajectory optimization                     | ✅     | SciPy-based trajectory optimization with constraint support and custom cost functions                                                                                                                                                                                                                                          |
| F12 | Muscle dynamics analysis                    | ✅     | IK, ID, and muscle dynamics computation with Hill-type and Millard muscle models                                                                                                                                                                                                                                               |
| F13 | Motion capture integration                  | 🔄     | Import and track motion capture data (C3D, BVH, TRC formats) and compare with simulation                                                                                                                                                                                                                                       |
| F14 | Reinforcement learning integration          | 🔄     | Gym-compatible interface for RL-based controller learning and policy optimization                                                                                                                                                                                                                                              |
| F15 | Sidekick AI assistant                       | 🔄     | In-app and standalone AI assistant surface (PyQt + React/Tauri + `sidekick.standalone.*`) with streaming, RAG, session history, persisted standalone preferences, onboarding, and agentic tool dispatch. See `docs/sidekick/README.md` and ADR-0018.                                                                           |
| F16 | Model-training controller                   | 🔄     | In-launcher training dashboard (PR3) with scheduler, dataset library, resource monitor, engine-compat gate, and ML/RL-aware stats. Backend contracts + scheduler land in `src/shared/python/training/` (PRs 1–2); GUI tab, tab-backgrounding refactor, and CVAE wiring in PRs 3–5.                                             |
| F17 | Tools ground-model integration              | 🔄     | Headless v1 consumer gateway validates the canonical Tools façade and degrades safely when absent; exact dependency pins, FastAPI, PyQt, React, parity, and protected release remain open under Tools #4276.                                                                                                                   |
| F18 | Source-backed strokes gained                | ✅     | Canonical Python and FastAPI contracts score complete course-state transitions against a versioned, hash-verified expected-strokes source; provenance, backing values, exclusions, uncertainty, and identity trust remain explicit, while outcome proxies cannot claim SG.                                                     |
| F19 | Immutable launch-monitor dataset jobs       | ✅     | Authenticated, aggregate-only FastAPI jobs bind an administrator-authorized private authority to exact repository/commit/manifest/content/count identity; fixed allowlisted operations, disclosure floors, bounded pages, structured unavailable states, and deterministic worker shutdown prevent inline row or path leakage. |
| F20 | Player covariation and population synthesis | ✅     | A versioned evidence-bearing contract separates pooled, within-player, between-player, and per-player associations; fixed/random Fisher-z synthesis reports heterogeneity, exclusions, unavailable states, explicit units, trusted player identity, vendor/model provenance, and source-linked backing rows.                   |

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
- `GET /tools/launch-monitor-analytics/contracts/strokes-gained/v1` — Publish the canonical source-backed SG result schema
- `POST /tools/launch-monitor-analytics/v2/strokes-gained` — Score explicit start/finish states against a supplied, source-verified expected-strokes baseline
- `POST /tools/launch-monitor-analytics/v2/outcome-proxy` — Compute target-relative radial error under a contract that forbids an SG claim
- `GET /tools/launch-monitor-analytics/contracts/dataset-jobs/v1` — Publish the immutable aggregate-job request schema
- `POST /tools/launch-monitor-analytics/v2/dataset-jobs` — Queue an authenticated aggregate job against an exact server-authorized dataset reference
- `GET /tools/launch-monitor-analytics/v2/dataset-jobs/{job_id}` — Return data-free job status and structured unavailable reasons
- `GET /tools/launch-monitor-analytics/v2/dataset-jobs/{job_id}/results` — Page bounded aggregate/source-provenance results without observations or private paths
- `GET /tools/launch-monitor-analytics/contracts/player-covariation/v1` — Publish the canonical selected-pair and exploratory-scan result schema
- `POST /tools/launch-monitor-analytics/v2/player-covariation` — Compare pooled, within-player, between-player, per-player, and population effects for an explicitly identified player variable pair
- `POST /tools/launch-monitor-analytics/v2/player-covariation/scan` — Rank a bounded exploratory variable-pair set with structured unavailable states and multiplicity warnings

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

- The classic launcher entry point reuses an active `QApplication` and creates
  one only when the process has none; it must never construct a second Qt
  application object in an embedded or test-hosted process.
- Model loader and parameter editor
- Real-time 3D simulation viewer with playback controls
- Cross-engine comparison visualizer
- IK/ID solver interface with result tables
- Trajectory optimization GUI with constraint editor
- Data export and report generation

**CLI Interface**:

- `upstream-drift` launches the web UI (console script → `launch_upstream_drift:parse_arguments`/`route_launch`).
- `upstream-drift --classic` launches the classic PyQt6 desktop launcher instead of the web UI.
- `upstream-drift --api-only [--port 8000]` starts the FastAPI server without any UI.
- `upstream-drift --engine mujoco [--no-browser]` launches a specific engine directly; `--engine` choices come from `EngineType`.
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

| Input                     | Format                         | Source                                             | Schema                                                                                       |
| ------------------------- | ------------------------------ | -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Biomechanical Models      | URDF                           | `shared/models/`                                   | URDF 1.0 standard with custom muscle actuator extensions                                     |
| Motion Capture Data       | C3D, BVH, TRC                  | External mocap systems or files                    | Standard formats with marker sets and frame data                                             |
| Launch Monitor Sessions   | CSV, TSV, TXT, XLS, XLSX, JSON | Common launch-monitor exports or user-mapped files | Canonical shot schema with source columns, unit/status metadata, and import manifest         |
| Expected-Strokes Baseline | JSON                           | Declared HTTP(S) source with license and SHA-256   | Versioned lie/context/target/distance states with unique points and optional standard errors |
| Optimization Constraints  | JSON                           | User input or configuration                        | Custom constraint schema in `src/config/`                                                    |
| Control Parameters        | YAML/JSON                      | Configuration files or API                         | Engine-specific parameter maps validated against schemas                                     |

### Output Data

| Output                   | Format                  | Destination                 | Description                                                                                                                                                           |
| ------------------------ | ----------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Simulation Trajectories  | JSON/HDF5               | API response or file export | Joint angles, muscle activations, forces over time                                                                                                                    |
| Cross-Validation Reports | JSON/PDF                | File export or API          | Engine comparison metrics, error margins, validation status                                                                                                           |
| IK/ID Solutions          | JSON/MATLAB             | API response or file        | Joint angles (IK) and joint torques (ID) with confidence metrics                                                                                                      |
| Optimized Trajectories   | URDF/MATLAB             | File export                 | Trajectory-optimized model definitions with optimal control inputs                                                                                                    |
| Visualization Data       | JSON (Three.js format)  | GUI or web client           | 3D geometry, animation keyframes, and rendering parameters                                                                                                            |
| Launch Monitor Analytics | CSV/JSON/project bundle | GUI, API, or file export    | Treated observations, provenance manifests, association/model results, dispersion, trends, source-backed SG with backing benchmark values, and non-SG outcome proxies |

### Configuration

Configuration is managed through:

- **Environment Variables**: read through `src/shared/python/config/environment.py` and re-exported by `src/shared/python/config/settings.py`, which is the canonical configuration reference. Key variables: `GOLF_API_HOST` / `GOLF_API_PORT`, `GOLF_API_SECRET_KEY`, `GOLF_ADMIN_PASSWORD`, `GOLF_AUTH_DISABLED`, `DATABASE_URL`, `ENVIRONMENT`, `HEADLESS`, `LOG_LEVEL`. `.env.example` documents the deployment-facing set (`API_HOST`, `API_PORT`, …).
- **YAML Config Files**: `src/config/interim_config.yaml` holds the declared defaults for CORS origins, trusted hosts, rate limits, quota tiers, simulation engine order, and video analysis. It documents intent and is not auto-loaded — callers that need its values load it explicitly.
- **API Request Parameters**: Engine selection, model path, solver options passed as JSON
- **Launcher Manifest**: `src/config/launcher_manifest.json` declares discoverable and hidden launcher surfaces, including shared Tools-hosted video/data utilities exposed to UpstreamDrift.
- **Theme API Settings**: `src/api/routes/theme.py` and `ui/src/api/themeClient.ts` expose launcher theme metadata to the desktop/web UI without duplicating theme lists in the frontend.
- **Web Settings**: `src/api/routes/settings.py` persists the validated web preferences document (`WebSettings`: appearance, notifications, simulation defaults) to `~/.upstreamdrift/web_settings.json` (override: `UPSTREAMDRIFT_WEB_SETTINGS_PATH`); `ui/src/pages/Settings.tsx` + `ui/src/api/settingsClient.ts` consume it with localStorage as cache only (#7457).

Excerpt from `src/config/interim_config.yaml`:

```yaml
# Server Configuration
server:
  # Use 127.0.0.1 (localhost) by default for security
  # Set to 0.0.0.0 in Docker or when external access is explicitly needed
  host: "127.0.0.1"
  port: 8000
  reload: true # Auto-reload on code changes (development)
  log_level: "info"

# Database Configuration
database:
  url: "sqlite:///./golf_modeling_suite.db"
  echo: false # Set to true for SQL debugging
  pool_size: 5
  max_overflow: 10
```

## 7. Testing Specification

### Testing Strategy

UpstreamDrift employs a comprehensive test pyramid with multiple specialized categories:

- **Unit Tests**: Test individual engine adapters, utilities, and validators in isolation
- **Integration Tests**: Test workflows combining multiple modules (e.g., load model → simulate → export)
- **Acceptance Tests**: End-to-end scenarios (e.g., full golf swing simulation with visualization)
- **Cross-Engine Tests**: Validate physics consistency across multiple engines with tolerance thresholds
- **Physics Validation Tests**: Verify results against known ground truth (analytical solutions, published benchmarks)
- **Golf Ball-Flight Source Contracts**: Validate documented aerodynamic, impact, and atmosphere assumptions against `docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md`
- **Launch-Monitor Scoring Contracts**: Verify baseline hashing and provenance,
  exact-stratum interpolation, fail-closed extrapolation, identity-safe grouping,
  uncertainty availability, API schemas, and the prohibition on relabeling an
  outcome proxy as strokes gained.
- **Dependency Source Contracts**: Validate generated dependency artifacts against `pyproject.toml` and fail CI when lockfiles or `environment.yml` drift
- **Documentation Governance Contracts**: Validate the canonical `docs/index.md` directory catalog, rendered documentation hub link, Markdown/Quarto size budget, and significant-word title capitalization for changed Markdown, Quarto, LaTeX, Word, and PDF documents.
- **Benchmark Tests**: Performance regression detection and optimization validation
- **Property-Based Tests**: Hypothesis-driven fuzzing for robustness

### Test Organization

| Category                    | Location                                                    | Framework           | Markers                                          |
| --------------------------- | ----------------------------------------------------------- | ------------------- | ------------------------------------------------ |
| Unit                        | `tests/unit/`                                               | pytest              | `@pytest.mark.unit`                              |
| Integration                 | `tests/integration/`                                        | pytest              | `@pytest.mark.integration`                       |
| Acceptance                  | `tests/acceptance/`                                         | pytest              | selected by path (no dedicated marker)           |
| Cross-Engine                | `tests/cross_engine/`                                       | pytest              | `@pytest.mark.gate` + `requires_<engine>`        |
| Physics Validation          | `tests/analytical/`, `tests/integration/conservation_laws/` | pytest              | `@pytest.mark.unit` / `@pytest.mark.integration` |
| Golf Source Contracts       | `tests/unit/shared_python/`                                 | pytest              | source-map contract tests                        |
| Launch-Monitor Scoring      | `tests/unit/launch_monitor/`, `tests/api/`                  | pytest              | source-backed SG domain and API contract tests   |
| Dependency Source Contracts | `tests/unit/scripts/`                                       | pytest              | generated dependency contract tests              |
| Benchmarks                  | `tests/benchmarks/`                                         | pytest-benchmark    | `@pytest.mark.benchmark`                         |
| Property-Based              | `tests/unit/`                                               | hypothesis + pytest | `@hypothesis.given` (no dedicated marker)        |

Issue #3841 moved stable flat tests and the launcher `src/**/tests` package into
topic directories under `tests/`, documented the fixture scopes in
`tests/README.md`, and added `scripts/check_test_layout.py` as the blocking CI
guard against new flat test files, new in-tree `src/**/tests` directories, and
overlapping fixture names in nested conftests.

### Coverage Requirements

| Scope                   | Minimum | Current          | Enforced By                                 |
| ----------------------- | ------- | ---------------- | ------------------------------------------- |
| Overall                 | 75%     | CI baseline      | `pyproject.toml` and `ci-standard.yml`      |
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
- [ ] Source-backed SG rejects unverifiable baselines, incomplete course states, extrapolation, and untrusted grouped identity while preserving row-level backing evidence
- [ ] Outcome-proxy responses remain explicitly non-SG in both domain and OpenAPI contracts
- [ ] FastAPI endpoints return 200 for valid requests and 400 for invalid schema
- [ ] GUI loads model and renders 3D visualization without crashing
- [ ] Trajectory optimization improves cost function by >20% over initial guess
- [ ] Muscle dynamics simulation produces realistic activation patterns
- [ ] Cross-platform build (Windows, macOS, Linux) produces functional binaries

## 8. Quality Standards

### Code Quality Tools

| Tool       | Version | Purpose                | Blocking? |
| ---------- | ------- | ---------------------- | --------- |
| ruff       | latest  | Linting and formatting | Yes       |
| mypy       | 1.7+    | Static type checking   | Yes       |
| pytest     | 7.0+    | Testing framework      | Yes       |
| pytest-cov | 4.0+    | Coverage measurement   | Yes       |
| bandit     | 1.7+    | Security scanning      | Yes       |
| hypothesis | 6.0+    | Property-based testing | No        |

### Design Principles

- **TDD**: Unit tests written before implementation; the current global coverage floor is 75%, with per-package production ratchets tracked toward higher thresholds (85% for API routes/engine adapters, 70% for shared utilities).
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
- **Docker Size Gates**: The canonical runtime and every modular profile must
  remain within the explicit budget registered for that image. Modular workflow
  budgets must match `docker/profiles.yaml`; slim is currently 900 MB and
  standard is 2200 MB after the governed core runtime added pandas-backed API
  routes.
- **Unit-Gate Quarantine Ratchet**: `scripts/ci/check_unit_gate_quarantine.py` requires every ledgered node ID to resolve to exactly one owned, executable failure cluster. Pull requests compare the current node-ID set with the fetched base branch and reject additions or replacements while allowing removals; duplicate, unassigned, ambiguous, or malformed entries fail closed. The checker can list or run one cluster in bounded batches without changing the Green-Suite-only skip boundary.

### CI/CD Pipeline

| Workflow                       | Trigger                                | Purpose                                                                               | Blocking?          |
| ------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------- | ------------------ |
| `ci-standard.yml`              | Push/PR (no PR path filter)            | Lint, type check, unit/integration tests, workflow inventory, blocking security scans | Yes                |
| `quality-gate.yml`             | PR/manual dispatch                     | Blocking repo-wide Law-of-Demeter ratchet for production `src/` Python code           | Yes                |
| `docs-ci.yml`                  | PR touching docs/markdown              | Docs governance for docs-only PRs                                                     | No (not required)  |
| `heavy-tests-opt-in.yml`       | Manual dispatch or `/heavy-test` label | Cross-engine and physics validation (long-running)                                    | No (opt-in)        |
| `nightly-cross-validation.yml` | Daily 2:00 UTC                         | Full multi-engine validation suite against all model variations                       | No (informational) |
| `tauri-build.yml`              | Tag release                            | Build desktop apps for Windows/macOS/Linux                                            | Yes (for releases) |
| `vendor-freshness.yml`         | Weekly                                 | Check for stale dependencies and security updates                                     | No (warning-only)  |
| `docker-size-gates.yml`        | Push                                   | Enforce the canonical runtime and per-profile image budgets                           | Yes                |

### Required Status Checks

Branch protection matches required checks by **context name**, so a check name is
a repository-level contract: any job carrying a required name publishes under that
context, and protection is satisfied by whichever job reported. Two rules follow.

**One publisher per required name.** Exactly one job may be named `quality-gate` —
the `ci-standard.yml` aggregate. Three jobs previously shared the name
(`ci-standard.yml`, `quality-gate.yml`, `docs-ci.yml`), which allowed a merge while
the aggregate was failing. `tests/ci/test_ci_infrastructure.py` enforces uniqueness.

**A required check must report on every PR.** A required context that never reports
blocks the PR indefinitely. A job skipped by a job-level `if:` still publishes a
check run and counts as satisfied, but a workflow skipped by a trigger **path
filter** publishes nothing. Required workflows therefore carry no PR path filter and
skip work per job instead: `ci-standard.yml` classifies the diff in its
`changed-paths` job and gates the substantive jobs on
`needs.changed-paths.outputs.code`, so docs-only PRs skip the suite while
`quality-gate` still reports. The aggregate accepts `skipped` gates only on
docs-only changes, and only once `pick-runner` and `changed-paths` themselves
succeeded — otherwise an infrastructure failure that skips everything would read as
a pass.

| Context             | Published by                | Required | Notes                                                          |
| ------------------- | --------------------------- | -------- | -------------------------------------------------------------- |
| `quality-gate`      | `ci-standard.yml` aggregate | Yes      | Also required org-wide by the `Repository_Protections` ruleset |
| `lod-quality-gate`  | `quality-gate.yml`          | Yes      | No path filter, so it reports on every PR                      |
| `docs-quality-gate` | `docs-ci.yml`               | **No**   | Docs-only trigger; requiring it would block every code-only PR |

Repository-specific required contexts belong on the repo-scoped `Protect Main`
ruleset. The organization ruleset `Repository_Protections` applies to every repo in
the org, so contexts added there must exist in all of them.

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

| Package    | Version | Purpose                                                      |
| ---------- | ------- | ------------------------------------------------------------ |
| pytest     | 7.0+    | Testing framework                                            |
| pytest-cov | 4.0+    | Coverage measurement                                         |
| hypothesis | 6.0+    | Property-based testing                                       |
| pip-tools  | 7.4+    | Regenerate Python dependency lockfiles from `pyproject.toml` |
| pyarrow    | 14.0+   | Parquet IO test coverage for compact swing dataset paths     |
| ruff       | latest  | Linting and formatting                                       |
| mypy       | 1.7+    | Type checking                                                |
| bandit     | 1.7+    | Security scanning                                            |
| black      | 23.0+   | Code formatter                                               |

### Fleet Dependencies

| Repo  | Relationship                              | Description                                                                                                                                                 |
| ----- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tools | Vendored Python and pinned Rust authority | `vendor/ud-tools` plus the Cargo `tools-core` revision provide reviewed shared contracts and kernels. Ground-model release pins remain pending Tools #4276. |

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
# For Desktop App: Cargo Install Tauri-Cli

# Running the FastAPI Server
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
# ...Or, Equivalently, Through the Launcher:
upstream-drift --api-only --port 8000

# Running the PyQt6 GUI
upstream-drift --classic
# ...Or Directly:
python launch_upstream_drift.py --classic

# Running the CLI
upstream-drift --engine mujoco --no-browser
python -m sidekick
python -m sidekick run --calculator unit-converter --inputs ./inputs.json

# Building the Tauri Desktop App
cd ui && npm install && npm run tauri build
# Outputs: UpstreamDrift.exe (Windows), UpstreamDrift.app (macOS), UpstreamDrift.AppImage (Linux)

# Running Tests
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/ --cov=src --cov-fail-under=75
```

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
- Ground-model execution is not yet available from the clean UpstreamDrift
  release: the headless consumer gateway exists, but exact Tools pins and
  FastAPI/PyQt/React surfaces await the protected Tools ground merge (#4276).

## 12. Change Log

Rows are keyed by pull request, not by a serial spec version: `| YYYY-MM-DD | #<pr> | summary |`. Add exactly one row for your own pull request and do not renumber anybody else's; the `Spec Version` field in section 1 is release-derived and is never bumped by an individual pull request. See [Repository_Management#1520](https://github.com/D-sorganization/Repository_Management/issues/1520).

Rows added before 2026-10-12 (about 1,100 of them), the per-pull-request sections that had accumulated before section 1 and after this section, and the pre-2026-10-12 change-log table live verbatim in [`docs/spec/CHANGELOG_ARCHIVE.md`](docs/spec/CHANGELOG_ARCHIVE.md) (UpstreamDrift#9415). This table is the live log: new rows go here, newest first.

<!-- prettier-ignore-start -->

| Date | PR | Changes |
| --- | --- | --- |
| 2026-10-15 | #12676 | Guard MOSAIC inertial parameter layouts: audit found no live Pinocchio/inertial.py mix; tests pin each module's ordering and block unconverted coupling |
| 2026-10-15 | #12627 | NSX-54: OpenSim closed-loop snapshot adapter delegating to OpenSimFullBodyParityAdapter, equal to Pinocchio |
| 2026-10-15 | #12629 | MuJoCo closed-loop snapshot by hard-KKT evaluation (CLOSED_LOOP_SNAPSHOT FULL), separate stock-soft-weld divergence entry |
| 2026-10-15 | #12668 | Correct the club y-sign in golfer forward kinematics, club Jacobians and the native grip_left wrapper; closure is a wrist-pin PLANAR_POINT2 with rank 2 on the manifold (TB-00 registry: 6 independent DOFs); planar-model and central-difference cross-checks |
| 2026-10-15 | #12623 | NSX-13: snapshot_from_adapter with CLOSED_LOOP_SNAPSHOT capability (Pinocchio FULL), canonical closure in the torque allocator, constraintDynamics parity |
| 2026-10-15 | #12613 | NSX-11 typed loop-nullspace registry loader with CI citation gate |
| 2026-10-15 | #12615 | NSX-50: pose_interchange dual velocity/acceleration/force maps and closure wrench duals |
| 2026-10-15 | #12614 | NSX-12: analytic closed-loop oracle fixtures (planar two-arm golfer, straight arm, envelope, spatial Pinocchio chain) reproducing NSX-00 goldens |
| 2026-10-15 | #12612 | NSX-10: ClosedLoopSnapshot, LoopTopology and evidence-axis contracts |
| 2026-10-15 | #12610 | NSX-02 canonical closure Jacobian convention (frame, row order, sign) with per-engine declared conventions, golden and cross-engine tests |
| 2026-10-15 | #12650 | Port MyoSuite runtime loading to a version-tolerant shim (env_base removed in 3.0.0) so the myosuite extras stay unpinned |
| 2026-10-15 | #12611 | NSX-04: per-slot torque-capacity registry and capacity_for loader with verified and assumed sources |
| 2026-10-15 | #12603 | Hand-on-club grip wrench uses the bare club inertia (club_solids_partition); legacy composite kept as wrist_on_hands_plus_club |
| 2026-10-15 | #12595 | Split force_nullspace.py into a force_nullspace package with re-export shims and public API contract test (NSX-03, no behavior change) |
| 2026-10-15 | #12598 | NSX-00: replace the provisional spec with the reviewed version (independent sympy derivations, 8 goldens, contracts.md, ADR-0054 D1-D6 accepted, D7 proposed) |
| 2026-10-15 | #12547 | Stop committing the generated agent-context views (README.md, index.html); render them on demand in CI and locally; reviews, catalog and contracts stay committed |
| 2026-10-15 | #12597 | NSX-20: convex solver backends (highspy LP session with basis reuse, Clarabel SOCP), loop-nullspace extra, benchmark |
| 2026-10-15 | #12596 | Add NSX-00 mathematical specification, SymPy golden fixtures, ADR-0054, hypothesis registry seed, and contract appendix; Add NSX-05 deferred-validation record DV-12554 and external validation protocol |
| 2026-10-15 | #12541 | BunkerShot3D: particle playback panel embedded in the shot viewport (Open particle trajectory action, play/pause/step, time scrubber in simulation seconds, playback time scale, camera presets, y=0 clipping toggle, visible frame stamp, reader errors shown for rejected artifacts, a stated reason for every disabled control, keyboard shortcuts and focus order). Workbench parity gap now points at follow-up #12536. |
| 2026-10-14 | #12542 | Complete BunkerShot3D workbench export views, save/reload, VTK stamped export, and API docs (#12536) |
| 2026-10-14 | #12538 | Qualify spatial carriage damper angular-momentum closure with central-force and closed-wrench laws (#11195); Add MOSAIC-09 leave-one-trial-out cross-validation study and update model-aware matching reference (#11541, #11532); Add OpenSim full-body musculoskeletal lift model composer and kinetics analyzer; \ Add lifting showcase video planner and headless smoke renderer\; Add five additional weightlifting movement specifications with split stance; Add frozen research replay provenance auditor and source-authority amendment ledger (#12398); Add transactional muscle allocation stepper, fresh controller-off replay, R2025b T01 binding, and F10 turnover packet (#12399, #11898, #11921, #11784); Add versioned pre-impact kernel bundle adapter and contact-interval playback/export integration (#9549, #9703, #9700) |
| 2026-10-14 | #12535 | BunkerShot3D figure exports carry tier, verdict and sources as sealed PNG data; the reader refuses unstamped, edited or predictive-claiming files |
| 2026-10-14 | #12534 | feat(lifting): regenerate back-squat bundle with back-bar lift body, torso shelf, standard grip width, and symmetric barbell bushing interface (LIFT-5 / #12347) |
| 2026-10-14 | #12532 | Add root-yaw target tracking support and evaluate finish-window root yaw feasibility for shooting fit; Adopt the #11671 foot torsional-friction patch (0.05 m) in canonical club-face fixture runs against ground yaw slip |
| 2026-10-14 | #12527 | Add OSV-5 golfer realism showcase render script, headless smoke test, and modelling reference updates (#11731, #11726) |
| 2026-10-14 | #12523 | Add MATLAB R2025b Simscape Mechanics Explorer 4-view export script and deferred catalog entry (NV-8) (#11681, #11673) |
| 2026-10-14 | #12525 | Add MATLAB R2025b Simscape same-input torque bundle replay harness and update parity reference (SIP P-8) (#11613, #11605) |
| 2026-10-14 | #12522 | BunkerShot3D: read-only 3-D particle viewer for the trajectory artifact (headless playback with scrub/step, time-synced club/ball without extrapolation, labelled display decimation, measured fps; instanced-glyph VTK layer for the canonical viewport with a stated no-VTK fallback). No production solver emits the artifact yet. |
| 2026-10-14 | #12517 | Regenerate the 7-iron club-face fixture with the opt-in 0.125 ms effort hold and adopt its #12221 grip refit (document, receipts, OpenSim export, MSK grip calibration); its feet stay within 5.5 mm of the tracked swing through 1.1 s. The OpenSim bushing grip exceeds 2 deg / 50 N only at 7-iron ball impact (2.98 deg, 58.2 N, t = 1.334-1.344 s): owner-ruled window-limited waiver (ceilings 3.07 deg, 59.9 N; #12272 rule), and both clubs now hold the plain bound outside their impact window. The iron7 grip-parity OpenSim reference and engine series are regenerated from the new fixture (reference_regeneration_12279.json; MuJoCo, Drake and Pinocchio pass, worst 4.3 % peak against 5 %) |
| 2026-10-14 | #12519 | BunkerShot3D: versioned 3-D particle-trajectory and contact-event artifact (stable grain ids, radii, active/lost masks, per-array SHA-256, bounded streaming, atomic finalisation marker) with a verifying reader |
| 2026-10-14 | #12515 | BunkerShot3D web page runs evaluate/compare as polled, cancellable jobs with a labelled progress bar, Cancel button, failed/cancelled states and a stale-result guard (slice 5) |
| 2026-10-14 | #12507 | LIFT-3 slice 3: re-measure the lift-pack parity baseline live with all four engines; update the audit name map and MuJoCo root placement to the current packs; cite open issues for every remaining gap |
| 2026-10-14 | #12506 | BunkerShot3D workbench solves run as bounded, cancellable API jobs (/v1/jobs/evaluate, /v1/jobs/compare, poll with per-shot progress, cancel) on the shared TaskManager; a cancelled or failed job never carries a result, and a completed one names its job id and input digest |
| 2026-10-14 | #12513 | Route the remaining dynamic OpenSim loads (import_module/__import__) in src and two grip tests through import_opensim/require_opensim; the import-route guard now flags literal dynamic loads |
| 2026-10-14 | #12476 | Add MockEngineQuarantineError, assert_not_mock_engine guard, and GOLF_FORBID_MOCK_ENGINE enforcement |
| 2026-10-14 | #12471 | Add frozen ImpactModelCapabilityMatrix and fail-closed normal-only spin and passivity guards |
| 2026-10-14 | #12466 | Add fail-closed impact-interval export schema and mutation tests for residual zeroing and unfinished contact |
| 2026-10-14 | #12440 | BunkerShot3D workbench comparison never names a predictive winner on the uncalibrated carry objective (#9239); leads are labelled exploratory with reasons |
| 2026-10-14 | #12495 | BunkerShot3D web page (/tools/bunker-workbench) consumes the v1 solve and A/B comparison with tier, verdict, stamp, source and objective-standing display; an unqualified lead is never shown as a winner |
| 2026-10-14 | #12449 | BunkerShot3D v1 API serves the A/B comparison with the model's carry-objective standing (winner null unless supported, exploratory_leader otherwise) and verifies saved records by digest |
| 2026-10-14 | #12489 | Extend rust optional extra to all 10 Maturin crates and populate deploy/README.md |
| 2026-10-14 | #12393 | Move SPEC.md section 12 rows and per-PR sections (about 1.3 MB) verbatim into docs/spec/CHANGELOG_ARCHIVE.md; cap SPEC.md at about 300 KB under a renewed dated exception; duplicate/key gates read the archive for history |
| 2026-10-14 | #12472 | Update stale wave8 root-clutter test for directory allowlist enforcement (main was red after #12464) |
| 2026-10-14 | #12464 | Enforce root directory allowlist, add .ai governance index, and record Argon audit and triage ledger |
| 2026-10-14 | #12334 | Document a blobless partial clone as the smaller-clone path (no history rewrite) and retarget the club-only epic native-receipt guard from closed #10602 to #10363 |
| 2026-10-14 | #12467 | Add per-engine optional extras in pyproject.toml and enforce requirements-tools.txt pin contract |
| 2026-10-14 | #12218 | Bootstrap rustup with --no-modify-path before dtolnay/rust-toolchain so CI stops appending CARGO_HOME env lines to runner shell profiles (Repository_Management#2064) |
| 2026-10-14 | #12264 | Sync the agent-identity managed block: the owner's personal token is an approved agent identity |
| 2026-10-14 | #12359 | Re-archive HANDOFF.md under the 200-line target |
| 2026-10-14 | #12512 | accommodate collated change-log rows in SPEC cap and test_spec_archive |
| 2026-10-14 | #12509 | Pack-audit adapters track the packs: recognise leaf/loop-closure hand_r weld to the bar (shared hand_attachment helper), measure grip width at the start pose, read start GRF through the Drake/Pinocchio pack contact APIs |
| 2026-10-14 | #12505 | Regenerate the grip-parity OpenSim reference series and engine series from the current club-face fixtures, with a regeneration receipt; MuJoCo, Drake and Pinocchio pass again on both clubs |
| 2026-10-14 | #12511 | Re-sync shared_scripts collator with Repository_Management (shipped dl_state now advisory) |
| 2026-10-14 | #12490 | collate-changes keeps going past a refused fragment and fails only after proposing the rest (port of Repository_Management #2046) |
| 2026-10-14 | #12474 | Remove private native capture files, original capture exports and club workbooks from the public tree; workbooks resolve only through the capture registry (CAPTURE_DATA_DIR); add a hash-denylist hygiene guard |
| 2026-10-14 | #12436 | Unblock Collate Change Fragments: the MOSAIC-13 multi-trial #11545 fragment no longer reopens shipped DL-#11545 as in_review; it records against the shipped entry |
| 2026-10-14 | #12423 | Make UI API type generation deterministic: share one GroupColumn Literal so ModelPayloadV2.group_column no longer flips order by import order |
| 2026-10-14 | #12417 | Move per-slice AGENT_HANDOFF.md history verbatim to docs/development/AGENT_HANDOFF_ARCHIVE.md; live handoff is current-state only (96 lines), the expiring doc-size exception is dropped. |
| 2026-10-14 | #12414 | \ Restore DBC level in test_ud_contract_level_alias_sync to prevent xdist pollution\ |
| 2026-10-14 | #12419 | Reconcile package collision gate with initialized Tools provider layout, distinguishing verified canonical aliases and seam counterparts while eliminating redundant nested Tools roots from pythonpath. |
| 2026-10-14 | #12426 | Reconcile optimized contract defaults across provider boundaries in test probes without creating an unapproved public contracts fork. |
| 2026-10-14 | #12396 | Restore Development Log capacity with source-backed DL-#10783 shipped metadata, the official zero-day archive routine, and two narrowly scoped title-case corrections; preserve all other metadata and canonical-function exception deadlines. |
| 2026-10-14 | #12433 | Add bounded invocation-local muscle path observation caching to avoid redundant solves during dense finite-difference probes; preserve complete allocation results, candidate accounting, and error propagation. |
| 2026-10-14 | #12389 | Expose a shared immutable quasistatic Hill force snapshot for allocator and future validated replay consumers; preserve allocator behavior and keep model qualification unchanged. |
| 2026-10-14 | #12367 | Allow native Moco preparation to skip only the short capability probe while recording runtime as not attempted and keeping solve readiness false |
| 2026-10-14 | #12488 | Back-squat bar geometry slice 1: torso bar shelf, standard grip width and medio-lateral bar axis from the parity standard (back_bar.py, lift_to_full_body geometry option, generate_lift_bundle --geometry back_bar); committed bundle and evidence not regenerated |
| 2026-10-14 | #12430 | Add Pinocchio OpenSim and MATLAB 3D adapters for swing report parity |
| 2026-10-14 | #12421 | Swing Report React/Tauri parity: web Full/Custom dialog over POST /api/swing-report/view (PyQt ReportViewModel rows) and zip package export via POST /api/swing-report/export |
| 2026-10-14 | #12406 | Swing Report S6: PyQt Full/Custom report dialog and Qt-free view-model over the validated report |
| 2026-10-14 | #12395 | Swing report S5: MuJoCo and Drake adapters with cross-engine speed parity harness |
| 2026-10-14 | #12394 | Swing report S4: package exporter and CLI (report.json, series CSVs, value-only plots, MANIFEST.json sha256s, markdown summary) with fail-closed round-trip tests |
| 2026-10-14 | #12366 | Swing report S3: plot registry and Full/Custom plot selection with unavailable reasons |
| 2026-10-14 | #12354 | Swing Report S2: build_swing_report assembles runs, quantities and provenance from engine results under the fail-closed #10382 policy (withheld stubs always value-free, advertised quantities need two-engine spread, nulls never zero-filled); reuses the S1 schema, policy and validator |
| 2026-10-14 | #12336 | Swing report contract v1 (S1): schemas/swing-report-v1.json, quantity policy table, fail-closed validate_swing_report with named I1-I12 errors, valid/invalid fixtures |
| 2026-10-14 | #12333 | R05: neural promotion gate fails closed on missing, zero, or non-comparable evidence |
| 2026-10-14 | #12335 | Export the 3-D putt simulation as the ud.putt_scene/1 mirror artifact for the public explorer Putting tab |
| 2026-10-14 | #12322 | Force redistribution reports optimizer convergence only when the optimizer reports success, independently of the candidate feasibility audit. |
| 2026-10-14 | #12484 | Widen the golf models' forearm pronation range to the AAOS normative 80 deg from neutral, from one shared source |
| 2026-10-14 | #12445 | Single FigureOptions mapping helper for all three final-figure writer calls; overhead writer accepts frozen presentation/dpi keywords with pixel-identical output |
| 2026-10-14 | #12496 | Compare address hip flexion, knee and pelvis tilt with capture and audit impact timing and landmark RMS |
| 2026-10-14 | #12313 | Frame the OpenSim same-input view from the trajectory so feet and clubhead stay in the face-on image |
| 2026-10-14 | #12311 | OSV-9 MSK turn_feet mode drives per-frame foot targets (heel and toe lift) from the capture foot markers |
| 2026-10-14 | #12314 | OSV-9 MSK grip: specified hand grip rotation (trail palm to target, lead palm away, thumb down the shaft) replaces the IK-inherited frame; calibration, models and MSK renders regenerated |
| 2026-10-14 | #12446 | Harden CI shot-pattern figure restore: strict path charset, symlink/alias/duplicate ZIP member rejection, size caps, atomic non-following writes |
| 2026-10-14 | #12305 | Vectorize MatplotlibRenderer face gathering into one (F, V, 3) array, restoring the 1 ms update_frame budget |
| 2026-10-14 | #12486 | Scale native hand masses with subject mass when match_subject_mass is enabled and document reference capture hand mass policy |
| 2026-10-14 | #12298 | Realtime soak nightly skips cleanly unless SOAK_RUNNER_AVAILABLE is true |
| 2026-10-14 | #12296 | Cross-Engine Equivalence keeps one running plus the newest pending main run instead of queuing one per push |
| 2026-10-14 | #12282 | Admit exact zero-coordinate structural WeldJoint in constrained muscle replay |
| 2026-10-14 | #12502 | Add explicit effort hold rate manifest policy for same-input bundles and document 0.125 ms ZOH findings |
| 2026-10-14 | #12277 | Retain hash-bound manufactured native diagnostics |
| 2026-10-14 | #12500 | Record OpenSim bushing grip ball-impact transient bounds decision 21 in DESIGN_DECISIONS_2.md |
| 2026-10-14 | #12273 | Add hash-bound one-pass nonuniform Moco mesh refinement planning; Analyze retained nonuniform activation replay evidence; Split native Moco ladder history into budget-compliant linked appendices |
| 2026-10-14 | #12268 | Retire the drifted in-tree Movement Optimizer copy; the legacy src.shared.python.movement_optimizer spelling now binds to the pinned Tools package with a DeprecationWarning, and the maturin parity gate builds the pinned crate. |
| 2026-10-14 | #12252 | Use MuJoCo's named muscle enum for native MyoSuite actuator classification. |
| 2026-10-14 | #12250 | Add source-bound native diagnostic admission, runtime, and qualification contract. |
| 2026-10-14 | #12232 | Bind native OpenSim visual meshes to a strict source-owned resource closure |
| 2026-10-14 | #12226 | CI: heavy cross-engine equivalence, optional-stack and PR leaderboard-artifact lanes run nightly, on dispatch or on the ci:full PR label instead of every PR push; non-PR cross-engine runs are never cancelled and a nightly failure files an issue |
| 2026-10-14 | #12230 | Club-face fixtures keep the capture address toe-out: the swing IK and every re-solve plant the stance at the calibrated address when foot progression is on; fixtures regenerated with --foot-progression capture (driver 16.64/3.57 vs 16.37/4.03 deg, 7-iron 15.43/-0.96 vs 15.00/-0.37 deg) |
| 2026-10-14 | #12251 | Opt-in --fix-club-markers keeps the club cluster at the per-club document offsets in the static trial; with it the 7-iron fixture flags reach -2.6 % impact speed, 0 ms peak timing and a 4.0 deg impact-face gap, and the driver document cluster is shown not to be a valid face frame |
| 2026-10-14 | #12270 | Per-capture grip rotation refit on the fixed clusters: tooling, receipts and evidence (documents unchanged) |
| 2026-10-14 | #12265 | Give the Driver Document Its Measured Club Marker Cluster and Regenerate the Driver Receipts |
| 2026-10-14 | #12274 | Adopt the per-capture grip rotation on the fixed driver club cluster and regenerate the three driver ground-support receipts, the driver OpenSim export, the driver club-face fixture (meets GCV-20) and the MSK driver grip calibration; the 7-iron refit is deferred because its same-input export fails GCV-20 and OSV-10 |
| 2026-10-14 | #12293 | Design manual index integrity: restore lost chapter 50, include orphaned chapter 31, drop three duplicate includes, renumber OpenSim geometry chapter 26 to 60, and add a ratchet test against new collisions |
| 2026-10-14 | #12219 | Give the 7-iron anthropometric document its own clubhead marker cluster (per-club club-markers file read by the builder); head-triad residual on the 7-iron capture 42.4 to 0.24 mm; both 7-iron receipts regenerated (nominal IK 67.9 to 45.2 mm; calibrated dynamics 54.5 to 109.9 mm, owner decision pending) |
| 2026-10-14 | #12295 | Real-engine tests skip on conftest MagicMock engines through one shared tests.support.real_engines.real_module helper |
| 2026-10-14 | #12202 | Make the engine-module teardown guard tolerate a concurrently removed package binding |
| 2026-10-14 | #12301 | Character Builder mass_kg now sets the compiled body mass: the Rajagopal legs scale to carry what the upper body does not. |
| 2026-10-14 | #12198 | Bind exact OpenSim Body properties to native physical-inertia admission; Require native source-body physical admission before OpenSim Moco solve; Add a separately identified source-bound generic shoulder triangle-lamina candidate with native physical and sampled kinematic checks |
| 2026-10-14 | #12188 | Reject impossible rotated inertia tensors and scale-masked asymmetry |
| 2026-10-14 | #12361 | Re-land the exact 557-muscle diagnostic replay profile (native_exact_557_profile, v3 opt-in) that stack consolidation dropped from main |
| 2026-10-14 | #12179 | Use the src.shared.python prefix for ud-canonical model_generation/humanoid_character_builder self-imports so the Drake path works without PYTHONPATH=src |
| 2026-10-14 | #12181 | Verify exact zero-MTP CustomJoint reduction on assembled 557-muscle native source |
| 2026-10-14 | #12184 | Replay Constrained Mixed OpenSim Moco Inputs Without Hiding Native State Projection |
| 2026-10-14 | #12172 | Verify exact zero-subtalar CustomJoint reduction with sampled native 557-muscle mechanics |
| 2026-10-14 | #12171 | Guard partial BUET abdominal zero-translation reduction with native source and state verification. |
| 2026-10-14 | #12480 | Reconcile shot pattern analysis paper prose and table values with saved scoring inputs |
| 2026-10-14 | #12165 | Compose exact BUET upper and Hamner distal native OpenSim sources with fresh-reload sampled mechanical checks |
| 2026-10-14 | #12152 | Add assistance-explicit OpenSim scalar replay with native force and work accounting |
| 2026-10-14 | #12153 | Add a source-bound native zero-MTP reduction with exact Rajagopal and scaled-club mechanical checks |
| 2026-10-14 | #12288 | Report OpenSim golf muscle support only from source-bound native readback; Bind observation scores to native marker replay evidence; Add source-bound provisional marker calibration artifacts; Exercise calibrated native marker replay through existing observation scoring |
| 2026-10-14 | #12447 | Prove paired-figure stems are unique and unchanged over the saved 384-comparison inventory and cover full-precision target, delimiter and fractional-label collision cases; Pin the paired-figure renderer protocol precondition: protocol_id-only references raise ValueError before any I/O or figure allocation for every selectable kind |
| 2026-10-14 | #12138 | Fix 11 motion-matching unit failures: DrakeFullBodyIK reads spec offset_m and keeps every spec-referenced body in pose_fn; event-alignment and Drake fit tests repointed to the current reader and dispatch seams |
| 2026-10-14 | #12163 | Declare and verify native OpenSim constrained cold starts with exact mechanical replay diagnostics; Bind declared native constrained muscle states to Moco and independent excitation replay; Link Assistance-Explicit Moco Controls to Independent Native Replay; Preserve canonical ground-support numerical receipt bytes through formatting hooks |
| 2026-10-14 | #12223 | Regenerate the club-face swing fixtures on the current spec (feet flat instead of 9-41 cm up), fix impact_frame's missing ball-radius check, check the address face against the capture's own face, regenerate the OSV-9 MSK grip calibration, and strict-xfail the 7-iron impact face (#12221) |
| 2026-10-14 | #12105 | LIFT-1 Drake pack-audit adapter reports segment inertia about the COM, not Bo; regenerate parity baseline |
| 2026-10-14 | #12482 | Route the remaining test-side OpenSim imports (64 files) through tests/helpers/opensim_support.require_opensim and add an AST guard test against direct opensim imports under tests/ (slice 7) |
| 2026-10-14 | #12365 | Route remaining ezc3d import sites through the shared native-DLL guard; add AST guard and a skip helper that skips only on ModuleNotFoundError (PR2 of 12065) |
| 2026-10-14 | #12475 | Route OpenSim test imports in tests/opensim through the native-runtime boundary via tests/helpers/opensim_support.require_opensim (slice 6) |
| 2026-10-14 | #12499 | Route every production src/ OpenSim import through engine_core.opensim_runtime.import_opensim and extend the AST import-route guard to src (slice 8) |
| 2026-10-14 | #12470 | Hold the native C3D/OpenSim runtime boundary inside the pytest process: tests/helpers/native_runtime_boundary skips a test with an explicit no-evidence reason before ezc3d loads beside OpenSim (require_ezc3d, ezc3d_installed) or OpenSim loads beside ezc3d (requires_opensim marker), reusing the diagnostic's public RUNTIME_ROLES and the Tools DLL probe; validated against real OpenSim 4.6 + ezc3d 1.7.2 where both import orders fail |
| 2026-10-14 | #12451 | Enforce the C3D/OpenSim process boundary: TRC I/O moves to shared trc_io so C3D preparation never probes opensim; source-bound receipts (hash, labels, mask, clock, units, points) verified bit-for-bit by load_transferred_capture; clean-process native_runtime_identity diagnostic reports python/opensim/ezc3d identities and ezc3d.dll provenance |
| 2026-10-14 | #12437 | Consume the Tools-owned ezc3d native-DLL guard (vendor pin bump to Tools #5581); data_io guard becomes a re-export, remaining importorskip(ezc3d) sites use require_ezc3d, AST guard forbids importorskip (step 2 of 12065) |
| 2026-10-14 | #12454 | Bump vendor/ud-tools to f7440f1823 (Tools #5596 c3d split); re-sync the sidekick c3d_reader child copy and its new characterization test, and retire the dated force_plate_dataframe architecture-budget exception |
| 2026-10-14 | #12350 | Add shared ezc3d import guard that diagnoses the OpenSim/ezc3d DLL conflict with an explicit NativeDllConflictError and process-boundary guidance |
| 2026-10-14 | #12144 | Correct SG optimizer CLI integration invocation |
| 2026-10-14 | #12113 | Right knee flexes negative like the left (spec builder + committed specs); canonical ground-support receipts regenerated: calibrated IK improves, dynamics replay regresses (#12110), trail toe-out misses 2 deg (#12109) |
| 2026-10-14 | #12048 | Move the expired realtime shadow-ledger entry to #12046 with a 30-day sunset to unblock the merge queue |
| 2026-10-14 | #12247 | Opt-in --hip-zero-twist neutral puts the hip rotation window at anatomical neutral, so the pelvis turn is within 3.2 deg of the markers at top and impact on both tour captures (evidence receipts committed, canonical default unchanged); the OSV-9 MSK club render can track with the capture's turn targets and planted feet. |
| 2026-10-14 | #12166 | Restore the canonical driver and 7-iron receipts re-solved on the consolidated #12145 code (adds dynamics.fd_phase, turn, GCV-20 ball impact and weld projection blocks) |
| 2026-10-14 | #12116 | Report forward-dynamics marker RMS split at the detected impact (address to impact, after impact) in matched-swing receipts and the ledger; reporting only |
| 2026-10-14 | #12078 | Map spec-ordered q into Pinocchio's depth-first q layout in PinocchioFullBodyIK.pose_fn (reusing CoordinateMap) and accept the 44-coordinate specs as Drake does |
| 2026-10-14 | #12108 | Diagnose the turn-split follow-through FD regression: IK branch selection sets a +/-7-14 mm FD floor; impact-windowed split passes turn bounds but not FD; split stays opt-in; Opt-in thorax-orientation target and shoulder-girdle weights split the MuJoCo turn (slice 7); IK meets turn bounds on capture-A/B but FD regresses, so default stays marker-only |
| 2026-10-14 | #12248 | Commit the golf_humanoid and golf_humanoid_scaled 7-iron MSK grip calibrations, so the OpenSim MSK retarget and render run for the 7-iron |
| 2026-10-14 | #12216 | Drake IK uses MuJoCo's principal-log weld rotation row and exact Jacobian; Drake and MuJoCo address stages now agree within 0.02 mm on both tour captures and the capture-A Drake receipt is regenerated |
| 2026-10-14 | #12208 | Fail closed when the clubhead triad offsets do not match the captured triad; the nominal 7-iron IK drops from 264.8 to 67.9 mm (frame 0 70.2 mm) and its dynamics from 1355 to 79.2 mm, with both nominal receipts regenerated |
| 2026-10-14 | #12024 | Extract one bounded candidate search loop behind finite cost and signed hard margins |
| 2026-10-14 | #12349 | Parity harness records the producing engine of each baseline and rejects a MuJoCo-produced baseline labelled Simscape (fail-closed); real Simscape regeneration remains deferred |
| 2026-10-14 | #12346 | Fix reversed toe/heel sign in legacy compute_gear_effect_spin; toe now curves left (draw) for right-handed +X shots |
| 2026-10-14 | #12348 | MyoSuite native export worker no longer imports the removed myosuite.envs.env_base: it builds a raw mujoco model, takes MJRenderer through myosuite_compat (3.x then 2.x), and unavailable_reason() probes that same import |
| 2026-10-14 | #12321 | Authenticate physics API endpoint tests via scoped dependency override and add explicit 401 contract coverage |
| 2026-10-14 | #12002 | Load the pinned Tools mocap package under a private name so native torque replay no longer grafts it onto sidekick.lab, fixing the order-dependent vendored-fallback test |
| 2026-10-14 | #12469 | Wire MatchedSwings native viewer handoff in React UI and add Vitest coverage |
| 2026-10-14 | #12025 | Web Trends section on the launch-monitor analytics page, calling POST /v2/trend with the PyQt Trends tab inputs |
| 2026-10-14 | #12106 | Web Relationships section on the launch-monitor analytics page, calling POST /v2/relationships and POST /v2/multivariate with the PyQt Relationships tab inputs |
| 2026-10-14 | #12008 | Web Launch Monitor Analytics page: CSV load and the Flexible Analysis tab against /analyze/v2 |
| 2026-10-14 | #12079 | Matched-swing web page gains frame-exact GIF Play/Pause/Restart controls, matching the desktop QMovie playback |
| 2026-10-14 | #12054 | Web Dispersion section on the launch-monitor analytics page, calling POST /v2/dispersion with the PyQt Dispersion tab inputs |
| 2026-10-14 | #12072 | Record the matched-swing browser API route and web page in the feature-parity registry |
| 2026-10-14 | #12019 | POST /v2/trend launch-monitor analytics route wrapping the PyQt Trends tab analyze_trend contract |
| 2026-10-14 | #12118 | POST /v2/treatment wrapping the PyQt Data Treatment tab |
| 2026-10-14 | #12037 | POST /v2/relationships and /v2/multivariate wrapping the PyQt Relationships tab |
| 2026-10-14 | #12028 | POST /v2/dispersion launch-monitor analytics route wrapping the PyQt Dispersion tab's analyze_dispersion |
| 2026-10-14 | #12112 | POST /v2/comparison and /v2/model wrapping the PyQt Monitor Comparison and Models tabs |
| 2026-10-14 | #12061 | Matched-swing web page ranks the ledger like the desktop, adds Drive Mode and Profile filters, and shows the qualification note; API gains a profile filter and qualification_note |
| 2026-10-14 | #12001 | Matched Swing web page gains capture/lane filters, reset, reason search, physical gates and receipt/parity JSON views (desktop parity) |
| 2026-10-14 | #12063 | Matched-swing web page and API export the same Markdown fit-quality report as the desktop Export Report action |
| 2026-10-14 | #12168 | Matched Swing Browser native-viewer handoff API (list backends, launch on the local API host) sharing a Qt-free helper with the desktop |
| 2026-10-14 | #12155 | Launch Monitor Analytics Sessions: ImportMappingDialog's header-read/auto-mapping/ImportOptions-building logic extracted to a shared Qt-free module; new POST /v2/import/preview and POST /v2/import routes call import_session |
| 2026-10-14 | #12130 | Launch Monitor Analytics Reports: report text, canonical CSV export and reproducibility manifest extracted to a shared Qt-free module; new POST /v2/report and /v2/export routes call it |
| 2026-10-14 | #12076 | Dispersion panel derives its coordinate and group selection instead of syncing state in effects (eslint set-state-in-effect on main) |
| 2026-10-14 | #12508 | Add LaunchMonitorSessionsPanel for web session import/preview and promote tools.launch_monitor_analytics to parity |
| 2026-10-14 | #12479 | Add LaunchMonitorReportsPanel and v2 report/export API client functions for web parity |
| 2026-10-14 | #12360 | Label the MyoSuite inventory MJCF compile probe as bare MuJoCo evidence and reject it as Gym-provider availability |
| 2026-10-14 | #12323 | Regenerate stale articulated-ground post-hoc sensitivity provenance (atlas SHA256) with its generator; add regenerate-equality test |
| 2026-10-14 | #12337 | Reconcile Development Log provenance, legacy headings, and source-backed delivery states without qualifying scientific work. |
| 2026-10-14 | #12441 | fix(drake): prevent offscreen widget abort during induced source combo clear |
| 2026-10-14 | #11931 | Observe actual native constrained state and preserve unverified full-model restart, lock-target and physiology gates. |
| 2026-10-14 | #12363 | Freeze training-only native capture registration with explicit geometry lineage |
| 2026-10-14 | #11911 | Bind explicit native OpenSim marker geometry and reject placeholder physical evidence |
| 2026-10-14 | #11902 | Add frozen marker scoring without fitting withheld observations |
| 2026-10-14 | #12269 | Add source-bound full-body MuJoCo generalized-effort frozen replay with exact q/v state and model-owned contact; The F09c full-body MuJoCo path records a state-dependent generalized effort once per fixed interval and returns the existing T01 frozen-input bundle, with manufactured-plant tests of eight-substep ZOH reproduction; no native full-body execution, bounded-control, support/grip, capture-matching, or physiology qualification is claimed; Make same-input package exports lazy so closure imports do not initialize reference, integrator, or optional renderer dependencies. |
| 2026-10-14 | #11897 | Reject unqualified concrete muscle laws before native initialization |
| 2026-10-14 | #12228 | Reproduce native moving-path force-length mismatch and reject nonfinite derivative evidence; Keep explicitly unqualified native Moco feasibility separate from solve convergence and failed or incomplete independent replay; Add a source-bound sampled native path-work and chart audit |
| 2026-10-14 | #12452 | Cast numeric refinement rows explicitly in numerics.py to resolve mypy static type errors. |
| 2026-10-14 | #12357 | Bound impact acceptance claims to the tested central-contact model |
| 2026-10-14 | #12418 | Correct Central Impact and Add Three-Club Shot Pattern Analysis; Add matched delivery, correlated noise, density graphics and finite-head contact study; Embed 13 reviewed shot-pattern figures in the 33-page LaTeX research report and verify its portable source archive.; Reorient shot-pattern figures, expand impact methods, and add a graphical summary; Move 281 shot-pattern binary archives and review photographs (1.64 GB) to the research release; keep a SHA-256 index and a verified fetch script; Record verified remote restoration of the illustrated shot-pattern research release; Derive final shot-pattern conclusions from qualified selected results and preserve protocol and population identities; Keep report numerical tokens intact using compact condition keys and separate scoring and population tables; Increase paired density row spacing while preserving scientific arrays, limits and population identities.; Wrap complete covariance captions and add selective, provenance-checked figure revision publication.; Prevent protocol collisions in supplemental figure paths and require unchanged numerical bytes before recovery rendering.; Correct phase-two article source fidelity and unsupported review claims; Qualify shot-pattern validation helpers for repository gates; Reject missing pair protocols before rendering; Keep Research PDF Headings With Their Following Tables and Paragraphs; Increase matched-supplement density figure gutters to clear legends, panel titles, and colorbars.; Reconcile shot-pattern integration and repair inherited validation blockers while preserving frozen science and the approved DRY baseline; Refactor shot-pattern duplication and architecture debt while preserving frozen research evidence; Batch Matplotlib face vertex gathering to preserve the 1 ms frame update contract; Protect revised LaTeX sources by verifying current figures and allowlisting legacy asset restoration; Migrate final figure builder calls to FigureOptions and verify real writer integration and intended-output parity; Shot-pattern article: 2,000-resample headline statistics, clubhead-assumption band (136 runs), generated hash-bound tables in SG per 100 shots and m (yd); shared figure palette with reproducible PDF/SVG metadata; preset-modification and driver-lean cautions; matched physics rejects an ignored nominal face; Shot-pattern second review pass: race-safe exclusive publishing, atomic bundle scoring, shared sha256/pattern names, earlier config contracts, async presenter failures, help and README corrections; Shot-pattern study hardening: fail-fast shaft-twist reachability, POSIX provenance keys with one shared hasher, delivery-mode matching, derived comparison titles, putt-range and pattern contracts, confined evidence paths; Reconcile the completed checkout-guard record and archive shipped development-log history without changing validation limits |
| 2026-10-14 | #12362 | Observe pinned native coupled-wrist geometry with explicit assembly and derivative gates |
| 2026-10-14 | #11830 | Sample native replay marker outputs on the observation clock |
| 2026-10-14 | #12141 | Reject truncated replay admission and require a separate observation sampling clock digest; Score native output against frozen observations with full replay identity; Add restricted native Drake frozen torque replay; Bind actual initial-state payload identity in versioned independent replay and same-input comparison admission; retain legacy identity/transcription receipts; Add a fail-closed F09 execution boundary that binds explicit F01 inventory rows to Tools T01 native replay bundles, calls the reviewed MuJoCo or Drake adapter and validates the native output against exact state, input, policy, model and time-grid identities; execution receipts remain unqualified; Add native Pinocchio frozen torque replay through the existing engine; Add a fail-closed native replay to marker-position bridge for MuJoCo and Drake with explicit ordered body/frame attachments, local offsets, source/model/provider identities and output hashes; unsupported engines stay in the required denominator and no qualification is inferred; Bind native OpenSim muscle replay to the shared experiment bundle; F09e adds Pinocchio native replay outputs to the F09 marker forward-kinematics path using the actual full-state q/v trajectory and explicit frame-local offsets, preserving all registered rows and six required engines; native receipts remain unqualified and MyoSuite, OpenSim and Simscape gaps stay visible; Add an F09 native MyoSuite muscle-excitation consumer for versioned T01 bundles that applies exact normalized excitation directly to native MuJoCo controls with full state restore and strict provider/model identity, and share wrapper-state capture and restoration in one chain-walk implementation; the elbow fixture remains test-only and the six-engine parity denominator stays unqualified; Add actual R2025b physical and discrete saved-operating-point restart prerequisite; Require explicit native passive-load readiness before muscle matching; Bind owned Tools native envelopes to actual R2025b diagnostic restart and changed future-force replay; Apply complete continuous initial-state and scalar-control bounds before native Moco guess creation; Add bounded native MyoSuite model resource closure admission.; Bind compiled actuator command profiles to native replay execution.; Observe exact native source reference frames and muscle path conventions; Add file-driven native OpenSim Moco preparation, exact-knot excitation export, independent replay and masked scoring; Materialize a source-bound numerical Moco guess on the frozen observation clock and reject incomplete native guess state tables.; Align Tools dependency pins to merged main; Keep F09 replay contracts on one private Tools module identity; Add frozen source-bound native tracking costs for mixed-command search.; Connect native tracking to shared bounded command search with independently replayed fallback and guarded promotion.; Integrate admitted Tools/private replay contracts into native command solving; Native prediction verifies complete live/source/model/handle/callback identity after bundle preparation before stepping, closing a late handle-mutation gap, and the execution validator reuses that verification, cutting repeated library-byte reads by 25 percent in controlled short native fixtures; full matching and parity acceptance remain open; Restore shared native feedback admission after platform integration |
| 2026-10-14 | #12356 | F01 docs: replay admission requires full_horizon and a separate observation_time_grid_sha256 (contract repair from PR #12141, not a numerical qualification) |
| 2026-10-14 | #11829 | Admit compliant native Thelen muscle replay with explicit subtype and state policy |
| 2026-10-14 | #11836 | Add a full-state native MuJoCo consumer of the Tools torque replay bundle |
| 2026-10-14 | #11822 | Add synthetic sparse inverse-dynamics collocation spike and benchmark existing multiple shooting with fresh replay diagnostics |
| 2026-10-14 | #11825 | Add pinned native OpenSim muscle-model structural admission without anatomical qualification |
| 2026-10-14 | #12351 | Raise canonical golfer root to 0.98 m so the feet rest on the floor; regenerate MJCF/Simscape, update inventory hashes |
| 2026-10-14 | #11820 | Add explicit model-owned native contact policies and complete wrench evidence to muscle replay |
| 2026-10-14 | #11817 | Generated MuJoCo golfer: segment geoms use contype 2 / conaffinity 1 so they touch the floor (and a ball) but never each other; golfer.xml regenerated and its inventory hash updated; the four-engine MuJoCo parity tests are strict-xfail because their 'Simscape' baseline is a rollout of the self-colliding MuJoCo golfer (#12013) |
| 2026-10-14 | #11814 | Prevent parameter-only muscle audits from fabricating native qualification evidence |
| 2026-10-14 | #11812 | Add early native continuous-state muscle replay and reconcile reviewed epic milestone dependencies |
| 2026-10-14 | #11833 | OpenSim FK no longer calls the missing State.isValid binding; an uninitialised state is rejected by its empty subsystem list |
| 2026-10-14 | #12401 | Extract the exact-state feedback request calculation as an immutable stateless policy while preserving the torque controller and its allocator boundary. |
| 2026-10-14 | #12355 | Persist and validate the native passive-readiness observation receipt |
| 2026-10-14 | #12442 | Isolate shared muscle kernel imports with a compatible lazy biomechanics facade (#12397). |
| 2026-10-14 | #12386 | Document bounded one-step muscle allocation, including a floor-aware optimizer start that preserves raw bounds and evaluation accounting. |
| 2026-10-14 | #12407 | Normalize contract-level enum values at the UD alias boundary |
| 2026-10-14 | #12402 | Add a strict analytic muscle replay input contract. |
| 2026-10-14 | #12318 | Add typed finite-difference scheme selection to native Moco tracking.; Instrument native Moco solve progress and solver-stage identity without changing model, request, physics, or solver policy; Unexpected Moco solver progress getter failures now propagate; expected RuntimeError and ValueError failures remain diagnostic receipts.; Bound Moco optimizer states to source moving-path function domains |
| 2026-10-14 | #11816 | Add phase-gated tangent feedback with constrained actuator allocation and auditable applied-torque evidence; Add a bounded native MuJoCo candidate benchmark for the synthetic rotary optimal-control fixture, with independent exact held-input truth, a fresh F06 replay, fail-closed torque/slew and numerical gates, and serial failure-retaining timing and receipt evidence; Add bounded coupled F02 loop tuning, frozen-controller sensitivity and honest heldout/perturbation reports on a two-coordinate synthetic forward plant; Add bounded robust shooting NMPC with fail-closed actuator fallback, native MuJoCo step execution and independent frozen-torque replay, benchmarked against F02 TVLQR on two predeclared perturbed one-hinge trials; TVLQR is retained for native-step operation because the shooting prototype cannot meet the 10 ms step budget, and F05 full-swing and capture qualification remain open; Add a strict native RK4 one-hinge BoxFDDP candidate with verified discrete derivatives, loaded-model identity, bounded post-solve admission and fail-closed fallback, compared with SciPy shooting on predeclared supported-MuJoCo trials using exact applied-torque replay and total measured time; F05 full-body and capture qualification remain open; Add a strict floating-root, two-hinge native MuJoCo tangent-derivative fixture with ordered bounded motors, quaternion-aware finite-difference checks and independent full-state replay of frozen ZOH torques, recording source-hashed supported-provider timing; optimal-control, contact, muscle and capture qualification remain open; Add a bounded native multi-DOF manifold BoxFDDP candidate with complete-state torque replay and a matched SciPy comparison.; Connect F02 feedback to native floating-root manifold state and independently replay post-limit torques.; Fit exact-clock masked native site markers with bounded manifold control and complete-state replay.; Exercise bounded F04 feedback tuning and paired motor interventions on a native floating-root fixture.; Add an activation-aware native floating-root manifold and complete-state command replay boundary.; Use SciPy's typed vector inequality constraint for F02 SLSQP and count its preparatory evaluations within the existing budget |
| 2026-10-14 | #11808 | Freeze six-engine feedback comparison admission and independent replay prerequisites |
| 2026-10-14 | #11798 | Document researched distributed feedback epic, six-engine parity and muscle-driven OpenSim target |
| 2026-10-14 | #11783 | Simscape grip fixture now comes from the qualified run-102 replay; model-workspace coefficients diagnosed as out of bounds (max 1.2e10) and divergent |
| 2026-10-14 | #11805 | Prettier ignores the generated matched-swing ledger and the budget-held DESIGN_DECISIONS.md so the hook no longer fights test_ledger_freshness and the doc size budget |
| 2026-10-14 | #11795 | OSV-9: Rajagopal golf models and the muscle model hold the shared club in both hands (lead weld + trail constraint or bushings), calibrated to the captured address; swing tracking and offscreen renders |
| 2026-10-14 | #11754 | Scrub capture-system identifiers from docs, research checkpoints and exploratory MATLAB comments per private data policy |
| 2026-10-14 | #12010 | Standalone LaTeX research reference for the lift model packs (exploratory parts: inventory, phases, spine load, parity numbers, gaps, negative results) |
| 2026-10-14 | #11835 | Provisional design-manual chapter for the LIFT-1 lift pack audit plus registry blocker (LIFT-11 slice 1) |
| 2026-10-14 | #11978 | Lifting modelling reference section in the Movement Optimizer help page (LIFT-11 slice 2) |
| 2026-10-14 | #12215 | Design manual: LIFT-4 static-hold per-hand bar wrench section in chapter 15 and its registry blocker |
| 2026-10-14 | #12075 | Web Lift Baseline page (/tools/lift-baseline) over the LIFT-1 baseline API |
| 2026-10-14 | #11971 | Record five-lift viewing and analysis as a tracked feature-parity gap for LIFT-8 |
| 2026-10-14 | #12070 | Read-only loader and API for the LIFT-1 five-lift cross-engine baseline |
| 2026-10-14 | #12341 | LIFT-7 slice: back-squat GRF breakdown (per foot, net, CoP, free moment) and per-hand grip wrench parity across MuJoCo, Drake, Pinocchio and OpenSim from the LIFT-5 bundle; KINETICS_PARITY.md report |
| 2026-10-14 | #12340 | LIFT-5 slice: cross-engine open-loop bar-path parity of the back-squat bundle in MuJoCo, Drake, Pinocchio and OpenSim (CI 20-step replay plus committed full-window and segmented record) |
| 2026-10-14 | #12245 | LIFT-5 slice 1: back-squat same-input bundle from the Movement Optimizer via full-body marker IK, with provenance, regeneration script and MuJoCo replay smoke test |
| 2026-10-14 | #12189 | LIFT-4: per-hand bar wrench in a settled static hold for the MuJoCo lift pack via the shared GCV-7 grip analysis (hand forces sum to the 120 kg bar weight within 3.3e-7, even split); other engines report unavailable |
| 2026-10-14 | #12345 | LIFT-4: barbell held at both hands through the OSV-7 compliant bushing on the back-squat same-input bundle, with per-hand bushing wrench, a recorded quasi-static hold and four-engine parity |
| 2026-10-14 | #12498 | Pack-parity audit gains a phase_names gap rule and a Names Match Standard report column built on names_match; a receipt measured before standard v2 marks the gap stale (no closed trackers cited); the baseline is re-rendered offline with the new --from-receipt flag |
| 2026-10-14 | #12477 | Lift definitions read ordered named phases, root joint and via-poses from the vendored biomech parity standard v2.1 (lift_spec accessor + conformance test) |
| 2026-10-14 | #12241 | LIFT-3: per-engine and per-lift start-pose conformance checks (bar, feet, closure, joint limits, static GRF) with a recorded matrix |
| 2026-10-14 | #11771 | Add lift model pack audit and cross-engine parity baseline (LIFT-1): baseline doc, receipt, audit package, stills script |
| 2026-10-14 | #11985 | OSV-7 phase 3: MuJoCo/Drake/Pinocchio contact grip, MyoSuite bushing parity, kinetics validation, grip clips |
| 2026-10-14 | #11963 | OSV-7 phase 2: same-input bushing grip parity of MuJoCo, Drake and Pinocchio against OpenSim (peak 5 %, RMS 2 % of peak) |
| 2026-10-14 | #11774 | OSV-7 impact phase: full-swing OpenSim bushing grip driven by the closure-consistent fits; deflection bounds met; flat 500 N internal-force bound replaced by shared squeeze (50 N) and couple-consistency (5 %) checks, both passing for driver and iron |
| 2026-10-14 | #12119 | OpenSim contact grip: ContactGripConfig.layout / origin_axial_m delegates remove the five LOD chains #12022 added, so lod-quality-gate passes on main again |
| 2026-10-14 | #12022 | OpenSim contact grip (ElasticFoundationForce on closed meshes): driver and 7-iron full swings, net force within 3 percent of bushing; Explain the 4x per-hand contact-grip force: input drift is zero, internal force is a hyperstatic ring redundant (diagnostics and evidence) |
| 2026-10-14 | #12051 | OSV-6: record the full address coordinate vector and evaluate address toe-out with Pinocchio, MyoSuite and MuJoCo forward kinematics |
| 2026-10-14 | #12049 | OSV-6 lead foot: lateral-marker zero-twist correction (knee_flexion_axis) and leg-marker calibration constraints under --foot-progression; MuJoCo address toe-out within 2 deg on both captures (driver +0.2/-0.3, iron -0.0/-1.4 deg); canonical receipts unchanged (default-on correction is a recorded failed experiment); OSV-6: mirror the left hip adduction and rotation axes like OpenSim; regenerate the anthro specs, OpenSim exports and the calibrated and nominal ground-support receipts (pre-OSV-6 nominal receipts kept as history; nominal 7-iron IK failure pre-exists on main, #12030) |
| 2026-10-14 | #12175 | OSV-4 final evidence: per-engine address toe-out on current main (all five engines within 2 deg), OpenSim FK in the evaluator and native-viewer address stills script |
| 2026-10-14 | #11736 | OSV-4: address foot progression (toe-out) measured from markers and fitted in the shared address seed, explicit --foot-progression option with receipt fields |
| 2026-10-14 | #12194 | OSV-3d: closed-loop gaze neck (--fd-neck gaze-closed) re-solves the neck on the simulated torso; release-window gaze error 17.2 -> 1.7 deg (driver), 8.6 -> 1.2 deg (7-iron) |
| 2026-10-14 | #12154 | OSV-3c: gaze-schedule neck for the forward-dynamics replay (--fd-neck gaze), dynamics.head_gaze schedule-tracking receipt, replay head-gaze clips, published head-motion evidence |
| 2026-10-14 | #11979 | OSV-3b: gaze-regularised reporting weight 0.1 chosen by the knee rule over a capture-A driver and capture-B iron sweep (qualified receipts stay at 0), with 1080p60 MuJoCo and native-viewer head-gaze clip scripts |
| 2026-10-14 | #11768 | MyoSuite anthro map: NeckInputY drives neck_flexion (sign -1 by FK), NeckInputX (lateral bending) unmapped, neck_rotation no longer overwritten; MyoSuite retarget map is one-to-one: weighted Hip/Scap secondary entries that overwrote six primaries dropped, loader fails closed; neck torque capacities relabelled (X lateral 36, Y flexion 30 N m) |
| 2026-10-14 | #12339 | OSV-2 slice 2b: both curled-finger grip hands drawn wrapped on the club grip in MuJoCo, MyoSuite (plain visual layer) and the Drake/Pinocchio MeshCat backends via model_appearance.grip_hands; restores the grip hand mesh reference subsection |
| 2026-10-14 | #12240 | OSV-2 slice 2a: parametric curled-finger grip hand mesh wrapped on the grip radius (visual only, no engine wiring) |
| 2026-10-14 | #12145 | OSV-2 slice 1: shared two-hand grip pose mirroring the OSV-9 spec grip points, two-hand closure checks on the OSV-9 topology, per-engine closure residual series; OSV-9 slice 2: tour-matching variants read the two-hand club from the model; static optimisation carries the club (lead hand, trail weld released) with a no-club control; GCV-20: ball contact impulse in every replay, release-preserving pre-contact cutoff and grip-weld projection of the tracked reference; peak clubhead-speed timing now within 5 ms of the capture, impact speed still 4-7 % low (strict xfail with measured numbers); Fix Drake full-body IK/model spec-vs-plant coordinate order: the torso twist was solved under the SpineInputX bound; capture-A Drake IK 47.2 -> 34.6 mm, FD 382 -> 60.1 mm, twist now matches MuJoCo; Opt-in trajectory IK restart policies (seeded, anchored, stable, continuous, off) and an address-pose posture prior; default stays free because no candidate stabilised whole-swing FD (evidence in the Simscape matching reference); OSV-9 MSK retarget: opt-in pelvis and upper-trunk turn targets from the capture markers and capture-planted feet; capture-A pelvis/trunk within 0.3/3.6 deg of the markers (slice 4 of #12042, refs #11726); Split thorax yaw into shoulder-girdle and upper-trunk turn lines, add X-factor, model-FK adapters and a shared receipt turn block (slices 1-2 of #12042, refs #11726); MyoSuite leg retarget maps the hip orientation through the calibrated hip frames and fixes the left knee sign; MyoSuite address feet within 2 deg on both captures; Hip rewrite uses the spec's own pelvis alignment; anthro hips no longer start 24-66 deg yawed, hip_rotation leaves its +-40 deg limits and both feet meet the 2 deg toe-out target at address; Split the reference ZMP rates at the ball impact so the ZMP filter stops spreading the contact velocity step over the downswing; the 7-iron GCV-20 replay on the #12125 hip spec no longer diverges (dynamics RMS 264 to 54.5 mm, impact speed -10.2 %, acceptance still open) |
| 2026-10-14 | #12439 | OSV-2 item 4: hand-on-grip orientation (palm diagonal, palm facing) moves from the OpenSim calibration into the shared grip_pose definition; no per-engine grip angle |
| 2026-10-14 | #12312 | GCV-19: MyoSuite impact frame grabs (grf, club_forces and clean) for the driver and the 7-iron, added to the showcase evidence page; the 15 clips per club were rendered on brick with the MyoSuite venv from main after #12292. |
| 2026-10-14 | #12292 | GCV-19: the native-export impact card fills Impact Location (toe, high) from the decorative ball (contact point = ball centre - radius x face normal at the impact sample, card extras ball_source records the placement source); without a ball the row stays unavailable with an accurate reason instead of the stale GCV-11/GCV-13 text |
| 2026-10-14 | #12287 | GCV-19: native exports draw the GCV-17 impact parameter card from the impact frame on and, on the grf and club_forces showcase clips, a synchronised inset plot strip (GRF in BW with CoP; grip force and couple); unavailable values are shown as unavailable and recorded in the showcase manifest |
| 2026-10-14 | #12237 | Add scripts/render_forces_showcase.py: reproducible owner showcase driver (clean, club forces, GRF clips) over native_viewer_export, with manifest and smoke test |
| 2026-10-14 | #11804 | GCV-18 slice 4: ADR-0052 grip derived labels and Simscape per-hand confirmation, native viewer export scale/grip/impact/ball docs, shared-infrastructure and C4 entries for grip, impact-parameter and video-timing modules |
| 2026-10-14 | #11803 | GCV-18 slice 3: provisional design-manual chapter for the GCV-15 impact parameters and registry blocker UP-D1-impact-parameters-inventory |
| 2026-10-14 | #11801 | GCV-18 slice 2: provisional design-manual chapter for the GCV-7/GCV-8 grip-wrench reduction and registry blocker UP-D1-grip-wrench-reduction-inventory |
| 2026-10-14 | #11782 | GCV-18 slice 1: provisional design-manual chapter and registry blocker for the GCV-1 ground-reaction breakdown; user manual 12.4/12.7 corrected (CoP and free-moment equations, metrics not computed) |
| 2026-10-14 | #11970 | Run the GCV-16 clubhead-series adapter and cross-engine parity tests in the engine-provisioned Cross-Engine Equivalence job and require them to pass |
| 2026-10-14 | #11861 | GCV-15: engine-free adapter contract tests lift impact_parameters coverage from 86% to 100% on hosts without Drake, OpenSim or Pinocchio |
| 2026-10-14 | #11735 | GCV-15: add impact_parameters package (speed, AoA, path, face, face-to-path, dynamic/spin loft, low point) with fail-closed Tools delivery gateway |
| 2026-10-14 | #12133 | PlaybackTiming groups render_playback timing (drops the budget exception); GIF playback is real time at 50 fps |
| 2026-10-14 | #12043 | native_viewer_export: MuJoCo Renderer backend showing the appearance body and club |
| 2026-10-14 | #12004 | GCV-14: native_viewer_export's detect_impact_time_s now returns model_appearance.club_face.ball_passage's sub-sample crossing instead of the frame-level impact_frame rule, so the HUD impact time matches the clubhead's true closest approach to the ball rather than a peak-speed sample the height check alone accepted |
| 2026-10-14 | #12124 | Move MuJoCo GIF playback, force overlay playback and capture-rig playback transport onto the shared time-based FrameSchedule and its 60 fps unified default |
| 2026-10-14 | #11734 | GCV-14: time-based FrameSchedule (video_timing), native export 60 fps with 1x/0.5x clips, impact-window clip, 720p CRF 18 preset |
| 2026-10-14 | #12026 | GCV-14: full-, half- and quarter-speed (1x, 0.5x, 0.25x at 60 fps) are now the default native export variants, and motion_matching.export.export_video_variants writes the same three clips for any engine |
| 2026-10-14 | #12481 | Promote render.club_head_and_ball to parity in feature_parity.json and update impact_zone_readiness.json evidence |
| 2026-10-14 | #12297 | GCV-13: the OpenSim native export now hides the decorative ball after impact. The worker hangs the sphere on a tiny render-only body on a slider joint from ground and slides it 1 km out of view once the ball is hidden, because the simbody visualizer redraws body poses every frame but caches fixed geometry. The OpenSim ball limitation note is removed from the export result. |
| 2026-10-14 | #11992 | GCV-13: decorative golf ball at address shared across engines (src/shared/python/model_appearance/ball.py resolve_ball_visual, schema.py BallSettings) and wired into the MuJoCo visual layer as a massless, non-colliding sphere fixed to the world body; reports unavailable with a reason instead of guessing when the exporter's reference pose is not grounded; Drake/Pinocchio MeshCat, OpenSim, MyoSuite, native export and the web scene are not yet wired (follow-up slices) |
| 2026-10-14 | #12283 | GCV-13: native viewer exports (MuJoCo, MyoSuite, Drake and Pinocchio MeshCat, OpenSim) draw the decorative golf ball at address from one shared per-swing placement (bundle ball_impact centre, else address geometry), visible until impact and visual only; --no-ball disables it; unavailable placements are reported with a reason; OpenSim cannot hide the ball after impact (recorded limitation); post-impact flight and the web scene remain follow-ups |
| 2026-10-14 | #12332 | Decorative golf ball in the web GolferModel scene at the address spot (ballGeometry.ts mirrors ball.py) |
| 2026-10-14 | #12140 | Web ClubHead (STL heads at spec loft/lie) and HeadModel (skull, face, ears, hair) in GolferModel; refs #11718; Decorative address ball in the Drake/Pinocchio MeshCat native viewer export; Decorative address ball in the MuJoCo, OpenSim and MyoSuite native-export backends; PyQt cross-engine lift baseline panel in the Exercise Dashboard; Matched-swing web page gains Export Video (GIF/MP4) and PDF report export, reusing the desktop exporters exactly; Web Launch Monitor Analytics gains Monitor Comparison, Models and Data Treatment sections (POST /v2/comparison, /v2/model, /v2/treatment); treated data feeds every other section as on the desktop; MyoSuite native-export worker supports myosuite 3.x (renderer and arena scene lookup with 2.x fallback; availability probe shares the worker import) |
| 2026-10-14 | #11775 | GCV-10 club force and torque overlays and plots: per-hand, net and couple glyphs, grip plot route, PyQt tile and web charts, hands close-up camera, grip export in native viewer |
| 2026-10-14 | #11779 | Simscape per-hand grip wrench channels (grip:hand_left/right) and an R2025b grip fixture with a reconstruction receipt (GCV-9) |
| 2026-10-14 | #11757 | GCV-8: per-hand GRIP wrench extraction for MuJoCo, MyoSuite, Drake and Pinocchio; OpenSim explicitly unavailable |
| 2026-10-14 | #12035 | GCV-8 gaps: grip-modelling tab reuses grip_efc, Drake URDF right-hand grip wrench with left unavailable, Pinocchio lambda_grip routed through grip_from_allocation |
| 2026-10-14 | #11732 | Add shared grip wrench core (per-hand wrench, midpoint net force and couple, contact-moment/free-torque split) for GCV-7 |
| 2026-10-14 | #11781 | GCV-5 ground-reaction plots: shared plot series, GET /api/analysis/ground-reaction, PyQt Ground Reaction Plots tile and web charts with per-foot force, load share, CoP path, free moment and moment about CoM |
| 2026-10-14 | #11969 | Matplotlib and QPainter force renderers draw the ADR-0052 double-chevron tip and legend count for clamped arrows, sharing one clamped_tip_shift helper with OpenCV (GCV-4). |
| 2026-10-14 | #11981 | Flip mujoco.force_overlays and drake.gui_force_overlay parity entries to parity: the web overlay serves both engines through the generic provider path (FTO-22/23); Reconcile twelve stale feature-parity gap entries that cited ten closed issues: eight flipped to parity with verified api/web paths, frankenstein canvas re-pointed to epic #11651, three real web gaps re-pointed to new tracker #11987 |
| 2026-10-14 | #12180 | Commit the canonical driver per-frame GRF series and assert the native-export arrows never clamp (GCV-4) |
| 2026-10-14 | #12483 | Simscape tier registry and cross-tier parity guard |
| 2026-10-14 | #11780 | Simscape compute_contact_forces raises SimscapeChannelUnavailableError instead of returning a zero GRF (GCV-3 step 1) |
| 2026-10-14 | #12468 | Define 3-tier mathematical engine equivalency contract and verify Simscape unavailable contact channel |
| 2026-10-14 | #11802 | Wire the shared ground-reaction core into MuJoCo, MyoSuite, Drake, Pinocchio and OpenSim contact output with a cross-engine parity test (GCV-2); GCV-5 recorded runs: MuJoCo get_ground_reaction_breakdown, the recorder stacks it per step and GET /api/analysis/ground-reaction reads the recorder's series (unavailable, never zero); Route every centre-of-pressure computation through the shared ground-reaction core; legacy helpers become deprecation shims (GCV-6) |
| 2026-10-14 | #11991 | Run the GCV-2 ground-reaction parity rows in the per-engine Force Overlay Parity CI lanes and require them to pass |
| 2026-10-14 | #12050 | GCV-2 ground-reaction parity test loads each engine module once (module-scoped breakdown): re-executing the Drake/Pinocchio test modules broke their native imports, so only the first Drake/Pinocchio row ran |
| 2026-10-14 | #11813 | NV-9: shared viewer FOV (0.7 rad) for MeshCat exports, projected-extent and fitted-distance framing helpers |
| 2026-10-14 | #11772 | Add centroidal feasibility filter v2 (full ZMP, vertical force and friction cone over the finish) as the opt-in --centroidal-filter pipeline stage; Contact-wrench tracking QP as opt-in --tracking wrench-qp; negative result recorded; Foot contact: torsional friction (opt-in --torsional-patch-m) and foot-width spheres (opt-in --foot-half-width-m); GCV-1: shared ground-reaction analysis core (per-foot and net GRF, CoP, free moment, moments about CoM); bundle provider consumes it; GCV-4: force arrow scale modes (fixed, body weight, peak), clamped-tip marker, group toggles across PyQt, web, API and native export; Realistic parametric clubhead meshes in the MuJoCo, OpenSim, Drake, Pinocchio and MyoSuite visual layers via one shared adapter, with committed STL fallback assets; GCV-12: visible parametric head, face and neck on the spec Head body in MuJoCo, Drake, Pinocchio, MyoSuite and OpenSim renders; visual-only gaze orientation channel; ClubheadSeries adapters for MuJoCo, MyoSuite, Drake, Pinocchio, OpenSim and Simscape with cross-engine parity (GCV-16); Impact parameters panel: API route, PyQt dock and web panel with parity registration (GCV-17); Visible shaft, grip and head meshes in every generated OpenSim model and the tour-matching golf humanoid; OSV-3 head gaze stabilisation: gaze module, shared ball placement, soft gaze residual (--gaze-weight) and receipt block; OSV-7 phase 1: grip_contact interface (designed damping, force decomposition, IK conditioning), OpenSim grip_model option (weld default, bushing) with per-hand wrenches, valid-window kinetics evidence; contact deferred to phase 2; Square the clubface at address in every engine: one shared head roll about the shaft (OSV-8); OSV-10: shared marker IK gains an explicit club-face orientation residual (weight 3) from the capture head triad; club-face fixtures regenerated by script with provenance; each engine's face tracks the capture within 5 degrees at address, top and impact |
| 2026-10-14 | #11690 | Add finish-feasibility metrics (ZMP-inside fraction, friction-cone utilisation, foot slide and yaw, pelvis yaw error, vertical force range) to the dynamics receipt, with a post-processing CLI and a regression ratchet |
| 2026-10-14 | #12206 | Desktop Character Builder walkthrough with reproducible offscreen screenshots (CMB-12 slice 2) |
| 2026-10-14 | #12294 | CMB-12: verify the tutorial MuJoCo simulate step with measured output and a test; AGENT_HANDOFF note |
| 2026-10-14 | #12235 | CMB-12 glTF slice: tutorial step for the skinned glTF export, spec-q animation and web preview |
| 2026-10-14 | #12200 | Character Builder docs: web builder and appearance-v1 sections, and a reproducible build-dress-export tutorial (CMB-12) |
| 2026-10-14 | #12497 | Add CMB-11 cross-engine L0 export parity verification across MJCF, URDF, OpenSim, and Pinocchio |
| 2026-10-14 | #11799 | CMB-10 slice 1: stateless Frankenstein assembly REST API (parts, plan replay with drop evaluation, URDF/MJCF export) over AssemblySession |
| 2026-10-14 | #12174 | Web Character Builder drives the spec-native presets, preview and export API (CMB-7a) |
| 2026-10-14 | #12193 | Web Character Builder appearance export bound to the current spec character (CMB-7d): the sidecar carries the compiled spec_sha256; Web Character Builder appearance picker with recoloured preview and sidecar export (CMB-7c) |
| 2026-10-14 | #12213 | Desktop Character Builder appearance picker and appearance-v1 sidecar export through the shared appearance service (CMB-7e) |
| 2026-10-14 | #12186 | Character appearance API: material library and validated appearance-v1 sidecar bound to the spec (CMB-7b) |
| 2026-10-14 | #12220 | CMB-6 slice 1: engine-free spec FK and a rigidly skinned procedural glTF body export (one bone per spec body, Y-up), wired into the character-builder API, desktop and web export buttons; CMB-6 slice 2: glTF animation of the skinned body driven by a spec-q trajectory (TRS bone nodes, per-bone LINEAR channels); CMB-7f: live web preview of the spec skinned glTF body, refetched as the spec sliders move and tinted by the appearance trunk colour |
| 2026-10-14 | #11702 | Pinned hash-verified MyoFullBody asset fetch with licence record; Map the 44 spec coordinates onto MyoFullBody by segment-orientation IK with exact joint-equality couplings, ROM clamp and muscle-length receipts; Per-frame muscle static optimisation on MyoFullBody with reserve actuators, ID-consistency check and fail-closed receipt; Activation-coloured four-view MyoFullBody swing renderer at quarter speed with activation and reserve plots; LaTeX reference for the MyoFullBody full-body muscle-driven swing study; Consolidate appearance, character builder, Frankenstein assembly, native viewer export and MyoFullBody PRs; reconcile registries; Spec-native character parameters compile to full_body_spec via the shared De Leva builder; Character presets library (JSON, schema-validated, compiled via CMB-1); Character Builder launcher tile, embed adapter and working spec API (build, preview, export); Add the engine-agnostic appearance-v1 schema and material library; Add game-quality MuJoCo appearance: materials, textures, skybox, smooth meshes; Typed attachment ports, part catalog and drag-and-drop assembly canvas in the Frankenstein editor (CMB-8, CMB-9); Add shared golf camera view presets (face-on, down-the-line, overhead, oblique) with MuJoCo, Drake MeshCat, MeshCat and simbody camera adapters; Initialise MjvGeom glyph slots with mjv_initGeom so Renderer.render no longer segfaults with arrows or arcs; Add engine-agnostic bundle overlay frame provider (joint torque arcs, per-foot GRF at the CoP, weight at the CoM) and a MuJoCo overlay source; MeshCat overlays for the Drake and Pinocchio native exports; fix Drake sink MeshcatCone argument order; Native viewer export CLI with Drake MeshCat, Pinocchio MeshCat, OpenSim simbody (xvfb) and MyoSuite backends; Labelled 2x2 multi-view compositor and clip writer for native viewer export; MyoFullBody reserve attribution, share IK and torque neck; reserves still above 10 percent (not qualified) |
| 2026-10-14 | #12493 | \ Synchronize same-input parity calculation reference and design manual chapter with 3-tier equivalency and hand-scaling contracts\ |
| 2026-10-14 | #11806 | SIP P-9: provisional design-manual chapter 16-same-input-parity.qmd and registry blocker UP-D1-same-input-parity-inventory; LaTeX reference cross-linked |
| 2026-10-14 | #12485 | Same-input parity reference and manual chapter: effort hold (#12279), merged child state and the governance blocker wording |
| 2026-10-14 | #12465 | Guard fsspec lock floor for CVE-2026-104851 (#11587) |
| 2026-10-14 | #12501 | Consume spec-exported URDF bundle across engines and ledger MyoSuite topology divergence |
| 2026-10-14 | #11868 | Simscape URDF exchange receipt: R2025b smimport round trip of the spec URDF and a canonical-vs-spec coordinate and mass diff (smexport is absent in R2025b). |
| 2026-10-14 | #11901 | Marker-overlay video export takes a frame size (size_px), so the run-102 Simscape playback renders at 1920x1080 60 fps (1x, 0.5x, 0.1x). |
| 2026-10-14 | #11699 | DIME dynamics window: multiple shooting with states and controls as decision variables, real transition defects d_k = x_{k+1} - Phi(x_k, u_k, dt) from the provider step, and real costs evaluated on the returned trajectory |
| 2026-10-14 | #12456 | Vectorised multi-trial batching (block-arrow Schur, time-parallel Riccati) and runtime receipts (MOSAIC-14) |
| 2026-10-14 | #11694 | MOSAIC-13: closed-form multi-trial benchmark for locked-prior rows and Schur-marginal covariance, fit_uncertainty consistency test, and LaTeX marginal equation |
| 2026-10-14 | #11695 | MOSAIC-13: MHE now applies its arrival factor in the window cost and propagates it by Schur-complement marginalisation when the window slides; matches batch Kalman/least-squares estimate to 1e-7 |
| 2026-10-14 | #11698 | Drift prediction returns every horizon step (controlled_trajectory), refuses non-integer horizons, and the model-aware matching reference documents the corrected df/dx covariance recursion and root-unactuated selection matrix |
| 2026-10-14 | #11693 | Fail closed on the last DIME stub constants: replay angular drift, cancellation and GRF equilibrium are None (not measured); solver cache records no fabricated 0.0 timing |
| 2026-10-14 | #11700 | Document the DIME manifold chart, retraction and local-coordinates Jacobians (closed forms from #11560) in the model-aware matching reference, with FD acceptance evidence |
| 2026-10-14 | #12461 | Humanoid/manipulation transfer: known-payload anchor and reference-plus-gains WBC export (MOSAIC-12) |
| 2026-10-14 | #12364 | MOSAIC-11 first slice: declared joint-level first-order activation kernel (time constants, bounds, rate limit) with DbC and time-constant recovery; capture-fit RMS comparison waits on MOSAIC-09 |
| 2026-10-14 | #12460 | Actuation layer: declared joint-level activation model and replay comparison |
| 2026-10-14 | #12462 | Multi-swing acquisition and phase registration with event anchoring (MOSAIC-10) |
| 2026-10-14 | #12458 | Analytic dynamics derivatives replacing finite differences (MOSAIC-08) |
| 2026-10-14 | #12444 | MOSAIC-07: kinematic jerk prior, whiteness gate and declared-discrepancy calculator |
| 2026-10-14 | #12455 | Observation factors: 2-D keypoint projection, robust kernels, marker attachment offsets (MOSAIC-06) |
| 2026-10-14 | #12459 | Contact wrenches in the inner solve: unilateral/friction-cone SOCP, measured GRF (MOSAIC-05) |
| 2026-10-14 | #12457 | MOSAIC-04: Floating-base manifold spline on SE(3) with retraction, exact Jacobians, and outer Gauss-Newton solve from marker observations |
| 2026-10-14 | #12463 | MuJoCo/MJX regressor provider with inertia mapping and planar conformance MOSAIC-03 |
| 2026-10-14 | #12450 | Provide the Pinocchio RegressorModel provider (computeJointTorqueRegressor, batched) with planar oracle conformance (MOSAIC-02) |
| 2026-10-14 | #12453 | Freeze benchmark protocol, tolerances and planar conformance oracle |
| 2026-10-14 | #12358 | Add neutral swing posture metrics (pelvis shift, trunk tilt change) with caller-supplied thresholds and provenance (#11007 package 3) |
| 2026-10-14 | #12424 | Wire the face-angle observability audit into the receipt (ik.face_observability), the exported report and the matched-swing viewer/API report; fix SHAPE_MISMATCH observed_frames consistency |
| 2026-10-14 | #12434 | Show per-output policy badges (advertised with value +/- uncertainty, not validated, unavailable) and the face-observability summary in the React matched-swing viewer; withheld outputs never render a number and an unknown output-policy schema_version fails closed |
| 2026-10-14 | #12431 | Show output policy status (validated / advertised / not validated / UNAVAILABLE) and face-angle coverage and gaps in the matched-swing PyQt viewer and the receipt API (output_policy_payload) |
| 2026-10-14 | #12338 | Fail-closed output policy for export-report: advertise clubhead position/speed/sequencing with uncertainty; withhold face angle, joint loads, impact and ball-flight as NOT VALIDATED unless calibration and evidence are supplied; CLI exits 1 on policy violation |
| 2026-10-14 | #12425 | Document club marker and calibration acquisition requirements (head triad, limits, REASON_* causes) with a code-sync contract test |
| 2026-10-14 | #12420 | Audit club face-angle observability (sparse, collinear, uncaptured, gapped or mismatched head markers) and keep face angle unavailable in the output policy when it is unobservable |
<!-- prettier-ignore-end -->
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
