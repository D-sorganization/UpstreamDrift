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

Rows dated up to 2026-10-14 (about 1,430 of them), the per-pull-request sections that had accumulated before section 1 and after this section, and the pre-2026-10-12 change-log table live verbatim in [`docs/spec/CHANGELOG_ARCHIVE.md`](docs/spec/CHANGELOG_ARCHIVE.md) (UpstreamDrift#9415, #13095). This table is the live log: new rows go here, newest first.

<!-- prettier-ignore-start -->

| Date | PR | Changes |
| --- | --- | --- |
| 2026-10-24 | #13749 | Port gs3dx_com_shift and the capture contact-solve test (PR6 of the #13270 series) |
| 2026-10-24 | #13727 | Make the grip MJCF independent of the BLAS kernel and libm: deterministic damping design and pad ring trig |
| 2026-10-24 | #13748 | Add the Simscape rigid-body stance exporter, R2025b fixture/receipt and Python GCV-2 parity test (rigid body at total mass/CoM on the six GS3DX sole contacts; net GRF +0.087% of weight, split within 0.43%, CoP within 1 mm) |
| 2026-10-24 | #13746 | Run tests/cross_engine/test_flex_shaft_parity.py in the cross-engine equivalence workflow and add its sources to the path filters. |
| 2026-10-24 | #13743 | Correct the OpenSim ten_finger swing xfail reason: net_force now passes; the remaining 0.04 s gap is net_moment_com (#13560) |
| 2026-10-24 | #13744 | Port gs3dx_vertical_share and gs3dx_contact_wrench with tests from the unmerged capture-contact commit 22419ee13 |
| 2026-10-24 | #13741 | Capture Rig theme-compliance fixture discards a ThemeManager singleton orphaned by QApplication teardown (fixes flaky QSettings-deleted teardown errors) |
| 2026-10-24 | #13724 | Route legacy chat UI builder imports to the pinned Tools module with shared module identity. |
| 2026-10-24 | #13566 | Disable pytest-xvfb autoload for motion-matching smoke; document -p no:xvfb in run_motion_matching_smoke.sh |
| 2026-10-24 | #13734 | Port the trail-foot preload chain, point momentum and capture segment and momentum tools with tests from the unmerged capture-contact commit 22419ee13 |
| 2026-10-24 | #13732 | Port gs3dx_nearest_hull_point, gs3dx_hull_moment, gs3dx_support_residual and gs3dx_foot_contacts with tests from the unmerged capture-contact commit 22419ee13 |
| 2026-10-24 | #13733 | Port gs3dx_foot_split and gs3dx_capture_foot_residual with tests from the unmerged capture-contact commit 22419ee13 |
| 2026-10-24 | #13740 | GS3DX rigid-body stance proxy: gs3dx_rigid_stance (one rigid body at the golfer's mass and CoM on the six sole contacts, same contact law) and gs3dx_log_force_mux, with a MATLAB test and GROUND_CONTACT.md section; exporter and parity fixture follow in the next PR |
| 2026-10-24 | #13736 | Wire the OSL-7 predictive-versus-tracked comparison into the predictive receipt (tracked_comparison row) |
| 2026-10-24 | #13737 | Windows: set CASADIPATH from the OpenSim wheel dir so Moco finds the ipopt plugin |
| 2026-10-24 | #13738 | R11: the R6 chain shaft runs in Drake (RevoluteSpring plus joint damping) and passes the same static, modal, topology and same-input parity gate as OpenSim and MyoSuite; Drake joins CHAIN_ENGINES |
| 2026-10-24 | #13645 | Install the governed canonical admission publisher copy and protected CI caller with independently pinned private provider runtime and trusted copy drift checking. |
| 2026-10-24 | #13735 | Port frame_rate, trunk_spin, hold_limit_jump, rebase_turns and reference_rate with tests from commit 22419ee13 and record the failed stance probes in FORWARD_DYNAMICS.md |
| 2026-10-24 | #13726 | GS3DX GRF exporter: opt-in lock_joints (gs3dx_lock_joints, in memory) and DRY shared contact-geometry / sole-point / CSV / SHA-256 helpers; the locked-joint stance costs ~140 s wall per simulated ms |
| 2026-10-24 | #13691 | Add opt-in allocate_muscle_excitations_until_feasible (stop_when_feasible) for bounded muscle allocation: stops during a counted evaluation on an exact feasible witness and reports declared_feasibility_satisfied |
| 2026-10-24 | #13721 | OSL-7: predictive-versus-tracked deadlift comparison (bar path, window-start joint angles, group activations, L5/S1 status) aligned by bar-rise fraction; receipt wiring follows in a second PR |
| 2026-10-24 | #13729 | Escape # in lift_models.tex so compile-latex-references passes (OSL-7 follow-up) |
| 2026-10-24 | #13723 | CI: the tests job checks out only the pinned myo_sim gitlink (depth 1) so the NSX-94 real-MyoSuite co-contraction tests run instead of skipping |
| 2026-10-24 | #13719 | Chain shaft runs through the ground-support pipeline: IK holds chain coordinates at rest, face targets accept the tip link, replay uses the 0.125 ms substep |
| 2026-10-24 | #13677 | Simscape grip kernel follows the shared capsule line-contact law; ten_finger verifier back to 1e-11; structural freshness check for generated JSON; goldens rebuilt from the rounded export params; committed R2025b evidence |
| 2026-10-24 | #13715 | Fix DrakeKinematicsService.set_pose/get_link_transforms against a real MultibodyPlant (floating-base quaternion layout, body enumeration) |
| 2026-10-24 | #13717 | NSX-94: run the co-contraction trade-off sweep on the real pinned MyoSuite elbow (myoelbow_1dof6muscles); replace the never-running importorskip(myo_sim) test |
| 2026-10-24 | #13716 | NSX-93: cross-engine conformance of the declared impedance and delayed-feedback layer through the torque-engine derivative seam (Pinocchio reference, MuJoCo, Drake; Simscape UNAVAILABLE) |
| 2026-10-24 | #13714 | static_envelopes retries a NUMERICAL warm-started HiGHS solve cold (anthro plate benchmark, highspy 1.15.1) |
| 2026-10-24 | #13689 | Forward-rollout root-residual test closes the #13179 axial-load acceptance |
| 2026-10-24 | #13688 | Guard test enforcing the single paired Monte Carlo runner rule for strategy comparisons (NSX-91) |
| 2026-10-24 | #13690 | NSX-45: user help and interpretation guide for loop null-space exploration (Refs #12577) |
| 2026-10-24 | #13692 | Link the loop null-space help page (docs/help/loop_nullspace.md) from the React web UI via a FEATURE_HELP topic and a help button on LoopNullSpacePanel (NSX-45) |
| 2026-10-24 | #13570 | Add conflict-disarm caller workflow (dry-run) and CI allowlists for RM reusable @main. |
| 2026-10-24 | #13370 | fix(motion_matching): reclassify grip_style as physical pick and clean sidecar |
| 2026-10-24 | #13623 | HRG-E3: lower-body, shoulder, elbow, and line turn terms in ISB and golf language |
| 2026-10-24 | #13685 | Record #13072 closure evidence (docs/operations/evidence/shipped-even-issues-20261024.md); Record #13104 closure evidence (docs/operations/evidence/shipped-even-issues-20261024.md); Record #13579 closure evidence (docs/operations/evidence/shipped-even-issues-20261024.md) |
| 2026-10-24 | #13684 | Route native F02 and BoxFDDP mass reads through full_mass_matrix on MuJoCo 3.13+. |
| 2026-10-24 | #13664 | Wire subscription CLI adapters (Claude, Codex, Antigravity), simulation HostActionPort controls, and local Whisper VTT into Sidekick |
| 2026-10-24 | #13680 | Pin live required-check export for UD-CI-031 |
| 2026-10-24 | #13551 | Bump vendor/ud-tools to Tools 37a7477 (owned restartable UI timers, Tools #5670; chat dock stops scroll timers on close, Tools #5674) and re-sync src/shared/python/chat |
| 2026-10-24 | #13565 | Drake samples the canonical mid_hands/club_head frames like Pinocchio; legacy grip names are aliases only and a missing frame fails closed instead of falling back to right_hand or the shaft; Align Drake autodiff fit theta with canonical URDF actuator count |
| 2026-10-24 | #13681 | Lock per-runner BuildKit cache regression guards |
| 2026-10-24 | #13682 | Use full_mass_matrix in grip-weld axial closure KKT tests for MuJoCo 3.13+. |
| 2026-10-24 | #13446 | Lock canonical driver seeds closure and weight-fraction gates in CI (#13109) |
| 2026-10-24 | #13512 | OSL-7 predictive deadlift: solved statically supported poses with free shoulders, a planted-feet goal, Hicks moment gate and a 17-run history; stage 1 still does not converge with foot contact |
| 2026-10-24 | #13667 | Extend KnowledgePack search characterization with repo scope and blank queries.; Isolate Capture Rig theme-compliance ThemeManager fixture from stale singletons. |
| 2026-10-24 | #13669 | Adopt fast-tier pytest changed-module selection for unit-test-gate |
| 2026-10-24 | #13666 | Drake samples the canonical mid_hands/club_head frames like Pinocchio; legacy grip names are aliases only and a missing frame fails closed instead of falling back to right_hand or the shaft; Align Drake autodiff fit theta with canonical URDF actuator count; Hand off the HRG-GS grip parity residuals: AGENT_HANDOFF entry for epic #13297 and xfail reasons pointing at the open tracking issues #13547 (MuJoCo deflection) and #13548 (OpenSim ten_finger); Grip contact: add antagonist pads opposite the heel, trigger and ten-finger little-finger pads so the neutral-pose closure preload exerts zero net force and moment on the club |
| 2026-10-24 | #13518 | CI: set rust-cache cache-bin false so post steps stop deleting the shared host rustup binary (Refs Repository_Management#2138) |
| 2026-10-24 | #13665 | Audit near-zero-mean CV callers: return NaN with reason in cross-engine runner and pendulum variability summary; Add load_coupling_law require_verified gate for SPN-3 cited coupling data; Pull club sole to ground at address IK; record sole height in calibrated receipt; Lock canonical driver seeds closure and weight-fraction gates in CI (#13109) |
| 2026-10-24 | #13674 | Bump transitive undici to 8.12.3 in ui/package-lock.json to clear HIGH npm audit advisories that ejected every merge-queue entry |
| 2026-10-24 | #13659 | Add .oss-scanner security-scan contract (project.yaml, offline audit Dockerfile, threat model) for the fleet scanner pilot |
| 2026-10-24 | #13668 | Skip redundant OpenSim CI runs when unrelated PR labels are applied |
| 2026-10-24 | #13671 | Bump ui lockfile (zod 4.6.7) to clear GHSA-mrhx-c7xc-m3w5 so the UI Security Audit and merge queue pass |
| 2026-10-24 | #13425 | Model explorer headless export: support NPZ trajectories with 'q', document and verify results replay for anthro driver mjx_package with stiff weld constraint |
| 2026-10-23 | #13661 | Critique physical grip-contact calibration and validation evidence |
| 2026-10-23 | #13649 | Characterize the existing KnowledgePack search contract before provider synchronization. |
| 2026-10-23 | #13652 | Exclude generated capability-atlas output from Tauri workflow path triggers |
| 2026-10-23 | #13607 | Add loop null-space interpretation help and salvage tracking |
| 2026-10-23 | #13651 | Clear deleted ThemeManager main-window wrappers during test cleanup |
| 2026-10-23 | #13643 | Defer generated collation branch updates when the canonical queue preflight cannot verify editability |
| 2026-10-23 | #13647 | Remove stale CI Requires OpenSim lane quarantine (squat JR + native muscle-effort tests pass for real since #13615) |
| 2026-10-22 | #13637 | Route OpenSim classification and skip reporting through the existing reversible public hosted policy; fail closed on diff errors while preserving real engine Docker validation. |
| 2026-10-22 | #13635 | Use Python 3.12 for the reviewed nightly cross-engine dependencies.; Prevent duplicate Xvfb startup in the motion-matching smoke command. |
| 2026-10-22 | #13614 | Quarantine CI Requires OpenSim lane (RM #2252) until #13564 fix lands. |
| 2026-10-22 | #13615 | Restore CI Requires OpenSim: squat JR kinematics revision 6 and native muscle-effort allocation. |
| 2026-10-22 | #13631 | Use curl for actionlint bootstrap script download (Semgrep merge_group) |
| 2026-10-22 | #13616 | Gate workflow edits with actionlint; fix cross-engine leaderboard publish concurrency expression |
| 2026-10-22 | #13628 | Schedule nightly development log archiver workflow and noop contract test |
| 2026-10-22 | #13621 | Reconcile development log against GitHub PR truth and archive terminal rows for fleet policy gate |
| 2026-10-22 | #13617 | Remove @dieterolson from CODEOWNERS; keep team-based ownership |
| 2026-10-21 | #13612 | Add fast PR/push and full merge-queue/scheduled tiers with a fail-closed required aggregate |
| 2026-10-21 | #13599 | Repair CI lint, trusted governed policy consumption, publication detection coverage, and pre-push failure guidance |
| 2026-10-21 | #13602 | Serialize Simscape grip committed-path generated export checks under xdist |
| 2026-10-21 | #13600 | Stop realtime file-tail poll thread when last subscriber unsubscribes |
| 2026-10-21 | #13603 | Register jaxsim_dashboard on engines.dashboards for feature-parity tile coverage |
| 2026-10-21 | #13598 | engine docker cache steps per fleet contract |
| 2026-10-21 | #13596 | Quarantine pilot for Simscape grip contact tests and audit validator (UD-CI-030) |
| 2026-10-21 | #13584 | Merge-queue-safe workflow cancel-in-progress audit and gh-readonly-queue push ignores (UD-CI-032). |
| 2026-10-21 | #13586 | Retire Jules push/PR noise workflows batch 1 with archived rollback copies and post-merge gh disable. |
| 2026-10-21 | #13589 | Add OpenSim CI failure triage snapshot, doc, and classifier script for UD-CI-002t. |
| 2026-10-21 | #13588 | Record merge-queue batch-size experiment snapshots and fleet merge settings checker |
| 2026-10-21 | #13583 | Move cross-engine leaderboard JSON publish off PR triggers (nightly and main push only). |
| 2026-10-21 | #13585 | Gate OpenSim CI on paths or ci:opensim label with classify_ci_changes job |
| 2026-10-21 | #13587 | Canonicalize grip-contact codegen floats and refresh Simscape MATLAB kernel/scenario exports.; Bump pillow to 13.0.0 and certifi to 2026.10.19 to clear pip-audit CVEs. |
| 2026-10-20 | #13558 | Grip topology parity compares numerically-zero reference channels (peak <= 1e-8 N or N·m) absolutely instead of relatively; fixes myoHand frozen-finger law-level parity on roundoff spin channels |
| 2026-10-20 | #13549 | Record the OSL epic handoff state (open PRs, owner decisions, rejected video work) |
| 2026-10-20 | #13546 | grip_contact exporter --check now logs the drifted file and a bounded unified diff (flake diagnostics) |
| 2026-10-20 | #13541 | HRG-E4 slice: event table detects pelvis reversal (peak pelvis_turn_isb between address and impact, distinct from max X-factor) and adds the six lab-frame pelvis/thorax terms; export schema v2 |
| 2026-10-20 | #13529 | Add stateless Simscape physical grip contact codegen kernel, GS3DX_GripContact builder/runner, and M12 parity harness |
| 2026-10-20 | #13538 | HRG-E4 slice: POST /biomechanics/anatomical-event-table serves the shared anatomical-angle event table (JSON/CSV) from calibrated arm frames |
| 2026-10-20 | #13536 | Count bench support reaction in Hicks residual limit |
| 2026-10-20 | #13532 | complete OSL-4 muscle solve and residual gating for supine bench press |
| 2026-10-20 | #13527 | R4: the dynamics receipt records IK and FD knee flexion through the first static window (acceptance item 4 instrumentation) |
| 2026-10-20 | #13525 | HRG-H2 diagnostic palm-thumb pad: drift-gated physical trail_palm_lead_thumb pair on the shared G2 law in separated_v2, impact timestep and preload sensitivity reports, and a new spec revision with its own spec_sha256; contract pinned by tests. |
| 2026-10-20 | #13520 | Drake native golfer grip tests pass with real pydrake: default loader parses the generator humanoid (not the #13312 canonical golfer), actuator retrofit no longer calls num_velocities() before Finalize, humanoid_urdf.py split under 1200 lines and its size exceptions removed |
| 2026-10-20 | #13504 | MuJoCo topology grip schedules each node friction row's solreffriction time to the shared law's secant slip gain every step (friction_secant_gain); ten_finger 0.04 s hold net_force 2.866% -> 0.413% RMS, ten_finger 0.10 s hold and swing pass all ten parity quantities, overlap/interlock per-hand forces 10-18% -> under 1% RMS |
| 2026-10-20 | #13517 | Symmetric capsule line-contact quadrature integrates the shorter capsule axis (#13508) |
| 2026-10-20 | #13515 | Add headless regression test for Drake-Simscape equivalence theta sizing (#13449). |
| 2026-10-20 | #13472 | Reclassify grip_style as a physical pick across CLI, API, desktop and web and migrate legacy sidecars |
| 2026-10-20 | #13498 | HRG-E3 slice: pelvis and thorax ISB lab-frame turn, tilt, obliquity and side-bend golf terms with golf-word aliases; thoracic/lumbar split marked unavailable |
| 2026-10-20 | #13511 | ci(opensim): Real OpenSim job reuses a validated per-runner venv and opensim-models clone and stops uploading the GitHub pip cache from self-hosted runners |
| 2026-10-20 | #13505 | Fix duplicate tex_skin_2 asset in MuJoCo with_head exports so the MJCF compiles |
| 2026-10-20 | #13497 | Hang watchdog dumps stacks from a GIL-held Python handler so SIGUSR1 no longer segfaults the xdist controller (exit 139) |
| 2026-10-20 | #13457 | Continuous capsule line-contact law (axial node quadrature) replaces the closest-point capsule law; MuJoCo emits node spheres with N-scaled friction time and drops the parallel-capsule workaround; MuJoCo 0.04 s ten_finger hold net_force RMS 2.866% is a strict xfail |
| 2026-10-20 | #13478 | Re-lands the salvaged source-bound native OpenSim muscle-effort adapter and runtime with SDK-free contract tests; native tests skip without the SDK and no Manager advance has been executed. |
| 2026-10-20 | #13491 | HRG-Q1 slice: full-body-v2 mass matrix and gravity qualified against a spec-only geometric-Jacobian reference on MuJoCo (measured rel. error <=4e-16, tol 1e-12) with a Pinocchio twin; Drake, OpenSim, MyoSuite, limits, closure, contact and ID remain open |
| 2026-10-20 | #13503 | HRG-E4 slice: anatomical-angle event table (address/top/impact/finish) with CSV/JSON export carrying units, sign conventions, observability and receipts; JSON schema validated |
| 2026-10-20 | #13499 | MuJoCo visual layer for full-body-v2: one rigid pelvis, abdomen and thorax visual per trunk segment, seam no-gap and physics-unchanged tests (HRG-D3 slice; renders deferred) |
| 2026-10-20 | #13501 | Forward-dynamics replay reproducibility: deterministic replay test, seeded Lyapunov-style sensitivity probe (--replay-sensitivity), bounded 0.1 s post-impact FD metric, and a base-of-support bound on the ZMP filter centre-of-mass shift |
| 2026-10-20 | #13496 | Stop CI Requires OpenSim queueing one run per main push |
| 2026-10-20 | #13476 | Heavy lanes: install real pin/pin-pink wheels and isolate each lane in a venv |
| 2026-10-20 | #13458 | Fix OpenSim capsule contact mesh ring alignment, CoP reporting, and BDF accuracy for ten-finger parity |
| 2026-10-20 | #13487 | HRG-E2: full-body-v1 wrist/forearm adapter with declared neutral Wu calibration, verified on compiled MuJoCo; best-fit report test gated on HRG_BEST_FIT_BUNDLE |
| 2026-10-20 | #13456 | keep the squat IK inside every coordinate range (OSL-12) |
| 2026-10-20 | #13486 | R6 criterion 6: MuJoCo visual and appearance layers draw a chain shaft per link (shaft pieces per chain body, head on the tip link) so deflection is visible; rigid MJCF byte-identical |
| 2026-10-20 | #13468 | Size Drake-Simscape equivalence fixture theta from the plant actuator count; Resolve the Drake autodiff fit grip body through the shared simulate fallback; Guard DrakePhysicsEngine.get_joint_names against unfinalized plants; Remove dead Pinocchio default branch from Drake simulate DEFAULT_GOLFER_URDF |
| 2026-10-20 | #13469 | Mark the torch CUDA wheel-index pip probe requires_network so the offline unit gate stops crashing an xdist worker |
| 2026-10-20 | #13442 | Prove hyperstatic per-hand split divergence between MuJoCo soft-constraint cone QP and shared penalty law with minimal two-pad test and update G8 reference; Align collinear capsule-cylinder midpoint convention and topology solimp impedance d0=0.001 so MuJoCo ten_finger 0.04 s hold and swing pass determinate parity |
| 2026-10-20 | #13415 | Add OpenSim 3D joint reaction analysis for barbell lifts (OSL-5) |
| 2026-10-20 | #13482 | Pin the trail-foot toe-out on the committed canonical receipts: driver 19.76 deg, 7-iron 20.00 deg against the 20 deg style target (OSV-4 2 deg acceptance) |
| 2026-10-20 | #13480 | Restore the #13396 grip follow-ups lost in the HRG-GS consolidation: one club_state per Pinocchio RHS and the HRG reference's assessments and layout fixes |
| 2026-10-20 | #13473 | Relax last-ulp geometry assertions in batched grip contact tests (Linux BLAS dot rounding) |
| 2026-10-20 | #13471 | Test the MyoSuite gymnasium-first import behaviourally in _engine_init |
| 2026-10-19 | #13445 | Verify bilateral myoHand grip against independent MuJoCo mj_inverse requirement and forward replay |
| 2026-10-19 | #13422 | Add Pinocchio grip adapter driven by ContactTopology and G2 kernels; Calibrate OpenSim topology grip foundation at 2x preload with Hollars friction mapping and preserve SimmSpline horizon across short swing windows; Implement OpenSim ContactTopology grip contact adapter; Add MuJoCo ContactTopology grip-contact adapter and parity tests (HRG-GS-G8); Refactor Drake ClubInHands to two hand bodies and add TopologyContactModel adapter; Extend grip_contact.parity with per-pair forces, contact-point drift, determinate/indeterminate classification, and cross-engine style parity report (HRG-GS-G10); Separate LHand and RHand in full-body-v2 (opt-in) and evaluate physical hand-hand contact pairs; Add shared model grip binding contract, GripProvider protocol, and PinocchioGripAdapter (HRG-GS-M1); Add versioned per-model grip binding manifests and frame/unit validator (HRG-GS-M2); Add cross-model grip qualification harness with 3-state availability envelope and pair-level parity (HRG-GS-M3); Add native OpenSim MSK grip adapter with weld/bushing binding, Bruno 2015 admission, and M3 qualification; Add native MuJoCo humanoid golf grip adapter for rigid-hand MJCFs (HRG-GS-M5); Add MuJoCo Shadow/Allegro grip laboratory adapter with phalanx-mapped sites and physical hand-hand contact; Add MyoSuiteGolferGripAdapter, grip-binding manifests, and fix MyoSuite club shaft geometry; Add MyoHub myoHand articulated bilateral grip fixture and segment-mapped phalanx adapter (HRG-GS-M8); Add native Pinocchio golfer separate-hand variant and NativePinocchioGolferGripAdapter (HRG-GS-M9).; Add explicit bilateral lead-hand bushing attachment and NativeDrakeGolferGripAdapter for the native Drake golfer (HRG-GS-M10).; Export deterministic Simscape GolfSwing3D grip topology and allocate pad and hand-hand forces from recorded states with zero injected forces; Emit grip-binding/v1 metadata on club-composed character builder exports, fail closed on clubless exports, and enforce 100% model-library grip binding admission; Extract single array-namespace contact kernel in motion_matching shared by contact_law, jax_contact, and grip_contact; Batch one side of the grip topology contact law: validate once, Python-float row kernel, 9.9x faster topology_side_wrench with bit-identical parity; HRG reference: record that an analytic Radau Jacobian is not worth building for the Pinocchio grip sim (num_jac is 5-15% of CPU; Newton iterations dominate); Pinocchio grip RHS: memoise hand kinematics per stage time, one club state per RHS, Python-float BushingState checks; bit-identical, -37% CPU; 0.04 s topology tests leave the slow lane |
| 2026-10-19 | #13443 | kinetics parity for all registered lifts |
| 2026-10-19 | #13312 | Align default engine models to canonical golfer and raise KeyError on unknown body in compute_jacobian |
| 2026-10-19 | #13441 | Declare websockets for realtime Python backend and document Qualisys C3D/TRC export path in formats.md |
| 2026-10-19 | #13440 | Add dated NM addendum superseding sealed golfer u_dim=5 records after #12608 |
| 2026-10-19 | #13435 | Retrofit Drake golfer joint actuators when URDF omits transmissions; fail closed on unwired torque (#13421). |
| 2026-10-19 | #13436 | Lock baseline receipt provenance to tibial-torsion anthropometry (#12789 closure). |
| 2026-10-19 | #13414 | Pinocchio grip-contact adapter from shared topology (HRG-GS-G6) |
| 2026-10-19 | #13430 | Docker CI builds on the host's shared ud-buildkit builder (one layer cache per host); per-runner local cache import/export removed, cold throwaway-builder fallback |
| 2026-10-19 | #13431 | Bench press supine OpenSim IK/ID with bench contact support |
| 2026-10-19 | #13429 | Record native Bruno capture-A runs: pelvis residuals (SPN-9c), static-optimisation reserve shares at address/top/impact for p=2 and p=3 (SPN-10b), and the fail-closed empty distribution-sensitivity result (SPN-14a); address reserve anomaly filed as #13427 |
| 2026-10-19 | #13419 | Record OSL-4 muscle forces, activations, and residual status across barbell lifts |
| 2026-10-19 | #13424 | Gate compose training service behind profile; pin PyTorch cu128 wheels for training stage; Resolve Tools gitlink for Docker explorer on WSL Windows worktrees; Repo-manage docker-cache-prune host script with keep-label spare and contract tests |
| 2026-10-19 | #13418 | Fix three Drake continuous-torque tests that failed with pydrake installed: hide every cached pydrake module and size the live theta via _resolve_n_actuators |
| 2026-10-19 | #13277 | Add scripts/ci/shared_buildkit.py: idempotent, flock-serialised manager for one persistent GC-bounded BuildKit builder per host (create/upgrade/repair, cold-build fallback) |
| 2026-10-19 | #13406 | Predictive Moco simulation of deadlift lift (OSL-7) |
| 2026-10-19 | #13397 | fix(pinocchio): CMB-11 parity evaluates the full neutral coordinate set |
| 2026-10-19 | #13343 | Named grip presets: Palmer overlap, Woods interlock, ten-finger with photo_estimate provenance (HRG-GS-G5) |
| 2026-10-19 | #13279 | Add Run Study panel and worker across PyQt and React spine viewers |
| 2026-10-19 | #13404 | add barbell bundles and OpenSim IK/ID receipts for remaining lifts |
| 2026-10-19 | #13338 | Drive finger pose from contact topology with per-hand roll, absorbing #13166 |
| 2026-10-19 | #13395 | Drop the removed Simulator.set_publish_every_time_step call from Drake simulate and fit_swing_autodiff; Drake already defaults to no per-step publish |
| 2026-10-19 | #13394 | Move the MOSAIC analytic-vs-finite-difference wall-clock speedup claim to the benchmark lane; unit lane asserts it deterministically via an injected clock and regressor-evaluation counts |
| 2026-10-19 | #13345 | Defer heavy route and backend imports so Sidekick API is ready in <15s; theme imports resolve via launcher_compat; Document ud-sim:latest compose source of truth and enforce ct-run contract in CI; Return dataset_id, name, rows, columns from generate; add export stream route; pipeline: wire ground-support CLI neural flags through MotionMatchingController and record neural disposition on receipts; Enforce OpenCap learned augmenter sidecar execution, fail-closed checks, and correct synthetic/model-conditioned provenance labeling; Enforce workspace, forbidden zone, Cartesian velocity, and contact force limits in SafetyMonitor; Fix CentroidalMPC contact count and mass from model; Align engine capability flags and interfaces across registered physics engines; fix(mujoco): eliminate phantom world-fused contact on double-pendulum fixture (#12989); Standardize get_joint_names across physics engines to match generalized velocity dimensions (nv) with per-DOF expansion for multi-DOF joints; Add results playback UI and real engine detection in Model Explorer (#12996); fix(model_explorer): per-engine export converts MJCF/Drake/Pinocchio and viewer handles errors visibly; Restore capture-contract CI coverage and guard addopts marker exclusions (#13000).; Stop c3d_simscape_preview false match success; load C3D via shared reader with unit conversion; learning/imitation: adopt He/Xavier init, Adam optimizer, and shared _mlp helper in BehaviorCloning and DAgger; DomainRandomizer fails loud on unsupported engines; MuJoCo parameter surface; remove dead config ranges |
| 2026-10-19 | #13342 | Allocated hand-hand contact reporting for all style pairs (HRG-GS-G3) |
| 2026-10-19 | #13378 | Bump vendor/ud-tools to Tools 96a1684 (Coriolis Christoffel term, Tools #5664; standard gravity, Tools #5668) |
| 2026-10-19 | #13344 | feat(lifting): high-bar rack reach sweep and wrist mode resolution; fix(lifting): resolve unstable open-loop wrist mode on high-bar rack |
| 2026-10-19 | #13318 | Unify grip frames across model_appearance and grip_contact, fix Drake contact normal recording |
| 2026-10-19 | #13337 | Add cross-model grip capability inventory manifest, DbC loader, and verification suite |
| 2026-10-19 | #13217 | Report the allocated trail-palm-on-lead-thumb force (patch frame, allocation, provider metadata); physics unchanged |
| 2026-10-19 | #13284 | Run real-OpenSim lift tests in a dedicated CI job (OSL-13) |
| 2026-10-19 | #13276 | LIFT-5 clean and jerk is the whole lift: Tools make_clean_and_jerk_config (Tools #5661) on the standard's eight phases with a hang/rack/overhead bar; bundle and parity regenerated, rack/jerk IK strict xfail |
| 2026-10-19 | #13366 | ci(quality-gate): aggregate frontend-tests and the other gating CI Standard jobs so a red job blocks the merge queue |
| 2026-10-19 | #13361 | fix(ui): include physical style in resolveGrip to resolve red main on frontend-tests |
| 2026-10-19 | #13358 | fix(launcher): renumber jaxsim_dashboard to order 105 to resolve duplicate tile order 33 |
| 2026-10-19 | #13359 | fix(data_io): make c3d_loader resilient to sidekick.lab.bio test pollution |
| 2026-10-19 | #13320 | Engine-agnostic grip-style contact topology: GripStyle and trigger finger join the physical GripSpec; per-style contact pair tables with anatomy scaling.; generalised shared contact law: sphere-capsule and capsule-capsule |
| 2026-10-19 | #13341 | test_check_windows_deps no longer races real engine imports against a 3s wait; the probe is tested synchronously with optional engines forced absent |
| 2026-10-19 | #13269 | Route web launcher to /tools/spine-viewer, update Ch68 manual and ignore licensed_assets |
| 2026-10-19 | #13288 | Compute muscle forces and activations for barbell squat (OSL-4) |
| 2026-10-19 | #13316 | Shard unit-test-gate into a 2-leg file-hash matrix so each leg stays well under the 35 min limit without raising it |
| 2026-10-19 | #13319 | generalised shared contact law: sphere-capsule and capsule-capsule |
| 2026-10-19 | #13274 | Feed finished spine study receipts into injury analysis via ModelSpinalLoadProvider; spine models stay out of the feedback inventory |
| 2026-10-19 | #13275 | Add spine study execution and model asset listing API |
| 2026-10-19 | #13257 | plot_style re-exports Qt widgets and the matplotlib renderer lazily so headless force_overlay/loop_nullspace imports no longer load PyQt6 or mplot3d |
| 2026-10-19 | #13247 | full-body-v2 landmark lumbar (L5/S1) and thoracic (T12/L1) joints: C3 L5/S1, Gray T12/L1 fraction, Yukawa curvature check, mass ledger, soft-limit ROM, back-marker observability (rank 5, split regularised); fixes OpenSim CustomJoint rotation order |
| 2026-10-19 | #13198 | HRG-B2: appearance CLI flags, precedence resolver, shared Qt colour field, desktop and web hair/skin/clothing colour pickers; Engine-neutral anatomical angles package with wrist and forearm terms (HRG-E2); HRG-G1: skin-coloured bare grip hands by default and an optional white glove (glove none/lead/trail/both) through CLI, desktop and web; one hand-material resolver; no duplicate RHandStandoff loft; shared core.handedness; Subject handedness in spec, receipts and GripPose (HRG-GR0); Add the anatomical grip contract (GripSpec, exact Euler bijection) and fix the trail grip-roll axis; HRG-GR2: physical grip configuration as an immutable spec revision with hand depth/spacing on the closure and grip provenance; fitted_grip uses provenance; Grip fitting modes (user_fixed, bounded_prior, free_fit), rom_set_identified provenance, golf-language grip report with profile interval and multistart sensitivity (HRG-GR3); HRG-GR4: grip presets and sliders in the spec builder, native viewer export, desktop and web, with one shared resolver |
| 2026-10-19 | #13311 | HRG-A2: parameterise visual grip strength and per-hand roll offsets in GripPose and grip selection |
| 2026-10-19 | #13286 | Align native Drake torque replay with reviewed 1.57/1.58 versions and skip unreviewed stacks in tests |
| 2026-10-19 | #13254 | Enforce imitation learning contracts: seed-based determinism, validation split bounds, activation execution, and DAgger dataset immutability |
| 2026-10-19 | #13268 | Reject unauthenticated diagnostics in remote mode, redact absolute repo root, and measure runtime probes |
| 2026-10-19 | #13272 | Abort RealTimeController on unexpected callback errors with zero torque; wire SafetyMonitor on non-simulation backends. |
| 2026-10-18 | #13264 | OSL-18: OpenSim lift foot carries the bundle's 10 deg tibial torsion and is sized from the bundle's calcaneus stations; squat IK marker RMS 1.93 cm / max 3.55 cm, inside the unchanged 2/4 cm gate; marker-gate xfails removed |
| 2026-10-18 | #13263 | Bump vendor/ud-tools to a459505 (Tools #5653 snatch pull phase) and regenerate the snatch bundle on the merged pin |
| 2026-10-18 | #13224 | Add JaxSim launcher dashboard tile and engine tile coverage test; Fail closed with ok=False and rejected cameras on degenerate or collinear camera pairs; full-body-v2 schema (regional lumbar L5/S1 and thoracic T12/L1 trunk), one schema registry for exporters and consumers, and a constrained v1->v2 trunk migration with residual and non-uniqueness report; v1 stays default and byte-identical |
| 2026-10-18 | #13252 | pipeline: FreeMoCap adapter deterministic CSV selection, NaN confidence handling, non-monotonic time rejection, and configurable min_confidence |
| 2026-10-18 | #13250 | robotics: eliminate silent velocity, acceleration, and mass fallbacks in ZMPComputer and support finite-difference acceleration |
| 2026-10-18 | #13244 | full-body-v2 schema (regional lumbar L5/S1 and thoracic T12/L1 trunk), one schema registry for exporters and consumers, and a constrained v1->v2 trunk migration with residual and non-uniqueness report; v1 stays default and byte-identical |
| 2026-10-18 | #13200 | Translucent tension/compression segment shading with independent force/torque vector toggles, user colours and opacity across MuJoCo GUI, Drake/Pinocchio GUIs, web and native_viewer_export (MuJoCo, Drake, Pinocchio, MyoSuite); Close the segment axial-load evaluation over the grip weld: physical KKT closure wrench replaces the soft weld in the prescribed inverse, ball-impact frames are unavailable, ground free moment is carried |
| 2026-10-18 | #13239 | Fail closed with ok=False and rejected cameras on degenerate or collinear camera pairs |
| 2026-10-18 | #13242 | Penalize behind-camera observations in bundle adjustment and propagate optimizer convergence |
| 2026-10-18 | #13249 | Verify all participating swing receipt hashes in leakage guard, preserve all participant hashes, and abstain on single candidates |
| 2026-10-18 | #13196 | Procedural shoe-last feet (HRG-F1): heel counter, medial arch, instep ramp, asymmetric toe box, flat sole sized from spec foot length and a cited breadth ratio; foot.model option |
| 2026-10-18 | #13243 | Align multi-view observations by physical time across different frame rates and report dropped frames |
| 2026-10-18 | #13182 | HRG-B1: default golfer is light skin with mid brown hair from one serialized appearance-defaults contract (Python + generated TS), skin_1..8 and hair palettes |
| 2026-10-18 | #13232 | Export NaN markers with negative residual, homogeneous coordinates, and axis convention in canonical_c3d_exporter |
| 2026-10-18 | #13237 | Map reconstruct and legacy joint names in L3 paired 3D comparisons, fail closed on missing or swapped bilateral joints, compute kinematic sequence order and timing error from angular velocity peaks with explicit unmeasured None, and isolate depth axis according to camera pose and view. |
| 2026-10-18 | #13229 | Reject non-finite inputs and validate shapes in SafetyMonitor |
| 2026-10-18 | #13193 | HRG-C4: pelvis skin width from measured hip breadth or stature prior with femoral-head containment |
| 2026-10-18 | #13234 | Handle numeric array wrench point attributes without boolean evaluation error and log warnings for unparseable points in load_trace_forces |
| 2026-10-18 | #13236 | Enforce unit, metric, and level preconditions in error-budget guidance classification |
| 2026-10-18 | #13233 | NSX-101: variability interpretation guide, ten publication-gate rows (all blocked or deferred), five preregistered external experiments in DV-12554, and a Variability Sensitivity section in the design-manual QMD chapter; no coaching prescription validated |
| 2026-10-18 | #13230 | OpenCap augmenter non-zero base joint derivation, NaN marker propagation, and gap counting |
| 2026-10-18 | #13227 | NSX-100: Variability & Consistency tab in the PyQt6 Grip Wrench window and a matching React section, driven by the plot model 2.0.0 sensitivity panels, with catalogue-backed request controls and parity entry loop_nullspace_variability |
| 2026-10-18 | #13047 | LIFT-4: the barbell hold starts in contact equilibrium (CoP under the combined CoM) and carries body plus bar within 1 % over the whole 1.5 s window; hold xfails flipped to passing; LIFT-5 bench press same-input bundle: raised contact supports in the shared contact law (all four engines), four OSL-2 back spheres on a bench_top support, supine hold, and cross-engine bar-path parity; LIFT-5 snatch same-input bundle from the Tools pull-phase snatch (Tools #5653): bent-elbow arm-hang retry, pull window parity in four engines, catch IK strict xfail; Lift body and every derived lift input (squat on the OSL-17 rack, deadlift, clean, snatch, bench) regenerated on the whole-row de Leva table (#13037) with main's tibial torsion; regression-snapshot IK bounds re-derived; bundle builder picks RK4-stable substeps (deadlift 8 -> 16, parity xfail removed); follow-ups OSL-18 #13157 (foot torsion marker gate, strict xfail), OSL-19 #13158 (arm reach), OSL-20 #13159 (open-loop wrist mode); OSL-3: OpenSim IK and inverse dynamics of the back-squat LIFT-5 bundle with GCV-1/GCV-7 external loads, contact-derived static-hold GRF for all five lifts, residual receipts; other four lifts unavailable (no bundle); OSL-12: OpenSim lift IK/ID credibility: pelvis width scaling and verified virtual ASIS/PSIS landmarks, 1e7 Pa contact (PR #476) within 8 mm, ankle-lean balanced holds at zero pelvis residual, in-range squat IK; OSL-12: Rajagopal hands reach the bundle bar - hold arms seeded on the bundle grips (two-pass IK), bar marker weight restored to 1; squat marker RMS 1.18 cm, max 2.32 cm; Hicks residuals still fail at the window's first state; OSL-14: lift bundles rebuilt on the AN-1 lift body (lumbar joint, cited-range TRF bounds, de Leva hips, wrist grip fix); OpenSim torso width scaling and pelvis tracking; LIFT-4 barbell re-track divergence measured as pre-existing; LIFT-4 barbell re-track: track the loaded plant with a rigid-bar controller model so the bar's floating-base load is on every row (drift removed); record per-landmark re-track deviation; re-tighten the pelvis gate to 0; split build_lift_model; source-squat balance (bar ahead of the forefoot) still blocks the LIFT-5 bounds; OSL-15: opt-in bar-load balance of the back-squat IK over mid-foot (0.190 m to 6.4e-5 m); loaded kkt tracking loses the feet, so the committed source and evidence stay unchanged; OSL-16: bar-loaded lifts track at 64 RK4 substeps with a 0.25 ms effort hold; the loaded feet stay within the LIFT-5 ankle bounds and the re-track and full-window bar-path parity close; OSL-17: back-squat bar on the cited high-bar rack (one shaft radius behind the C7 landmark, Glassbrook 2019 / Wretenberg 1996) and a 0.1 s lead-in so the bundle and barbell re-track start settled; OpenSim squat Hicks gate passes on all 41 states |
| 2026-10-18 | #13225 | Reject duplicate source labels and handle inf values in capture export |
| 2026-10-18 | #13219 | Surface searched paths and submodule remedy in tile launch failure toast |
| 2026-10-18 | #13222 | Resolve MATLAB default slx path under production src root and raise GolfModelingError on missing default model |
| 2026-10-18 | #13215 | NSX-44: React loop null-space panel rendering the shared plot-model golden |
| 2026-10-18 | #13184 | HRG-00: measured coordinate sign map, ISB basis map, human-range characterisation and HRG calculation reference |
| 2026-10-18 | #13195 | NSX-99: sensitivity stage, report schema v2 and variability panels |
| 2026-10-18 | #13180 | HRG-E1: wrist lever angle renamed with NaN-plus-reason on missing markers; joint_stress requires signed, side-tagged WristDeviation |
| 2026-10-18 | #13197 | NSX-43: tabbed Grip Wrench window with NSX plot-model tabs, preset selector and QThreadPool runner |
| 2026-10-18 | #13194 | HRG-C3: pelvis frame re-expression (invariant) and documented HJC/COM/L5S1 relocation as a new spec revision |
| 2026-10-18 | #13192 | HRG-C2: MAP hip joint centre estimator with covariance, status and stature prior (estimation only) |
| 2026-10-18 | #13206 | Grip hand frame rolled -90 deg about the shaft so azimuth 0 is the top of the grip (HRG-A1) |
| 2026-10-18 | #13183 | HRG-C1: loft the pelvis visual on the hip landmarks (centred on the HJC midpoint, ends at the lumbar plane) and shrink the hip blend ball; visual only |
| 2026-10-18 | #13156 | Plate import forms (PyQt and web) and the plate-overlay route declare channel timing skew and the CoP skew relation (DeclaredReading); only independent allows the CoP covariance |
| 2026-10-18 | #13152 | Ground-reaction overlay draws measured CoP 1-sigma covariance ellipses, tilted with the plate yaw (cop_cov_random_xy_m2 = R_xy diag R_xy^T) |
| 2026-10-18 | #13155 | Sanitize 400/403/404 error details in handle_api_errors to prevent path leaks |
| 2026-10-18 | #13154 | NSX-42: framework-free NSX plot model and cross-UI golden |
| 2026-10-18 | #13138 | Label the Dockerfile.modular runtime stage keep=true so host prune jobs spare every profile image; contract test guards it |
| 2026-10-18 | #13149 | NSX-41: display_decision output-policy integration (ADR-0054 D7 Option A) |
| 2026-10-18 | #13147 | Ground reaction overlay draws measured CoP uncertainty as per-world-axis 1-sigma ellipses |
| 2026-10-18 | #13141 | Ground reaction overlay rings measured CoP points by their 1-sigma per axis |
| 2026-10-18 | #13139 | Ground reaction overlay shades measured plate load by its 1-sigma uncertainty and names the declared uncertainty source |
| 2026-10-17 | #13117 | Shoulder orientation solve keeps its Euler branch (continuity rule); fullbody reports orientation_max_step_rad |
| 2026-10-17 | #13134 | Plate declarations dialog and web form offer the cross_correlation alignment and require the model it was fitted against |
| 2026-10-17 | #13125 | Web Analysis Tools ground reaction charts gain the plate declarations form (same fields as the PyQt6 dialog, guarded by a cross-language parity test); analysis.ground_reaction back to parity |
| 2026-10-17 | #13129 | Add DV-13082 deferred-validation record for OF-HW open-fp device qualification; refresh DV-13061 now that the software children have landed |
| 2026-10-17 | #13124 | Ground reaction tile asks for plate declarations, stance placement and clock alignment in a dialog; API and tile share open_fp_format.declared_import; web form tracked as gap #13122 |
| 2026-10-17 | #13079 | Add the family-wise plate consistency screen and plate observation provider that eliminate candidate swing solutions inconsistent with vertical force-plate data (OF-12) |
| 2026-10-17 | #13097 | Add optional live open-fp bridge to realtime IPC (OF-09) |
| 2026-10-17 | #13118 | Fix order-dependent KeyError in the pivot parent-healing regression test |
| 2026-10-17 | #13063 | Prettier-format the back-squat barbell_hold.json evidence record (data unchanged) so the commit-stage prettier hook no longer rewrites it. |
| 2026-10-17 | #13111 | Sidekick smoke-test checks fixtures out into an isolated path so the shared self-hosted workspace is never left a sparse partial clone |
| 2026-10-17 | #13094 | Add read-only open-fp LAN client (OF-06) |
| 2026-10-17 | #13100 | Add force-event namespace and sourced impact anchor (OF-04); Overlay measured plate trials on the ground reaction tile (OF-07) |
| 2026-10-17 | #13102 | Derive the fit-path de Leva table from whole Table 4 rows, keep the capture trunk length by rescaling trunk_scale, and measure the head and trunk-part CM from the cranial row origin; canonical receipts regenerated |
| 2026-10-17 | #13107 | Src-pivot hygiene test accepts the pivot's documented parent healing, removing an order-dependent unit-test-gate failure |
| 2026-10-17 | #13098 | Add research proposal for measured plate rows in the contact-wrench QP (OF-10) |
| 2026-10-17 | #13077 | Add per-foot vertical-cell force-plate observation rows with the exact sensing-plane shear lever to the ground null space, with sympy goldens and spec (OF-11) |
| 2026-10-17 | #13096 | Add fail-closed open-fp swing file reader (JSON+NPZ, CSV) producing VerticalPlateTrial (OF-02); Add plate readings at model time with an interpolation bias estimate, skew term, world-axis CoP uncertainty and common-mode clock-offset alignment (OF-05); Add model-versus-measured vertical comparison in sigma units (OF-08) |
| 2026-10-17 | #13090 | Add force_plates package: vertical-only two-plate trial contract, ground-aligned placement and observed ground-reaction view (OF-01); Derive plate placement from a model target frame without mirroring (OF-03) |
| 2026-10-17 | #13080 | Update the open-fp integration plan with the sensing-plane lever finding and the OF-11 and OF-12 null-space constraint children |
| 2026-10-17 | #13101 | Full-body spine stage poses arms and GRF on the drive-unlocked trunk; capture-A pelvis residual 20.7 to 5.9 BW |
| 2026-10-17 | #12814 | Static 10 deg external tibial-torsion offset in the shared full-body spec (ankle base frames, cited Seber 2000 range); 7-iron trail toe-out reaches 19.97 deg with hip_rotation_r off its limit; the four canonical ground-support receipts are regenerated on the default style feet (calibrated 7-iron dynamics 64.35 mm, G3 passes) |
| 2026-10-17 | #13099 | Rotate the 315 oldest SPEC.md section 12 rows (dated up to 2026-10-14) verbatim into a second pinned block of docs/spec/CHANGELOG_ARCHIVE.md; SPEC.md 349,957 to 276,739 bytes under the unchanged 350,000-byte cap; the change-log gate reads every archived row block |
| 2026-10-17 | #13071 | Capture Rig no longer aborts when its panel is destroyed with Help or Status dialogs alive |
| 2026-10-17 | #13062 | Evaluate open-fp force plates and add the vertical-only integration plan (licence boundary, data contract, frames, timing, backlog) |
| 2026-10-17 | #13059 | Sole-load view note names its matplotlib renderer when a 3-D provider is installed; adapter golden accepts verified numpy 2.5.3 runtime |
| 2026-10-17 | #13076 | Spine ID and joint reactions keep driven coordinates unlocked under external loads; JR reserve check passes on capture-A |
| 2026-10-17 | #13060 | OS-8 branch-continuous capture-A IK: constant marker set, warm-started tie-break IK, continuous through impact |
| 2026-10-17 | #13055 | Make tests/tools a package so its test packages stop shadowing src/tools; collision gate green and unquarantined |
| 2026-10-17 | #13050 | Bump vendor/ud-tools to Tools 46c78180, carrying the Rust golfer solver closure-row parity with Python (Tools#5644) and the ChatDockWidget mixin split (Tools#5642); pins and divergence inventory kept in step |
| 2026-10-17 | #13046 | OSL-2: compose real OpenSim musculoskeletal lift models (Rajagopal-Lai-Uhlrich body, hand-held barbell, smooth sphere contact) replacing the LIFT-6 stub |
| 2026-10-17 | #13039 | de Leva 1996 Table 4 transcribed whole (both sexes, all rows) with load-time invariants; the humanoid table derives from it and closes to body mass without rescaling; fit-path mixed rows ratcheted (#13037) |
| 2026-10-17 | #13034 | OpenSim 4.6 engine: inverse dynamics, bias, gravity, drift, counterfactuals and reset work and raise on failure, joint names follow the state; committed golfer models load their club meshes (#13016) |
| 2026-10-17 | #13018 | ci: propagate pytest exit status through tee in heavy-test, soak and optional-stack lanes; add workflow pipefail lint test |
| 2026-10-17 | #13028 | Keep the runtime cancel event out of persisted chat stream metadata and validate JSON-serialisability before persisting; Model Explorer Docker: explorer-headless MP4 export, explorer-gui (noVNC) and explorer-gui-wslg compose profiles on one explorer-base stage; DbC entrypoint and contract guard; Launcher theme imports resolve via launcher_compat; Sidekick API child runs without uvicorn auto-reload (ready in ~10s); closing an engine tab resets its Running status (#12962); OpenSim Run Simulation no longer calls the removed Manager.setInitialTime.; Model Explorer opens swing-datastore results (MuJoCo, Drake, OpenSim, Pinocchio) and plays them on the shared-appearance golfer with timeline, plots and MP4 export; OpenSim explorer: resolve meshes from repo/package/OPENSIM_GEOMETRY, geometry-missing banner with capsule fallback, anthro golfer default; OpenSim preview: orient by model gravity (Z-up golf models no longer lie on their side), capsules only for bodies with missing meshes; Results Playback: wall-clock play with 0.25x/0.5x/1x speed selector, viewport scales to the dock, y-label no longer clipped; fix(drake): real pydrake forces/reset APIs and gravity sign (#12980, #12981, #12982); fix(gui): restore dead settings menu actions, vendored tile fallback, embedded C3D viewer (#12992, #12993, #12995); Resolve the club-head body across engine naming schemes via one shared alias table; add strict mode |
| 2026-10-17 | #13026 | Decompose launcher UI widgets (rebuild_grid, init_ui, LibraryWidget._setup_ui, CustomTitleBar) under the architecture budget and drop the expiring exceptions; Split launch_monitor_analytics gui.py into cohesive _gui_* mixin modules and decompose TerrainExplorerWidget._build_ui; remove expiring size/architecture budget exceptions.; Split MuJoCo motion-matching simulate_with_coefficients, compute_qpos_jacobian and compute_cost_gradient_analytical into helpers under the function-lines budget (bit-identical outputs); drop their expiring exceptions; Make pinocchio/drake SimOut real dataclasses and decompose simulate_with_coefficients, _record_rollout and fit_swing_drake under the architecture budget; delete 8 expiring exceptions; Introduce shared frozen SimulationConfig for pendulum run_simulation variants and BatchSwingConfig for BatchSwingOptimizer; delete 5 expiring architecture-budget exceptions; Group IK/eval tuning parameters into option objects (TrajectoryIkOptions, DualFrameIkOptions, EvaluationOptions) and delete the three expiring architecture-budget parameter exceptions; Split oversized development and agent narrative docs into index plus linked parts; drop four doc-size exceptions |
| 2026-10-17 | #12862 | Re-vendor refactored deferred_validation/handoff_validator (RM#2112) with handoff_schema and archive_policy; delete 3 expiring function-lines exceptions |
| 2026-10-17 | #12858 | Re-synced the chat child copy with the pinned Tools tree: ChatDockWidget.__init__ now takes ChatConnectionConfig/ChatPresentationConfig/ChatIntegrationHooks (Tools#4897), call sites updated, and the expiring architecture-budget exception removed. |
| 2026-10-17 | #12958 | Opt-in muscle_model_policy and unlock_driven_coordinates, Clarabel stage two for large trunks, SO solver/screen provenance, asset_id in distribution sensitivity |
| 2026-10-17 | #12917 | SPN-9c: full-body driving of the Bruno model (coordinate map, shoulder orientation match, club hand wrench and ground reaction loads, study option and receipt record) |
| 2026-10-17 | #13029 | Dedupe MakeHumanMeshGenerator: one class and one _convert_params_to_makehuman in _mesh_makehuman; mesh-generator tests marked unit; humanoid_character_builder: one Primitive and one SMPL-X mesh generator class, URDF generator and URDFWriter collision exclusions restored after #8322, precondition binding by name, and the builder unit tests now run in the unit gate; Pin the Bruno 2015 full-body model hash with the owner's 2026-10-16 SimTK licence acceptance; Senteler stays unpinned; Cross-engine _safe_cv returns NaN at near-zero mean; robustness_score fails closed and the API reports null with a reason instead of perfect robustness; Motion-matching smoke: headless GL probe and best-effort playback render (render_skipped in receipt) |
| 2026-10-17 | #12951 | R11: the R6 chain shaft runs in OpenSim (SpringGeneralizedForce) and MyoSuite (MuJoCo export) with static, modal, topology and same-input parity gates; chain bundles export with --shaft-model chain |
| 2026-10-17 | #12910 | R9b: OpenSim and MyoSuite viewers render the shared appearance (OpenSim registers R8 body meshes in the spec body frame; MyoSuite uses the MuJoCo appearance layer) |
| 2026-10-17 | #13033 | NSX-23 anthro trajectory allocation and NSX-61 (#12586) anthro kinematic ensemble acceptance reports on the NSX-60 foot-contact fixture; NSX-23 trajectory conic allocation: capacity effort, torque rate, c+/c- gross power with strict-convexity check; Consolidate ContactForceAllocator and redistribute_forces/trajectory onto the conic core (qp_core); DeprecationWarning shims, no SLSQP on convex allocation paths; NSX-25: epsilon-constraint Pareto fronts and dual-certified affine probes over the trajectory feasible set; NSX-26: reaction-restriction sensitivity (WELD6 to SHAFT_FREE5 to BALL3) with anthro report; NSX-31: open-loop versus feedback-assisted replay of allocated torques with a separate feedback budget (loop_nullspace/replay.py); NSX-40: loop-nullspace report schema v1, facade, service, async cancellable job routes with result cache; Lower-body loop: two-foot ground contact null space, GRF observations, real anthro envelope benchmark (NSX-60, NSX-22); NSX-62: inertial parameter ensemble over identifiability-gated parameters with LMI rejection, sensitivity report and anthro acceptance; NSX-70 hit-and-run sampling research spike on a low-rank temporal basis with mixing diagnostics; NSX-71 inverse optimal control of allocation weights: pre-registered held-out report, KKT-residual module and synthetic recovery tests; NSX-72: true closure alternatives with kinematics re-derived per grip model (research report, per-model admission); NSX-86: Drake AutoDiff constrained-dynamics derivatives (closed_loop_derivatives FULL) through the NSX-84 seam, canonical closure convention, capability rejection instead of FD fallback; NSX-91 paired nonlinear Monte Carlo on constrained replay (CRN, hold-out seeds, retained failures); NSX-92 noise-aware null-space allocation: minimum face-angle variance under declared M1/M2 with convex and local-certified solves, fronts and held-out MC verification; NSX-93: declared joint-impedance and delayed minimum-intervention feedback layer for torque-driven models; NSX-94: muscle-level co-contraction experiments (activation fibre, signal-dependent muscle noise, short-range stiffness) for OpenSim and MuJoCo-muscle, fail-closed capability registry; NSX-95 strategy catalogue S1-S12 with matched-CHS/matched-effort campaigns, prior reversals, held-out stability and variance budget; NSX-96: Saltelli/Jansen Sobol and PRCC global sensitivity screening with grouped indices on the planar oracle and low-rank basis; NSX-97: MOSAIC multi-swing variability consumer with UCM/ORT, Muller-Sternad TNC, direction comparison, phase and physical-time residuals and session-level train/test split (real-capture falsification deferred); NSX-98: two-track conformance rows for closed-loop sensitivities and variance across Pinocchio, Drake, MuJoCo, MyoSuite and OpenSim |
| 2026-10-17 | #13052 | Deadlift and clean-and-jerk (first pull) same-input bundles with hand-held bar IK, cross-engine bar-path parity and literature comparison; bench press and snatch deferred with measured reasons |
| 2026-10-16 | #12897 | Web spine viewer: /api/tools/spine-viewer routes and React page with always-visible scientific status and distribution source (SPN-15c) |
| 2026-10-16 | #12867 | Fujii 2007 coupled flexion is even in trunk rotation: new ParityCouplingLaw, fujii2007_lumbar law version 2, row numbers unchanged; Golf-term contributions attribute base, terminal and alignment rotations as named entries (segmental_contribution_breakdown) and close on the full pelvis-to-thorax term; Study requests carry the drive torso alignment and its calibration source; receipts record and hash it, and an unsupplied alignment makes the study unqualified; Pelvis residual height is the model extent along gravity at the first frame, with its source recorded in the budget and receipt; Movement-variant undeclared-term tolerance is enforced at study run time; a violation makes the run unqualified; A missing ExternalLoads data file raises before any study stage runs; relative datafiles resolve against the XML directory; Static optimization holds zero-capacity muscles at zero, fixing the early-sample failure on the synthetic fixture; Correct the golf-term evaluate docstring for side_bend_tilt input and output shapes |
| 2026-10-16 | #12896 | SPN-8c: translational DOFs in SpineInventory, extract_passive_elements bushing reader, Bruno names-only level map, asset-gated tests |
| 2026-10-16 | #12960 | Adversarial review 2026-10-16: 43 findings with draft PR pitches under docs/development/adversarial-review/ |
| 2026-10-16 | #13021 | Pin the Bruno 2015 full-body model hash with the owner's 2026-10-16 SimTK licence acceptance; Senteler stays unpinned |
| 2026-10-16 | #13020 | Confine launcher logo route to its asset directory (fixes unauthenticated arbitrary file read) |
| 2026-10-16 | #12966 | Render the address ball at the club's own face on the ground for every club (drop the stale capture ball and the duplicate static-pose MuJoCo ball) |
| 2026-10-16 | #12956 | CC0 short-hair head is now the default HeadSettings.model in every engine; parametric stays selectable |
| 2026-10-16 | #12796 | NSX-21: FeasibleSet with declared torque, friction-cone and grip constraints in null-space coordinates; NSX-73 research report scoping a muscle-redundancy layer above NSX envelopes, with elbow null-space example test; Add spine coupling laws (linear, state-dependent, MyoFullBody adapter) and regional fraction (SPN-2a); Spine distribution problem: geodesic least-squares distribution of pelvis-thorax rotation over levels under a fixed coupling law (SPN-2b); SPN-3 partial: cited coupling-law data and loader; verified laws only (bruno2015_regional, fujii2007_lumbar, uniform_reference); per-level Bruno and Pearcy pending; Add spine.golf_terms versioned golf-term registry (SPN-4a); Migrate SpinalLoadAnalyzer X-factor and crunch factor to the golf-term registry (SPN-4b), behaviour-preserving; SPN-4c: swing-metrics and API X-factor delegate to golf-term registry; SPN-7a: synthetic segmental spine OpenSim fixture generator and native composition check; SPN-7b: fixture coupler constraints, massless driver and lumped trunk muscles; SPN-7c: OpenSim lane guard turns absent-OpenSim spine skips into failures (UD_REQUIRE_OPENSIM=1); SPN-8a: OpenSim spine inventory, level maps and ISB axis check; SPN-8b: extract OpenSim CoordinateCouplerConstraints into a CouplingLaw; SPN-9a: golf humanoid motion reader with native pelvis-to-thorax rotation and SLERP resampling; SPN-9b: drive a coupled segmental model from golf trunk input with per-frame residuals and .mot export; SPN-9d: read OpenSim motions into SegmentalKinematicsSeries (inverse of export_mot); Spinal inverse dynamics with external loads: generalized effort per level and pelvis residual budget (SPN-10a); SPN-10b: trunk static optimisation with cost exponent p in {2,3}, lexicographic reserves with flagging, shared numeric core with the leg SO, per-coordinate ID accessor; SPN-10c: per-level joint reactions with SO muscle forces as a JOINT_REACTION SpinalLoadSeries; SPN-11b: golf literature context bands (Hosea 1990, Lim 2012) annotated beside model peaks, never acceptance; SPN-12: SpinalLoadAnalyzer heuristic/model provider split; SPN-13a: deterministic hash-pinned spine study request -> receipt (spine.study); Spine study requests can run a movement variant with fingerprint and receipt provenance (SPN-13d); SPN-13b movement-variant generator: five versioned pelvis->thorax variants by JCS-axis composition, model range check; SPN-13c: spine study aggregation tables; SPN-14a: distribution sensitivity within one model across shipped coupling laws, with recorded law restriction to the model's levels; SPN-15a: pure-data spine angle-stack and load-heatmap adapter plus Agg report figures; Add PyQt6 Spine Viewer tool with embed adapter, tile and feature-parity entry (SPN-15b); Rewrite segmental spine manual chapter 66 as a calculation-level reference after panel review; declare the Fujii flexion parity defect; correct the gravity-only residual wording; Record owner approval of the segmental spine chapter text (2026-10-15), split it into chapters 66 and 67 to fit the doc size budget, track the spine viewer web gap in #12815, and fix part-2 CI (lazy matplotlib import, SciPy read-only arrays, LOD/DRY, markers, help page, atlas regeneration); SPN-16: consolidate segmental spine manual chapter 66, registry blocker text, shared-infrastructure module directory and study run guide; Record segmental spine contract ownership; open Tools#5630 and OpenSim_Models#464; Run the Prettier pre-commit hook at commit time only so rebased branches can push; Fix main breakage from #12704: pinocchio JointModelRZ test lane and pinocchio_reference row typing |
| 2026-10-16 | #12950 | Explorer: OpenSim plot resolves markers from the model (no KeyError 'Hand'), MuJoCo Export Video uses the viewport camera, app exports write H.264 (#12946, #12947) |
| 2026-10-16 | #12880 | NSX-89: Simscape closed-loop derivative capability NONE with UNAVAILABLE reason, R2025b-only linearization payload gate, DV-12718 deferred record |
| 2026-10-16 | #12853 | NSX-85: Pinocchio analytic closed-loop derivatives, FULL reference for CLOSED_LOOP_DERIVATIVES |
| 2026-10-16 | #12881 | NSX-90 covariance propagation, adjoint sensitivity kernels and nonlinearity gate |
| 2026-10-16 | #12645 | Kinematic ensemble (admitted samples, rejection accounting, envelope union/intersection) and reproject_onto_closure delegating to project_to_closure (NSX-61) |
| 2026-10-16 | #12922 | Closing the launcher after a MuJoCo session releases renderers before widgets die. |
| 2026-10-16 | #12902 | MuJoCo explorer: 3D view no longer renders black in the launcher (renderer freed under its own GL context), Side quick-camera no longer crashes, Visualization tab scrolls instead of overlapping (#12884, #12885, #12889) |
| 2026-10-16 | #12852 | NSX-87: MuJoCo/MyoSuite hard-KKT finite-difference closed-loop derivatives (PARTIAL), reusing the NSX-84 routine; NSX-88: OpenSim closed-loop FD derivatives via NSX-84 routine, capability PARTIAL, muscle-insufficiency flag |
| 2026-10-16 | #12640 | NSX-22 static torque and hand-split envelopes with Chebyshev margin and sign classes |
| 2026-10-16 | #12921 | Club Head Speed no longer reads 0.0 for models naming the body clubhead. |
| 2026-10-16 | #12904 | Launcher survives slot errors, accurate engine status/hints, shutdown and first-click launch fixes (#12890 #12891 #12892 #12894) |
| 2026-10-16 | #12899 | Offscreen per-engine explorer contract tests; explorer opens MJCF, keeps every MJCF DOF, loads results, scrubs and exports MP4 via one shared frame source; OSIM loader accepts Class::Nested tags; results browser Export button follows playability; MuJoCo GUI gains Export Video (#12888) and plots no longer crash on unrecorded channels (#12887) |
| 2026-10-16 | #12869 | Refactor chat_service stream worker, motion_pipeline API routes and model-pack manifest parsing under the architecture budget; remove six expiring exceptions |
| 2026-10-16 | #12788 | R4: address knee flexion (owner 15 +-5 deg prior, reliable capture wins, hard no-hyperextension bound, per-knee target_source in the receipt; --knee-flexion defaults to capture) |
| 2026-10-16 | #12908 | Drake and Pinocchio MeshCat backends draw the shared model_appearance spec through one engine-neutral scene builder; local grey palettes removed (R9a) |
| 2026-10-16 | #12799 | Add DisturbanceSpec timestep-consistent disturbance contract, wrapped OutcomeDispersion statistics, and NaN (not 0) MetricStatistics.cv at near-zero mean (NSX-82) |
| 2026-10-16 | #12909 | R9c: GS3DX Simscape golfer takes its colours and trunk size from the shared model_appearance spec via a versioned JSON exporter (fail-closed MATLAB reader) |
| 2026-10-16 | #12907 | R7: realistic CC0 head for the shared parametric head slot - head.model cc0_mesh loads a MakeHuman-derived CC0 skin and short-hair mesh (assets/head_cc0, provenance + sha256 verified, fail closed) through build_head_parts, so MuJoCo and the OBJ export (R9a/b/c) get it with no per-engine code; default stays the procedural head |
| 2026-10-16 | #12903 | OpenSim loads alongside ezc3d: C3D stack imported lazily at startup, structured OpenSim probe reports the real import failure instead of Not Installed |
| 2026-10-16 | #12900 | Fix classic launcher startup: parent contract guard, ChatService ToolRegistry identity, data_explorer launcher_embed alias (#12893, #12895) |
| 2026-10-16 | #12876 | NSX-84: constrained tangent linearization (M2 preload) and closed-loop derivative capability seam with engine-free FD reference; NSX-84b: tangent state space for multi-DOF closed loops (k > 1), five-bar golden and gauge-law tests |
| 2026-10-16 | #12801 | Make the vendored Tools double_pendulum_golf the canonical golfer kinematics and closure Jacobian; golfer_kinematics/golfer_constraints become thin adapters pinned by a CI parity test (ADR-0056) |
| 2026-10-16 | #12734 | R6: opt-in flexible golf shaft as a rigid-link spring-damper chain (MuJoCo); generic STIFF profile flagged default_not_measured; rigid default byte-identical; other engines refuse chain specs |
| 2026-10-16 | #12790 | R8: one shared appearance and anthropometry spec (model_appearance.body_meshes) - engine-neutral skin/garment/shoe meshes in the spec body frame, trunk sized from spec visual_hints as one solid, default charcoal trousers hip to ankle, one plain palette, shared OBJ export; MuJoCo is now a thin translator |
| 2026-10-16 | #12877 | Generate the user manual TOC from real chapter headings and test that every manual link resolves |
| 2026-10-16 | #12835 | Divergence inventory: commit only stable per-path facts (v2 JSONL, one sorted record per line), drop the committed markdown, move totals/bytes/authorship to --report; different-path PRs no longer conflict |
| 2026-10-16 | #12865 | Split the user manual and functional test plan into an index plus chapter files under the doc size budget; delete both expiring exceptions. |
| 2026-10-16 | #12854 | Split generated ORG-01 capability baseline into index plus linked parts; drop three doc-size exceptions |
| 2026-10-16 | #12863 | Split scripts/_urdf_campaign_issues.py into a domain package and drop the stale/expiring file-size exceptions; research scripts left pinned. |
| 2026-10-16 | #12857 | Refactor model_converter exporters and schema validator under the architecture budget; delete 5 expiring exceptions; add golden-output tests |
| 2026-10-16 | #12834 | Golfer generator adapter derives coordinate and control widths from the TB-00 registry (u_dim=6) instead of a literal 5, and rejects a sealed u_dim that disagrees; sealed NM-01/NM-09 records untouched pending owner decision |
| 2026-10-16 | #12837 | Perturbation panel renders the worker's PerturbationSummary on batch finish and shows an undefined CV as n/a |
| 2026-10-16 | #12830 | Receipt base_spec_sha256 hashes the committed spec before foot-progression seeding; foot_progression_mode and foot_seed_deg recorded as explicit receipt fields so style-feet canonical receipts chain to the committed spec |
| 2026-10-16 | #12641 | Constrained ZTCF/ZVCF via the KKT projector, ConstrainedEquationsOfMotionProvider; counterfactual rollouts integrate real dynamics (m_eff mock removed) |
| 2026-10-16 | #12800 | NSX-80: variability-sensitivity specification, symbolic goldens, hypotheses and unverified citations; NSX-83: event-triggered impact outcome map with saltation correction, face-angle availability and Tools ball-flight bridge |
| 2026-10-16 | #12644 | NSX-55: Simscape closed-loop export format, loader and R2025b exporter (planar rig validated; golf model deferred as DV-12583) |
| 2026-10-16 | #12812 | Fix main breakage from #12704: pinocchio JointModelRZ test lane and pinocchio_reference row typing |
| 2026-10-16 | #12809 | Run the Prettier pre-commit hook at commit time only so rebased branches can push |
| 2026-10-16 | #12780 | NSX-80: variability-sensitivity specification, symbolic goldens, hypotheses and unverified citations |
| 2026-10-16 | #12663 | NSX-73 research report scoping a muscle-redundancy layer above NSX envelopes, with elbow null-space example test |
| 2026-10-16 | #12637 | NSX-21: FeasibleSet with declared torque, friction-cone and grip constraints in null-space coordinates |
| 2026-10-16 | #12798 | Fix garment_classification.tex row terminators so the golfer appearance reference compiles; latex-references CI now builds the full-body model reference masters |
| 2026-10-16 | #12804 | Split full-body HANDOFF and anthropometry REVIEW under the doc-size budget; renew only the stale chat child-copy architecture exception for 14 days (#12803) |
| 2026-10-15 | #12792 | Bump vendor/ud-tools to Tools e3cf5a43, carrying the corrected golfer club Jacobians (Tools#5627 Rust pendulum-core, Tools#5628 Python/JAX forward kinematics y-sign, Tools#5626); pins and divergence inventory kept in step |
| 2026-10-15 | #12782 | Re-solve the MSK address calibration on the widened forearm range so the lead pronation leaves its old 90 deg bound |
| 2026-10-15 | #12786 | Nightly motion-matching-smoke.yml runs tests/integration/motion_matching (slow, --timeout=0) via scripts/ci/run_motion_matching_smoke.sh on the fleet; a failure opens or updates an issue and the red scheduled run reaches fleet-main-health; not coupled to heavy-tests-opt-in parity |
| 2026-10-15 | #12781 | ADR-0055: segmental spine kinematics, load observables and golf-term definitions (SPN-0); Primary data extraction for segmental spine (Bruno, Fujii, Wilke, golf loads) with provenance flags; Segmental spine research reference scaffold, provisional manual chapter 66 and registry blocker; SPN-1a: spinal level enumeration, regions and adjacency (spine.levels); SPN-1b: segmental spine per-level transforms and ordered composition (spine.segmental); SPN-5 spinal load series, sign conventions and tension-positive axial bridge; SPN-6a: spine model asset manifest and local-source SHA-256 admission |
| 2026-10-15 | #12783 | Smoke test dynamics RMS 0.51 m was not a code regression: test_mujoco_native_match_smoke[driver] ran the pipeline's default native v2 spec, not the canonical anthro_driver document its 0.15 m bound and 0.089 m figure come from; the driver now passes --spec full_body_spec_anthro_driver.json (current main reproduces receipt anthro_driver: IK 0.0465, dynamics 0.0719), pinned by a fast unit test; bounds unchanged |
| 2026-10-15 | #12779 | OpenSim MSK renders draw the wrapped-finger grip hands on the club and hide the flat Rajagopal hand bones |
| 2026-10-15 | #12776 | Fix the 'spec has no club body' crash on the MuJoCo-native driver run: assembly_from_spec now recognises the native Simscape club (segmented 2.5in Butt grip, no Grip solid) so the default v2 spec carries a club body; Lane.set_club_targets fails fast via club_face_target.require_club_assembly naming the spec source. The smoke test now gets past set_club_targets but still fails on dynamics marker RMS (0.51 m vs 0.15 m bound), a separate regression; the smoke is marked slow and runs in no CI lane |
| 2026-10-15 | #12733 | NSX-81: add orthogonal conditioning axis to QuantityRecord so INVARIANT quantities can declare hypotheses they are conditioned on |
| 2026-10-15 | #12731 | R3: trail-foot toe-out owner style target (TRAIL_TOE_OUT_STYLE_DEG = 20 deg, single-sourced from DEFAULT_TOE_OUT_DEG) via a new --foot-progression style mode, now the canonical default; capture mode still lets the measurement win; per-foot target_source in receipts; driver trail -2.9 -> 19.8 deg, 7-iron -9.8 -> 13.5 deg (hip_rotation_r at its -40 deg limit) |
| 2026-10-15 | #12696 | R1: trunk, pelvis and shoe appearance cross-sections now oriented from spec landmarks (ML > AP); lofted_segment requires an explicit width axis for non-round sections; new model_appearance.cross_section; cross-engine trunk guard; golfer_appearance_reference.tex master + trunk_cross_section section |
| 2026-10-15 | #12704 | Pinocchio reference adapter finite-differences the trajectory on the base Lie group (world linear, body angular via the quaternion log) and feeds the result through the NSX-50 velocity map, instead of differentiating raw native q |
| 2026-10-15 | #12679 | Correct the club y-sign, hub direction and club Jacobians in the JAX golfer path; club points and club-angle Jacobian columns now share backend-agnostic formulas with the NumPy kinematics, pinned by NumPy-vs-JAX parity tests |
| 2026-10-15 | #12664 | NSX-74 publication gate checklist for loop null-space results |
| 2026-10-15 | #12638 | NSX-17 invariant quantities with validity gates |
| 2026-10-15 | #12639 | Two-track cross-engine closed-loop conformance harness (numerical snapshot equality plus propagated scientific envelopes), registry entries per engine pair, native OpenSim adapter registered in the engine factory |
| 2026-10-15 | #12695 | R5: one clubhead measurement point (face centre via club_face_target.face_centre_path) for the fd_phase impact split, the exporter impact detector and a new dynamics.fd_phase.clubhead_speed receipt block (simulated/reference/capture speed, % error, peak-timing offset); scripts/diagnose_clubhead_speed.py decomposition |
| 2026-10-15 | #12694 | Pelvis wears the active outfit's pants material: body classification is order-independent (leaf segment before parent folder, exact before prefix before substring, equal-specificity ties raise), GolfSwing3D LowerTorso maps to pelvis, and check_clothing enforces pelvis garment == pants for every preset |
| 2026-10-15 | #12636 | NSX-52: MyoSuite CLOSED_LOOP_SNAPSHOT via MuJoCo adapter reuse; kkt_regularization recorded, FULL only at 0 |
| 2026-10-15 | #12632 | NSX-16: grip-frame split coordinates, grasp map kernel and bare-club net wrench |
| 2026-10-15 | #12699 | Pinocchio reference adapter maps the free-flyer world linear velocity/acceleration into Pinocchio's LOCAL frame via the NSX-50 velocity map |
| 2026-10-15 | #12631 | NSX-15 closed-loop decomposition: dimension accounting, zero-side three-case solve, minimum-metric torque |
| 2026-10-15 | #12630 | NSX-53 native Drake force adapter with closed-loop snapshot; synthetic Drake route quarantined to tests |
| 2026-10-15 | #12625 | NSX-14 kinematic and contact admission gate (admit, AdmittedTrajectory) |
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
