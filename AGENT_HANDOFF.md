# Agent Handoff — UpstreamDrift

Last updated: 2026-08-26
This is current operational state; Git/GitHub retain history, and epic #8557
is the single proximal-to-distal completion authority.

## Markerless Mocap Program (#9063)

- ADR-0041 assigns camera, observation, calibration, timing, session, reconstruction, and C3D contracts to Tools #4706; UpstreamDrift owns orchestration, UX, persistence, and biomechanics adapters; AffineDrift owns sanitized publication. Tools PR #4734 remains a protected candidate; do not repin `vendor/ud-tools` to a feature head, and let UpstreamDrift #9069 follow its immutable merge. Existing ingestion #4558 and duplicate-reader debt #8865 are inputs, not live-lab implementation: there is no physical-lab qualification or camera, inference, C3D round-trip, commercial, or human-performance claim.

## Repository and Publication Authority

- UpstreamDrift owns scientific sources, models, evidence registers, and the release bundle. AffineDrift is an immutable, revision-pinned public projection; Tools owns reusable consumers and source contracts.
- Remote `main` is `99acc997a97b3d97cb4ddd857b79bedd4a66f290`.
- The current branch publication is 245 pages, SHA-256 `48800a40a899406a13787c93c8282f33688eec819f664c961ff42f093efa28fa`, with 194 URI links and 255 outline entries. Archival qualification remains false because the PDF is untagged and retains Type 3 and unembedded fonts.
- The complete audit contains 1,142 reviewed candidates, 313 atomic claims,
  132 numeric contracts, and 427/427 verified literals. Outcomes are 293
  supported only within declared estimands and boundaries, five inconclusive,
  15 untested, and zero contradicted. All 47 public release claims have review
  dispositions and scientifically open gates. The bundle has 636 artifacts.

## Coordinate Force-Source Attribution (#9059)

- Tools squash `8dc4512184` is the `force-attribution/v1` authority, consumed
  through the vendor pin and thin biomechanics gateway. The 135-program grid
  retains 91 qualified impacts and separates impulse, work, power, mapping
  residuals, and speed optima in declared coordinates.
- The force-only endpoint map is rank deficient; its residual couple remains
  explicit. This is synthetic planar evidence, not a measured grip wrench,
  muscle attribution, human strategy, or coaching authority.

## Hybrid-System Topology Contract (#9027)

- PR #9049 is merged as `a4b7d3f9b`. Its eight-tier typed topology covers
  states, controls, constraints, modes, guards, resets, impacts, actuator
  dynamics, uncertain events, observables, limitations, and blockers.
- Three tiers are implemented, three are partial, and the participant-
  calibrated and governed-human tiers are explicitly unavailable. This does
  not establish observability, controllability, stability, controller ranking,
  participant validity, or coaching interpretation.

## Double-Pendulum Rank and Identifiability (#9092, #9104)

- PR #9100 is merged as protected squash `b8508197e`. It retains raw dimensional
  matrices for traceability but interprets rank and condition only after
  declared state, control, output, and time scaling.
- Four synthetic operating points, three finite-difference multipliers, three
  scale scenarios, and 16 countermodels in each class are registered. The
  zero-input and zero-output killswitches return rank zero.
- Published floats use six significant digits for cross-platform identity;
  rank decisions are computed at full precision before serialization. These
  local results do not establish structural/global rank or human strategy.
- PR #9108 is merged as `6f9b068a5`. It proves analytic
  physical-map rank seven/nullity four, retains three exact nonunique families,
  and gives registered finite-record rank seven with condition 180.853;
  equivalent-unit change is `4.44089e-16`, while zero motion has rank zero.
- Its Gaussian Fisher screen is an oracle-kinematics lower bound only: at 1 N m
  noise, the worst relative 95% half-width is 0.123266 for the full record and
  498.504 for the first 10%. Practical/participant identifiability remains
  unestablished. Its protected 309-claim snapshot is preserved in migration
  history and has been superseded by the current #9027 projection.

## Scaled Constraints and Feasible Closed-Loop Margins (#9027, #9113)

- PR #9110 is merged as `d16212ac7`. It adds source-bound planar
  closure and bilateral wrench-map audits. It requires explicit 1 rad/0.75 m
  generalized-coordinate scales, a 0.10 m wrench scale, and `1e-12` rank tolerance.
- Regular planar rank/nullity is 4/1; the constructed adverse alignment is
  3/2 but is not a qualified anatomical pose. Separated/coincident point-force
  maps are 5/1 and 3/3; near-coincident rank is tolerance sensitive.
- #9113 is implemented in open PR #9114 on
  `research/9113-closed-loop-margins`. Exact triangle
  closure covers both branches and 181 phases each; all 362 nominal samples
  close below `1.67e-16` m with rank/nullity 4/1. Exact 0.03 m and 1.53 m
  degeneracies are 3/2. A five-offset by five-tolerance matrix retains the
  observed numerical rank boundary without calling it physical.
- Phase, scale, feasible/impossible geometry, equivalent-unit, and manufactured
  rank controls pass. Python 3.12 CI exposed platform SVD roundoff; the corrected
  publication bounds all expected-zero SVD diagnostics by conservative powers
  of ten, including exact null singular values, while rank decisions retain raw
  precision. The 58-test focused gate and 636-artifact release validation pass.
  This is planar kinematics only: no force, anatomy, passive torque, human
  occurrence, strategy, or coaching inference is authorized.

## Structural Campaign and Recovery Boundary (#8800)

- The separate #8752 constitutive campaign completed on ControlTower and was
  transferred at branch commit `2fa6cf886`; it is not on remote `main`.
- #8800 is frozen at source `1bd4d57da7bd257b76b42b3cc19524b283b5f748`.
  The seven-corner plan is 93/830; 737 checkpoints and 27 nominal ground
  branches are absent. `release_evidence=false`; a persisted `running` state
  is stale and no campaign process is active.
- Checkpoints remain under `C:\Users\diete\Campaigns\UpstreamDrift-8800-1bd4d57da`.
  ControlTower's WSL VHDX is unreadable (`0x80070570`); preserve it and the C:
  checkpoints. Do not retry mounts, run CHKDSK, mutate frozen plans, or start
  a replacement campaign without an explicit recoverability decision.
- DeskComputer is fully runner-drained and must not run uncertainty campaigns
  or large parallel tests. Use serial bounded tests with `-n 0` only.

## Human-Evidence Boundaries

- PR #9017 is merged at `ce6fce1c2b8a6e50e410d16d31e219fabcb154e1`
  and provides fail-closed participant split, processing, frame-transform, and
  event-detector authorities for #9004.
- #9004 remains open because no qualifying governed participant trajectory
  dataset or held-out human outcome is registered. Simscape exports, fixtures,
  tutorials, GolfDB labels, and launch-monitor records are not substitutes.
- #8556 remains externally blocked by the absence of governed synchronized
  bilateral six-axis grip-wrench participant data. Synthetic traces and
  paper-level curves must never substitute for human validation.

## Active Dependencies

- #8800 blocks the final #8752/#8668 audit. #8443, #8448, #8449, #8450,
  #8595, #8668, #8684, and #8796 remain open.
- #9099 branch has no PR. Protected/OGLaptop-4 `33018178822`/`98341732390`
  pins exit 139; RED `33018492146`/`98342734690` fails the ownership contract.
  `c550c6f9` restores process ownership; 3.11 `33019029879`/`98344479337` and
  3.12 `33019364615`/`98345614325` are GREEN at `361a914d`; review is pending.
- #8358's Tools gateway and analyses are merged; UI, localized-perturbation, and presentation criteria remain unaudited. Merged #9096 changes no production API, schema, or vendor pin.
- #9107 exposes one canonical top-level `bunkershot3d` identity, excludes `src.bunkershot3d`, and passed 36 build-hook/unit contracts plus an isolated exact-wheel probe (`ec3b6c6223f08ebfe1a256f5a3eda3b00209a081fe2cbbe01bd9a0e8ae6f0d18`). Full wheel runtime remains unqualified because inherited `src.api`/Tools-config alias and `sidekick --help` failures remain.
- #9107 now transfers separate immutable wheel/source artifacts through a fixture-only sparse smoke checkout within a 20-minute bound; fresh protected evidence is required, and no runner change or smoke exemption is authorized.
- Tools #4142 remains open. AffineDrift PR #3942 is merged; the next public projection must pin a qualified UpstreamDrift merge commit.

## Scientific and Review Invariants

- Distinguish energy transfer, momentum redistribution, joint work,
  constraint forces, and clubhead speed. Preserve falsifiers, adverse cases,
  identifiability limits, uncertainty, countermodels, and unavailable states.
- Model evidence is never promoted to human validation. No current result
  establishes anatomy, physiology, equipment calibration, injury, coaching
  efficacy, or a universal speed benefit.
- PRs target `main`; use full PRs and let required protected checks govern
  readiness. The live ruleset requires zero approvals, so do not request a
  named maintainer as a standing gate. Never force-push, admin-merge, bypass
  checks, add quarantine debt, or edit the vendor pin independently of a
  protected Tools authority.

## Focused Validation

```powershell
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.run_constraint_internal_force_diagnostics validate
python -m scripts.research.proximal_distal_energy.run_closed_loop_singularity_margin validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python -m pytest -n 0 -q tests/research/test_closed_loop_singularity_margin.py tests/research/test_closed_loop_singularity_margin_evidence.py tests/research/test_constraint_internal_force_diagnostics.py tests/research/test_constraint_internal_force_diagnostics_evidence.py tests/research/test_numeric_evidence.py tests/research/test_proximal_distal_release_bundle.py tests/research/test_claim_numeric_registry.py
python scripts/ci/check_file_size_budget.py
```

Passing shared gates does not close narrower scientific or external-data gates.
