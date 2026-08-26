# Agent Handoff — UpstreamDrift

Last updated: 2026-08-26
This is current operational state; Git/GitHub retain history, and epic #8557
is the single proximal-to-distal completion authority.

## Markerless Mocap Program (#9063)

- ADR-0041 assigns camera, observation, calibration, timing, session, reconstruction, and C3D contracts to Tools #4706; UpstreamDrift owns orchestration, UX, persistence, and biomechanics adapters; AffineDrift owns sanitized publication. Tools PR #4734 remains a protected candidate; do not repin `vendor/ud-tools` to a feature head, and let UpstreamDrift #9069 follow its immutable merge. Existing ingestion #4558 and duplicate-reader debt #8865 are inputs, not live-lab implementation: there is no physical-lab qualification or camera, inference, C3D round-trip, commercial, or human-performance claim.

## Repository and Publication Authority

- UpstreamDrift owns scientific sources, models, evidence, and release; AffineDrift
  is the immutable public projection and Tools owns consumers.
- Remote `main` is `99acc997a97b3d97cb4ddd857b79bedd4a66f290` (PR #9109); active branch: `research/9123-trajectory-control-authority`.
- The 247-page publication is SHA-256 `8356e14e33fda24c1cbd79ad48485a76dfe6240c34fb9ed5d505ef95d7251907`,
  with 194 URI links and 255 outline entries; archival qualification is false.
- The complete audit contains 1,154 reviewed candidates, 317 atomic claims,
  136 numeric contracts, and 455/455 verified literals. Outcomes are 297
  supported only within declared estimands and boundaries, five inconclusive,
  15 untested, and zero contradicted; all 49 release claims are reviewed with
  zero open release reviews.

## Coordinate Force-Source Attribution (#9059)

- Tools squash `8dc4512184` is the `force-attribution/v1` authority, consumed
  through the vendor pin and thin biomechanics gateway. Its 135-program grid
  retains 91 impacts and separates impulse, work, power, residuals, and optima.
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

## Closed-Loop Margins and Phase/Event Stability (#9027, #9113, #9116)

- PR #9110 is merged as `d16212ac7`. It adds source-bound planar
  closure and bilateral wrench-map audits. It requires explicit 1 rad/0.75 m
  generalized-coordinate scales, a 0.10 m wrench scale, and `1e-12` rank tolerance.
- Regular planar rank/nullity is 4/1; the constructed adverse alignment is
  3/2 but is not a qualified anatomical pose. Separated/coincident point-force
  maps are 5/1 and 3/3; near-coincident rank is tolerance sensitive.
- #9113/#9114 is merged as `847b9abd3`. Both exact triangle branches and 181
  phases each close below `1.67e-16` m with rank/nullity 4/1; exact 0.03 m and
  1.53 m degeneracies are 3/2. Controls retain numerical rank/tolerance
  dependence without labeling it a physical or anatomical threshold.
- #9116 is published in PR #9117 with the Python 3.11/3.12 portability fix; it also reconciles the lock-header and architecture-budget gates.
  It propagates the exact discrete RK4 variational map at 0.125 ms and separates
  finite-window gains from transverse event-time derivatives, rejects grazing
  guards, tests saltation controls, and suppresses Floquet output because the
  scaled periodicity residual is 1.48546 versus a `1e-6` tolerance.
- Python 3.11/3.12 reproduce the report exactly. Step refinement is reported at
  two significant digits (`6.9e-6`), and direct-rollout agreement is an upward-
  rounded `4e-7`; raw `1e-5`/`1e-4` gates and full NPZ arrays remain. Unit
  controls use a raw `1e-12` gate and conservative decade bounds. Gains span 0.093456--8.33244; the event
  denominator is 35.0258 s^-1 and direct/implicit agreement is `5.98e-5` s.
  These are local synthetic diagnostics, not asymptotic stability, neural
  timing demand, participant robustness, passive torque, or coaching evidence.

## Trajectory-Varying Control Authority (#9123)

- Local branch commits `5d82cd890` through `3c9358636` implement and publish
  the exact-discrete trajectory-varying authority milestone on the immutable
  #9116 source identity. Do not push or open its PR until dependency PR #9117
  is protected-merged and the branch is reconciled with the resulting main.
- The registered event is at 0.349256 s with guard residual `1.30562e-13` and
  transversality 35.0258 s^-1. Full-state rank is four and the explicit
  orthonormal event-tangent rank is three; the guard-normal null direction is
  not classified as actuator loss.
- Four channel cases, six direct nonlinear pulse checks, four matched frozen-
  local countermodels, three input-step refinements, three integration-step
  refinements, additivity, zero-input, and equivalent-unit controls pass. The
  maximum direct-pulse residual is `4.11321e-08`; the maximum integration-step
  residual is `1.26519e-07`.
- The architecture refactor introduces owned numerical, pulse, and guard
  contracts and bounded figure/report helpers. It is evidence-inert: JSON,
  NPZ, and PDF hashes are unchanged, and the regenerated figure is pixel-
  identical. The architecture, claim, release, title-case, and file-size gates
  pass with 652 release artifacts and no mismatches.
- These results remain local first-order diagnostics on one synthetic open-loop
  analytical trajectory. Bounded nonlinear feasibility, controller ranking,
  human capacity, passive torque, robustness, and coaching inference are
  explicitly unavailable.

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
- #9049, #9100, #9108, #9110, and #9114 are merged; #9116/PR #9117 is
  protected and capacity-bound. #9123 is complete locally but cannot enter its
  protected publication workflow until #9117 merges.
- #9124 is the registered next control milestone: bounded nonlinear event-
  reaching feasibility under explicit model-scenario torque and torque-rate
  bounds. It is a child of #8557/#9027/#8449 and depends on protected #9123;
  do not infer physiological bounds, controller ranking, or coaching guidance.
- #8358's Tools gateway and analyses are merged; UI and presentation criteria
  remain unaudited. Merged #9096 changes no production API, schema, or vendor pin.
- #9107 exposes one canonical top-level `bunkershot3d` identity, excludes `src.bunkershot3d`, and passed 36 build-hook/unit contracts plus an isolated exact-wheel probe (`ec3b6c6223f08ebfe1a256f5a3eda3b00209a081fe2cbbe01bd9a0e8ae6f0d18`). Full wheel runtime remains unqualified because inherited `src.api`/Tools-config alias and `sidekick --help` failures remain.
- #9107 now transfers separate immutable wheel/source artifacts through a fixture-only sparse smoke checkout within a 20-minute bound. This branch also restores the tested Linux-only native PyInstaller release job after the consolidated matrix regressed that contract; fresh protected evidence is required, and no runner change or smoke exemption is authorized.
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
python -m scripts.research.proximal_distal_energy.run_phase_event_stability validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python -m pytest -n 0 -q tests/research/test_phase_event_stability.py tests/research/test_phase_event_stability_evidence.py tests/research/test_proximal_distal_release_bundle.py tests/research/test_claim_numeric_registry.py tests/unit/research/test_proximal_distal_claim_audit.py
python scripts/ci/check_file_size_budget.py
```

Passing shared gates does not close narrower scientific or external-data gates.
