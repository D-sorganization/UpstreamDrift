# Swing-to-Impact Dynamics and Acoustic Sweetness Program

Parent [#9700](https://github.com/D-sorganization/UpstreamDrift/issues/9700).
This delivery: [#9701](https://github.com/D-sorganization/UpstreamDrift/issues/9701).
Provider: [Tools #5068](https://github.com/D-sorganization/Tools/issues/5068).
Theory: [AffineDrift #4253](https://github.com/D-sorganization/AffineDrift/issues/4253).
Reviewed 2026-09-07; current phase is inventory and integration design.

## Outcome and Scientific Boundary

Determine how delivered rigid state, impact position, applied force/torque,
shaft deformation/velocity and prestress, and grip impedance change ball launch,
head response, hand vibration and radiated sound. These are separate outcomes.
Perceived sweetness requires a controlled perception study, not a scalar mass
or ball-speed score.

The older heavy-hit treatment asserted universal isolation that its 1-D fixture
does not prove. Local shaft response and existing preload are distinct from
late reflections at the hands. Centrifugal tension produces geometric stiffness;
Coriolis coupling is not an added material modulus. The detailed derivation and
primary-source ledger live in AffineDrift's canonical heavy-hit route and
`docs/development/impact-acoustics/`. This document owns integration and study
execution requirements, not a second copy of the theory.

## Baseline Inventory and Reuse

Reviewed base: `40308a0cf103ceb9a8b81b0fa62d4f5a34b24b83`. The tracked
`vendor/ud-tools` pin is `eab74a901a7c8467e1997049a73e2cfd2df74428`.
Tools-main capabilities at `b4875be19` must be checked against that pin before
use. This change does not bump or edit it.

| Surface                                                                                                                      | Existing Capability                                                                            | Reuse / Limitation                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `src/shared/python/physics/impact_model/` and `_impact_physics.py`, `_impact_recorder.py`                                    | Impact models, solver and event recording; scalar effective-MOI approximation remains explicit | Use as baseline and regression surface; reconcile provider ownership before migration              |
| `src/shared/python/physics/flexible_shaft.py`, `_shaft_fem.py`, `_shaft_model.py`                                            | Rigid/FE shaft interfaces, distributed stiffness/mass, Rayleigh damping and cantilever support | Reuse assembly and limits; no automatic qualification of rotating contact-band head/shaft dynamics |
| `scripts/research/proximal_distal_energy/shaft_beam_reference.py`                                                            | Existing FE plus head mass/inertia and synthetic modal identification                          | Numerical reference; not equipment calibration                                                     |
| Same research root: `moving_base_modal_shaft.py`                                                                             | Moving-base/two-hand dynamics with distributed bending basis                                   | Preserve mode/basis identity and rigid/modal coupling when exporting states                        |
| Same research root: `articulated_shaft.py`, `articulated_shaft_forward.py`, `articulated_shaft_atlas.py`                     | Articulated bending/torsion comparisons and evidence pipeline                                  | Keep synthetic parameters, small-deformation bounds and protected authority                        |
| Same research root: `articulated_forward_attribution.py`, `articulated_contact_events.py`, distributed contact/event modules | Work, contact events and attribution infrastructure                                            | Same-trajectory bookkeeping is not a causal divergent-trajectory experiment                        |
| Same research root: `two_hand_wrench.py`, `bilateral_wrench_identifiability.py`, `bilateral_wrench_sensor_qualification.py`  | Bilateral wrench and observation/qualification pathways                                        | Read governed contracts before new sensors or participant claims                                   |
| Same research root: `e1c_impact_sensitivity.py`                                                                              | Sensitivity to the definition of a swing impact/score event                                    | Not a finite ball–club collision simulator                                                         |
| `tests/api/test_impact_explorer_mount.py`, launcher/API infrastructure                                                       | Existing Impact Explorer integration                                                           | Keep host behavior and provider APIs separated                                                     |
| `docs/research/proximal_distal_energy_transfer/`, `AGENT_HANDOFF.md`                                                         | Protected publication and experiment-registration contracts                                    | No regeneration or relabeling of old evidence                                                      |

Existing test entry points include `tests/unit/test_flexible_shaft.py`,
`test_shaft_engine_integration.py`, `tests/unit/physics/test_impact_physics_value_assertions.py`,
`test_impact_friction_axis_and_gear_offset.py`, and the research tests
`test_shaft_beam_reference.py`, `test_moving_base_modal_shaft.py`,
`test_moving_base_modal_shaft_evidence.py`, `test_articulated_shaft.py`,
`test_articulated_shaft_forward.py`, `test_articulated_shaft_atlas.py`.
These are discovered validation entry points, not a claim that all engines or
research campaigns were executed during the inventory.

Tools supplies `golf_club` profile/statics/modal/fitting/mesh-inertia utilities,
the 1-D coupling model, `swing_sim.impact`, model/delivery interchange, variation
and impact-report contracts. Its first new reference is `impact_mobility` under
#5069. Generic audio/FFT/fitting facilities exist in Tools, but the inspected
fleet code does not establish a calibrated clubhead-radiation-to-microphone
pipeline or a blinded player-sweetness dataset.

## Implementable Work Breakdown

| Slice | Issue                                                                 | Deliverable                                                                              | Dependencies                                                  |
| ----- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| IA-U1 | [#9701](https://github.com/D-sorganization/UpstreamDrift/issues/9701) | This inventory, integration design and turnover                                          | Current source review                                         |
| IA-U2 | [#9703](https://github.com/D-sorganization/UpstreamDrift/issues/9703) | Frame/power/energy-compatible swing and elastic state adapters, pinned Tools consumption | U1; Tools #5069 and #5072                                     |
| IA-U3 | [#9704](https://github.com/D-sorganization/UpstreamDrift/issues/9704) | Registered counterfactuals and spatial/sensitivity studies                               | U2; Tools #5071, #5072, #5073, #5075                          |
| IA-U4 | [#9705](https://github.com/D-sorganization/UpstreamDrift/issues/9705) | Synchronized physical/acoustic validation and blinded listening                          | U2/U3; Tools #5074/#5075; actual equipment/protocol readiness |

All code is tests first, DbC, LoD and DRY. Numerical laws and shared wires belong
in Tools. Engine adapters, research registration and execution belong here.
AffineDrift consumes qualified evidence through its existing provenance rules.
Keep epics open until their actual numerical and empirical acceptance criteria
are met; a merged planning document completes only U1.

## Pre-Impact State Contract

Define a versioned immutable bundle before implementing a new engine adapter:

- Equipment, ball and calibration IDs, exact source hashes, model tier and units.
- Source time, event bracket/interpolant and uncertainty; world/head/grip frames
  and proper rotations with explicitly declared conventions.
- Head COM pose, linear/angular velocity and full COM inertia; contact-point
  offset, actual local normal/face curvature, ball state and spin.
- Shaft distributed profile, modal basis/version/normalization, amplitudes and
  velocities, prestress field and compatible boundary state.
- Each hand's applied wrench with origin/frame, impedance identification and
  support/control assumptions; generalized constraint and velocity conventions.
- Origin of every field: measured, identified, prescribed, synthetic or absent.

Missing elastic state is not zero elastic state. An adapter that cannot represent
a requested degree of freedom must refuse the request or return a documented
lower model tier. Summing masses from MJCF/URDF/OSIM cannot supply constrained
contact inertia or a measured six-axis grip impedance.

Transform forces, moments and velocities together and test wrench power.
Interpolate on a declared interval, not an unbounded extrapolation through
contact. Exporting a reduced mode basis must preserve its represented energy
and report projection residuals; do not reset shaft energy when changing solvers.

## Counterfactual Design

Two distinct families are required:

1. **Matched input:** identical declared control/wrench program and initial
   conditions, varying equipment/boundary model. Delivery may change, so report
   the total effect including that mediation.
2. **Matched rigid delivery:** identical head/ball pose/twist, contact point,
   normal and equipment; vary explicitly declared elastic phase, preload or
   grip condition. Construct physically compatible states and record the
   energy/work difference induced by the intervention.

Within each family compare detached rigid head, existing relaxed lumped chain,
distributed unloaded/preloaded shaft and flexible head/contact tiers. Sweep
heel/toe/high/low contact, speed/loft/path, tensor inertia/CG, force and torque
origin, axial tension, bending/torsion phase, damping and grip impedance.
Include interactions and negative/null results, not only a favorable baseline.
Changing stiffness may change restitution in a penalty model; hold calibration
fixed where the causal question requires it and disclose every retuning.

Zero applied control is not zero passive load, zero initial strain energy or
detachment. Frozen-state and forward-trajectory comparisons must be labeled
separately. Discrete event attribution in current research fixtures is not
evidence of a real sub-millisecond ball collision.

## Reproducibility and Acceptance

Use immutable registration, deterministic IDs and random seeds, exact code/data
hashes, atomic resumable outputs and typed failures. No silently skipped cases.
Keep raw histories plus derived launch/spin, head rotation, contact impulse and
duration, early/late vibration and qualified observer pressure. Every summary
must carry model assumptions, units, calibration source and validity limits.

Numerical gates precede effect claims: analytic rigid/beam limits, rotation
covariance, momentum and energy/work closure, passive dissipation and gyroscopic
zero work, event refinement, independent time/mesh/modal convergence and
installed-provider compatibility. Tolerances must be registered before outcome
inspection; report absolute floors near zero and separate contact, structural
and acoustic convergence. Cross-engine agreement does not validate physiology.

Physical validation progresses from measured inertia/static profiles and
complex FRFs to tensioned/gripped modal tests, plate/ball fixtures, assembled
clubs, repeatable rigs and players. Inventory actual available hardware before
scheduling; no equipment purchase or participant recruitment is authorized by
this document. Store calibration, sensor loading, clock alignment, bandwidth,
clipping checks, microphone geometry, environment and trial exclusions.

For sweetness, randomize/blind identity and order; separate original-level and
level-matched audio, tactile and combined stimuli. Retain within-player and
within-equipment replication. Predefine primary estimands, meaningful/equivalence
margins, pilot-based power, hierarchical/repeated-measures analysis and
multiplicity correction. Match measured delivery before interpreting residual
player differences. Neither an attractive synthesized sound nor a favorable
uncontrolled recording establishes a grip mechanism.

## Turnover

- Worktree: `C:/Users/diete/Repositories/UpstreamDrift-impact-acoustics`.
- Branch: `docs/9700-impact-acoustics-program`; commit `SELF`; PR #9706; initial implementation `077ae9df9`.
- Completed: relevant implementation/test inventory, source ownership, state
  contract, study/validation design and linked implementation issues.
- This slice changes documentation only; no engine solver or vendor pin changes.
- Validation: document title-case and SPEC checks pass; no engine tests run
  because this slice is documentation only. PR #9706 is open for normal protected review.
- Preserve #8557/#9153 protected records and frozen ControlTower/WSL recovery
  restrictions in the root handoff. Do not import old checkpoints as new data.
- Next implementation is #9703 after its provider dependency is reviewed and
  qualified. Physical and acoustic effect determination remains open research.
