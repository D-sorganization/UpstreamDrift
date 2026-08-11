# Model Completion Falsification Matrix

## Interpretation

This register connects every principal mechanism claim to the evidence that
currently supports it, the strongest competing explanations, and the next test
that can contradict it. `Supported` is reserved for an executed prediction at
the declared model tier. A finding does not automatically inherit support when
it is transported to a higher tier.

The machine-readable prediction definitions are stored in
`data/model_completion_predictions.json`. Status changes require a reviewed
evidence bundle generated after its tolerance and intervention were registered.

## Claim–Evidence–Falsifier Register

| Hypothesis                            | Current Evidence                                                                                                                                                               | Current Boundary                                                                                     | Strongest Competing Explanations                                                       | Decisive Next Test                                                                           | Current Status                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| H1: Interaction-Dynamics Contribution | Exact planar matched-state drift/control closure; forward double-pendulum and constrained two-hand killswitches; coupled moving-base/flexible-club branch; one-arm attribution | The coupled branch is planar, uses one lumped flex mode, and does not identify biological effort     | Gravity, passive damping, moving-base work, shaft recoil, stabilization                | Carry the registered split and branched killswitch into the spatial cross-engine tier        | Supported through the coupled planar tier; untested in spatial tiers         |
| H2: Geometry-Dependent Transfer       | Planar force–velocity projections, grip-separation sweeps, reference transport, and proper-frame invariance                                                                    | The 3-D result rotates planar evidence rather than creating out-of-plane dynamics                    | Force magnitude, event alignment, reference-point error                                | Registered moment-arm and projection interventions in forward two-hand and spatial models    | Supported in planar interventions; untested in full 3-D dynamics             |
| H3: Passive Late Negative Couple      | Archived WSCG reconstruction; rigid-club and moving-base/flexible-club forward branches with exact state inheritance; geometric negative controls; convergence                 | The coupled result remains planar with ideal contacts, linear base/shaft compliance, and one command | Active wrist torque, moving-base work, shaft recoil, contact compliance, stabilization | Repeat in spatial, parameter-uncertainty, contact-compliance, and measured-data tiers        | Supported through the coupled planar tier; untested in spatial/human tiers   |
| H4: Preactivation Under Delay         | Bounded first-order preview signal study                                                                                                                                       | Reference traces are prescribed; there is no forward physiological actuator or EMG evidence          | Objective leakage, unbounded control authority, missing co-contraction                 | Forward delayed-actuator robust-control comparison with held-out ensembles and matched costs | Hypothesis only; untested                                                    |
| H5: Transport Across Fidelity         | Three-link surrogate, prescribed-hub audit, forward coupled moving-base/flexible-club model, closed-loop rank audit, and proper-frame transformation                           | No full-body 3-D rollout or independent-engine common-observable comparison                          | Convention mismatch, contact-model discrepancy, structural inadequacy                  | Event-aligned common experiment through two independent full-body engines                    | Supported through the coupled planar tier; cross-engine spatial test pending |

## Model-Discrepancy Register

| Model Tier                | Executed State                                                                  | Preserved Observable                                                                       | Material Missing Physics                                                          | Prohibited Inference                            | Completion Gate                               |
| ------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------- |
| Planar Double Pendulum    | Forward dynamics complete                                                       | Wrist force/couple, power, work, drift/control, branched interventions                     | Moving base, two hands, 3-D motion, distributed shaft                             | Universal golfer or coaching prescription       | Retain as analytical ground truth             |
| One-Arm Three-Link        | Forward dynamics complete                                                       | Shoulder/elbow/wrist transfer quantities                                                   | Coupled second arm, floating club, 3-D motion                                     | Two-hand allocation or human effort             | Preserve v1 values through v2 migration       |
| Two-Arm Closed Loop       | Forward constrained state evolution plus exact same-state zero-command branches | Contact modes, rank, force/couple/power closure, finite-horizon persistence                | Moving base, compliant club, 3-D motion, contact compliance                       | Human use or higher-tier persistence            | Preserve as the planar forward reference tier |
| Moving-Base Flexible Club | Forward coupled finite-mass base, two arms, two contacts, and one flex mode     | Contact force/couple/power, base and shaft energy, same-state branch, common wrench schema | Base rotation, anatomical body, contact compliance, distributed shaft, 3-D motion | Equipment, physiological, or body-general claim | Preserve as the coupled planar reference tier |
| Full-Body 3-D             | Frame transformation audit only                                                 | Spatial frame/reference invariants                                                         | Out-of-plane dynamics, ground pathway, cross-engine simulation                    | 3-D mechanism validation                        | Complete Phase 3 in two independent engines   |
| Human Experimental        | Not executed                                                                    | Preregistered observable definitions only                                                  | Governed synchronized measurements                                                | Skill, physiology, or causal human inference    | Complete Phase 5 held-out validation          |

## Status Rules

- **Supported:** the registered expected result passes its preoutcome decision
  rule and applicable negative controls.
- **Contradicted:** the registered falsifier occurs outside numerical and
  declared uncertainty bounds.
- **Inconclusive:** the observable is unavailable, non-identifiable, singular,
  or too uncertain to distinguish the alternatives.
- **Untested:** the required experiment or model tier has not been executed.

Publication language must preserve these statuses. A visually similar curve,
successful trajectory fit, or higher force magnitude is not a substitute for
the registered estimand and intervention.
