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

| Hypothesis                            | Current Evidence                                                                                                   | Current Boundary                                                                                  | Strongest Competing Explanations                                            | Decisive Next Test                                                                                       | Current Status                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| H1: Interaction-Dynamics Contribution | Exact planar matched-state drift/control closure; forward double-pendulum killswitch ensemble; one-arm attribution | Two-hand evidence is not a forward trajectory                                                     | Gravity, passive damping, prescribed-base work, shaft recoil, stabilization | Forward two-hand same-state split and branched killswitch with a complete term inventory                 | Supported in lower tiers; untested in forward two-hand and 3-D tiers |
| H2: Geometry-Dependent Transfer       | Planar force–velocity projections, grip-separation sweeps, reference transport, and proper-frame invariance        | The 3-D result rotates planar evidence rather than creating out-of-plane dynamics                 | Force magnitude, event alignment, reference-point error                     | Registered moment-arm and projection interventions in forward two-hand and spatial models                | Supported in planar interventions; untested in full 3-D dynamics     |
| H3: Passive Late Negative Couple      | Archived WSCG pointwise ZTCF reconstruction and two-hand force-couple geometry                                     | The archived trajectory and current local sweep do not establish forward persistence              | Active wrist torque, internal-force allocation, shaft recoil, stabilization | Nonsingular forward two-hand killswitch across contact allocation, step size, and stabilization variants | Plausible and pointwise-supported; forward claim untested            |
| H4: Preactivation Under Delay         | Bounded first-order preview signal study                                                                           | Reference traces are prescribed; there is no forward physiological actuator or EMG evidence       | Objective leakage, unbounded control authority, missing co-contraction      | Forward delayed-actuator robust-control comparison with held-out ensembles and matched costs             | Hypothesis only; untested                                            |
| H5: Transport Across Fidelity         | Three-link surrogate, prescribed mobile hub, closed-loop rank audit, and proper-frame transformation               | No dynamic moving base, full-body 3-D rollout, or independent-engine common-observable comparison | Convention mismatch, contact-model discrepancy, structural inadequacy       | Event-aligned common experiment through two independent full-body engines                                | Untested beyond the executed surrogate audits                        |

## Model-Discrepancy Register

| Model Tier                | Executed State                                          | Preserved Observable                                                   | Material Missing Physics                                       | Prohibited Inference                         | Completion Gate                              |
| ------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| Planar Double Pendulum    | Forward dynamics complete                               | Wrist force/couple, power, work, drift/control, branched interventions | Moving base, two hands, 3-D motion, distributed shaft          | Universal golfer or coaching prescription    | Retain as analytical ground truth            |
| One-Arm Three-Link        | Forward dynamics complete                               | Shoulder/elbow/wrist transfer quantities                               | Coupled second arm, floating club, 3-D motion                  | Two-hand allocation or human effort          | Preserve v1 values through v2 migration      |
| Two-Arm Closed Loop       | Prescribed local sweep and pointwise constrained solves | Contact modes, rank, force/couple/power closure                        | Forward state evolution and event-level persistence            | Passive forward negative-couple claim        | Complete Phase 1 forward constrained rollout |
| Moving-Base Flexible Club | Prescribed hub and reduced shaft surrogate              | Declared base/shaft terms and common wrench schema                     | Coupled torso dynamics and calibrated beam behavior            | Equipment or body-general claim              | Complete Phase 2 energy-closed dynamic model |
| Full-Body 3-D             | Frame transformation audit only                         | Spatial frame/reference invariants                                     | Out-of-plane dynamics, ground pathway, cross-engine simulation | 3-D mechanism validation                     | Complete Phase 3 in two independent engines  |
| Human Experimental        | Not executed                                            | Preregistered observable definitions only                              | Governed synchronized measurements                             | Skill, physiology, or causal human inference | Complete Phase 5 held-out validation         |

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
