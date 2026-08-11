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

| Hypothesis                            | Current Evidence                                                                                                                                                               | Current Boundary                                                                                       | Strongest Competing Explanations                                                       | Decisive Next Test                                                                           | Current Status                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| H1: Interaction-Dynamics Contribution | Exact planar matched-state drift/control closure; forward double-pendulum and constrained two-hand killswitches; coupled moving-base/flexible-club branch; one-arm attribution | The spatial common-state tier does not execute a drift/control branch or solve hand contact            | Gravity, passive damping, moving-base work, shaft recoil, stabilization                | Carry the registered split and branched killswitch into forward spatial closed contact       | Supported through the coupled planar tier; untested in spatial forward contact |
| H2: Geometry-Dependent Transfer       | Planar force–velocity projections; grip-separation sweeps; proper-frame invariance; reduced full-body nonplanar moment-arm reversal and coincident-hand control                | Spatial hand loads are prescribed, and the common-state tree is reduced rather than subject-specific   | Force magnitude, event alignment, reference-point error                                | Repeat the registered geometry intervention with independently solved spatial contact        | Supported through reduced full-body common-state inverse dynamics              |
| H3: Passive Late Negative Couple      | Archived WSCG reconstruction; rigid-club and moving-base/flexible-club forward branches; spatial prescribed-load couple and geometric controls                                 | The spatial tier prescribes contact loads, so it cannot determine whether the late couple is passive   | Active wrist torque, moving-base work, shaft recoil, contact compliance, stabilization | Execute a spatial same-state killswitch with independently solved compliant two-hand contact | Supported in coupled planar forward dynamics; inconclusive in spatial tier     |
| H4: Preactivation Under Delay         | Bounded first-order preview signal study                                                                                                                                       | Reference traces are prescribed; there is no forward physiological actuator or EMG evidence            | Objective leakage, unbounded control authority, missing co-contraction                 | Forward delayed-actuator robust-control comparison with held-out ensembles and matched costs | Hypothesis only; untested                                                      |
| H5: Transport Across Fidelity         | Three-link surrogate; coupled planar forward model; frame audit; nonplanar reduced full-body common-state inverse dynamics in two independent formulations                     | The spatial result holds achieved states fixed and prescribes hand loads; forward contact remains open | Convention mismatch, contact-model discrepancy, structural inadequacy                  | Event-aligned forward experiment through independent spatial two-hand contact solvers        | Supported for common-state inverse dynamics; forward spatial contact untested  |

## Model-Discrepancy Register

| Model Tier                         | Executed State                                                                                     | Preserved Observable                                                                       | Material Missing Physics                                                          | Prohibited Inference                            | Completion Gate                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| Planar Double Pendulum             | Forward dynamics complete                                                                          | Wrist force/couple, power, work, drift/control, branched interventions                     | Moving base, two hands, 3-D motion, distributed shaft                             | Universal golfer or coaching prescription       | Retain as analytical ground truth                 |
| One-Arm Three-Link                 | Forward dynamics complete                                                                          | Shoulder/elbow/wrist transfer quantities                                                   | Coupled second arm, floating club, 3-D motion                                     | Two-hand allocation or human effort             | Preserve v1 values through v2 migration           |
| Two-Arm Closed Loop                | Forward constrained state evolution plus exact same-state zero-command branches                    | Contact modes, rank, force/couple/power closure, finite-horizon persistence                | Moving base, compliant club, 3-D motion, contact compliance                       | Human use or higher-tier persistence            | Preserve as the planar forward reference tier     |
| Moving-Base Flexible Club          | Forward coupled finite-mass base, two arms, two contacts, and one flex mode                        | Contact force/couple/power, base and shaft energy, same-state branch, common wrench schema | Base rotation, anatomical body, contact compliance, distributed shaft, 3-D motion | Equipment, physiological, or body-general claim | Preserve as the coupled planar reference tier     |
| Reduced Full-Body 3-D Common State | Nonplanar inverse dynamics executed in MuJoCo and an independent Lagrange--Christoffel formulation | Named wrench, geometry intervention, generalized action, model hash, numerical residual    | Solved two-hand contact, forward divergence, subject anatomy, distributed club    | Passive-contact or human-mechanism validation   | Retain as the spatial common-state reference tier |
| Full-Body 3-D Forward Contact      | Not executed                                                                                       | Preregistered observable definitions and common-state reference only                       | Independent compliant contact rollouts, anatomical parameters, ground pathway     | Spatial passive persistence or optimal control  | Complete forward cross-engine contact gate        |
| Human Experimental                 | Not executed                                                                                       | Preregistered observable definitions only                                                  | Governed synchronized measurements                                                | Skill, physiology, or causal human inference    | Complete Phase 5 held-out validation              |

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
