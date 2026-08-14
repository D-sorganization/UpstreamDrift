# Data Dictionary

## Common Rules

All physical values use SI units unless a field name explicitly ends in
`_deg`. Time arrays are seconds and monotonically increasing. Forces act in the
direction declared by the evidence interface; couples and moments use the
right-hand rule. Every model result is conditional on its named tier and may
not be promoted to a human or physiological result.

| Artifact                                     | Primary Contents                                                                                                                   | Interpretation Boundary                                 |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `results_summary.json`                       | Double-pendulum delivery, energy, and timing summaries                                                                             | Planar fixed-hub model                                  |
| `representative_traces.npz`                  | State, command, force, power, work, and energy histories                                                                           | Selected deterministic programs                         |
| `interaction_force_summary.json`             | Exact drift/control closure, geometry, work, and killswitch metrics                                                                | Two-link analytical tier                                |
| `counterfactual_ensemble.json`               | Cut-time, horizon, timestep, gravity, and damping interventions                                                                    | Branched model counterfactuals                          |
| `two_hand_wscg_analysis.{json,npz}`          | Reconstructed hand forces, wrench, couple, power, and geometry controls                                                            | Archived pointwise BASE/ZTCF paths                      |
| `shaft_contribution_study.json`              | Rigid/flexible, termwise, ablation, robustness, and convergence summaries                                                          | One lumped flex surrogate                               |
| `forward_two_arm_study.{json,npz}`           | Forward constrained states, grip reactions, force modes, branches, closure                                                         | Fixed-shoulder planar model                             |
| `moving_base_flexible_study.{json,npz}`      | Coupled base/flex states, reactions, energy, branches, and convergence                                                             | Translating-base planar model                           |
| `shaft_beam_reference.{json,npz}`            | Modal identification, mesh convergence, reduced/reference response, and energy                                                     | Synthetic isolated beam comparison                      |
| `moving_base_modal_shaft_study.{json,npz}`   | Moving-base, two-hand, transported modal-shaft states, branches, controls, and closure                                             | Synthetic planar forward model                          |
| `torque_allocation_preload_study.{json,npz}` | Matched-task arm/wrist allocations and declared dead-zone transmission traces                                                      | Generalized controls and phenomenological channel       |
| `spatial_full_body_study.{json,npz}`         | Nonplanar common states, prescribed wrenches, two-formulation inverse dynamics                                                     | No solved forward spatial contact                       |
| `spatial_forward_contact_study.{json,npz}`   | Native MuJoCo/Pinocchio forward states, compliant wrench, killswitch, energy, ground pathway                                       | Reduced hand carriages; no anatomy or human data        |
| `uncertainty_control_study.{json,npz}`       | Parameter ensembles, PRCC, identifiability, programs, Pareto metrics, rollouts                                                     | Engineering envelopes and actuator proxies              |
| `experimental_protocol_v1.json`              | Frozen modalities, outcomes, split, residuals, and inference rules                                                                 | Protocol, not participant evidence                      |
| `experimental_protocol_readiness.json`       | Synthetic intake qualification and untested claim status                                                                           | No human observations                                   |
| `advanced_biological_bridge.{json,npz}`      | Frame/power invariance, muscle redundancy, activation history, and engine-role records                                             | Reduced synthetic biology; no subject validation        |
| `transmission_robustness_study.{json,npz}`   | Paired clock/state-trigger perturbations, robust objectives, pathway metrics, local outcome Jacobian, and adversarial gap register | Conditional planar robustness; no human stability       |
| `momentum_transfer_question_registry.json`   | Seven critical questions with current status, estimands, controls, falsifiers, and governing issues                                | Research-program contract; not experimental evidence    |
| `momentum_transfer_experiment_registry.json` | Registered model tiers, interventions, controls, outcomes, uncertainty, falsifiers, required data, and execution status            | Prospective experiment contract; not a favorable result |
| `momentum_transfer_human_registration.json`  | Participant-held-out Q1--Q7 estimands, nulls, falsifiers, sensitivities, negative controls, adverse margins, and data requirements | Frozen protocol; no governed human outcome acquired     |
| `typed_slack_study.json`                     | One-class-at-a-time engagement, transmission, stored energy, work, and closure for five slack definitions                          | Synthetic constitutive screen; no performance result    |
| `timing_failure_mode_study.json`             | Twenty-seven proximal-acceleration, braking, and release timing cases with two casting-event definitions and delivery status       | Unmatched-work fixed-hub screen; no coaching conclusion |

## Recurring Field Families

- `schema_version`, `study_id`, and `model_tier` identify the contract and
  evidential scope.
- `source_sha256`, `model_hash`, and `array_artifact` connect summaries to
  executable sources and dense arrays.
- `time_s`, `q`, `qd`, and `qdd` are time, generalized position, velocity, and
  acceleration.
- `force_*_n`, `wrench_*`, and `couple_*_nm` are forces, six-dimensional
  wrenches, and scalar couples at named references.
- `power_*_w`, `work_*_j`, and `energy_*_j` are instantaneous power, integrated
  work, and stored/mechanical energy.
- `constraint_*`, `kkt_*`, `power_residual_*`, and `work_energy_residual_*`
  quantify numerical closure rather than physical effects.
- `training`, `held_out`, `q05`, `median`, `q95`, `q10`, and `q90` identify
  declared ensemble partitions or quantiles; held-out model cases are not human
  validation.

NPZ member names and shapes are inventoried programmatically by the release
manifest and preserved exactly by the checksum file. Reviewers should load NPZ
files with `allow_pickle=False`.
