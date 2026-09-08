# Reviewer Workbench

## Adversarial Transmission and Robustness Package

The second adversarial pass is indexed by
[`ADVERSARIAL_TRANSMISSION_REVIEW.md`](ADVERSARIAL_TRANSMISSION_REVIEW.md).
Its executable evidence is
[`transmission_robustness_study.json`](data/transmission_robustness_study.json)
with dense arrays in
[`transmission_robustness_study.npz`](data/transmission_robustness_study.npz).
Reviewers should inspect every Pareto member, the paired clock/state outcomes,
the local outcome Jacobian, the 12-item gap register, and both numerical
closures. The projected whole-system work--energy ledger has a declared 4 ms
residual; the algebraic two-contact power identity is the machine-precision
check.

## Start With Claim Status

Start with the complete [normalized claim JSON](data/claim_adjudication_summary.json)
or [accessible claim CSV](data/claim_adjudication_summary.csv), then read the
[claim–evidence–falsifier matrix](MODEL_COMPLETION_FALSIFICATION_MATRIX.md)
before interpreting figures. The generated paper table summarizes outcomes,
evidence tiers, source independence, model tiers, unresolved replication, and
claim-family concentration. A supported model-conditional estimand is not an
independent replication or human finding. Every panel is model-tier specific. The
[release manifest](release_manifest.json) supplies exact hashes and canonical
commands; the [data dictionary](DATA_DICTIONARY.md) defines recurring fields.
The [adversarial review adjudication](ADVERSARIAL_REVIEW_ADJUDICATION.md)
records every external criticism, its verification status, and the resulting
code, evidence, or claim-boundary response.

## Analytical Mechanism

- Machine-readable summary:
  [`interaction_force_summary.json`](data/interaction_force_summary.json)

> Figure references removed 2026-09-03 (issue #8851). Three figures listed here
> -- `fig_force_vectors_geometry.svg`, `fig_pointwise_vs_forward.svg`, and
> `fig_power_transfer.svg` -- were never present in `figures/` and are not
> recoverable from the available git history. Several existing figures cover
> adjacent ground (for example
> [`fig_interaction_force_components.svg`](figures/fig_interaction_force_components.svg),
> [`fig_interaction_geometry_coefficients.svg`](figures/fig_interaction_geometry_coefficients.svg),
> [`fig_ztcf_drift_control.svg`](figures/fig_ztcf_drift_control.svg), and
> [`fig_interaction_force_power.svg`](figures/fig_interaction_force_power.svg)),
> but the original caption-to-panel mapping could not be verified, so the dead
> references were removed rather than re-pointed at a guess. See
> [`figures/`](figures/) for the full generated set.

## Two-Hand and Coupled Planar Models

- Evidence:
  [`forward_two_arm_study.json`](data/forward_two_arm_study.json) and
  [`moving_base_flexible_study.json`](data/moving_base_flexible_study.json)

> Figure references removed 2026-09-03 (issue #8851). Three figures listed here
> -- `fig_two_hand_wrench_reconstruction.svg`,
> `fig_forward_two_arm_killswitch.svg`, and
> `fig_moving_base_flexible_energy.svg` -- were never present in `figures/` and
> are not recoverable from the available git history. Nearest surviving panels
> are [`fig_wscg_registered_hand_forces.svg`](figures/fig_wscg_registered_hand_forces.svg),
> [`fig_forward_two_hand_couple_killswitch.svg`](figures/fig_forward_two_hand_couple_killswitch.svg),
> and [`fig_coupled_base_flex_transfer.svg`](figures/fig_coupled_base_flex_transfer.svg);
> the mapping is unverified, so the dead references were removed rather than
> silently re-pointed.

## Distributed-Shaft Structural Reference

- Synthetic identification and convergence:
  [`fig_shaft_beam_identification.svg`](figures/fig_shaft_beam_identification.svg)
- Reduced and six-mode responses:
  [`fig_shaft_beam_response.svg`](figures/fig_shaft_beam_response.svg)
- Input, damping, and energy closure:
  [`fig_shaft_beam_energy.svg`](figures/fig_shaft_beam_energy.svg)
- Evidence:
  [`shaft_beam_reference.json`](data/shaft_beam_reference.json)

This tier is a synthetic structural comparison. It is not an equipment
calibration. The later forward modal-shaft tier performs the coupling test at
the declared planar mechanism level; measured equipment calibration remains
open.

## Spatial Common-State Tier

- Model and force geometry:
  [`fig_spatial_full_body_force_geometry.svg`](figures/fig_spatial_full_body_force_geometry.svg)
- Two-formulation comparison:
  [`fig_spatial_cross_formulation_inverse_dynamics.svg`](figures/fig_spatial_cross_formulation_inverse_dynamics.svg)
- Falsification status:
  [`fig_spatial_full_body_falsification.svg`](figures/fig_spatial_full_body_falsification.svg)
- Evidence:
  [`spatial_full_body_study.json`](data/spatial_full_body_study.json)

## Reduced Spatial Forward-Contact Tier

- Achieved geometry and projected compliant-force vectors:
  [`fig_spatial_forward_contact_geometry.svg`](figures/fig_spatial_forward_contact_geometry.svg)
- Inertia-and-bias transport trajectory and wrench comparison:
  [`fig_spatial_forward_cross_engine.svg`](figures/fig_spatial_forward_cross_engine.svg)
- Exact same-state driver killswitch and pathway observables:
  [`fig_spatial_forward_killswitch.svg`](figures/fig_spatial_forward_killswitch.svg)
- Energy closure, geometry controls, and claim boundary:
  [`fig_spatial_forward_energy_controls.svg`](figures/fig_spatial_forward_energy_controls.svg)
- Evidence:
  [`spatial_forward_contact_study.json`](data/spatial_forward_contact_study.json)

This tier uses two finite-mass translational hand carriages and one rigid club.
It is shared-contact, shared-integrator inertia-and-bias transport evidence,
not independent contact-solver evidence or anatomical, tissue, physiological,
equipment, or human validation. The separate native-constraint discrepancy
control is
[`fig_articulated_native_constraint_discrepancy.pdf`](figures/fig_articulated_native_constraint_discrepancy.pdf).

## Uncertainty, Identifiability, and Control

- Global intervals and PRCC:
  [`fig_uncertainty_intervals_and_prcc.svg`](figures/fig_uncertainty_intervals_and_prcc.svg)
- Identifiability audit:
  [`fig_identifiability_audit.svg`](figures/fig_identifiability_audit.svg)
- Training/held-out Pareto comparison:
  [`fig_control_pareto_train_holdout.svg`](figures/fig_control_pareto_train_holdout.svg)
- Strategy tradeoffs:
  [`fig_control_strategy_tradeoffs.svg`](figures/fig_control_strategy_tradeoffs.svg)
- Evidence:
  [`uncertainty_control_study.json`](data/uncertainty_control_study.json)

## Reference Frames, Biological Redundancy, and Engine Roles

- Phase-resolved model geometry:
  [`fig_advanced_model_motion_plate.svg`](figures/fig_advanced_model_motion_plate.svg)
- Wrench/twist transport and invariant power:
  [`fig_frame_power_invariance.svg`](figures/fig_frame_power_invariance.svg)
- Matched-moment muscle redundancy:
  [`fig_biological_redundancy.svg`](figures/fig_biological_redundancy.svg)
- Activation and series-force history:
  [`fig_biological_role_reversal.svg`](figures/fig_biological_role_reversal.svg)
- Cross-engine question ladder:
  [`fig_cross_engine_question_ladder.svg`](figures/fig_cross_engine_question_ladder.svg)
- Evidence:
  [`advanced_biological_bridge.json`](data/advanced_biological_bridge.json) and
  [`advanced_biological_bridge.npz`](data/advanced_biological_bridge.npz)

The frame and virtual-work closures are executed numerical invariance checks.
The reduced muscle study demonstrates non-uniqueness and preparation-history
effects under declared synthetic parameters. The MuJoCo, Pinocchio, Drake,
OpenSim, and MyoSuite pose-adapter round trips are executed coordinate-mapping
checks only; they are not five-engine forward-dynamics or human-validation
results. Use the [terminology and conventions contract](TERMINOLOGY_AND_CONVENTIONS.md)
when translating quantities between tiers.

## Experimental Readiness

The [experimental protocol](EXPERIMENTAL_FALSIFICATION_PROTOCOL.md) and
[`experimental_protocol_readiness.json`](data/experimental_protocol_readiness.json)
are readiness artifacts only. They contain no human observations. The review
surface must not render synthetic readiness as empirical support.

## Arm--Wrist Allocation and Transmission Preload

- Matched-task allocation and moment closure:
  [`fig_torque_allocation_moment_closure.svg`](figures/fig_torque_allocation_moment_closure.svg)
- Geometry-dependent internal-demand surface:
  [`fig_torque_allocation_geometry_surface.svg`](figures/fig_torque_allocation_geometry_surface.svg)
- Persistent-direction and role-reversal traces:
  [`fig_torque_role_reversal_transmission.svg`](figures/fig_torque_role_reversal_transmission.svg)
- Evidence:
  [`torque_allocation_preload_study.json`](data/torque_allocation_preload_study.json)

The allocation poles are generalized actuator subspaces. The transmission
channel is a declared phenomenological dead-zone model. Neither identifies
scapular action, muscular inactivity, biological slack, or a preferred human
technique.
