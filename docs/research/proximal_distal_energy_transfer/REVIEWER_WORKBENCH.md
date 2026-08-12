# Reviewer Workbench

## Start With Claim Status

Read the [claim–evidence–falsifier matrix](MODEL_COMPLETION_FALSIFICATION_MATRIX.md)
before interpreting figures. Every panel is model-tier specific. The
[release manifest](release_manifest.json) supplies exact hashes and canonical
commands; the [data dictionary](DATA_DICTIONARY.md) defines recurring fields.

## Analytical Mechanism

- Interaction-force vectors and geometry:
  [`fig_force_vectors_geometry.svg`](figures/fig_force_vectors_geometry.svg)
- Pointwise drift/control and forward killswitch comparisons:
  [`fig_pointwise_vs_forward.svg`](figures/fig_pointwise_vs_forward.svg)
- Timing, power, and work:
  [`fig_power_transfer.svg`](figures/fig_power_transfer.svg)
- Machine-readable summary:
  [`interaction_force_summary.json`](data/interaction_force_summary.json)

## Two-Hand and Coupled Planar Models

- Archived WSCG wrench reconstruction:
  [`fig_two_hand_wrench_reconstruction.svg`](figures/fig_two_hand_wrench_reconstruction.svg)
- Forward two-arm branch:
  [`fig_forward_two_arm_killswitch.svg`](figures/fig_forward_two_arm_killswitch.svg)
- Moving-base/flexible-club energy and couple:
  [`fig_moving_base_flexible_energy.svg`](figures/fig_moving_base_flexible_energy.svg)
- Evidence:
  [`forward_two_arm_study.json`](data/forward_two_arm_study.json) and
  [`moving_base_flexible_study.json`](data/moving_base_flexible_study.json)

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
calibration and is not yet coupled into the constrained two-hand rollout.

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

- Achieved geometry and engine-solved force vectors:
  [`fig_spatial_forward_contact_geometry.svg`](figures/fig_spatial_forward_contact_geometry.svg)
- Native forward trajectory and wrench comparison:
  [`fig_spatial_forward_cross_engine.svg`](figures/fig_spatial_forward_cross_engine.svg)
- Exact same-state driver killswitch and pathway observables:
  [`fig_spatial_forward_killswitch.svg`](figures/fig_spatial_forward_killswitch.svg)
- Energy closure, geometry controls, and claim boundary:
  [`fig_spatial_forward_energy_controls.svg`](figures/fig_spatial_forward_energy_controls.svg)
- Evidence:
  [`spatial_forward_contact_study.json`](data/spatial_forward_contact_study.json)

This tier uses two finite-mass translational hand carriages and one rigid club.
It is forward cross-engine mechanism evidence, not anatomical, tissue,
physiological, equipment, or human validation.

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

## Experimental Readiness

The [experimental protocol](EXPERIMENTAL_FALSIFICATION_PROTOCOL.md) and
[`experimental_protocol_readiness.json`](data/experimental_protocol_readiness.json)
are readiness artifacts only. They contain no human observations. The review
surface must not render synthetic readiness as empirical support.
