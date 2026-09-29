# Drake Native Dual-Club Dynamics Evidence (MMR-10D #11094)

Verifiable dynamic qualification evidence for Drake native multibody simulations across dual-club models (Driver and 7-Iron).

## Qualification Scope and Criteria

In accordance with matched motion replay standards:

1. **Fresh Simulation Requirement**: Saved controls must drive fresh forward dynamic simulation in Drake; state trajectories copied from kinematic fits or reference motion are rejected.
2. **Dynamic Actuation**: Playback with kinematic-only (FK) position updates without actuation torques is rejected.
3. **Derivative Consistency**: Joint velocities `v` must be independently consistent with position derivatives `dq/dt`.
4. **Energy Conservation**: Total mechanical energy (kinetic + gravitational potential) is tracked and accounted for.
5. **Aligned Marker Metrics**: Whole-body, phase-segmented (early, terminal), clubhead RMS, and pelvis yaw orientation error percentages are captured.
6. **Declared Engine Limitations**: Engine limitations are explicitly disclosed:
   - `upper_body_27dof_float_pathway`: Pelvis 6-DoF translation and orientation are free floating coordinates.
   - `rigid_weld_closure`: Dual-grip closed kinematic chain is enforced via 6D rigid weld constraint.
   - `continuous_polynomial_actuation`: 6th-order continuous Bernstein/power torque polynomials per joint.
   - `ground_contact_requires_full_body`: Upper-body model slice does not resolve ground reaction contact wrenches.

## Artifacts

- `driver_receipt.json`: Qualified receipt for driver model (`model_hash: a376e6889b2b9d3b05a0b2f110f386c83f17f4ec0c94c9f4b9289a887ba4643e`).
- `iron_receipt.json`: Qualified receipt for 7-iron model (`model_hash: b75a9c056633c41cc400fe1a0b64253669028abb8edb27d6fe26735b10659c5a`).
- `../nightly/drake_receipt.json`: Automated nightly CI lane receipt reporting native runner test health.
