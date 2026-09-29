# OpenSim Native Dual-Club Dynamics Evidence (MMR-10O #11095)

Verifiable dynamic qualification evidence for OpenSim native musculoskeletal simulations across dual-club models (Driver and 7-Iron).

## Qualification Scope and Criteria

In accordance with matched motion replay standards:

1. **Fresh Simulation Requirement**: Saved controls must drive fresh forward dynamic simulation in OpenSim; state trajectories copied from kinematic fits or reference motion are rejected.
2. **Dynamic Actuation & Musculoskeletal Dynamics**: Playback with kinematic-only (FK) position updates without actuation torques or muscle excitations is rejected.
3. **Physiological Bounds**: Muscle activations must respect physiological excitation/activation limits `[0.0, 1.0]`.
4. **Derivative Consistency**: Joint velocities `v` must be independently consistent with position derivatives `dq/dt`.
5. **Energy Conservation**: Mechanical energy (kinetic + gravitational potential) and work balance are accounted for.
6. **Aligned Marker Metrics**: Whole-body, phase-segmented (early, terminal), clubhead RMS, and pelvis yaw orientation error percentages are captured.
7. **Declared Engine Limitations**: Engine limitations are explicitly disclosed:
   - `hill_type_activation_dynamics`: Muscle excitation to activation first-order lag.
   - `force_velocity_length_multipliers`: Active and passive force-length-velocity curves restrict instantaneous force generation.
   - `tendon_elasticity_equilibrium`: Tendon compliance and equilibrium dynamics.
   - `coordinate_limit_forces`: Penalty contact or coordinate limit forces active near physiological extrema.
   - `ground_contact_requires_external_wrenches`: Foot-ground reaction forces require calibrated multi-component contact meshes or external force profiles.

## Artifacts

- `driver_receipt.json`: Qualified receipt for driver model (`model_hash: 6fa980a3ccda49b1051c55fb025da596c9a7b66e1687a4b3254df19d49536a21`).
- `iron_receipt.json`: Qualified receipt for 7-iron model (`model_hash: 3e7be1486821578efd6474916f21a25e794c353bf698b61b5cac28d725778688`).
- `../nightly/opensim_receipt.json`: Automated nightly CI lane receipt reporting native runner test health.
