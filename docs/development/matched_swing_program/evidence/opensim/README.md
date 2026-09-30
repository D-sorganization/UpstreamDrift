# OpenSim Native Dual-Club Dynamics Evidence (MMR-10O #11095)

Qualification contracts and evidence records for OpenSim native
musculoskeletal simulations across dual-club models (Driver and 7-Iron).

## Status: FAIL-CLOSED / NOT YET QUALIFIED

No native OpenSim execution has occurred in this repository. The `opensim`
bindings are not installed on any available host, so `driver_receipt.json` and
`iron_receipt.json` are honest **fail-closed UNAVAILABLE records**: every
numerical evidence field is empty, the missing evidence is enumerated, and the
remedy names the exact command that regenerates real receipts. A previous
version of these files contained placeholder sha256 digests and invented
marker metrics; those fabricated values were removed in this PR and must
never be restored without real execution.

## Qualification Contract

A club receipt may only reach `QUALIFIED` when **all** of the following are
satisfied on a pinned host; any missing item is recorded as missing evidence
and the receipt stays `REJECTED` or `UNAVAILABLE`:

1. **Fresh Simulation Requirement**: Saved controls must drive fresh forward dynamic simulation in OpenSim; `is_fresh_simulation`/`actuation_applied` flags that are absent are treated as unverified, never assumed fresh.
2. **Dynamic Actuation & Musculoskeletal Dynamics**: Playback with kinematic-only (FK) position updates without actuation torques or muscle excitations is rejected.
3. **Nonzero native tests**: a recorded `native_tests_executed > 0`; an unrecorded count blocks qualification.
4. **Physiological Bounds**: Muscle activations must respect physiological excitation/activation limits `[0.0, 1.0]`.
5. **Derivative Consistency**: Joint velocities `v` are independently checked against `dq/dt` central differences; mismatch rejects the candidate.
6. **Energy Accounting**: Mechanical energy (kinetic + gravitational potential) computed from the recorded rollout; non-finite values reject.
7. **Aligned Marker Metrics**: only the `whole_rms_m` metric actually computed
   from the recorded `markers_m`/`target_m` observations is emitted. Previously
   present "phase-segmented", "clubhead", and "pelvis yaw" values were not
   derived from data and are removed.
8. **Declared Engine Limitations**: Engine limitations are explicitly disclosed:
   - `hill_type_activation_dynamics`: Muscle excitation to activation first-order lag.
   - `force_velocity_length_multipliers`: Active and passive force-length-velocity curves restrict instantaneous force generation.
   - `tendon_elasticity_equilibrium`: Tendon compliance and equilibrium dynamics.
   - `coordinate_limit_forces`: Penalty contact or coordinate limit forces active near physiological extrema.
   - `ground_contact_requires_external_wrenches`: Foot-ground reaction forces require calibrated multi-component contact meshes or external force profiles.

## Artifacts

- `driver_receipt.json`: fail-closed UNAVAILABLE record (driver).
- `iron_receipt.json`: fail-closed UNAVAILABLE record (7-iron).
- `../nightly/opensim_receipt.json`: automated nightly CI lane receipt
  reporting native runner test health (fail-closed: 0 executed tests on
  unavailable engines yields `status: fail`).

## Regenerating Real Club Receipts

Install the OpenSim bindings on a pinned host, execute the native replay for
each club, then produce receipts via `assess_opensim_qualification(...)` with
the recorded `native_state`/`time_s`/`marker` data, nonzero executed test
count, and the committed model/candidate/capture hashes. Software-side
qualification logic is complete; native qualification remains explicitly
**blocked** until the engine is available.
