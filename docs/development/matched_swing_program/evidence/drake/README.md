# Drake Native Dual-Club Dynamics Evidence (MMR-10D #11094)

Qualification contracts and evidence records for Drake native multibody
simulations across dual-club models (Driver and 7-Iron).

## Status: FAIL-CLOSED / NOT YET QUALIFIED

No native Drake execution has occurred in this repository. `pydrake` is not
installed on any available host, so `driver_receipt.json` and
`iron_receipt.json` are honest **fail-closed UNAVAILABLE records**: every
numerical evidence field is empty, the missing evidence is enumerated, and the
remedy names the exact command that regenerates real receipts. A previous
version of these files contained placeholder sha256 digests (`c0ffee…`,
`deadbeef…`, `cafebabe…`) and invented marker metrics; those fabricated values
were removed in this PR and must never be restored without real execution.

## Qualification Contract

A club receipt may only reach `QUALIFIED` when **all** of the following are
satisfied on a pinned host; any missing item is recorded as missing evidence
and the receipt stays `REJECTED` or `UNAVAILABLE`:

1. **Native runtime**: `pydrake` importable on the pinned host (`runtime_available`).
2. **Fresh Simulation**: saved controls drive a fresh forward dynamic
   simulation in Drake; `is_fresh_simulation`/`actuation_applied` flags that
   are absent are treated as unverified, never assumed fresh.
3. **Dynamic Actuation**: FK-only playback without actuation torques is rejected.
4. **Nonzero native tests**: a recorded `native_tests_executed > 0`; an
   unrecorded count blocks qualification.
5. **Derivative Consistency**: joint velocities `v` are independently checked
   against `dq/dt` central differences; mismatch rejects the candidate.
6. **Energy Accounting**: kinetic/potential energy summary computed from the
   recorded rollout; non-finite values reject.
7. **Aligned Marker Metrics**: only the `whole_rms_m` metric actually computed
   from the recorded `markers_m`/`target_m` observations is emitted. Previously
   present "phase-segmented", "clubhead", and "pelvis yaw" values were not
   derived from data and are removed.

## Declared Engine Limitations

- `upper_body_27dof_float_pathway`: Pelvis 6-DoF translation and orientation are free floating coordinates.
- `rigid_weld_closure`: Dual-grip closed kinematic chain is enforced via 6D rigid weld constraint.
- `continuous_polynomial_actuation`: 6th-order continuous Bernstein/power torque polynomials per joint.
- `ground_contact_requires_full_body`: Upper-body model slice does not resolve ground reaction contact wrenches.

## Artifacts

- `driver_receipt.json`: fail-closed UNAVAILABLE record (driver).
- `iron_receipt.json`: fail-closed UNAVAILABLE record (7-iron).
- `../nightly/drake_receipt.json`: automated nightly lane receipt; recorded
  locally (`status: fail`, 0 executed native tests, engine unavailable) —
  honest, and regenerated on a qualified runner by:

```bash
bash scripts/ci/run_native_engine_lane.sh \
  --engine drake \
  --out docs/development/matched_swing_program/evidence/nightly
```

## Regenerating Real Club Receipts

Install `pydrake` on a pinned host, execute the native replay for each club,
then produce receipts via `assess_drake_qualification(...)` with the recorded
`native_state`/`time_s`/`marker` data, nonzero executed test count, and the
committed model/candidate/capture hashes. Software-side qualification logic is
complete; native qualification remains explicitly **blocked** until the engine
is available.