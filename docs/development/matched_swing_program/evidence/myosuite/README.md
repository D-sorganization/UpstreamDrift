# MyoSuite Native Dual-Club Dynamics Evidence Package (MMR-10M, #11096)

Qualification contracts and evidence records for the MyoSuite native dynamic
simulation lane across Driver and 7-Iron models.

## Status: FAIL-CLOSED / NOT YET QUALIFIED

No native MyoSuite execution has occurred in this repository. The
`myosuite`/MuJoCo stack is not installed on any available host, so
`driver_receipt.json` and `iron_receipt.json` are honest **fail-closed
UNAVAILABLE records**: every numerical evidence field is empty, the missing
evidence is enumerated, and the remedy names the exact command that
regenerates real receipts. A previous version of these files contained
placeholder sha256 digests and invented marker metrics; those fabricated
values were removed in this PR and must never be restored without real
execution. The model hashes below are recorded *targets* for regeneration,
not evidence.

## Engine Scope and Model Hashes

- **Engine:** MyoSuite (MuJoCo-backed musculoskeletal model)
- **Host Platform:** Linux (ControlTower) / Windows (Development)
- **Actuation:** Musculoskeletal forward excitation-activation dynamics with Hill-type force-length-velocity multipliers
- **Driver Model Hash:** `8e941be1c5a7d4b2a32a6fb7949ff527200eda069fee00a05664a81850fbc39b`
- **7-Iron Model Hash:** `ddd93a125771f5712c1f8e62d676510b6df138290a58b20c2555817906f31086`

## Qualification Criteria and Checks

1. **Fresh Simulation:** Saved controls drive an independent forward numerical integration rollout. Copied or cloned state trajectories from candidate references are strictly rejected.
2. **Forward Dynamics:** Rejects kinematics-only playback without dynamic actuation or muscle excitations.
3. **Nonzero Native Execution:** Live qualification requires execution on a pinned host running qualified tests.
4. **Independent Derivative and Energy Consistency:** Central-difference verification of joint velocities against state derivatives; energy summary computed from the recorded rollout and rejected when non-finite.
5. **Physiological Muscle Limits:** Muscle activations bounded in $[0.0, 1.0]$.
6. **Free-Joint Unit Quaternion:** Root orientation quaternion $|\|q_{\text{root}}\| - 1.0| \le 1\times 10^{-4}$.
7. **Aligned Common-Marker Metrics:** 3D marker RMSE evaluated against measured optical markers.

## Declared Biomechanical Limitations

- Musculoskeletal excitation-activation dynamics governed by first-order filter with activation/deactivation time constants.
- Hill-type muscle tendon unit (MTU) force-length-velocity multipliers with physiological bounds.
- Free-joint quaternion orientation normalization ($w^2 + x^2 + y^2 + z^2 = 1$) and spatial velocity state layout.
- Dual-grip weld constraint kinematics coupling hands to club shaft.
- Four-foot Hunt-Crossley contact spheres with normal compliance and friction cone limits.
- Absence of native joint-torque inverse dynamics (forward muscle excitation driven only).

## Artifacts

- `driver_receipt.json`: fail-closed UNAVAILABLE record (driver).
- `iron_receipt.json`: fail-closed UNAVAILABLE record (7-iron).
- `../nightly/myosuite_receipt.json`: automated nightly CI lane receipt
  (fail-closed: 0 executed tests on unavailable engines yields
  `status: fail`).

## Regenerating Real Club Receipts

Install the myosuite/MuJoCo stack on a pinned host, execute the native replay
for each club, then produce receipts via `assess_myosuite_qualification(...)`
with the recorded `native_state`/`time_s`/`marker` data, nonzero executed test
count, and the committed model/candidate/capture hashes. Software-side
qualification logic is complete; native qualification remains explicitly
**blocked** until the engine is available.
