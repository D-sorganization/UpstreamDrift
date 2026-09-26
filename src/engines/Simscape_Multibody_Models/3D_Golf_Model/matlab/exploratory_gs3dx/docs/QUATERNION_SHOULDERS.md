# Quaternion Shoulders (`GS3DX_Quat`)

Issue [#10955](https://github.com/D-sorganization/UpstreamDrift/issues/10955),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b.

## What Changed

`gs3dx_build_quat` copies `GS3DX_Slim` to `GS3DX_Quat` and
`GS3DX_KDS_Gimbal` to `GS3DX_KDS_Spherical`, and then re-points both
shoulders at the new subsystem. In that subsystem,
`gs3dx_gimbal_to_spherical` replaces the Gimbal Joint (three revolutes with
Euler-angle states) with a Spherical Joint (quaternion state). The
subsystem interface does not change: the same ports, the same mask
parameters and the same 21 bus elements with the same names and units.

| Signal                            | Gimbal                           | Spherical stand-in                                                |
| --------------------------------- | -------------------------------- | ----------------------------------------------------------------- |
| Torque X/Y/Z inputs               | Per-axis joint torque            | `XYZ Torque`: follower torque T = E⁻ᵀ(τ − c·q̇)                    |
| Angular position, velocity, accel | Sensed per revolute              | `XYZ Kinematics`: from Q, ω and the sensed follower angular accel |
| Start angles and rates            | Per-axis targets                 | Follower-axes X-Y-Z rotation sequence, and ω = E·q̇                |
| Composite force/torque            | Base on follower, follower frame | Same settings, copied from the Gimbal                             |

E is the X-Y-Z rate matrix (`gs3dx_xyz_rate_matrix`), and
`gs3dx_xyz_map` does the mapping (7 unit tests). `Angle Reference`
integrates q̇ only to pick the 360° branch of the X and Z angles. The
original left-shoulder X angle reaches -523°.

Joint damping stays in Gimbal-axis terms (N·m per deg/s on q̇), so it keeps
its meaning.

**Block count:** 609 → 599 non-virtual (10 saved).

## Equivalence

### Isolated Joint Rig (`gs3dx_joint_rig`)

In the rig, each subsystem carries an offset 1.8 kg brick under gravity,
with sine torques on all three axes. The solver is `ode15s` with
RelTol = AbsTol = 1e-8. All 21 bus signals of the Spherical stand-in match
the Gimbal to ≤ 2e-7 of each signal's peak. The test bound is 1e-5.
Warm rig wall times are comparable: Gimbal 3.2–4.1 s, Spherical 3.4–3.6 s
over 1 s simulated.

The first attempt used larger torques, and the **Gimbal** rig stopped in
gimbal lock (middle angle ±90°). That is the failure the quaternion joint
removes.

### Full Model

Under the plain impact drive, the left shoulder (High priority) starts in
the same state as the original, but the right shoulder (Low priority) does
not. For example, its X angle is -7.77° in the original and 0.51° in
`GS3DX_Quat`. Both hands hold the club, so assembly cannot meet the right
shoulder's targets and settles on a compromise. The Gimbal compromises per
axis, while the Spherical Joint has one rotation target and compromises
differently. The runs then diverge.

`gs3dx_pinned_drive` sets the right shoulder's start state to the one the
original assembles and raises its priority to High. With it, both models
start from the same state (checked to the printed digit). They still differ
at the default RelTol 1e-3, because different state variables give
different truncation errors. That difference converges away:

| RelTol | \|Quat − Slim\| clubhead | \|Slim − ref\| | \|Quat − ref\| | Steps Quat / Slim |
| ------ | ------------------------ | -------------- | -------------- | ----------------- |
| 1e-3   | 13.9 mm                  | 17.2 mm        | 4.4 mm         | 344 / 367         |
| 1e-5   | 0.54 mm                  | 0.70 mm        | 0.22 mm        | 969 / 1,070       |
| 1e-7   | 0.039 mm                 | (ref)          | 0.039 mm       | 3,566 / 3,863     |

The reference ("ref") is `GS3DX_Slim` at RelTol 1e-7. At every tolerance
`GS3DX_Quat` is closer to the converged solution than `GS3DX_Slim`, and it
takes 7–9% fewer steps. `test_gs3dx_quat/quat_converges_to_the_slim_solution`
checks this. Under the same drive, a 1e-9 RelTol perturbation moves the Slim
clubhead only 6.7e-8 m, so the drive is well conditioned.

The pinned drive reaches a left-shoulder middle angle of 87°, close to the
Gimbal's singularity. Near that angle the Euler-angle outputs of both models
are ill-conditioned, which is why the left-shoulder acceleration signals
have the largest differences at RelTol 1e-3.

## Limits and Follow-Up

- A Spherical Joint has one position target and one velocity target. The
  mask init therefore requires the same priority on Rx/Ry/Rz for each
  quantity, and errors otherwise.
- A Low-priority shoulder target assembles differently from the Gimbal (see
  above). To reproduce a Gimbal assembly exactly, pin the state as
  `gs3dx_pinned_drive` does.
- The torque-path cost of the Euler outputs is kept only for bus
  compatibility. A quaternion-native bus would drop `XYZ Kinematics` and
  `Angle Reference`.
- Input-function mode sweep on the impact inputs: mode 3 is ill-conditioned
  (a 1e-9 RelTol change moves the clubhead 122 m), mode 2 is torqued but
  reaches 1e8 N·m, and mode 1 stops at 35 ms with **hip gimbal lock**. That
  lock is the case for the quaternion hip (#10956).
