# Ground Contact (`GS3DX_FullBodyContact`)

Issue [#10986](https://github.com/D-sorganization/UpstreamDrift/issues/10986),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b. The parameter sources are in [DATA_AUDIT.md](DATA_AUDIT.md).

## What Was Built

`gs3dx_build_contact` copies `GS3DX_FullBody` to `GS3DX_FullBodyContact`.
FullBody itself is only read. The build makes four changes.

| Part   | GS3DX_FullBody                         | GS3DX_FullBodyContact                                                                                                 |
| ------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Feet   | welded to World 0.18 m apart           | three sole spheres per foot (heel centre, toe inside, toe outside) on one Infinite Plane, smooth stick-slip friction  |
| Stance | under the hips                         | captured address stance: ankles ±0.29 m (0.65 m marker width less 3.5 cm malleolus inset), foot yaw −2.15° / +8.15°   |
| Pelvis | 6-DOF joint driven by force and torque | same joint, **unactuated**: `NoTorque` on all axes, its four drive converters deleted; start targets and sensing kept |
| Legs   | passive (`LegTorqueCommand` = 0)       | stance-hold servo on all 12 axes; start angles and rates from inverse kinematics, High priority                       |

**Stance and inverse kinematics.**

- The foot targets are the captured ankle positions, placed below the Quat
  pelvis at t = 0 in the FullBody leg frame [facing, lateral, up].
- `gs3dx_leg_ik` solves the six leg angles per side. It uses damped
  Gauss-Newton on `gs3dx_leg_fk`, and the FK matches Simscape to round-off
  (`tests/test_gs3dx_leg_kinematics.m`).
- The start rates come from the IK over Quat's first 4 ms. They keep the
  feet still while the pelvis moves at its start velocity.
- The knees start at −28.3° (L) and −26.6° (R).

**Servo.**

- τ = `LegTorqueCommand` + Kp ⊙ (`LegAngleReference` − q) − Kd ⊙ q̇
- Kp = 100 N·m/deg (hip, knee) and 50 (ankle); Kd = 2 and 1 N·m·s/deg.
- `LegAngleReference` holds the t = 0 stance.
- It uses the existing Constant (now `LegTorqueCommand + LegServoKp .* LegAngleReference`),
  one matrix Gain `[diag(Kp) diag(Kd)]` on the stacked [q; q̇] and one Sum.

**Contact.**

- Sphere radius 1 cm, stiffness 1e5 N/m, damping 1e3 N·s/m, μs 0.9, μk 0.7.
- These values are assumptions. No GRF data exists to fit them.
- Each contact logs the ground-on-foot force in the ground frame into
  `FootContactForces` (18×1).

## Block Budget

The Home license counts the **compiled** model (see
[BLOCK_BUDGET_FINDINGS.md](BLOCK_BUDGET_FINDINGS.md)). FullBody already
compiles to 945 of 1,000 blocks, so the design is deliberately lean:

- three contacts per foot, not four;
- no From Workspace references;
- the pelvis converters removed, not gated.

It compiles to **967**. That leaves room for the 25 blocks of in-memory
sensing that `gs3dx_contact_check` adds. The builder asserts this reserve.

## Validation

`gs3dx_contact_check` simulates the model for 0.3 s with mass and COM sensing
added in memory. Its main test is Newton's second law: with no pelvis drive,
the contacts and gravity are the only external forces, so

M (v_com(t) − v_com(0)) = ∫ (ΣF_contact + M g) dt,

to within 1% of M·|g|·T (3.2 N·s).

| Run                     | Newton residual | Foot slip L / R | Foot lift L / R | Support (× body weight) | COM drop |
| ----------------------- | --------------- | --------------- | --------------- | ----------------------- | -------- |
| From rest (`rest=true`) | 0.71 N·s ✓      | 1.8 / 2.1 mm    | 0 / 0 mm        | up to 1.08              | 8.3 cm   |
| Impact drive            | 0.98 N·s ✓      | 114 / 71 mm     | 214 / 1 mm      | up to 1.33              | 22 cm    |

**From rest the golfer stands.** The feet stay planted and the ground carries
the weight. The COM drop is the upper body sagging: its joints hold the drive's
torques, not a standing posture.

**With the impact drive the golfer tips over the right foot.** This does not
depend on the gains: a 20× range of Kp and Kd changes nothing. The cause is
the start state. The drive starts mid-downswing, and the whole body's COM
already moves at 1.21 m/s away from the target: 131 N·s of sideways momentum.
The upper body was fitted with a driven pelvis that could supply any force,
and planted feet cannot absorb that momentum without tipping. The fix is the
full-body refit
([#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979))
with a capture-derived pelvis path. Tuning contact or gains will not fix it.

## Rigid-Body Stance Check (Cross-Engine Parity, #11709)

The full articulated `GS3DX_FullBodyContact` cannot give the cross-engine
stance parity numbers GCV-2 defines
(`tests/integration/cross_engine/test_ground_reaction_parity.py`: net
vertical GRF within 2% of weight, left/right split within 5%, CoP between
the feet) because it has no affordable held posture:

| hold method                                       | cost                                                        | outcome                                                                                                                                                                                                                                                        |
| ------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| passive (`rest=true`, unactuated)                 | ~0.5 s wall per ms                                          | self-consistent (Newton check passes) but the body topples: feet slip up to 1.50 m and lift 0.16 m over 2 s, so there is no settled stance to compare                                                                                                          |
| joint-space PD hold at the start pose             | ~20 s wall per ms                                           | momentum residual grows without bound; not physically consistent                                                                                                                                                                                               |
| every non-root joint locked (`gs3dx_lock_joints`) | **~140 s wall per ms** (a 2 s stance would take about 78 h) | an algebraic loop (36 blocks, lumped Simscape output through the leg servo and every `XYZ Torque` input) is paid on every model evaluation regardless of the solver; the loop's torques are ignored once the joints are locked, but the evaluation cost is not |

Director-approved option (b) (2026-10-10): **a rigid-body stance check.**
One rigid body carries the golfer's total mass and world centre of mass
(read from a 4 ms probe of `GS3DX_FullBodyContact` at rest — cheap, because
it is the model's own start pose, not a settled trajectory) on the same
six sole-sphere contacts, placed at the same world points, with the same
Spatial Contact Force law and parameter values (by workspace variable name)
as `GS3DX_FullBodyContact`. This tests exactly what the stance parity
compares — the contact law under the body's weight and centre of mass —
without paying for the full articulated tree: one 6-DOF joint and six
zero-DOF contact points compile to a few dozen blocks, not 967.

Built fresh in memory by `gs3dx_rigid_stance` (never saved, never touching
`GS3DX_FullBodyContact.slx`), exported by
`scripts/matlab/export_gs3dx_rigid_stance_fixture.m` into
`tests/fixtures/simscape/simscape_gs3dx_rigid_stance_fixture.csv` /
`simscape_gs3dx_rigid_stance_receipt.json`, in the same column schema as
`export_gs3dx_grf_fixture.m` so the Python loader
(`src.engines.simscape.force_channels.load_simscape_force_series`) reads it
unchanged. `gs3dx_contact_geometry` and `gs3dx_sole_contact_points` are
shared between the two exporters (DRY): the rigid proxy reuses the exact
contact-law values and sphere placement the full model simulates with,
rather than recomputing them from `gs3dx_leg_table`.

**Approximation.** The body's rotational inertia is a solid sphere of the
same total mass (default radius 0.15 m, a torso-scale guess with no
measurement behind it). A settled standing stance carries no net torque
about the centre of mass, so the forces this check reports at rest do not
depend on the inertia tensor — only the transient settling trajectory and
its duration do.

**Result** (MATLAB R2025b, 2 s settle, `ode23t`, 109.388 kg, 44 compiled
blocks, 2.8 s wall clock). Measured at the last logged frame against the
unchanged GCV-2 tolerances (2% weight, 5% split, CoP within 0.05 m x / 0.01 m y):

| criterion                        | measured           | GCV-2 bound        |
| -------------------------------- | ------------------ | ------------------ |
| net vertical GRF vs weight       | +0.087%            | 2%                 |
| left foot vs half weight         | -0.251%            | 5%                 |
| right foot vs half weight        | +0.425%            | 5%                 |
| net CoP vs foot midpoint (x, y)  | -0.97 mm, +0.02 mm | 0.05 m x, 0.01 m y |
| settled slip / sink (last 0.5 s) | 9.8 mm / 2.2 mm    | n/a                |

All four checks pass, so Simscape now has stance-parity numbers against the
GCV-2 criteria. The check is a rigid-body proxy for the contact law, weight
and centre of mass, not a full-body held posture; it does not test the
articulated model's joint dynamics. A failing criterion would be recorded
as an `xfail(strict=True)` with the measured number, never loosened.

## Open Items

- The trunk mass double-count (109.4 kg model) is resolved in
  `GS3DX_Golfer` (80 kg, [ANTHROPOMETRY.md](ANTHROPOMETRY.md)).
- There is no GRF data to fit the contact parameters. The total GRF from the
  capture kinematics (`gs3dx_kinematic_grf`) is the validation reference.
- The servo holds a constant stance. A swing needs time-varying leg
  references or torques from inverse dynamics, and at this budget they must
  replace the Constant rather than add blocks.
