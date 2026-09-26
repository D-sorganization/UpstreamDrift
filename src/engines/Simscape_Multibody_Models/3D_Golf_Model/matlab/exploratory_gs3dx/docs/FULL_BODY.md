# Full Body (`GS3DX_FullBody`)

Issues [#10957](https://github.com/D-sorganization/UpstreamDrift/issues/10957)
(lower body) and [#10958](https://github.com/D-sorganization/UpstreamDrift/issues/10958)
(ground and integration), epic
[#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b.

## What Was Built

`gs3dx_build_lower_body` copies `GS3DX_Quat` to `GS3DX_FullBody` and adds
one top-level subsystem, `Lower Body`. It connects to the pelvis
(`Lower Torso`) and World frames of `Hips and Torso Inputs`. Each side is
built from the segment/joint table in `gs3dx_leg_table`:

| Joint | Subsystem             | Distal segment   | Start target (deg) | Target priority |
| ----- | --------------------- | ---------------- | ------------------ | --------------- |
| Hip   | `GS3DX_KDS_Spherical` | Thigh (cylinder) | X-Y-Z [0 −θ 0]     | None            |
| Knee  | `GS3DX_KDS_Revolute`  | Shank (cylinder) | −2θ                | Low             |
| Ankle | `GS3DX_KDS_Universal` | Foot (brick)     | X-Y [0 −θ]         | Low             |

θ = 14.07°. At that angle the ankle sits under the hip at 97% of the
straight-leg length, with the knee forward and the foot level. Velocity
targets are None throughout.

Segment masses and lengths are de Leva (1996) male fractions of
`LegBodyMass` = 80 kg and `LegBodyHeight` = 1.80 m. These are assumptions,
not fitted to the golfer; they live in the model workspace. Leg torques come
from one Constant, `LegTorqueCommand` (12×1, zeros), through a Demux. The
legs are therefore passive until leg inputs are fitted.

**Placement.** `gs3dx_stance_frames` measures the pelvis and shoulder frames
of `GS3DX_Quat` at t = 0 under the impact drive. It adds sensors in memory
and closes without saving. The leg frame is then:

- z: opposite to gravity.
- y: the horizontal shoulder line, pointing to the left-shoulder side (the
  target side).
- x = y × z: the facing direction.

At t = 0 the golfer faces −Y and the pelvis frame is tilted 22.5°. The
hip centres sit 0.10 m below the pelvis frame and 0.18 m apart.

**Feet (weld stance, #10958).** Each foot frame is connected rigidly to
World under its ankle at the start posture, with no Weld Joint block. This
closes one kinematic loop per leg through the pelvis joint. Simscape ignores
targets when every joint in a loop has one, so the hip joint carries none.

**Block count:** 751 non-virtual (`GS3DX_Quat` 594, legs +157). The budget
with a 10% margin is 900, which leaves 149 spare.

## Integration Run (Impact Drive, 0.3 s)

| Quantity                           | `GS3DX_Quat` | `GS3DX_FullBody` |
| ---------------------------------- | ------------ | ---------------- |
| Status                             | success      | success          |
| Solver steps                       | 348          | 348              |
| Wall time (fresh session)          | 24 s         | 20–23 s          |
| Clubhead speed at 0.3 s            | 19.3 m/s     | 17.1 m/s         |
| Start state (clubhead, hands, hip) | reference    | equal to 3e-15   |
| Knee angle at t = 0 (both sides)   | —            | −28.14° = −2θ    |

**Start and assembly.** The full body starts in exactly `GS3DX_Quat`'s
state, so the legs do not move the pelvis at assembly. Both knees and ankles
assemble on their targets.

**Divergence.** The two models then separate because the passive legs load
the pelvis:

- The pelvis Euler angles differ by up to 10° at 0.1 s and 8–17° at 0.2 s.
- The pelvis translation stays identical.
- The clubhead ends 0.52 m away.

The legs add about 32 kg that the fitted pelvis inputs never accounted for,
so this divergence is expected.

**Energy and physical sanity.**

- The leg actuator torques are zero throughout, so the leg joints do no
  work.
- The weld frames are rigid and do no work, so all leg energy enters through
  the pelvis.
- The knee and left-ankle constraint torques peak at 40 N·m.
- The left knee flexes to −95°.
- The **right knee reaches +67°, which is hyperextension**. Passive knees
  follow wherever the pelvis rotation twists the closed loop.
- No kinetic-energy audit is computed yet: it would need per-body velocity
  sensing (see next steps).

## Foot Contact Trial (#10958)

`gs3dx_contact_trial` loads `GS3DX_FullBody`, replaces both weld frames with
an Infinite Plane at sole height and a Spatial Contact Force per foot
(1e6 N/m, 1e3 N·s/m), and simulates. It closes the model without saving.

| Stance  | Non-virtual blocks | Solver steps | Warm wall time (0.3 s) |
| ------- | ------------------ | ------------ | ---------------------- |
| Weld    | 751                | 348          | 20.1 s                 |
| Contact | 753 (+4, −2)       | 590          | 36.6 s                 |

Contact is cheap in blocks and costs 1.8× in wall time. With passive legs its
motion is not meaningful: the loops open and nothing holds the knees. It is
therefore only a price, not a candidate model yet.

## Next Steps Toward Matching

Tracked in [#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979).

1. Give the legs torques. Either fit `LegTorqueCommand` as input functions,
   like the arm joints, or drive the knees and ankles with motion targets
   taken from capture data. Once the legs are no longer passive, the right
   knee can no longer hyperextend.
2. Refit the pelvis inputs with the legs attached. The existing hip and
   translation input functions were fitted to an upper body with no legs.
3. Add an energy audit (kinetic plus potential energy per body) to separate
   drive work from numerical drift.
4. Move to contact once the legs carry load, and tune the stiffness against
   the extra solver cost measured above.

## Visual QA (#10959)

`gs3dx_layout_qa` prints diagrams headlessly to `docs/screenshots/` and lists
overlapping blocks. A Sonnet agent reviewed the images and reported issues
only.

**Fixed:**

- `GS3DX_KDS_Spherical` and the quaternion hip printed the `ActuatorTorque*`
  name once per branch of the torque-command line, so the labels overwrote
  each other. The named signals now leave the `Axis Torques` mux through a
  virtual `Actuator Torque` Demux, at no block cost.
- `Angle Reference` covered the `Axis Rate` and `XYZ Kinematics` labels. It
  has been moved.
- `Lower Body` is placed below `Hips and Torso Inputs`, at the first spot
  that overlaps no block, instead of far across the canvas.

**Result:** every rendered diagram has 0 overlapping blocks.

**Inherited from the original model, left as is:**

- The dense sensor bundle under the arm chain at the top level.
- The ~30-line fan-out into `HipLogs` in the hip subsystem.

**Not done:** opening each model in the Simulink GUI through computer use.
Agents here run headless by fleet rule, and computer use needs the owner to
grant app access interactively.
