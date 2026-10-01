# GS3DX_Neck: A Motion-Driven Neck

Issue [#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b, measured 2026-09-28.

In `GS3DX_Shape` the neck and head are rigid with the upper trunk. The head
therefore follows the trunk's turn and tilt, and its centre travels 213 mm
vertically before impact where the golfer's travels 55 mm (`docs/SHAPE.md`,
"Where the Remaining 12 mm Comes From"). `GS3DX_Neck` is `GS3DX_Shape` with a
two-axis neck driven by the capture's head markers. It is built by
`gs3dx_build_neck(info, reference=neck)`.

## The Joint

- **Where.** `Rigid Transform5` joins the top of `UpperTorsoTop` to the Neck's
  "Bottom of Neck" frame with no offset. A Universal Joint is inserted
  between the two, so it pivots at the base of the neck, and at zero angles
  every solid is where it is in `GS3DX_Shape` (1e-15 m).
- **Axes.** The neck frame has the upper trunk's orientation, z along the
  spine. The joint turns the neck and head about x, then y. The turn about z
  is left out, because the head is an ellipsoid of revolution about z with its centre on
  that axis, so the turn neither shows nor moves mass.
- **Drive.** Both angles are input motion from `NeckReference` (2 × n, rad,
  model workspace) on `LegReferenceTime`. A From Workspace block feeds two
  Simulink-PS Converters with second-order filtering
  (`NeckFilterTime`, 5 ms), which supply the derivatives. The joint computes its
  torque.
- **Blocks.** The neck costs 6 compiled blocks. The four massless elbow and
  shoulder spheres (1 compiled block each) are removed to make room, each
  checked massless and single-port first: 973 → 975, the cap under the
  25-block reserve.

## The Reference

The neck angles are the head's rotation relative to the upper trunk, from
address:

`N(t) = R_ut(t)ᵀ R_head(t) R_head(0)ᵀ R_ut(0)`

`R_ut` is the model's `UpperTorsoTop` posed by the regularised whole-body IK
(`gs3dx_render`). `R_head` is the capture's head frame
(`gs3dx_capture_head_frame`: HeadTop, HeadFront and HeadSide, rigid to 0.3
mm). `N` is split into X-Y-Z angles, and x and y are kept. To impact: x
−16.0 to 13.5°, y −17.6 to 4.8°. The dropped z runs from −75.7 to 1.3°: the
trunk turns under a head that stays on the ball.

## Result

Head centre to impact, every third capture frame (`gs3dx_render` poses),
against the centroid of the capture's three head markers:

| Model         | Vertical range (mm) | At impact (mm) | Vertical error RMS / max (mm) | Facing error RMS / max (mm) |
| ------------- | ------------------- | -------------- | ----------------------------- | --------------------------- |
| `GS3DX_Shape` | 213.2               | −129.2         | 52.2 / 106.0                  | 72.4 / 170.6                |
| `GS3DX_Neck`  | 112.8               | −81.4          | 36.8 / 62.2                   | 67.7 / 169.9                |
| capture       | 54.8                | −54.8          | —                             | —                           |

The neck halves the head's vertical travel and cuts the vertical error by 30
%. The remainder, and the facing error, which barely moves, belong to the
trunk. The neck base follows the model's upper trunk, and the model's trunk
is not the capture's (its C7 proxy, `docs/SHAPE.md`). The marker centroid
also sits on the surface of the head, not at its centre.

The orientation error is almost all the dropped axial turn: against the
capture's full head rotation it is 44.9° RMS for Shape and 44.1° for Neck.

![GS3DX_Neck face-on at impact](screenshots/GS3DX_Neck_fo_imp.png)

## Rendering

Adding a joint renumbers Simscape's joint IDs, which are numbered in
block-path order. The new neck takes `j2`, and the right-arm loop becomes
`j16`, `j19` and `j20`. `gs3dx_joint_keys` names each joint variable by its
block path below the model and its primitive, for example
`Right Elbow Joint/Revolute Joint/Kinetically Driven Revolute|Rz.q`.
`gs3dx_render` now finds the closed right-arm loop by these keys, and it maps
an IK solved on another variant (`ik.model`), or rows named by
`ik.joint_keys`, onto the model's joints by key. To pose the neck, append its rows:

```matlab
[k0, id0] = gs3dx_joint_keys(char(ik.model));
[~, r] = ismember(string(ik.joint_ids), id0);
ik.joint_keys = [k0(r); "Hips and Torso Inputs/Neck Joint|Rx.q"; "Hips and Torso Inputs/Neck Joint|Ry.q"];
ik.joint = [ik.joint; rad2deg(neck_at_ik_frames)];
```

## Tests

`tests/test_gs3dx_neck.m` (5 tests):

- the neck joint is a Universal Joint with input motion, between Rigid
  Transform5 and the Neck, with a finite `NeckReference` on
  `LegReferenceTime`;
- the four spheres are the only solids removed, and the mass is unchanged;
- the model compiles within the 25-block reserve;
- `GS3DX_Shape`'s joint keys all survive, plus the two neck keys;
- with the neck straight every solid has its `GS3DX_Shape` pose to 1e-12.
  With the neck turned, only the Neck and Head move, rigidly about the
  neck base.
