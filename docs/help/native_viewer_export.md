---
title: What Am I Looking At
tile_id: native_viewer_export
status: active
---

# What Am I Looking At

This page explains each picture the native viewer export draws on a golfer
video. Each section says what the picture is, what it is computed from, which
bundle it needs, and what it is not. To make the clips, see the
[Quick Start](../user_guide/native_viewer_export.md).

All force and torque pictures are displays of a model. None is a measurement.
When every recorded effort is zero, the exporter treats the bundle as a
kinematic replay. This is a display rule: it does not read provenance and does
not prove how the motion was produced. By default the clip then says
"Kinematic Replay: Forces Not Shown" and leaves out the force and torque
vectors. Explicit `--force-vectors` or `--torque-vectors` draws them anyway;
that does not make them measured or applied forces.

## Joint Torque Arcs

- **What it is:** Curved arrows at the joints showing the effort applied there.
- **Computed from:** The efforts recorded in the bundle, which the tracking
  controller applied. Joint positions and axes come from the displayed pose.
- **Needs:** A bundle with non-zero efforts.
- **What it is not:** Not a measured torque and not inverse dynamics. With a
  receipt file the pose comes from the rollout but the arcs still show the
  bundle's efforts. The free root of the body is not drawn.

## Ground Reaction Force Vectors

- **What it is:** Green arrows at each foot's centre of pressure, with the net
  force and free moment.
- **Computed from:** A sphere-ground contact model evaluated at the displayed
  pose and the bundle's reference velocity.
- **Needs:** A bundle with non-zero efforts to be shown by default.
- **What it is not:** No force plates are read. The values are not balanced
  against the motion on a kinematic replay.

## Weight

- **What it is:** One arrow at the centre of mass for the model's body weight.
- **Computed from:** Model mass and centre of mass.
- **Needs:** Non-zero efforts to be shown by default.
- **What it is not:** The model's mass, not the golfer's weighed mass.

## Grip Forces

- **What it is:** The force of each hand on the club, the net force and the
  equivalent couple at the grip midpoint (`--grip`).
- **Computed from:** The constraint force needed to hold the club in the hands
  under the bundle's efforts. The Drake clip uses Drake's own solve; other
  engines use MuJoCo.
- **Needs:** A bundle with non-zero efforts.
- **What it is not:** Not measured. The left and right split is a method choice
  named in the legend.

## Palm-Thumb Force

- **What it is:** One arrow for the trail palm pressing on the lead thumb
  (`--palm-thumb-force`), labelled "allocated (not measured)".
- **Computed from:** The grip closure force, split onto a contact patch using
  the contact topology and static friction.
- **Needs:** A bundle with non-zero efforts.
- **What it is not:** One possible split of many; not a measured contact.

## Compression and Tension Shading

- **What it is:** Blue (tension) and red (compression) colouring on body
  segments (`--segment-shading`); loads beyond the limit saturate.
- **Computed from:** Inverse dynamics on the displayed motion, with
  accelerations from an unfiltered finite difference, plus the modelled ground
  forces and the grip force. Segments are left unshaded around impact unless
  you pass `--no-axial-impact-blackout`.
- **Needs:** A bundle with non-zero efforts to mean anything. MuJoCo viewer only.
- **What it is not:** Not a measured tissue load. Mismatches between motion and
  modelled forces end up as a residual at the root; noisy accelerations show up
  as noisy colour.

## Mobility, Effective-Mass, and Capacity Ellipsoids

- **What it is:** Ellipsoids at body points (hands by default) describing how
  the body can move or push there: mobility (how easily the point moves in each
  direction), effective mass (inertia felt at the point, in kg), and force or
  acceleration capacity (what it can deliver).
- **Computed from:** The pose: the Jacobian and mass matrix. Capacity kinds also
  use joint torque limits.
- **Needs:** Any bundle; valid on a kinematic replay. Capacity kinds need torque
  limits in the model specification.
- **What it is not:** Not a measurement. Capacity depends on assumed torque
  limits. Under the default per-point scaling each point has its own scale, so
  sizes cannot be compared between points; a minimum drawn axis means a thin
  axis may not be to scale.

## Impact Factors Panel and Card

- **What it is:** Club speed, path, face angle, face to path, attack angle, loft
  and strike point, frozen at impact (`--impact-factors`).
- **Computed from:** Forward kinematics of the club face, differences of the
  face-centre position for velocity, and the ball placed where the face passes.
- **Needs:** Any bundle with a valid impact.
- **What it is not:** Kinematic values, not launch-monitor readings. The target
  line is assumed from the address face normal. A value that cannot be computed
  shows as unavailable, never as zero.

## Ball Flight Tracer and Inset

- **What it is:** After impact, the ball flies along a tracer line; a small
  inset plots the flight; the clip holds until landing (`--ball-flight`).
- **Computed from:** The repository's ball-flight model, launched from the
  clubhead state at impact. It is labelled "model prediction".
- **Needs:** Any bundle with a valid impact.
- **What it is not:** Not a measured shot. It ends at first ground contact: no
  bounce and no roll yet, so no total distance.

## Ball and Tee

- **What it is:** A decorative golf ball shown from the start until impact
  (hide it with `--no-ball`).
- **Computed from:** The address geometry of the club face, then moved to where
  the face centre passes nearest it. For a driver the ball can sit up to
  0.1016 m (the longest tee the rules allow) above the ground; other clubs lie
  on the ground.
- **Needs:** Any bundle.
- **What it is not:** Not part of the dynamics model, and not the captured
  ball. Moving it with `--ball-position-shift` is a display choice.

## Claims Not to Make

- Do not call torques, ground forces, grip forces, or palm-thumb forces measured.
- Do not say a ground force came from force plates.
- Do not say a given engine computed the forces: they come from one MuJoCo
  evaluation at that engine's pose (the Drake grip solve excepted).
- Do not say a force or shading picture describes the golfer on a kinematic
  replay.
- Do not compare ellipsoid sizes between points under per-point scaling.
- Do not quote impact factors as launch-monitor data or flight as total distance.
- Do not say an open-loop clip is accepted for a full swing.

## Known Issues

As of this writing, tracked in epic #15150 with fixes on other branches:

- Segment shading and the palm-thumb arrow are still drawn on a kinematic
  replay, and the grip data file and strips still use the zero efforts.
- A capacity ellipsoid requested without torque limits falls back to mobility
  sizes while keeping the capacity label.
- `--drift-arrow` is accepted but nothing draws it.
