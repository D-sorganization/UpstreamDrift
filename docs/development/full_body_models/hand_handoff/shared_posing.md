# Shared Hand Posing Handoff

This file is the handoff for the shared posing track of epic
[#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150).
Read it before changing the poser. The branch is
`cursor/shared-hand-posing-6902`, based on
`cursor/hrg-gb1b-shaft-roll-2882`. The work is partial. The contact tests
pass. The skin pictures do not.

Cloud lanes stopped on 2026-11-15. The next owner is a subscription fleet
agent (Claude Code on Brick, or Antigravity on DeskComputer). Post the
plug-in note from [How Each Backend Plugs in Its Chain](#how-each-backend-plugs-in-its-chain)
as a comment on #15150. Use `Refs #15150`. Do not use `Fixes` or `Closes`.
The epic stays open. Dieter has not signed the visual checklist.

## Goal and Quality Bar

Build one backend-agnostic poser. Every visual backend (MakeHuman skinned,
Shadow, MyoHand, procedural skin, Allegro, and a signed-distance grip)
consumes the same joint angles and the same contact report. This pull
request does not own `registry.py`. Sibling pull requests each add their
own registry, and `hand_backends/__init__.py` will conflict when they merge.

The quality bar on #15150, which this branch does not yet meet:

- Renders of the lead hand, the trail hand, and both hands.
- Three views, at shaft roll 0°, +20°, and −20°.
- Solid smooth skin, with a key light and a fill light.
- The hand about 70% of a frame at least 1024 px on a side.
- The club in every view.
- Fingertip and middle-phalanx contact within a tolerance setting.
- No penetration beyond a tolerance setting.
- Left and right hands mirror.
- Anthropometric ratios within a tolerance.
- A written checklist that Dieter signs before merge.
- A licence next to any third-party asset.

The contact tolerances are settings, not literals:
`contact_gap_tolerance_m` is 0.002 and `penetration_tolerance_m` is 0.0005.
The capsule contact tests meet those two numbers on the default grip.
The renders on this branch are contact diagrams drawn as thick lines.
They fail the skin bar. They are not Dieter's sign-off.

## Flat-Hand-Then-Curl Grip Spec

Dieter's grip is a procedure, not a rest pose copied from a robot hand.

1. Place the open hand so the palm faces the grip and the fingers are
   straight. The proximal tangent point of each finger, which is the
   midpoint of that bone, lies on a line perpendicular to the shaft.
   On the default cylinder that line is 90° from the shaft and the index
   axis distance is `grip radius + finger radius + base_clearance_m`.
2. Curl each joint from proximal to distal until the capsule surface is
   inside the contact band. The metacarpal arch is part of the curl:
   the index stays nearer the grip and the little sits just below it.
3. Take the flat snapshot before the arch subtracts clearance. The
   snapshot is `HandPose.flat`.
4. Overlap, interlock, ten-finger, and the thumb come from `GripSpec`.
   They are not a second poser.
5. Apply shaft roll after the curl, about the appearance shaft
   (`+Z`, butt toward the club head). Positive roll is the right-hand
   rule about that axis. A left-handed spec mirrors the sign through
   `resolved_shaft_roll_deg` unless the caller passes an explicit
   `shaft_roll_deg`.

`GripSpec.legacy()` is the default: overlap, `club_cross_angle_deg` 40,
`palm_finger_depth` 0.5, `thumb_straightness` 1, lead hand origin 0.032 m
below the butt, hand spacing 0.0762 m. Lead and trail V azimuths are 20°.
The lead thumb azimuth is 10° and the trail thumb azimuth is 0°. Handedness
`"left"` flips the V azimuth and the curl sign.

## Current State

### Files

| Path                                                              | Role                                                      |
| ----------------------------------------------------------------- | --------------------------------------------------------- |
| `src/shared/python/model_appearance/hand_backends/posing.py`      | `pose_grip` and `pose_chain`                              |
| `src/shared/python/model_appearance/hand_backends/chain.py`       | Default kinematic chain                                   |
| `src/shared/python/model_appearance/hand_backends/solve.py`       | Tangent wrap, shaft roll, hinge bisection                 |
| `src/shared/python/model_appearance/hand_backends/report.py`      | Pose and contact dataclasses                              |
| `src/shared/python/model_appearance/hand_backends/metrics.py`     | `hand_contact_quality_v1`                                 |
| `src/shared/python/model_appearance/hand_backends/settings.py`    | Loader for the JSON settings                              |
| `src/shared/python/model_appearance/hand_backends/grip_solid.py`  | Cylinder and tapered cone                                 |
| `src/shared/python/model_appearance/hand_backends/gb2_adapter.py` | Read-only adapter for GB2 #15134                          |
| `src/shared/python/model_appearance/hand_backends/__init__.py`    | Public exports. Sibling pull requests will conflict here. |
| `scripts/config/hand_pose_contact.json`                           | Every tolerance and gain                                  |
| `scripts/render_shared_hand_pose.py`                              | Headless contact diagrams                                 |
| `tests/unit/model_appearance/hand_backends/test_shared_posing.py` | 18 unit tests                                             |
| `docs/development/full_body_models/hand_handoff/figures/`         | 28 views plus `contact_sheet.png`                         |

Do not add this poser to `model_appearance/__init__.py`. Do not add
`registry.py` on this branch. Do not edit `vendor/ud-tools`.

### Functions

`pose_grip(spec, grip=None, *, shaft_roll_deg=None, settings=None)` builds
the lead chain, poses it, builds the style override from the lead index,
poses the trail hand against those capsules, and returns a
`GripPoseSolution`. `grip=None` uses `default_grip()`, which is
`get_grip_preset()` (`legacy_spec_standoff`, a cylinder of radius 0.0127 m).

`pose_chain(chain, spec, grip, *, side, shaft_roll_deg=None, settings=None, neighbour_capsules=(), neighbour_finger="")`
poses one chain. A joint whose `drive` is unknown, or a joint left without
an angle, raises `ValueError`.

Joint names are `{side}_{finger}_{segment}` from `joint_name`. Fingers are
`index`, `middle`, `ring`, `little`, and `thumb`. Segments are
`metacarpal`, `proximal`, `middle`, and `distal`. The thumb has no middle
segment.

| `drive`   | What it does                                                                                                                             |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `tangent` | Three equal-radius phalanges on one circle, copied from the construction in `grip_hand_mesh` without calling the private `_finger`.      |
| `arch`    | `angle = palm_finger_depth * arch_weight * arch_gain_rad`. No capsule. It does not move the joint origin.                                |
| `thumb`   | Straightness 1 is a straight bone along `+Z`. Straightness 0 is the tangent curl. Values between blend and stay outside the seat circle. |
| `hinge`   | Bisection on a shaft-parallel axis (`+Z` only).                                                                                          |

`curl_sign_for` is −1 for the lead hand and +1 for the trail hand. A
left-handed spec flips both.

`contact_quality_metrics` and `metrics_from_tables` emit schema
`hand_contact_quality_v1`. The keys are `CONTACT_QUALITY_METRIC_KEYS`:
`schema`, `contact_gap_tolerance_m`, `penetration_tolerance_m`,
`per_segment_gap_m`, `per_segment_grip_gap_m`, `per_segment_penetration_m`,
`per_segment_neighbour_gap_m`, `wrap_angle_rad`, `palm_contact_area_m2`,
`max_gap_m`, `max_penetration_m`, `within_tolerance`.

`within_tolerance` checks the grip gap only for segments with `check_grip`,
the neighbour gap only for segments with `check_neighbour`, and penetration
on every capsule. A negative neighbour gap fails the pose even when that
segment was not a required contact. `GripPoseSolution.gb2["applied"]` is
always false. `trigger_finger_applied` and `coupling_overrides_applied` are
always false.

### Settings

`load_pose_settings()` reads `scripts/config/hand_pose_contact.json`.
`PoseSettings` stores radians. The loader converts the degree fields.
Finger radii, phalanx lengths, and the thumb radius are not in the JSON.
They come from `HandMeshParams` / `DEFAULT_HAND_MESH_PARAMS` and scale with
`hand_mesh_scale` (hand length divided by 0.1932 m). Validation is in the
loader. `dataclasses.replace` can tighten `base_clearance_m` without
re-running that check.

| Setting                                                  | Value                                               | Use                                                                         |
| -------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------- |
| `contact_gap_tolerance_m`                                | 0.002                                               | Upper edge of the contact band                                              |
| `penetration_tolerance_m`                                | 0.0005                                              | Lower edge of the band                                                      |
| `base_clearance_m`                                       | 0.0004                                              | Index target gap                                                            |
| `just_below_pinky_m`                                     | 0.0015                                              | Extra target gap at the little, scaled by stack fraction                    |
| `open_gap_m`                                             | 0.02                                                | Seat radius where the curl search starts                                    |
| `curl_samples`                                           | 25                                                  | Retained search count. The closed form does not use it.                     |
| `mcp_limit_deg` / `pip_limit_deg` / `dip_limit_deg`      | 110 / 120 / 100                                     | Flexion limits. A larger angle raises.                                      |
| `arch_gain_rad`                                          | 0.45                                                | Full-depth arch angle at weight 1                                           |
| `arch_travel_m`                                          | 0.004                                               | Reserved radial travel. The arch does not apply it to the joint origin yet. |
| `arch_weight`                                            | index 0.15, middle 0.4, ring 0.75, little 1         | Little must be at least the index.                                          |
| `cross_azimuth_scale`                                    | 0.25                                                | 40° club cross becomes 10° of knuckle azimuth                               |
| `thumb_lengths_m`                                        | 0.046, 0.032, 0.022                                 | Metacarpal, proximal, distal                                                |
| `thumb_flex_limit_deg`                                   | 80                                                  | Thumb curl limit                                                            |
| `palm_grid_nu` × `palm_grid_nv`                          | 7 × 5                                               | Uniform palm patch                                                          |
| `palm_standoff_m` / `palm_arch_travel_m` / `palm_cell_m` | 0.012 / 0.012 / 0.008                               | One gap for every cell                                                      |
| `min_wrap_rad`                                           | 1.0                                                 | Floor used by the wrap test                                                 |
| `style_separation_m`                                     | 0.005                                               | Ten-finger little must stay at least this far from the lead index           |
| `hinge_bisection_steps`                                  | 24                                                  | Contact bisection                                                           |
| `axis_samples`                                           | 13                                                  | Samples along a capsule, including the midpoint                             |
| `metacarpal_length_m`                                    | index 0.068, middle 0.066, ring 0.058, little 0.052 | Arch bone length. The arch emits an angle only.                             |

`HandMeshParams` radii in metres: index 0.0080, middle 0.0082, ring 0.0078,
little 0.0068, thumb 0.0095. Phalanx lengths (proximal, middle, distal):
index (0.040, 0.024, 0.019), middle (0.045, 0.028, 0.020), ring
(0.042, 0.027, 0.020), little (0.034, 0.020, 0.018). `finger_gap_m` is
0.0003.

The loader rejects `base_clearance + just_below` above the gap tolerance,
a palm standoff already inside the gap band, and a depth-1 palm gap outside
the band.

### Tests

```bash
python3 -m pytest tests/unit/model_appearance/hand_backends/test_shared_posing.py -m unit --tb=short
python3 -m ruff check src/shared/python/model_appearance/hand_backends tests/unit/model_appearance/hand_backends scripts/render_shared_hand_pose.py
python3 -m ruff format --check src/shared/python/model_appearance/hand_backends tests/unit/model_appearance/hand_backends scripts/render_shared_hand_pose.py
```

All 18 tests passed on this branch before the handoff commit.
`pytest-timeout` is not installed in the cloud environment that produced
this branch, so `--timeout=60` is an unrecognized argument there. CI may
add it. The file is marked `pytest.mark.unit`.

The tests cover the settings file, flat placement, knuckle span, default
overlap contact, arch depth, ten-finger and interlock, shaft roll 0° and
±20°, left/right mirroring, thumb straightness 0 and 1, the
`mens_standard` taper, hand length, the settings clearance, one
shaft-parallel hinge, rejected inputs, the untouched trigger-finger flag,
the GB2 adapter, and `metrics_from_tables`.

### Relation to GB2

GB2 is pull request
[#15134](https://github.com/D-sorganization/UpstreamDrift/pull/15134),
branch `cursor/hrg-gb2-grip-builder-0932`. It has not merged. It poses the
lead fingers only. `gb2_curl_status` tries
`src.shared.python.grip_build`. When the import fails, `available` is false
and `applied` is false, and the note names #15134 and that branch. When
the import works, the adapter stores `build_lead_fingers(spec).joint_angles_rad`
and `BUILDER_VERSION`, still with `applied` false. Local tangent angles are
never replaced. `ensure` in `pose_grip` rejects a true `applied` flag.

Two GB2 behaviours must not be copied:

- `ClubSDF.signed_distance_capsule` takes the maximum of the sampled signed
  distances. Clearance is the minimum sample. A max sample reports the
  bone as clear while one end is inside the grip.
- A phalanx placed along the shaft and hinged about the shaft does not
  wrap the grip. The tangent construction is the one that puts the pad on
  the cylinder.

GB2 also writes the distal angle as two thirds of the proximal interphalangeal
angle after an independent distal solve. Leave the distal contact
independent.

## What Worked

Measured on `GripSpec.legacy()` with the default cylinder (radius 0.0127 m)
and the committed settings.

Flat placement, taken before the arch subtracts clearance:

- Index gap 0.0004 m. Little gap 0.0019 m.
- Index axis distance 0.0211 m, which is radius + index radius + base clearance.
- Index-to-shaft angle 90.0°.
- Knuckle azimuths −90.00° (index) and −100.00° (little). The span is 10.00°,
  which is 40° times `cross_azimuth_scale` 0.25.

Lead stations on `+Z`, with the hand origin at 0.032 m: index 0.05525 m,
middle 0.03875 m, ring 0.02245 m, little 0.00755 m. Index-to-middle
centerline separation is 0.0165 m. The radii sum to 0.0162 m, so the
surface clearance is 0.30 mm. Bones that stay at constant `z` cannot
intersect when the station gap exceeds the sum of the radii.

Lead flexion after the arch, in degrees (proximal / middle / distal):

| Finger | Metacarpal | Proximal | Middle | Distal |      Wrap |
| ------ | ---------: | -------: | -----: | -----: | --------: |
| Index  |      1.93° |   86.93° | 54.20° | 46.14° | 3.128 rad |
| Middle |      5.16° |   93.14° | 61.05° | 48.72° | 3.326 rad |
| Ring   |      9.67° |   90.27° | 61.15° | 49.83° | 3.315 rad |
| Little |     12.89° |   81.01° | 49.08° | 45.58° | 3.005 rad |

Every on-grip finger and both thumb capsules finish at a grip gap of
0.0004 m. `max_gap_m` is 0.0004 m. `max_penetration_m` is 0.
`within_tolerance` is true. At depth 0.5 the arch cancels `just_below` on
the final little, so the final little gap equals the index gap. The flat
snapshot still holds the 0.0019 m little gap.

Overlap trail little, posed as a shorter centered copy of the lead index
and shifted out by the sum of the radii:

- Proximal grip gap 16.40 mm, neighbour gap 0.00 mm. It is off the grip and
  on the lead index.
- Middle grip gap 17.04 mm, neighbour gap 0.44 mm.
- Distal grip gap 16.41 mm, neighbour gap 0.010 mm.

Interlock puts the trail little on the grip (gap = base clearance) and
keeps the proximal neighbour gap inside 2 mm, at
`z = index station + index radius + little radius`, with the trail curl
sign. Ten-finger leaves the trail little on the grip and more than
`style_separation_m` (5 mm) from the lead index. All three styles report
`within_tolerance` true.

Shaft roll ±20° leaves flexion and grip gaps unchanged and shifts the lead
index proximal midpoint azimuth by exactly ±20°. An explicit
`shaft_roll_deg` overrides both hands. A left-handed spec with lead roll
+20° resolves to −20°.

Left versus right: absolute flexion matches, `x` and `z` match, `y`
negates, and the gaps match.

Thumb straightness 1: flexion 0, grip gap 0.0004 m, azimuth equal to the
resolved thumb azimuth (lead 10°). Straightness 0 uses the tangent curl and
stays inside the band.

`mens_standard` is a tapered cone (`d_upper` 0.02286 m, `d_lower` 0.01981 m,
stations 0.0508 m to 0.1524 m). `GripSolid.radius_at` matches
`GripGeometry.radius_at` at 0.05, 0.08, and 0.12 m. The slope is about
−0.015 m per metre. The tapered pose is `within_tolerance`.

One shaft-parallel hinge, origin (0.0307, 0, 0.05) m, rest direction `+X`,
length 0.030 m, radius 0.008 m, limit 180°, reaches contact at 132.32°
with a gap of exactly 2.0 mm. That is the outer edge of the band, the least
angle `bisect_contact` accepts. A `+Y` axis raises `ValueError` and the
message contains `shaft`.

A longer hand (length 0.22 m, breadth `0.22 * 0.449`) produces a longer
index proximal. Replacing `base_clearance_m` with 0.0002 m moves the flat
index gap to 0.0002 m.

## What Failed

The pictures fail the skin bar. See [Score on This Branch](#score-on-this-branch).
They were opened and scored on 2026-11-15. The long axis of the strokes is
72.3% of the 1024 px frame. The width is 31.7% for both hands and 45.4%
for the lead hand alone, because the drawing is a set of thin capsules.

The palm is not a mesh. Every palm cell shares one gap,
`palm_standoff_m - depth * palm_arch_travel_m`. Area is 0 at depth 0 and
at the default depth 0.5. At depth 1 the area is `7 * 5 * 0.008²` =
0.00224 m². The whole patch switches on together.

The arch angle changes with depth. The metacarpal joint origin does not.
`arch_travel_m` is unused by the solver.

At depth 0.5 the final little grip gap collapses to the index gap
(0.0004 m). The "just below" offset survives only on `HandPose.flat`.

The overlap little's middle and distal bones are parallel copies stepped
outward until they clear the lead index capsules. In
`trail_oblique_roll_0.png` those copies read as short disconnected dashes,
not as one finger lying on the lead index. The proximal neighbour gap is 0. The middle neighbour gap is 0.44 mm.

`trigger_finger` true still reports `trigger_finger_applied` false, and the
trail index stays on the grip. `joint_coupling_overrides` are ignored.

A serial hinge whose root origin is already on the clearance circle returns
0° for that root, because `bisect_contact` takes the least angle already
inside the band. The child is then left short of the grip. Reproduction:
middle-finger radius 0.0082 m, root origin
`(0.0127 + 0.0082 + 0.0004, 0, 0.04)` m, rest direction `+Y`, proximal
length 0.045 m with a 110° limit, middle length 0.028 m with a 120° limit.
The child raises:

`hinge bisection missed the contact window: gap 0.010239924508031835 m at 2.0943951023931953 rad`

That is 10.24 mm at the 120° limit.

Hinge axes that are not parallel to the shaft raise before any solve.
Anatomical flexion axes are not the shaft axis. A Shadow or Allegro chain
cannot call `drive="hinge"` until its axes are expressed as `+Z`, or until
the solver accepts a general axis.

## Dead Ends to Avoid

These were measured. Repeating them costs a full contact cycle.

- Putting the full 40° `club_cross_angle_deg` on the knuckle azimuth.
  The stations span 47.7 mm. `sin(40°) * 47.7 mm` is 30.7 mm of radial
  offset. At the 21.1 mm index seat radius that offset is an azimuth of
  about 55°, and the little leaves the grip. `cross_azimuth_scale` 0.25
  is why the measured span is 10.00°.
- Treating "the first segment on the club axis" as the joint center on the
  surface, with a 40 mm proximal kept tangent. The joint then sits at
  `hypot(seat, length / 2)`, about 29 mm from the axis, 16 mm outside the
  grip. The tangent point is the midpoint of the bone.
- Running the proximal bone along `Z`, or shrinking the station gap by
  `cos(40°)`. Adjacent clearance is only 0.30 mm. A `z` change of that
  size makes the capsules intersect.
- Posing the overlap little on one larger tangent circle. The lead index
  capsule ends reach `hypot(seat, proximal_length / 2)` and stab the outer
  circle by 5 to 8 mm. A single parallel copy still left about 0.96 mm of
  penetration on the cylinder (1.08 mm and 0.54 mm on the taper). The
  committed fix is the parallel copy plus `_parallel_lift`, which steps
  the middle and distal bones out by `penetration_tolerance_m` until the
  sampled neighbour gap is at least 0. That passes the number test and
  fails the picture. The next step is one continuous bone, not a return
  to the outer circle.
- A coarse angular grid. A step near 4.5° skips the 2 mm band.
  `bisect_contact` uses 24 bisections and assumes the gap decreases with
  angle. The single-hinge result stops at 2.0 mm, the outer edge, at 132.32°.
- GB2's maximum-of-samples clearance, and a bone whose rest direction is
  the shaft. Both are described under [Relation to GB2](#relation-to-gb2).
- Using −0.15 m/m as the `mens_standard` radius slope. The stations are
  0.1016 m apart and the diameter drop is 3.05 mm, so the radius slope is
  about −0.015 m/m.
- Citing an unreproduced distal angle as the serial-hinge failure. The
  measured failure is the 10.24 mm gap at 120° above.

## Next Steps

Do these in order. Each one keeps the existing 18 tests green and adds the
acceptance test named here. After each one, run the
[Visual Self-Review Procedure](#visual-self-review-procedure) and write the
new score. A green contact test is not a visual pass.

1. Drive one skinned backend from these angles. The angle keys are
   `{side}_{finger}_{proximal|middle|distal}`, `{side}_{finger}_metacarpal`,
   and `{side}_thumb_{metacarpal|proximal|distal}`. Pass them to that
   backend's `build_hand_visual(grip_spec, side, joint_angles)`.
   Acceptance: the comparison renders at roll 0° and ±20° score Pass on
   "no tubes" and "skin shading smooth" for that backend, the contact
   metrics stay `within_tolerance`, and the hand's long axis stays about
   70% of a frame at least 1024 px. The capsule script will keep failing
   the skin items. Do not retune it until it looks like skin.

2. Keep the flat little clearance on the curled pose, and move the
   metacarpal origin with depth. Acceptance: at depth 0.5 the final little
   grip gap exceeds the index gap by about `just_below_pinky_m` (today it
   collapses to 0.0004 m), the little arch angle stays larger than the
   index angle, and depth 1 places the little metacarpal origin closer to
   the shaft than depth 0 by about `arch_travel_m`. Middle and distal gaps
   stay inside the band.

3. Make the overlap little one continuous finger on the lead index.
   Acceptance: the three trail-little joints meet within 1 mm, every
   trail-little neighbour gap is inside ±0.5 mm, the proximal neighbour
   gap stays inside 2 mm, the grip gap stays above the 2 mm band, and
   `within_tolerance` stays true. `trail_oblique_roll_0.png` shows one
   finger, not disconnected dashes.

4. Honour `trigger_finger` and `joint_coupling_overrides`. Acceptance:
   `trigger_finger` true moves the trail index grip gap above 2 mm and
   sets `trigger_finger_applied` true. A named override changes that
   joint's angle and sets `coupling_overrides_applied` true.

5. After GB2 #15134 has merged, and after its clearance uses the minimum
   sample, compare the two angle sets. Acceptance: `gb2_curl_status`
   reports `available` true and `applied` false, and a test records both
   sets. Swap GB2 angles in only when a test shows those angles inside
   this contact band.

6. Teach the hinge solver a general axis, and make a root that is already
   inside the band keep curling when a child would otherwise miss.
   Acceptance: the two-bone chain in [What Failed](#what-failed) reaches
   a child gap inside the band, and a chain with a non-shaft axis either
   meets the band or still raises `ValueError` with the refusal recorded
   in the test. Today the child raises at 10.24 mm and 120°.

7. Compute palm area from a palm mesh. Acceptance: the cells are not all
   inside or all outside at an intermediate depth, and the depth-1 area
   is still greater than the depth-0 area.

8. Post the plug-in note on #15150, using the next section as the text.
   File a child issue of #15150 if the posing track needs its own tracker.
   Use `Refs`. Leave the epic open until Dieter signs the checklist.

## How Each Backend Plugs in Its Chain

The shared call is `build_hand_visual(grip_spec, side, joint_angles)` on
that backend's branch. `joint_angles` is `HandPose.angles_rad`. Report
contact with `contact_quality_metrics` or, for a backend that measures its
own mesh, `metrics_from_tables`. The appearance shaft is `+Z`. Shared grip
frames that use `+X` have to be converted before the chain is built.
Positive shaft roll is the right-hand rule about butt toward head.

Do not import `_finger` or `_stack_z`. Build a `KinematicChain` of
`JointDescription` rows, or consume `pose_grip` and retarget the angles.
`drive="hinge"` accepts only a shaft-parallel axis. Robot chains whose
flexion axis is not the shaft should retarget the human tangent angles
until step 6 lands.

### MakeHuman Skinned Hand

Pull request #15149, branch `cursor/skinned-human-hand-backend-751f`.
Settings `scripts/config/skinned_human_hand.json` and
`scripts/config/hand_visual_quality.json`. The mesh is the CC0 MakeHuman
hm08 hand, about 115 vertices. Its fingers currently sit off the club.
Retarget `angles_rad` onto the hm08 flexion degrees of freedom. An
along-bone rest pose will not wrap the grip. Put the licence next to the
asset. This poser does not ship the mesh.

### Shadow Hand

Pull request #15143, branch `cursor/shadow-hand-grip-visuals-4437`.
Registry name `"shadow"`. Settings
`scripts/config/grip_shadow_hand_visual.json`. The current evidence splays
the fingers in the air because the pose is not curl-to-contact. Pass
`angles_rad` into `build_hand_visual`. A shaft-parallel hinge chain can
call `pose_chain` with `drive="hinge"` only after the axes are `+Z` in the
appearance frame.

### MyoHand

Pull request #15147, branch `cursor/myohand-visual-backend-dcca`. Settings
`scripts/config/hand_backend_myohand.json`. The model has 23 joints. Map
the four fingers' proximal, middle, and distal angles onto the flexion
degrees of freedom. The current registration is approximate, and the
evidence has no club. The licence is Apache-2.0 and the asset gate is
#13763. Keep that licence beside the asset.

### Procedural Skin

Pull request #15146, branch `cursor/hand-procedural-skin-ff59`. It owns
`hand_backends/capsule.py`, `procedural_skin.py`, `registry.py`, and
`settings.py` on that branch, plus
`scripts/render_hand_backend_comparison.py` and
`scripts/render_hand_backend_roll_evidence.py`. Feed it the same
`GripSpec`, the same `angles_rad`, and the same shaft roll. Report metrics
through `metrics_from_tables` when the skin mesh, rather than the capsule,
is what touches the grip. Expect a merge conflict in
`hand_backends/__init__.py`.

### Allegro Hand

No posing code on this branch. Build a `KinematicChain` with the Allegro
joint origins, axes, and limits in the appearance frame. Use
`drive="hinge"` only when the axis is shaft-parallel. Otherwise retarget
the human tangent angles onto the Allegro flexion joints. Add the registry
entry in the Allegro pull request, not in this one.

### Signed Distance Field

Replace `GripSolid.radius_at` with a signed-distance callback when a mesh
exists. Until then the contact report is a capsule clearance. Sample the
minimum distance along the bone. Do not copy GB2's maximum-of-samples
clearance. `per_segment_gap_m` is the objective the visual backends should
match.

## Dependencies

- This branch is based on `cursor/hrg-gb1b-shaft-roll-2882`, which carries
  `GripSpec`, `resolved_shaft_roll_deg`, and `apply_shaft_roll`.
- GB2 #15134 is unmerged. The adapter records it and does not require it.
- `HandMeshParams` in `grip_hand_mesh.py` is the anthropometry source.
- `get_grip_preset` and `GripGeometry.radius_at` supply the default
  cylinder and the tapered cone.
- Importing the poser pulls in `grip_hand_mesh`, which imports the contact
  topology stack. That stack needs `scipy`. The cloud environment that
  produced this branch used user-site `numpy` 2.5.4, `scipy` 1.18.1,
  `pytest` 9.2.0, `ruff` 0.17.5, and `matplotlib` 3.11.2. Invoke them as
  `python3 -m`.
- Sibling pull requests #15142 (capsule render), #15143 (Shadow), #15146
  (procedural), #15147 (MyoHand), and #15149 (skinned MakeHuman) all sit
  on `cursor/hrg-gb1b-shaft-roll-2882` except #15142 and #15134. They will
  conflict on `hand_backends/__init__.py`. Leave `registry.py` to them.
- `vendor/ud-tools` must stay a clean submodule. A missing checkout raises
  `SeamResolutionError` on import. Initialize it locally. Do not commit a
  submodule pointer change.
- The render script inserts the repository root on `sys.path` because
  running a file does not. `python3 -m scripts.render_shared_hand_pose`
  also works when the working directory is the repository root.

## Visual Self-Review Procedure

Run this after every posing or visual change, before the commit. The unit
tests do not look at the pictures. The render script does not score them.
You have to open the PNG files and score them yourself.

### Render the Standard Views

From the repository root, with `MPLBACKEND=Agg`:

```bash
python3 scripts/render_shared_hand_pose.py --out docs/development/full_body_models/hand_handoff/figures
```

The script writes 28 views at 1024 px and then `contact_sheet.png`.
Names are `{lead|trail|both}_{face|side|oblique}_roll_{0|p20|m20}.png`,
plus `both_oblique_left_roll_0.png`. Face looks along `−Y`. Side looks
along `+X`. Oblique looks from `normalize(1, −1, 0.35)` toward the origin.
The frame fraction in the script is `_FRAME = 0.76`, chosen so the long
axis of these strokes lands near 70% of the PNG after the title. Lead is
`#e0a888`. Trail is `#c4845a`. The shaft is gray, with two edge lines at
the grip radius. Commit the PNGs with the code that produced them.

### Open the PNG Files

Open these, in this order, and look at them. A file listing is not a review.

1. `contact_sheet.png`
2. `both_face_roll_0.png`, `both_side_roll_0.png`, `both_oblique_roll_0.png`
3. `both_oblique_roll_p20.png` and `both_oblique_roll_m20.png`
4. `both_oblique_left_roll_0.png` next to `both_oblique_roll_0.png`
5. `lead_oblique_roll_0.png` and `trail_oblique_roll_0.png`

### Score the Checklist

Score every item Pass, Partial, or Fail. Write one sentence about what you
saw in the PNG. Do not copy the previous score forward.

| Item                               | What a pass looks like                                                                                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Separate fingers and thumb         | Four fingers and a thumb read as separate digits. They do not fuse into one bar.                                                                    |
| Cupped palm                        | A palm surface cups the grip. An arch angle with no surface is a Fail.                                                                              |
| Knuckle line                       | The knuckles form a line across the hand, and the little sits just below the index.                                                                 |
| Fingers wrap the grip              | The fingers curl around the grip. Straight bars that only cross the shaft are a Partial at best.                                                    |
| No tubes, blobs, or torn triangles | The surface is a closed skin. Round-capped strokes are tubes and Fail.                                                                              |
| Skin shading smooth                | Key and fill shading, with no faceted bands and no flat stroke color.                                                                               |
| Hand about 70% of the frame        | The hand fills about 70% of a frame at least 1024 px. Measure the skin bounding box. Thin capsules can pass the long axis and still fail the width. |
| Club visible                       | The shaft is visible in every view you opened.                                                                                                      |
| Lead and trail mirror correctly    | Opposite curl, and the left-handed oblique is the mirror of the right-handed oblique.                                                               |

A skinned backend has to pass the skin rows. This capsule script is
expected to fail them until step 1 replaces the picture. Do not mark those
rows Pass because the contact test is green.

### Score on This Branch

Scored 2026-11-15 by opening `contact_sheet.png`, `both_face_roll_0.png`,
`both_side_roll_0.png`, `both_oblique_roll_0.png`,
`both_oblique_roll_p20.png`, `both_oblique_left_roll_0.png`, and the lead
and trail oblique views. The long-axis fraction below was measured on the
reframed PNGs (skin-colored pixels, frame fraction 0.76).

| Item                               | Score   | What the PNG files show                                                                                                                                                                                                                                 |
| ---------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Separate fingers and thumb         | Partial | Face and side show four stacked bars and a thumb bar. The trail overlap finger in `trail_oblique_roll_0.png` breaks into short dashes.                                                                                                                  |
| Cupped palm                        | Fail    | No palm surface. The arch is only the metacarpal angle in the table. Palm area is 0 at depth 0.5.                                                                                                                                                       |
| Knuckle line                       | Fail    | The bars stack along the shaft. The 10° azimuth does not read as a knuckle line.                                                                                                                                                                        |
| Fingers wrap the grip              | Partial | The lead oblique shows bent segments meeting the shaft. Face and side are straight bars across the shaft. The overlap little does not read as a wrap.                                                                                                   |
| No tubes, blobs, or torn triangles | Fail    | Round-capped strokes. There is no triangle mesh, so there is nothing to tear, and there is no skin.                                                                                                                                                     |
| Skin shading smooth                | Fail    | Flat stroke color. No key light and no fill light.                                                                                                                                                                                                      |
| Hand about 70% of the frame        | Partial | Long axis 72.3% of 1024 px on the both-hands views. Width 31.7% (both) and 45.4% (lead only).                                                                                                                                                           |
| Club visible                       | Pass    | The gray shaft and its two edge lines are in every opened view.                                                                                                                                                                                         |
| Lead and trail mirror correctly    | Partial | The left-handed oblique flips the chevron direction relative to the right-handed oblique. Lead and trail use opposite curl and different colors. The face view of a right-handed pair is not a mirror: both hands extend to the same side of the shaft. |

This score is an agent review of contact diagrams. It is not Dieter's
sign-off, and it is not a pass of the #15150 skin bar.
