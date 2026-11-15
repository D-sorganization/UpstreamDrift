# SDF Hand Handoff

This note is for an agent that has not seen the SDF lane. It is the state of
branch `cursor/sdf-sculpted-hand-c70f` (base `cursor/hrg-gb1b-shaft-roll-2882`).
The hand is a partial, test-backed visual backend. It is not visually accepted
and it does not have Dieter's sign-off.

## Goal and Quality Bar

Epic [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)
asks for a grip hand a person would call a human hand on a golf club. This
lane is the implicit SDF sculpt, tracked as child issue
[#15155](https://github.com/D-sorganization/UpstreamDrift/issues/15155).
Use `Refs #15155` and `Refs #15150`. Do not use Fixes or Closes until the
quality bar is actually met.

The bar, from the epic:

- Renders of the lead hand, the trail hand, and both hands. Three views.
  Shaft roll 0 and ±20 degrees. Solid skin, a key light and a fill light.
  The hand fills about 70 percent of a frame at least 1024 pixels on a side.
  The club is in every view.
- Fingers wrap the grip. Fingertip and middle-phalanx contact stay inside a
  tolerance setting. Penetration past a second tolerance fails. Lead and
  trail mirror.
- Anthropometric ratios stay inside tolerance of the reference settings.
  Hand length is about 0.19 m (wrist to middle fingertip, uncurled). Palm
  width is about 0.085 m. Finger diameters are 14–20 mm and taper.
- A written checklist with Dieter's sign-off before merge.
- The licence sits next to the assets. MANO and SMPL-X are research-only
  and are not used here. Contact physics stay on the existing capsules.
  Every visual backend is driven by the same `GripSpec`, joint angles, and
  shaft roll.

Sibling lanes, do not copy their mesh method: procedural lofts #15146,
shadow #15143, MyoHand #15147, skinned human #15149, capsule fixes #15142.
The shared contract those lanes invented is
`build_hand_visual(grip_spec, side, joint_angles)` in the grip hand frame.
This branch has its own copy of that contract under `hand_backends/`.
Expect a merge conflict in `registry.py` and `__init__.py`. Keep one
contract module when the lanes merge.

## Current State

Package: `src/shared/python/model_appearance/hand_backends/`.

| File                | Role                                                                                                      |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `contract.py`       | `HandRole`, `HandJointAngles.from_grip_spec`                                                              |
| `sdf_settings.py`   | `load_sdf_hand_settings`, frozen `SdfHandSettings`                                                        |
| `sdf_field.py`      | Round cones, spheres, ellipsoids, polynomial smooth-min, `evaluate_hand_sdf`, `analytic_grip_penetration` |
| `sdf_primitives.py` | Primitive builders, directional ellipsoid clearance, skeleton measurements                                |
| `sdf_pose.py`       | `pose_curl_to_contact`, `pose_hand_skeleton`, `phalanx_contact_points`                                    |
| `sdf_mesh.py`       | `marching_cubes_mesh`, `taubin_smooth`, `mesh_is_watertight`                                              |
| `sdf_hand.py`       | `build_hand_visual`, `HandVisualResult.sdf`                                                               |
| `registry.py`       | Registers only `sdf_hand`. Dispatcher takes the backend id first                                          |
| `sdf_render.py`     | Z-buffer, `render_standard_views`, `render_meshes`, `project_hand`                                        |
| `__init__.py`       | Exports the dispatcher                                                                                    |

Settings file: `scripts/config/sdf_hand.json`. Every sculpt number lives
there. The loader rejects a middle-finger chain that is not
`hand_length_m`, a finger radius outside 6–12 mm, a phalanx triple that
cannot wrap, a non-positive blend, and Taubin weights that break
`0 < lambda < -mu`.

Hand frame, same as `grip_hand_mesh`: origin on the grip axis at the hand
centre, +z from butt toward the club head, +x at azimuth 0, +y at azimuth 90. The grip cylinder is `x**2 + y**2 = R**2`. Lead curl decreases azimuth.
Trail curl increases it. A left-handed spec mirrors the finished skeleton
through y. Shaft roll is applied to mesh vertices after marching cubes
(`vertices @ R.T`, `R` from `apply_shaft_roll` about +z). `HandVisualResult.sdf`
inverse-rotates query points with `points @ R` before evaluating the
unrolled field.

Skeleton construction in `pose_curl_to_contact`:

- Uniform scale is `GripSpec.hand_size.length_m / settings.hand_length_m`.
  Breadth is not a separate scale.
- Each finger is a tangent chain on the circle
  `R + pad_clearance + proximal_radius + lift`. `lift` is
  `open_hand_lift_m` when `palm_finger_depth` is 0 and is 0 at
  `full_contact_palm_finger_depth` (0.5, the legacy default).
- A palmar sphere on each phalanx brings the tangent point out to the
  proximal contact radius so the taper does not leave a gap.
- Metacarpals run proximally from the knuckle (`head + wrist_dir * length`) and gather monotonically in `z` toward the carpal arch (`wrist_convergence`), keeping fingers uncrossed across the palm.
- Ellipsoid centres for the palm fill, thenar, hypothenar, thumb web, and
  wrist use directional support (`clear_of_grip_ellipsoid` in `sdf_primitives.py`) with `ellipsoid_support_margin_m`.
- `HandSize.breadth_m` scales palm width relative to `resolve_hand_breadth_m(length_m)`, `club_cross_angle_deg` pivots outer digit MCP azimuths via `cross_angle_mcp_tilt_gain`, and non-`TEN_FINGER` trail styles lift `little` by `overlap_little_lift_m`.
- Groups: hard min across the four fingers, smooth-min into the palm with
  `blend_m.mcp`, smooth-min of the thumb with `blend_m.web`, then crease
  subtraction. Nails are unioned last with `blend_m.nail`.

Default grip radius in the settings is 0.014 m. `build_hand_visual`
rejects a radius outside `GRIP_RADIUS_RANGE_M` (0.008–0.03 m) before it
meshes.

### Tests

```bash
python3 -m pytest tests/unit/model_appearance/hand_backends/test_sdf_hand.py -m unit
```

Nineteen tests, all passing on this branch. Mesh tests call
`pytest.importorskip("skimage")`. Pose, SDF, mirror, contact, and
renderer-math tests run without scikit-image. SciPy is a core dependency
(Taubin). Do not raise `contact_tolerance_m` or `penetration_tolerance_m`
to hide a penetrating sculpt.

What the tests lock:

- `smin(a, a, k) == a - k/6`, and a wide gap equals `min`.
- A round cone is `-radius` on its axis and 0 on the side surface.
- Analytic penetration is the overlap of the hand interior and the
  cylinder interior.
- Blend radii equal the JSON file.
- Middle-finger chord plus three phalanges equals `hand_length_m * scale`
  within 1e-6 m. Palm width equals `palm_width_m * scale`. Middle proximal
  diameter is 20 mm times scale. Distal diameter is 14 mm times scale.
  All four fingers taper inside 14–20 mm.
- `HandSize(length_m=0.22, breadth_m=0.10)` measures 0.22 m.
- Legacy thumb straightness 1: distal axis dot +z is above 0.95.
- Every phalanx contact sample (grip-surface point nearest the bone axis)
  has SDF in `[-penetration_tolerance_m, contact_tolerance_m]`.
- Samples on the cylinder `rho = R - penetration_tolerance_m` stay at or
  outside `-penetration_tolerance_m`.
- `palm_finger_depth` 0 lifts the closed middle-phalanx contact by more
  than half of `open_hand_lift_m`.
- Left-hand SDF at `(x, -y, z)` matches the right-hand SDF at `(x, y, z)`.
- Lead and trail match under a y mirror when both azimuths are 0.
- Bad side, radius, joint-angle type, and resolution raise before the
  mesher runs.
- Registry id `sdf_hand`. An unknown id raises `KeyError` matching
  `unknown`.
- Orthographic span / (2 \* half-extent) equals `frame_fill` (0.70).
  `frame_px` is at least 1024.
- The rasterizer paints a triangle and leaves the background.
- A dark grip-coloured triangle survives the shader.
- The mesh at `test_resolution` (30) is watertight (every edge count is 2)
  and has positive volume.
- A 20 degree shaft roll is exactly `unrolled_vertices @ R.T` with the
  same faces. `result.sdf` matches the unrolled field after `points @ R`.

### Evidence Script

```bash
MPLBACKEND=Agg python3 scripts/render_sdf_hand_evidence.py
```

Optional `--out`, `--frame-px`, `--resolution`. The default directory is
`docs/development/full_body_models/evidence/sdf_hand/`. The default frame
is 1024 and the default marching-cubes resolution is 48. The script prints
one PNG path per line.

Twenty-seven images are already in that directory, plus `NOTICE`. Names:

`{lead|trail|both}_{dorsal|oblique|butt}_roll{+000|+020|-020}.png`

Cameras, all orthographic, up axis +z except the butt view:

- Dorsal: camera on `skeleton.dorsal_hint` (in the xy plane), looking at
  the origin.
- Oblique: `0.75 * dorsal + 0.35 * side - 0.45 * shaft`, with `side`
  perpendicular to dorsal and +z.
- Butt: camera on -z, looking toward the club head, up along the dorsal hint.

The frame is fit to hand vertices only. The grip cylinder uses the hand's
grip radius from the minimum hand z to the maximum. A thinner shaft
(`shaft_radius_m`, `shaft_rgb`) continues past the head end by
`shaft_extension_fraction` of the hand's z span. Skin colour is
`skin_2` from `src/config/appearance_defaults.json` (sRGB 0.93, 0.76, 0.66).
Both-hands places the trail mesh at `z = -hand_spacing_m` (legacy 0.0762 m)
and frames the pair, so each hand is smaller than in a single-hand view.

Generating the sheet on this machine took about 25 seconds.

## What Worked and What Failed

Worked, and locked by tests:

- Polynomial smooth-min, round-cone SDF, and analytic grip penetration.
- Anthropometry measured on the bone chain, not on the curled bounding box.
- Y mirror for handedness, and lead/trail mirror when the azimuths match.
- Shaft roll as a rigid vertex rotation, with SDF queries inverse-rotated.
- Watertight marching-cubes output at resolution 30 (the test) and 48
  (the evidence sheet, about 7800 triangles).
- Opening the hand lifts the finger off the closed contact point.
- The club reads as a dark cylinder in the z-buffer. Matplotlib 3D
  polygons were avoided because they tear.

Failed, then fixed in this branch. Do not revert these two guards:

- The first metacarpal placement pulled wrist bases toward their 3D
  centroid (`wrist_convergence` 0.72). The chords crossed the grip. Little,
  ring, and index metacarpal midpoints landed at rho about 0.001–0.008 m
  on a 0.014 m grip. Gathering is now z-only, and a chord that still points
  inward is flattened until it does not.
- `palm_fill` was 13 mm inside an inset grip sample
  (`rho = R - 0.0015`, SDF about -0.013). The hypothenar ellipsoid was the
  nearest solid at the little-finger contact (gap about -2.5 mm). The thumb
  web centre sat at rho about 0.013 m. Those centres are now clamped by
  `_clear_of_grip`. The clamp is why the palm and the thumb web look like
  lumps standing off the shaft. A closer palm has to be sculpted without
  deleting the clamp, or the inset test goes red.

Still failing the visual bar. Scored by opening the 1024 px sheet, not by
the unit tests:

- `lead_dorsal_roll+000.png`: four fingers are separate and the black grip
  shows between them. The thumb is a radial lump, not a two-segment thumb.
  The palm is one swollen mass. There is no knuckle line. Fingers read as
  tapered tubes. Shading is faceted. The club, including the grey shaft
  past the grip, is visible.
- `trail_dorsal_roll+000.png`: mirror of the lead dorsal. Thumb on the
  opposite side. Same tube fingers and blob palm.
- `lead_butt_roll+000.png`: the grip end is a black disc and the digits
  arch around it. That is the clearest "wrap" view. Still faceted. The
  thumb is a thick pad.
- `both_oblique_roll+000.png`: two hands stacked on one club, curls
  opposing, club visible. Each hand is well under 70 percent of the frame
  because the pair is framed together.

A person would not yet say "that is a human hand holding a golf club."
Creases and nails are in the field. At resolution 48 a crease is about
1.5 mm and a voxel is about 4 mm, so they do not show. `club_cross_angle_deg`,
`HandSize.breadth_m`, and grip style (overlap, interlock, trigger finger)
do not change the mesh. There is no Dieter sign-off. Do not describe this
sheet as accepted.

scikit-image 0.26 on NumPy 2.5 prints a `DeprecationWarning` from
`skimage.measure._marching_cubes_lewiner` (`ar.shape = shape`). The tests
pass. Do not silence it by editing scikit-image.

## Next Steps

Do these in order. After each one, run the visual self-review below and
the unit tests. A green test with a worse picture is a failed step.

1. Smooth the skin. The evidence mesh is faceted at resolution 48 with
   Taubin `lambda` 0.5, `mu` -0.53, 6 iterations. Raise `resolution`
   and/or the Taubin settings in `sdf_hand.json`, then regenerate the
   sheet. Acceptance: `lead_dorsal_roll+000.png` at 1024 px no longer
   shows obvious triangle facets on the fingers, and
   `test_test_resolution_mesh_is_watertight` still passes. Do not switch
   the rasterizer back to a painter's-algorithm 3D plot.

2. Replace the palm lump and the thumb lump. The dorsal palm should read
   as a cupped back of the hand, with a thenar and a hypothenar, and the
   thumb should read as a metacarpal plus two phalanges. Acceptance:
   `test_inset_grip_samples_do_not_penetrate_past_the_tolerance` and
   `test_on_grip_phalanges_meet_the_clearance_band` still pass, and the
   visual checklist marks cupped palm and a separate thumb as yes on
   `lead_oblique_roll+000.png` and `lead_butt_roll+000.png`. If you move
   an eminence toward the grip, keep `_clear_of_grip` or an equivalent
   so the bounding sphere cannot enter `R + pad_clearance`.

3. Make the knuckle line and the flexion creases visible at the evidence
   resolution. The spheres and the crease cones already exist
   (`knuckle_radius_fraction`, `knuckle_dorsal_fraction`,
   `crease_radius_fraction`, `crease_depth_fraction`,
   `crease_span_fraction`). They are smaller than a voxel. Acceptance:
   the dorsal view shows four knuckle bumps in a row, and the oblique
   view shows a crease at the finger joints. Add a unit test that the
   SDF just dorsal of the middle MCP is closer to zero than a sample
   midway between two MCP knuckles, and that the contact tests still pass.
   A crease that tunnels a finger will fail the watertight test. Shallow
   the crease in JSON if that happens.

4. Close the visual gap between the finger pads and the grip without
   new penetration. The analytic gap is already inside
   `contact_tolerance_m` (0.0035 m) and `penetration_tolerance_m`
   (0.0015 m). The dorsal picture still shows black slots between the
   fingers because that camera looks at the back of a wrap. Acceptance:
   `lead_oblique_roll+000.png` shows the palmar side of the middle
   phalanx against the grip, and both contact tests stay green. Smooth-min
   expands the solid by up to about `k/6` where two fields meet
   (`blend_m.joint` is 0.0055 m, `blend_m.web` is 0.011 m). Shrinking a
   blend to win the picture can open a gap the contact test will catch.

5. When `hand_backends/posing.py` defines `pose_hand_skeleton` and it
   returns a `HandSkeleton`, `pose_hand_skeleton` in `sdf_pose.py` uses
   it. Until that file exists, keep `pose_curl_to_contact`. Acceptance
   after the switch: the mirror tests, the anthropometry tests, and the
   contact tests still pass. Do not block step 1 on this.

6. When `hand_backends/render.py` defines `render_grip_views`,
   `render_standard_views` delegates to it. Until that file exists, keep
   `sdf_render.py`. Acceptance: the evidence script still writes the 27
   names above, the club is in every PNG, and
   `test_orthographic_frame_fills_seventy_percent` still passes. Do not
   block step 1 on this.

7. Apply `club_cross_angle_deg` as a palm shear, scale palm width by
   `HandSize.breadth_m` inside the ANSUR breadth/length band, and honour
   overlap / interlock / trigger finger. The capsule helper
   `_solve_topology_finger` in `grip_hand_mesh.py` is the existing
   topology reference. Acceptance: a unit test shows the angle, the
   breadth, and `GripStyle.OVERLAP` each move a measured landmark, and
   the contact tests still pass at the legacy defaults.

8. Stop for a person. Write the checklist score into the pull request.
   Dieter signs it before merge. A self-score of yes is not sign-off.

## Shared Posing and the Renderer

Neither shared module is on this branch. Confirmed by import: there is no
`hand_backends/posing.py` and no `hand_backends/render.py`.

`pose_hand_skeleton` tries
`src.shared.python.model_appearance.hand_backends.posing.pose_hand_skeleton`.
`ImportError`, `TypeError`, or a return that is not a `HandSkeleton` falls
through to `pose_curl_to_contact`. Proceed with the local poser. Keep the
hook. When the shared poser lands, pass the same arguments
(`grip_spec`, `side`, `joint_angles`, `grip_radius_m`, `settings`) and
return a `HandSkeleton` whose `primitives` use the `Primitive` fields in
`sdf_field.py` (`group`, `kind`, `operation`, `shape`, `blend`).

`render_standard_views` tries
`hand_backends.render.render_grip_views(output_dir, frame_px=, resolution=, settings=)`.
`ImportError` or `TypeError` falls through to the local z-buffer. Proceed
with `sdf_render.py`. The local rasterizer is what produced the sheet in
this directory. It is fast enough (about half a second for one 1024 px
frame of an 8000-triangle hand on this machine).

Do not vendor the procedural loft mesher. Do not wait for the other lanes
before the visual steps above.

## Licences and Dependencies

`docs/development/full_body_models/evidence/sdf_hand/NOTICE` records this.
The hand is generated from `scripts/config/sdf_hand.json`. No MANO, SMPL-X,
or other third-party hand mesh is shipped.

| Dependency   | Role                             | Licence                                                                                                                   |
| ------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| NumPy        | Fields, meshes, images           | BSD-3-Clause. Core dependency (`numpy>=2`)                                                                                |
| SciPy        | Taubin smooth (`scipy.sparse`)   | BSD-3-Clause. Core dependency                                                                                             |
| scikit-image | `skimage.measure.marching_cubes` | BSD-3-Clause. Optional extra `sdf-hand` in `pyproject.toml` (`scikit-image>=0.22`). Imported inside `marching_cubes_mesh` |
| Matplotlib   | `plt.imsave` only                | Matplotlib licence (BSD-compatible). Imported inside `_save_png`                                                          |

Skin colour is the repo `skin_2` entry, not a scanned texture. Install the
mesher with `pip install 'scikit-image>=0.22'` or the `sdf-hand` extra.
Missing scikit-image skips the two mesh tests and raises `ImportError` from
`build_hand_visual`, with that install hint.

## Visual Self-Review Procedure

Run this yourself after every change that can affect the picture. Do not
delegate it, and do not treat a previous score, including the one in
"What Worked and What Failed", as the review of your change. Unit tests
do not grade the picture.

1. Regenerate the sheet from the repo root:

   ```bash
   MPLBACKEND=Agg python3 scripts/render_sdf_hand_evidence.py
   ```

   Use the defaults (1024 px, resolution from `sdf_hand.json`) for the
   score. A smaller `--frame-px` is only a smoke test.

2. Open the PNGs. Read the files. A directory listing is not a review.
   At least these six, and the matching trail views:

   - `docs/development/full_body_models/evidence/sdf_hand/lead_dorsal_roll+000.png`
   - `docs/development/full_body_models/evidence/sdf_hand/lead_oblique_roll+000.png`
   - `docs/development/full_body_models/evidence/sdf_hand/lead_butt_roll+000.png`
   - `docs/development/full_body_models/evidence/sdf_hand/trail_dorsal_roll+000.png`
   - `docs/development/full_body_models/evidence/sdf_hand/trail_oblique_roll+000.png`
   - `docs/development/full_body_models/evidence/sdf_hand/trail_butt_roll+000.png`
   - `docs/development/full_body_models/evidence/sdf_hand/both_oblique_roll+000.png`

   Also open one `roll+020` and one `roll-020` dorsal and confirm the hand
   has rotated about the shaft relative to `roll+000`.

3. Score every item yes or no. Write the score next to the image name.
   Partial is a no.

   - Separate fingers and thumb
   - Cupped palm
   - Knuckle line
   - Fingers wrap the grip
   - No tubes, blobs or torn triangles
   - Skin shading smooth
   - The hand about 70 percent of the frame
   - Club visible
   - Lead and trail mirror correctly

4. Score "the hand about 70 percent of the frame" on the lead and trail
   images. On `both_*` images, score club visibility and the mirror, and
   say that the pair shares one frame.

5. If any item that your change was supposed to fix is still no, the
   change is not done. Fix it and start again at step 1. Do not weaken
   `contact_tolerance_m`, `penetration_tolerance_m`, or the watertight
   test to keep a picture.

6. When every item is yes on the lead and trail sheets, stop and ask for
   Dieter's sign-off. Your yes is not that sign-off.
