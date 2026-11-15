# Allegro Hand Backend Handoff

Wind-down note for the next agent. This branch does not implement the backend.
It records what was verified on 2026-11-15 so the work does not start over.

Issue: [#15152](https://github.com/D-sorganization/UpstreamDrift/issues/15152)
(sub-issue of epic [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)).
Use `Refs #15152` in the implementation PR. Do not use `Fixes` or `Closes`
until the quality bar below is actually met. Do not arm auto-merge. Do not
merge. Do not push to this branch or to any other agent's branch.

## Goal and Quality Bar

Build a Wonik Allegro visual backend beside the other hand options. There is
no winner. Contact physics stays on the procedural capsules.

Epic: [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)
(child of grip epic #15122). Read the epic again before coding; its body is
the quality bar.

Target API, which is not in the tree yet:

- Package `src/shared/python/model_appearance/hand_backends/`.
- `allegro.py` exposes `build_hand_visual(grip_spec, side, joint_angles)`.
- The returned mesh is in the hand-grip frame (`grip_hands.grip_hand_frame`:
  origin on the grip axis, `z` butt toward head).
- Registry entry `allegro`.
- Allegro has four fingers and 16 DOF. The little finger is absent. Document
  that. Do not invent a fifth digit. The backend is for 4-finger contact and
  for later per-finger clamp controllers with force limits.
- Humanize what is reasonable: a skin-tone material, scale to the
  anthropometric `GripSpec.hand_size`, and hide hardware that is not the hand.
- Pose by curling joints until the fingers meet the grip. Do not push
  vertices.
- Renders: lead, trail, and both hands; three views; shaft roll 0° and ±20°.
  Solid smooth skin shading, key light and fill light, the hand filling about
  70% of a frame of at least 1024 px, and the club in every view.
- Tests: contact tolerance, penetration, mirroring, and shaft roll. Tolerances
  live in settings, not as literals in Python.

Quality bar copied from the epic:

1. The renders above.
2. Fingertip and middle-phalanx contact within a tolerance setting, no
   penetration beyond a tolerance setting, and left/right mirroring holds.
3. Anthropometric ratios within tolerance of the reference (settings).
4. A written human checklist in the PR. Dieter signs off before merge. The
   question is whether it reads as a hand holding a club: separate fingers and
   thumb, a cupped palm, a knuckle line, and no tubes or blobs.
5. The licence recorded in a NOTICE next to the assets.

Sibling lanes already reported two failures the Allegro work must not repeat
(comments on #15150, 2026-11-15):

- Skinned human (#15149): cropping a hand on an ad-hoc band, and moving palm
  vertices radially to fake a clamp, tore the mesh. Pose by joint rotation.
- Procedural skin (#15146): finger tubes about 4–6 mm across, no palm, fingers
  pointing at random, and a frame that is about 85% empty. Write the
  anthropometric test and the knuckle-line test before any shading work.
  Target, from that lane: finger diameters 13–20 mm, palm about 0.085 m by
  0.10 m, hand length 0.18–0.20 m.

## Current State

No Allegro backend, registry, settings file, test, or evidence PNG exists.
`src/shared/python/model_appearance/hand_backends/` is not on this branch.
`grip_hands.grip_hand_meshes` still bakes the procedural capsule hand.
Nothing in `src/` or `tests/` changed for this lane.

The hand model is already vendored and already used as a MuJoCo contact
fixture (not a qualified golfer) by
`src/engines/physics_engines/mujoco/python/grip_lab_adapter.py`.

Canonical asset directory:

`src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/hand_assets/wonik_allegro/`

Files there: `LICENSE`, `README.md`, `CHANGELOG.md`, `left_hand.xml`,
`right_hand.xml`, `scene_left.xml`, `scene_right.xml`, and `assets/*.stl`.
Do not copy the meshes to a second directory. Add `NOTICE` and
`PROVENANCE.json` beside `LICENSE`.

Binding `src/config/grip_bindings/mujoco_allegro_hand_lab.json` maps
`trail_little_finger` to `rh_rf_distal`. That is a 5-site contact alias for
the lab fixture. It is not a fifth finger. The visual backend must not draw
one.

### Kinematic Facts to Build On

Verified by loading `right_hand.xml` in MuJoCo 3.17 (`MUJOCO_GL=disable`).
`nq` is 16. Visual geoms are mesh group 2. Collision boxes and capsules are
group 3 and must not be drawn.

Finger bodies, in curl order: `ff_*` index, `mf_*` middle, `rf_*` ring,
`th_*` thumb. Each chain is base, proximal, medial, distal, tip. There is no
`lf_*` body.

Joints and ranges (radians). Actuator names are in parentheses. Each actuator
is a `<position>` whose `ctrlrange` equals the joint range. A future clamp
controller can command these sixteen actuators. This visual task does not
apply forces.

| Joint | Actuator | Range (rad) | Axis in the zero pose |
| --- | --- | --- | --- |
| `ffj0` | `ffa0` | −0.47 to 0.47 | about +X (abduction) |
| `ffj1` | `ffa1` | −0.196 to 1.61 | about −Y (flexion) |
| `ffj2` | `ffa2` | −0.174 to 1.709 | about −Y |
| `ffj3` | `ffa3` | −0.227 to 1.618 | about −Y |
| `mfj0` | `mfa0` | −0.47 to 0.47 | about +X |
| `mfj1` | `mfa1` | −0.196 to 1.61 | about −Y |
| `mfj2` | `mfa2` | −0.174 to 1.709 | about −Y |
| `mfj3` | `mfa3` | −0.227 to 1.618 | about −Y |
| `rfj0` | `rfa0` | −0.47 to 0.47 | about +X |
| `rfj1` | `rfa1` | −0.196 to 1.61 | about −Y |
| `rfj2` | `rfa2` | −0.174 to 1.709 | about −Y |
| `rfj3` | `rfa3` | −0.227 to 1.618 | about −Y |
| `thj0` | `tha0` | 0.263 to 1.396 | about −X |
| `thj1` | `tha1` | −0.105 to 1.163 | about −Y |
| `thj2` | `tha2` | −0.189 to 1.644 | about +Z |
| `thj3` | `tha3` | −0.162 to 1.719 | about +Z |

`thj0` does not include 0. A zero `qpos` is outside that joint. Initialize
every joint inside `jnt_range` (the lower end is the open pose).

At the open pose (`thj0` at 0.263, other joints at their lower limits), tip
body positions in the model frame are about:

- `ff_tip` (0.107, −0.053, 0)
- `mf_tip` (0.110, 0, 0)
- `rf_tip` (0.107, 0.053, 0)
- `th_tip` (−0.082, −0.121, 0.015)

Adding 0.8 rad on `ffj1`–`ffj3`, `mfj1`–`mfj3`, and `rfj1`–`rfj3` moves each
fingertip by about (−0.056, 0, +0.077). Positive flexion curls the fingers
from +X toward +Z. The finger row lies along Y. A mount that maps model Y
onto the grip shaft (`z` in `grip_hand_frame`) is the family that lets that
curl wrap the grip. `Rx(±90°)` does that mapping. It is not a finished mount.

Palm visual bounds at that open pose: x from −0.095 to 0.002, y from −0.057
to 0.057, z from −0.030 to 0.011. The whole visual cloud spans about
0.246 m by 0.241 m by 0.067 m. Fingertip meshes sit near x = 0.12 to 0.15.

`left_hand.xml` uses `base_link_left.stl` and `link_12.0_left.stl` and the
same sixteen joint names. Load the single-hand XML. Do not load the bimanual
lab XML from `grip_lab_adapter`, which prefixes every name with `lh_` or
`rh_`.

Right-handed golfer: lead hand is anatomical left (`left_hand.xml`), trail
hand is anatomical right (`right_hand.xml`). Left-handed golfer: swap. That
matches golf, and it matches the Shadow lane's left/right split.

`link_4.0.stl` is in `assets/` and is not referenced by either hand XML. It
is not a little finger. Do not attach it.

### Settings to Introduce

Put every tolerance and gain in `scripts/config/hand_backend_allegro.json`.
Suggested keys, values to be tuned by the tests rather than copied as magic
numbers into Python:

- `contact_tolerance_m`, `penetration_tolerance_m`,
  `finger_finger_penetration_tolerance_m`
- `grip_radius_nominal_m` (the iron grip is about 0.012 m; read it from the
  club when the caller passes one)
- `curl_bisect_iterations`, `curl_joint_fractions` for proximal, medial, and
  distal (do not use one scalar for every joint, including the thumb)
- `mount_rotation` and `mount_translation_m` once a search actually meets
  the contact test
- `skin_tone_rgba` and an optional `nail_tone_rgba` for the tip meshes
- `hidden_geom_group` = 3 (collision). `hidden_mesh_names` should include
  nothing that is a finger. Do not list `base_link`: it is the palm. Allegro
  has no forearm mesh to hide. The palm shell includes the mounting flange;
  keep it and say so.
- `anthropometry`: reference length and breadth measured from the extended
  pose, plus the acceptance band 0.18–0.20 m length, palm about 0.085 m by
  0.10 m, finger diameter 13–20 mm
- `little_finger`: the string `absent`, so the documentation is data

Scale about the grip axis after the curl solve, or curl against
`grip_radius / length_scale`, then scale. Scaling after a solve that targeted
the real radius breaks contact. Breadth scales along the shaft. Length scales
in the radial plane. Read the target from `GripSpec.hand_size`.

### Tests and Evidence, Once They Exist

There are no tests and no evidence script yet. Create:

- `tests/unit/model_appearance/hand_backends/test_allegro_visual_backend.py`
- `docs/development/full_body_models/evidence/hand_backend_allegro/render_allegro_hand_evidence.py`

Run the tests with:

```bash
MUJOCO_GL=disable MPLBACKEND=Agg python3 -m pytest \
  tests/unit/model_appearance/hand_backends/test_allegro_visual_backend.py \
  -m unit --timeout=120
```

`mujoco` is not installed in the cloud image used for this note. The probe
used `python3 -m pip install --user 'mujoco>=3.6,<4'` and got MuJoCo 3.17.
That same install upgraded user-site NumPy to 2.5.4. Prefer the repo extra
`mujoco` over a bare user install. Skip the tests with
`pytest.importorskip("mujoco")` when the package is absent.

Regenerate evidence with:

```bash
MUJOCO_GL=disable MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen \
  python3 docs/development/full_body_models/evidence/hand_backend_allegro/render_allegro_hand_evidence.py
```

That script does not exist yet. Writing it is a next step, not a command
that works today.

### Contract to Match

Shadow PR #15143 (`cursor/shadow-hand-grip-visuals-4437` at
`b347adc2380e5db30db944e26ae58705ec3550d6`) is the closest analogue: a
Menagerie robot hand, skin tint, hidden hardware, and curl-to-contact inside
the backend. Copy that shape:

- `types.py`: `HandSide = "lead" | "trail"`, `JointAngles` as a mapping, an
  array, or `None`, and `HandVisualResult(mesh, material_rgba)`.
- Module function `build_hand_visual(grip_spec, side, joint_angles, *, grip_radius_m=None)`.
- `registry.py`: `register_hand_backend`, `list_hand_backends`, and
  `build_hand_visual(backend, grip_spec, side, joint_angles, **kwargs)`.
- Register `"allegro"`. Call it by name from the evidence script. Do not
  retarget `grip_hands.py` to Allegro. Shadow did retarget it; that edit
  will conflict, and the capsule hand must keep working.

Other lanes each ship their own `registry.py`. Re-read them before copying.
Heads moved during this session:

- Procedural #15146, `cursor/hand-procedural-skin-ff59` at
  `d8492c235f47acee7958dc8afa266b1c5f956d18`. Has `metrics.py`, `preview.py`,
  `placement.py`. The comparison harness discovers every registered backend.
- MyoHand #15147, `cursor/myohand-visual-backend-dcca` at
  `7a83d231181c9b5194d0170ea67ce5cb24f9f6fb`. Has `mujoco_extract.py` and
  `render_evidence.py`. Useful for pulling group-2 meshes out of an `MjModel`.
- Skinned human #15149, `cursor/skinned-human-hand-backend-751f` at
  `acce425175bd4e9dd9995902620ce6bff8dda1ee`.
- Capsule render fixes #15142, `cursor/grip-hand-render-mesh-fix-8498` at
  `1297671eeb68dd3b58a99aea950c3a18ec536acc`. Lambert shading and equal-aspect
  crops live on that branch as `grip_preview_matplotlib.py`. They are not on
  this branch. That helper frames the whole club, so the hand is small, and
  its figure is under 1024 px. Do not copy its framing.

`posing.py` and a shared `render.py` are not on any of those branches. See
the section below.

Shaft roll is already on this branch (`grip_spec.shaft_roll`,
`appearance_frame_with_shaft_roll` inside `grip_hand_frame`). Solve the curl
in the unrolled hand frame, then rotate vertices about appearance `z`
(`APPEARANCE_GRIP_SHAFT_AXIS`). Radial contact is unchanged by that rotation.
When placing the mesh on a club, build `grip_hand_frame` from a pose whose
shaft-roll fields are 0, because `grip_hand_frame` applies roll itself.
Applying both doubles the roll.

## What Worked

- The vendored Allegro package loads in MuJoCo 3.17. Sixteen joints, four
  fingers, visual meshes on group 2.
- `link_3.0.stl` is byte-identical to
  `google-deepmind/mujoco_menagerie` `main` `wonik_allegro/assets/link_3.0.stl`
  (sha256 `fe333088176395aa7d8a913aa953d3cd649c4c77753224239b78b2c99e3e314c`).
  Every STL in the directory has the same byte size as the upstream contents
  listing. Only `link_3.0.stl` was compared byte for byte.
- The package `LICENSE` sha256 is
  `09c187439ecece8b00330e71eb988d3d30f29373c580dc31ab54200d0aa4deb4`, the same
  file as upstream Menagerie `wonik_allegro/LICENSE`.
- Positive finger flexion curls from +X toward +Z, so a shaft-aligned mount
  is a rotation of the model Y axis onto grip `z`, not a mystery.
- Issue #15152 is open and parented under #15150.
- Sibling learnings on #15150 are specific enough to write the first tests
  (anthropometry and knuckle line) before any shading.

## What Failed

No backend was written. There is no red test, no green test, and no PNG.
Do not describe this branch as a visual result.

A coarse mount search did not achieve a grip. It tried `Rx(−90°)`,
`Rx(+90°)`, `Ry(90°)`, and `Rz(90°)` then `Rx(−90°)`. For each, it curled
every finger joint and the thumb with one scalar `t` (eight samples from
0.15 to 0.85) and tried four shaft distances (0.02, 0.03, 0.04, 0.05 m)
from the middle-finger proximal centroid along the curl direction. Best
scores, as `(score, tip gap m, medial gap m, penetration m, distance m, t)`:

- `Ry(90°)`: (0.0699, 0.0213, 0.0010, 0.0095, 0.05, 0.15). The middle
  phalanx can get close while the tips stay 21 mm off the surface and some
  vertex penetrates by 9.5 mm.
- The other three rotations: (0.0803, 0.0508, 0.0296, −0.0055, 0.02, 0.15).
  Negative penetration means the cloud stayed outside the cylinder. Tips
  were about 51 mm away.

That search is not evidence that curl-to-contact is impossible. It used one
curl scalar for the thumb and the fingers, a single translation heuristic,
and a 15th-percentile radius. The next search should binary-search proximal,
medial, and distal separately, search the thumb on its own, and score the
actual tip meshes (`ff_tip`, `mf_tip`, `rf_tip`, `th_tip`) and the medial
meshes (`ff_medial`, `mf_medial`, `rf_medial`).

The cloud image did not have MuJoCo, pytest, SciPy, or Matplotlib. Installing
them with `pip install --user` also upgraded NumPy. That is an environment
side effect, not a repository change.

Shared `posing.py` and `render.py` had not landed by the end of this run.
Issue comments from this token work (the child issue was created here). Other
cloud lanes reported HTTP 403 on issue comments. Put a `## Learnings` section
in the PR body either way, so it can be relayed onto #15150.

## Next Steps

Do these in order. Each step is done only when its test passes. Write the
test first. Keep going down the list only after the previous test is green.
Re-read #15150 and the sibling PRs before step 1; their heads moved during
this session.

1. **Licence and DOF gate.** Add `NOTICE` and `PROVENANCE.json` next to
   `wonik_allegro/LICENSE`. The test reads the NOTICE, checks the licence
   string `BSD-2-Clause`, checks the sha256 of `assets/link_3.0.stl` against
   the provenance record, asserts `model.nq == 16`, asserts the joint names
   are exactly the sixteen in the table above, and asserts no joint or body
   name contains `little` or `lf`. Acceptance: that test passes with
   `MUJOCO_GL=disable` and no network.

2. **Anthropometry before styling.** From `GripSpec.legacy().hand_size`, the
   extended-pose hand length is inside 0.18–0.20 m, the palm bounds are
   within the settings band of 0.085 m by 0.10 m, and a cross-section of
   `link_1.0` (proximal) has diameter 13–20 mm. Acceptance: the test fails
   if the scale is left at 1 and the native span (about 0.25 m) is outside
   that band, and it passes after scaling from `hand_size`. A second case
   with `length_m` 0.15 versus 0.23 changes the mesh extent by more than 8%.

3. **Knuckle line.** On the open pose, the index, middle, and ring base
   bodies are ordered along the shaft, and the thumb base is not on that
   line. Acceptance: the signed shaft coordinate increases index to middle
   to ring (or the reverse, documented), and the thumb's distance to that
   line exceeds a settings threshold.

4. **Curl-to-contact.** For lead and trail, the 15th percentile of radial
   distance of each of `ff_tip`, `mf_tip`, `rf_tip`, and `th_tip`, and of
   each `*_medial` finger mesh, lies within `contact_tolerance_m` of the
   grip radius. The little-finger metric is the string `absent`, not a
   number. Acceptance: the test fails on the open pose and passes on the
   solved pose. Store the mount that achieves this in the JSON settings.

5. **Penetration.** Every visual vertex has radial distance at least
   `grip_radius_m - penetration_tolerance_m`. The minimum distance between
   finger clouds is at least `-finger_finger_penetration_tolerance_m`.
   Acceptance: both asserts use the JSON values, and a deliberately
   over-curled `qpos` fails the test (so the test can see penetration).

6. **Mirroring.** Right-handed lead loads `left_hand.xml` and trail loads
   `right_hand.xml`. Left-handed swaps. Lead and trail vertex counts match
   and the vertex arrays are not allclose. Reflecting the lead mesh through
   `y = 0` moves the thumb to the other side of the finger row.
   Acceptance: those four asserts.

7. **Shaft roll.** `shaft_roll_deg` +20 and −20 rotate the mesh about `z`
   by that angle relative to roll 0. A point on the axis is unchanged.
   Radial contact gaps from step 4 stay inside the same tolerance.
   Acceptance: the rotation residual is under a settings epsilon (start at
   1e-6 if roll is applied as a pure rotation after the solve).

8. **Joint override.** `build_hand_visual(..., {"ffj1": 0.2})` moves the
   index proximal vertices by more than 5 mm relative to `{"ffj1": 1.2}`.
   Acceptance: that displacement test.

9. **Registry.** `"allegro"` is listed, and
   `build_hand_visual("allegro", spec, "trail", None)` returns a finite mesh
   and the skin RGBA from the JSON file. Acceptance: the test does not
   import a private helper to see the colour.

10. **Evidence and the visual self-review.** Implement the script in the
    current-state section. Then run the procedure in the last section of
    this note, yourself, on the PNGs. Acceptance: `receipt.json` records
    width and height at least 1024, a measured hand-fill fraction between
    0.60 and 0.80 for every single-hand view, and `club_in_frame: true` for
    every view. The written checklist in the PR matches what you saw. An
    honest fail on "reads as a human hand" is a pass of this step if the
    other items are scored truthfully. Allegro is a robot hand; skin tint
    will not make it a human hand. Say that.

11. **PR hygiene.** One PR, based on `cursor/hrg-gb1b-shaft-roll-2882`.
    `Refs #15152` and `Refs #15150`. A `## Learnings` section: what worked,
    what failed, what other lanes should copy. A `## Handoff` section that
    links this file. Do not arm auto-merge. Dieter's checklist sign-off is
    still required before merge.

## Shared Posing and Renderer

Epic #15150 lists two lanes that had not published files by the end of this
run:

- Shared curl-to-contact posing, cloud lane `bc-c9c36777`. No
  `hand_backends/posing.py` on the Shadow, procedural, MyoHand, or skinned
  branches.
- Shared renderer, cloud lane `bc-59b6630a`. No shared `hand_backends/render.py`
  on those branches. MyoHand has its own `render_evidence.py`. Procedural has
  `preview.py`.

Before writing the solver or the evidence script, look again:

```bash
git fetch origin cursor/hand-procedural-skin-ff59 \
  cursor/myohand-visual-backend-dcca \
  cursor/shadow-hand-grip-visuals-4437 \
  cursor/skinned-human-hand-backend-751f
git grep -n "def pose_hand\|def render_hand" \
  $(git rev-parse origin/cursor/hand-procedural-skin-ff59) \
  -- src/shared/python/model_appearance/hand_backends || true
```

If `posing.py` exists and its function accepts a grip spec, a side, and a
joint-range model, call it from `allegro.py` instead of a private solver.
If it does not know Allegro's sixteen joint names, keep the Allegro curl
loop local and pass it the shared contact metric only.

If `render.py` exists and can draw a mesh with a key light, a fill light, a
1024 px frame, and a grip cylinder, use it. If it frames the full club the
way #15142 does, do not use its camera. The hand has to fill about 70% of
the frame while the grip the hand is holding stays visible. A full 0.95 m
iron makes the hand a speck. Draw the grip tube plus a short shaft stub
inside the crop. Record that the club head is outside the crop if it is.

Until those modules exist, implement curl inside `allegro.py` the way
`shadow.py` does (`solve_shadow_hand_qpos` on #15143): forward kinematics,
binary search on each flexion joint, stop when the tip and the medial mesh
are within `contact_tolerance_m` of the cylinder and not inside
`penetration_tolerance_m`. Implement the evidence script locally. Delete the
private copies when the shared modules land and the same tests still pass.

## Licences and Assets

The meshes ship in-tree. Do not download a second copy.

Directory `hand_assets/wonik_allegro/`:

- `LICENSE` is BSD-2-Clause, Copyright 2016 SimLab. Menagerie
  `wonik_allegro/README.md` says the models are released under that licence.
  The local file matches upstream Menagerie byte for byte (sha256 above).
- `README.md` traces the MJCF to the public SimLab URDF at
  `simlabrobotics/allegro_hand_ros` (`allegro_hand_description`). That
  repository has no top-level `LICENSE` file (a fetch of `/LICENSE` returned
  404). The licence next to the bytes we ship is the Menagerie BSD-2-Clause
  file.
- STL headers are not a licence. Most of the 80-byte headers are blank.
  `base_link_left.stl`, `link_0.0.stl`, `link_12.0_left.stl`, and
  `link_12.0_right.stl` start with `STLB ASM 218.00.00.0000` (a CAD assembly
  id). No header contains the words copyright or licence.
- `link_3.0.stl` matches upstream Menagerie. The other STL sizes match the
  upstream listing. A later agent should hash the rest once and store the
  digests in `PROVENANCE.json`. Digests of the local files, sha256:

| File | Bytes | sha256 |
| --- | --- | --- |
| `base_link.stl` | 163884 | `4060e1ae52bca21367e15a1dfab16bc7c2e81849566631b7f55e2cf376b7bc85` |
| `base_link_left.stl` | 433984 | `9f66c1cb8d27b9f07e49101a7997a49c60d3cc3ccbc8de96065322df68f47a0b` |
| `link_0.0.stl` | 57584 | `32b90a6744d472a6ffaee02cbf32eb8a5600b1da0944e86aeb9f53503a9f8ac9` |
| `link_1.0.stl` | 70084 | `e3dfb35b6682f62f18ca6ed0b39cb94ff627b68e0e3373cf7212390574604680` |
| `link_12.0_left.stl` | 217084 | `1749ba567bf36729a81eaecf384bc4f5483be8c496af54d230500b96f1d81dcc` |
| `link_12.0_right.stl` | 204284 | `c5418fc127311eefee1d8ee69dfcfde2157d038286900bee91608d2701b00f13` |
| `link_13.0.stl` | 22484 | `e906896afc078a12c9c377e3a2b07f375583d3b869581f9962030cabc888be29` |
| `link_14.0.stl` | 46884 | `a9e4983e4bff7fc7d868b4965c9865edb2669b946434095703766202b5de01ff` |
| `link_15.0.stl` | 43284 | `bf963adb775b79b27649f4f5dff4ced58ef676ce86cc677de49d3810382d11e0` |
| `link_15.0_tip.stl` | 71984 | `a7147c765d4b6d2eb16fa14bba2e8eceec4a750bc346260893ca02aca1fd7a63` |
| `link_2.0.stl` | 52884 | `9c98477c65b45f5a3de1df0e1b54281da2385f4eafd057d103ab40893771c2d0` |
| `link_3.0.stl` | 18084 | `fe333088176395aa7d8a913aa953d3cd649c4c77753224239b78b2c99e3e314c` |
| `link_3.0_tip.stl` | 71984 | `9c3fa04fea0dfa1257c8a1b02d2b7d982b0d985c1ddda1edf148827944835f39` |
| `link_4.0.stl` | 22884 | `f80ac4d443169c56507afaa0b3c72ec3679312bc80cbaebd03e73e96dd18c408` |

`link_4.0.stl` is unused by both MJCF files.

Upstream model page:
`https://github.com/google-deepmind/mujoco_menagerie/tree/main/wonik_allegro`

Wonik product page, named by the README:
`https://www.wonikrobotics.com/research-robot-hand`

MANO and SMPL-X are not used and must not be vendored.

NOTICE text to add, next to `LICENSE`:

```text
Wonik Allegro Hand V3 visual meshes and MJCF
Copyright 2016 SimLab. Licensed under the BSD 2-Clause License.
The licence file in this directory is the Menagerie package licence.
Individual STL headers contain no licence text. Some headers carry the
CAD id "STLB ASM 218.00.00.0000".
link_4.0.stl is not referenced by left_hand.xml or right_hand.xml.
The kinematic tree has four fingers and 16 DOF. There is no little finger.
Visual only. Grip contact in UpstreamDrift stays on procedural capsules.
```

## Visual Self-Review Procedure

Run this after every change that touches the mesh, the mount, the curl, the
scale, the material, or the camera. Metrics in `receipt.json` do not replace
looking at the pictures.

1. Render. From the repo root:

```bash
MUJOCO_GL=disable MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen \
  python3 docs/development/full_body_models/evidence/hand_backend_allegro/render_allegro_hand_evidence.py
```

The script must write 27 PNGs: lead, trail, and both; views side, top, and
butt; shaft roll −20°, 0°, and +20°. It must also write `receipt.json` with
the pixel size, the hand-fill fraction, and whether the grip cylinder is
inside the crop. Minimum frame is 1024 by 1024. Solid shading with a key
light and a fill light. No wireframe as the only shading.

2. Open the PNGs. Use an image reader on the files, not a thumbnail and not
   the receipt alone. At least the nine images at roll 0° (three roles, three
   views) and one ±20° pair for lead and for trail. Open a failed view from
   the previous commit as well when you are checking a fix.

3. Score every opened image. Write the score in the PR. Each item is pass,
   fail, or not applicable. A fail blocks the change unless the PR says why
   the fail is the honest robot-hand limit and names the next edit.

- Separate fingers and thumb. Index, middle, ring, and thumb are four
  distinguishable parts. No extra little finger.
- Cupped palm. The palm faces the grip. A flat robot palm is a fail of
  "human palm", and it must be labeled that way. It is still a pass of
  "palm present" if the palm shell is on the grip rather than floating.
- Knuckle line. The three finger bases line up along the shaft.
- Fingers wrap the grip. Tips and middle links meet the cylinder. An open
  hand is a fail.
- No tubes, blobs, or torn triangles. Look for spikes, holes, and flipped
  black faces. Allegro's own link seams are acceptable. A torn palm slab is
  not.
- Skin shading is smooth. Banding, a single flat colour with no light
  direction, or a black robot shader left in place is a fail.
- The hand fills about 70% of the frame. Under half the frame, or a crop
  that cuts off the fingertips, is a fail.
- The club is visible. The grip cylinder is in frame in every view. A hand
  floating in empty space is a fail.
- Lead and trail mirror correctly. The trail hand is the other Allegro
  mesh, on the other side of the grip or stacked along the shaft, not a
  copy of the lead mesh in the same place. Shaft roll ±20° turns the hand
  around the grip and does not translate it off the shaft.

4. If any required item fails, fix the cause and start again at step 1.
   Do not add a tolerance or a crop to hide the failure.

5. Stop claiming success when the pictures disagree with the tests. A green
   penetration test with a picture that shows an open hand means the metric
   is measuring the wrong vertices.
