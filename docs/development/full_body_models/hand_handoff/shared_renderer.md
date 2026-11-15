# Shared Hand Renderer Handoff

This file is the handoff for the shared grip-hand renderer. No renderer
code landed in this run. Read this file before writing any of it.

Epic: [Human Hand Model on the Club](https://github.com/D-sorganization/UpstreamDrift/issues/15150)
(parent grip epic [#15122](https://github.com/D-sorganization/UpstreamDrift/issues/15122)).

## Goal and Quality Bar

Build `src/shared/python/model_appearance/hand_backends/render.py` plus a
CLI that any registered hand backend can call. The picture has to be good
enough for a human to judge the hand. Earlier lane evidence is not: tiny
matplotlib wireframes, washed out, hands filling about 10% of the frame.

Epic [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)
quality bar, which every visual PR still has to meet:

1. Renders of the lead hand, the trail hand, and both hands, from 3 views,
   at shaft roll 0° and ±20°. Solid smooth skin shading, key and fill
   lights, the hand filling about 70% of a frame of at least 1024 px, and
   the club in every view.
2. Fingers wrap the grip. Fingertip and middle-phalanx contact stay inside
   a tolerance setting. Penetration stays inside a tolerance setting.
   Left and right mirroring holds.
3. Anthropometric ratios stay inside the reference tolerance (settings).
4. A written human checklist in the PR. Dieter signs off before merge:
   the picture reads as a hand holding a club.
5. Licence recorded next to any asset (`NOTICE`). MANO and SMPL-X stay
   research-only.

The renderer track adds these views on top of the epic's three: palm,
back, thumb side, pinky side, butt-end down the shaft, and the golfer's
eye view. It also needs a deterministic camera rig loaded from settings,
a contact sheet across registry backends at shaft roll −20°, 0°, and +20°,
golden-image tests with a perceptual tolerance, and a smoke test that
runs headless.

Engine order, fixed by the task:

1. MuJoCo offscreen (`MUJOCO_GL=osmesa`, then `MUJOCO_GL=egl`).
2. pyrender with trimesh, only when MuJoCo cannot render.
3. A matplotlib fallback only when neither of those can render. The
   fallback must still auto-frame the hand. A receipt that says
   `engine=matplotlib` is not quality-bar evidence.

Contact physics stays on the capsule grip. This renderer is visual only.

Parallel tracks, no winner. Do not delete or rewrite another lane's
backend to make the sheet look better.

## Current State

Branch `cursor/hrg-shared-hand-renderer-cec5` is cut from
`cursor/hrg-gb1b-shaft-roll-2882` at `c00b28590` (shaft roll per hand,
[#15138](https://github.com/D-sorganization/UpstreamDrift/issues/15138)).
`main` does not have that shaft-roll API. Keep this stack until that
branch is on `main`.

### Files That Exist

There is no `hand_backends` package on this branch. These are the pieces
the renderer must call, and they already exist:

| Piece | Path | What it gives the renderer |
| --- | --- | --- |
| Capsule hand | `src/shared/python/model_appearance/grip_hand_mesh.py` | `build_grip_hand_mesh(hand, *, grip_radius_m, pose)` returns `GripHandMesh` with `.mesh`, `.components`, `.grip_radius_m` |
| Hand-grip frame | same module, module docstring | Origin on the grip axis at the hand centre. `z` runs butt to head. `x` is azimuth 0. `y` is azimuth 90°. The grip cylinder is `x**2 + y**2 = R**2` |
| Shaft roll | `src/shared/python/grip_spec/shaft_roll.py` | `APPEARANCE_GRIP_SHAFT_AXIS = (0, 0, 1)`. `apply_shaft_roll(eye(3), axis, roll_deg)` rotates column vectors. The mesh baker does not apply roll |
| Authoring spec | `src/shared/python/grip_spec/types.py` | `GripSpec.legacy()`, `HandGripSpec(shaft_roll_deg=...)`, `spec.to_grip_pose()` |
| Club tube | `src/shared/python/model_appearance/club_shaft_mesh.py` | `tapered_tube(start, end, r_start, r_end, sides)` |
| Materials | `src/shared/python/model_appearance/library.py` | `MATERIALS["grip_rubber"]`, shaft finishes `graphite` and `satin_steel`, skin swatches via `Material.rgba` |
| Spacing | `src/shared/python/model_appearance/grip_pose.py` | `LEAD_BELOW_BUTT_M = 0.032`, `NATIVE_HAND_SPACING_M = 0.0762` |
| Contracts | `src/shared/python/contracts.py` | `require` / `ensure` raise only when `DBC_LEVEL` is `enforce` (the default). Public entry points must also raise `ValueError` or `TypeError`, because `DBC_LEVEL=off` silences `require` |
| Mesh type | `src/shared/python/model_appearance/geometry.py` | `Mesh(vertices (N, 3), faces (M, 3))` |

Unit tests for the capsule mesh already pass through
`tests/unit/model_appearance/test_grip_hand_mesh.py`. They use
`grip_radius_m=0.014`. Put that radius in settings. Do not invent a new
one in the renderer.

There is no `scripts/config/hand_render.json`. There is no `render.py`,
no CLI, no golden PNG, and no `tests/unit/model_appearance/hand_backends/`
directory. Nothing in this run calls `build_grip_hand_mesh`.

### Settings to Add

One JSON file, `scripts/config/hand_render.json`. The renderer reads it.
Numbers below are the contract for that file, not literals to scatter
through Python.

| Key | Value to record | Why |
| --- | --- | --- |
| `frame_px` | at least 1024 | Epic quality bar |
| `fill_fraction` | 0.70 | Hand, not the whole club, occupies this share of width and height |
| `fill_tolerance` | a band around 0.70, suggested 0.12 until a real render is scored | Auto-frame test |
| `fov_deg` | a modest perspective, suggested 28 | Keep distortion off the fingers |
| `grip_radius_m` | 0.014 | Matches existing capsule tests |
| `shaft_radius_m` | thinner than the grip, suggested 0.0045 | Steel above the grip rubber |
| `grip_length_margin_m` | suggested 0.04 past the hand | Club visible without shrinking the hand |
| `shaft_extension_m` | suggested 0.12 toward the head | Shaft continues past the rubber |
| `rolls_deg` | −20, 0, +20 | Contact sheet |
| `views` | `palm`, `back`, `thumb_side`, `pinky_side`, `butt`, `eye` | Task list |
| `background_rgba` | dark neutral, suggested `(0.12, 0.13, 0.15, 1)` | So skin is not the same colour as the backdrop |
| `skin_rgba` | from the library skin swatch, suggested a mid tone near `(0.77, 0.56, 0.46, 1)` | Fallback when a backend sends no colour |
| `grip_rgba` | `MATERIALS["grip_rubber"].rgba` | Club rubber |
| `shaft_rgba` | `MATERIALS["satin_steel"].rgba` or `graphite` | Shaft |
| `lights` | `key`, `fill`, `rim` in camera space | Directions, colours, intensities, and which one casts a shadow |
| `wrap` | suggested 0.45 | Subsurface-like wrap term |
| `sss_tint` | a red-brown, suggested `(0.55, 0.18, 0.12)` | Wrap approximation |
| `sss_strength` | suggested 0.22 | Wrap approximation |
| `shadows` | true when the engine can do it cheaply | Soft shadows are optional |
| `perceptual_mae_max` | suggested 0.06 on a 64×64 box-filtered luminance image | Golden tolerance |
| `min_luminance_std` | a floor that rejects a flat wash, suggested 0.04 on foreground pixels in 0–1 | Smoke test |

View directions are coefficients in an anatomical basis
`(thumb, dorsum, shaft)`, not hard-coded world axes. Suggested
coefficients, to be stored in the JSON:

| View | Look direction `(thumb, dorsum, shaft)` | Up |
| --- | --- | --- |
| `palm` | `(0, -1, 0)` | shaft |
| `back` | `(0, 1, 0)` | shaft |
| `thumb_side` | `(1, 0, 0)` | shaft |
| `pinky_side` | `(-1, 0, 0)` | shaft |
| `butt` | `(0, 0, -1)` | dorsum |
| `eye` | `(0.15, 0.55, -0.75)` | mix of dorsum and shaft |

Derive the basis from skin vertices in the hand-grip frame:

- Shaft is `+z`.
- Dorsum is the unit `xy` direction of the outer skin vertices (the bulk
  of a grip hand sits off the shaft, on the back of the hand).
- Thumb is the unit `xy` direction of the skin vertex with the highest
  `z` that is also far from the shaft. The capsule thumb points toward
  the club head.

Auto-frame on skin vertices only. Place the eye along the view direction
so the projected skin bounds fill `fill_fraction` of both axes, then
take the longer requirement. Include the grip rubber and a shaft stub
in the scene even when they extend past the frame. The club has to
appear inside the hand, and the hand stays near 70%.

Both hands share the grip-frame axes. Put the trail mesh at
`z = -(trail_distance_below_butt - lead_distance_below_butt)`, which is
about `-NATIVE_HAND_SPACING_M` for the legacy spec. Check that against
`GripPose` before freezing it.

### How to Run Tests Today

There are no renderer tests yet. The capsule mesh suite is the regression
net for the geometry you will wrap:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH=. \
  python3 -m pytest tests/unit/model_appearance/test_grip_hand_mesh.py \
  -m unit --timeout=60
```

After `render.py` exists, the suite to add and keep green is:

```bash
MUJOCO_GL=osmesa QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH=. \
  python3 -m pytest tests/unit/model_appearance/hand_backends \
  -m unit --timeout=120
```

Title case on this handoff:

```bash
python3 scripts/check_document_title_case.py \
  docs/development/full_body_models/hand_handoff/shared_renderer.md
```

### Sibling Backends, Read Only

Each lane invented its own registry. They are not on this branch. Do not
copy their trees into the commit. Import them from a scratch checkout
when you need a demo sheet.

| PR | Branch | Head | Return value |
| --- | --- | --- | --- |
| [#15142](https://github.com/D-sorganization/UpstreamDrift/pull/15142) | `cursor/grip-hand-render-mesh-fix-8498` | `1297671eeb68dd3b58a99aea950c3a18ec536acc` | Matplotlib helper `grip_preview_matplotlib`. No registry. Base is `main` |
| [#15143](https://github.com/D-sorganization/UpstreamDrift/pull/15143) | `cursor/shadow-hand-grip-visuals-4437` | `b347adc2380e5db30db944e26ae58705ec3550d6` | `HandVisualResult(mesh, material_rgba)`. `registry.build_hand_visual(backend, spec, side, joint_angles, **kwargs)`. `register_hand_backend`. Default name `shadow` |
| [#15146](https://github.com/D-sorganization/UpstreamDrift/pull/15146) | `cursor/hand-procedural-skin-ff59` | `600db07bb2acc30bdc8e1307742b484dd0e1df18` | `HandVisualResult(backend_id, role, mesh, grip_radius_m, material, palm_arch_angle_deg)`. `material` is a library name string. `registry.build_hand_visual(backend_id, spec, side, joint_angles, *, grip_radius_m, settings)`. `register_backend` |
| [#15147](https://github.com/D-sorganization/UpstreamDrift/pull/15147) | `cursor/myohand-visual-backend-dcca` | `9e11da5e6447f87f191f8973f6960146b9f30c56` | `dict[str, Mesh]`, one mesh per bone. `get_hand_backend(name)(spec, side, joint_angles)`. Names `myohand` and `capsule` |
| [#15149](https://github.com/D-sorganization/UpstreamDrift/pull/15149) | `cursor/skinned-human-hand-backend-751f` | `31d467f057ba99ad8dc565802275ed8f8d99e9bb` | `HandVisualResult(backend, side, mesh, skin_rgb)`. `build_hand_visual(spec, side, joint_angles, *, grip_radius_m, backend="skinned_human")` |

[#15146](https://github.com/D-sorganization/UpstreamDrift/pull/15146)
also has `hand_backends/preview.py`, which draws a flat-shaded
`Poly3DCollection` plus a grip wireframe. That is the picture this
renderer replaces. Do not extend that preview.

## What Worked and What Failed

Probed on this Linux cloud VM (Python 3.12.3) on 2026-11-15. No hand was
rendered. The probes drew a sphere only, so they prove the GL path and
nothing about framing, skin, or the club.

### What the Fresh VM Had

`numpy` 1.26.4 was importable. These were not: `mujoco`, `pyrender`,
`trimesh`, `PIL`, `matplotlib`, `imageio`. `ldconfig` had `libGL.so.1`,
`libGLX`, and `libGLESv2`. It did not have `libOSMesa` or `libEGL`.
`DISPLAY` was `:1`. `sudo` apt worked.

A fresh Ubuntu image of this kind cannot offscreen-render until the
system libraries are installed. Do not write the engine probe as
"import mujoco and hope".

### What Rendered

Installed for the probe only, not as a repo dependency change:

- `libosmesa6` 25.1.7-1ubuntu2~24.04.2
- `libegl1` 1.7.0-1build1 and `libegl-mesa0` 25.2.8
- `mujoco` 3.17.0 (satisfies `pyproject.toml` `mujoco>=3.6,<4`)
- `pillow` 13.0.0

`MUJOCO_GL=osmesa` with `DISPLAY` unset rendered a 64×64 RGB sphere
(`min 0`, `max 142`, `mean 8`) and `Renderer.close()` returned cleanly.
Use this as the primary engine. A fixed camera (`pos` plus `xyaxes`)
was enough. The default headlight was turned off in the MJCF
(`<headlight active="0"/>`).

`MUJOCO_GL=egl` with `DISPLAY` unset also returned a 64×64 frame
(`max 139`, `mean 7`). Process teardown then raised
`OpenGL.raw.EGL._errors.EGLError` inside `Renderer.__del__` →
`mujoco.egl.GLContext.free`. A green image is not a clean exit on EGL.
Prefer OSMesa. If you must use EGL, call `close()` yourself and do not
let the destructor fail the test.

pyrender 0.1.45 with `PYOPENGL_PLATFORM=egl`, an icosphere, a
directional light, and a `PerspectiveCamera` returned a 64×64 frame
(`max 150`, `mean 51`). That path can be the second engine.

### What Failed

`MUJOCO_GL=glfw` with `DISPLAY` unset:

- `GLFWError: (65550) b'X11: The DISPLAY environment variable is missing'`
- then `mujoco.FatalError: an OpenGL platform library has not been loaded into this process`

Do not select GLFW for CI.

pyrender 0.1.45 with `PYOPENGL_PLATFORM=osmesa`:

```text
ImportError: cannot import name 'OSMesaCreateContextAttribs'
from 'OpenGL.osmesa'
```

Do not use pyrender's OSMesa platform. MuJoCo's own OSMesa binding is
the one that worked.

`pip install pyrender` in the same user environment downgraded
`PyOpenGL` 3.1.10 to 3.1.0 and upgraded `numpy` 1.26.4 to 2.5.4.
MuJoCo OSMesa still rendered afterwards (`max 81` on a second sphere).
Pin `PyOpenGL` to the version MuJoCo wants, and do not let a pyrender
install float numpy.

pyrender with no camera raises `ValueError: Cannot render scene without a camera`. That one is a caller bug, not a platform bug.

### Dead Ends

- A new `registry.py` that imports shadow, procedural skin, MyoHand, and
  the skinned human will not import on this branch, and it will conflict
  with four incompatible registries. Duck-type the result object instead.
- MuJoCo's free camera (`azimuth`, `elevation`, `distance`) is Z-up in
  the world. In the hand-grip frame Z is the shaft. A free-camera butt
  view and a free-camera palm view will not match the table above. Set
  `pos` and `xyaxes` from the look-at basis: camera `z` points from the
  target back to the eye, camera `x` is `normalize(cross(up, z))`,
  camera `y` is `cross(z, x)`.
- Leaving the MuJoCo headlight on is the washed-out look the epic is
  rejecting.
- Matplotlib `Poly3DCollection` plus a cylinder wireframe is the picture
  [#15142](https://github.com/D-sorganization/UpstreamDrift/pull/15142)
  and [#15146](https://github.com/D-sorganization/UpstreamDrift/pull/15146)
  already produced. It is the fallback, and the receipt must say so.
- Reading only `.mesh` drops the MyoHand backend, which returns
  `dict[str, Mesh]`.
- Reading only `material_rgba` drops the procedural backend (`material`
  is a string) and the skinned backend (`skin_rgb`).
- Committing another lane's backend, assets, or meshes. Scratch
  checkouts only.
- Pushing this work onto `cursor/hrg-gb1b-shaft-roll-2882` or any
  sibling branch.
- Editing `vendor/ud-tools/` or policy workflows.
- Arming auto-merge or merging. Dieter reviews the checklist before
  merge.
- Treating a sphere probe, a changed golden, or a raised tolerance as
  acceptance. The ledger rule in `CLAUDE.md` applies in spirit here too:
  an image of a sphere is not a hand.
- Vendoring MANO or SMPL-X. They stay research-only.

## Next Steps

Do these in order. Each step's tests go in the same commit as the code
(red, then green). Stay under 1200 lines per file. Split engines into
`render_mujoco.py`, `render_pyrender.py`, and `render_matplotlib.py` if
`render.py` would cross that line. `render.py` stays the public facade.

No `print()` under `src/`. The CLI lives in `scripts/render_hand_backend.py`
(scripts may print) or writes through `sys.stdout` from `__main__.py`.

1. Settings loader. Acceptance: a unit test loads
   `scripts/config/hand_render.json`; `frame_px >= 1024`;
   `fill_fraction == 0.70`; the six view names and the three rolls are
   present; key, fill, and rim lights are present. A missing file, a
   non-finite number, or an unknown view name raises `ValueError`.

2. Scene assembly with no GL. Acceptance: `build_grip_hand_mesh` for
   `Hand.LEAD` and `Hand.TRAIL` at `grip_radius_m` from settings yields
   parts tagged skin, grip, and shaft. A shaft roll of +20° moves skin
   vertices about `z` and matches `apply_shaft_roll` on a column vector.
   Roll 0° leaves vertices unchanged. Duck-typed inputs covered by
   tests: an object with `.mesh` and `.material_rgba`; an object with
   `.mesh` and a string `.material` that resolves in `MATERIALS`; an
   object with `.mesh` and `.skin_rgb`; a `dict` of `Mesh` values.
   Empty vertices, a non-finite roll, and a roll outside `[-180, 180]`
   raise `ValueError` or `TypeError`.

3. Camera rig, still no GL. Acceptance: two calls on the same scene
   return the same eye, target, and up. Palm and back look in opposite
   directions along dorsum. Butt looks along `-z`. The projected skin
   bounds are `fill_fraction` within `fill_tolerance`. The rig reads
   coefficients from settings.

4. MuJoCo OSMesa engine. Acceptance, marked so a machine without
   `libOSMesa` skips with the import or GL error in the message, and a
   machine that has it does not skip: one 1024 px palm image; a second
   render is byte-identical; foreground fill is inside the tolerance of
   0.70; foreground luminance standard deviation is at least
   `min_luminance_std`; pixels near `grip_rgba` are present; the palm
   image differs from the back image and from palm at +20°. Headlight
   off. Key, fill, and rim from the camera-space settings. `Renderer.close()`
   is called. Run it as `MUJOCO_GL=osmesa` with `DISPLAY` unset.

5. pyrender engine, used only when step 4 cannot create a context.
   Acceptance: the same structural assertions as step 4, with
   `PYOPENGL_PLATFORM=egl` and an explicit camera in the scene. Do not
   set `PYOPENGL_PLATFORM=osmesa` (the `OSMesaCreateContextAttribs`
   import is broken on pyrender 0.1.45). Pin `PyOpenGL` so a pyrender
   install cannot downgrade MuJoCo's copy.

6. Matplotlib fallback, used only when steps 4 and 5 both fail.
   Acceptance: the image is still at least 1024 px, the hand fill test
   still passes, and the returned metadata says `engine=matplotlib`.
   Smooth vertex normals and one Lambert light are enough for the
   fallback. It is not the golden source.

7. Golden perceptual test for the OSMesa palm view at roll 0°.
   Acceptance: mean absolute error of a 64×64 box-filtered luminance
   image against the committed PNG is at most `perceptual_mae_max`.
   A flat grey image fails. Store the engine name beside the PNG.
   Compare goldens only when the running engine name matches. Do not
   loosen the tolerance to hide an engine change.

8. Contact-sheet CLI. Acceptance: a grid whose rows are registered
   sources and whose columns are the three rolls; each cell is one
   standard view (palm is enough for the sheet); labels name the
   backend and the roll; two runs write identical PNG bytes. On this
   branch the only source is the local capsule. Suggested invocation
   once the module exists:

   ```bash
   MUJOCO_GL=osmesa QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH=. \
     python3 scripts/render_hand_backend.py \
     --backend capsule \
     --views palm,back,thumb_side,pinky_side,butt,eye \
     --rolls -20,0,20 \
     --contact-sheet \
     --out docs/development/full_body_models/evidence/hand_render
   ```

9. Scratch-only demos of sibling backends. Acceptance: a flag such as
   `--scratch-import PATH` loads that tree's `build_hand_visual`,
   adapts the return value, and writes PNGs plus a `receipt.json` that
   lists which backends imported and the exception for any that did
   not. The commit contains none of their source. Shadow needs its
   Menagerie assets. MyoHand needs the pinned `myo_sim` tree and is
   research-only until [#13763](https://github.com/D-sorganization/UpstreamDrift/issues/13763)
   is signed off. The skinned human needs `assets/hand_cc0/` from
   [#15149](https://github.com/D-sorganization/UpstreamDrift/pull/15149).
   Procedural skin in [#15146](https://github.com/D-sorganization/UpstreamDrift/pull/15146)
   is pure NumPy and is the easiest second row.

10. Run the visual self-review procedure below after step 4 and again
    after every later change. Commit the PNGs and the score. A passing
    pytest with no opened PNG is not done.

File a child issue of [#15150](https://github.com/D-sorganization/UpstreamDrift/issues/15150)
when you open the implementation PR. Use `Refs`, never `Fixes` or
`Closes`. This handoff PR does not close the epic. Do not arm
auto-merge.

## How Backend Tracks Call It

Nothing here is importable yet. Implement this surface and keep it
stable so the other lanes can call it without taking a dependency on
MuJoCo in their own modules.

```python
from src.shared.python.model_appearance.hand_backends.render import (
    register_render_source,
    render_contact_sheet,
    render_view,
)

register_render_source("shadow", build_hand_visual)

image = render_view(result, "palm", shaft_roll_deg=0.0, side="lead")
sheet = render_contact_sheet(
    backend_ids=("capsule", "shadow"),
    rolls_deg=(-20.0, 0.0, 20.0),
    view="palm",
    side="lead",
)
```

`result` may be any of these, all in the hand-grip frame:

- `GripHandMesh` or any object with `.mesh` (`vertices`, `faces`) and
  optional `.grip_radius_m`
- shadow: `.mesh` and `.material_rgba`
- procedural: `.mesh` and `.material` (library name)
- skinned human: `.mesh` and `.skin_rgb`
- MyoHand: `dict[str, Mesh]`

`render_view` returns `uint8` RGB of shape `(frame_px, frame_px, 3)` and
metadata naming the engine (`osmesa`, `egl`, `pyrender`, or
`matplotlib`). `register_render_source(name, builder)` stores a callable
the contact sheet invokes. The local capsule builder is registered on
import and calls `build_grip_hand_mesh`. It does not import the other
lanes.

Other lanes should keep their own `build_hand_visual(grip_spec, side,
joint_angles, ...)` and pass the object they already return. They should
stop baking matplotlib evidence once this function exists. They should
not each grow a camera rig.

Shaft roll stays in the renderer (or is already baked by the backend).
If the backend has already applied `apply_shaft_roll`, pass
`shaft_roll_deg=0` so the hand is not rolled twice. Document which
choice that backend made in its PR.

## Licences and Dependencies

Renderer code written for this track is under the repository MIT
licence. No third-party hand mesh is added by the renderer itself.

| Dependency | Licence and note |
| --- | --- |
| `mujoco>=3.6,<4` | Already in `pyproject.toml`. Apache-2.0, DeepMind. Probe used 3.17.0. System library `libosmesa6` is required for the preferred GL path and is not a pip package |
| `pillow>=13` | Already in `pyproject.toml`. Used to write PNG evidence |
| pyrender 0.1.45 | MIT. Secondary engine only. Its OSMesa platform is broken. An unpinned install downgraded `PyOpenGL` and upgraded `numpy` on this VM |
| trimesh | MIT. Pulled in by the `urdf` extra (`trimesh>=4.11.5`). Needed only for the pyrender path |
| matplotlib | Fallback only. Do not make the unit default depend on a window |
| Capsule hand | In-repo procedural mesh. No extra asset licence |
| Shadow hand ([#15143](https://github.com/D-sorganization/UpstreamDrift/pull/15143)) | Apache-2.0 MuJoCo Menagerie meshes. That PR records provenance under `assets/third_party/shadow_dexterous_hand/`. Do not copy the meshes into this branch |
| MyoHand ([#15147](https://github.com/D-sorganization/UpstreamDrift/pull/15147)) | Apache-2.0 [MyoHub/myo_sim](https://github.com/MyoHub/myo_sim). `NOTICE` in that PR. [#13763](https://github.com/D-sorganization/UpstreamDrift/issues/13763) is still the sign-off gate. Research-only until that closes |
| Skinned human ([#15149](https://github.com/D-sorganization/UpstreamDrift/pull/15149)) | CC0 MakeHuman hm08 extract, with `LICENSE.CC0` and `NOTICE` in that PR. MANO and SMPL-X are named there as research-only and are not vendored |
| Procedural skin ([#15146](https://github.com/D-sorganization/UpstreamDrift/pull/15146)) | No bundled hand asset. NumPy and, for its own preview, Matplotlib |

## Visual Self-Review Procedure

Run this yourself after every change that touches rendering, cameras,
materials, or the capsule scene. Pytest does not replace it.

1. Render, at `frame_px` from settings (at least 1024), the six views
   for the lead hand and the trail hand at shaft roll −20°, 0°, and
   +20°, plus one both-hands frame at roll 0°. Write them under
   `docs/development/full_body_models/evidence/hand_render/` with the
   engine name in `receipt.json`.
2. Open each PNG and look at it. On a headless host, load the pixels
   and still write a score from what is in the picture. The test log is
   not the review.
3. Score every line below as pass or fail for each image. Record the
   score in the PR and in `receipt.json`. Any fail means the change is
   not done.

Checklist:

- Separate fingers and a separate thumb.
- A cupped palm.
- A knuckle line.
- Fingers wrap the grip.
- No tubes, blobs, or torn triangles.
- Skin shading is smooth (no facet glitter, no flat white wash).
- The hand fills about 70% of the frame.
- The club is visible (grip rubber and shaft).
- Lead and trail mirror correctly.

This run has no hand PNGs to score. The sphere probes above are not a
score. The first implementation PR is incomplete until a person can
open the six views and mark this list.
