# Shared Hand Renderer Handoff

This file describes the shared grip-hand renderer as it now exists on
`cursor/hrg-shared-hand-renderer-cec5`. The renderer, its settings, its
tests, a CLI, and a scored set of capsule-hand pictures are on the branch.

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

Engine order, as built:

1. MuJoCo offscreen through OSMesa, then through EGL.
2. A matplotlib fallback only when neither can create a GL context. The
   fallback still auto-frames the hand. A manifest that says
   `engine=matplotlib` is not quality-bar evidence.

pyrender is not wired in. It is not installed on the build host, and the
task for this lane said to skip it.

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

| Piece | Path |
| --- | --- |
| Settings | `scripts/config/hand_render.json` |
| Settings loader | `src/shared/python/model_appearance/hand_backends/render_settings.py` |
| Scene, basis, camera rig, normals | `src/shared/python/model_appearance/hand_backends/render_rig.py` |
| MuJoCo engine | `src/shared/python/model_appearance/hand_backends/render_mujoco.py` |
| Matplotlib fallback | `src/shared/python/model_appearance/hand_backends/render_matplotlib.py` |
| Public facade | `src/shared/python/model_appearance/hand_backends/render.py` |
| CLI | `scripts/render_hand_backend.py` |
| Tests | `tests/unit/model_appearance/hand_backends/test_render.py` |
| Pictures and manifest | `docs/development/full_body_models/evidence/hand_render/` |

`hand_backends/__init__.py` holds a docstring only, so the backend lanes
can add their own without a conflict that matters.

### What Rendered

On the build host (Linux, Python 3.12, MuJoCo 3.8.0, `libEGL` present,
`libOSMesa` absent) every frame came from the `egl` engine. The OSMesa
probe failed with `AttributeError: 'NoneType' object has no attribute
'glGetError'` and the renderer moved on to EGL. OSMesa has not been run
through this code. A repeat render is byte-identical on EGL.

### Settings

Every number is in `scripts/config/hand_render.json`. The loader raises
`ValueError` for a missing file, a missing key, a non-finite number, a
value outside its range, or an unknown view name.
`settings.with_overrides(frame_px=1536)` returns a re-validated copy.

Views are stored as an `eye` and an `up` vector in the anatomical basis
`(thumb, dorsum, shaft)`. `eye` points from the hand toward the camera,
so the camera looks along `-eye`.

| View | `eye` | `up` | What It Shows |
| --- | --- | --- | --- |
| `palm` | `(0, -1, 0)` | shaft | Fingers across the grip |
| `back` | `(0, 1, 0)` | shaft | Back of the hand |
| `thumb_side` | `(1, 0, 0)` | shaft | Thumb (radial) side |
| `pinky_side` | `(-1, 0, 0)` | shaft | Little-finger side |
| `butt` | `(0, -0.55, -0.83)` | dorsum | From the butt end, tilted to the palm side so the heel pad on the grip shows |
| `eye` | `(0.15, 0.55, -0.75)` | `(0, 0.35, 1)` | The golfer's eye |

The first handoff called this column a look direction. It is the eye
direction: with a look direction of `(0, -1, 0)` the `palm` view would
show the back of the hand.

Grip and shaft colours are named library materials (`grip_material`,
`shaft_material`), read from `MATERIALS`, so the colour has one source.

## How Backend Tracks Call It

```python
from src.shared.python.model_appearance.hand_backends.render import (
    ScenePart,
    build_club_parts,
    load_hand_render_settings,
    register_render_source,
    render_contact_sheet,
    render_scene,
    render_view,
    save_png,
)

frame = render_view(result, "palm", shaft_roll_deg=0.0, side="lead")
save_png(frame, "lead_palm.png")
frame.image          # uint8 RGB, (frame_px, frame_px, 3)
frame.engine         # "osmesa", "egl" or "matplotlib"
frame.fill_fraction  # measured skin span of the frame, longer axis
frame.kind_mask      # uint8: 0 background, 1 skin, 2 grip, 3 shaft
```

Signatures:

```python
render_view(result, view, *, shaft_roll_deg=0.0, side="lead", settings=None,
            grip_radius_m=None, basis=None, engine=None) -> RenderedView
render_scene(parts, view, *, basis=None, settings=None, side="lead",
             engine=None) -> RenderedView
build_club_parts(z_min, z_max, *, settings, grip_radius_m=None)
    -> (grip ScenePart, shaft ScenePart)
anatomical_basis(skin_vertices, *, side, settings=None) -> AnatomicalBasis
save_png(rendered, path) -> Path
register_render_source(name, builder, *, replace=False) -> None
render_contact_sheet(backend_ids=None, rolls_deg=None, view="palm",
                     side="lead", settings=None, *, engine=None) -> ContactSheet
```

`result` is duck-typed, in the hand-grip frame (grip axis `z`, butt
toward head): an object with `.mesh` (`vertices`, `faces`) and an
optional `.skin_rgb`, `.material_rgba`, `.material` (library name) or
`.grip_radius_m`; a bare mesh; or a `dict` of meshes.

`ScenePart(mesh, rgba, kind)` takes `kind` in `skin`, `grip`, `shaft`.
`render_scene` frames the camera on every skin part, which is how to draw
both hands: place two skin meshes in one frame and add
`build_club_parts(z_min, z_max, settings=...)`.

### Passing Your Own Basis

A backend that knows its palm normal and thumb side should pass them:

```python
render_view(result, "palm", basis=(thumb_side_vector, dorsum_vector))
```

`basis` is `(thumb, dorsum)`, `(thumb, dorsum, shaft)`, or an
`AnatomicalBasis`. The renderer orthonormalises in the order shaft,
dorsum, thumb. `thumb` here is the radial side of the hand, not the
thumb bone: a vector along the shaft raises `ValueError`. Give the basis
for the hand before the roll; the camera does not follow the roll, so a
roll shows as the hand turning.

Without `basis`, `render_view` looks for `result.render_basis`, then for
`result.components` tagged `dorsum` and `thumb` (the capsule hand), then
falls back to a vertex heuristic. The heuristic put the capsule trail
hand about a quarter turn off, because its overlapped little finger
pulls the centroid. Do not rely on it for a real hand.

### Shaft Roll

`shaft_roll_deg` turns the skin about `+z` with `apply_shaft_roll`. If
the backend already baked the roll, pass `0` so it is not applied twice.

### Contact Sheet Sources

`register_render_source(name, builder)` stores a callable that is called
as `builder(side="lead" | "trail", grip_radius_m=<float>)` and returns a
`result`. Wrap a lane's own `build_hand_visual` in a lambda. The local
`capsule` source is registered when `render.py` is imported.

### Command Line

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH=. \
  python3 scripts/render_hand_backend.py \
  --backend capsule \
  --views palm,back,thumb_side,pinky_side,butt,eye \
  --rolls -20,0,20 --both-hands --contact-sheet \
  --out docs/development/full_body_models/evidence/hand_render
```

It writes `<backend>_<side>_<view>_roll<m20|p00|p20>.png`, a both-hands
frame per view, `contact_sheet_<side>_<view>.png`, and
`render_manifest.json` (engine, GL probe errors, settings hash, and a
SHA-256 and fill fraction per frame).

The manifest is not named `receipt.json` on purpose. Any JSON with
`receipt` in its name under an evidence root is swept into the
matched-swing ledger (`src/shared/python/motion_matching/ledger.py`),
which then wants a lane classification, an acceptance block, and a
refreshed `reports/matched_swing_ledger.json`. A picture manifest is not
a swing receipt. `--manifest-name receipt.json` is there for whoever
decides to take that on.

## How to Run Tests

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH=. \
  python3 -m pytest tests/unit/model_appearance/hand_backends \
  -m unit --timeout=120
```

The pinned Tools tree has to be initialised first
(`git submodule update --init vendor/ud-tools`), or `src/__init__.py`
refuses to import under pytest.

Result on the build host: 55 passed, 0 skipped, with the GL tests running
on EGL. The GL tests skip only when no GL context can be created, and the
skip message carries the GL error text.

The tests cover: the settings loader and its rejections; scene assembly
and the four duck-typed result shapes; shaft roll against
`apply_shaft_roll`; welded smooth normals and the crease angle; camera
determinism, palm opposite back, the butt eye position, and fill within
tolerance for six views and both hands; the GL frame size, byte-identical
repeat, skin luminance spread, background and club colours, palm against
back and against a rolled palm, the shaft in the eye view, and a
two-hand `render_scene`; the matplotlib fallback; the source registry;
PNG round trip; the contact sheet; and two CLI runs writing identical
bytes.

There is no golden-image test. `golden_px` and `perceptual_mae_max` are
in the settings and nothing reads them yet.

Title case on this handoff:

```bash
python3 scripts/check_document_title_case.py \
  docs/development/full_body_models/hand_handoff/shared_renderer.md
```

## What Worked and What Failed

### Worked

- Smooth skin. Each skin mesh is sent to MuJoCo unindexed with one normal
  per face corner. A corner averages the faces around its vertex that lie
  within `normal_crease_deg` (60°) of its own face, after welding
  vertices closer than `normal_weld_tolerance_m`. Seams with duplicated
  vertices shade smooth, and a hard rim stays sharp.
- Framing. The eye distance and target are iterated so the projected
  skin bounds are centred and the longer axis spans `fill_fraction`. The
  measured fill was 0.64 to 0.70 over the 42 committed frames; the low
  end is the club hiding the outermost skin pixels.
- A fixed camera from `pos` and `xyaxes`, the headlight off, and key,
  fill, and rim directional lights given in camera space.
- An exact part mask from a second, non-multisampled segmentation pass.
- 8x multisampling through `<quality offsamples>`.

### Failed or Left Out

- Shadows. MuJoCo shadow maps showed acne specks and serrated edges on
  the fingers at this scale, at every `shadowclip`, `shadowscale`, and
  `shadowsize` tried. `shadows` is `false` in the settings. Setting it to
  `true` turns the key-light shadow back on with those artefacts. The
  pictures therefore have no contact shadow between fingers and grip.
- Wrap and subsurface tint. MuJoCo's fixed pipeline has no wrap term.
  The MuJoCo engine only adds `wrap * sss_strength * sss_tint` to the
  ambient light. The matplotlib fallback applies the real wrap formula.
- `mujoco.Renderer`. Under pytest something imports MuJoCo before
  `tests/conftest.py` sets `MUJOCO_GL`, which binds it to GLFW, and GLFW
  cannot render headless. The engine opens `mujoco.egl.GLContext` or
  `mujoco.osmesa.GLContext` itself and drives `MjrContext` directly, and
  it frees both explicitly.
- Probing OSMesa in-process. A failed OSMesa import breaks `import
  mujoco` for the rest of the process, so each backend is probed in a
  child process and the result is cached.
- The shaft stub in the side views. With the hand at 70% of the frame
  and a 4 cm rubber margin, the steel starts outside the frame in
  `palm`, `back`, `thumb_side`, and `pinky_side`. It shows in `eye`. The
  grip rubber shows in every view.
- The matplotlib fallback is faceted (one colour per triangle) and uses
  a painter's sort with no depth buffer.
- No pyrender engine, no golden-image test, no `--scratch-import` flag
  for sibling backends.
- `argparse` reads `--rolls -20,0,20` as an option. The CLI rewrites it
  to `--rolls=-20,0,20` before parsing.

### Still True From the First Probe

- Do not select GLFW for headless runs.
- A new shared `registry.py` importing every lane's backend will not
  import on this branch. The renderer duck-types the result instead.
- Do not commit another lane's backend, assets, or meshes. MANO and
  SMPL-X stay research-only.

## Self-Review Score

Scored by opening the 42 frames, the six both-hands frames, and the two
contact sheets. The same text is in
`evidence/hand_render/self_review.json` and in the manifest.

Renderer:

| Check | Result |
| --- | --- |
| Smooth skin shading | Pass. No facet glitter |
| Hand fills about 70% | Pass. 0.64 to 0.70 measured |
| Key, fill, rim; no headlight | Pass, but no cast shadows, so the fingers have no contact shadow |
| Club visible | Partial. Grip rubber in every view; the shaft stub only in `eye` |
| Lead and trail mirror | Pass |
| Repeat render identical | Pass on EGL |

| View | Score out of 5 | Note |
| --- | --- | --- |
| `palm` | 4 | Reads clearly; flat without contact shadows |
| `back` | 4 | Clean gradient over the dorsum |
| `thumb_side` | 4 | Depth reads from key and rim |
| `pinky_side` | 4 | Same |
| `butt` | 3 | Framing right; the butt cap is a flat near-black disc |
| `eye` | 4 | Grip and shaft both visible |

The capsule hand itself fails the cupped palm, knuckle line, and
"no tubes" lines of the checklist below. That is the capsule source, and
it is expected.

## Next Steps

1. Get OSMesa run once on a host that has `libOSMesa`.
2. Fix or replace shadows (a baked ambient-occlusion term would need
   vertex colours, which MuJoCo meshes do not take).
3. Add the golden-image test for `palm` at roll 0°, compared only when
   the engine name matches.
4. Decide whether the manifest should join the matched-swing ledger.
5. Add a `--scratch-import PATH` flag to the CLI for sibling backends.
6. The implementation PR is over the 500-line limit; it needs the
   `large-pr-approved` label or a split.

## Sibling Backends, Read Only

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
