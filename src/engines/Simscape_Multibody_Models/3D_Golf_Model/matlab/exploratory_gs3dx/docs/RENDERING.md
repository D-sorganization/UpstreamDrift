# Rendering: Headless Stills and Swing Video

Issue [#10979](https://github.com/D-sorganization/UpstreamDrift/issues/10979),
epic [#10950](https://github.com/D-sorganization/UpstreamDrift/issues/10950).
MATLAB R2025b.

Mechanics Explorer needs a desktop and keeps `matlab -batch` alive, so it
cannot run unattended. `gs3dx_render` draws the model with ordinary MATLAB
graphics in an invisible figure instead, and writes PNG stills
(`exportgraphics`) and MP4 video (`VideoWriter`).

## How It Works

1. Every Solid block of the model (`sm_lib/Body Elements/* Solid`, except
   `GraphicType = None`) is read: cylinder, sphere, brick or ellipsoid
   dimensions, evaluated in the model workspace and converted to metres, and
   its diffuse colour and opacity. A File Solid is read from its STL (found
   on the MATLAB path, scaled by its `ExtGeomFileUnits`), with coincident
   vertices merged so the facets shade smoothly.
2. Each solid's reference frame port is exposed in memory (the model is
   closed without saving) and a `KinematicsSolver` gets a World translation
   and rotation variable for every solid.
3. For each requested frame the solver is driven by the joint targets (the
   grip loop's right elbow, shoulder and wrist are initial guesses) and every
   solid is drawn at its pose. Rotation variables are intrinsic XYZ Euler
   angles in degrees, `R = Rx(a) Ry(b) Rz(c)`.
4. Optional marker dots (for example the capture joint centres) and a ground
   plane are overlaid.

```matlab
ik = gs3dx_whole_body_ik(jc, model='GS3DX_Fit', frames=1:jc.impact_frame);
out = gs3dx_render('GS3DX_FitBalance', ik, stills=[1 320 jc.impact_frame], ...
    still_files=["addr.png" "top.png" "imp.png"], view="down-the-line", markers=mk);
```

Poses are an IK struct (`.joint_ids`, `.joint`, optional `.t`) or a joint
matrix. Views are `"face-on"` (camera on +X, the way the golfer faces),
`"down-the-line"` (camera on −Y, behind the golfer looking at the target),
`"top"` or `[azimuth elevation]` as for `view`; `out.view` is the camera used.
`focus` gives a close-up: a solid's name (the view follows it, for example
`focus="Driver Head"`) or a fixed point `[x y z]`, with half-width
`focus_width` (0.2 m); `out.focus` holds the centres used.

```matlab
gs3dx_render('GS3DX_Human', pose, stills=1, view="down-the-line", focus="Driver Head", focus_width=0.15);
```

Before 2026-09-28 the two names were swapped (face-on drew down the line, and
down-the-line drew a face-on view from behind the golfer).

## Tests

`tests/test_gs3dx_render.m` (5 tests):

- face-on puts the camera on +X and down-the-line on −Y, and an unknown view
  is an error;
- one pose renders headless to a PNG that is not a single colour;
- the drawn `L Thigh` cylinder is its solver pose applied to its dimensions,
  to 1e-9 m;
- the same for the `L Foot` brick;
- `GS3DX_Human`'s driver head (a File Solid, STL in mm) is drawn from the
  file with the file's extent, in its solver pose to 1e-9 m, and `focus`
  centres the close-up on it; an unknown focus solid is an error.

The ellipsoid path is checked on `GS3DX_Shape` (`tests/test_gs3dx_shape.m`).

## Current Matching and Historical Examples

New matching defaults to `GS3DX_Human`, the refined ellipsoid model.
Use `gs3dx_match_export` with a configured capture registry for shareable
outputs; the maintained research reference records capture identity, actual
geometry, timing, pose-driving mode and validation limits. The explicit
`GS3DX_Fit`/`GS3DX_FitBalance` example above reproduces the historical model
progression. It is not the standard for new matching outputs.
