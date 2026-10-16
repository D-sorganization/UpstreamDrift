"""Render worker: OpenSim's simbody-visualizer inside a virtual X server.

Run only through ``xvfb-run`` (the backend does this): the worker refuses to
start unless ``NATIVE_VIEWER_XVFB=1`` and ``DISPLAY`` is not the real
display ``:0``. Frames are grabbed with ``xwd`` from the visualizer window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import club_visuals
from src.shared.python.force_overlay.glyphs import GlyphSet
from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.body_meshes import (
    ResolvedAppearance,
    resolve_appearance,
)
from src.shared.python.model_appearance.schema import AppearanceDocument
from src.shared.python.golf_view_presets import (
    VIEWER_FOV_Y_RAD,
    get_view_preset,
    simbody_camera_transform,
)
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.ball import BALL_RGBA, BallPlacement
from src.tools.native_viewer_export.backends._body import (
    body_mesh_files,
    default_appearance,
)
from src.tools.native_viewer_export.backends._club import club_parts
from src.tools.native_viewer_export.backends._opensim_framing import (
    fitted_distance_m,
    fitted_lookat_m,
    framing_body_names,
    principal_crop,
    simbody_fov_y_rad,
    sole_points,
)
from src.tools.native_viewer_export.backends._head import head_mesh_files
from src.tools.native_viewer_export.backends._scene import (
    fit_to_size,
    y_axis_rotation,
)
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.overlay2d import draw_glyphs_rgb, pinhole_for_view
from src.shared.python.engine_core.opensim_runtime import import_opensim

UI_STRIP_PX = 40
FOV_Y_RAD = VIEWER_FOV_Y_RAD
SETTLE_S = 0.6
BALL_RGB = BALL_RGBA[:3]
BALL_BODY = "ud_ball"
BALL_BODY_MASS_KG = 1e-6
#: slide of the hidden ball (m): a 21 mm sphere 1 km away is sub-pixel
BALL_HIDE_OFFSET_M = 1000.0


def require_virtual_display() -> str:
    """Return ``DISPLAY`` after asserting it is a virtual (xvfb) display."""
    display = os.environ.get("DISPLAY", "")
    if os.environ.get("NATIVE_VIEWER_XVFB") != "1":
        raise RuntimeError("refusing to run outside the xvfb launcher")
    if not display or display == ":0" or display.startswith(":0."):
        raise RuntimeError(f"refusing to use the real display {display!r}")
    return display


def grab_window() -> np.ndarray:
    """RGB image of the OpenSim visualizer window (via xwininfo + xwd)."""
    wid = None
    for _ in range(20):
        tree = subprocess.run(  # noqa: S603, S607 - fixed argv
            ["xwininfo", "-root", "-tree"], capture_output=True, text=True, check=False
        ).stdout
        ids = [line.split()[0] for line in tree.splitlines() if "OpenSim" in line]
        if ids:
            wid = ids[0]
            break
        time.sleep(0.5)
    if wid is None:
        raise RuntimeError("OpenSim visualizer window not found")
    raw = subprocess.run(  # noqa: S603, S607 - fixed argv
        ["xwd", "-id", wid, "-silent"], capture_output=True, check=True
    ).stdout
    head = struct.unpack(">25I", raw[:100])
    header_size, width, height, bytes_per_line, n_colors = (
        head[0],
        head[4],
        head[5],
        head[12],
        head[19],
    )
    offset = header_size + n_colors * 12
    flat = np.frombuffer(raw[offset : offset + bytes_per_line * height], np.uint8)
    pix = flat.reshape(height, bytes_per_line)[:, : width * 4].reshape(height, width, 4)
    return np.ascontiguousarray(pix[:, :, [2, 1, 0]])


@dataclass(frozen=True)
class VisualPart:
    """One visual geometry on one body, in the spec (= OpenSim) body frame.

    ``geometry`` is ``mesh`` (``params = (obj path,)``), ``cylinder``
    (radius, half length), ``sphere`` (radius), ``ellipsoid`` or ``brick``
    (half sizes); ``rgba`` comes from ``model_appearance.library`` only.
    """

    body: str
    geometry: str
    params: tuple[Any, ...]
    rotation: np.ndarray
    position: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


_ORIGIN = (0.0, 0.0, 0.0)


def _point(v: Any) -> tuple[float, float, float]:
    """A 3-vector as a float triple. Precondition: exactly three components."""
    x, y, z = (float(c) for c in v)
    return (x, y, z)


def _plain_parts(
    skeleton: Any, skip: set[str], club_body: str | None
) -> list[VisualPart]:
    """The un-styled capsule skeleton and hint shapes in the library plain palette."""
    capsule, shape = library.plain_rgba("capsule"), library.plain_rgba("shape")
    out = []
    for cap in skeleton.capsules:
        if cap.body in skip:
            continue
        a, b = np.array(cap.start_m), np.array(cap.end_m)
        half = float(np.linalg.norm(b - a)) / 2
        mid = _point((a + b) / 2)
        out.append(
            VisualPart(
                cap.body,
                "cylinder",
                (cap.radius_m, half),
                y_axis_rotation(b - a),
                mid,
                capsule,
            )
        )
        for end in (a, b):
            pos = _point(end)
            out.append(
                VisualPart(cap.body, "sphere", (cap.radius_m,), np.eye(3), pos, capsule)
            )
    for shp in skeleton.shapes:
        if shp.body == club_body:
            continue  # the exported model carries the club meshes instead
        half_size = tuple(float(v) for v in shp.half_size_m)
        pos = _point(shp.center_m)
        out.append(VisualPart(shp.body, shp.kind, half_size, np.eye(3), pos, shape))
    return out


def _dressed_parts(
    resolved: ResolvedAppearance, directory: Path, club_body: str | None
) -> list[VisualPart]:
    """R8's resolved body meshes as OBJ mesh geometry, untransformed.

    The OpenSim exporter attaches each joint's ``child_to_follower`` as an
    offset frame on the child body, so the OpenSim body frame IS the spec
    body frame: the identity is the correct placement (unlike MuJoCo's
    follower-frame body, which needs ``inv(child_to_follower)``).
    """
    return [
        VisualPart(f.body, "mesh", (str(f.path),), np.eye(3), _ORIGIN, f.rgba)
        for f in body_mesh_files(resolved, directory)
        if f.body != club_body  # the exported model carries the club meshes
    ]


def visual_parts(
    spec: Mapping[str, Any],
    appearance: AppearanceDocument | None,
    directory: Path,
) -> list[VisualPart]:
    """Every visual part of the render model: head, then body meshes or skeleton.

    ``appearance`` dresses the bodies with the shared resolved meshes
    (R9b, #12689); ``None`` is the plain mode (capsules and hint shapes in
    the library plain palette). OBJ files are written under ``directory``.
    Postcondition: no colour outside ``model_appearance.library``.
    """
    heads = head_mesh_files(dict(spec), directory / "head")
    out = [
        VisualPart(h.body, "mesh", (str(h.path),), np.eye(3), _ORIGIN, h.rgba)
        for h in heads
    ]
    club = club_parts(dict(spec))
    club_body = club[0] if club is not None else None
    if appearance is None:
        skip = {heads[0].body} if heads else set()
        return out + _plain_parts(derive_visual_skeleton(spec), skip, club_body)
    resolved = resolve_appearance(spec, appearance)
    return out + _dressed_parts(resolved, directory / "body", club_body)


def _geometry(osim: Any, part: VisualPart) -> Any:
    """The OpenSim geometry object of one visual part."""
    if part.geometry == "mesh":
        return osim.Mesh(part.params[0])
    if part.geometry == "cylinder":
        return osim.Cylinder(*part.params)
    if part.geometry == "sphere":
        return osim.Sphere(*part.params)
    if part.geometry == "ellipsoid":
        return osim.Ellipsoid(*part.params)
    if part.geometry == "brick":
        return osim.Brick(osim.Vec3(*part.params))
    raise ValueError(f"unknown visual geometry {part.geometry!r}")


def _attach(osim: Any, body: Any, part: VisualPart, n: int) -> None:
    from scipy.spatial.transform import Rotation

    frame = osim.PhysicalOffsetFrame(
        f"vf{n}", body, osim.Transform(osim.Vec3(*map(float, part.position)))
    )
    frame.set_orientation(
        osim.Vec3(*map(float, Rotation.from_matrix(part.rotation).as_euler("XYZ")))
    )
    geom = _geometry(osim, part)
    geom.setColor(osim.Vec3(*part.rgba[:3]))
    geom.setOpacity(float(part.rgba[3]))
    frame.attachGeometry(geom)
    geom.thisown = False
    body.addComponent(frame)
    frame.thisown = False


def build_model(
    osim: Any,
    spec_bytes: bytes,
    appearance: AppearanceDocument | None = None,
) -> tuple[Any, float]:
    """OpenSim model of the specification with its visual parts attached.

    ``appearance`` selects the dressed bodies (see :func:`visual_parts`);
    ``None`` keeps the plain skeleton. Visual only: no mass or joint changes.
    """
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        clean_osim_body_name,
        export_full_body_osim,
    )

    club_visuals.register_geometry_path()
    spec = json.loads(spec_bytes)
    xml, _ = export_full_body_osim(spec_bytes)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.osim"
        path.write_text(xml, encoding="utf-8")
        model = osim.Model(str(path))
    osim.Logger.setLevelString("Warn")
    for clear in (
        "updMarkerSet",
        "updForceSet",
        "updConstraintSet",
        "updContactGeometrySet",
    ):
        getattr(model, clear)().clearAndDestroy()
    model.finalizeFromProperties()
    bodies = model.getBodySet()
    mesh_dir = Path(tempfile.mkdtemp(prefix="ud_visual_"))
    for n, part in enumerate(visual_parts(spec, appearance, mesh_dir)):
        _attach(osim, bodies.get(clean_osim_body_name(part.body)), part, n)
    return model, float(derive_visual_skeleton(spec).ground.height_m)


def attach_ball(osim: Any, model: Any, ball: dict[str, Any]) -> str:
    """Attach the decorative ball to this render model; return its slide coordinate.

    The simbody visualizer caches fixed geometry at ``initSystem`` (toggling
    Appearance visibility afterwards does not redraw), but it redraws body
    poses from the state every frame. So the sphere rides a tiny body on a
    slider joint from ground at the ball position: slide 0 is the address
    ball, :data:`BALL_HIDE_OFFSET_M` puts it out of view. Visual only: this
    model is built per render and never integrated.
    """
    placement = BallPlacement.from_dict(ball)
    tiny = BALL_BODY_MASS_KG
    body = osim.Body(BALL_BODY, tiny, osim.Vec3(0), osim.Inertia(tiny, tiny, tiny))
    joint = osim.SliderJoint(
        f"{BALL_BODY}_slide",
        model.updGround(),
        osim.Vec3(*map(float, placement.position_m)),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    geom = osim.Sphere(placement.radius_m)
    geom.setColor(osim.Vec3(*BALL_RGB))
    body.attachGeometry(geom)
    geom.thisown = False
    model.addBody(body)
    body.thisown = False
    model.addJoint(joint)
    joint.thisown = False
    return str(joint.get_coordinates(0).getName())


def ball_slide_m(visible: list[bool] | None, position: int) -> float:
    """Slide (m) of the ball body at frame ``position`` of the job.

    ``visible`` is the job's per-position ball visibility; ``None`` keeps the
    ball in place. Raises ``ValueError`` for a position outside the list.
    """
    if visible is None:
        return 0.0
    if not 0 <= position < len(visible):
        raise ValueError(f"position {position} outside {len(visible)} frames")
    return 0.0 if visible[position] else BALL_HIDE_OFFSET_M


def collect_framing_points(
    model: Any,
    state: Any,
    coordinate_names: list[str],
    q: np.ndarray,
    ground_h: float,
) -> np.ndarray:
    """``(n, 3)`` foot, head and clubhead points (plus soles) over every row of ``q``.

    Postcondition: finite points, ``2 * rows * framing bodies`` of them.
    """
    coords = model.getCoordinateSet()
    bodies = model.getBodySet()
    names = framing_body_names(
        [bodies.get(i).getName() for i in range(bodies.getSize())]
    )
    if not names:
        raise ValueError("model has no foot, head or clubhead body to frame")
    out: list[list[float]] = []
    for row in q:
        for name, value in zip(coordinate_names, row, strict=True):
            coords.get(name).setValue(state, float(value), False)
        model.realizePosition(state)
        for body in names:
            p = bodies.get(body).getPositionInGround(state)
            out.append([p.get(0), p.get(1), p.get(2)])
    return sole_points(np.asarray(out, dtype=float), ground_h)


def set_camera(
    osim: Any,
    viz: Any,
    view: str,
    distance_m: float | None,
    lookat_m: list[float],
) -> None:
    """Point the simbody camera for a golf view preset at ``lookat_m``."""
    rows, pos = simbody_camera_transform(view, lookat_m, distance_m)
    mat = osim.Mat33()
    for i in range(3):
        for j in range(3):
            mat.set(i, j, float(rows[i][j]))
    viz.setCameraTransform(osim.Transform(osim.Rotation(mat), osim.Vec3(*pos)))


def main(job_path: str) -> None:
    require_virtual_display()
    osim = import_opensim()

    job = WorkerJob.load(Path(job_path))
    bundle = InputBundle.load(Path(job.bundle_path))
    q = np.load(job.q_path)
    model, ground_h = build_model(osim, bundle.spec_bytes, default_appearance())
    ball_coord = attach_ball(osim, model, job.ball) if job.ball else None
    model.setUseVisualizer(True)
    state = model.initSystem()
    viz = model.updVisualizer().updSimbodyVisualizer()
    viz.setBackgroundType(viz.GroundAndSky)
    viz.setShowFrameRate(False)
    viz.setShowSimTime(False)
    viz.setSystemUpDirection(osim.CoordinateDirection(osim.CoordinateAxis(2)))
    viz.setGroundHeight(ground_h)
    coords = model.getCoordinateSet()
    glyph_sets = job.load_glyph_sets()
    time.sleep(3.0)
    # The window is not the clip's aspect: crop about the principal point and set
    # the simbody lens so the visible crop has the shared viewer lens (#12308).
    viz.drawFrameNow(state)
    time.sleep(SETTLE_S)
    window_hw = grab_window().shape[:2]
    y0, y1, x0, x1 = principal_crop(window_hw, UI_STRIP_PX, (job.width, job.height))
    fov = simbody_fov_y_rad(FOV_Y_RAD, window_hw[0], y1 - y0)
    viz.setCameraFieldOfView(fov)
    points = collect_framing_points(
        model, state, list(bundle.coordinate_order), q[job.indices], ground_h
    )
    aspect = job.width / job.height
    # fixed views are fitted to the trajectory unless the caller fixed the distance
    fitted = [
        v
        for v in job.views
        if get_view_preset(v).tracks is None and job.distance_m is None
    ]
    fit_looks = {v: fitted_lookat_m(v, points, job.lookat_m) for v in fitted}
    distances = {
        v: fitted_distance_m(v, points, fit_looks[v], None, FOV_Y_RAD, aspect)
        if v in fit_looks
        else job.distance_m
        for v in job.views
    }
    for pos, k in enumerate(job.indices):
        for name, value in zip(bundle.coordinate_order, q[k], strict=True):
            coords.get(name).setValue(state, float(value), False)
        if ball_coord is not None:
            slide = ball_slide_m(job.ball_visible, pos)
            coords.get(ball_coord).setValue(state, slide, False)
        model.realizePosition(state)
        for view in job.views:
            look = fit_looks.get(view) or job.lookat_for(view, pos)
            set_camera(osim, viz, view, distances[view], look)
            viz.drawFrameNow(state)
            time.sleep(SETTLE_S)
            window = grab_window()
            if glyph_sets is not None:
                cam = pinhole_for_view(
                    view,
                    look,
                    distances[view],
                    fov,
                    (window.shape[1], window.shape[0]),
                )
                window = draw_glyphs_rgb(window, glyph_sets[pos], cam)
            # crop about the principal point, clear of the visualizer's toolbar
            # strip (drawn after overlays so the projection matches the window)
            frame = fit_to_size(window[y0:y1, x0:x1], job.width, job.height)
            np.save(job.frame_path(view, pos), frame)


if __name__ == "__main__":
    main(sys.argv[1])
