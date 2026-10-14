"""Render worker: OpenSim's simbody-visualizer inside a virtual X server.

Run only through ``xvfb-run`` (the backend does this): the worker refuses to
start unless ``NATIVE_VIEWER_XVFB=1`` and ``DISPLAY`` is not the real
display ``:0``. Frames are grabbed with ``xwd`` from the visualizer window.
"""

from __future__ import annotations

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
from src.shared.python.golf_view_presets import (
    VIEWER_FOV_Y_RAD,
    get_view_preset,
    simbody_camera_transform,
)
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.ball import BALL_RGBA, BallPlacement
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
_GREY = (0.75, 0.78, 0.85)
_SHAPE_GREY = (0.7, 0.72, 0.8)
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


def _attach(
    osim: Any, body: Any, rot: np.ndarray, pos: Any, geom: Any, rgb: Any, n: int
) -> None:
    from scipy.spatial.transform import Rotation

    frame = osim.PhysicalOffsetFrame(
        f"vf{n}", body, osim.Transform(osim.Vec3(*map(float, pos)))
    )
    frame.set_orientation(
        osim.Vec3(*map(float, Rotation.from_matrix(rot).as_euler("XYZ")))
    )
    geom.setColor(osim.Vec3(*rgb))
    frame.attachGeometry(geom)
    geom.thisown = False
    body.addComponent(frame)
    frame.thisown = False


def build_model(osim: Any, spec_bytes: bytes) -> tuple[Any, float]:
    """OpenSim model of the specification with the shared visual skeleton attached."""
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        clean_osim_body_name,
        export_full_body_osim,
    )

    club_visuals.register_geometry_path()
    skeleton = derive_visual_skeleton(json.loads(spec_bytes))
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
    n = 0
    # The shared visual head replaces the head capsule (visual only).
    head_dir = tempfile.mkdtemp(prefix="ud_head_")
    heads = head_mesh_files(json.loads(spec_bytes), Path(head_dir))
    head_body = heads[0].body if heads else None
    for head in heads:
        body = bodies.get(clean_osim_body_name(head.body))
        _attach(
            osim,
            body,
            np.eye(3),
            (0.0, 0.0, 0.0),
            osim.Mesh(str(head.path)),
            head.rgba[:3],
            n,
        )
        n += 1
    for cap in skeleton.capsules:
        if cap.body == head_body:
            continue
        body = bodies.get(clean_osim_body_name(cap.body))
        a, b = np.array(cap.start_m), np.array(cap.end_m)
        rot = y_axis_rotation(b - a)
        _attach(
            osim,
            body,
            rot,
            (a + b) / 2,
            osim.Cylinder(cap.radius_m, np.linalg.norm(b - a) / 2),
            _GREY,
            n,
        )
        for end in (a, b):
            n += 1
            _attach(osim, body, np.eye(3), end, osim.Sphere(cap.radius_m), _GREY, n)
        n += 1
    club = club_parts(json.loads(spec_bytes))
    for shp in skeleton.shapes:
        if club is not None and shp.body == club[0]:
            continue  # the exported model carries the club meshes instead
        body = bodies.get(clean_osim_body_name(shp.body))
        half = [float(v) for v in shp.half_size_m]
        geom = (
            osim.Ellipsoid(*half)
            if shp.kind == "ellipsoid"
            else osim.Brick(osim.Vec3(*half))
        )
        _attach(osim, body, np.eye(3), shp.center_m, geom, _SHAPE_GREY, n)
        n += 1
    return model, float(skeleton.ground.height_m)


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
    model, ground_h = build_model(osim, bundle.spec_bytes)
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
