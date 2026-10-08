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
from src.shared.python.golf_view_presets import simbody_camera_transform
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.backends._club import club_parts
from src.tools.native_viewer_export.backends._head import head_mesh_files
from src.tools.native_viewer_export.backends._scene import (
    fit_to_size,
    y_axis_rotation,
)
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.overlay2d import draw_glyphs_rgb, pinhole_for_view

UI_STRIP_PX = 40
FOV_Y_RAD = 0.7
SETTLE_S = 0.6
_GREY = (0.75, 0.78, 0.85)
_SHAPE_GREY = (0.7, 0.72, 0.8)


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


def set_camera(osim: Any, viz: Any, view: str, job: WorkerJob) -> None:
    """Point the simbody camera for a golf view preset."""
    rows, pos = simbody_camera_transform(view, job.lookat_m, job.distance_m)
    mat = osim.Mat33()
    for i in range(3):
        for j in range(3):
            mat.set(i, j, float(rows[i][j]))
    viz.setCameraTransform(osim.Transform(osim.Rotation(mat), osim.Vec3(*pos)))


def main(job_path: str) -> None:
    require_virtual_display()
    import opensim as osim

    job = WorkerJob.load(Path(job_path))
    bundle = InputBundle.load(Path(job.bundle_path))
    q = np.load(job.q_path)
    model, ground_h = build_model(osim, bundle.spec_bytes)
    model.setUseVisualizer(True)
    state = model.initSystem()
    viz = model.updVisualizer().updSimbodyVisualizer()
    viz.setBackgroundType(viz.GroundAndSky)
    viz.setShowFrameRate(False)
    viz.setShowSimTime(False)
    viz.setSystemUpDirection(osim.CoordinateDirection(osim.CoordinateAxis(2)))
    viz.setGroundHeight(ground_h)
    viz.setCameraFieldOfView(FOV_Y_RAD)
    coords = model.getCoordinateSet()
    glyph_sets = job.load_glyph_sets()
    time.sleep(3.0)
    for pos, k in enumerate(job.indices):
        for name, value in zip(bundle.coordinate_order, q[k], strict=True):
            coords.get(name).setValue(state, float(value), False)
        model.realizePosition(state)
        for view in job.views:
            set_camera(osim, viz, view, job)
            viz.drawFrameNow(state)
            time.sleep(SETTLE_S)
            window = grab_window()
            if glyph_sets is not None:
                cam = pinhole_for_view(
                    view,
                    job.lookat_m,
                    job.distance_m,
                    FOV_Y_RAD,
                    (window.shape[1], window.shape[0]),
                )
                window = draw_glyphs_rgb(window, glyph_sets[pos], cam)
            # drop the visualizer's own toolbar strip (drawn after overlays so the
            # projection still matches the full window)
            frame = fit_to_size(window[:-UI_STRIP_PX], job.width, job.height)
            np.save(job.frame_path(view, pos), frame)


if __name__ == "__main__":
    main(sys.argv[1])
