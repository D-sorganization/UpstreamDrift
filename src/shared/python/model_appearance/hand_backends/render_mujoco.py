"""MuJoCo offscreen engine of the hand renderer (OSMesa, then EGL).

A failed OSMesa import poisons PyOpenGL for the rest of the process, so each
backend is probed in a child process first and only the winner is imported
here. The engine opens ``mujoco.osmesa.GLContext`` or ``mujoco.egl.GLContext``
itself instead of going through ``mujoco.Renderer``: that class uses whatever
platform MuJoCo bound at import (GLFW on a default Linux import, which cannot
render headless), while the explicit context works either way.

Skin shading is smooth because every skin mesh is compiled with explicit,
area-weighted vertex normals. The headlight is off; the key, fill and rim
lights come from the camera-space settings. The render context and the GL
context are always freed explicitly, because the EGL destructor raises at
interpreter teardown otherwise.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure
from src.shared.python.model_appearance.hand_backends.render_rig import (
    CameraRig,
    ScenePart,
    corner_normals,
)
from src.shared.python.model_appearance.hand_backends.render_settings import (
    GL_ENGINES,
    PART_KINDS,
    HandRenderSettings,
)

logger = logging.getLogger(__name__)

#: Mask id of each part kind (zero is the background).
_KIND_ID = {kind: index + 1 for index, kind in enumerate(PART_KINDS)}

_PROBE_SOURCE = """
import importlib
import sys
import numpy as np
import mujoco
backend, px = sys.argv[1], int(sys.argv[2])
context_type = importlib.import_module("mujoco." + backend).GLContext
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom size="1"/></worldbody></mujoco>'
)
gl = context_type(px, px)
try:
    gl.make_current()
    context = mujoco.MjrContext(model, int(mujoco.mjtFontScale.mjFONTSCALE_150))
    try:
        mujoco.mjr_setBuffer(int(mujoco.mjtFramebuffer.mjFB_OFFSCREEN), context)
        pixels = np.empty((px, px, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, px, px), context)
    finally:
        context.free()
finally:
    gl.free()
"""

#: Probe results, keyed by the candidate order: ``(backend or None, errors)``.
_PROBE_CACHE: dict[tuple[str, ...], tuple[str | None, dict[str, str]]] = {}


class GLUnavailableError(RuntimeError):
    """No MuJoCo GL context could be created. The message holds each error."""


def _probe_child(backend: str, settings: HandRenderSettings) -> str | None:
    """``None`` when a child process rendered with ``backend``, else the error."""
    env = dict(os.environ, MUJOCO_GL=backend)
    env.pop("PYOPENGL_PLATFORM", None)
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _PROBE_SOURCE, backend, str(settings.gl_probe_px)],
            env=env,
            capture_output=True,
            text=True,
            timeout=settings.gl_probe_timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}"
    if done.returncode == 0:
        return None
    lines = [line for line in done.stderr.strip().splitlines() if line.strip()]
    return lines[-1] if lines else f"probe exited with {done.returncode}"


def probe_gl_backend(
    settings: HandRenderSettings,
) -> tuple[str | None, dict[str, str]]:
    """First usable MuJoCo GL backend in ``settings.engine_order``.

    Returns ``(backend, errors)``; ``backend`` is ``None`` when none works and
    ``errors`` maps each rejected backend to its error text. The result is
    cached. When MuJoCo is not imported yet, ``MUJOCO_GL`` is set to the
    winner so the import itself cannot fail on a missing platform library.
    """
    candidates = tuple(name for name in settings.engine_order if name in GL_ENGINES)
    if candidates in _PROBE_CACHE:
        return _PROBE_CACHE[candidates]
    errors: dict[str, str] = {}
    chosen: str | None = None
    for name in candidates:
        error = _probe_child(name, settings)
        if error is None:
            chosen = name
            break
        errors[name] = error
    if chosen is not None and "mujoco" not in sys.modules:
        os.environ["MUJOCO_GL"] = chosen
    _PROBE_CACHE[candidates] = (chosen, errors)
    logger.info("hand renderer GL backend: %s (rejected: %s)", chosen, errors)
    return chosen, errors


def _fmt(values: Any) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(values, dtype=float).reshape(-1))


def _mesh_asset(name: str, part: ScenePart, settings: HandRenderSettings) -> str:
    mesh = part.mesh
    if part.kind != "skin":
        faces = " ".join(str(int(i)) for i in np.asarray(mesh.faces).reshape(-1))
        return (
            f'<mesh name="{name}" vertex="{_fmt(mesh.vertices)}" '
            f'face="{faces}" inertia="shell"/>'
        )
    normals = corner_normals(
        mesh,
        weld_tolerance_m=settings.normal_weld_tolerance_m,
        crease_deg=settings.normal_crease_deg,
    )
    corners = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces = " ".join(str(i) for i in range(len(corners)))
    return (
        f'<mesh name="{name}" vertex="{_fmt(corners)}" normal="{_fmt(normals)}" '
        f'face="{faces}" inertia="shell"/>'
    )


def _light_xml(rig: CameraRig, settings: HandRenderSettings) -> str:
    ambient = np.asarray(settings.ambient_rgb) + (
        settings.wrap * settings.sss_strength * np.asarray(settings.sss_tint)
    )
    out = []
    for index, light in enumerate(settings.lights.values()):
        travel = -rig.light_to_world(light.direction)
        colour = np.asarray(light.color)
        shadow = settings.shadows and light.cast_shadow
        out.append(
            f'<light name="{light.name}" directional="true" '
            f'pos="{_fmt(rig.target - travel * rig.distance)}" dir="{_fmt(travel)}" '
            f'diffuse="{_fmt(colour * light.intensity)}" '
            f'specular="{_fmt(colour * light.specular)}" '
            f'ambient="{_fmt(ambient if index == 0 else np.zeros(3))}" '
            f'castshadow="{"true" if shadow else "false"}"/>'
        )
    return "".join(out)


def scene_xml(
    parts: Sequence[ScenePart], rig: CameraRig, settings: HandRenderSettings
) -> str:
    """MJCF of the scene: mesh geoms, a fixed camera, three lights, no headlight."""
    size = settings.frame_px * settings.supersample
    points = np.vstack([p.mesh.vertices for p in parts])
    centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    reach = float(np.linalg.norm(rig.eye - centre)) + extent
    assets, geoms = [], []
    for kind in PART_KINDS:
        surface = settings.surface[kind]
        assets.append(
            f'<material name="surface_{kind}" specular="{surface.specular!r}" '
            f'shininess="{surface.shininess!r}"/>'
        )
    for index, part in enumerate(parts):
        assets.append(_mesh_asset(f"mesh_{index}", part, settings))
        geoms.append(
            f'<geom name="{part.kind}:{index}" type="mesh" mesh="mesh_{index}" '
            f'material="surface_{part.kind}" rgba="{_fmt(part.rgba)}" '
            'contype="0" conaffinity="0"/>'
        )
    background = _fmt(settings.background_rgba[:3])
    return (
        "<mujoco>"
        f'<statistic center="{_fmt(centre)}" extent="{extent!r}"/>'
        "<visual>"
        f'<global offwidth="{size}" offheight="{size}"/>'
        f'<quality offsamples="{settings.msaa_samples}" '
        f'shadowsize="{settings.shadow_map_px}"/>'
        '<headlight active="0"/>'
        f'<map znear="{settings.znear_ratio * rig.distance / extent!r}" '
        f'zfar="{settings.zfar_ratio * reach / extent!r}"/>'
        "</visual>"
        "<asset>"
        f'<texture name="backdrop" type="skybox" builtin="flat" rgb1="{background}" '
        f'rgb2="{background}" width="{settings.backdrop_texture_px}" '
        f'height="{settings.backdrop_texture_px}"/>' + "".join(assets) + "</asset>"
        "<worldbody>"
        + _light_xml(rig, settings)
        + f'<camera name="rig" pos="{_fmt(rig.eye)}" '
        f'xyaxes="{_fmt(rig.right)} {_fmt(rig.up)}" fovy="{settings.fov_deg!r}"/>'
        + "".join(geoms)
        + "</worldbody></mujoco>"
    )


def _capture(
    model: Any, size: int, backend: str, *, segmentation: bool
) -> NDArray[np.uint8]:
    """One offscreen frame, top row first, through an explicit GL context."""
    import mujoco

    try:
        context_type = importlib.import_module(f"mujoco.{backend}").GLContext
    except Exception as exc:  # noqa: BLE001 - PyOpenGL raises many types here
        raise GLUnavailableError(
            f"cannot load the {backend!r} GL platform in this process: {exc}"
        ) from exc
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rig")
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom + model.nlight + model.ncam)
    scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = int(segmentation)
    scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = int(segmentation)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        camera,
        int(mujoco.mjtCatBit.mjCAT_ALL),
        scene,
    )
    viewport = mujoco.MjrRect(0, 0, size, size)
    pixels = np.empty((size, size, 3), dtype=np.uint8)
    gl = context_type(size, size)
    try:
        gl.make_current()
        context = mujoco.MjrContext(model, int(mujoco.mjtFontScale.mjFONTSCALE_150))
        try:
            mujoco.mjr_setBuffer(int(mujoco.mjtFramebuffer.mjFB_OFFSCREEN), context)
            mujoco.mjr_render(viewport, scene, context)
            mujoco.mjr_readPixels(pixels, None, viewport, context)
        finally:
            context.free()
    finally:
        gl.free()
    if not segmentation:
        return np.flipud(pixels)
    packed = pixels.astype(np.int64) @ (np.iinfo(np.uint8).max + 1) ** np.arange(3)
    kind_of_segment = np.zeros(scene.ngeom + 1, dtype=np.uint8)
    for segment in range(scene.ngeom):
        geom = scene.geoms[segment]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom.objid)
            kind_of_segment[segment + 1] = _KIND_ID[name.split(":")[0]]
    return np.flipud(kind_of_segment[np.clip(packed, 0, scene.ngeom)])


def render_mujoco(
    parts: Sequence[ScenePart],
    rig: CameraRig,
    settings: HandRenderSettings,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], str]:
    """Render ``parts`` through MuJoCo. Returns ``(image, kind_mask, backend)``.

    Raises :class:`GLUnavailableError` when no GL context can be created.
    ``kind_mask`` comes from a second, non-multisampled segmentation pass.
    """
    backend, errors = probe_gl_backend(settings)
    if backend is None:
        raise GLUnavailableError(f"no MuJoCo GL backend rendered: {errors}")
    import mujoco

    model = mujoco.MjModel.from_xml_string(scene_xml(parts, rig, settings))
    scale = settings.supersample
    size = settings.frame_px * scale
    colour = _capture(model, size, backend, segmentation=False)
    quality = model.vis.quality
    quality.offsamples = 0
    mask = _capture(model, size, backend, segmentation=True)
    if scale > 1:
        frame = settings.frame_px
        blocks = colour.reshape(frame, scale, frame, scale, 3).astype(np.float64)
        colour = np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)
        mask = mask[::scale, ::scale]
    image = np.ascontiguousarray(colour, dtype=np.uint8)
    ensure(
        image.shape == (settings.frame_px, settings.frame_px, 3),
        "the MuJoCo frame is frame_px square RGB",
    )
    return image, np.ascontiguousarray(mask, dtype=np.uint8), backend
