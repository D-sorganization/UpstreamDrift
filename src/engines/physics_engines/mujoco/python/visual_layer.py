"""MuJoCo translation of the shared visual skeleton.

Adds capsule/sphere geoms in visual group 1 with collisions disabled and no
mass, a ground plane opposite gravity, lights and a default camera. The
physics model (bodies, joints, inertias, contact spheres, closure sites) is
untouched; the exporter's plain output remains the qualified representation.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.model_appearance.ball import BALL_RADIUS_M, resolve_ball_visual
from src.shared.python.model_appearance.club_assembly import (
    ClubAssembly,
    assembly_from_spec,
    club_body_name,
    clubface_centre,
    clubface_vector,
)
from src.shared.python.model_appearance.schema import AppearanceDocument, BallSettings
from src.shared.python.motion_matching.visual_skeleton import (
    VisualSkeleton,
    derive_visual_skeleton,
)
from src.shared.python.video_timing.frame_schedule import DEFAULT_FPS, FrameSchedule

_VISUAL_CLASS = "visual"
_CAPSULE_RGBA = "0.75 0.78 0.85 1"
_SHAPE_RGBA = "0.7 0.72 0.8 1"
# Centre-of-mass and frame spheres sit in a group the renderer hides by
# default (groups 0 to 2 are shown); enable group 3 to inspect them.
_MARKER_SPHERE_GROUP = "3"
_COM_RGBA = "0.9 0.35 0.2 1"
_FRAME_RGBA = "0.2 0.6 0.95 1"
_FLOOR_RGBA = "0.35 0.45 0.3 1"
_BALL_RGBA = "0.95 0.95 0.95 1"


def _numbers(values: Any) -> str:
    return " ".join(format(float(v), ".9g") for v in np.asarray(values).ravel())


def _to_mjcf_frame(offset: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Map a point from the spec body frame into the MJCF body frame."""
    return offset[:3, :3] @ point + offset[:3, 3]


def _plane_quat(normal: np.ndarray) -> str:
    """Quaternion rotating MuJoCo's +z plane normal onto ``normal`` (wxyz)."""
    z = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    axis = np.cross(z, n)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(z, n))
    if s < 1e-12:
        return "1 0 0 0" if c > 0 else "0 1 0 0"
    axis /= s
    half = float(np.arctan2(s, c)) / 2.0
    return _numbers([np.cos(half), *(np.sin(half) * axis)])


def _club_face_world(
    club: ClubAssembly, club_body: str, offsets: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """World face centre and face normal of ``club`` at the exporter's reference pose.

    Uses the same ``offsets[club_body]`` transform the club head mesh itself
    is drawn with, so the decorative ball is placed consistently with
    whatever pose this exporter renders (GCV-13, #11719).
    """
    offset = offsets[club_body]
    centre = offset[:3, :3] @ clubface_centre(club) + offset[:3, 3]
    normal = offset[:3, :3] @ clubface_vector(club)
    return centre, normal


# A clubhead sitting higher than this above the ground cannot be an address
# stance (real clubs rest within a few centimetres of the ground). The
# exporter's static reference pose is a kinematic rest configuration, not
# necessarily the biomechanical address pose (OSV-8, #11755), so the
# geometric fallback below is only trusted when it looks grounded; otherwise
# the caller must resolve the real address pose and pass ``ball.position_m``.
_MAX_ADDRESS_HEIGHT_M = 0.5


def _attach_decorative_ball(
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    club: ClubAssembly | None,
    club_body: str | None,
    ball: BallSettings,
    ground_height_m: float,
) -> dict[str, Any]:
    """Attach the decorative address ball (GCV-13, #11719), or report why not.

    The ball is a massless, non-colliding sphere fixed to the world body:
    adding it changes no dynamics and it never moves on its own (an optional
    post-impact launch is future work, GCV-15). Returns a summary dict with
    ``enabled`` and either ``position_m``/``source`` or a ``reason`` the ball
    was skipped — "unavailable" is reported, never silently drawn as zero.

    ``ball.position_m`` (e.g. a mocap-matched swing's measured ball) is used
    verbatim regardless of this exporter's reference pose. Without it, the
    face centre at the reference pose is used only when it is plausibly
    grounded (within :data:`_MAX_ADDRESS_HEIGHT_M` of ``ground_height_m``).
    """
    if club is None or club_body is None or club_body not in elements:
        return {"enabled": False, "reason": "spec has no club body"}
    centre_world, normal_world = _club_face_world(club, club_body, offsets)
    if ball.position_m is None and ball.enabled:
        height_above_ground = abs(float(centre_world[2]) - ground_height_m)
        if height_above_ground > _MAX_ADDRESS_HEIGHT_M:
            return {
                "enabled": False,
                "reason": (
                    "exporter reference pose is not grounded "
                    f"({height_above_ground:.3f} m above ground_height_m); "
                    "supply appearance.ball.position_m from a resolved address pose"
                ),
            }
    resolved = resolve_ball_visual(
        enabled=ball.enabled,
        position_m=ball.position_m,
        source=ball.source,
        face_centre_m=centre_world,
        face_normal=normal_world,
        ground_height_m=ground_height_m,
    )
    if resolved is None:
        return {"enabled": False, "reason": "ball.enabled is False"}
    position, source = resolved
    ET.SubElement(
        elements["world"],
        "geom",
        name="visual_ball",
        type="sphere",
        pos=_numbers(position),
        size=_numbers([BALL_RADIUS_M]),
        rgba=_BALL_RGBA,
        attrib={"class": _VISUAL_CLASS},
    )
    return {"enabled": True, "position_m": position.tolist(), "source": source}


def attach_visual_layer(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    appearance: AppearanceDocument | None = None,
    *,
    with_head: bool = False,
) -> dict[str, Any]:
    """Attach the shared skeleton to an MJCF document; returns a summary.

    With an ``appearance`` document the bare capsules are replaced by smooth
    textured mesh segments, garments, shoes and a club head, plus a skybox,
    textured ground and shadowed lights; still visual only.

    ``elements`` and ``offsets`` are the exporter's per-body MJCF elements and
    spec-body-to-MJCF-body transforms. Every added geom is class ``visual``:
    group 1, zero contype/conaffinity, zero mass.
    """
    skeleton: VisualSkeleton = derive_visual_skeleton(spec)
    default_root = root.find("default")
    if default_root is None:
        default_root = ET.SubElement(root, "default")
    visual_default = ET.SubElement(
        default_root, "default", attrib={"class": _VISUAL_CLASS}
    )
    ET.SubElement(
        visual_default, "geom", contype="0", conaffinity="0", group="1", mass="0"
    )
    appearance_meta: dict[str, Any] = {}
    if appearance is not None:
        from src.engines.physics_engines.mujoco.python.appearance_layer import (
            attach_appearance,
        )

        appearance_meta = attach_appearance(
            root, elements, offsets, skeleton, appearance, assembly_from_spec(spec)
        )
    head_doc = appearance or (AppearanceDocument() if with_head else None)
    if head_doc is not None:  # visible head and neck, visual only (GCV-12)
        from src.engines.physics_engines.mujoco.python.head_visual import (
            attach_head_visual,
        )

        appearance_meta["head"] = attach_head_visual(
            root, elements, offsets, spec, head_doc
        )
    head_meta = appearance_meta.get("head") or {}
    if appearance is not None and head_meta.get("enabled"):
        appearance_meta["meshes"] += len(head_meta["parts"])
    head_body = head_meta["body"] if head_meta.get("source") == "head_body" else None
    for index, capsule in enumerate(
        () if appearance is not None else skeleton.capsules
    ):
        if capsule.body == head_body:
            continue  # the head mesh replaces the head capsule
        offset = offsets[capsule.body]
        start = _to_mjcf_frame(offset, np.asarray(capsule.start_m))
        end = _to_mjcf_frame(offset, np.asarray(capsule.end_m))
        ET.SubElement(
            elements[capsule.body],
            "geom",
            name=f"visual_capsule_{index}_{capsule.body}",
            type="capsule",
            fromto=_numbers(np.concatenate((start, end))),
            size=_numbers([capsule.radius_m]),
            rgba=_CAPSULE_RGBA,
            attrib={"class": _VISUAL_CLASS},
        )
    club = assembly_from_spec(spec)
    club_body = club_body_name(spec)
    mesh_club = appearance is None and club is not None and club_body in elements
    if mesh_club:
        from src.engines.physics_engines.mujoco.python.appearance_layer import (
            attach_club_meshes,
        )

        attach_club_meshes(root, elements, offsets, club_body, club)  # type: ignore[arg-type]
    ball_settings = appearance.ball if appearance is not None else BallSettings()
    ball_meta = _attach_decorative_ball(
        elements, offsets, club, club_body, ball_settings, skeleton.ground.height_m
    )
    for index, shape in enumerate(() if appearance is not None else skeleton.shapes):
        if mesh_club and shape.body == club_body:
            continue  # the mesh head replaces the ellipsoid hint
        ET.SubElement(
            elements[shape.body],
            "geom",
            name=f"visual_{shape.kind}_{index}_{shape.body}",
            type=shape.kind,
            pos=_numbers(
                _to_mjcf_frame(offsets[shape.body], np.asarray(shape.center_m))
            ),
            size=_numbers(shape.half_size_m),
            rgba=_SHAPE_RGBA,
            attrib={"class": _VISUAL_CLASS},
        )
    for i, sphere in enumerate(skeleton.spheres):
        ET.SubElement(
            elements[sphere.body],
            "geom",
            name=f"visual_{sphere.kind}_{i}",
            type="sphere",
            pos=_numbers(
                _to_mjcf_frame(offsets[sphere.body], np.asarray(sphere.center_m))
            ),
            size=_numbers([sphere.radius_m]),
            rgba=_COM_RGBA if sphere.kind == "com" else _FRAME_RGBA,
            group=_MARKER_SPHERE_GROUP,
            attrib={"class": _VISUAL_CLASS},
        )
    world = elements["world"]
    normal = np.asarray(skeleton.ground.normal)
    ET.SubElement(
        world,
        "geom",
        name="visual_floor",
        type="plane",
        size="3 3 0.05",
        pos=_numbers(normal * skeleton.ground.height_m),
        quat=_plane_quat(normal),
        attrib={"class": _VISUAL_CLASS},
        **(
            {"material": appearance_meta["ground_material"]}
            if appearance is not None
            else {"rgba": _FLOOR_RGBA}
        ),
    )
    up = normal * 3.0
    ET.SubElement(
        world,
        "light",
        name="visual_key",
        pos=_numbers(up + np.array([1.5, -1.5, 0.0])),
        dir=_numbers(-(up + np.array([1.5, -1.5, 0.0]))),
        diffuse="0.8 0.8 0.8",
        castshadow="true" if appearance is not None else "false",
    )
    ET.SubElement(
        world,
        "light",
        name="visual_fill",
        pos=_numbers(up + np.array([-2.0, 1.0, 0.0])),
        dir=_numbers(-(up + np.array([-2.0, 1.0, 0.0]))),
        diffuse="0.4 0.4 0.4",
    )
    ET.SubElement(
        world,
        "camera",
        name="visual_default",
        pos=_numbers(normal * 1.2 + np.array([2.8, -2.8, 0.0])),
        mode="targetbody",
        target=next(e.get("name", "") for k, e in elements.items() if k != "world"),
    )
    lights = 2
    if appearance is not None:
        rim = up + np.array([0.0, 2.5, 0.0])
        ET.SubElement(
            world,
            "light",
            name="visual_rim",
            pos=_numbers(rim),
            dir=_numbers(-rim),
            diffuse="0.45 0.47 0.55",
            castshadow="false",
        )
        lights = 3
    return {
        **appearance_meta,
        "capsules": len(skeleton.capsules),
        "spheres": len(skeleton.spheres),
        "floor": True,
        "ground_calibrated": skeleton.ground.calibrated,
        "lights": lights,
        "ball": ball_meta,
    }


def whole_body_com(model: Any, data: Any) -> np.ndarray:
    """Centre of mass of every body in the tree (the body-plus-club system) at
    the current ``data`` state, from MuJoCo's subtree centre of mass of the
    first body under the world. Postcondition: a finite 3-vector."""
    import mujoco

    if model.nbody < 2:
        raise ValueError("Model has no bodies below the world")
    mujoco.mj_comPos(model, data)
    com = np.asarray(data.subtree_com[1], dtype=float).copy()
    if not np.isfinite(com).all():
        raise ValueError("Centre of mass is not finite")
    return com


def add_scene_marker(
    scene: Any,
    position: np.ndarray,
    radius_m: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Add a sphere marker to a rendered scene (an overlay, not a model geom).
    Precondition: the scene has a free geom slot and the radius is positive."""
    import mujoco

    if radius_m <= 0:
        raise ValueError("Marker radius must be positive")
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("Scene has no free geom slot for a marker")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius_m, 0.0, 0.0]),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


COM_MARKER_RGBA = (0.95, 0.15, 0.15, 1.0)
COM_GROUND_RGBA = (0.95, 0.85, 0.1, 1.0)


def add_com_markers(
    scene: Any, model: Any, data: Any, ground_height_m: float
) -> np.ndarray:
    """Overlay the whole-body centre of mass (red) and its vertical projection
    onto the ground plane (yellow); returns the centre of mass."""
    com = whole_body_com(model, data)
    add_scene_marker(scene, com, 0.03, COM_MARKER_RGBA)
    add_scene_marker(
        scene,
        np.array([com[0], com[1], ground_height_m + 0.005]),
        0.025,
        COM_GROUND_RGBA,
    )
    return com


def render_playback(
    spec_bytes: bytes,
    names: Sequence[str],
    q: np.ndarray,
    lookat: np.ndarray,
    path: Path,
    show_com: bool = True,
    playback_stride: int | None = None,
    rate_hz: float = 120.0,
    fps: float = DEFAULT_FPS,
    speed: float = 1.0,
) -> None:
    """Render an animated GIF of a motion from a spec and joint trajectory.

    Time-based sampling (GCV-14, #11720): GIF frame ``j`` shows the ``q``
    sample nearest to swing time ``j * speed / fps``, via a
    :class:`~src.shared.python.video_timing.frame_schedule.FrameSchedule`
    over the ``rate_hz``-spaced trajectory. ``playback_stride`` is a
    deprecated alias for a fixed index step: it is converted to the
    equivalent ``speed`` at ``fps`` and ``rate_hz`` (mirrors
    ``ExportSettings.stride`` in ``native_viewer_export/core.py``) and emits
    a ``DeprecationWarning``.

    Raises:
        ValueError: if ``rate_hz`` is not positive and finite, if
            ``playback_stride`` is given but is not a positive integer, or
            if the resulting ``fps``/``speed`` are invalid (raised by
            :class:`FrameSchedule`).
    """
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be positive and finite, got {rate_hz}")
    if playback_stride is not None:
        if not isinstance(playback_stride, int) or playback_stride < 1:
            raise ValueError("playback_stride must be a positive integer")
        warnings.warn(
            "render_playback(playback_stride=...) is deprecated: playback is "
            "time-based now; use fps and speed",
            DeprecationWarning,
            stacklevel=2,
        )
        speed = playback_stride * fps / rate_hz
    schedule = FrameSchedule(np.arange(q.shape[0]) / rate_hz, fps, speed)

    import json

    import imageio
    import mujoco

    from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

    xml, _ = exporter.export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    addresses = [model.joint(n).qposadr[0] for n in names]
    ground_height = float(
        json.loads(spec_bytes)["contact"].get("ground_height_m") or 0.0
    )
    renderer = mujoco.Renderer(model, 240, 320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135.0, -12.0
    frames_out = []
    for k in schedule.nearest_indices():
        data.qpos[addresses] = q[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        if show_com:
            add_com_markers(renderer.scene, model, data, ground_height)
        frames_out.append(renderer.render().copy())
    imageio.mimsave(path, frames_out, duration=1000.0 / fps, loop=0)
