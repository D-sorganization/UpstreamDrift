"""Muscle-activation rendering helpers for MyoFullBody swings (issue #11646).

Pure functions (colour mapping, frame selection, camera placement) are importable
without MuJoCo; :func:`render_view` imports it lazily and needs an offscreen GL
context (``MUJOCO_GL=egl``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.contracts import require

Array = np.ndarray

COLD_RGB = np.array([0.10, 0.30, 1.00])  # activation 0: blue
HOT_RGB = np.array([1.00, 0.10, 0.10])  # activation 1: red
VIEW_NAMES = ("face_on", "down_the_line", "overhead", "oblique")
BONE_RGBA = (0.80, 0.80, 0.78, 0.30)


def activation_rgba(activation: Array, alpha: float = 1.0) -> Array:
    """Blue (activation 0) to red (activation 1) colour per muscle, ``(n, 4)``.

    Raises:
        ValueError: if an activation is not finite or lies outside ``[0, 1]``
            by more than 1e-6, or ``alpha`` is outside ``[0, 1]``.
    """
    a = np.asarray(activation, dtype=float)
    require(bool(np.isfinite(a).all()), "activation must be finite")
    require(
        bool((a >= -1e-6).all() and (a <= 1.0 + 1e-6).all()),
        "activation must lie in [0, 1]",
    )
    require(0.0 <= alpha <= 1.0, "alpha must lie in [0, 1]")
    t = np.clip(a, 0.0, 1.0)[..., None]
    rgb = (1.0 - t) * COLD_RGB + t * HOT_RGB
    return np.concatenate([rgb, np.full(rgb.shape[:-1] + (1,), alpha)], axis=-1)


def select_frames(times_s: Array, fps: float = 50.0, slowdown: float = 0.25) -> Array:
    """Source-frame index shown by each video frame at ``slowdown`` x real time.

    Video frame ``j`` lasts ``1 / fps`` seconds of video, which is
    ``slowdown / fps`` seconds of the swing, so it shows the sample nearest to
    ``times[0] + j * slowdown / fps``.

    Returns:
        Integer indices, non-decreasing, covering the whole swing.

    Raises:
        ValueError: if ``times_s`` is not a strictly increasing 1-D array of at
            least two samples, or ``fps``/``slowdown`` are not positive.
    """
    t = np.asarray(times_s, dtype=float)
    require(t.ndim == 1 and t.size >= 2, "times_s must be 1-D with >= 2 samples")
    require(bool((np.diff(t) > 0.0).all()), "times_s must be strictly increasing")
    require(fps > 0.0 and slowdown > 0.0, "fps and slowdown must be positive")
    n = int(np.floor((t[-1] - t[0]) * fps / slowdown)) + 1
    wanted = t[0] + np.arange(n) * slowdown / fps
    right = np.clip(np.searchsorted(t, wanted), 1, t.size - 1)
    left = right - 1
    nearer_left = (wanted - t[left]) <= (t[right] - wanted)
    return np.where(nearer_left, left, right).astype(int)


def circular_mean_deg(a: float, b: float) -> float:
    """Mean of two headings in degrees, correct across the +-180 wrap."""
    ra, rb = np.radians([a, b])
    return float(
        np.degrees(np.arctan2(np.sin(ra) + np.sin(rb), np.cos(ra) + np.cos(rb)))
    )


def view_cameras(
    forward: Array, target_dir: Array, centre: Array, distance: float = 3.2
) -> dict[str, dict[str, Any]]:
    """Free-camera parameters of the four views.

    Args:
        forward: horizontal direction the golfer faces at address (world frame).
        target_dir: horizontal direction of the target line (club head velocity
            at impact).
        centre: look-at point (m).
        distance: camera distance (m).

    Returns:
        ``{view: {"lookat", "distance", "azimuth", "elevation"}}`` in degrees,
        following MuJoCo's free camera (azimuth is the heading of the viewing
        direction in the horizontal plane, elevation is negative when looking
        down).

    Raises:
        ValueError: if a direction has no horizontal component.
    """

    def heading(v: Array) -> float:
        h = np.asarray(v, dtype=float)[:2]
        require(float(np.linalg.norm(h)) > 1e-9, "direction needs a horizontal part")
        return float(np.degrees(np.arctan2(h[1], h[0])))

    look_face = heading(
        -np.asarray(forward)
    )  # camera in front, looking back at the golfer
    look_dtl = heading(target_dir)  # camera behind the golfer, looking at the target
    common = {"lookat": np.asarray(centre, dtype=float), "distance": float(distance)}
    return {
        "face_on": {**common, "azimuth": look_face, "elevation": -8.0},
        "down_the_line": {**common, "azimuth": look_dtl, "elevation": -8.0},
        "overhead": {
            **common,
            "azimuth": look_dtl,
            "elevation": -89.0,
            "distance": float(distance) * 1.1,
        },
        "oblique": {
            **common,
            "azimuth": circular_mean_deg(look_face, look_dtl),
            "elevation": -22.0,
        },
    }


def muscle_tendon_ids(model: Any) -> Array:
    """Tendon index of each muscle actuator (``-1`` for non-tendon transmissions)."""
    import mujoco

    tendon = int(mujoco.mjtTrn.mjTRN_TENDON)
    return np.array(
        [
            int(model.actuator_trnid[i, 0])
            if int(model.actuator_trntype[i]) == tendon
            else -1
            for i in range(model.nu)
        ]
    )


def style_bones(model: Any) -> None:
    """Make mesh geoms translucent grey so the coloured muscles read clearly."""
    import mujoco

    mesh = model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
    model.geom_rgba[mesh] = BONE_RGBA
    model.geom_rgba[~mesh, 3] = 0.0


def render_view(
    model: Any,
    data: Any,
    qpos: Array,
    activation: Array,
    camera: dict[str, Any],
    renderer: Any,
    club_length_m: float | None = None,
) -> Array:
    """One RGB frame of the pose with muscles coloured by ``activation``.

    ``club_length_m`` adds the illustrative club of :func:`club_segment`.
    """
    import mujoco

    require(activation.shape == (model.nu,), "activation must be (nu,)")
    ids = muscle_tendon_ids(model)
    ok = ids >= 0
    model.tendon_rgba[ids[ok]] = activation_rgba(activation[ok])
    data.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, data)  # tendon paths are drawn from this
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = camera["lookat"]
    cam.distance = camera["distance"]
    cam.azimuth = camera["azimuth"]
    cam.elevation = camera["elevation"]
    renderer.update_scene(data, camera=cam)
    if club_length_m is not None:
        add_club(renderer, data, model, club_length_m)
    return np.asarray(renderer.render()).copy()


def club_segment(
    elbow: Array, lead_wrist: Array, trail_wrist: Array, length_m: float
) -> tuple[Array, Array]:
    """Illustrative club: grip at the wrists' midpoint, shaft along the lead forearm.

    MyoFullBody has no club.  This is a drawing aid, not the spec club: the
    shaft is simply the lead forearm line extended to ``length_m``.

    Raises:
        ValueError: if ``length_m`` is not positive or the forearm has no length.
    """
    require(length_m > 0.0, "length_m must be positive")
    axis = np.asarray(lead_wrist, float) - np.asarray(elbow, float)
    norm = float(np.linalg.norm(axis))
    require(norm > 1e-9, "the forearm has no length")
    grip = 0.5 * (np.asarray(lead_wrist, float) + np.asarray(trail_wrist, float))
    return grip, grip + length_m * axis / norm


def add_club(renderer: Any, data: Any, model: Any, length_m: float) -> None:
    """Draw the illustrative club (see :func:`club_segment`) into the scene."""
    import mujoco

    pos = lambda name: np.asarray(data.xpos[model.body(name).id])  # noqa: E731
    grip, head = club_segment(pos("ulna_l"), pos("lunate_l"), pos("lunate_r"), length_m)
    scene = renderer.scene
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array([0.25, 0.25, 0.28, 1.0], dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, grip, head)
    scene.ngeom += 1
