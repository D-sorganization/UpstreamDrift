"""Enhanced visualization module for MuJoCo humanoid golf simulation.

This module provides:
- Force and torque visualization overlays rendered via MuJoCo's native
  ``mjv_initGeom`` API so arrows, traces, and labels appear in the 3-D
  viewer without an extra rendering back-end.
- Trajectory tracers that persist through simulation.
- Desired-vs-actual trajectory comparison overlays.
- Real-time data display helpers.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Trajectory recording
# ---------------------------------------------------------------------------
class TrajectoryTracer:
    """Manages trajectory traces for bodies in the simulation."""

    def __init__(self, max_points: int = 1000) -> None:
        self.traces: dict[str, deque] = {}
        self.max_points = max_points
        self._desired_traces: dict[str, list[np.ndarray]] = {}

    def add_point(self, body_name: str, position: np.ndarray) -> None:
        """Append a position sample to a body's trajectory trace."""
        if body_name not in self.traces:
            self.traces[body_name] = deque(maxlen=self.max_points)
        self.traces[body_name].append(np.asarray(position, dtype=np.float64).copy())

    def get_trace(self, body_name: str) -> list[np.ndarray]:
        """Return the recorded trajectory points for a body."""
        return list(self.traces.get(body_name, []))

    def set_desired_trajectory(
        self, body_name: str, positions: list[np.ndarray] | np.ndarray
    ) -> None:
        """Register a *desired* trajectory for overlay comparison.

        Args:
            body_name: Body whose desired path is being stored.
            positions: Sequence of (3,) world-frame positions.
        """
        self._desired_traces[body_name] = [
            np.asarray(p, dtype=np.float64) for p in positions
        ]

    def get_desired_trace(self, body_name: str) -> list[np.ndarray]:
        """Return the desired reference trajectory for a body."""
        return list(self._desired_traces.get(body_name, []))

    def clear(self, body_name: str | None = None) -> None:
        """Remove trajectory data for one body or all bodies."""
        if body_name:
            self.traces.pop(body_name, None)
            self._desired_traces.pop(body_name, None)
        else:
            self.traces.clear()
            self._desired_traces.clear()


# ---------------------------------------------------------------------------
# Force / torque data extraction
# ---------------------------------------------------------------------------
class ForceVisualizer:
    """Extracts force and torque data from MuJoCo simulation state."""

    def __init__(self, physics: Any) -> None:
        self.physics = physics
        self.model = physics.model
        self.data = physics.data

    def get_contact_forces(self) -> list[dict]:
        """Return per-contact dicts with position, normal, magnitudes."""
        import mujoco

        contacts: list[dict] = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)

            geom1, geom2 = contact.geom1, contact.geom2
            body1_id = self.model.geom_bodyid[geom1]
            body2_id = self.model.geom_bodyid[geom2]

            try:
                body1_name = self.model.body(body1_id).name
            except (RuntimeError, ValueError, OSError):
                body1_name = f"body_{body1_id}"
            try:
                body2_name = self.model.body(body2_id).name
            except (RuntimeError, ValueError, OSError):
                body2_name = f"body_{body2_id}"

            contacts.append(
                {
                    "position": contact.pos.copy(),
                    "normal": contact.frame[:3].copy(),
                    "normal_force": force[0],
                    "friction_force": np.linalg.norm(force[1:3]),
                    "total_force": np.linalg.norm(force[:3]),
                    "body1": body1_name,
                    "body2": body2_name,
                }
            )
        return contacts

    def get_joint_torques(self) -> dict[str, float]:
        """Return actuator_name -> torque mapping."""
        torques: dict[str, float] = {}
        for i in range(self.model.nu):
            try:
                name = self.model.actuator(i).name
            except (RuntimeError, ValueError, OSError):
                name = f"actuator_{i}"
            torques[name] = float(self.data.actuator_force[i])
        return torques

    def get_center_of_mass(self) -> dict[str, np.ndarray]:
        """Return whole-body center of mass position and velocity."""
        return {
            "position": self.data.subtree_com[0].copy(),
            "velocity": self.data.cvel[0][:3].copy(),
        }


# ---------------------------------------------------------------------------
# Native MuJoCo geometry helpers (mjvGeom population)
# ---------------------------------------------------------------------------


def _init_arrow_geom(
    geom: Any,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: np.ndarray | list[float],
) -> None:
    """Populate a ``mjvGeom`` as a capsule (arrow shaft) from *start* to *end*.

    Uses ``mujoco.mjv_initGeom`` which is available in MuJoCo >= 2.3.
    """
    import mujoco

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    diff = end - start
    length = float(np.linalg.norm(diff))
    if length < 1e-8:
        return

    midpoint = (start + end) / 2.0
    direction = diff / length

    # Build rotation matrix whose z-axis aligns with *direction*.
    z = direction
    if abs(z[0]) < 0.9:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    else:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    rot = np.column_stack([x, y, z]).flatten()  # row-major 3x3

    size = np.array([radius, radius, length / 2.0])
    rgba_arr = np.array(rgba[:4], dtype=np.float32)

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        size,
        midpoint,
        rot,
        rgba_arr,
    )


def _init_line_geom(
    geom: Any,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    rgba: np.ndarray | list[float],
) -> None:
    """Populate a ``mjvGeom`` as a thin cylinder between two points."""
    _init_arrow_geom(geom, p0, p1, radius, rgba)


def _init_sphere_geom(
    geom: Any,
    pos: np.ndarray,
    radius: float,
    rgba: np.ndarray | list[float],
) -> None:
    """Populate a ``mjvGeom`` as a sphere at *pos*."""
    import mujoco

    size = np.array([radius, 0.0, 0.0])
    rot = np.eye(3).flatten()
    rgba_arr = np.array(rgba[:4], dtype=np.float32)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        np.asarray(pos, dtype=np.float64),
        rot,
        rgba_arr,
    )


# ---------------------------------------------------------------------------
# Public overlay entry point
# ---------------------------------------------------------------------------


def add_visualization_overlays(
    viewer: Any,
    physics: Any,
    config: dict,
    tracer: TrajectoryTracer,
) -> None:
    """Render force arrows, torque indicators, and trajectory traces into
    a live MuJoCo viewer by appending ``mjvGeom`` objects to the viewer's
    scene (``viewer.user_scn`` or ``viewer.scn``).

    Args:
        viewer: MuJoCo viewer (``mujoco.viewer`` handle) that exposes a
            writable ``user_scn`` (preferred) or ``scn`` attribute.
        physics: Physics wrapper exposing ``.model`` / ``.data``.
        config: Dict controlling which overlays are active.  Recognised
            keys: ``show_contact_forces``, ``show_joint_torques``,
            ``show_tracers``, ``show_desired_trajectory``,
            ``tracer_bodies``, ``force_scale``, ``torque_scale``.
        tracer: ``TrajectoryTracer`` instance accumulating body paths.
    """

    show_forces = config.get("show_contact_forces", True)
    show_torques = config.get("show_joint_torques", True)
    show_tracers = config.get("show_tracers", True)
    show_desired = config.get("show_desired_trajectory", True)
    tracer_bodies: list[str] = config.get("tracer_bodies", [])
    force_scale: float = config.get("force_scale", 0.01)
    torque_scale: float = config.get("torque_scale", 0.005)

    visualizer = ForceVisualizer(physics)

    scene = getattr(viewer, "user_scn", None) or getattr(viewer, "scn", None)
    if scene is None:
        return

    if show_tracers:
        _record_tracer_positions(physics, tracer, tracer_bodies)

    def _add_geom() -> Any | None:
        if scene.ngeom >= scene.maxgeom:
            return None
        g = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return g

    if show_forces:
        _overlay_contact_forces(visualizer, _add_geom, force_scale)

    if show_torques:
        _overlay_joint_torques(visualizer, physics, _add_geom, torque_scale)

    if show_tracers:
        _overlay_trajectory_traces(tracer, tracer_bodies, _add_geom)

    if show_desired:
        _overlay_desired_trajectory(tracer, tracer_bodies, _add_geom)


def _record_tracer_positions(
    physics: Any, tracer: TrajectoryTracer, tracer_bodies: list[str]
) -> None:
    for body_name in tracer_bodies:
        try:
            body_id = physics.model.body(body_name).id
            tracer.add_point(body_name, physics.data.xpos[body_id].copy())
        except (RuntimeError, ValueError, AttributeError):
            pass


def _overlay_contact_forces(
    visualizer: ForceVisualizer,
    add_geom: Callable[[], Any | None],
    force_scale: float,
) -> None:
    contacts = visualizer.get_contact_forces()
    for contact in contacts:
        if contact["total_force"] < 0.1:
            continue
        arrow_len = min(contact["total_force"] * force_scale, 0.5)
        direction = contact["normal"]
        start = contact["position"]
        end = start + direction * arrow_len

        g = add_geom()
        if g is not None:
            _init_arrow_geom(g, start, end, 0.005, FORCE_COLORS["contact_normal"])

        if contact["friction_force"] > 0.05:
            _add_friction_arrow(add_geom, contact, direction, start, force_scale)


def _add_friction_arrow(
    add_geom: Callable[[], Any | None],
    contact: dict,
    direction: Any,
    start: Any,
    force_scale: float,
) -> None:
    fric_len = min(contact["friction_force"] * force_scale, 0.3)
    n = np.asarray(direction, dtype=np.float64)
    if abs(n[0]) < 0.9:
        tangent = np.cross(n, [1.0, 0.0, 0.0])
    else:
        tangent = np.cross(n, [0.0, 1.0, 0.0])
    tangent /= np.linalg.norm(tangent) + 1e-12
    fric_end = start + tangent * fric_len
    g = add_geom()
    if g is not None:
        _init_arrow_geom(g, start, fric_end, 0.003, FORCE_COLORS["contact_friction"])


def _overlay_joint_torques(
    visualizer: ForceVisualizer,
    physics: Any,
    add_geom: Callable[[], Any | None],
    torque_scale: float,
) -> None:
    torques = visualizer.get_joint_torques()
    for i, (name, tau) in enumerate(torques.items()):
        if abs(tau) < 0.01:
            continue
        jnt_idx = None
        for j in range(physics.model.njnt):
            try:
                if physics.model.joint(j).name == name:
                    jnt_idx = j
                    break
            except (RuntimeError, ValueError, AttributeError):
                pass
        body_id = min(i + 1, physics.model.nbody - 1)
        if jnt_idx is not None:
            body_id = physics.model.jnt_bodyid[jnt_idx]

        pos = physics.data.xpos[body_id]
        colour = (
            FORCE_COLORS["joint_torque_positive"]
            if tau > 0
            else FORCE_COLORS["joint_torque_negative"]
        )
        radius = min(abs(tau) * torque_scale, 0.04)
        g = add_geom()
        if g is not None:
            _init_sphere_geom(g, pos, max(radius, 0.005), colour)


def _overlay_trajectory_traces(
    tracer: TrajectoryTracer,
    tracer_bodies: list[str],
    add_geom: Callable[[], Any | None],
) -> None:
    for body_name in tracer_bodies:
        trace = tracer.get_trace(body_name)
        if len(trace) < 2:
            continue
        color = get_trace_color(body_name)
        step = max(1, len(trace) // 200)
        for k in range(0, len(trace) - step, step):
            g = add_geom()
            if g is None:
                break
            _init_line_geom(g, trace[k], trace[k + step], 0.002, color)


def _overlay_desired_trajectory(
    tracer: TrajectoryTracer,
    tracer_bodies: list[str],
    add_geom: Callable[[], Any | None],
) -> None:
    for body_name in tracer_bodies:
        desired = tracer.get_desired_trace(body_name)
        if len(desired) < 2:
            continue
        desired_color = [0.0, 1.0, 1.0, 0.5]
        step = max(1, len(desired) // 200)
        for k in range(0, len(desired) - step, step * 2):
            g = add_geom()
            if g is None:
                break
            end_k = min(k + step, len(desired) - 1)
            _init_line_geom(g, desired[k], desired[end_k], 0.003, desired_color)


# ---------------------------------------------------------------------------
# Standalone geometry factories (for external consumers)
# ---------------------------------------------------------------------------


def create_force_arrow_geom(
    position: np.ndarray,
    direction: np.ndarray,
    magnitude: float,
    color: list[float],
) -> dict:
    """Return a plain dict describing a force arrow (engine-agnostic)."""
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    length = min(magnitude / 100.0, 0.5)
    end_pos = np.asarray(position) + direction * length
    return {
        "type": "arrow",
        "start": np.asarray(position),
        "end": end_pos,
        "radius": 0.005,
        "color": color,
        "magnitude": magnitude,
    }


def create_trace_line_geom(
    points: list[np.ndarray],
    color: list[float],
    radius: float = 0.002,
) -> list[dict]:
    """Return a list of line-segment dicts for a trajectory trace."""
    segments = []
    for i in range(len(points) - 1):
        segments.append(
            {
                "type": "line",
                "start": points[i],
                "end": points[i + 1],
                "radius": radius,
                "color": color,
            }
        )
    return segments


# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------

FORCE_COLORS = {
    "contact_normal": [1.0, 0.0, 0.0, 0.8],
    "contact_friction": [1.0, 0.5, 0.0, 0.8],
    "joint_torque_positive": [0.0, 1.0, 0.0, 0.8],
    "joint_torque_negative": [0.0, 0.0, 1.0, 0.8],
}

TRACE_COLORS = {
    "pelvis": [1.0, 1.0, 0.0, 0.6],
    "torso": [0.0, 1.0, 1.0, 0.6],
    "head": [1.0, 0.0, 0.0, 0.6],
    "r_hand": [0.0, 1.0, 0.0, 0.6],
    "l_hand": [0.0, 0.0, 1.0, 0.6],
    "r_foot": [1.0, 0.5, 0.0, 0.6],
    "l_foot": [0.5, 0.0, 1.0, 0.6],
    "club_head": [1.0, 0.0, 1.0, 0.8],
}


def get_trace_color(body_name: str) -> list[float]:
    """Get RGBA colour for a body's trajectory trace."""
    return TRACE_COLORS.get(body_name, [0.5, 0.5, 0.5, 0.6])
