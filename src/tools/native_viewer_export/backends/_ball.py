"""Backend helpers that draw the decorative golf ball (GCV-13, #11719).

The ball is visual only. MuJoCo-family workers append a sphere to the render
scene each frame it is visible; the MeshCat backends hold one world-fixed
sphere object and toggle its ``visible`` property. Nothing here touches a
dynamics model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from src.tools.native_viewer_export.ball import BALL_RGBA, BallPlacement
from src.tools.native_viewer_export.core import SwingInput

BALL_PATH = "/ball"


def ball_visibility(swing: SwingInput, indices: Sequence[int]) -> list[bool] | None:
    """Per-position ball visibility of ``swing`` at ``indices``, or ``None``.

    ``None`` means no ball is drawn at all (disabled or unavailable).
    """
    placement = swing.ball
    if placement is None:
        return None
    return [placement.visible_at(swing.time_s(k)) for k in indices]


def draw_job_ball(scene: Any, job: Any, position: int) -> bool:
    """Append the job's ball to a MuJoCo render ``scene`` when visible there.

    ``job`` is a :class:`WorkerJob`; returns whether a sphere was added.
    """
    if job.ball is None or not job.ball_visible or not job.ball_visible[position]:
        return False
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
        add_ball_to_scene,
    )

    placement = BallPlacement.from_dict(job.ball)
    return add_ball_to_scene(scene, placement.position_m, placement.radius_m, BALL_RGBA)


class BallSink(Protocol):
    """What a MeshCat flavour must offer to host the ball."""

    def set_sphere(
        self,
        path: str,
        position_m: tuple[float, float, float],
        radius_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None: ...

    def set_visible(self, path: str, visible: bool) -> None: ...


class MeshcatBall:
    """One world-fixed ball in a MeshCat scene, shown or hidden per frame."""

    def __init__(self, sink: BallSink, placement: BallPlacement) -> None:
        self._sink = sink
        self.placement = placement
        sink.set_sphere(BALL_PATH, placement.position_m, placement.radius_m, BALL_RGBA)

    def update(self, t_s: float) -> bool:
        """Show the ball while ``t_s`` is at or before impact; returns visibility."""
        visible = self.placement.visible_at(t_s)
        self._sink.set_visible(BALL_PATH, visible)
        return visible


def update_meshcat_frame(
    ball: MeshcatBall | None,
    glyphs: Any,
    overlay: Any,
    swing: SwingInput,
    index: int,
) -> None:
    """Per-frame MeshCat decorations: ball visibility, then the force glyphs.

    Shared by the Drake and Pinocchio backends; ``ball``, ``glyphs`` and
    ``overlay`` may each be ``None`` (that decoration is then skipped).
    """
    if ball is not None:
        ball.update(swing.time_s(index))
    if glyphs is not None and overlay is not None:
        glyphs.update(overlay.glyphs_at(index))


class DrakeBallSink:
    """:class:`BallSink` over a ``pydrake`` ``Meshcat``."""

    def __init__(self, meshcat: Any) -> None:
        self._meshcat = meshcat

    def set_sphere(
        self,
        path: str,
        position_m: tuple[float, float, float],
        radius_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        from pydrake.geometry import Rgba, Sphere
        from pydrake.math import RigidTransform

        self._meshcat.SetObject(path, Sphere(radius_m), Rgba(*rgba))
        self._meshcat.SetTransform(path, RigidTransform(list(position_m)))  # type: ignore[arg-type]

    def set_visible(self, path: str, visible: bool) -> None:
        self._meshcat.SetProperty(path, "visible", visible)


class PythonMeshcatBallSink:
    """:class:`BallSink` over a meshcat-python ``Visualizer`` node."""

    def __init__(self, node: Any) -> None:
        self._node = node

    def set_sphere(
        self,
        path: str,
        position_m: tuple[float, float, float],
        radius_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        import meshcat.geometry as geometry
        import meshcat.transformations as tf

        colour = (
            (int(round(rgba[0] * 255)) << 16)
            | (int(round(rgba[1] * 255)) << 8)
            | int(round(rgba[2] * 255))
        )
        item = self._node[path]
        item.set_object(geometry.Sphere(radius_m), geometry.MeshLambertMaterial(colour))
        item.set_transform(tf.translation_matrix(list(position_m)))

    def set_visible(self, path: str, visible: bool) -> None:
        self._node[path].set_property("visible", visible)
