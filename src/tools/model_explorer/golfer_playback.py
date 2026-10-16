"""Play a joint-space run back on the shared-appearance golfer (#12915).

Qt-free: :class:`GolferPlayback` renders any :class:`PoseFrameSource` frame
with MuJoCo's offscreen renderer on the full-body model exported with the
shared appearance document (body segments, garments, head, club meshes,
ground and sky), the same scene ``native_viewer_export`` films. Results from
engines without a Windows build (Drake, Pinocchio) and from OpenSim are
replayed here from their stored ``q``; the source's ``playback_label`` says
so. :func:`results_time_series` gives the plotted channels (joint angles and
club-head speed) and video export goes through the one shared writer,
``motion_matching.export.write_video_frames``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.golf_view_presets.presets import VIEW_ORDER
from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.model_explorer.pose_results import PoseFrameSource, ResultsSchemaError

__all__ = [
    "CLUB_HEAD_SPEED",
    "DEFAULT_LOOKAT_M",
    "DEFAULT_SIZE",
    "GolferPlayback",
    "TimeSeries",
    "results_time_series",
]

#: Camera look-at for the golfer at address (matches ``ExportSettings``).
DEFAULT_LOOKAT_M = (1.0, 0.0, 0.9)
DEFAULT_SIZE = (960, 540)
CLUB_HEAD_SPEED = "Club-head speed"
_MPS_TO_MPH = 2.2369362920544
logger = get_logger(__name__)


@dataclass(frozen=True)
class TimeSeries:
    """One plotted channel: ``values`` sampled at the source ``time_s``."""

    name: str
    unit: str
    values: NDArray[np.float64]


def _translational(spec: dict[str, Any]) -> set[str]:
    """Coordinates driven by a prismatic primitive (``Px``/``Py``/``Pz``)."""
    names: set[str] = set()
    for joint in spec.get("joints", []):
        for primitive in joint.get("primitives", []):
            if str(primitive.get("primitive", "")).startswith("P"):
                names.add(str(primitive.get("coordinate")))
    return names


def _club_head_speed(source: PoseFrameSource) -> TimeSeries | None:
    """Face-centre speed (mph) from MuJoCo FK, or ``None`` without a club.

    Uses the repository's one club-head measurement point, the rendered face
    centre carried by the spec ``Clubhead`` frame (``club_face_target``).
    """
    try:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
        from src.shared.python.motion_matching.club_face_target import (
            FACE_FRAME,
            face_centre_in_frame,
            face_centre_path,
        )

        centre = face_centre_in_frame(source.spec)
        fk = MujocoOverlaySource(source.spec_bytes)
    except (ImportError, KeyError, ValueError) as exc:
        logger.info("club-head speed unavailable: %s", exc)
        return None
    names = source.coordinate_order
    poses = np.array(
        [
            fk.frame_poses(dict(zip(names, map(float, row), strict=True)))[FACE_FRAME]
            for row in source.q
        ],
        dtype=float,
    )
    head = face_centre_path(poses[:, :3, :3], poses[:, :3, 3], centre)
    velocity = np.gradient(head, source.time_s, axis=0)
    speed = np.linalg.norm(velocity, axis=1) * _MPS_TO_MPH
    return TimeSeries(CLUB_HEAD_SPEED, "mph", np.asarray(speed, dtype=float))


def results_time_series(
    source: PoseFrameSource, *, with_club: bool = True
) -> dict[str, TimeSeries]:
    """Plot channels of ``source``: club-head speed first, then every joint.

    Rotational coordinates are reported in degrees and translational ones in
    metres. Club-head speed is omitted when the specification has no club or
    MuJoCo is unavailable. Postcondition: every series has
    ``source.frame_count`` samples.
    """
    series: dict[str, TimeSeries] = {}
    if with_club:
        club = _club_head_speed(source)
        if club is not None:
            series[club.name] = club
    linear = _translational(source.spec)
    for column, name in enumerate(source.coordinate_order):
        values = source.q[:, column]
        if name in linear:
            series[name] = TimeSeries(name, "m", values.copy())
        else:
            series[name] = TimeSeries(name, "deg", np.degrees(values))
    return series


class GolferPlayback:
    """Offscreen MuJoCo renders of a run on the shared-appearance golfer.

    Raises ``ImportError`` without MuJoCo and :class:`ResultsSchemaError`
    when a run coordinate has no joint in the exported model.
    """

    def __init__(
        self, source: PoseFrameSource, size: tuple[int, int] = DEFAULT_SIZE
    ) -> None:
        width, height = size
        if width < 16 or height < 16:
            raise ValueError(f"frame size must be at least 16x16, got {size}")
        import mujoco

        from src.tools.native_viewer_export.backends.mujoco_worker import (
            build_scene_xml,
            grow_offscreen_buffer,
        )

        self._mj = mujoco
        self.source = source
        self.size = (int(width), int(height))
        xml, _ = build_scene_xml(source.spec_bytes)
        self._model = mujoco.MjModel.from_xml_string(xml)
        grow_offscreen_buffer(self._model, *self.size)
        self._data = mujoco.MjData(self._model)
        self._qpos_adr = self._address_map(source)
        self._renderer: Any = mujoco.Renderer(self._model, self.size[1], self.size[0])
        self._camera = mujoco.MjvCamera()
        self.view = VIEW_ORDER[0]

    def _address_map(self, source: PoseFrameSource) -> list[int]:
        addresses: list[int] = []
        for name in source.coordinate_order:
            try:
                addresses.append(int(self._model.joint(name).qposadr[0]))
            except KeyError as exc:
                raise ResultsSchemaError(
                    source.path,
                    "coordinate_order",
                    f"coordinate {name!r} has no joint in the golfer model",
                ) from exc
        return addresses

    @property
    def view(self) -> str:
        return self._view

    @view.setter
    def view(self, name: str) -> None:
        """Camera preset name; one of ``VIEW_ORDER``."""
        if name not in VIEW_ORDER:
            raise ValueError(f"unknown view {name!r}; expected one of {VIEW_ORDER}")
        from src.shared.python.golf_view_presets.adapters import (
            mujoco_camera_params,
        )

        params = mujoco_camera_params(name, DEFAULT_LOOKAT_M)
        cam = self._camera
        cam.type = self._mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = params.lookat
        cam.distance, cam.azimuth, cam.elevation = (
            params.distance,
            params.azimuth,
            params.elevation,
        )
        self._view = name

    def render(self, index: int) -> NDArray[np.uint8]:
        """RGB frame ``(height, width, 3)`` of the golfer at sample ``index``.

        Raises ``ValueError`` when ``index`` is outside the run.
        """
        if not 0 <= index < self.source.frame_count:
            raise ValueError(
                f"frame {index} is outside the run (0..{self.source.frame_count - 1})"
            )
        self._data.qpos[self._qpos_adr] = self.source.q[index]
        self._mj.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def export_video(
        self,
        path: str | Path,
        *,
        fps: int = 30,
        speed: float = 1.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Render the whole run to ``path`` (``.mp4`` or ``.gif``) at ``speed``.

        Frames follow the shared time-based ``FrameSchedule``, so a 1x clip
        lasts as long as the swing. Postcondition: the file exists and is
        non-empty.
        """
        if not (math.isfinite(speed) and speed > 0):
            raise ValueError(f"speed must be positive, got {speed}")
        from src.shared.python.motion_matching.export import write_video_frames
        from src.shared.python.video_timing.frame_schedule import FrameSchedule

        indices = FrameSchedule(self.source.time_s, fps, speed).nearest_indices()
        cache: dict[int, NDArray[np.uint8]] = {}
        frames = []
        for count, index in enumerate(indices):
            key = int(index)
            if key not in cache:
                cache[key] = self.render(key)
            frames.append(cache[key])
            if progress is not None:
                progress(count + 1, len(indices))
        out = write_video_frames(frames, path, fps)
        if not out.is_file() or out.stat().st_size == 0:
            raise RuntimeError(f"video export to {out} wrote no data")
        return out

    def close(self) -> None:
        """Release the renderer; idempotent."""
        renderer, self._renderer = self._renderer, None
        if renderer is not None:
            renderer.close()


def spec_summary(source: PoseFrameSource) -> str:
    """One-line description of the run for the panel header."""
    spec = source.spec
    subject = spec.get("subject", {})
    club = spec.get("club", {})
    parts = [
        f"{source.frame_count} samples",
        f"{source.time_s[-1] - source.time_s[0]:.3f} s",
        f"{len(source.coordinate_order)} coordinates",
    ]
    if isinstance(club, dict) and club.get("name"):
        parts.append(f"club {club['name']}")
    if isinstance(subject, dict) and subject.get("name"):
        parts.append(str(subject["name"]))
    return ", ".join(parts)
