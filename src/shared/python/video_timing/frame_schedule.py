"""Time-based frame schedule for video export (GCV-14, #11720).

Playback speed used to be an accident of ``stride x dt``. A
:class:`FrameSchedule` fixes it: video frame ``j`` lasts ``1 / fps`` seconds of
video and therefore ``speed / fps`` seconds of the swing, so it shows the swing
state at ``t0 + j * speed / fps``. When the source ``dt`` is coarser than that
step the state is interpolated (linear for joints, spherical for the unit
quaternions of free and ball joints) instead of held.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require

Array = NDArray[np.float64]
DEFAULT_FPS = 60.0
MAX_SPEED = 4.0
SPEED_VARIANTS = (1.0, 0.5, 0.25)  # full, half and quarter speed (GCV-14)
_EPS = 1e-9
_MUJOCO_FREE, _MUJOCO_BALL = 0, 1  # mjtJoint values


def speed_suffix(speed: float) -> str:
    """File-name suffix of a playback speed: ``1`` -> ``_1x``, ``0.5`` -> ``_0p5x``."""
    require(math.isfinite(speed) and speed > 0.0, "speed must be positive and finite")
    text = f"{speed:.4f}".rstrip("0").rstrip(".")
    return "_" + text.replace(".", "p") + "x"


def stride_for_speed(dt_s: float, fps: float = DEFAULT_FPS, speed: float = 1.0) -> int:
    """Index stride nearest to ``speed / fps`` seconds of swing per video frame.

    For renderers that can only take every Nth source sample. Returns at least 1;
    prefer :class:`FrameSchedule` interpolation when the renderer accepts states.
    """
    require(math.isfinite(dt_s) and dt_s > 0.0, "dt_s must be positive and finite")
    require(math.isfinite(fps) and fps > 0.0, "fps must be positive and finite")
    require(math.isfinite(speed) and speed > 0.0, "speed must be positive")
    return max(1, round(speed / (fps * dt_s)))


def slerp(q0: Array, q1: Array, t: float) -> Array:
    """Shortest-path spherical interpolation of two unit quaternions.

    Postcondition: the result has unit norm. Falls back to a normalised lerp
    when the quaternions are nearly parallel.
    """
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    require(
        bool(a.shape == (4,) and b.shape == (4,)), "quaternions must have shape (4,)"
    )
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    require(bool(na > 0.0 and nb > 0.0), "quaternions must be non-zero")
    a, b = a / na, b / nb
    dot = float(a @ b)
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 1.0 - 1e-10:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)  # type: ignore[no-any-return]
    theta = math.acos(min(1.0, dot))
    s = math.sin(theta)
    return (math.sin((1.0 - t) * theta) * a + math.sin(t * theta) * b) / s  # type: ignore[no-any-return]


def quaternion_groups_from_model(model: Any) -> tuple[tuple[int, int, int, int], ...]:
    """``qpos`` index groups of the unit quaternions of a MuJoCo-style model.

    Reads ``jnt_type`` / ``jnt_qposadr``: a free joint holds its quaternion in
    ``qpos[adr + 3 : adr + 7]``, a ball joint in ``qpos[adr : adr + 4]``.
    """
    groups: list[tuple[int, int, int, int]] = []
    for j in range(int(model.njnt)):
        adr = int(model.jnt_qposadr[j])
        kind = int(model.jnt_type[j])
        if kind == _MUJOCO_FREE:
            groups.append((adr + 3, adr + 4, adr + 5, adr + 6))
        elif kind == _MUJOCO_BALL:
            groups.append((adr, adr + 1, adr + 2, adr + 3))
    return tuple(groups)


class FrameSchedule:
    """Sample times of a video of a swing at ``speed`` x real time.

    Args:
        times_s: strictly increasing source sample times (>= 2), seconds.
        fps: video frames per second (> 0).
        speed: playback speed relative to real time (> 0; 0.5 is half speed).
        window: optional ``(start_s, end_s)`` swing interval to show, clipped
            to the source range; it must overlap the samples.

    Postconditions: ``sample_times_s`` is strictly increasing, starts at the
    window start (or ``times_s[0]``) and never exceeds the window end.
    """

    def __init__(
        self,
        times_s: Array,
        fps: float = DEFAULT_FPS,
        speed: float = 1.0,
        window: tuple[float, float] | None = None,
    ) -> None:
        t = np.asarray(times_s, dtype=float)
        require(t.ndim == 1 and t.size >= 2, "times_s must be 1-D with >= 2 samples")
        require(bool(np.isfinite(t).all()), "times_s must be finite")
        require(bool((np.diff(t) > 0.0).all()), "times_s must be strictly increasing")
        require(math.isfinite(fps) and fps > 0.0, "fps must be positive and finite")
        require(math.isfinite(speed) and speed > 0.0, "speed must be positive")
        start, end = float(t[0]), float(t[-1])
        if window is not None:
            lo, hi = float(window[0]), float(window[1])
            require(hi > lo, "window must satisfy start < end")
            start, end = max(start, lo), min(end, hi)
            require(end > start, "window does not overlap the sampled swing")
        self.times_s = t
        self.fps = float(fps)
        self.speed = float(speed)
        self.window = window
        self._start, self._end = start, end

    @property
    def n_frames(self) -> int:
        return (
            int(math.floor((self._end - self._start) * self.fps / self.speed + _EPS))
            + 1
        )

    @property
    def sample_times_s(self) -> Array:
        """Swing time shown by each video frame."""
        return self._start + np.arange(self.n_frames) * self.speed / self.fps  # type: ignore[no-any-return,return-value]

    def nearest_indices(self) -> NDArray[np.int_]:
        """Source sample nearest to each video frame (non-decreasing)."""
        t, wanted = self.times_s, self.sample_times_s
        right = np.clip(np.searchsorted(t, wanted), 1, t.size - 1)
        left = right - 1
        nearer_left = (wanted - t[left]) <= (t[right] - wanted)
        return np.where(nearer_left, left, right).astype(int)

    def bracket(self) -> tuple[NDArray[np.int_], NDArray[np.int_], Array]:
        """``(left, right, fraction)`` so a frame is ``left + fraction * (right - left)``."""
        t, wanted = self.times_s, self.sample_times_s
        right = np.clip(np.searchsorted(t, wanted, side="left"), 1, t.size - 1)
        left = right - 1
        frac = np.clip((wanted - t[left]) / (t[right] - t[left]), 0.0, 1.0)
        return left, right, frac

    def interpolate(
        self,
        states: Array,
        quaternion_groups: tuple[tuple[int, int, int, int], ...] = (),
    ) -> Array:
        """States at the schedule times, shape ``(n_frames, nq)``.

        Linear in every coordinate except the ``quaternion_groups`` (index
        quadruples ``w, x, y, z``), which are slerped.

        Raises:
            ValueError: if ``states`` does not have one row per source sample or
                a quaternion index is out of range.
        """
        x = np.asarray(states, dtype=float)
        require(
            x.ndim == 2 and x.shape[0] == self.times_s.size,
            "states must have one row per source sample",
        )
        for group in quaternion_groups:
            require(
                len(group) == 4 and all(0 <= i < x.shape[1] for i in group),
                "quaternion group index outside the state vector",
            )
        left, right, frac = self.bracket()
        out = x[left] + frac[:, None] * (x[right] - x[left])
        for group in quaternion_groups:
            cols = list(group)
            for k in range(out.shape[0]):
                out[k, cols] = slerp(
                    x[left[k], cols], x[right[k], cols], float(frac[k])
                )
        return out


def select_frames(times_s: Array, fps: float = 50.0, slowdown: float = 0.25) -> Array:
    """Source-frame index shown by each video frame at ``slowdown`` x real time.

    Kept for the MyoFullBody renderer: the nearest-sample form of
    :class:`FrameSchedule`.

    Raises:
        ValueError: if ``times_s`` is not a strictly increasing 1-D array of at
            least two samples, or ``fps``/``slowdown`` are not positive.
    """
    return FrameSchedule(times_s, fps, slowdown).nearest_indices()  # type: ignore[return-value]
