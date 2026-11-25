"""Smooth grip-driven club motion and the validated settings (#15684).

Pose-track input, settings loading and the C2 grip/orientation splines that
prescribe the club body in :mod:`grip_driven_shaft`. Same hand motion for every
shaft: nothing here adapts the golfer to the flex.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, RotationSpline

from src.shared.python.motion_matching import club_models as cm

SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "club_fitting.json"

_FLOAT_FIELDS = (
    "integration_dt_s",
    "lead_in_s",
    "post_impact_s",
    "history_before_start_s",
    "extend_s",
    "outlier_threshold_k",
)
_COUNT_FIELDS = (
    "record_every",
    "outlier_half_window",
    "lowpass_order",
    "filter_pad_samples",
    "filter_fit_samples",
    "chain_links",
)


def _positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not (
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    ):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


@dataclass(frozen=True)
class FittingSettings:
    """Validated tunables of the grip-driven study (``club_fitting.json``).

    ``lowpass_cutoff_hz`` may be ``None`` (use the raw samples).
    """

    integration_dt_s: float
    lead_in_s: float
    post_impact_s: float
    record_every: int
    history_before_start_s: float
    extend_s: float
    outlier_half_window: int
    outlier_threshold_k: float
    lowpass_cutoff_hz: float | None
    lowpass_order: int
    filter_pad_samples: int
    filter_fit_samples: int
    chain_links: int

    def __post_init__(self) -> None:
        for name in _FLOAT_FIELDS:
            _positive(name, getattr(self, name))
        for name in _COUNT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.filter_fit_samples < 3:
            raise ValueError("filter_fit_samples must be at least 3 (quadratic fit)")
        if self.lowpass_cutoff_hz is not None:
            _positive("lowpass_cutoff_hz", self.lowpass_cutoff_hz)


_SETTING_KEYS = {f.name for f in fields(FittingSettings)}


def load_settings(path: Path | None = None) -> FittingSettings:
    """Settings from ``path`` (default ``src/config/club_fitting.json``).

    Raises:
        ValueError: a missing or unknown key, or a value out of range.
    """
    data = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    if set(data) != _SETTING_KEYS:
        diff = sorted(set(data) ^ _SETTING_KEYS)
        raise ValueError(f"club_fitting settings keys differ from the schema: {diff}")
    return FittingSettings(**data)


def unit_vector(name: str, vector: Any) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-9:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class ClubPoseTrack:
    """Club pose samples: ``grip_m`` (n, 3) and club-body ``rotation`` (n, 3, 3).

    The grip point is the butt of the exposed shaft span; the club-body frame
    has the shaft along +y (grip at ``y = -span``) and the head at the origin.
    """

    times_s: np.ndarray
    grip_m: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.times_s)
        if self.times_s.ndim != 1 or n < 4 or not np.all(np.diff(self.times_s) > 0):
            raise ValueError("times_s must be a strictly increasing 1-D array (n>=4)")
        if self.grip_m.shape != (n, 3) or self.rotation.shape != (n, 3, 3):
            raise ValueError("grip_m must be (n, 3) and rotation (n, 3, 3)")
        if not (
            np.all(np.isfinite(self.grip_m)) and np.all(np.isfinite(self.rotation))
        ):
            raise ValueError("pose samples must be finite")


def grip_point_in_club(club: cm.ClubSpec) -> np.ndarray:
    """Grip point in the club-body frame: ``(0, -span, HEAD_HEIGHT)``."""
    return np.array(
        [0.0, -(club.length_m - club.grip_length_m), cm.HEAD_HEIGHT_M], dtype=float
    )


def repair_outliers(
    t: np.ndarray, sig: np.ndarray, half: int, k: float
) -> tuple[np.ndarray, np.ndarray]:
    """Replace single-sample outliers by a leave-one-out local quadratic fit.

    A sample is an outlier when its residual from the quadratic through its
    +-``half`` neighbours (itself excluded) exceeds ``k`` times the median
    residual in the position block (first 3 columns) or quaternion block.
    Returns ``(repaired signal, times of repaired samples)``.
    """
    n = len(t)
    fit, res = sig.copy(), np.zeros((n, 2))
    for i in range(n):
        idx = [j for j in range(max(0, i - half), min(n, i + half + 1)) if j != i]
        if len(idx) < 4:
            continue
        x = t[idx] - t[i]
        coef, *_ = np.linalg.lstsq(
            np.vstack([np.ones_like(x), x, x**2]).T, sig[idx], None
        )
        fit[i] = coef[0]
        d = sig[i] - coef[0]
        res[i] = np.linalg.norm(d[:3]), np.linalg.norm(d[3:])
    bad = np.any(res > k * np.median(res, axis=0), axis=1)
    bad[:2] = bad[-2:] = False  # keep the window ends
    out = sig.copy()
    out[bad] = fit[bad]
    return out, t[bad]


def _quadratic_pad(sig: np.ndarray, dt: float, pad: int, nfit: int) -> np.ndarray:
    """``sig`` padded at both ends by quadratic extrapolation (end-accel safe)."""
    x_end = np.arange(-nfit + 1, 1) * dt
    x_new = np.arange(1, pad + 1) * dt
    basis = lambda x: np.vstack([np.ones_like(x), x, x**2]).T  # noqa: E731
    end, *_ = np.linalg.lstsq(basis(x_end), sig[-nfit:], rcond=None)
    start, *_ = np.linalg.lstsq(basis(x_end), sig[:nfit][::-1], rcond=None)
    return np.vstack([(basis(x_new) @ start)[::-1], sig, basis(x_new) @ end])


class GripMotion:
    """Smooth (C2) club-body motion driven by the grip point.

    Samples of ``track`` in ``[t_first, t_last]`` have single-frame outliers
    repaired and are zero-phase low-passed (``settings.lowpass_cutoff_hz``),
    then extended past ``t_last`` with constant acceleration (position) and
    constant body angular rate (orientation) for ``settings.extend_s``.
    """

    def __init__(
        self,
        track: ClubPoseTrack,
        r_grip: np.ndarray,
        t_first: float,
        t_last: float,
        settings: FittingSettings,
    ) -> None:
        sel = (track.times_s >= t_first) & (track.times_s <= t_last + 1e-9)
        t = track.times_s[sel]
        need = max(settings.filter_fit_samples, 2 * settings.outlier_half_window + 2)
        if len(t) < need:
            raise ValueError(f"need at least {need} pose samples before impact")
        self.r_grip = r_grip
        quat = Rotation.from_matrix(track.rotation[sel]).as_quat()
        for i in range(1, len(quat)):  # sign continuity
            if quat[i] @ quat[i - 1] < 0:
                quat[i] = -quat[i]
        sig, self.repaired_times_s = repair_outliers(
            t,
            np.hstack([track.grip_m[sel], quat]),
            settings.outlier_half_window,
            settings.outlier_threshold_k,
        )
        dt = float(np.median(np.diff(t)))
        if settings.lowpass_cutoff_hz is not None:
            if settings.lowpass_cutoff_hz >= 0.5 / dt:
                raise ValueError("lowpass cutoff must lie below the Nyquist rate")
            b, a = butter(settings.lowpass_order, settings.lowpass_cutoff_hz * 2 * dt)
            pad = settings.filter_pad_samples
            ext = _quadratic_pad(sig, dt, pad, settings.filter_fit_samples)
            sig = filtfilt(b, a, ext, axis=0, padtype="odd", padlen=pad)[pad:-pad]
        grip, quat = (
            sig[:, :3],
            sig[:, 3:] / np.linalg.norm(sig[:, 3:], axis=1)[:, None],
        )
        pre, rpre = (
            CubicSpline(t, grip, axis=0),
            RotationSpline(t, Rotation.from_quat(quat)),
        )
        te = t[-1]
        p0, v0, a0 = pre(te), pre(te, 1), pre(te, 2)
        w0, al0, r0 = np.asarray(rpre(te, 1)), np.asarray(rpre(te, 2)), rpre(te)
        ext_t = te + dt * np.arange(1, int(np.ceil(settings.extend_s / dt)) + 1)
        s = ext_t - te
        ext_p = p0 + np.outer(s, v0) + 0.5 * np.outer(s**2, a0)
        ext_q = [
            (r0 * Rotation.from_rotvec(w0 * x + 0.5 * al0 * x * x)).as_quat() for x in s
        ]
        tt = np.concatenate([t, ext_t])
        self.pos = CubicSpline(tt, np.vstack([grip, ext_p]), axis=0)
        self.rot = RotationSpline(tt, Rotation.from_quat(np.vstack([quat, ext_q])))
        self.t_range = (float(t[0]), float(tt[-1]))

    def state(self, t: float) -> tuple[np.ndarray, ...]:
        """``R, origin, v_origin, w_world, a_origin, alpha_world`` at ``t``."""
        rot = self.rot(t)
        R = rot.as_matrix()
        # scipy RotationSpline rates are in the body frame (checked numerically)
        w, al = R @ np.asarray(self.rot(t, 1)), R @ np.asarray(self.rot(t, 2))
        r = R @ self.r_grip
        po = self.pos(t) - r
        vo = self.pos(t, 1) - np.cross(w, r)
        ao = self.pos(t, 2) - np.cross(al, r) - np.cross(w, np.cross(w, r))
        return R, po, vo, w, ao, al
