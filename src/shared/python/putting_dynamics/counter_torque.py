"""Counter-torque stabilization of the putting stroke (#14469, epic #14452).

The studio's closed-form two-hand couple model (``js/state.js``
``evalPhysicsAtTime``), moved into the shared core so every surface shows
the same values (ADR-0058). It is illustrative, not calibrated: the
constants below are the studio's, and the model has no measured inputs.

* **Stroke.** The shaft angle ``theta`` follows quintic (minimum-jerk)
  segments: back to ``-THETA_MAX`` by ``T_BACK``, then through to impact at
  ``T_IMPACT`` (``theta = 0``) and on to the finish at ``T_STROKE``, where
  the head has slowed by the restitution law's post-impact speed ratio.
* **Couples.** The in-plane moment the stroke needs is
  ``tau_net = 0.38 alpha + 0.35 omega - tau_gravity``; the off-plane one
  balances gravity and the centripetal couple of the lie. Each is split
  between the hands with a smooth (C-infinity) positive/negative part, on
  top of a preload couple the golfer holds (``preload_nm`` in plane,
  ``off_preload_nm`` off plane): without noise the two hands' couples sum
  to exactly the required moment. Forces are couples over ``HAND_LEVER_M``.
* **Noise and gust.** Hand noise is a smooth multi-sine (an
  Ornstein-Uhlenbeck look-alike) scaled by ``noise_scale``. A gust pulse of
  ``gust_impulse_nm`` near 0.72 s twists the face: ``phi_slack`` is the
  response of a slack grip, ``phi_preload`` the same pulse damped by the
  preload (``1.8 / (1.8 + 14.5 preload)``) plus a tremor that grows with
  over-gripping.
* **Impact.** Face and path errors shrink with grip stiffness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    PUTTER_FACE_COR,
)

T_STROKE_S = 1.05
T_BACK_S = 0.60
T_IMPACT_S = 0.8168  # downswing crosses theta = 0
THETA_MAX_DEG = 8.4  # backswing of a ~10 ft putt
DOWNSWING_GAIN = 2.15
HAND_LEVER_M = 0.10
STUDIO_HEAD_MASS_KG = 0.35
_CLUB_MASS_KG = 0.52
_CLUB_COM_M = 0.58
_STUDIO_GRAVITY_M_S2 = 9.81  # the studio's rounded g, kept for parity
_SMOOTH_IN_NM = 0.12
_SMOOTH_OFF_NM = 0.08
PRELOAD_RANGE_NM = (0.0, 2.0)
NOISE_RANGE = (0.0, 2.0)
GUST_RANGE_NM = (0.0, 2.0)
LIE_RANGE_DEG = (55.0, 85.0)


@dataclass(frozen=True)
class PutterSpec:
    """Putter head as the couple model reads it: name and face offset (m)."""

    name: str
    e_face_m: float


PUTTER_SPECS = {  # studio PUTTER_SPECS
    "th": PutterSpec("Toe-Hang Blade (45°)", 0.015),
    "fb": PutterSpec("Face-Balanced Mallet", 0.000),
    "lab": PutterSpec("Zero-Torque / LAB", 0.000),
    "cb": PutterSpec("Heavy Counterbalanced", 0.002),
}


def _within(value: float, bounds: tuple[float, float], name: str) -> None:
    if not (math.isfinite(value) and bounds[0] <= value <= bounds[1]):
        raise ValueError(f"{name} must be in [{bounds[0]}, {bounds[1]}], got {value}")


@dataclass(frozen=True)
class CounterTorqueInputs:
    """The explorer's settings (studio defaults)."""

    preload_nm: float = 0.60
    off_preload_nm: float = 0.55
    noise_scale: float = 0.25
    gust_impulse_nm: float = 0.25
    lie_deg: float = 70.0
    putter: str = "th"

    def __post_init__(self) -> None:
        _within(self.preload_nm, PRELOAD_RANGE_NM, "preload_nm")
        _within(self.off_preload_nm, PRELOAD_RANGE_NM, "off_preload_nm")
        _within(self.noise_scale, NOISE_RANGE, "noise_scale")
        _within(self.gust_impulse_nm, GUST_RANGE_NM, "gust_impulse_nm")
        _within(self.lie_deg, LIE_RANGE_DEG, "lie_deg")
        if self.putter not in PUTTER_SPECS:
            raise ValueError(f"putter must be one of {sorted(PUTTER_SPECS)}")


@dataclass(frozen=True)
class CounterTorqueSample:
    """Stroke state, hand couples (N m), hand forces (N) and face response (deg)."""

    t_s: float
    theta_deg: float
    phi_preload_deg: float
    phi_slack_deg: float
    tau_net_nm: float
    tau_off_required_nm: float
    tau_left_in_nm: float
    tau_right_in_nm: float
    tau_left_off_nm: float
    tau_right_off_nm: float
    force_left_in_n: float
    force_right_in_n: float
    force_left_off_n: float
    force_right_off_n: float
    face_angle_impact_deg: float
    club_path_impact_deg: float


def post_impact_speed_ratio() -> float:
    """Head speed after/before impact: ``(M - e m) / (M + m)`` (restitution law)."""
    m, head = float(GOLF_BALL_MASS_KG), STUDIO_HEAD_MASS_KG
    return (head - float(PUTTER_FACE_COR) * m) / (head + m)


def ou_noise(t: float, seed_offset: float) -> float:
    """Smooth multi-sine hand noise in [-1, 1] (the studio's ``smoothOUNoise``)."""
    return (
        0.45 * math.sin(2.0 * math.pi * 1.7 * t + seed_offset)
        + 0.32 * math.cos(2.0 * math.pi * 3.1 * t + 1.9 * seed_offset)
        + 0.23 * math.sin(2.0 * math.pi * 4.8 * t - 0.7 * seed_offset)
    )


def _quintic(s: float, duration: float) -> tuple[float, float, float]:
    """Minimum-jerk position, rate and acceleration at phase ``s`` of ``duration``."""
    p = 10 * s**3 - 15 * s**4 + 6 * s**5
    dp = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    ddp = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
    return p, dp, ddp


def stroke_kinematics(t_s: float) -> tuple[float, float, float]:
    """Shaft angle (rad), rate and acceleration; the finish is held after ``T_STROKE``."""
    ti = min(t_s, T_STROKE_S)
    th_max = math.radians(THETA_MAX_DEG)
    if ti <= T_BACK_S:
        p, dp, ddp = _quintic(ti / T_BACK_S, T_BACK_S)
        return -th_max * p, -th_max * dp, -th_max * ddp
    rest = T_STROKE_S - T_BACK_S
    p, dp, ddp = _quintic(min((ti - T_BACK_S) / rest, 1.0), rest)
    gain = DOWNSWING_GAIN * th_max
    if ti <= T_IMPACT_S:
        return -th_max + gain * p, gain * dp, gain * ddp
    decel = post_impact_speed_ratio()
    moving = 0.0 if t_s > T_STROKE_S else decel
    return (
        gain * (p - 1.0 / DOWNSWING_GAIN) * decel,
        gain * dp * moving,
        gain * ddp * moving,
    )


def _split(required: float, smooth: float) -> tuple[float, float]:
    """Smooth positive and negative parts of ``required`` (they differ by it)."""
    root = math.sqrt(required * required + smooth * smooth)
    return 0.5 * (required + root), 0.5 * (-required + root)


def evaluate(t_s: float, inputs: CounterTorqueInputs) -> CounterTorqueSample:
    """The couple model at time ``t_s`` (s) of the stroke."""
    ti = min(t_s, T_STROKE_S)
    th, om, al = stroke_kinematics(t_s)
    beta = math.radians(90.0 - inputs.lie_deg)
    weight = _CLUB_MASS_KG * _STUDIO_GRAVITY_M_S2 * _CLUB_COM_M
    tau_net = 0.38 * al + 0.35 * om + weight * math.cos(beta) * math.sin(th)

    pre, off_pre, noise = inputs.preload_nm, inputs.off_preload_nm, inputs.noise_scale
    n_scale = 0.045 * noise
    pos, neg = _split(tau_net, _SMOOTH_IN_NM)
    tau_l_in = -(pre + neg) + n_scale * (0.8 + pre) * ou_noise(ti, 1.1)
    tau_r_in = +(pre + pos) + n_scale * (0.8 + pre) * ou_noise(ti, 2.7)

    tau_cent = 0.22 * om * om * math.cos(th) + 0.045 * al
    tau_off = -(-weight * math.sin(beta) * math.cos(th) + tau_cent)
    pos_off, neg_off = _split(tau_off, _SMOOTH_OFF_NM)
    tau_l_off = +(off_pre + pos_off) + n_scale * (0.6 + off_pre) * ou_noise(ti, 4.3)
    tau_r_off = -(off_pre + neg_off) + n_scale * (0.6 + off_pre) * ou_noise(ti, 5.9)

    gust = inputs.gust_impulse_nm / 0.25
    wave = math.exp(-(((ti - 0.72) / 0.065) ** 2)) * math.sin(
        14.0 * max(0.0, ti - 0.66)
    )
    damping = 1.80 / (1.80 + 14.5 * pre)
    tremor = 0.018 * noise * max(0.05, pre) ** 1.42 * ou_noise(ti, 3.7)
    slack_noise = 0.045 * noise * ou_noise(ti, 3.7)

    stiff = max(0.22, pre + 0.5 * off_pre)
    face_offset = PUTTER_SPECS[inputs.putter].e_face_m * 8.0 / stiff
    face = (0.14 + 0.35 * noise / stiff) * ou_noise(T_IMPACT_S, 2.1) + face_offset
    path = (0.08 + 0.22 * noise / stiff) * ou_noise(T_IMPACT_S, 4.7)
    return CounterTorqueSample(
        t_s=t_s,
        theta_deg=math.degrees(th),
        phi_preload_deg=1.401 * gust * damping * wave + tremor,
        phi_slack_deg=1.401 * gust * wave + slack_noise,
        tau_net_nm=tau_net,
        tau_off_required_nm=tau_off,
        tau_left_in_nm=tau_l_in,
        tau_right_in_nm=tau_r_in,
        tau_left_off_nm=tau_l_off,
        tau_right_off_nm=tau_r_off,
        force_left_in_n=-tau_l_in / HAND_LEVER_M,
        force_right_in_n=tau_r_in / HAND_LEVER_M,
        force_left_off_n=tau_l_off / HAND_LEVER_M,
        force_right_off_n=tau_r_off / HAND_LEVER_M,
        face_angle_impact_deg=face,
        club_path_impact_deg=path,
    )


def stroke_series(
    inputs: CounterTorqueInputs, n_points: int = 210
) -> tuple[CounterTorqueSample, ...]:
    """``n_points + 1`` samples evenly over the stroke (the studio's chart grid)."""
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    times = np.linspace(0.0, T_STROKE_S, n_points + 1)
    return tuple(evaluate(float(t), inputs) for t in times)
