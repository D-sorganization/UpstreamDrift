"""Track D bounded-PD free-root MuJoCo controller and duration sweep (#15837)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
import logging
import math
from typing import Any, Literal

import mujoco
import numpy as np

from src.shared.python.core.physics_constants import GRAVITY_M_S2
from ._validation import check_nonneg_float, check_pos_float

logger = logging.getLogger(__name__)
SCORECARD_SCHEMA_VERSION = "markerless-scorecard/1.0.0"
TIMEBASE_UNVERIFIED = "unverified"
QUALIFICATION_IMAGE_ONLY = "image_observations_only"
HELD_OUT_PHASE_NAMES = ("address", "top", "downswing", "impact", "finish")
DEFAULT_PHASE_FRACTIONS = (0.0, 0.35, 0.55, 0.75, 1.0)
ClaimLabel = Literal["dynamic_feasible_relative", "below_gate"]
Arr = np.ndarray


def _unit(value: object, name: str) -> float:
    """Validate finite float in [0.0, 1.0]."""
    val = check_nonneg_float(value, name)
    if val > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
    return val


def _honest_ok(tb: str, verified: bool, qual: str) -> None:
    """Verify unverified-timebase and image-observations-only invariants."""
    if tb != TIMEBASE_UNVERIFIED or verified or qual != QUALIFICATION_IMAGE_ONLY:
        raise ValueError("Honest timing/qualification flags cannot be elevated")


@dataclass(frozen=True, slots=True)
class ConfidenceGainSchedule:
    """Monotonic confidence-dependent PD gain schedule from min to max."""

    kp_min: float = 15.0
    kp_max: float = 80.0
    kd_min: float = 1.5
    kd_max: float = 6.0
    exponent: float = 1.0

    def __post_init__(self) -> None:
        """Validate monotonic gain bounds and positive exponent."""
        for v in (self.kp_min, self.kp_max, self.kd_min, self.kd_max):
            check_nonneg_float(v, "gain")
        check_pos_float(self.exponent, "exponent")
        if self.kp_max < self.kp_min or self.kd_max < self.kd_min:
            raise ValueError("Max PD gains cannot be less than min PD gains")

    def gains(self, confidence: float) -> tuple[float, float]:
        """Map confidence in [0, 1] monotonically to (kp, kd)."""
        w = _unit(confidence, "confidence") ** self.exponent
        kp = float(self.kp_min + w * (self.kp_max - self.kp_min))
        kd = float(self.kd_min + w * (self.kd_max - self.kd_min))
        return kp, kd


@dataclass(frozen=True, slots=True)
class PdControllerConfig:
    """Configuration and physical feasibility gates for BoundedPdController."""

    gain_schedule: ConfidenceGainSchedule = field(
        default_factory=ConfidenceGainSchedule
    )
    tau_limit_nm: float = 120.0
    max_tracking_rmse_rad: float = 0.35
    max_grip_drift_m: float = 0.025
    max_foot_slip_m: float = 0.020
    max_saturation_fraction: float = 0.85

    def __post_init__(self) -> None:
        """Validate torque limit and feasibility gate thresholds."""
        if not isinstance(self.gain_schedule, ConfidenceGainSchedule):
            raise TypeError("gain_schedule must be ConfidenceGainSchedule")
        for n in ("tau_limit_nm", "max_tracking_rmse_rad", "max_grip_drift_m"):
            check_pos_float(getattr(self, n), n)
        check_pos_float(self.max_foot_slip_m, "max_foot_slip_m")
        _unit(self.max_saturation_fraction, "max_saturation_fraction")


@dataclass(frozen=True, slots=True)
class PdStepTelemetry:
    """Per-step telemetry logged during closed-loop MuJoCo PD rollout."""

    step_index: int
    time_s: float
    confidence: float
    kp: float
    kd: float
    joint_error_rmse_rad: float
    grip_drift_m: float
    foot_slip_m: float
    peak_torque_nm: float
    saturation_fraction: float
    applied_torques_nm: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate step index, non-empty torques, and unit bounds."""
        if self.step_index < 0 or not self.applied_torques_nm:
            raise ValueError("step_index >= 0 and non-empty torques required")
        _unit(self.confidence, "confidence")
        _unit(self.saturation_fraction, "saturation_fraction")


@dataclass(frozen=True, slots=True)
class DurationSweepResult:
    """Closed-loop MuJoCo evaluation for one assumed swing duration."""

    assumed_duration_s: float
    claim_label: ClaimLabel
    gates_passed: bool
    failing_phases: tuple[str, ...]
    tracking_rmse_rad: float
    max_grip_drift_m: float
    max_foot_slip_m: float
    peak_torque_nm: float
    saturation_fraction: float
    phase_metrics: dict[str, dict[str, float]]
    step_telemetry: tuple[PdStepTelemetry, ...]
    timebase: str = TIMEBASE_UNVERIFIED
    physical_time_verified: bool = False
    qualification: str = QUALIFICATION_IMAGE_ONLY

    def __post_init__(self) -> None:
        """Validate honest claim labels, timing flags, and positive duration."""
        check_pos_float(self.assumed_duration_s, "assumed_duration_s")
        if self.claim_label not in ("dynamic_feasible_relative", "below_gate"):
            raise ValueError(f"Unsupported claim_label: {self.claim_label!r}")
        _honest_ok(self.timebase, self.physical_time_verified, self.qualification)

    def to_dict(self) -> dict[str, Any]:
        """Serialize sweep result to a JSON-compatible dictionary."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["failing_phases"] = list(self.failing_phases)
        d["step_count"] = len(d.pop("step_telemetry"))
        return d


@dataclass(frozen=True, slots=True)
class TrackDDurationSweepReport:
    """Markerless-scorecard/1.0.0 report across assumed swing durations."""

    assumed_durations_s: tuple[float, ...]
    results: tuple[DurationSweepResult, ...]
    best_duration_s: float | None
    claim_label: ClaimLabel
    failing_phases: tuple[str, ...]
    schema_version: str = SCORECARD_SCHEMA_VERSION
    track: str = "track_d_bounded_pd"
    phase_names: tuple[str, ...] = HELD_OUT_PHASE_NAMES
    timebase: str = TIMEBASE_UNVERIFIED
    physical_time_verified: bool = False
    qualification: str = QUALIFICATION_IMAGE_ONLY

    def __post_init__(self) -> None:
        """Validate scorecard schema version, timing flags, and alignment."""
        if self.schema_version != SCORECARD_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCORECARD_SCHEMA_VERSION!r}")
        _honest_ok(self.timebase, self.physical_time_verified, self.qualification)
        if not self.results or len(self.results) != len(self.assumed_durations_s):
            raise ValueError("assumed_durations_s and results must match")

    def to_dict(self) -> dict[str, Any]:
        """Serialize full Track D sweep report matching markerless-scorecard/1.0.0."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["assumed_durations_s"] = list(self.assumed_durations_s)
        d["failing_phases"] = list(self.failing_phases)
        d["held_out_phases"] = list(d.pop("phase_names"))
        d["duration_sweeps"] = [r.to_dict() for r in d.pop("results")]
        return d


def build_free_root_test_mjcf() -> str:
    """Build a minimal free-root articulated golfer+club MJCF for tests."""
    gz, f = -float(GRAVITY_M_S2), 'contype="1" conaffinity="1" solref="0.01 1"'
    cl = '<body pos="0.20 -0.09 -0.08"><joint name="wc" axis="0 1 0" range="-1.8 1.8" damping="0.15" armature="0.05"/><geom fromto="0 0 0 0.15 0 -0.55" size="0.01" mass="0.35"/><geom type="sphere" pos="0.15 0 -0.55" size="0.03" mass="0.2"/></body>'
    arm = '<body pos="0.05 {1} 0.40"><joint name="{0}y" axis="0 0 1" range="-2.2 2.2" damping="0.4" armature="0.08"/><joint name="{0}p" axis="0 1 0" range="-2.2 2.2" damping="0.4" armature="0.08"/><geom fromto="0 0 0 0.16 {2} -0.22" size="0.035" mass="1.8"/><body pos="0.16 {2} -0.22"><joint name="{0}e" axis="0 1 0" range="-2.0 2.0" damping="0.3" armature="0.06"/><geom fromto="0 0 0 0.20 {2} -0.08" size="0.03" mass="1.2"/><site name="{0}_hand_grip" pos="0.20 {2} -0.08" size="0.02"/>{3}</body></body>'
    arms = arm.format("lead", 0.18, -0.09, cl) + arm.format("trail", -0.18, 0.09, "")
    return f"""<mujoco model="tiger_pd"><compiler angle="radian"/><option timestep="0.002" gravity="0 0 {gz:.6f}" integrator="RK4"/>
<default><joint type="hinge"/><geom type="capsule" contype="0" conaffinity="0"/></default><worldbody>
<geom type="plane" pos="0 0 0" size="5 5 0.1" condim="3" {f} friction="1.2 0.05 0.01"/>
<body name="pelvis" pos="0 0 0.50"><freejoint name="root"/><geom fromto="0 -0.15 0 0 0.15 0" size="0.08" mass="35"/>
<geom type="sphere" pos="0.2 0.2 -0.47" size="0.03" mass="1.2" {f} friction="1.2 0.05 0.01"/><geom type="sphere" pos="-0.2 0.2 -0.47" size="0.03" mass="1.2" {f} friction="1.2 0.05 0.01"/><geom type="sphere" pos="0.2 -0.2 -0.47" size="0.03" mass="1.2" {f} friction="1.2 0.05 0.01"/><geom type="sphere" pos="-0.2 -0.2 -0.47" size="0.03" mass="1.2" {f} friction="1.2 0.05 0.01"/>
<site name="lead_foot_contact" pos="0 0.20 -0.50" size="0.02"/><body name="torso" pos="0 0 0.15">
<joint name="ty" axis="0 0 1" range="-1.8 1.8" damping="1.5" armature="0.15"/><joint name="tp" axis="0 1 0" range="-0.8 0.8" damping="1.5" armature="0.15"/>
<geom fromto="0 0 0 0 0 0.45" size="0.09" mass="20"/>{arms}</body></body></worldbody>
<actuator><motor name="ty" joint="ty" gear="1"/><motor name="tp" joint="tp" gear="1"/><motor name="leady" joint="leady" gear="1"/><motor name="leadp" joint="leadp" gear="1"/><motor name="leade" joint="leade" gear="1"/><motor name="wc" joint="wc" gear="1"/><motor name="traily" joint="traily" gear="1"/><motor name="trailp" joint="trailp" gear="1"/><motor name="traile" joint="traile" gear="1"/></actuator></mujoco>"""


class BoundedPdController:
    """Confidence-scheduled, torque-clamped PD controller for free-root MuJoCo."""

    def __init__(self, config: PdControllerConfig | None = None) -> None:
        """Initialize the controller with validated configuration."""
        self.config = config or PdControllerConfig()

    def schedule_gains(self, confidence: float) -> tuple[float, float]:
        """Return scheduled (kp, kd) for observation confidence in [0, 1]."""
        sched = self.config.gain_schedule
        return sched.gains(confidence)

    def compute_torques(
        self, q_err: Arr, qd_err: Arr, confidence: float
    ) -> tuple[Arr, float, float, float]:
        """Compute bounded PD torques and saturation fraction."""
        qa = np.asarray(q_err, dtype=np.float64).reshape(-1)
        qda = np.asarray(qd_err, dtype=np.float64).reshape(-1)
        if qa.size == 0 or qa.shape != qda.shape:
            raise ValueError("q_err and qd_err must match")
        if not (np.all(np.isfinite(qa)) and np.all(np.isfinite(qda))):
            raise ValueError("q_err and qd_err must be finite")
        kp, kd = self.schedule_gains(confidence)
        raw, lim = kp * qa + kd * qda, self.config.tau_limit_nm
        sat = float(np.sum(np.abs(raw) > lim) / qa.size)
        return np.clip(raw, -lim, lim), kp, kd, sat


def _model_indices(model: mujoco.MjModel) -> tuple[Arr, Arr, tuple[int, int, int]]:
    """Extract actuator joint indices and verify required grip/foot site IDs."""
    if model.nu <= 0:
        raise ValueError("Model must define at least one actuator")
    jids = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
    qp = np.asarray([int(model.jnt_qposadr[j]) for j in jids], dtype=np.int64)
    df = np.asarray([int(model.jnt_dofadr[j]) for j in jids], dtype=np.int64)
    sites = ("lead_hand_grip", "trail_hand_grip", "lead_foot_contact")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in sites]
    if any(sid < 0 for sid in ids):
        raise ValueError("Model missing required grip or foot contact site(s)")
    return qp, df, (int(ids[0]), int(ids[1]), int(ids[2]))


def _resample(
    model: mujoco.MjModel, tq: Arr, confs: Sequence[float] | Arr | None, dur_s: float
) -> tuple[Arr, Arr, Arr, float]:
    """Validate and resample target trajectory, velocities, and confidences."""
    check_pos_float(dur_s, "duration_s")
    qt = np.asarray(tq, dtype=np.float64)
    if qt.ndim != 2 or qt.shape[0] < 2 or qt.shape[1] != model.nu:
        raise ValueError("Invalid target_trajectory shape")
    if not np.all(np.isfinite(qt)):
        raise ValueError("target_trajectory must be finite")
    dt = float(model.opt.timestep)
    n = max(2, round(dur_s / dt))
    uk, us = np.linspace(0.0, 1.0, qt.shape[0]), np.linspace(0.0, 1.0, n)
    qg = np.column_stack([np.interp(us, uk, qt[:, j]) for j in range(model.nu)])
    qdg = np.gradient(qg, dt, axis=0)
    if confs is None:
        return qg, qdg, np.ones(n, dtype=np.float64), dt
    cvec = np.asarray(confs, dtype=np.float64).reshape(-1)
    if cvec.shape[0] != qt.shape[0]:
        raise ValueError("confidences length mismatch")
    cg = np.interp(us, uk, np.asarray([_unit(float(v), "c") for v in cvec]))
    return qg, qdg, cg, dt


def run_pd_rollout(
    model: mujoco.MjModel,
    target_trajectory: Arr,
    confidences: Sequence[float] | Arr | None = None,
    *,
    duration_s: float = 1.0,
    controller: BoundedPdController | None = None,
    initial_qvel: Arr | None = None,
) -> tuple[PdStepTelemetry, ...]:
    """Run closed-loop free-root MuJoCo mj_step rollout under BoundedPdController."""
    ctrl = controller or BoundedPdController()
    qg, qdg, cg, dt = _resample(model, target_trajectory, confidences, duration_s)
    qp, df, (lid, tid, fid) = _model_indices(model)
    data = mujoco.MjData(model)
    data.qpos[qp] = qg[0]
    if initial_qvel is not None:
        v0 = np.asarray(initial_qvel, dtype=np.float64).reshape(-1)
        if v0.shape != (model.nv,) or not np.all(np.isfinite(v0)):
            raise ValueError("Invalid initial_qvel")
        data.qvel[:] = v0
    mujoco.mj_forward(model, data)
    foot0, tel = data.site_xpos[fid][:2].copy(), []
    for i in range(qg.shape[0]):
        c, qe, qde = float(cg[i]), qg[i] - data.qpos[qp], qdg[i] - data.qvel[df]
        tau, kp, kd, sat = ctrl.compute_torques(qe, qde, c)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        mujoco.mj_kinematics(model, data)
        sp, err = data.site_xpos, qg[i] - data.qpos[qp]
        rmse = float(np.sqrt(np.mean(err * err)))
        drift = float(np.linalg.norm(sp[lid] - sp[tid]))
        slip = float(np.linalg.norm(sp[fid][:2] - foot0))
        pk, tq = float(np.max(np.abs(tau))), tuple(float(v) for v in tau)
        args = (i, float(i * dt), c, kp, kd, rmse, drift, slip, pk, sat, tq)
        tel.append(PdStepTelemetry(*args))
    return tuple(tel)


def _score_rollout(
    dur_s: float, tel: tuple[PdStepTelemetry, ...], cfg: PdControllerConfig
) -> DurationSweepResult:
    """Audit held-out phase frames and score rollout metrics."""
    n, fr, nms = len(tel), DEFAULT_PHASE_FRACTIONS, HELD_OUT_PHASE_NAMES
    pidx = {nms[k]: min(n - 1, max(0, round(fr[k] * (n - 1)))) for k in range(len(nms))}
    keys = ("tracking_rmse_rad", "grip_drift_m", "foot_slip_m", "saturation_fraction")
    lims = {k: float(getattr(cfg, f"max_{k}")) for k in keys}
    attrs = ("joint_error_rmse_rad",) + keys[1:]
    pm = {
        nm: {k: float(getattr(tel[i], a)) for k, a in zip(keys, attrs, strict=True)}
        for nm, i in pidx.items()
    }
    failing = [nm for nm in nms if any(pm[nm][k] > lims[k] for k in keys)]
    for lb, at in zip(keys[1:3], attrs[1:3], strict=True):
        val, step_i = max((float(getattr(s, at)), s.step_index) for s in tel)
        _, ph = min((abs(pidx[nm] - step_i), nm) for nm in nms)
        if val > lims[lb] and ph not in failing:
            failing.append(ph)
    ok = not failing
    rmse = math.sqrt(sum(s.joint_error_rmse_rad**2 for s in tel) / n)
    sat = sum(s.saturation_fraction for s in tel) / n
    mg, ms = max(s.grip_drift_m for s in tel), max(s.foot_slip_m for s in tel)
    pt = max(s.peak_torque_nm for s in tel)
    lbl: ClaimLabel = "dynamic_feasible_relative" if ok else "below_gate"
    h = (float(dur_s), lbl, ok, tuple(failing), float(rmse))
    return DurationSweepResult(*h, mg, ms, pt, float(sat), pm, tel)


def run_duration_sweep(
    model: mujoco.MjModel,
    target_trajectory: Arr,
    assumed_durations_s: Sequence[float] = (0.85, 1.05, 1.25),
    *,
    confidences: Sequence[float] | Arr | None = None,
    controller: BoundedPdController | None = None,
    initial_qvel: Arr | None = None,
) -> TrackDDurationSweepReport:
    """Sweep assumed swing durations over closed-loop MuJoCo PD rollouts."""
    if not assumed_durations_s:
        raise ValueError("assumed_durations_s must be non-empty")
    ctrl = controller or BoundedPdController()
    durs = tuple(check_pos_float(d, "assumed_durations_s") for d in assumed_durations_s)
    t, c, v, res_l = target_trajectory, confidences, initial_qvel, []
    for d in durs:
        tel = run_pd_rollout(model, t, c, duration_s=d, controller=ctrl, initial_qvel=v)
        res_l.append(_score_rollout(d, tel, ctrl.config))
    res = tuple(res_l)
    passing = [r for r in res if r.gates_passed]
    best_r = min(passing, key=lambda r: r.tracking_rmse_rad) if passing else None
    best = best_r.assumed_duration_s if best_r else None
    seen = {p for r in res for p in r.failing_phases}
    fail_all = () if passing else tuple(p for p in HELD_OUT_PHASE_NAMES if p in seen)
    lbl: ClaimLabel = "dynamic_feasible_relative" if passing else "below_gate"
    return TrackDDurationSweepReport(durs, res, best, lbl, fail_all)
