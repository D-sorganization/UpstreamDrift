"""Finish-feasibility metrics for matched full-body swings (Balance-1, #11668).

The matched swings lose balance after impact. The diagnosis behind epic #11667
showed that the tracked *reference* is itself dynamically infeasible there: its
zero-moment point leaves the foot hull, its required vertical force collapses and
its feet would have to slide. This module turns that diagnosis into receipt
metrics, computed identically for the reference (inverse dynamics: the load the
reference demands) and the simulation (the load the plant actually delivers).

Metrics, over the finish window ``[1.0 s, end]``:

* ZMP-inside-support fraction (also over the epic's ``1.0 - 1.5 s`` window).
  A frame counts as supported only if it carries load and its pressure point
  lies in the foot hull, so an unloaded frame is never "inside".
* Friction-cone utilisation: tangential over normal force, divided by the
  dynamic friction coefficient (the limit the contact law sustains once a foot
  slides, so utilisation near 1 means the foot is slipping). Reported as its maximum and as the fraction of
  loaded frames above ``0.95 mu``. The reference has no foot-force split, so it
  uses the net ground reaction; the simulation uses the larger of the two feet.
* Foot slide (mm) and foot yaw pivot (degrees): worst-foot displacement of the
  foot centre and of the heel-to-toe axis relative to the start of the window.
* Pelvis yaw error (degrees): simulated minus tracked pelvis yaw. For the
  reference it is the tracked-minus-IK yaw when the IK trajectory is supplied.
* Minimum and maximum vertical force in body weights.

The pure functions take arrays only; ``reference_history`` and
``simulation_history`` extract those arrays from a simulator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.contracts import postcondition

if TYPE_CHECKING:
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
    from src.shared.python.motion_matching.contact_law import GroundPlane

FINISH_START_S = 1.0
ZMP_WINDOW_END_S = 1.5
FRICTION_SATURATION = 0.95
MIN_LOAD_BW = 0.1
FOOT_NORMAL_FLOOR_BW = 0.05
PELVIS_YAW_COORDINATE = "HipInputZ"

ReportDict = dict[str, Any]

# Regression ratchet (#11668): these may only improve against a recorded baseline.
FLOOR_METRICS = ("zmp_inside_fraction_1_0_to_1_5s", "zmp_inside_fraction_finish")
CEILING_METRICS = ("friction_saturated_fraction",)
RATCHET_TOLERANCE = 1e-6


def _check_series(name: str, values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}")
    return arr


@dataclass(frozen=True)
class FootTrack:
    """Heel, toe and contact-centre positions of one foot, each ``(N, 3)``."""

    heel: np.ndarray
    toe: np.ndarray
    centre: np.ndarray

    def __post_init__(self) -> None:
        shapes = {np.shape(a) for a in (self.heel, self.toe, self.centre)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError(f"foot tracks must share one (N, 3) shape, got {shapes}")
        if next(iter(shapes))[1] != 3:
            raise ValueError("foot tracks must be (N, 3) positions")


@dataclass(frozen=True)
class FinishHistory:
    """Per-frame quantities the finish metrics are computed from.

    Attributes:
        time_s: Strictly increasing times (N,).
        vertical_bw: Vertical ground reaction in body weights (N,).
        friction_ratio: Tangential over normal force (N,); only read where the
            frame is loaded (``vertical_bw >= MIN_LOAD_BW``).
        supported: True where the frame is loaded and its pressure point is
            inside the support polygon (N,).
        feet: Foot tracks by side label.
        pelvis_yaw_error_rad: Optional pelvis yaw error (N,).
    """

    time_s: np.ndarray
    vertical_bw: np.ndarray
    friction_ratio: np.ndarray
    supported: np.ndarray
    feet: Mapping[str, FootTrack]
    pelvis_yaw_error_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=float)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("time_s must be 1-D, strictly increasing, length >= 2")
        n = t.size
        _check_series("vertical_bw", self.vertical_bw, n)
        _check_series("friction_ratio", self.friction_ratio, n)
        if not np.isfinite(np.asarray(self.vertical_bw, dtype=float)).all():
            raise ValueError("vertical_bw must be finite")
        if np.shape(self.supported) != (n,):
            raise ValueError(f"supported must have shape ({n},)")
        if not self.feet:
            raise ValueError("at least one foot track is required")
        for side, track in self.feet.items():
            if track.heel.shape != (n, 3):
                raise ValueError(f"foot {side!r} must have shape ({n}, 3)")
        if self.pelvis_yaw_error_rad is not None:
            _check_series("pelvis_yaw_error_rad", self.pelvis_yaw_error_rad, n)


def _window(times: np.ndarray, start_s: float, end_s: float | None) -> np.ndarray:
    mask = times >= start_s
    if end_s is not None:
        mask &= times < end_s
    if not mask.any():
        raise ValueError(f"no samples in window [{start_s}, {end_s}) s")
    return mask


def supported_fraction(
    times: np.ndarray,
    supported: np.ndarray,
    *,
    start_s: float,
    end_s: float | None = None,
) -> float:
    """Fraction of frames in ``[start_s, end_s)`` whose pressure point is supported."""
    t = np.asarray(times, dtype=float)
    if np.shape(supported) != t.shape:
        raise ValueError("times and supported must share a shape")
    return float(np.asarray(supported, dtype=bool)[_window(t, start_s, end_s)].mean())


def friction_utilisation(
    ratio: np.ndarray, vertical: np.ndarray, mu: float, *, min_vertical: float
) -> np.ndarray:
    """Tangential-over-normal ``ratio`` divided by ``mu``; NaN where unloaded."""
    if not mu > 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    r = np.asarray(ratio, dtype=float)
    v = np.asarray(vertical, dtype=float)
    if r.shape != v.shape:
        raise ValueError("ratio and vertical force must share a shape")
    return np.where(v >= min_vertical, r / mu, np.nan)


def friction_summary(
    times: np.ndarray,
    utilisation: np.ndarray,
    *,
    start_s: float,
    saturation: float = FRICTION_SATURATION,
) -> tuple[float, float]:
    """Peak utilisation and the fraction of loaded frames above ``saturation``.

    Frames with NaN utilisation (unloaded) are excluded from both. A window with
    no loaded frame reports ``(0.0, 0.0)``: nothing is pressing on the ground.
    """
    t = np.asarray(times, dtype=float)
    u = np.asarray(utilisation, dtype=float)[_window(t, start_s, None)]
    u = u[np.isfinite(u)]
    if u.size == 0:
        return 0.0, 0.0
    return float(u.max()), float((u > saturation).mean())


def foot_slide_and_yaw(
    feet: Mapping[str, FootTrack], times: np.ndarray, *, start_s: float
) -> tuple[float, float]:
    """Worst-foot slide (mm) and yaw pivot (degrees) relative to ``start_s``."""
    t = np.asarray(times, dtype=float)
    idx = np.flatnonzero(_window(t, start_s, None))
    slide_mm = yaw_deg = 0.0
    for track in feet.values():
        centre = track.centre[idx, :2]
        slide_mm = max(
            slide_mm, float(np.linalg.norm(centre - centre[0], axis=1).max() * 1e3)
        )
        axis = (track.toe - track.heel)[idx, :2]
        yaw = np.degrees(np.unwrap(np.arctan2(axis[:, 1], axis[:, 0])))
        yaw_deg = max(yaw_deg, float(np.abs(yaw - yaw[0]).max()))
    return slide_mm, yaw_deg


def pelvis_yaw_error_deg(
    times: np.ndarray, error_rad: np.ndarray, *, start_s: float
) -> tuple[float, float]:
    """Largest absolute and final pelvis yaw error (degrees) in the window."""
    t = np.asarray(times, dtype=float)
    err = np.degrees(np.asarray(error_rad, dtype=float)[_window(t, start_s, None)])
    return float(np.abs(err).max()), float(err[-1])


def vertical_force_range(
    times: np.ndarray, vertical_bw: np.ndarray, *, start_s: float
) -> tuple[float, float]:
    """Minimum and maximum vertical force (body weights) in the window."""
    t = np.asarray(times, dtype=float)
    fz = np.asarray(vertical_bw, dtype=float)[_window(t, start_s, None)]
    return float(fz.min()), float(fz.max())


@postcondition(
    lambda result: 0.0 <= result["zmp_inside_fraction_finish"] <= 1.0,
    "inside fraction is a fraction",
)
def summarise_history(
    history: FinishHistory,
    *,
    mu: float,
    start_s: float = FINISH_START_S,
    zmp_end_s: float = ZMP_WINDOW_END_S,
) -> ReportDict:
    """Finish-feasibility metrics of one trajectory (reference or simulation)."""
    if not mu > 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    t = np.asarray(history.time_s, dtype=float)
    util = friction_utilisation(
        history.friction_ratio, history.vertical_bw, mu, min_vertical=MIN_LOAD_BW
    )
    util_max, util_frac = friction_summary(t, util, start_s=start_s)
    slide_mm, yaw_pivot = foot_slide_and_yaw(history.feet, t, start_s=start_s)
    fz_min, fz_max = vertical_force_range(t, history.vertical_bw, start_s=start_s)
    yaw_max: float | None = None
    yaw_final: float | None = None
    if history.pelvis_yaw_error_rad is not None:
        yaw_max, yaw_final = pelvis_yaw_error_deg(
            t, history.pelvis_yaw_error_rad, start_s=start_s
        )
    return {
        "zmp_inside_fraction_1_0_to_1_5s": supported_fraction(
            t, history.supported, start_s=start_s, end_s=zmp_end_s
        ),
        "zmp_inside_fraction_finish": supported_fraction(
            t, history.supported, start_s=start_s
        ),
        "friction_utilisation_max": util_max,
        "friction_saturated_fraction": util_frac,
        "foot_slide_mm_max": slide_mm,
        "foot_yaw_pivot_deg_max": yaw_pivot,
        "pelvis_yaw_error_deg_max": yaw_max,
        "pelvis_yaw_error_deg_final": yaw_final,
        "vertical_force_bw_min": fz_min,
        "vertical_force_bw_max": fz_max,
    }


# -- extraction from a simulator ------------------------------------------------


def _sphere_centres(sim: fs.FullBodySimulator, q: np.ndarray) -> dict[str, np.ndarray]:
    """World centres of the contact spheres at ``q`` (any engine adapter)."""
    adapter = sim.adapter
    names = list(adapter._spheres)
    if hasattr(adapter, "get_sphere_kinematics"):
        coords = sim._map(q)
        zero = dict.fromkeys(sim.names, 0.0)
        return {
            s: np.asarray(adapter.get_sphere_kinematics(s, coords, zero)[0], float)
            for s in names
        }
    adapter.frame_poses(sim._map(q))
    return {
        s: np.array(adapter.data.site_xpos[adapter._spheres[s]["site_id"]], float)
        for s in names
    }


def _side_groups(names: list[str]) -> dict[str, list[str]]:
    """Contact spheres by foot side; a side needs a heel and a toe or forefoot."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(name.rsplit("_", 1)[-1], []).append(name)
    return {
        side: members
        for side, members in groups.items()
        if f"heel_{side}" in members and _toe_name(members, side) is not None
    }


def _toe_name(members: list[str], side: str) -> str | None:
    for stem in ("toe", "forefoot"):
        if f"{stem}_{side}" in members:
            return f"{stem}_{side}"
    return None


def _foot_tracks(
    positions: list[dict[str, np.ndarray]], groups: dict[str, list[str]]
) -> dict[str, FootTrack]:
    tracks = {}
    for side, members in groups.items():
        series = {m: np.array([p[m] for p in positions]) for m in members}
        toe = _toe_name(members, side)
        tracks[side] = FootTrack(
            heel=series[f"heel_{side}"],
            toe=series[str(toe)],
            centre=np.mean([series[m] for m in members], axis=0),
        )
    return tracks


def _ground_normal(ground: GroundPlane) -> np.ndarray:
    n = np.asarray(ground.normal, dtype=float)
    n = n / np.linalg.norm(n)
    if abs(n[2]) < 0.999:
        raise ValueError("finish metrics assume a z-up ground plane")
    return n


def _yaw_index(sim: fs.FullBodySimulator) -> int:
    return list(sim.names).index(PELVIS_YAW_COORDINATE)


def reference_history(
    sim: fs.FullBodySimulator,
    times: np.ndarray,
    q_track: np.ndarray,
    zmp: Mapping[str, np.ndarray],
    ground: GroundPlane,
    *,
    q_ik: np.ndarray | None = None,
) -> FinishHistory:
    """History of the tracked reference from its ``reference_zmp`` demand."""
    n = _ground_normal(ground)
    grf = np.asarray(zmp["grf_over_weight"], dtype=float)
    vertical = grf @ n
    horizontal = np.linalg.norm(grf - np.outer(vertical, n), axis=1)
    ratio = np.divide(
        horizontal, vertical, out=np.zeros_like(vertical), where=vertical > 1e-9
    )
    supported = (np.asarray(zmp["outside_m"]) <= 0.0) & ~np.asarray(
        zmp["unloaded"], dtype=bool
    )
    groups = _side_groups(list(sim.adapter._spheres))
    feet = _foot_tracks([_sphere_centres(sim, q) for q in q_track], groups)
    yaw_err = None
    if q_ik is not None:
        col = _yaw_index(sim)
        yaw_err = np.asarray(q_track)[:, col] - np.asarray(q_ik)[:, col]
    return FinishHistory(
        time_s=np.asarray(times, dtype=float),
        vertical_bw=vertical,
        friction_ratio=ratio,
        supported=supported,
        feet=feet,
        pelvis_yaw_error_rad=yaw_err,
    )


def _foot_friction_ratio(
    samples: Mapping[str, Any],
    groups: dict[str, list[str]],
    n: np.ndarray,
    weight_n: float,
) -> float:
    """Larger tangential/normal ratio of the loaded feet (sum of sphere magnitudes)."""
    worst = 0.0
    for members in groups.values():
        normal = sum(float(samples[m].normal_force_n @ n) for m in members)
        if normal < FOOT_NORMAL_FLOOR_BW * weight_n:
            continue
        tangential = sum(
            float(np.linalg.norm(samples[m].friction_force_n)) for m in members
        )
        worst = max(worst, tangential / normal)
    return worst


def simulation_history(
    sim: fs.FullBodySimulator,
    record: fs.SimulationRecord,
    times_track: np.ndarray,
    q_track: np.ndarray,
    ground: GroundPlane,
) -> FinishHistory:
    """History of the simulated plant: contact forces, feet and pelvis yaw."""
    n = _ground_normal(ground)
    adapter = sim.adapter
    weight_n = sim.mass_kg * float(np.linalg.norm(sim.gravity))
    groups = _side_groups(list(adapter._spheres))
    positions, ratios = [], []
    for q_row, v_row in zip(record.q, record.v, strict=True):
        q, v = np.asarray(q_row), np.asarray(v_row)
        positions.append(_sphere_centres(sim, q))
        samples = adapter.evaluate_contact_samples(sim._map(q), sim._map(v))
        ratios.append(_foot_friction_ratio(samples, groups, n, weight_n))
    col = _yaw_index(sim)
    yaw_ref = np.interp(record.time_s, times_track, np.asarray(q_track)[:, col])
    vertical = np.asarray(record.weight_fraction, dtype=float)
    return FinishHistory(
        time_s=np.asarray(record.time_s, dtype=float),
        vertical_bw=vertical,
        friction_ratio=np.asarray(ratios),
        supported=np.asarray(record.inside_support_polygon, dtype=bool)
        & (vertical >= MIN_LOAD_BW),
        feet=_foot_tracks(positions, groups),
        pelvis_yaw_error_rad=np.asarray(record.q)[:, col] - yaw_ref,
    )


def finish_feasibility_report(
    sim: fs.FullBodySimulator,
    *,
    times_track: np.ndarray,
    q_track: np.ndarray,
    record: fs.SimulationRecord,
    zmp: Mapping[str, np.ndarray],
    ground: GroundPlane,
    q_ik: np.ndarray | None = None,
) -> ReportDict:
    """Finish-feasibility block of the dynamics receipt (reference and simulation)."""
    contacts = sim.adapter.contact_parameters
    mu = float(contacts.dynamic_friction)
    ref = reference_history(sim, times_track, q_track, zmp, ground, q_ik=q_ik)
    plant = simulation_history(sim, record, times_track, q_track, ground)
    return {
        "description": (
            "finish-window feasibility of the tracked reference (the load its "
            "inverse dynamics demands) and of the simulation (the load the plant "
            "delivers); reference friction uses the net ground reaction, the "
            "simulation the larger of the two feet; an unloaded frame is never "
            "inside the support polygon"
        ),
        "window_s": [FINISH_START_S, float(times_track[-1])],
        "zmp_window_s": [FINISH_START_S, ZMP_WINDOW_END_S],
        "friction_limit_mu": mu,  # dynamic friction coefficient
        "friction_saturation_fraction": FRICTION_SATURATION,
        "min_load_bw": MIN_LOAD_BW,
        "reference": summarise_history(ref, mu=mu),
        "simulation": summarise_history(plant, mu=mu),
    }


def ratcheted_metrics(block: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """The ratcheted fractions of a full finish-feasibility block, per side.

    A receipt (or companion baseline) block also carries its description,
    windows and non-ratcheted metrics; :func:`regressions` takes only the
    ratcheted ones. Raises ``ValueError`` if a side is missing.
    """
    out: dict[str, dict[str, float]] = {}
    for side in ("reference", "simulation"):
        if side not in block:
            raise ValueError(f"finish-feasibility block lacks {side!r}")
        out[side] = {
            name: float(block[side][name])
            for name in FLOOR_METRICS + CEILING_METRICS
            if name in block[side]
        }
    return out


def regressions(
    block: Mapping[str, Any], baseline: Mapping[str, Mapping[str, float]]
) -> list[str]:
    """Metrics of a finish-feasibility ``block`` that are worse than ``baseline``.

    ``baseline`` maps ``reference`` and ``simulation`` to the recorded value of
    each ratcheted metric: an inside fraction may not fall below it and the
    saturated fraction may not rise above it. Returns one message per violation.
    """
    found: list[str] = []
    for side, recorded in baseline.items():
        if side not in block:
            raise ValueError(f"finish-feasibility block lacks {side!r}")
        for name, floor in recorded.items():
            if name not in FLOOR_METRICS + CEILING_METRICS:
                raise ValueError(f"{name!r} is not a ratcheted metric")
            value = float(block[side][name])
            worse = (
                value < floor - RATCHET_TOLERANCE
                if name in FLOOR_METRICS
                else value > floor + RATCHET_TOLERANCE
            )
            if worse:
                found.append(
                    f"{side}.{name} regressed to {value:.4f} (baseline {floor:.4f})"
                )
    return found


#: Finish-window marker RMS baselines (m) for canonical clubs (#11672, #11667).
FINISH_MARKER_RMS_BASELINES_M: dict[str, float] = {
    "driver": 0.176,
    "iron": 0.113,
    "iron7": 0.113,
}

#: Pre-impact marker RMS baselines (m) for canonical clubs (#11672).
PRE_IMPACT_MARKER_RMS_BASELINES_M: dict[str, float] = {
    "driver": 0.085,
    "iron": 0.090,
    "iron7": 0.090,
}


def evaluate_finish_root_yaw_feasibility(
    times: np.ndarray,
    marker_errors: np.ndarray,
    valid: np.ndarray,
    pelvis_yaw_error_rad: np.ndarray,
    *,
    club: str = "driver",
    finish_start_s: float = FINISH_START_S,
    impact_time_s: float | None = None,
) -> ReportDict:
    """Evaluate finish-window feasibility and root-yaw metrics against baseline (#11672).

    Calculates marker RMS and root-yaw error over the finish window ([finish_start_s, end]),
    checks for improvement over the baseline (driver 176 mm, 7-iron 113 mm), and verifies
    absence of pre-impact regression.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a 1D array of length >= 2")
    errs = np.asarray(marker_errors, dtype=float)
    vld = np.asarray(valid, dtype=bool)
    if errs.ndim != 2 or vld.shape != errs.shape:
        raise ValueError("marker_errors and valid must share a 2D shape")
    if len(t) != len(errs):
        raise ValueError("times and marker_errors row counts must match")

    finish_mask = t >= finish_start_s
    if not finish_mask.any():
        raise ValueError(f"no samples in finish window >= {finish_start_s} s")

    pre_mask = t <= impact_time_s if impact_time_s is not None else t < finish_start_s
    finish_vld = vld[finish_mask]
    finish_err = errs[finish_mask]
    finish_rms = (
        float(np.sqrt(np.mean(finish_err[finish_vld] ** 2)))
        if finish_vld.any()
        else 0.0
    )

    pre_vld = vld[pre_mask]
    pre_err = errs[pre_mask]
    pre_rms = float(np.sqrt(np.mean(pre_err[pre_vld] ** 2))) if pre_vld.any() else 0.0

    yaw_max, yaw_final = pelvis_yaw_error_deg(
        t, pelvis_yaw_error_rad, start_s=finish_start_s
    )
    finish_yaw_rad = np.asarray(pelvis_yaw_error_rad, dtype=float)[finish_mask]
    yaw_rms = (
        float(np.sqrt(np.mean(np.degrees(finish_yaw_rad) ** 2)))
        if finish_yaw_rad.size > 0
        else 0.0
    )

    club_key = club.lower().strip()
    baseline_finish = FINISH_MARKER_RMS_BASELINES_M.get(club_key, 0.176)
    baseline_pre = PRE_IMPACT_MARKER_RMS_BASELINES_M.get(club_key, 0.085)

    improved = bool(finish_rms < baseline_finish)
    no_pre_regression = bool(pre_rms <= baseline_pre + 1e-4)

    return {
        "club": club,
        "finish_window_s": [finish_start_s, float(t[-1])],
        "finish_marker_rms_m": finish_rms,
        "finish_marker_rms_baseline_m": baseline_finish,
        "finish_marker_rms_delta_mm": float((finish_rms - baseline_finish) * 1e3),
        "finish_marker_rms_improved": improved,
        "pre_impact_marker_rms_m": pre_rms,
        "pre_impact_marker_rms_baseline_m": baseline_pre,
        "pre_impact_regression": not no_pre_regression,
        "pelvis_yaw_error_deg_max": yaw_max,
        "pelvis_yaw_error_deg_final": yaw_final,
        "pelvis_yaw_error_deg_rms": yaw_rms,
        "feasible": improved and no_pre_regression,
    }
