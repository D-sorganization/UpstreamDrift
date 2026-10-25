"""Receipt helpers for the predictive Moco deadlift (OSL-7, #13044).

Bundle metrics, bar-path forward-kinematics replay, residual and bar
kinematics extraction, and the comparison table. Every value is computed
from a bundle file or a Moco solution; a missing value is ``None`` plus a
reason, never a default number.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from src.engines.physics_engines.opensim.python.msk_lift_ik_id import (
    RESIDUAL_MOMENT_FRACTION,
)
from src.shared.python.contracts import require
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.motion_matching.same_input import InputBundle

if TYPE_CHECKING:
    from src.engines.physics_engines.opensim.python.msk_lift_predictive import (
        PredictiveDeadliftConfig,
    )

logger = logging.getLogger(__name__)


#: OSL-3 IK/ID receipts; the tracked row's status is read from here, never
#: asserted in code.
OSL3_IK_ID_RECEIPT = (
    Path(__file__).resolve().parents[5]
    / "docs/development/lifting/evidence/opensim_ik_id/receipts.json"
)


#: The LIFT-5 deadlift same-input bundle; its planar optimizer solution sets
#: the predictive setup and lockout poses.
DEADLIFT_BUNDLE = (
    Path(__file__).resolve().parents[5]
    / "docs/development/lifting/evidence/deadlift/deadlift_bundle.npz"
)


def bundle_pose_targets(bundle_path: Path | str = DEADLIFT_BUNDLE) -> dict[str, float]:
    """Setup bar height and setup/lockout trunk inclinations from the bundle.

    The trunk inclination is the optimizer's third segment angle (hip to
    shoulder, radians from vertical, positive forward); the bar height is the
    bundle's starting bar height (``compute_bundle_metrics``).
    """
    sagittal = InputBundle.load(Path(bundle_path)).provenance.get("sagittal_solution")
    require(
        isinstance(sagittal, dict) and "segment_angles_rad" in sagittal,
        f"bundle at {bundle_path} has no sagittal_solution segment_angles_rad",
    )
    angles = cast("dict[str, Any]", sagittal)["segment_angles_rad"]
    trunk = np.asarray(angles, dtype=np.float64)[:, 2]
    return {
        "setup_trunk_rad": float(trunk[0]),
        "lockout_trunk_rad": float(trunk[-1]),
        "setup_bar_height_m": compute_bundle_metrics(bundle_path)["bar_start_height_m"],
    }


def tracked_opensim_record(path: Path = OSL3_IK_ID_RECEIPT) -> dict[str, str]:
    """Status of the tracked OpenSim deadlift row, derived from the OSL-3 receipt.

    The predictive solution is not yet compared with the tracked IK/ID one, so
    the row is never a measured value; its reason states which input exists.
    """
    try:
        lifts = json.loads(path.read_text(encoding="utf-8"))["lifts"]
        ik_id = lifts["deadlift"]["ik_id"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "unavailable", "reason": f"OSL-3 receipt unreadable: {exc}"}
    if ik_id.get("status") != "complete":
        return {
            "status": "unavailable",
            "reason": f"OSL-3 deadlift IK/ID is {ik_id.get('status')!r}: "
            f"{ik_id.get('reason')}",
        }
    return {
        "status": "not compared",
        "reason": "the OSL-3 deadlift IK/ID receipt is complete; the "
        "comparison runs only on a converged predictive solution (see "
        "tracked_comparison)",
    }


def compute_bundle_metrics(bundle_path: Path | str) -> dict[str, float]:
    """Compute kinematic summary metrics directly from deadlift bundle arrays.

    In the bundle coordinate frame, Z is vertical.
    """
    p = Path(bundle_path)
    if not p.is_file():
        raise FileNotFoundError(f"bundle file not found: {p}")

    bundle = InputBundle.load(p)
    ik_path = bundle.provenance.get("ik_bar_path_m")
    sagittal = bundle.provenance.get("sagittal_solution")

    if ik_path is not None and len(ik_path) > 0:
        path = np.asarray(ik_path, dtype=float)
    elif isinstance(sagittal, dict) and "bar_m" in sagittal:
        path = np.asarray(sagittal["bar_m"], dtype=float)
    else:
        raise ValueError(f"bundle at {p} contains no ik_bar_path_m or bar_m")

    if isinstance(sagittal, dict) and "time_s" in sagittal:
        times = np.asarray(sagittal["time_s"], dtype=float)
    else:
        raise ValueError(f"bundle at {p} has no sagittal_solution time_s")

    z = path[:, 2]
    start_h = float(z[0])
    final_h = float(z[-1])
    rise = float(final_h - start_h)
    duration = float(times[-1] - times[0])
    dt = np.diff(times)
    dz = np.diff(z)
    peak_v = float(np.max(dz / dt))

    return {
        "bar_start_height_m": start_h,
        "bar_final_height_m": final_h,
        "bar_vertical_rise_m": rise,
        "peak_vertical_velocity_m_s": peak_v,
        "lift_duration_s": duration,
    }


def replay_bar_path_fk(model: Any, states_table: Any) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics replay of barbell body in ground frame from states table.

    Returns (times, bar_positions) where bar_positions has shape (N, 3).
    In OpenSim ground coordinates: X is forward, Y is vertical, Z is right.
    """
    times, positions, _com_heights = _replay_states(model, states_table)
    return times, positions


def _replay_states(
    model: Any, states_table: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay a states table: times, bar positions (N, 3), whole-model COM heights."""
    state = model.initSystem()
    bar_body = model.getBodySet().get("barbell")
    times = np.array(states_table.getIndependentColumn(), dtype=float)
    n_times = len(times)
    positions = np.zeros((n_times, 3), dtype=float)
    com_heights = np.zeros(n_times, dtype=float)

    cset = model.getCoordinateSet()
    col_indices: dict[str, int] = {}
    for i in range(states_table.getNumColumns()):
        label = states_table.getColumnLabel(i)
        if label.endswith("/value"):
            c_name = label.split("/")[-2]
            if cset.contains(c_name):
                col_indices[c_name] = i

    mat = states_table.getMatrix()
    arr = mat.to_numpy()
    for step in range(n_times):
        state.setTime(times[step])
        for c_name, c_idx in col_indices.items():
            val = float(arr[step, c_idx])
            cset.get(c_name).setValue(state, val, False)
        model.realizePosition(state)
        pos = bar_body.getPositionInGround(state)
        positions[step, 0] = pos.get(0)
        positions[step, 1] = pos.get(1)
        positions[step, 2] = pos.get(2)
        com_heights[step] = model.calcMassCenterPosition(state).get(1)

    return times, positions, com_heights


def _extract_residual_metrics(
    solution: Any, total_weight_n: float, residual_optimal_force_n: float
) -> tuple[float | None, float | None, float | None, bool | None]:
    """Compute peak force, RMS force, weight ratio, and residual-assisted flag.

    Residual forces are control times ``residual_optimal_force_n``.
    """
    try:
        ctable = solution.exportToControlsTable()
        labels = [ctable.getColumnLabel(i) for i in range(ctable.getNumColumns())]
        idx = {
            axis: next(
                (
                    i
                    for i, lab in enumerate(labels)
                    if f"residual_pelvis_t{axis}" in lab
                ),
                None,
            )
            for axis in "xyz"
        }
        mat = ctable.getMatrix().to_numpy()
        if idx["x"] is not None and idx["y"] is not None:
            # Force norm over every translational residual present (tz is
            # absent from solutions predating the six-coordinate residuals).
            cols = [i for i in idx.values() if i is not None]
            f_norm = np.linalg.norm(mat[:, cols] * residual_optimal_force_n, axis=1)
            peak = float(np.max(f_norm))
            rms = float(np.sqrt(np.mean(f_norm**2)))
            ratio = float(rms / total_weight_n)
            assisted = bool(ratio > 0.05)
            return peak, rms, ratio, assisted
    except (RuntimeError, ValueError, IndexError) as exc:
        logger.warning("Could not extract residual forces from solution: %s", exc)
    return None, None, None, None


def residual_moment_metrics(
    solution: Any, residual_optimal_force_n: float
) -> tuple[float | None, float | None]:
    """Peak and RMS (N m) of the rotational pelvis residual moment norm.

    The rotational residuals share the translational optimal force, so a
    control of 1 is ``residual_optimal_force_n`` newton-metres. Returns
    ``(None, None)`` when the solution has no rotational residual columns.
    """
    try:
        ctable = solution.exportToControlsTable()
        labels = [ctable.getColumnLabel(i) for i in range(ctable.getNumColumns())]
        cols = [
            i
            for i, lab in enumerate(labels)
            if any(
                lab.endswith(f"residual_pelvis_{axis}")
                for axis in ("tilt", "list", "rotation")
            )
        ]
        if cols:
            mat = ctable.getMatrix().to_numpy()
            m_norm = np.linalg.norm(mat[:, cols] * residual_optimal_force_n, axis=1)
            return float(np.max(m_norm)), float(np.sqrt(np.mean(m_norm**2)))
    except (RuntimeError, ValueError, IndexError) as exc:
        logger.warning("Could not extract residual moments from solution: %s", exc)
    return None, None


def residual_moment_gate(
    moment_peak_nm: float | None, com_height_m: float | None, total_weight_n: float
) -> dict[str, float | bool | None]:
    """Compare the peak residual moment with the OSL-3 Hicks moment limit.

    The limit is ``RESIDUAL_MOMENT_FRACTION`` of COM height times total
    weight (Hicks et al. 2015), the same fraction as the OSL-3 IK/ID gate.
    Without a COM height or a moment the limit and verdict are ``None``.
    """
    require(total_weight_n > 0.0, f"total_weight_n must be positive: {total_weight_n}")
    if com_height_m is None:
        return {"residual_moment_limit_nm": None, "residual_moment_within_limit": None}
    limit = RESIDUAL_MOMENT_FRACTION * com_height_m * total_weight_n
    within = None if moment_peak_nm is None else bool(moment_peak_nm <= limit)
    return {"residual_moment_limit_nm": limit, "residual_moment_within_limit": within}


def _processed_model(study: Any) -> tuple[Any, Any]:
    """The problem rep and a replayable copy of its processed model.

    The rep holds the processed model (welds applied, muscles replaced); FK
    replays on a copy of it, since the rep itself exposes no initSystem().
    The caller must keep the rep alive while it uses the copy: a temporary
    rep leaves a dangling model.
    """
    osim = import_opensim()
    rep = study.getProblem().createRep()
    return rep, osim.Model(rep.getModelBase())


def _extract_bar_kinematics(study: Any, solution: Any) -> dict[str, float]:
    """Replay FK from states table and return kinematic metrics."""
    rep, model = _processed_model(study)
    times, bar_pos, com_heights = _replay_states(model, solution.exportToStatesTable())
    start_h = float(bar_pos[0, 1])
    final_h = float(bar_pos[-1, 1])
    rise = float(final_h - start_h)
    dur = float(times[-1] - times[0])
    dt = np.diff(times)
    dy = np.diff(bar_pos[:, 1])
    peak_v = float(np.max(dy / dt))
    return {
        "bar_start_height_m": start_h,
        "bar_final_height_m": final_h,
        "bar_vertical_rise_m": rise,
        "peak_vertical_velocity_m_s": peak_v,
        "duration_s": dur,
        "com_height_m": float(np.mean(com_heights)),
    }


def build_comparison_table(
    results: dict[str, Any],
    config: PredictiveDeadliftConfig,
    bundle_path: Path | str | None = None,
) -> dict[str, Any]:
    """Construct comparison table with bundle kinematics and unavailable tracked row.

    Missing predictive quantities must be None with an explicit reason, never a
    hardcoded fallback.
    """
    if bundle_path is None:
        bundle_path = DEADLIFT_BUNDLE

    bundle_metrics: dict[str, float] = {}
    bundle_err: str | None = None
    try:
        bundle_metrics = compute_bundle_metrics(bundle_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        bundle_err = str(exc)

    converged = bool(results.get("converged", False))
    cv = results.get("constraint_violation")
    status = results.get("solver_status", "unknown")

    if not converged:
        infeasible_reason = (
            f"infeasible iterate (constraint violation {cv})"
            if cv is not None and not (isinstance(cv, float) and np.isnan(cv))
            else f"solve did not converge ({status})"
        )
    else:
        infeasible_reason = None

    quantity_keys = (
        ("bar_start_height_m", "m"),
        ("bar_final_height_m", "m"),
        ("bar_vertical_rise_m", "m"),
        ("peak_vertical_velocity_m_s", "m/s"),
        ("lift_duration_s", "s"),
    )

    quantities: dict[str, Any] = {}
    for key, unit in quantity_keys:
        if key in bundle_metrics:
            b_val = bundle_metrics[key]
            b_reason = None
        else:
            b_val = None
            b_reason = bundle_err or "missing from bundle"

        if not converged:
            p_val = None
            p_reason = infeasible_reason
        else:
            if key in results:
                p_val = results[key]
                p_reason = None
            elif key == "lift_duration_s" and "duration_s" in results:
                p_val = results["duration_s"]
                p_reason = None
            else:
                p_val = None
                p_reason = f"metric '{key}' missing from results"

        entry: dict[str, Any] = {
            "bundle_kinematics": b_val,
            "predictive": p_val,
            "unit": unit,
        }
        if b_reason is not None:
            entry["bundle_reason"] = b_reason
        if p_reason is not None:
            entry["reason"] = p_reason
        quantities[key] = entry

    return {
        "bundle_kinematics": {
            "label": "bundle kinematics",
            "source": str(bundle_path),
            "reference_lifter": (
                f"{config.body_mass_kg:g} kg, {config.height_m:g} m, "
                f"{config.bar_mass_kg:g} kg bar"
            ),
        },
        "tracked_opensim": tracked_opensim_record(),
        "tracked_comparison": _tracked_comparison(results, infeasible_reason),
        "quantities": quantities,
    }


def _tracked_comparison(
    results: dict[str, Any], infeasible_reason: str | None
) -> dict[str, Any]:
    """Predictive-versus-tracked comparison, or the reason it did not run."""
    from src.engines.physics_engines.opensim.python import (
        msk_lift_predictive_compare as compare,
    )

    if infeasible_reason is not None:
        return compare.unavailable(infeasible_reason)
    if "trajectories" not in results:
        return compare.unavailable("the solve record has no trajectories")
    return compare.compare_with_tracked(
        results["trajectories"], compare.load_tracked_reference()
    )
