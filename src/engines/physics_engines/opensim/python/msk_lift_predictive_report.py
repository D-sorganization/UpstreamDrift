"""Receipt helpers for the predictive Moco deadlift (OSL-7, #13044).

Bundle metrics, bar-path forward-kinematics replay, residual and bar
kinematics extraction, and the comparison table. Every value is computed
from a bundle file or a Moco solution; a missing value is ``None`` plus a
reason, never a default number.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.motion_matching.same_input import InputBundle

if TYPE_CHECKING:
    from src.engines.physics_engines.opensim.python.msk_lift_predictive import (
        PredictiveDeadliftConfig,
    )

logger = logging.getLogger(__name__)


TRACKED_SOLUTION_UNAVAILABLE_REASON = (
    "unavailable: no OSL-3 deadlift IK/ID receipts on main yet "
    "(OSL-3 #13041 is still open)"
)


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
        times = np.linspace(0.0, 2.0, len(path))

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
    state = model.initSystem()
    bar_body = model.getBodySet().get("barbell")
    times = np.array(states_table.getIndependentColumn(), dtype=float)
    n_times = len(times)
    positions = np.zeros((n_times, 3), dtype=float)

    cset = model.getCoordinateSet()
    col_indices: dict[str, int] = {}
    for i in range(states_table.getNumColumns()):
        label = states_table.getColumnLabel(i)
        if label.endswith("/value"):
            c_name = label.split("/")[-2]
            if cset.has(c_name):
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

    return times, positions


def _extract_residual_metrics(
    solution: Any, total_weight_n: float, residual_optimal_force_n: float
) -> tuple[float | None, float | None, float | None, bool | None]:
    """Compute peak force, RMS force, weight ratio, and residual-assisted flag.

    Residual forces are control times ``residual_optimal_force_n``.
    """
    try:
        ctable = solution.exportToControlsTable()
        labels = [ctable.getColumnLabel(i) for i in range(ctable.getNumColumns())]
        tx_idx = next(
            (i for i, label in enumerate(labels) if "residual_pelvis_tx" in label), None
        )
        ty_idx = next(
            (i for i, label in enumerate(labels) if "residual_pelvis_ty" in label), None
        )
        mat = ctable.getMatrix().to_numpy()
        if tx_idx is not None and ty_idx is not None:
            fx = mat[:, tx_idx] * residual_optimal_force_n
            fy = mat[:, ty_idx] * residual_optimal_force_n
            f_norm = np.sqrt(fx**2 + fy**2)
            peak = float(np.max(f_norm))
            rms = float(np.sqrt(np.mean(f_norm**2)))
            ratio = float(rms / total_weight_n)
            assisted = bool(ratio > 0.05)
            return peak, rms, ratio, assisted
    except (RuntimeError, ValueError, IndexError) as exc:
        logger.warning("Could not extract residual forces from solution: %s", exc)
    return None, None, None, None


def _extract_bar_kinematics(study: Any, solution: Any) -> dict[str, float]:
    """Replay FK from states table and return kinematic metrics."""
    rep = study.getProblem().createRep()
    times, bar_pos = replay_bar_path_fk(rep, solution.exportToStatesTable())
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
        bundle_path = Path(
            "docs/development/lifting/evidence/deadlift/deadlift_bundle.npz"
        )

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
        "tracked_opensim": {
            "status": "unavailable",
            "reason": TRACKED_SOLUTION_UNAVAILABLE_REASON,
        },
        "quantities": quantities,
    }
