"""Evaluate club-trajectory matching quality and torque efficiency.

This script is intentionally diagnostic rather than pass/fail. It can be run
after each surrogate optimization and MATLAB replay to produce a small report
that connects target tracking error, impact-window error, and torque effort.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MODEL_GROUPS = {
    "position": [
        "ClubLogs_CHGlobalPosition_1",
        "ClubLogs_CHGlobalPosition_2",
        "ClubLogs_CHGlobalPosition_3",
    ],
    "velocity": [
        "ClubLogs_CHGlobalVelocity_1",
        "ClubLogs_CHGlobalVelocity_2",
        "ClubLogs_CHGlobalVelocity_3",
    ],
    "acceleration": [
        "ClubLogs_CHGlobalAcceleration_1",
        "ClubLogs_CHGlobalAcceleration_2",
        "ClubLogs_CHGlobalAcceleration_3",
    ],
}

SOURCE_TO_MODEL = {
    "clubface_x": "ClubLogs_CHGlobalPosition_1",
    "clubface_y": "ClubLogs_CHGlobalPosition_2",
    "clubface_z": "ClubLogs_CHGlobalPosition_3",
    "clubface_vx": "ClubLogs_CHGlobalVelocity_1",
    "clubface_vy": "ClubLogs_CHGlobalVelocity_2",
    "clubface_vz": "ClubLogs_CHGlobalVelocity_3",
    "clubface_ax": "ClubLogs_CHGlobalAcceleration_1",
    "clubface_ay": "ClubLogs_CHGlobalAcceleration_2",
    "clubface_az": "ClubLogs_CHGlobalAcceleration_3",
}

TARGET_WEIGHTS = {
    "position": 1.0,
    "velocity": 0.25,
    "acceleration": 0.25,
}
EFFORT_SCALE_EPS = 1.0e-12
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "data" / "processed" / "matching_reports"
)


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty")
    return frame


def _canonical_club_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {"time": frame["time"] if "time" in frame.columns else np.arange(len(frame))}
    )
    for model_column in {
        column for columns in MODEL_GROUPS.values() for column in columns
    }:
        if model_column in frame.columns:
            output[model_column] = frame[model_column]
    for source_column, model_column in SOURCE_TO_MODEL.items():
        if source_column in frame.columns and model_column not in output.columns:
            output[model_column] = frame[source_column]
    return output


def _time_values(frame: pd.DataFrame) -> np.ndarray:
    if "time" not in frame.columns:
        return np.arange(len(frame), dtype=float)
    time = frame["time"].to_numpy(dtype=float)
    if len(time) == 1:
        return np.zeros_like(time, dtype=float)
    if not np.all(np.isfinite(time)):
        raise ValueError("time contains non-finite values")
    return time


def _normalized_time(frame: pd.DataFrame) -> np.ndarray:
    time = _time_values(frame)
    if len(time) == 1:
        return np.zeros_like(time, dtype=float)
    span = float(time[-1] - time[0])
    if abs(span) < EFFORT_SCALE_EPS:
        return np.linspace(0.0, 1.0, len(time))
    return (time - float(time[0])) / span


def _interpolate(
    frame: pd.DataFrame, columns: list[str], query_time: np.ndarray
) -> np.ndarray:
    source_time = _normalized_time(frame)
    values = np.zeros((len(query_time), len(columns)), dtype=float)
    for idx, column in enumerate(columns):
        values[:, idx] = np.interp(
            query_time,
            source_time,
            frame[column].to_numpy(dtype=float),
        )
    return values


def _vector_metrics(
    target_values: np.ndarray,
    predicted_values: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is not None:
        target_values = target_values[mask]
        predicted_values = predicted_values[mask]
    if len(target_values) == 0:
        return {}

    error = predicted_values - target_values
    rmse_axis = np.sqrt(np.mean(error**2, axis=0))
    mae_axis = np.mean(np.abs(error), axis=0)
    max_axis = np.max(np.abs(error), axis=0)
    # ⚡ Bolt: np.sqrt(np.einsum('ij,ij->i', x, x)) fast norm
    vector_error = np.sqrt(np.einsum("ij,ij->i", error, error))
    target_span = np.ptp(target_values, axis=0)
    denom = float(np.linalg.norm(target_span))
    if denom < EFFORT_SCALE_EPS:
        denom = float(
            # ⚡ Bolt: np.sqrt(np.einsum('ij,ij->i', x, x)) fast norm
            np.mean(np.sqrt(np.einsum("ij,ij->i", target_values, target_values)))
        )
    if denom < EFFORT_SCALE_EPS:
        denom = 1.0

    return {
        "samples": int(len(target_values)),
        "rmse_axis": rmse_axis.tolist(),
        "mae_axis": mae_axis.tolist(),
        "max_abs_axis": max_axis.tolist(),
        "vector_rmse": float(np.sqrt(np.mean(vector_error**2))),
        "vector_mae": float(np.mean(vector_error)),
        "vector_max_abs": float(np.max(vector_error)),
        "normalized_vector_rmse": float(np.sqrt(np.mean(vector_error**2)) / denom),
        "normalizer": denom,
    }


def _impact_mask(
    time: np.ndarray, impact_time: float | None, impact_window_s: float
) -> np.ndarray:
    if len(time) == 0:
        return np.zeros(0, dtype=bool)
    center = float(time[-1] if impact_time is None else impact_time)
    half_width = max(float(impact_window_s), 0.0) / 2.0
    if half_width <= EFFORT_SCALE_EPS:
        index = int(np.argmin(np.abs(time - center)))
        mask = np.zeros(len(time), dtype=bool)
        mask[index] = True
        return mask
    return np.abs(time - center) <= half_width


def _matching_metrics(
    target: pd.DataFrame,
    simulated: pd.DataFrame,
    impact_time: float | None,
    impact_window_s: float,
) -> dict[str, Any]:
    query_time = _normalized_time(target)
    raw_time = _time_values(target)
    mask = _impact_mask(raw_time, impact_time, impact_window_s)
    metrics: dict[str, Any] = {
        "target_rows": int(len(target)),
        "simulated_rows": int(len(simulated)),
        "impact_time": float(raw_time[-1] if impact_time is None else impact_time),
        "impact_window_s": float(impact_window_s),
        "groups": {},
        "impact_window": {},
    }

    for group, columns in MODEL_GROUPS.items():
        available = [
            column
            for column in columns
            if column in target.columns and column in simulated.columns
        ]
        if not available:
            continue
        target_values = target[available].to_numpy(dtype=float)
        sim_values = _interpolate(simulated, available, query_time)
        metrics["groups"][group] = {
            "columns": available,
            **_vector_metrics(target_values, sim_values),
        }
        metrics["impact_window"][group] = {
            "columns": available,
            **_vector_metrics(target_values, sim_values, mask),
        }

    return metrics


def _time_step_seconds(time: np.ndarray) -> np.ndarray:
    if len(time) <= 1:
        return np.ones(max(len(time), 1), dtype=float)
    diffs = np.diff(time)
    finite = diffs[np.isfinite(diffs) & (np.abs(diffs) > EFFORT_SCALE_EPS)]
    fallback = float(np.median(np.abs(finite))) if len(finite) else 1.0
    return np.concatenate([np.abs(diffs), [fallback]])


def _torque_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = [
        column
        for column in frame.columns
        if column != "time" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    preferred = [
        column
        for column in numeric_columns
        if "Torque" in column or "Force" in column or "Input" in column
    ]
    return preferred or numeric_columns


def _joint_velocity_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = [
        column
        for column in frame.columns
        if column != "time" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    preferred = [
        column
        for column in numeric_columns
        if "Velocity" in column
        or "velocity" in column
        or "qdot" in column
        or "QDot" in column
    ]
    return preferred or numeric_columns


def _column_tokens(column: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for char in column:
        if char.isalnum():
            current += char.lower()
        elif current:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)
    return tokens


def _normalized_column_name(column: str) -> str:
    return "".join(_column_tokens(column))


def _signal_stem(column: str) -> str:
    normalized = _normalized_column_name(column)
    for token in ("angularvelocity", "velocity", "actuatortorque", "torque"):
        normalized = normalized.replace(token, "")
    return normalized


def _axis_suffix(column: str) -> str | None:
    tokens = _column_tokens(column)
    if not tokens:
        return None
    last = tokens[-1]
    if last in {"x", "y", "z"} or (len(last) == 1 and last.isdigit()):
        return last
    for suffix in ("x", "y", "z"):
        if last.endswith(suffix):
            return suffix
    return None


def pair_torque_velocity_columns(
    torque_columns: list[str], velocity_columns: list[str]
) -> dict[str, str]:
    """Pair torque columns to joint-velocity columns with explicit heuristics."""
    available = list(velocity_columns)
    normalized_velocity = {
        _normalized_column_name(column): column for column in available
    }
    stem_velocity = {_signal_stem(column): column for column in available}
    pairs: dict[str, str] = {}

    for torque_column in torque_columns:
        candidates = [
            torque_column,
            torque_column.replace("Torque", "Velocity"),
            torque_column.replace("torque", "velocity"),
            torque_column.replace("Torque", "AngularVelocity"),
            torque_column.replace("torque", "angularvelocity"),
        ]
        matched = next(
            (
                normalized_velocity[_normalized_column_name(candidate)]
                for candidate in candidates
                if _normalized_column_name(candidate) in normalized_velocity
            ),
            None,
        )
        if matched is None:
            matched = stem_velocity.get(_signal_stem(torque_column))
        if matched is None:
            torque_axis = _axis_suffix(torque_column)
            if torque_axis is not None:
                axis_matches = [
                    column
                    for column in available
                    if _axis_suffix(column) == torque_axis
                ]
                if len(axis_matches) == 1:
                    matched = axis_matches[0]
        if matched is not None:
            pairs[torque_column] = matched
            available.remove(matched)
            normalized_velocity = {
                _normalized_column_name(column): column for column in available
            }
            stem_velocity = {_signal_stem(column): column for column in available}

    return pairs


def _trapz(values: np.ndarray, time: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, time))
    return float(np.trapz(values, time))


def compute_mechanical_work(
    torque_frame: pd.DataFrame, velocity_frame: pd.DataFrame | None
) -> dict[str, Any]:
    torque_columns = _torque_columns(torque_frame)
    if not torque_columns:
        return {"available": False, "reason": "no numeric torque columns found"}
    if velocity_frame is None:
        return {
            "available": False,
            "reason": "joint velocity data was not provided",
        }

    velocity_columns = _joint_velocity_columns(velocity_frame)
    if not velocity_columns:
        return {
            "available": False,
            "reason": "no numeric joint velocity columns found",
        }

    pairs = pair_torque_velocity_columns(torque_columns, velocity_columns)
    if not pairs:
        return {
            "available": False,
            "reason": "no torque columns could be mapped to joint velocity columns",
            "torque_columns": torque_columns,
            "velocity_columns": velocity_columns,
        }

    time = _time_values(torque_frame)
    query_time = _normalized_time(torque_frame)
    torque_values = torque_frame[list(pairs)].to_numpy(dtype=float)
    velocity_values = _interpolate(velocity_frame, list(pairs.values()), query_time)
    joint_power = torque_values * velocity_values
    positive_power = np.maximum(joint_power, 0.0)
    negative_power_abs = np.maximum(-joint_power, 0.0)
    net_power = np.sum(joint_power, axis=1)

    per_joint: list[dict[str, Any]] = []
    for idx, (torque_column, velocity_column) in enumerate(pairs.items()):
        positive_work = _trapz(positive_power[:, idx], time)
        negative_work_abs = _trapz(negative_power_abs[:, idx], time)
        net_work = _trapz(joint_power[:, idx], time)
        per_joint.append(
            {
                "torque_column": torque_column,
                "velocity_column": velocity_column,
                "positive_mechanical_work": positive_work,
                "negative_mechanical_work_abs": negative_work_abs,
                "net_mechanical_work": net_work,
            }
        )
    per_joint.sort(
        key=lambda item: float(item["positive_mechanical_work"]), reverse=True
    )

    return {
        "available": True,
        "rows": int(len(torque_frame)),
        "paired_columns": pairs,
        "unmapped_torque_columns": [
            column for column in torque_columns if column not in pairs
        ],
        "positive_mechanical_work": _trapz(np.sum(positive_power, axis=1), time),
        "negative_mechanical_work_abs": _trapz(
            np.sum(negative_power_abs, axis=1), time
        ),
        "net_mechanical_work": _trapz(net_power, time),
        "per_joint_ranking": per_joint,
    }


def _effort_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    columns = _torque_columns(frame)
    if not columns:
        return {"available": False, "reason": "no numeric torque columns found"}

    values = frame[columns].to_numpy(dtype=float)
    time = _time_values(frame)
    dt = _time_step_seconds(time).reshape(-1, 1)
    diff_dt = max(float(np.median(dt)), EFFORT_SCALE_EPS)
    derivatives = np.diff(values, axis=0) / diff_dt if len(values) > 1 else values * 0.0
    smoothness = float(np.vdot(derivatives, derivatives) * diff_dt)
    l2_effort = float(np.vdot(values, values * dt))
    l1_impulse = float(np.sum(np.abs(values) * dt))

    return {
        "available": True,
        "columns": columns,
        "rows": int(len(frame)),
        "l2_torque_effort": l2_effort,
        "l1_torque_impulse": l1_impulse,
        "peak_abs_torque": float(np.max(np.abs(values))),
        "mean_abs_torque": float(np.mean(np.abs(values))),
        "smoothness_l2": smoothness,
        "note": (
            "Mechanical work requires joint velocities paired with torques. "
            "This report uses torque impulse, squared torque, and torque-rate "
            "smoothness when qdot is not provided."
        ),
    }


def _weighted_tracking_score(
    matching: dict[str, Any], weights: dict[str, float]
) -> float | None:
    groups = matching.get("groups", {})
    score = 0.0
    used = False
    for group, weight in weights.items():
        value = groups.get(group, {}).get("normalized_vector_rmse")
        if value is None:
            continue
        used = True
        score += float(weight) * float(value) ** 2
    return score if used else None


def _weighted_objective(
    matching: dict[str, Any],
    effort: dict[str, Any] | None,
    mechanical_work: dict[str, Any] | None,
    effort_weight: float,
    smoothness_weight: float,
) -> dict[str, Any]:
    tracking = _weighted_tracking_score(matching, TARGET_WEIGHTS)
    objective = 0.0 if tracking is None else tracking
    terms: dict[str, Any] = {"tracking": tracking}
    if effort and effort.get("available"):
        smoothness = float(effort["smoothness_l2"])
        terms["smoothness"] = smoothness_weight * smoothness
        objective += float(terms["smoothness"])
        if mechanical_work and mechanical_work.get("available"):
            effort_value = float(mechanical_work["positive_mechanical_work"])
            terms["effort_source"] = "positive_mechanical_work"
        else:
            effort_value = float(effort["l2_torque_effort"])
            terms["effort_source"] = "l2_torque_effort_proxy"
        terms["effort"] = effort_weight * effort_value
        objective += float(terms["effort"])
    return {
        "value": objective,
        "terms": terms,
        "weights": {
            "tracking": TARGET_WEIGHTS,
            "effort": effort_weight,
            "smoothness": smoothness_weight,
        },
    }


def _plot_matching(
    target: pd.DataFrame,
    simulated: pd.DataFrame,
    output_png: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    query_time = _normalized_time(target)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis_index, (group, columns) in enumerate(MODEL_GROUPS.items()):
        available = [
            column
            for column in columns
            if column in target.columns and column in simulated.columns
        ]
        if not available:
            axes[axis_index].set_visible(False)
            continue
        sim_values = _interpolate(simulated, available, query_time)
        target_values = target[available].to_numpy(dtype=float)
        residual = sim_values - target_values
        for idx, column in enumerate(available):
            axes[axis_index].plot(query_time, residual[:, idx], label=column[-1])
        axes[axis_index].axhline(0.0, color="black", linewidth=0.8)
        axes[axis_index].set_ylabel(f"{group} error")
        axes[axis_index].legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("normalized time")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return True


def _plot_torques(frame: pd.DataFrame, output_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    columns = _torque_columns(frame)
    if not columns:
        return False
    time = _time_values(frame)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    frame[columns].plot(ax=axes[0], legend=False)
    axes[0].set_ylabel("torque/force")
    axes[0].set_title("Optimized controls")
    axes[1].bar(columns, np.max(np.abs(frame[columns].to_numpy(float)), axis=0))
    axes[1].set_ylabel("peak abs")
    axes[1].tick_params(axis="x", labelrotation=90)
    axes[0].set_xlabel(f"time index, start={time[0]:.6g}")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return True


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Golf ML Matching Diagnostics",
        "",
        f"- Scenario: `{report['scenario']}`",
        f"- Run label: `{report['run_label']}`",
        f"- Weighted objective: `{report['objective']['value']:.8g}`",
    ]
    matching = report.get("matching")
    if matching:
        lines.append("")
        lines.append("## Tracking Error")
        for group, metrics in matching["groups"].items():
            lines.append(
                "- "
                f"{group}: normalized RMSE "
                f"`{metrics['normalized_vector_rmse']:.8g}`, "
                f"vector RMSE `{metrics['vector_rmse']:.8g}`"
            )
        lines.append("")
        lines.append("## Impact Window")
        for group, metrics in matching["impact_window"].items():
            if metrics:
                lines.append(
                    "- "
                    f"{group}: samples `{metrics['samples']}`, "
                    f"normalized RMSE `{metrics['normalized_vector_rmse']:.8g}`"
                )
    effort = report.get("effort")
    if effort and effort.get("available"):
        lines.extend(
            [
                "",
                "## Torque Effort",
                f"- L2 torque effort: `{effort['l2_torque_effort']:.8g}`",
                f"- L1 torque impulse: `{effort['l1_torque_impulse']:.8g}`",
                f"- Peak absolute torque/force: `{effort['peak_abs_torque']:.8g}`",
                f"- Torque smoothness L2: `{effort['smoothness_l2']:.8g}`",
            ]
        )
    mechanical_work = report.get("mechanical_work")
    if mechanical_work and mechanical_work.get("available"):
        lines.extend(
            [
                "",
                "## Mechanical Work",
                "- Positive mechanical work: "
                f"`{mechanical_work['positive_mechanical_work']:.8g}`",
                "- Negative mechanical work abs: "
                f"`{mechanical_work['negative_mechanical_work_abs']:.8g}`",
                "- Net mechanical work: "
                f"`{mechanical_work['net_mechanical_work']:.8g}`",
            ]
        )
    elif mechanical_work:
        lines.extend(
            [
                "",
                "## Mechanical Work",
                f"- Unavailable: {mechanical_work['reason']}",
            ]
        )
    return "\n".join(lines) + "\n"


def evaluate(
    target_csv: Path | None,
    sim_csv: Path | None,
    torque_csv: Path | None,
    output_dir: Path,
    scenario: str,
    run_label: str,
    impact_time: float | None,
    impact_window_s: float,
    effort_weight: float,
    smoothness_weight: float,
    body_state_csv: Path | None = None,
    joint_velocity_csv: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matching: dict[str, Any] | None = None
    plots: dict[str, str] = {}

    target = _canonical_club_frame(_load_frame(target_csv)) if target_csv else None
    simulated = _canonical_club_frame(_load_frame(sim_csv)) if sim_csv else None
    if target is not None and simulated is not None:
        matching = _matching_metrics(target, simulated, impact_time, impact_window_s)
        plot_path = output_dir / f"{run_label}_tracking_residuals.png"
        if _plot_matching(target, simulated, plot_path):
            plots["tracking_residuals"] = str(plot_path)

    effort = None
    mechanical_work = None
    if torque_csv is not None:
        torques = _load_frame(torque_csv)
        effort = _effort_metrics(torques)
        velocity_csv = joint_velocity_csv or body_state_csv
        velocities = _load_frame(velocity_csv) if velocity_csv else None
        mechanical_work = compute_mechanical_work(torques, velocities)
        if velocity_csv:
            mechanical_work["velocity_csv"] = str(velocity_csv)
        plot_path = output_dir / f"{run_label}_torque_effort.png"
        if _plot_torques(torques, plot_path):
            plots["torque_effort"] = str(plot_path)

    report = {
        "scenario": scenario,
        "run_label": run_label,
        "target_csv": str(target_csv) if target_csv else None,
        "sim_csv": str(sim_csv) if sim_csv else None,
        "torque_csv": str(torque_csv) if torque_csv else None,
        "body_state_csv": str(body_state_csv) if body_state_csv else None,
        "joint_velocity_csv": str(joint_velocity_csv) if joint_velocity_csv else None,
        "matching": matching,
        "effort": effort,
        "mechanical_work": mechanical_work,
        "objective": _weighted_objective(
            matching or {},
            effort,
            mechanical_work,
            effort_weight=effort_weight,
            smoothness_weight=smoothness_weight,
        ),
        "plots": plots,
    }
    metrics_path = output_dir / f"{run_label}_matching_metrics.json"
    summary_path = output_dir / f"{run_label}_matching_summary.md"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path.write_text(_summary_markdown(report), encoding="utf-8")
    # Also emit the canonical Metrics record (METRICS_SCHEMA.md) so the
    # leaderboard / cross-language consumers can rank this fit alongside
    # MATLAB-emitted records.  Best-effort: missing fields fall back to
    # safe defaults rather than failing the diagnostic run.
    canonical_path = output_dir / f"{run_label}_metrics_canonical.json"
    try:
        canonical_path.write_text(
            _emit_canonical_metrics(report, run_label), encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover - best effort  # noqa: BLE001
        LOGGER.warning("Skipped canonical Metrics emission: %s", exc)
    LOGGER.info("Wrote matching diagnostics to %s", output_dir)
    return report


def _emit_canonical_metrics(report: dict, run_label: str) -> str:
    """Convert a diagnostic report dict to a canonical Metrics JSON string."""
    import platform
    import subprocess
    from datetime import datetime

    from src.shared.python.motion_matching.metrics import (
        SCHEMA_VERSION,
        Metrics,
    )

    matching = report.get("matching") or {}
    effort = report.get("effort") or {}
    mech = report.get("mechanical_work") or {}

    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        sha = "0" * 40
    if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
        sha = "0" * 40

    m = Metrics(
        swing_id=run_label,
        option=1,
        solver="evaluate_matching_workflow",
        n_iterations=0,
        rmse_clubhead_mm=float(matching.get("rmse_position_mm", 0.0)),
        rmse_butt_mm=float(matching.get("rmse_butt_mm", 0.0)),
        rmse_orientation_deg=float(matching.get("rmse_orientation_deg", 0.0)),
        clubhead_speed_at_impact_mph=float(matching.get("clubhead_speed_sim_mph", 0.0)),
        clubhead_speed_meas_mph=float(matching.get("clubhead_speed_meas_mph", 0.0)),
        total_work_J=float(mech.get("total_work_J", 0.0)) if mech else 0.0,
        peak_power_W=float(effort.get("peak_power_W", 0.0)),
        wall_clock_s=float(report.get("wall_clock_s", 0.0)),
        git_commit=sha,
        matlab_version="",
        python_version=platform.python_version(),
        timestamp_iso8601=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        schema_version=SCHEMA_VERSION,
    )
    return m.to_json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path)
    parser.add_argument("--sim-csv", type=Path)
    parser.add_argument("--torque-csv", type=Path)
    parser.add_argument("--body-state-csv", type=Path)
    parser.add_argument("--joint-velocity-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", default="downswing")
    parser.add_argument("--run-label", default="latest")
    parser.add_argument("--impact-time", type=float)
    parser.add_argument("--impact-window-s", type=float, default=0.02)
    parser.add_argument("--effort-weight", type=float, default=1.0e-8)
    parser.add_argument("--smoothness-weight", type=float, default=1.0e-10)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    evaluate(
        target_csv=args.target_csv,
        sim_csv=args.sim_csv,
        torque_csv=args.torque_csv,
        body_state_csv=args.body_state_csv,
        joint_velocity_csv=args.joint_velocity_csv,
        output_dir=args.output_dir,
        scenario=args.scenario,
        run_label=args.run_label,
        impact_time=args.impact_time,
        impact_window_s=args.impact_window_s,
        effort_weight=args.effort_weight,
        smoothness_weight=args.smoothness_weight,
    )


if __name__ == "__main__":
    main()
