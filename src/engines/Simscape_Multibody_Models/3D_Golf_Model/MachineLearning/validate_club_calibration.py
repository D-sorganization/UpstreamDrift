"""Validate calibrated golf-club targets against Simscape club logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "processed" / "calibration_validation"
EPSILON = 1.0e-12

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
    if abs(span) < EPSILON:
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


def _impact_mask(
    time: np.ndarray, impact_time: float | None, impact_window_s: float
) -> np.ndarray:
    if len(time) == 0:
        return np.zeros(0, dtype=bool)
    center = float(time[-1] if impact_time is None else impact_time)
    half_width = max(float(impact_window_s), 0.0) / 2.0
    if half_width <= EPSILON:
        mask = np.zeros(len(time), dtype=bool)
        mask[int(np.argmin(np.abs(time - center)))] = True
        return mask
    return np.abs(time - center) <= half_width


def _vector_metrics(
    target_values: np.ndarray,
    simulated_values: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mask is not None:
        target_values = target_values[mask]
        simulated_values = simulated_values[mask]
    if len(target_values) == 0:
        return {"samples": 0}

    finite = np.all(np.isfinite(target_values), axis=1) & np.all(
        np.isfinite(simulated_values), axis=1
    )
    target_values = target_values[finite]
    simulated_values = simulated_values[finite]
    if len(target_values) == 0:
        return {"samples": 0}

    residual = simulated_values - target_values
    rmse_axis = np.sqrt(np.mean(residual**2, axis=0))
    # ⚡ Bolt: np.sqrt(np.einsum('ij,ij->i', x, x)) fast norm
    vector_error = np.sqrt(np.einsum("ij,ij->i", residual, residual))
    target_span = np.ptp(target_values, axis=0)
    normalizer = float(math.sqrt(np.vdot(target_span, target_span)))  # ⚡ Bolt: math.sqrt(np.vdot) is faster than np.linalg.norm for small 1D arrays
    if normalizer < EPSILON:
        normalizer = float(
            # ⚡ Bolt: np.sqrt(np.einsum('ij,ij->i', x, x)) fast norm
            np.mean(np.sqrt(np.einsum("ij,ij->i", target_values, target_values)))
        )
    if normalizer < EPSILON:
        normalizer = 1.0

    anisotropy = float(np.max(rmse_axis) / max(float(np.min(rmse_axis)), EPSILON))
    return {
        "samples": int(len(target_values)),
        "rmse_axis": rmse_axis.tolist(),
        "mae_axis": np.mean(np.abs(residual), axis=0).tolist(),
        "max_abs_axis": np.max(np.abs(residual), axis=0).tolist(),
        "vector_rmse": float(np.sqrt(np.mean(vector_error**2))),
        "vector_mae": float(np.mean(vector_error)),
        "vector_max_abs": float(np.max(vector_error)),
        "normalized_vector_rmse": float(np.sqrt(np.mean(vector_error**2)) / normalizer),
        "normalizer": normalizer,
        "residual_anisotropy": anisotropy,
    }


def _residual_metrics(
    target: pd.DataFrame,
    simulated: pd.DataFrame,
    impact_time: float | None,
    impact_window_s: float,
) -> dict[str, Any]:
    query_time = _normalized_time(target)
    raw_time = _time_values(target)
    impact = _impact_mask(raw_time, impact_time, impact_window_s)
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
            **_vector_metrics(target_values, sim_values, impact),
        }
    return metrics


def _first_nested(payload: Any, names: tuple[str, ...]) -> Any:
    if isinstance(payload, dict):
        for name in names:
            if name in payload:
                return payload[name]
        for value in payload.values():
            found = _first_nested(value, names)
            if found is not None:
                return found
    return None


def _transform_summary(transform_json: Path | None) -> dict[str, Any] | None:
    if transform_json is None:
        return None
    payload = json.loads(transform_json.read_text(encoding="utf-8"))
    matrix_raw = _first_nested(payload, ("matrix", "transform_matrix"))
    rotation_raw = _first_nested(payload, ("rotation", "rotation_matrix"))
    translation_raw = _first_nested(payload, ("translation", "offset"))
    scale_raw = _first_nested(payload, ("scale", "scale_estimate"))

    matrix = None
    if matrix_raw is not None:
        matrix = np.asarray(matrix_raw, dtype=float)
    elif rotation_raw is not None:
        matrix = np.asarray(rotation_raw, dtype=float)

    determinant = None
    orientation_sign = None
    matrix_shape = None
    if matrix is not None:
        matrix_shape = list(matrix.shape)
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            determinant = float(np.linalg.det(matrix))
            orientation_sign = "positive" if determinant >= 0.0 else "negative"

    scale_estimate = None
    if scale_raw is not None:
        scale_estimate = float(scale_raw)
    elif matrix is not None and matrix.ndim == 2:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if len(singular_values):
            scale_estimate = float(np.mean(np.abs(singular_values)))

    return {
        "path": str(transform_json),
        "matrix_shape": matrix_shape,
        "determinant": determinant,
        "orientation_sign": orientation_sign,
        "scale_estimate": scale_estimate,
        "translation": (
            np.asarray(translation_raw, dtype=float).reshape(-1).tolist()
            if translation_raw is not None
            else None
        ),
    }


def _warning_flags(  # noqa: C901
    before: dict[str, Any],
    after: dict[str, Any],
    transform: dict[str, Any] | None,
    min_finite_samples: int,
    poor_impact_threshold: float,
    anisotropy_threshold: float,
    extreme_scale_min: float,
    extreme_scale_max: float,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if transform is not None:
        determinant = transform.get("determinant")
        if determinant is not None and determinant < 0.0:
            warnings.append(
                {
                    "code": "mirror_flip",
                    "message": (
                        "Transform determinant is negative; calibration mirrors "
                        "handedness."
                    ),
                    "value": determinant,
                }
            )
        scale = transform.get("scale_estimate")
        if scale is not None and (
            float(scale) < extreme_scale_min or float(scale) > extreme_scale_max
        ):
            warnings.append(
                {
                    "code": "extreme_scale",
                    "message": (
                        "Transform scale estimate is outside the expected range."
                    ),
                    "value": scale,
                }
            )

    for label, metrics in (("before", before), ("after", after)):
        for section in ("groups", "impact_window"):
            for group, group_metrics in metrics.get(section, {}).items():
                samples = int(group_metrics.get("samples", 0))
                if samples < min_finite_samples:
                    warnings.append(
                        {
                            "code": "too_few_finite_samples",
                            "message": (
                                f"{label} {section} {group} has only {samples} "
                                "finite samples."
                            ),
                            "samples": samples,
                        }
                    )

    after_position_impact = after.get("impact_window", {}).get("position", {})
    impact_error = after_position_impact.get("normalized_vector_rmse")
    if impact_error is not None and impact_error > poor_impact_threshold:
        warnings.append(
            {
                "code": "poor_impact_window_fit",
                "message": (
                    "Calibrated target still has high impact-window position error."
                ),
                "value": impact_error,
                "threshold": poor_impact_threshold,
            }
        )

    for label, metrics in (("before", before), ("after", after)):
        for section in ("groups", "impact_window"):
            for group, group_metrics in metrics.get(section, {}).items():
                anisotropy = group_metrics.get("residual_anisotropy")
                if anisotropy is not None and anisotropy > anisotropy_threshold:
                    warnings.append(
                        {
                            "code": "residual_anisotropy",
                            "message": (
                                f"{label} {section} {group} residuals are highly "
                                "axis-skewed."
                            ),
                            "value": anisotropy,
                            "threshold": anisotropy_threshold,
                        }
                    )
    return warnings


def _plot_overlay_3d(
    measured: pd.DataFrame,
    calibrated: pd.DataFrame,
    simulated: pd.DataFrame,
    output_png: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    columns = MODEL_GROUPS["position"]
    if not all(
        column in measured.columns and column in calibrated.columns
        for column in columns
    ) or not all(column in simulated.columns for column in columns):
        return False
    query_time = _normalized_time(measured)
    sim_values = _interpolate(simulated, columns, query_time)
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    for frame, label, style in (
        (measured, "measured", "-"),
        (calibrated, "calibrated", "-"),
    ):
        values = frame[columns].to_numpy(dtype=float)
        axis.plot(values[:, 0], values[:, 1], values[:, 2], style, label=label)
    axis.plot(sim_values[:, 0], sim_values[:, 1], sim_values[:, 2], "--", label="sim")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")  # type: ignore[attr-defined]
    axis.legend(loc="best")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return True


def _plot_residuals(
    calibrated: pd.DataFrame,
    simulated: pd.DataFrame,
    output_png: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    columns = [
        column
        for column in MODEL_GROUPS["position"]
        if column in calibrated.columns and column in simulated.columns
    ]
    if not columns:
        return False
    query_time = _normalized_time(calibrated)
    raw_time = _time_values(calibrated)
    residual = _interpolate(simulated, columns, query_time) - calibrated[
        columns
    ].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(10, 5))
    for idx, column in enumerate(columns):
        axis.plot(raw_time, residual[:, idx], label=column[-1])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("time")
    axis.set_ylabel("sim - calibrated residual")
    axis.legend(loc="best")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return True


def _plot_speed_acceleration(
    measured: pd.DataFrame,
    calibrated: pd.DataFrame,
    simulated: pd.DataFrame,
    output_png: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    groups = (("velocity", "speed"), ("acceleration", "acceleration magnitude"))
    query_time = _normalized_time(measured)
    raw_time = _time_values(measured)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plotted = False
    for axis, (group, label) in zip(axes, groups, strict=True):
        columns = MODEL_GROUPS[group]
        if not all(column in simulated.columns for column in columns):
            continue
        val = _interpolate(simulated, columns, query_time)
        # ⚡ Bolt: np.sqrt(np.einsum('ij,ij->i', x, x)) fast norm
        sim_norm = np.sqrt(np.einsum("ij,ij->i", val, val))
        axis.plot(raw_time, sim_norm, "--", label="sim")
        for frame, frame_label in (
            (measured, "measured"),
            (calibrated, "calibrated"),
        ):
            if all(column in frame.columns for column in columns):
                arr = frame[columns].to_numpy(dtype=float)
                # ⚡ Bolt: np.sqrt(np.einsum('ij,ij->i', x, x)) fast norm
                values = np.sqrt(np.einsum("ij,ij->i", arr, arr))
                axis.plot(raw_time, values, label=frame_label)
                plotted = True
        axis.set_ylabel(label)
        axis.legend(loc="best")
    if not plotted:
        plt.close(fig)
        return False
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return True


def _write_plots(
    measured: pd.DataFrame,
    calibrated: pd.DataFrame,
    simulated: pd.DataFrame,
    output_dir: Path,
    run_label: str,
) -> dict[str, str]:
    plots: dict[str, str] = {}
    overlay_png = output_dir / f"{run_label}_overlay_3d.png"
    if _plot_overlay_3d(measured, calibrated, simulated, overlay_png):
        plots["overlay_3d"] = str(overlay_png)

    residual_png = output_dir / f"{run_label}_position_residuals.png"
    if _plot_residuals(calibrated, simulated, residual_png):
        plots["position_residuals"] = str(residual_png)

    speed_png = output_dir / f"{run_label}_speed_acceleration.png"
    if _plot_speed_acceleration(measured, calibrated, simulated, speed_png):
        plots["speed_acceleration"] = str(speed_png)
    return plots


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = ["# Golf ML Club Calibration Validation", ""]
    transform = report.get("transform")
    if transform:
        lines.extend(
            [
                "## Transform",
                f"- Matrix shape: `{transform.get('matrix_shape')}`",
                f"- Determinant: `{transform.get('determinant')}`",
                f"- Orientation sign: `{transform.get('orientation_sign')}`",
                f"- Scale estimate: `{transform.get('scale_estimate')}`",
                f"- Translation: `{transform.get('translation')}`",
                "",
            ]
        )
    lines.append("## Residuals")
    for phase in ("before", "after"):
        phase_metrics = report["residuals"][phase]
        lines.append(f"### {phase.title()} Calibration")
        for section in ("groups", "impact_window"):
            lines.append(f"- {section}:")
            for group, metrics in phase_metrics.get(section, {}).items():
                norm = metrics.get("normalized_vector_rmse")
                vector = metrics.get("vector_rmse")
                samples = metrics.get("samples")
                lines.append(
                    f"  - {group}: samples `{samples}`, vector RMSE `{vector}`, "
                    f"normalized RMSE `{norm}`"
                )
        lines.append("")
    warnings = report.get("warnings", [])
    lines.append("## Warnings")
    if warnings:
        for warning in warnings:
            lines.append(f"- `{warning['code']}`: {warning['message']}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def validate(
    measured_target_csv: Path,
    calibrated_target_csv: Path,
    sim_csv: Path,
    output_dir: Path,
    run_label: str = "club_calibration",
    transform_json: Path | None = None,
    impact_time: float | None = None,
    impact_window_s: float = 0.02,
    write_plots: bool = True,
    min_finite_samples: int = 3,
    poor_impact_threshold: float = 0.05,
    anisotropy_threshold: float = 10.0,
    extreme_scale_min: float = 0.2,
    extreme_scale_max: float = 5.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measured = _canonical_club_frame(_load_frame(measured_target_csv))
    calibrated = _canonical_club_frame(_load_frame(calibrated_target_csv))
    simulated = _canonical_club_frame(_load_frame(sim_csv))
    before = _residual_metrics(measured, simulated, impact_time, impact_window_s)
    after = _residual_metrics(calibrated, simulated, impact_time, impact_window_s)
    transform = _transform_summary(transform_json)
    warnings = _warning_flags(
        before=before,
        after=after,
        transform=transform,
        min_finite_samples=min_finite_samples,
        poor_impact_threshold=poor_impact_threshold,
        anisotropy_threshold=anisotropy_threshold,
        extreme_scale_min=extreme_scale_min,
        extreme_scale_max=extreme_scale_max,
    )
    plots = (
        _write_plots(measured, calibrated, simulated, output_dir, run_label)
        if write_plots
        else {}
    )
    report = {
        "measured_target_csv": str(measured_target_csv),
        "calibrated_target_csv": str(calibrated_target_csv),
        "sim_csv": str(sim_csv),
        "residuals": {
            "before": before,
            "after": after,
        },
        "transform": transform,
        "warnings": warnings,
        "plots": plots,
    }
    metrics_path = output_dir / f"{run_label}_calibration_validation.json"
    summary_path = output_dir / f"{run_label}_calibration_validation.md"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path.write_text(_summary_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-target-csv", type=Path, required=True)
    parser.add_argument("--calibrated-target-csv", type=Path, required=True)
    parser.add_argument("--sim-csv", type=Path, required=True)
    parser.add_argument("--transform-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-label", default="club_calibration")
    parser.add_argument("--impact-time", type=float)
    parser.add_argument("--impact-window-s", type=float, default=0.02)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--min-finite-samples", type=int, default=3)
    parser.add_argument("--poor-impact-threshold", type=float, default=0.05)
    parser.add_argument("--anisotropy-threshold", type=float, default=10.0)
    parser.add_argument("--extreme-scale-min", type=float, default=0.2)
    parser.add_argument("--extreme-scale-max", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(
        measured_target_csv=args.measured_target_csv,
        calibrated_target_csv=args.calibrated_target_csv,
        sim_csv=args.sim_csv,
        output_dir=args.output_dir,
        run_label=args.run_label,
        transform_json=args.transform_json,
        impact_time=args.impact_time,
        impact_window_s=args.impact_window_s,
        write_plots=not args.no_plots,
        min_finite_samples=args.min_finite_samples,
        poor_impact_threshold=args.poor_impact_threshold,
        anisotropy_threshold=args.anisotropy_threshold,
        extreme_scale_min=args.extreme_scale_min,
        extreme_scale_max=args.extreme_scale_max,
    )


if __name__ == "__main__":
    main()
