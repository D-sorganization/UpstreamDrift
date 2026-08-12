"""Evaluate a bounded preview-control hypothesis on the archived WSCG trace.

This study does not infer muscle activation and does not simulate clubhead
speed.  It asks a narrower control question: if the archived BASE equivalent
couple is treated as a reference, can a delayed actuator track the small
BASE-minus-pointwise-ZTCF residual more accurately when that residual is
previewed?  The pointwise ZTCF trace remains a stitched same-state evaluation,
not a forward trajectory.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE_ROOT = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
SOURCE_NPZ = ARTICLE_ROOT / "data/two_hand_wscg_analysis.npz"
FIGURE_STEM = "fig_two_hand_preactivation_hypothesis"
SCHEMA_VERSION = "two-hand-preactivation-hypothesis-v1"
LATE_WINDOW_S = (0.16, 0.25)
REFERENCE_TIME_CONSTANT_S = 0.03
TIME_CONSTANTS_S = (0.01, 0.02, 0.03, 0.04, 0.05)
PREVIEW_GRID_S = np.linspace(0.0, 0.08, 81)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_source(
    time: np.ndarray,
    base: np.ndarray,
    pointwise_ztcf: np.ndarray,
    residual: np.ndarray,
) -> None:
    if time.ndim != 1 or time.size < 3 or np.any(np.diff(time) <= 0.0):
        raise ValueError("source time must be a strictly increasing vector")
    for name, values in (
        ("base", base),
        ("pointwise_ztcf", pointwise_ztcf),
        ("residual", residual),
    ):
        if values.shape != time.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite and match source time")
    closure = float(np.max(np.abs(base - pointwise_ztcf - residual)))
    if closure > 2e-12:
        raise ValueError(f"archived BASE/ZTCF/DELTA closure failed: {closure:.3e}")
    if time[0] > LATE_WINDOW_S[0] or time[-1] < LATE_WINDOW_S[1]:
        raise ValueError("source trace does not cover the declared late window")


def _first_order_response(
    time: np.ndarray, command: np.ndarray, time_constant_s: float
) -> np.ndarray:
    """Return exact zero-order-hold response of ``tau da/dt + a = u``."""
    if not np.isfinite(time_constant_s) or time_constant_s <= 0.0:
        raise ValueError("time_constant_s must be finite and positive")
    response = np.empty_like(command)
    response[0] = command[0]
    for index in range(1, time.size):
        decay = np.exp(-(time[index] - time[index - 1]) / time_constant_s)
        response[index] = (
            decay * response[index - 1] + (1.0 - decay) * command[index - 1]
        )
    return response


def _previewed_command(
    time: np.ndarray, required_residual: np.ndarray, preview_s: float
) -> np.ndarray:
    if not np.isfinite(preview_s) or preview_s < 0.0:
        raise ValueError("preview_s must be finite and non-negative")
    return np.interp(
        time + preview_s,
        time,
        required_residual,
        left=float(required_residual[0]),
        right=float(required_residual[-1]),
    )


def _rmse(values: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values[mask] - target[mask]))))


def _evaluate_time_constant(
    time: np.ndarray,
    base: np.ndarray,
    pointwise_ztcf: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
    time_constant_s: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    reactive_activation = _first_order_response(time, residual, time_constant_s)
    reactive_total = pointwise_ztcf + reactive_activation
    reactive_rmse = _rmse(reactive_total, base, mask)

    preview_rows: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for preview_s in PREVIEW_GRID_S:
        command = _previewed_command(time, residual, float(preview_s))
        activation = _first_order_response(time, command, time_constant_s)
        delivered_total = pointwise_ztcf + activation
        preview_rows.append(
            (
                _rmse(delivered_total, base, mask),
                float(preview_s),
                command,
                delivered_total,
            )
        )
    best_rmse, best_preview, best_command, best_total = min(
        preview_rows, key=lambda row: (row[0], row[1])
    )

    naive_activation = _first_order_response(time, base, time_constant_s)
    naive_total = pointwise_ztcf + naive_activation
    result = {
        "time_constant_s": float(time_constant_s),
        "reactive_rmse_nm": reactive_rmse,
        "best_preview_s": best_preview,
        "best_preview_rmse_nm": best_rmse,
        "improvement_percent": 100.0 * (reactive_rmse - best_rmse) / reactive_rmse,
        "naive_net_target_rmse_nm": _rmse(naive_total, base, mask),
    }
    arrays = {
        "reactive_activation_nm": reactive_activation,
        "reactive_total_nm": reactive_total,
        "best_preview_command_nm": best_command,
        "best_preview_total_nm": best_total,
        "naive_total_nm": naive_total,
        "preview_grid_s": PREVIEW_GRID_S.copy(),
        "preview_rmse_nm": np.asarray([row[0] for row in preview_rows]),
    }
    return result, arrays


def build_study(
    source_npz: Path = SOURCE_NPZ,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build deterministic hypothesis evidence from the registered source trace."""
    source_path = Path(source_npz)
    with np.load(source_path) as source:
        time = np.asarray(source["base__time"], dtype=float)
        base = np.asarray(source["base__equivalent_couple"], dtype=float)
        pointwise_ztcf = np.asarray(source["ztcf__equivalent_couple"], dtype=float)
        residual = np.asarray(source["delta__equivalent_couple"], dtype=float)
    _validate_source(time, base, pointwise_ztcf, residual)
    mask = (time >= LATE_WINDOW_S[0]) & (time <= LATE_WINDOW_S[1])

    sensitivity: list[dict[str, float]] = []
    sensitivity_arrays: dict[float, dict[str, np.ndarray]] = {}
    for time_constant in TIME_CONSTANTS_S:
        result, arrays = _evaluate_time_constant(
            time, base, pointwise_ztcf, residual, mask, time_constant
        )
        sensitivity.append(result)
        sensitivity_arrays[time_constant] = arrays

    reference_row = next(
        row
        for row in sensitivity
        if row["time_constant_s"] == REFERENCE_TIME_CONSTANT_S
    )
    reference_arrays = sensitivity_arrays[REFERENCE_TIME_CONSTANT_S]
    minimum_index = int(np.argmin(pointwise_ztcf))
    base_at_minimum = float(base[minimum_index])
    pointwise_minimum = float(pointwise_ztcf[minimum_index])
    if abs(base_at_minimum) < 1e-12:
        raise ValueError("BASE couple is too small at the pointwise ZTCF minimum")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_type": "model_only_delayed_actuator_preview_hypothesis",
        "source": {
            "path": source_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _digest(source_path),
            "trace_semantics": (
                "archived stitched same-state BASE, pointwise ZTCF, and DELTA; "
                "not a forward zero-torque trajectory"
            ),
        },
        "protocol": {
            "target": "archived BASE equivalent midpoint couple",
            "drift": "archived stitched pointwise ZTCF equivalent midpoint couple",
            "required_control": "BASE minus pointwise ZTCF (archived DELTA)",
            "actuator": "first-order unit-gain lag with exact zero-order-hold update",
            "late_window_s": list(LATE_WINDOW_S),
            "preview_grid_s": [round(float(value), 3) for value in PREVIEW_GRID_S],
            "initial_condition": "actuator output initialized to first command sample",
        },
        "source_reversal": {
            "time_of_pointwise_ztcf_minimum_s": float(time[minimum_index]),
            "pointwise_ztcf_minimum_nm": pointwise_minimum,
            "base_at_ztcf_minimum_nm": base_at_minimum,
            "control_residual_at_ztcf_minimum_nm": float(residual[minimum_index]),
            "drift_fraction_at_ztcf_minimum": pointwise_minimum / base_at_minimum,
            "base_peak_abs_nm": float(np.max(np.abs(base[mask]))),
            "control_residual_peak_abs_nm": float(np.max(np.abs(residual[mask]))),
            "closure_max_abs_nm": float(
                np.max(np.abs(base - pointwise_ztcf - residual))
            ),
        },
        "reference_actuator": {
            "time_constant_s": REFERENCE_TIME_CONSTANT_S,
            "best_preview_s": reference_row["best_preview_s"],
            "reactive_rmse_nm": reference_row["reactive_rmse_nm"],
            "preview_rmse_nm": reference_row["best_preview_rmse_nm"],
            "improvement_percent": reference_row["improvement_percent"],
            "naive_net_target_rmse_nm": reference_row["naive_net_target_rmse_nm"],
        },
        "time_constant_sensitivity": sensitivity,
        "claim_boundary": {
            "human_preactivation": "not_established",
            "clubhead_speed_outcome": "not_evaluated",
            "muscle_activation": "not_modeled",
            "interpretation": (
                "preview reduces delayed residual-couple tracking error in this "
                "declared signal model; it does not establish a human strategy"
            ),
        },
    }
    arrays_out = {
        "time_s": time,
        "late_window_mask": mask,
        "base_couple_nm": base,
        "pointwise_ztcf_couple_nm": pointwise_ztcf,
        "control_residual_nm": residual,
        **reference_arrays,
    }
    for row in sensitivity:
        key = f"tau_{int(round(row['time_constant_s'] * 1000)):02d}ms"
        arrays_out[f"{key}_preview_grid_s"] = sensitivity_arrays[
            row["time_constant_s"]
        ]["preview_grid_s"]
        arrays_out[f"{key}_preview_rmse_nm"] = sensitivity_arrays[
            row["time_constant_s"]
        ]["preview_rmse_nm"]
    return record, arrays_out


def _render_figure(
    record: dict[str, Any], arrays: dict[str, np.ndarray], figure_dir: Path
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "pdf.use14corefonts": True,
            "pdf.compression": 9,
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
        }
    )
    time = arrays["time_s"]
    mask = arrays["late_window_mask"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.3))

    axes[0].plot(time, arrays["base_couple_nm"], color="#172B4D", lw=2.0, label="BASE")
    axes[0].plot(
        time,
        arrays["pointwise_ztcf_couple_nm"],
        color="#007C91",
        lw=1.8,
        label="Stitched Pointwise ZTCF",
    )
    axes[0].plot(
        time,
        arrays["control_residual_nm"],
        color="#B23A48",
        lw=1.6,
        label="Required Control Residual",
    )
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].axvspan(*LATE_WINDOW_S, color="#E2E8F0", alpha=0.55)
    axes[0].set_title(
        "The Late Negative Couple Is Predominantly a Pointwise Drift Reaction"
    )
    axes[0].set_ylabel("Equivalent Couple (N m)")
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(
        time[mask],
        arrays["base_couple_nm"][mask],
        color="#172B4D",
        lw=2.2,
        label="Target BASE",
    )
    axes[1].plot(
        time[mask],
        arrays["reactive_total_nm"][mask],
        color="#D97706",
        lw=1.5,
        label="Reactive Residual Command",
    )
    axes[1].plot(
        time[mask],
        arrays["best_preview_total_nm"][mask],
        color="#2A9D8F",
        lw=1.7,
        label="Previewed Residual Command",
    )
    axes[1].plot(
        time[mask],
        arrays["naive_total_nm"][mask],
        color="#7C3AED",
        lw=1.2,
        ls="--",
        label="Naive Net-Couple Command Plus Drift",
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_title(
        "Drift-Aware Preview Improves a Delayed Actuator's Couple Tracking"
    )
    axes[1].set_ylabel("Delivered Couple (N m)")
    axes[1].legend(ncol=2, fontsize=8)

    colors = ("#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51")
    for row, color in zip(record["time_constant_sensitivity"], colors, strict=True):
        key = f"tau_{int(round(row['time_constant_s'] * 1000)):02d}ms"
        axes[2].plot(
            1_000.0 * arrays[f"{key}_preview_grid_s"],
            arrays[f"{key}_preview_rmse_nm"],
            color=color,
            lw=1.6,
            label=f"Time Constant = {1_000 * row['time_constant_s']:.0f} ms",
        )
        axes[2].scatter(
            1_000.0 * row["best_preview_s"],
            row["best_preview_rmse_nm"],
            color=color,
            s=24,
            zorder=4,
        )
    axes[2].set_title("The Best Preview Depends on the Assumed Actuator Time Constant")
    axes[2].set_xlabel("Command Preview (ms)")
    axes[2].set_ylabel("Late-Window RMSE (N m)")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].grid(alpha=0.2)

    fig.suptitle(
        "Two-Hand Residual-Couple Preview Is a Bounded Preactivation Hypothesis",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    figure_dir.mkdir(parents=True, exist_ok=True)
    svg = figure_dir / f"{FIGURE_STEM}.svg"
    pdf = figure_dir / f"{FIGURE_STEM}.pdf"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return svg, pdf


def write_study(output_root: Path = ARTICLE_ROOT) -> dict[str, Path]:
    """Write JSON, NPZ, and paired publication figures under ``output_root``."""
    root = Path(output_root)
    data_dir = root / "data"
    figure_dir = root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    record, arrays = build_study()
    json_path = data_dir / "two_hand_preactivation_hypothesis.json"
    npz_path = data_dir / "two_hand_preactivation_hypothesis.npz"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    svg, pdf = _render_figure(record, arrays, figure_dir)
    return {"json": json_path, "npz": npz_path, "svg": svg, "pdf": pdf}


def main() -> None:
    """Generate the committed hypothesis evidence."""
    write_study()


if __name__ == "__main__":
    main()
