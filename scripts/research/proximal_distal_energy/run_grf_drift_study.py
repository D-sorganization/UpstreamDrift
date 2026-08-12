"""Generate deterministic fixed-support reaction drift evidence.

This is a model-internal falsification benchmark.  The fixed shoulder is an
idealized support, not a human foot-ground contact model, and the resulting
planar reaction is not a bilateral force-plate prediction.
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

from scripts.research.proximal_distal_energy.double_pendulum_attribution import (
    double_pendulum_support_reaction_decomposition,
)
from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.physics.contact_reaction_decomposition import (
    evaluate_reaction_prediction,
)
from src.shared.python.simulation_backends import GolfModelParams

SCHEMA_VERSION = "grf-drift-attribution-v1"
FIGURE_STEMS = (
    "fig_grf_drift_components",
    "fig_grf_drift_vectors",
    "fig_grf_falsification_ladder",
)
COLORS = {
    "total": "#111827",
    "configuration": "#6B7280",
    "velocity": "#0072B2",
    "control": "#D55E00",
    "ztcf": "#009E73",
    "zvcf": "#CC79A7",
}


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _manifest() -> list[dict[str, str]]:
    root = _source_root()
    paths = (
        Path(__file__).resolve(),
        root / "scripts/research/proximal_distal_energy/double_pendulum_attribution.py",
        root / "src/shared/python/physics/contact_reaction_decomposition.py",
    )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]


def _reference_trace() -> tuple[
    GolfModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    params = GolfModelParams.default()
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    time, q, velocity, controls = rollout_program(params, program)
    inertials = PlanarInertials.from_params(params)
    impact = find_impact(time, q, velocity, inertials)
    if impact is None:
        raise ValueError("reference case has no valid first impact")
    sampled_time = np.linspace(float(time[0]), float(impact[0]), 161)

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.interp(sampled_time, time, values[:, column])
                for column in range(values.shape[1])
            ]
        )

    return (
        params,
        sampled_time,
        interpolate(q),
        interpolate(velocity),
        interpolate(controls),
    )


def _json_array(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)]


def build_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the deterministic record and trace arrays without writing files."""
    params, time, q, velocity, controls = _reference_trace()
    result = double_pendulum_support_reaction_decomposition(
        time, q, velocity, controls, params
    )
    arrays = {
        "time_s": time,
        "q_rad": q,
        "velocity_rad_s": velocity,
        "control_Nm": controls,
        "total": result.total,
        "configuration": result.configuration,
        "velocity": result.velocity,
        "control": result.control,
        "ztcf": result.ztcf,
        "zvcf": result.zvcf,
    }
    inertials = PlanarInertials.from_params(params)
    total_mass = float(inertials.m1 + inertials.m2)
    body_weight = total_mass * float(inertials.g_proj)
    scale = np.full(2, body_weight)
    metrics = evaluate_reaction_prediction(
        time,
        result.total,
        result.ztcf,
        normalization_scale=scale,
        component_names=("target_horizontal", "swing_plane_vertical"),
    )
    total_impulse = np.trapezoid(result.total, time, axis=0)
    component_impulses = {
        name: np.trapezoid(arrays[name], time, axis=0)
        for name in ("configuration", "velocity", "control", "ztcf", "zvcf")
    }
    total_closure = result.total - (
        result.configuration + result.velocity + result.control
    )
    zvcf_closure = result.zvcf - (result.configuration + result.control)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "model_internal_falsification_benchmark",
        "human_validation": False,
        "model": {
            "name": "fixed-base planar double pendulum",
            "support_interpretation": "shoulder support reaction on mechanism",
            "frame": result.frame,
            "units": result.units,
            "samples": int(time.size),
            "analysis_end_s": float(time[-1]),
            "planar_weight_scale_N": body_weight,
        },
        "counterfactual_contract": {
            "pointwise": True,
            "ztcf": "configuration plus velocity reaction at zero applied control",
            "zvcf": "configuration plus control reaction at zero velocity",
            "overlap_warning": "ZTCF and ZVCF share configuration reaction and are not additive",
        },
        "drift_only_prediction": {
            "target": "modeled_total_support_reaction",
            "predictor": "pointwise_ZTCF_support_reaction",
            "component_names": list(metrics.component_names),
            "bias_N": _json_array(metrics.bias),
            "rmse_N": _json_array(metrics.rmse),
            "nrmse_planar_weight": _json_array(metrics.nrmse),
            "r_squared": _json_array(metrics.r_squared),
            "impulse_error_Ns": _json_array(metrics.impulse_error),
            "total_impulse_Ns": _json_array(total_impulse),
            "component_impulses_Ns": {
                name: _json_array(value) for name, value in component_impulses.items()
            },
        },
        "closure": {
            "max_abs_total_N": float(np.max(np.abs(total_closure))),
            "max_abs_zvcf_N": float(np.max(np.abs(zvcf_closure))),
        },
        "non_identifiable": [
            "bilateral foot-force allocation",
            "center of pressure without a spatial contact model",
            "free moment without a three-dimensional contact wrench",
            "muscle torques from reaction residuals alone",
        ],
        "required_human_falsification_data": [
            "synchronized bilateral six-axis force plates",
            "whole-body and club kinematics",
            "segment inertial parameters and coordinate transforms",
            "declared filtering, event, and held-out evaluation protocols",
        ],
        "source_manifest": _manifest(),
    }
    return record, arrays


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    svg_path = output / f"{stem}.svg"
    figure.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    figure.savefig(
        output / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None}
    )
    plt.close(figure)


def _make_figures(
    record: dict[str, Any], arrays: dict[str, np.ndarray], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    time = arrays["time_s"]
    phase = 100.0 * (time - time[0]) / (time[-1] - time[0])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True)
    for component, axis in enumerate(axes):
        for name in ("total", "configuration", "velocity", "control", "ztcf", "zvcf"):
            label = name.upper() if name in {"ztcf", "zvcf"} else name.title()
            axis.plot(
                phase,
                arrays[name][:, component],
                label=label,
                color=COLORS[name],
                linewidth=1.7,
            )
        axis.axhline(0.0, color="#9CA3AF", linewidth=0.7)
        axis.set_ylabel(
            ("Target-Horizontal" if component == 0 else "Swing-Plane Vertical")
            + " Force (N)"
        )
        axis.grid(alpha=0.22)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Normalized Time to Model Impact (%)")
    fig.suptitle("Pointwise Support-Reaction Attribution")
    fig.tight_layout()
    _save_figure(fig, output, FIGURE_STEMS[0])

    indices = np.linspace(0, time.size - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.6), sharex=True, sharey=True)
    for axis, index in zip(axes.flat, indices, strict=True):
        for name in ("total", "ztcf", "control"):
            vector = arrays[name][index]
            axis.quiver(
                0.0,
                0.0,
                vector[0],
                vector[1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=COLORS[name],
                label=name.upper() if name == "ztcf" else name.title(),
            )
        axis.set_title(f"{phase[index]:.0f}%")
        axis.grid(alpha=0.22)
        axis.set_aspect("equal", adjustable="box")
    limit = 1.08 * max(
        float(np.max(np.abs(arrays["total"]))), float(np.max(np.abs(arrays["ztcf"])))
    )
    for axis in axes.flat:
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Target-Horizontal Force (N)")
    fig.supylabel("Swing-Plane Vertical Force (N)")
    fig.suptitle("Support-Reaction Vectors Along the Achieved Trajectory")
    fig.tight_layout()
    _save_figure(fig, output, FIGURE_STEMS[1])

    fig, axis = plt.subplots(figsize=(8.2, 4.7))
    axis.axis("off")
    boxes = (
        (0.12, "Measured Inputs\nKinematics + Force Plates"),
        (0.38, "Declared Model\nFrames + Inertials + Contacts"),
        (0.64, "Pointwise Predictions\nTotal + ZTCF + ZVCF"),
        (0.88, "Held-Out Tests\nRMSE + Impulse + COP"),
    )
    for x, label in boxes:
        axis.text(
            x,
            0.60,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#F3F4F6",
                "edgecolor": "#374151",
            },
        )
        if x < 0.88:
            axis.annotate(
                "",
                xy=(x + 0.145, 0.60),
                xytext=(x + 0.115, 0.60),
                arrowprops={"arrowstyle": "->", "color": "#374151"},
            )
    axis.text(
        0.02,
        0.22,
        "Failure is informative: rank deficiency, dynamic inconsistency, poor held-out waveform fit, impulse bias, or unstable parameter sensitivity rejects the stated model.",
        fontsize=10,
        wrap=True,
    )
    axis.set_title(
        "Falsification Ladder for Human Ground-Reaction Attribution",
        fontsize=14,
        pad=16,
    )
    fig.tight_layout()
    _save_figure(fig, output, FIGURE_STEMS[2])


def write_study(output_root: Path | str) -> dict[str, Path]:
    """Write machine-readable evidence and publication figures."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    record, arrays = build_study()
    json_path = output / "grf_drift_study.json"
    npz_path = output / "grf_drift_traces.npz"
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(npz_path, **arrays)
    figure_path = output / "figures"
    _make_figures(record, arrays, figure_path)
    return {"json": json_path, "npz": npz_path, "figures": figure_path}


def main() -> None:
    """Regenerate the committed study evidence."""
    output = (
        _source_root() / "docs/research/proximal_distal_energy_transfer/data/grf_drift"
    )
    write_study(output)


if __name__ == "__main__":
    main()
