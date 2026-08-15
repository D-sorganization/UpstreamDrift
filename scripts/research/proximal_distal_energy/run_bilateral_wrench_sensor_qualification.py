"""Generate trajectory-level bilateral point-force sensor qualification evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .bilateral_wrench_sensor_qualification import (
    RecoveryMetrics,
    SensorQualificationConfig,
    run_sensor_qualification,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
EVIDENCE = ARTICLE / "data/bilateral_wrench_sensor_qualification.json"
FIGURE_PDF = ARTICLE / "figures/fig_bilateral_wrench_sensor_qualification.pdf"
FIGURE_SVG = ARTICLE / "figures/fig_bilateral_wrench_sensor_qualification.svg"
SAMPLE_COUNT = 301
TRIAL_COUNT = 32
SEED = 20260814


def _base_config(**changes: object) -> SensorQualificationConfig:
    values: dict[str, object] = {
        "sample_count": SAMPLE_COUNT,
        "trial_count": TRIAL_COUNT,
        "seed": SEED,
    }
    values.update(changes)
    return SensorQualificationConfig(**values)


def _metric_record(metrics: RecoveryMetrics) -> dict[str, float]:
    return {name: float(value) for name, value in asdict(metrics).items()}


def build_record() -> dict[str, object]:
    """Build the deterministic synthetic qualification matrix."""

    ideal = run_sensor_qualification(
        _base_config(
            normalized_noise_std=0.0,
            normalized_cross_talk=0.0,
            contact_migration_m=0.0,
        )
    )
    case_configs = {
        "noise_only": _base_config(
            normalized_noise_std=0.002,
            normalized_cross_talk=0.0,
            contact_migration_m=0.0,
        ),
        "cross_talk_uncorrected": _base_config(
            normalized_noise_std=0.002,
            normalized_cross_talk=0.01,
            contact_migration_m=0.0,
            apply_cross_talk_correction=False,
        ),
        "cross_talk_calibrated": _base_config(
            normalized_noise_std=0.002,
            normalized_cross_talk=0.01,
            cross_talk_calibration_error_fraction=0.0,
            contact_migration_m=0.0,
        ),
        "cross_talk_calibration_residual": _base_config(
            normalized_noise_std=0.002,
            normalized_cross_talk=0.01,
            cross_talk_calibration_error_fraction=0.10,
            contact_migration_m=0.0,
        ),
        "contact_migration_fixed": _base_config(
            normalized_noise_std=0.0,
            normalized_cross_talk=0.0,
            contact_migration_m=0.008,
            track_contact_centers=False,
        ),
        "contact_migration_tracked": _base_config(
            normalized_noise_std=0.0,
            normalized_cross_talk=0.0,
            contact_migration_m=0.008,
            track_contact_centers=True,
        ),
        "combined_registered": _base_config(),
    }
    cases = {
        "ideal_augmented": _metric_record(ideal.augmented),
        "net_wrench_only": _metric_record(ideal.net_wrench_only),
    }
    for name, config in case_configs.items():
        cases[name] = _metric_record(run_sensor_qualification(config).augmented)
    return {
        "schema_version": "bilateral-wrench-sensor-qualification/v1",
        "analysis_type": "synthetic_point_force_sensor_qualification",
        "sample_count": SAMPLE_COUNT,
        "trial_count": TRIAL_COUNT,
        "seed": SEED,
        "channel_scaling": {
            "force_channels_n": 100.0,
            "moment_channels_nm": 10.0,
            "internal_axial_channel_n": 100.0,
            "purpose": "dimensionless cross-talk, noise, and net-wrench error audit",
        },
        "registered_combined_assumptions": asdict(case_configs["combined_registered"]),
        "cases": cases,
        "qualification": {
            "net_wrench_only_allocation": "fails_by_structure",
            "declared_synthetic_augmented_map": (
                "recoverable_with_error_conditioned_on_sensor_and_contact_assumptions"
            ),
            "cross_talk_calibration": "beneficial_in_declared_synthetic_case",
            "contact_center_tracking": "removes_declared_migration_model_bias",
        },
        "boundaries": {
            "bilateral_full_wrench_allocation": "not_addressed",
            "distributed_contact_pressure": "not_addressed",
            "sensor_values": "synthetic_not_device_calibration",
            "human_validation": "untested",
            "muscle_or_scapular_strategy": "not_identified",
            "coaching_inference": "unsupported",
        },
    }


def _plot(record: dict[str, object]) -> None:
    cases = record["cases"]
    assert isinstance(cases, dict)
    names = list(cases)
    allocation = np.array([cases[name]["allocation_rmse_n"] for name in names])
    net_error = np.array([cases[name]["normalized_net_wrench_rmse"] for name in names])
    labels = [name.replace("_", " ").title() for name in names]

    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(10.4, 8.0), constrained_layout=True)
    x = np.arange(len(names))
    axes[0].bar(x, np.maximum(allocation, 1e-13), color="#2F5597")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Point-Force Allocation RMSE (N)")
    axes[0].set_title("A. Recovery Error Depends on the Measurement Contract")
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, np.maximum(net_error, 1e-15), color="#C55A11")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Normalized Net-Wrench RMSE")
    axes[1].set_title("B. Net-Wrench Closure Can Hide Allocation Error")
    axes[1].set_xticks(x, labels, rotation=24, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    null_error = cases["net_wrench_only"]["axial_mode_rmse_n"]
    figure.suptitle(
        "Synthetic Bilateral Point-Force Sensor Qualification\n"
        f"Net-Wrench-Only Axial-Mode RMSE: {null_error:.2f} N",
        fontsize=14,
    )
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_SVG, bbox_inches="tight")
    plt.close(figure)
    svg_lines = FIGURE_SVG.read_text(encoding="utf-8").splitlines()
    FIGURE_SVG.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Write the qualification record and publication figure."""

    record = build_record()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _plot(record)
    print(json.dumps({"evidence": str(EVIDENCE), "figure": str(FIGURE_PDF)}, indent=2))


if __name__ == "__main__":
    main()
