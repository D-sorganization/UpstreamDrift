"""Generate analytical, planar, bilateral, and spatial geometry evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.mechanism_ladder import (
    InteractionSample,
    rotation_matrix,
)
from scripts.research.proximal_distal_energy.momentum_geometry_atlas import (
    bilateral_force_couple,
    distributed_contact_couple,
    force_velocity_projection,
    relative_link_gates,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data"
FIGURES = ARTICLE / "figures"
JSON_PATH = DATA / "momentum_geometry_atlas.json"
NPZ_PATH = DATA / "momentum_geometry_atlas.npz"
FIGURE_PATH = FIGURES / "fig_momentum_geometry_atlas.pdf"
SCHEMA_VERSION = "momentum-transfer-geometry-atlas/v2"
# The station lattice mirrors the executed distributed-grip tier so the
# analytical reduction is evaluated on the same discretization it covers.
STATION_COUNTS_PER_HAND = (1, 3, 5)
STATION_WIDTH_M = 0.03


def _source_hashes() -> dict[str, str]:
    names = ("momentum_geometry_atlas.py", "run_momentum_geometry_atlas.py")
    directory = Path(__file__).parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in names
    }


def _frame_audit() -> dict[str, float]:
    sample = InteractionSample(
        model_tier="spatial_reference",
        time_s=0.0,
        frame="world",
        reference_point_m=np.array([0.12, -0.08, 0.04]),
        force_n=np.array([43.0, -19.0, 11.0]),
        couple_nm=np.array([1.2, -0.7, 3.1]),
        linear_velocity_m_s=np.array([2.4, 0.8, -0.3]),
        angular_velocity_rad_s=np.array([0.4, -0.2, 6.0]),
    )
    rotations = [
        rotation_matrix(axis, angle)
        for axis in (np.eye(3)[0], np.eye(3)[1], np.array([1.0, 1.0, 1.0]))
        for angle in (-1.1, -0.4, 0.7, 1.6)
    ]
    residuals = [
        abs(sample.rotate(value, frame="rotated").total_power_w - sample.total_power_w)
        for value in rotations
    ]
    return {
        "rotation_count": len(rotations),
        "maximum_power_residual_w": max(residuals),
    }


def _station_offsets(count: int, center_m: float) -> np.ndarray:
    """Return symmetric station offsets spanning one hand about ``center_m``."""

    if count == 1:
        return np.array([center_m])
    half_width = STATION_WIDTH_M / 2.0
    return center_m + np.linspace(-half_width, half_width, count)


def _opposed_station_couple(
    count: int, signed_separation_m: float, force_angle_rad: float
) -> float:
    """Return the midpoint couple of an opposed distributed grip."""

    axis = np.array([1.0, 0.0, 0.0])
    force = np.array([np.cos(force_angle_rad), np.sin(force_angle_rad), 0.0]) / count
    offsets = np.concatenate(
        (
            _station_offsets(count, 0.5 * signed_separation_m),
            _station_offsets(count, -0.5 * signed_separation_m),
        )
    )
    forces = np.vstack((np.tile(force, (count, 1)), np.tile(-force, (count, 1))))
    return float(distributed_contact_couple(offsets, axis, forces)[2])


def _common_mode_station_couple(count: int, signed_separation_m: float) -> float:
    """Return the couple of a common-mode distributed grip about its midpoint."""

    axis = np.array([1.0, 0.0, 0.0])
    force = np.array([0.0, 1.0, 0.0]) / (2 * count)
    offsets = np.concatenate(
        (
            _station_offsets(count, 0.5 * signed_separation_m),
            _station_offsets(count, -0.5 * signed_separation_m),
        )
    )
    forces = np.tile(force, (2 * count, 1))
    return float(distributed_contact_couple(offsets, axis, forces)[2])


def _cross_tier_controls() -> dict[str, dict[str, float | str]]:
    """Collect existing achieved-state geometry controls without relabeling them."""

    moving = json.loads(
        (DATA / "moving_base_flexible_study.json").read_text(encoding="utf-8")
    )
    rotating = json.loads(
        (DATA / "rotating_base_torso_velocity_study.json").read_text(encoding="utf-8")
    )
    spatial = json.loads(
        (DATA / "spatial_forward_contact_study.json").read_text(encoding="utf-8")
    )
    ladder = json.loads(
        (DATA / "mechanism_ladder_study.json").read_text(encoding="utf-8")
    )
    prescribed = json.loads(
        (DATA / "subject_scaled_spatial_geometry.json").read_text(encoding="utf-8")
    )
    closed = json.loads(
        (DATA / "subject_scaled_closed_contact.json").read_text(encoding="utf-8")
    )
    grip = json.loads(
        (DATA / "articulated_distributed_grip_atlas.json").read_text(encoding="utf-8")
    )
    shaft = json.loads(
        (DATA / "articulated_shaft_atlas.json").read_text(encoding="utf-8")
    )
    sensor = json.loads(
        (DATA / "bilateral_wrench_sensor_qualification.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "moving_base_planar": {
            "baseline_minimum_couple_nm": moving["zero_command_branch"][
                "minimum_force_generated_couple_nm"
            ],
            "coincident_couple_nm": moving["coincident_grip_negative_control"][
                "maximum_abs_force_generated_couple_nm"
            ],
            "provenance": "moving_base_flexible_study.json",
        },
        "rotating_base_two_hand": {
            "baseline_couple_nm": rotating["negative_controls"][
                "baseline_separated_grip_couple_nm"
            ],
            "coincident_couple_nm": rotating["negative_controls"][
                "coincident_grip_max_couple_nm"
            ],
            "reversed_couple_nm": rotating["negative_controls"][
                "reversed_grip_couple_nm"
            ],
            "provenance": "rotating_base_torso_velocity_study.json",
        },
        "spatial_two_engine": {
            "minimum_killswitch_couple_nm": spatial["mechanism_tests"][
                "same_state_killswitch_minimum_couple_nm"
            ],
            "coincident_couple_nm": spatial["mechanism_tests"][
                "coincident_grip_couple_max_nm"
            ],
            "reversal_residual_nm": spatial["mechanism_tests"][
                "reversed_geometry_sign_residual_max_nm"
            ],
            "provenance": "spatial_forward_contact_study.json",
        },
        "closed_loop_conditioning": {
            "minimum_condition_number": ladder["closed_loop_diagnostics"][
                "minimum_condition_number"
            ],
            "maximum_condition_number": ladder["closed_loop_diagnostics"][
                "maximum_condition_number"
            ],
            "provenance": "mechanism_ladder_study.json",
        },
        "subject_scaled_articulated_contact": {
            "couple_per_span_invariance_residual": prescribed["geometry_tests"][
                "couple_per_span_invariance_residual"
            ],
            "prescribed_maximum_contact_error_m": prescribed["closure_tests"][
                "maximum_hand_to_grip_distance_m"
            ],
            "closed_configuration_count": closed["results"]["feasible_sample_count"],
            "closed_maximum_contact_error_m": closed["results"][
                "maximum_contact_error_m"
            ],
            "provenance": (
                "subject_scaled_spatial_geometry.json; subject_scaled_closed_contact.json"
            ),
        },
        "distributed_club": {
            "coincident_station_couple_residual_nm": grip["results"][
                "maximum_coincident_couple_residual_nm"
            ],
            "reversed_station_couple_sign_residual_nm": grip["results"][
                "maximum_reversed_couple_sign_residual_nm"
            ],
            "maximum_station_load_concentration": grip["results"][
                "maximum_station_load_concentration"
            ],
            "grip_trajectory_count": grip["design"]["trajectory_count"],
            "shaft_trajectory_count": shaft["design"]["trajectory_count"],
            "shaft_maximum_peak_force_couple_nm": shaft["results"][
                "maximum_peak_force_couple_nm"
            ],
            "provenance": (
                "articulated_distributed_grip_atlas.json; articulated_shaft_atlas.json"
            ),
        },
        "held_out_measurement": {
            "net_wrench_only_allocation_rmse_n": sensor["cases"]["net_wrench_only"][
                "allocation_rmse_n"
            ],
            "net_wrench_only_normalized_net_wrench_rmse": sensor["cases"][
                "net_wrench_only"
            ]["normalized_net_wrench_rmse"],
            "combined_registered_allocation_rmse_n": sensor["cases"][
                "combined_registered"
            ]["allocation_rmse_n"],
            "net_wrench_only_qualification": sensor["qualification"][
                "net_wrench_only_allocation"
            ],
            "provenance": "bilateral_wrench_sensor_qualification.json",
        },
    }


def build_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return the preregistered atlas and its machine-readable arrays."""

    projection_angle = np.linspace(-np.pi, np.pi, 361)
    normalized_power = cast(
        np.ndarray, force_velocity_projection(1.0, 1.0, projection_angle)
    )
    relative_angle = projection_angle.copy()
    tangential, centripetal = relative_link_gates(relative_angle)
    separation = np.linspace(-0.30, 0.30, 121)
    force_angle = np.linspace(-np.pi, np.pi, 181)
    couple = np.empty((separation.size, force_angle.size))
    for row, distance in enumerate(separation):
        for column, angle in enumerate(force_angle):
            force = np.array([np.cos(angle), np.sin(angle), 0.0])
            couple[row, column] = bilateral_force_couple(
                float(distance), np.array([1.0, 0.0, 0.0]), force
            )[2]

    station_counts = np.array(STATION_COUNTS_PER_HAND, dtype=np.int64)
    distributed_couple = np.array(
        [
            [
                _opposed_station_couple(int(count), float(distance), np.pi / 2)
                for distance in separation
            ]
            for count in station_counts
        ]
    )
    point_pair_couple = np.array(
        [
            bilateral_force_couple(float(distance), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])[2]
            for distance in separation
        ]
    )
    station_count_deviation = float(
        np.max(np.abs(distributed_couple - point_pair_couple))
    )
    common_mode_residuals = [
        abs(_common_mode_station_couple(int(count), float(distance)))
        for count in station_counts
        for distance in (0.0, 0.12, 0.24)
    ]

    null_residuals = [
        abs(float(force_velocity_projection(1.0, 1.0, np.pi / 2))),
        abs(float(relative_link_gates(np.array([0.0]))[1][0])),
        float(np.linalg.norm(bilateral_force_couple(0.0, [1, 0, 0], [0, 1, 0]))),
        float(np.linalg.norm(bilateral_force_couple(0.2, [1, 0, 0], [1, 0, 0]))),
        max(common_mode_residuals),
        abs(_opposed_station_couple(5, 0.0, np.pi / 2)),
        abs(_opposed_station_couple(5, 0.24, 0.0)),
    ]
    reversal_residuals = [
        abs(
            float(force_velocity_projection(1.0, 1.0, 0.0))
            + float(force_velocity_projection(1.0, 1.0, np.pi))
        ),
        float(np.max(np.abs(couple + couple[::-1]))),
        float(np.max(np.abs(distributed_couple + distributed_couple[:, ::-1]))),
    ]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "reference-explicit-momentum-transfer-geometry-atlas-v1",
        "registered_before_preferred_result": True,
        "geometry_gates": {
            "force_velocity_projection": "F v cos(phi)",
            "distal_tangential_projection": "cos(relative_link_angle)",
            "distal_centripetal_projection": "-sin(relative_link_angle)",
            "bilateral_differential_couple": "signed separation cross differential force",
            "distributed_station_reduction": "axis cross first moment of station forces",
            "closed_loop_conditioning": "constraint Jacobian singular values; magnitude requires dynamics",
        },
        "negative_controls": {
            "orthogonal_force_velocity": "zero force power",
            "relative_angle_gate_zeros": "distinct tangential and centripetal nulls",
            "coincident_grips": "zero bilateral couple",
            "axial_differential_force": "zero bilateral couple",
            "reversed_force_or_moment_arm": "sign reversal",
            "common_mode_station_distribution": "zero midpoint couple",
            "station_count_refinement": "invariant couple at a fixed first moment",
            "maximum_null_residual": max(null_residuals),
            "maximum_reversal_residual": max(reversal_residuals),
            "maximum_station_count_couple_deviation_nm": station_count_deviation,
        },
        "frame_audit": _frame_audit(),
        "cross_tier_controls": _cross_tier_controls(),
        "tier_coverage": {
            "analytical": "executed_exact_geometry",
            "fixed_hub_planar": "executed_relative_link_gates",
            "moving_base_two_hand": "linked_existing_evidence",
            "spatial_forward_contact": "linked_existing_evidence_and_frame_audit",
            "subject_scaled_articulated_contact": (
                "linked_prescribed_state_rejection_and_closed_reduced_tree_evidence"
            ),
            "distributed_club": (
                "executed_exact_first_moment_reduction_and_linked_grip_shaft_evidence"
            ),
            "held_out_measurement": (
                "linked_synthetic_sensor_qualification_not_device_or_participant_data"
            ),
            "subject_scaled": "open_requires_anthropometry_and_contact_calibration",
            "governed_human": "open_blocked_by_bilateral_wrench_dataset_acquisition",
        },
        "claim_status": {
            "force_magnitude_alone_determines_transfer": "rejected",
            "geometry_can_gate_sign_and_zero": "supported_analytically_and_in_declared_model_controls",
            "coordinate_rotation_changes_physical_power": "rejected_for_proper_rotations",
            "universal_human_geometry": "untested",
            "coaching_prescription": "unsupported",
            "station_spread_changes_couple_at_fixed_first_moment": "rejected",
            "net_wrench_identifies_bilateral_contact_geometry": "rejected",
            "calibrated_subject_scaled_contact_geometry": "untested",
        },
        "limitations": [
            "The atlas normalizes force and speed and therefore maps geometry rather than attainable human magnitude.",
            "The bilateral couple surface assumes an ideal opposed differential force mode at a declared midpoint.",
            "Closed-loop conditioning does not determine reaction-force magnitude without dynamics and control.",
            "The distributed reduction is exact for rigid station offsets and says nothing about measured grip pressure, finger anatomy, friction, or tissue.",
            "The subject-scaled tier is linked reduced-tree inverse kinematics with engineering limits and coarse collision bounds, not calibrated anatomy or a participant distribution.",
            "The held-out measurement tier is a synthetic sensor qualification; it is not device calibration and cannot close a human claim.",
            "Scapular, wrist, and grip-compliance geometry and every governed human geometry question remain open.",
        ],
        "source_sha256": _source_hashes(),
        "array_artifact": NPZ_PATH.name,
    }
    arrays = {
        "projection_angle_rad": projection_angle,
        "normalized_force_power_w": normalized_power,
        "relative_link_angle_rad": relative_angle,
        "distal_tangential_gate": tangential,
        "distal_centripetal_gate": centripetal,
        "signed_grip_separation_m": separation,
        "differential_force_angle_rad": force_angle,
        "couple_normalized_nm": couple,
        "station_count_per_hand": station_counts,
        "station_width_m": np.array([STATION_WIDTH_M]),
        "point_pair_couple_normalized_nm": point_pair_couple,
        "distributed_couple_normalized_nm": distributed_couple,
    }
    return record, arrays


def _write_figure(arrays: dict[str, np.ndarray]) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].plot(
        np.degrees(arrays["projection_angle_rad"]),
        arrays["normalized_force_power_w"],
        color="#4C78A8",
    )
    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0].set(
        title="Force–Velocity Projection",
        xlabel="Included Angle (deg)",
        ylabel="Normalized Force Power",
    )
    angle = np.degrees(arrays["relative_link_angle_rad"])
    axes[1].plot(
        angle, arrays["distal_tangential_gate"], label="tangential", color="#F58518"
    )
    axes[1].plot(
        angle, arrays["distal_centripetal_gate"], label="centripetal", color="#54A24B"
    )
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set(
        title="Relative-Link Gates",
        xlabel="Relative Angle (deg)",
        ylabel="Projection Coefficient",
    )
    axes[1].legend(frameon=False)
    image = axes[2].imshow(
        arrays["couple_normalized_nm"],
        origin="lower",
        aspect="auto",
        extent=(-180, 180, -0.30, 0.30),
        cmap="coolwarm",
        vmin=-0.30,
        vmax=0.30,
    )
    axes[2].set(
        title="Bilateral Force-Couple Geometry",
        xlabel="Differential-Force Angle (deg)",
        ylabel="Signed Grip Separation (m)",
    )
    figure.colorbar(image, ax=axes[2], label="Normalized Couple (N m)")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)
    return FIGURE_PATH


def write_outputs() -> tuple[Path, Path, Path]:
    """Write deterministic JSON, arrays, and the reader-facing atlas."""

    record, arrays = build_study()
    DATA.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cast(Any, np.savez_compressed)(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH, _write_figure(arrays)


if __name__ == "__main__":
    for path in write_outputs():
        print(path)
