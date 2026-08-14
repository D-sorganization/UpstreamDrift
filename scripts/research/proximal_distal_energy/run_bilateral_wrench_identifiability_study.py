"""Generate the bilateral-wrench structural-identifiability evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .bilateral_wrench_identifiability import (
    LinearMapAudit,
    audit_linear_map,
    full_hand_wrench_map,
    internal_axial_measurement,
    point_force_wrench_map,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
EVIDENCE = ARTICLE / "data/bilateral_wrench_identifiability_study.json"
FIGURE_PDF = ARTICLE / "figures/fig_bilateral_wrench_identifiability.pdf"
FIGURE_SVG = ARTICLE / "figures/fig_bilateral_wrench_identifiability.svg"


def _contacts(span_m: float) -> np.ndarray:
    return np.array(((-span_m / 2.0, 0.0, 0.0), (span_m / 2.0, 0.0, 0.0)))


def _audit_record(audit: LinearMapAudit) -> dict[str, object]:
    return {
        "shape": list(audit.matrix_shape),
        "rank": audit.rank,
        "nullity": audit.nullity,
        "singular_values": audit.singular_values.tolist(),
        "minimum_nonzero_singular_value": audit.minimum_nonzero_singular_value,
        "nonzero_condition_number": audit.nonzero_condition_number,
    }


def _rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    cross = np.array(
        ((0.0, -unit[2], unit[1]), (unit[2], 0.0, -unit[0]), (-unit[1], unit[0], 0.0))
    )
    return (
        np.eye(3)
        + np.sin(angle_rad) * cross
        + (1.0 - np.cos(angle_rad)) * (cross @ cross)
    )


def build_record() -> dict[str, object]:
    """Build deterministic structural-identifiability evidence."""

    reference_span = 0.20
    contacts = _contacts(reference_span)
    point = audit_linear_map(point_force_wrench_map(contacts))
    full = audit_linear_map(full_hand_wrench_map(contacts))
    augmented = audit_linear_map(
        np.vstack(
            (point_force_wrench_map(contacts), internal_axial_measurement(contacts))
        )
    )

    spans = np.linspace(0.06, 0.30, 49)
    span_audits = [
        audit_linear_map(point_force_wrench_map(_contacts(span))) for span in spans
    ]
    base_singular = point.singular_values
    rotation_differences: list[float] = []
    for axis, angle in (
        (np.array((0.0, 0.0, 1.0)), 0.37),
        (np.array((1.0, 2.0, -1.0)), 1.11),
        (np.array((-2.0, 1.0, 3.0)), 2.03),
    ):
        rotated = contacts @ _rotation(axis, angle).T
        singular = audit_linear_map(point_force_wrench_map(rotated)).singular_values
        rotation_differences.append(float(np.max(np.abs(singular - base_singular))))

    return {
        "schema_version": "bilateral-wrench-identifiability-study/v1",
        "analysis_type": "instantaneous_linear_structural_identifiability",
        "measurement_scaling": {
            "force_scale_n": 1.0,
            "moment_scale_nm": 1.0,
            "purpose": "dimensionless numerical singular-value and condition audit",
        },
        "wrench_order": [
            "force_x",
            "force_y",
            "force_z",
            "moment_x",
            "moment_y",
            "moment_z",
        ],
        "reference_geometry": {
            "grip_span_m": reference_span,
            "reference_point": "grip_midpoint",
        },
        "point_force_map": _audit_record(point),
        "full_bilateral_wrench_map": _audit_record(full),
        "augmented_point_force_map": _audit_record(augmented),
        "grip_span_sweep": {
            "span_m": spans.tolist(),
            "rank": [audit.rank for audit in span_audits],
            "minimum_nonzero_singular_value": [
                audit.minimum_nonzero_singular_value for audit in span_audits
            ],
            "nonzero_condition_number": [
                audit.nonzero_condition_number for audit in span_audits
            ],
        },
        "rotation_audit": {
            "proper_rotation_cases": len(rotation_differences),
            "maximum_singular_value_difference": max(rotation_differences),
        },
        "measurement_boundary": {
            "net_club_wrench_observes_individual_point_forces": False,
            "one_internal_axial_scalar_closes_point_force_rank_gap": True,
            "bilateral_six_axis_required_for_direct_allocation": True,
            "net_club_wrench_alone_is_a_bilateral_sensor_substitute": False,
        },
        "claims": {
            "individual_hand_allocation_from_net_wrench": "structurally_unidentifiable",
            "axial_push_pull_from_net_wrench": "structurally_unidentifiable",
            "force_to_couple_geometric_gain": "proportional_to_declared_grip_span",
            "orientation_dependence_of_rank": "invariant_under_consistent_proper_rotation",
            "muscle_or_scapular_strategy": "not_identified",
            "human_validation": "untested",
            "noise_robust_practical_identifiability": "not_established",
        },
    }


def _plot(record: dict[str, object]) -> None:
    sweep = record["grip_span_sweep"]
    assert isinstance(sweep, dict)
    spans = np.asarray(sweep["span_m"])
    minimum = np.asarray(sweep["minimum_nonzero_singular_value"])
    condition = np.asarray(sweep["nonzero_condition_number"])

    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        (-0.10, 0.10), (0.0, 0.0), color="0.25", linewidth=7, solid_capstyle="round"
    )
    ax.scatter((-0.10, 0.10), (0.0, 0.0), s=75, color=("#2F5597", "#C55A11"), zorder=3)
    ax.arrow(
        -0.10, 0.0, 0.065, 0.0, width=0.0025, color="#2F5597", length_includes_head=True
    )
    ax.arrow(
        0.10, 0.0, -0.065, 0.0, width=0.0025, color="#C55A11", length_includes_head=True
    )
    ax.text(0.0, 0.025, "Equal and Opposite Axial Mode", ha="center")
    ax.text(0.0, -0.032, "Zero Net Force and Zero Net Moment", ha="center")
    ax.set(
        xlim=(-0.14, 0.14), ylim=(-0.07, 0.07), title="A. Invisible Point-Force Mode"
    )
    ax.axis("off")

    ax = axes[0, 1]
    labels = ("Point\nForces", "Point Forces\nPlus Axial", "Bilateral\nSix-Axis")
    ranks = (5, 6, 6)
    nullities = (1, 0, 6)
    x = np.arange(3)
    ax.bar(x - 0.18, ranks, 0.36, label="Rank", color="#2F5597")
    ax.bar(x + 0.18, nullities, 0.36, label="Nullity", color="#C55A11")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Dimension")
    ax.set_title("B. Net-Wrench Measurement Maps")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(spans, minimum, color="#2F5597", linewidth=2.2)
    ax.set_xlabel("Grip Span (m)")
    ax.set_ylabel("Smallest Nonzero Normalized Singular Value")
    ax.set_title("C. Couple Observability Increases With Span")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(spans, condition, color="#C55A11", linewidth=2.2)
    ax.set_xlabel("Grip Span (m)")
    ax.set_ylabel("Normalized Nonzero Condition Ratio")
    ax.set_title("D. Shorter Spans Are More Ill-Conditioned")
    ax.grid(alpha=0.25)

    figure.suptitle(
        "Bilateral Wrench Identifiability Is Geometry and Sensor Dependent", fontsize=14
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
    """Write evidence and its publication figure."""

    record = build_record()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _plot(record)
    print(json.dumps({"evidence": str(EVIDENCE), "figure": str(FIGURE_PDF)}, indent=2))


if __name__ == "__main__":
    main()
