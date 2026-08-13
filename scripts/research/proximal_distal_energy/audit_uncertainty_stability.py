"""Audit small-sample stability of uncertainty and control conclusions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .uncertainty_control import nondominated_indices, partial_rank_correlations

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
OUTPUT = DATA / "uncertainty_control_stability_audit.json"


def _objectives(outputs: np.ndarray) -> np.ndarray:
    """Return five minimization objectives for programs by samples by metrics."""
    return np.column_stack(
        [
            -np.quantile(outputs[:, :, 0], 0.10, axis=1),
            np.mean(outputs[:, :, 1], axis=1),
            np.quantile(outputs[:, :, 2], 0.90, axis=1),
            np.mean(outputs[:, :, 3], axis=1),
            np.std(outputs[:, :, 0], axis=1, ddof=1),
        ]
    )


def build_audit() -> dict[str, object]:
    """Compute leave-one-out rankings, Pareto membership, and rank thresholds."""
    record = json.loads(
        (DATA / "uncertainty_control_study.json").read_text(encoding="utf-8")
    )
    with np.load(DATA / "uncertainty_control_study.npz") as arrays:
        design = arrays["global_physical_design"]
        outputs = arrays["global_outputs"]
        held = arrays["held_out_outputs"]
        sensitivity = arrays["standardized_sensitivity"]

    parameter_names = record["global_sensitivity"]["parameter_names"]
    metric_names = record["global_sensitivity"]["metric_names"]
    point = np.column_stack(
        [
            partial_rank_correlations(design, outputs[:, index])
            for index in range(outputs.shape[1])
        ]
    )
    leave_one_out = np.stack(
        [
            np.column_stack(
                [
                    partial_rank_correlations(
                        np.delete(design, omitted, axis=0),
                        np.delete(outputs[:, metric], omitted),
                    )
                    for metric in range(outputs.shape[1])
                ]
            )
            for omitted in range(design.shape[0])
        ]
    )
    point_leaders = np.argmax(np.abs(point), axis=0)
    leader_counts: dict[str, dict[str, int]] = {}
    sign_flip_counts: dict[str, dict[str, int]] = {}
    for metric, metric_name in enumerate(metric_names):
        leaders = np.argmax(np.abs(leave_one_out[:, :, metric]), axis=1)
        leader_counts[metric_name] = {
            parameter_names[index]: int(np.sum(leaders == index))
            for index in sorted(set(leaders.tolist()))
        }
        sign_flip_counts[metric_name] = {
            parameter_names[index]: int(
                np.sum(
                    np.sign(leave_one_out[:, index, metric])
                    != np.sign(point[index, metric])
                )
            )
            for index in range(len(parameter_names))
            if np.any(
                np.sign(leave_one_out[:, index, metric])
                != np.sign(point[index, metric])
            )
        }

    names = [row["name"] for row in record["control_comparison"]["candidates"]]
    point_pareto = nondominated_indices(_objectives(held))
    jackknife_pareto = [
        nondominated_indices(_objectives(np.delete(held, omitted, axis=1)))
        for omitted in range(held.shape[1])
    ]
    membership_counts = {
        name: int(sum(index in members for members in jackknife_pareto))
        for index, name in enumerate(names)
    }

    singular = np.linalg.svd(sensitivity, compute_uv=False)
    threshold_fractions = (0.01, 0.05, 0.10, 0.20)
    threshold_ranks = {
        str(fraction): int(np.sum(singular > fraction * singular[0]))
        for fraction in threshold_fractions
    }
    return {
        "schema_version": "proximal-distal-uncertainty-stability-audit-v1",
        "prcc_leave_one_out": {
            "replicates": int(design.shape[0]),
            "point_leaders": {
                metric_names[metric]: parameter_names[index]
                for metric, index in enumerate(point_leaders)
            },
            "leader_counts": leader_counts,
            "sign_flip_counts": sign_flip_counts,
        },
        "held_out_pareto_jackknife": {
            "replicates": int(held.shape[1]),
            "point_members": [names[index] for index in point_pareto],
            "membership_counts": membership_counts,
        },
        "identifiability_threshold_sensitivity": {
            "singular_values": singular.tolist(),
            "rank_by_fraction_of_largest": threshold_ranks,
        },
        "interpretation": (
            "small-sample perturbation audit only; instability weakens ranking "
            "and Pareto-membership claims and cannot quantify population uncertainty"
        ),
    }


def main() -> None:
    """Write the deterministic stability audit."""
    OUTPUT.write_text(json.dumps(build_audit(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
