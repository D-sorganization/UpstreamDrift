"""Audit Pareto and local-linear stability of the transmission study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.research.proximal_distal_energy.transmission_robustness import (
    nondominated_indices,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _objectives(outputs: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            -np.quantile(outputs[:, :, 0], 0.10, axis=1),
            np.std(outputs[:, :, 0], axis=1, ddof=1),
            np.mean(outputs[:, :, 1], axis=1),
            np.quantile(outputs[:, :, 2], 0.90, axis=1),
            np.mean(outputs[:, :, 3], axis=1),
        )
    )


def build_audit() -> dict[str, object]:
    record = json.loads((DATA / "transmission_robustness_study.json").read_text())
    with np.load(DATA / "transmission_robustness_study.npz") as archive:
        held = archive["held_out_outcomes"]
        perturbations = archive["held_out_perturbations"]
        jacobian = archive["local_outcome_jacobian"]
        nominal = archive["nominal_outcomes"][1, :3]
    jackknife = []
    counts = np.zeros(len(record["programs"]), dtype=int)
    for omitted in range(held.shape[1]):
        indices = nondominated_indices(_objectives(np.delete(held, omitted, axis=1)))
        counts[list(indices)] += 1
        jackknife.append([record["programs"][index] for index in indices])
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    thresholds = (0.01, 0.05, 0.10, 0.20)
    predicted = nominal + perturbations @ jacobian.T
    actual = held[1, :, :3]
    rmse = np.sqrt(np.mean((actual - predicted) ** 2, axis=0))
    ranges = np.ptp(actual, axis=0)
    return {
        "schema_version": "transmission-stability-audit-v1",
        "study_id": record["study_id"],
        "held_out_case_count": int(held.shape[1]),
        "pareto_leave_one_case_out": jackknife,
        "pareto_membership_counts": dict(
            zip(record["programs"], counts.tolist(), strict=True)
        ),
        "local_task_map": {
            "singular_values_raw_units": singular_values.tolist(),
            "effective_rank_by_relative_threshold": {
                str(threshold): int(
                    np.sum(singular_values > threshold * singular_values[0])
                )
                for threshold in thresholds
            },
            "held_out_linear_prediction_rmse": dict(
                zip(
                    record["local_task_map"]["outcome_names"],
                    rmse.tolist(),
                    strict=True,
                )
            ),
            "held_out_observed_range": dict(
                zip(
                    record["local_task_map"]["outcome_names"],
                    ranges.tolist(),
                    strict=True,
                )
            ),
            "boundary": (
                "singular values depend on units and scaling; the local Jacobian "
                "does not validate linear prediction across the held-out envelope"
            ),
        },
    }


def main() -> None:
    output = DATA / "transmission_stability_audit.json"
    output.write_text(json.dumps(build_audit(), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
