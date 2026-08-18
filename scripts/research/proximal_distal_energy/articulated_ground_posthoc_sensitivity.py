"""Transparent post-hoc sensitivity for the preregistered ground match failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/articulated_ground_atlas.json",
    "docs/research/proximal_distal_energy_transfer/data/articulated_ground_atlas.npz",
    "scripts/research/proximal_distal_energy/articulated_ground_posthoc_sensitivity.py",
    "tests/research/test_articulated_ground_posthoc_sensitivity.py",
)
TOLERANCES = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)


def _relative(left: np.ndarray, right: np.ndarray, floor: float) -> np.ndarray:
    scale = np.maximum(floor, 0.5 * (np.abs(left) + np.abs(right)))
    return np.abs(left - right) / scale


def _signed_summary(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=float).ravel()
    return {
        "count": int(flat.size),
        "minimum": float(np.min(flat)),
        "median": float(np.median(flat)),
        "maximum": float(np.max(flat)),
        "positive_count": int(np.count_nonzero(flat > 0.0)),
        "negative_count": int(np.count_nonzero(flat < 0.0)),
        "zero_count": int(np.count_nonzero(flat == 0.0)),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_ground_posthoc_sensitivity() -> dict[str, Any]:
    """Explain the primary match failure without replacing its estimand."""

    atlas_record = json.loads(
        (DATA / "articulated_ground_atlas.json").read_text(encoding="utf-8")
    )
    with np.load(DATA / "articulated_ground_atlas.npz") as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    fixed, coupled = 0, 3
    load_error = np.asarray(arrays["load_match_relative_error"], dtype=float)
    primary_work_error = np.asarray(arrays["work_match_relative_error"], dtype=float)
    speed_delta = (
        arrays["primary_final_speed"][:, coupled]
        - arrays["primary_final_speed"][:, fixed]
    )
    nonground_work = (
        arrays["primary_terminal_total_work"] - arrays["primary_terminal_ground_work"]
    )
    nonground_work_error = _relative(
        nonground_work[:, coupled], nonground_work[:, fixed], 1.0e-6
    )
    primary_tolerance = []
    nonground_tolerance = []
    for tolerance in TOLERANCES:
        primary = (load_error <= tolerance) & (primary_work_error <= tolerance)
        alternative = (load_error <= tolerance) & (nonground_work_error <= tolerance)
        primary_tolerance.append(
            {
                "tolerance": tolerance,
                "matched_cell_count": int(np.count_nonzero(primary)),
            }
        )
        nonground_tolerance.append(
            {
                "tolerance": tolerance,
                "matched_cell_count": int(np.count_nonzero(alternative)),
                "speed_difference_m_s": (
                    _signed_summary(speed_delta[alternative])
                    if np.any(alternative)
                    else None
                ),
            }
        )
    horizons = np.asarray(arrays["horizons_s"], dtype=float)
    fine_speed = arrays["primary_final_speed"][:, :, :, -1, :, :]
    coupled_speed = fine_speed[:, coupled]
    pathway = {}
    for slot, name in enumerate(arrays["ground_activation_names"]):
        pathway[str(name)] = [
            {
                "horizon_s": float(horizon),
                "coupled_or_pathway_minus_fixed_m_s": _signed_summary(
                    fine_speed[:, slot, ..., horizon_slot]
                    - fine_speed[:, fixed, ..., horizon_slot]
                ),
            }
            for horizon_slot, horizon in enumerate(horizons)
        ]
    controls = {}
    fine_controls = arrays["control_final_speed"][:, :, :, -1, :, :]
    for slot, name in enumerate(arrays["control_names"]):
        controls[str(name)] = [
            {
                "horizon_s": float(horizon),
                "coupled_minus_control_m_s": _signed_summary(
                    coupled_speed[..., horizon_slot]
                    - fine_controls[:, slot, ..., horizon_slot]
                ),
            }
            for horizon_slot, horizon in enumerate(horizons)
        ]
    return {
        "schema_version": "articulated-ground-posthoc-sensitivity/v1",
        "study_id": "finite-ground-primary-match-failure-explanation",
        "analysis_status": "post_hoc_sensitivity_does_not_replace_preregistered_primary_estimand",
        "primary_result": {
            "matched_cell_count": atlas_record["results"][
                "matched_load_work_cell_count"
            ],
            "total_cell_count": atlas_record["results"][
                "matched_load_work_total_cell_count"
            ],
            "load_error_range": [float(np.min(load_error)), float(np.max(load_error))],
            "total_dissipated_work_error_range": [
                float(np.min(primary_work_error)),
                float(np.max(primary_work_error)),
            ],
            "tolerance_sensitivity": primary_tolerance,
        },
        "nonground_dissipation_sensitivity": {
            "definition": "grip plus shaft dissipated work; ground damping work excluded after observing the primary match failure",
            "work_error_range": [
                float(np.min(nonground_work_error)),
                float(np.max(nonground_work_error)),
            ],
            "tolerance_sensitivity": nonground_tolerance,
        },
        "unmatched_pathway_speed_sensitivity": pathway,
        "independent_control_speed_sensitivity": controls,
        "interpretation_boundary": (
            "The preregistered total-work match failed and remains the primary result. "
            "Alternative work definitions and unmatched speed differences are descriptive "
            "post-hoc sensitivities, not causal ground-pathway effects."
        ),
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCE_PATHS},
    }


__all__ = ["build_ground_posthoc_sensitivity"]
