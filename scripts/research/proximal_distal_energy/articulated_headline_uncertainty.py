"""Registered uncertainty corners for articulated shaft and ground headlines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
    run_articulated_ground_atlas,
)
from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
    run_articulated_shaft_atlas,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
Pathway = Literal["shaft", "ground"]


@dataclass(frozen=True, slots=True)
class UncertaintyAxis:
    """One registered scalar axis with low, nominal, and high values."""

    name: str
    low: float
    nominal: float
    high: float
    pathways: tuple[Pathway, ...]

    def __post_init__(self) -> None:
        values = np.asarray((self.low, self.nominal, self.high), dtype=float)
        if (
            not self.name
            or np.any(~np.isfinite(values))
            or not np.all(np.diff(values) > 0)
        ):
            raise ValueError(
                "uncertainty axes require a name and increasing finite values"
            )
        if not self.pathways or any(
            value not in {"shaft", "ground"} for value in self.pathways
        ):
            raise ValueError("uncertainty axes require shaft and/or ground pathways")


@dataclass(frozen=True, slots=True)
class RegisteredCorner:
    """Nominal or one-at-a-time low/high uncertainty corner."""

    corner_id: str
    axis_name: str
    level: str
    value: float
    pathways: tuple[Pathway, ...]


@dataclass(frozen=True, slots=True)
class HeadlineUncertaintyConfig:
    """Registered bounds and execution controls for the headline campaign."""

    worker_count: int = 4
    axes: tuple[UncertaintyAxis, ...] = (
        UncertaintyAxis("grip_stiffness_scale", 0.6, 1.0, 1.4, ("shaft", "ground")),
        UncertaintyAxis("grip_damping_scale", 0.5, 1.0, 1.5, ("shaft", "ground")),
        UncertaintyAxis(
            "shaft_bending_frequency_scale", 0.8, 1.0, 1.2, ("shaft", "ground")
        ),
        UncertaintyAxis(
            "shaft_torsional_stiffness_scale", 0.64, 1.0, 1.44, ("shaft", "ground")
        ),
        UncertaintyAxis(
            "shaft_damping_ratio", 0.009, 0.018, 0.036, ("shaft", "ground")
        ),
        UncertaintyAxis(
            "ground_translation_stiffness_scale", 0.5, 1.0, 1.5, ("ground",)
        ),
        UncertaintyAxis("ground_translation_damping_scale", 0.5, 1.0, 1.5, ("ground",)),
        UncertaintyAxis(
            "ground_free_moment_stiffness_scale", 0.5, 1.0, 1.5, ("ground",)
        ),
        UncertaintyAxis("ground_free_moment_damping_scale", 0.5, 1.0, 1.5, ("ground",)),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.worker_count, int) or not 1 <= self.worker_count <= 12:
            raise ValueError("worker_count must be an integer from one through twelve")
        names = [axis.name for axis in self.axes]
        if not names or len(names) != len(set(names)):
            raise ValueError("uncertainty axis names must be nonempty and unique")


def registered_corners(
    config: HeadlineUncertaintyConfig = HeadlineUncertaintyConfig(),
) -> tuple[RegisteredCorner, ...]:
    """Return nominal plus every registered low/high one-at-a-time corner."""

    result = [
        RegisteredCorner("nominal", "nominal", "nominal", 1.0, ("shaft", "ground"))
    ]
    for axis in config.axes:
        result.extend(
            (
                RegisteredCorner(
                    f"{axis.name}-low", axis.name, "low", axis.low, axis.pathways
                ),
                RegisteredCorner(
                    f"{axis.name}-high", axis.name, "high", axis.high, axis.pathways
                ),
            )
        )
    return tuple(result)


def _shaft_config(
    corner: RegisteredCorner, config: HeadlineUncertaintyConfig
) -> ArticulatedShaftAtlasConfig:
    result = ArticulatedShaftAtlasConfig(worker_count=config.worker_count)
    updates: dict[str, float] = {}
    if corner.axis_name == "grip_stiffness_scale":
        updates["total_stiffness_n_m"] = result.total_stiffness_n_m * corner.value
    elif corner.axis_name == "grip_damping_scale":
        updates["total_damping_n_s_m"] = result.total_damping_n_s_m * corner.value
    elif corner.axis_name == "shaft_bending_frequency_scale":
        updates["bending_frequency_scale"] = corner.value
    elif corner.axis_name == "shaft_torsional_stiffness_scale":
        updates["torsional_stiffness_scale"] = corner.value
    elif corner.axis_name == "shaft_damping_ratio":
        updates["shaft_damping_ratio"] = corner.value
    return replace(result, **updates)


def _ground_config(
    corner: RegisteredCorner, config: HeadlineUncertaintyConfig
) -> ArticulatedGroundAtlasConfig:
    result = ArticulatedGroundAtlasConfig(worker_count=config.worker_count)
    mapping = {
        "shaft_bending_frequency_scale": "shaft_bending_frequency_scale",
        "shaft_torsional_stiffness_scale": "shaft_torsional_stiffness_scale",
        "shaft_damping_ratio": "shaft_damping_ratio",
        "ground_translation_stiffness_scale": "ground_translation_stiffness_scale",
        "ground_translation_damping_scale": "ground_translation_damping_scale",
        "ground_free_moment_stiffness_scale": "ground_free_moment_stiffness_scale",
        "ground_free_moment_damping_scale": "ground_free_moment_damping_scale",
    }
    updates: dict[str, float] = {}
    if corner.axis_name == "grip_stiffness_scale":
        updates["total_stiffness_n_m"] = result.total_stiffness_n_m * corner.value
    elif corner.axis_name == "grip_damping_scale":
        updates["total_damping_n_s_m"] = result.total_damping_n_s_m * corner.value
    elif corner.axis_name in mapping:
        updates[mapping[corner.axis_name]] = corner.value
    return replace(result, **updates)


def _run_pathway(
    pathway: Pathway,
    corner: RegisteredCorner,
    config: HeadlineUncertaintyConfig,
) -> dict[str, Any]:
    try:
        if pathway == "shaft":
            record, _ = run_articulated_shaft_atlas(_shaft_config(corner, config))
        else:
            record, _ = run_articulated_ground_atlas(_ground_config(corner, config))
    except (
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
        FloatingPointError,
    ) as error:
        return {
            "status": "failed_retained",
            "failure_class": type(error).__name__,
            "failure_message": str(error),
            "matched_cell_count": None,
            "total_cell_count": 384,
            "all_registered_gates_passed": False,
        }
    results = record["results"]
    return {
        "status": "completed",
        "failure_class": None,
        "failure_message": None,
        "matched_cell_count": results["matched_load_work_cell_count"],
        "total_cell_count": results["matched_load_work_total_cell_count"],
        "all_registered_gates_passed": results["all_registered_gates_passed"],
    }


def _movement(rows: list[dict[str, Any]], pathway: Pathway) -> None:
    nominal = next(row for row in rows if row["corner_id"] == "nominal")
    baseline = nominal[pathway]["matched_cell_count"]
    for row in rows:
        count = row[pathway]["matched_cell_count"]
        row[pathway]["matched_cell_count_change_from_nominal"] = (
            int(count - baseline)
            if count is not None and baseline is not None
            else None
        )


def run_headline_uncertainty(
    config: HeadlineUncertaintyConfig = HeadlineUncertaintyConfig(),
) -> dict[str, Any]:
    """Run all registered full-atlas corners and retain every failure."""

    rows: list[dict[str, Any]] = []
    for corner in registered_corners(config):
        row: dict[str, Any] = asdict(corner)
        for pathway in ("shaft", "ground"):
            row[pathway] = (
                _run_pathway(pathway, corner, config)
                if pathway in corner.pathways or corner.corner_id == "nominal"
                else {
                    "status": "not_affected",
                    "matched_cell_count": None,
                    "total_cell_count": 384,
                    "all_registered_gates_passed": None,
                }
            )
        rows.append(row)
    for pathway in ("shaft", "ground"):
        _movement(rows, pathway)
    return {
        "schema_version": "articulated-headline-uncertainty/v1",
        "study_id": "articulated-shaft-ground-headline-uncertainty",
        "design": {
            "method": "registered_nominal_plus_one_at_a_time_low_high_corners",
            "corner_count": len(rows),
            "axes": [asdict(axis) for axis in config.axes],
            "controls": "each full atlas retains both engines, velocity reversal, timestep refinement, and pathway killswitches",
        },
        "configuration": asdict(config),
        "corners": rows,
        "limitations": {
            "interaction_order": "one-at-a-time corners do not estimate higher-order parameter interactions",
            "calibration": "bounds are engineering ranges, not measured participant or equipment properties",
            "human_inference": "survival does not promote any result to a human or coaching claim",
        },
    }


def main() -> None:
    record = run_headline_uncertainty()
    path = DATA / "articulated_headline_uncertainty.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
