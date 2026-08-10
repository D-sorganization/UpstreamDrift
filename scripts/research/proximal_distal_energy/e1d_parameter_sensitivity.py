"""Model-parameter sensitivity analysis for distal-handoff timing (epic #8426).

The analysis perturbs one parameter family at a time around the canonical
two-link model and repeats the timing comparison.  It is a local sensitivity
study, not a population model: the ranges are deliberately transparent and do
not claim to represent percentile anthropometry.

Output:
``docs/research/proximal_distal_energy_transfer/data/e1d_parameter_sensitivity.json``
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from scripts.research.proximal_distal_energy.run_experiments import (
    DATA_DIR,
    ONSET_GRID,
    RESTRAIN_LEVELS,
    WRIST_DRIVE,
    _git_sha,
    rollout_program,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    TorqueProgram,
    drive_only_program,
    passive_program,
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams

logger = logging.getLogger(__name__)

SHOULDER_TORQUE_NM = 60.0


@dataclass(frozen=True)
class ParameterCase:
    """One declared perturbation of the canonical two-link model."""

    name: str
    varied_parameter: str
    value: float
    unit: str
    params: GolfModelParams


def _upper_case(
    base: GolfModelParams, name: str, field: str, factor: float, unit: str
) -> ParameterCase:
    upper = base.upper.model_copy(update={field: getattr(base.upper, field) * factor})
    params = base.model_copy(update={"upper": upper})
    return ParameterCase(
        name, f"upper.{field}", float(getattr(upper, field)), unit, params
    )


def _lower_case(
    base: GolfModelParams, name: str, field: str, factor: float, unit: str
) -> ParameterCase:
    lower = base.lower.model_copy(update={field: getattr(base.lower, field) * factor})
    params = base.model_copy(update={"lower": lower})
    return ParameterCase(
        name, f"lower.{field}", float(getattr(lower, field)), unit, params
    )


def _scalar_case(
    base: GolfModelParams, name: str, field: str, value: float, unit: str
) -> ParameterCase:
    params = base.model_copy(update={field: value})
    return ParameterCase(name, field, value, unit, params)


def _damping_case(base: GolfModelParams, name: str, factor: float) -> ParameterCase:
    params = base.model_copy(
        update={
            "damping_shoulder": base.damping_shoulder * factor,
            "damping_wrist": base.damping_wrist * factor,
        }
    )
    return ParameterCase(
        name,
        "damping_shoulder_and_wrist",
        factor,
        "multiple of baseline",
        params,
    )


def build_parameter_cases() -> tuple[ParameterCase, ...]:
    """Return the preregistered one-at-a-time sensitivity cases.

    Postcondition: names are unique and every case contains validated,
    immutable :class:`GolfModelParams`.
    """
    base = GolfModelParams.default()
    cases = (
        ParameterCase("baseline", "none", 1.0, "baseline", base),
        _upper_case(base, "arm_length_low", "length_m", 0.90, "m"),
        _upper_case(base, "arm_length_high", "length_m", 1.10, "m"),
        _upper_case(base, "arm_mass_low", "mass_kg", 0.85, "kg"),
        _upper_case(base, "arm_mass_high", "mass_kg", 1.15, "kg"),
        _lower_case(base, "club_length_low", "length_m", 0.90, "m"),
        _lower_case(base, "club_length_high", "length_m", 1.10, "m"),
        _lower_case(base, "clubhead_mass_low", "clubhead_mass_kg", 0.80, "kg"),
        _lower_case(base, "clubhead_mass_high", "clubhead_mass_kg", 1.20, "kg"),
        _scalar_case(
            base, "plane_inclination_low", "plane_inclination_deg", 25.0, "deg"
        ),
        _scalar_case(
            base, "plane_inclination_high", "plane_inclination_deg", 45.0, "deg"
        ),
        _damping_case(base, "joint_damping_low", 0.50),
        _damping_case(base, "joint_damping_high", 1.50),
    )
    if len({case.name for case in cases}) != len(cases):
        raise AssertionError("parameter sensitivity case names must be unique")
    return cases


def _score_program(
    params: GolfModelParams, inertials: PlanarInertials, program: TorqueProgram
) -> dict[str, float | None]:
    t, q, v, _ = rollout_program(params, program)
    impact = find_impact(t, q, v, inertials)
    return {
        "clubhead_speed_mps": None if impact is None else impact[1],
        "impact_time_s": None if impact is None else impact[0],
        "onset_s": None if program.onset_s == float("inf") else program.onset_s,
        "wrist_restrain_nm": program.wrist_restrain_nm,
    }


def _best_valid_strategy(
    params: GolfModelParams,
    inertials: PlanarInertials,
    programs: Iterable[TorqueProgram],
) -> dict[str, float | None]:
    scored = [_score_program(params, inertials, program) for program in programs]
    valid = [row for row in scored if row["clubhead_speed_mps"] is not None]
    if not valid:
        raise ValueError("parameter case produced no valid first-pass impacts")
    return max(valid, key=lambda row: float(row["clubhead_speed_mps"]))


def evaluate_parameter_case(case: ParameterCase) -> dict[str, Any]:
    """Evaluate early, passive, optimized-drive, and optimized-retain strategies."""
    if not case.name.strip():
        raise ValueError("parameter case name must be non-empty")
    inertials = PlanarInertials.from_params(case.params)
    early = _score_program(
        case.params,
        inertials,
        drive_only_program(SHOULDER_TORQUE_NM, WRIST_DRIVE, 0.0),
    )
    passive = _score_program(
        case.params, inertials, passive_program(SHOULDER_TORQUE_NM)
    )
    best_drive = _best_valid_strategy(
        case.params,
        inertials,
        (
            drive_only_program(SHOULDER_TORQUE_NM, WRIST_DRIVE, float(onset))
            for onset in ONSET_GRID
        ),
    )
    best_restrain = _best_valid_strategy(
        case.params,
        inertials,
        (
            restrain_then_drive_program(
                SHOULDER_TORQUE_NM, WRIST_DRIVE, restraint, float(onset)
            )
            for restraint in RESTRAIN_LEVELS
            for onset in ONSET_GRID
        ),
    )
    strategies = {
        "early_drive": early,
        "passive": passive,
        "best_drive": best_drive,
        "best_restrain": best_restrain,
    }
    if any(row["clubhead_speed_mps"] is None for row in strategies.values()):
        ordering: list[str] = []
    else:
        ordering = sorted(
            strategies,
            key=lambda name: float(strategies[name]["clubhead_speed_mps"]),
        )
    return {
        "case": case.name,
        "varied_parameter": case.varied_parameter,
        "value": case.value,
        "unit": case.unit,
        "strategies": strategies,
        "ordering": ordering,
        "ordering_confirmed": ordering
        == ["early_drive", "passive", "best_drive", "best_restrain"],
    }


def run_parameter_sensitivity(
    cases: Iterable[ParameterCase] | None = None,
) -> dict[str, Any]:
    """Evaluate and persist all supplied cases; defaults to the declared set."""
    selected = tuple(build_parameter_cases() if cases is None else cases)
    if not selected:
        raise ValueError("at least one parameter case is required")
    results = [evaluate_parameter_case(case) for case in selected]
    output = {
        "provenance": {
            "git_sha": _git_sha(),
            "experiment": "E1d one-at-a-time model-parameter sensitivity",
            "shoulder_torque_nm": SHOULDER_TORQUE_NM,
            "wrist_drive_nm": WRIST_DRIVE,
            "interpretation": "local model sensitivity; not a population distribution",
        },
        "all_cases_confirm_ordering": all(row["ordering_confirmed"] for row in results),
        "cases": results,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "e1d_parameter_sensitivity.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)
    logger.info("Wrote parameter sensitivity results to %s", output_path)
    return output


def main() -> None:
    """Run the declared E1d sensitivity set."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_parameter_sensitivity()
    logger.info(
        "Parameter sensitivity complete; all cases confirm ordering: %s",
        result["all_cases_confirm_ordering"],
    )


if __name__ == "__main__":
    main()
