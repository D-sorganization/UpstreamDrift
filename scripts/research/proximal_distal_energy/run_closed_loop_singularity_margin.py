"""Generate or validate feasible closed-loop singular-margin evidence."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.closed_loop_singularity_margin import (
    ClosedLoopOrbitAudit,
    PlanarClosedGeometry,
    PlanarCoordinateScale,
    audit_closed_loop_orbit,
    audit_feasible_configuration,
    audit_triangle_degeneracies,
    feasible_closed_loop_configuration,
    planar_closure_residual,
)
from scripts.research.proximal_distal_energy.constraint_internal_force_diagnostics import (
    PlanarClosedLoopAudit,
    audit_scaled_planar_closure_jacobian,
)
from scripts.research.proximal_distal_energy.mechanism_ladder import (
    closed_loop_grip_jacobian,
)
from scripts.research.proximal_distal_energy.numeric_evidence import (
    canonicalize_published_numbers,
)

ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "closed_loop_singularity_margin.json"
)
SCHEMA_VERSION = "proximal-distal-closed-loop-singularity-margin/v1"
GEOMETRY = PlanarClosedGeometry(0.75, 0.78, 0.25)
SCALE = PlanarCoordinateScale(1.0, 0.75)
NOMINAL_RELATIVE_TOLERANCE = 1e-12
POSITION_CLOSURE_TOLERANCE_M = 1e-12
PHASE_SAMPLE_COUNT = 181
NEAR_BOUNDARY_OFFSETS_M = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
RELATIVE_TOLERANCES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
ROUND_OFF_DIAGNOSTIC_REPORTING = "conservative_power_of_ten_upper_bound"


def _roundoff_upper_bound(value: float) -> float:
    """Bound an expected-zero floating diagnostic without platform false precision."""

    if value < 0.0 or not math.isfinite(value):
        raise ValueError("roundoff diagnostic must be finite and nonnegative")
    if value == 0.0:
        return 0.0
    return 10.0 ** math.ceil(math.log10(value))


def _orbit_payload(audit: ClosedLoopOrbitAudit) -> dict[str, object]:
    """Serialize an orbit while conservatively bounding SVD roundoff fields."""

    payload = asdict(audit)
    for key in (
        "maximum_scaled_nullspace_residual_m",
        "maximum_scaled_singular_value_spread_m",
    ):
        payload[key] = _roundoff_upper_bound(float(payload[key]))
    return payload


def _audit_payload(audit: PlanarClosedLoopAudit) -> dict[str, object]:
    return {
        "angular_coordinate_scale_rad": audit.angular_coordinate_scale_rad,
        "maximum_scaled_nullspace_residual_m": _roundoff_upper_bound(
            audit.maximum_scaled_nullspace_residual_m
        ),
        "nullity": audit.nullity,
        "rank": audit.rank,
        "scaled_condition_number": (
            audit.scaled_condition_number
            if math.isfinite(audit.scaled_condition_number)
            else None
        ),
        "scaled_singular_values_m": list(audit.scaled_singular_values_m),
        "smallest_scaled_singular_value_m": audit.smallest_scaled_singular_value_m,
        "translation_coordinate_scale_m": audit.translation_coordinate_scale_m,
    }


def _jacobian(
    geometry: PlanarClosedGeometry,
    *,
    phase_rad: float = 0.0,
    branch: int = 1,
) -> tuple[np.ndarray, float]:
    configuration = feasible_closed_loop_configuration(
        geometry,
        phase_rad=phase_rad,
        branch=branch,
    )
    matrix = closed_loop_grip_jacobian(
        lead_angle_rad=configuration.lead_angle_rad,
        trail_angle_rad=configuration.trail_angle_rad,
        grip_angle_rad=configuration.grip_angle_rad,
        lead_arm_length_m=geometry.lead_arm_length_m,
        trail_arm_length_m=geometry.trail_arm_length_m,
        grip_separation_m=geometry.grip_separation_m,
    )
    residual = float(np.max(np.abs(planar_closure_residual(configuration, geometry))))
    return matrix, residual


def _equivalent_unit_control() -> dict[str, object]:
    matrix_m, closure_residual_m = _jacobian(GEOMETRY, phase_rad=0.4)
    audit_m = audit_scaled_planar_closure_jacobian(
        matrix_m,
        (SCALE.angular_coordinate_scale_rad, SCALE.translation_coordinate_scale_m),
        relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
    )
    matrix_cm = 100.0 * matrix_m
    matrix_cm[:, 2:4] /= 100.0
    audit_cm = audit_scaled_planar_closure_jacobian(
        matrix_cm,
        (
            SCALE.angular_coordinate_scale_rad,
            100.0 * SCALE.translation_coordinate_scale_m,
        ),
        relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
    )
    singular_values_m = np.asarray(audit_m.scaled_singular_values_m)
    singular_values_cm = np.asarray(audit_cm.scaled_singular_values_m)
    return {
        "closure_residual_m": closure_residual_m,
        "condition_number_after": audit_cm.scaled_condition_number,
        "condition_number_before": audit_m.scaled_condition_number,
        "conversion": "metre residual/translation coordinates to centimetres",
        "maximum_abs_scaled_spectrum_conversion_residual_cm": _roundoff_upper_bound(
            float(np.max(np.abs(singular_values_cm - 100.0 * singular_values_m)))
        ),
        "rank_after": audit_cm.rank,
        "rank_before": audit_m.rank,
        "singular_value_scale_factor": 100.0,
    }


def _near_boundary_control() -> list[dict[str, object]]:
    lower_span_m = abs(GEOMETRY.lead_arm_length_m - GEOMETRY.trail_arm_length_m)
    cases = []
    for offset_m in NEAR_BOUNDARY_OFFSETS_M:
        geometry = PlanarClosedGeometry(0.75, 0.78, lower_span_m + offset_m)
        configuration = feasible_closed_loop_configuration(
            geometry,
            phase_rad=0.0,
            branch=1,
        )
        closure_residual_m = float(
            np.max(np.abs(planar_closure_residual(configuration, geometry)))
        )
        tolerance_cases = []
        for tolerance in RELATIVE_TOLERANCES:
            audit = audit_feasible_configuration(
                configuration,
                geometry,
                SCALE,
                relative_tolerance=tolerance,
            )
            tolerance_cases.append(
                {"relative_tolerance": tolerance, **_audit_payload(audit)}
            )
        cases.append(
            {
                "closure_residual_m": closure_residual_m,
                "distance_to_lower_degeneracy_m": offset_m,
                "tolerance_cases": tolerance_cases,
                "triangle_sine_margin": configuration.triangle_sine_margin,
            }
        )
    return cases


def _scale_control() -> list[dict[str, object]]:
    return [
        {
            "translation_coordinate_scale_m": translation_scale_m,
            **_orbit_payload(
                audit_closed_loop_orbit(
                    GEOMETRY,
                    PlanarCoordinateScale(1.0, translation_scale_m),
                    phase_sample_count=41,
                    relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
                )
            ),
        }
        for translation_scale_m in (0.50, 0.75, 1.00)
    ]


def _geometry_control() -> list[dict[str, object]]:
    geometries = (
        PlanarClosedGeometry(0.65, 0.70, 0.20),
        GEOMETRY,
        PlanarClosedGeometry(0.90, 0.72, 0.40),
    )
    return [
        {
            "geometry_m": asdict(geometry),
            "orbit": _orbit_payload(
                audit_closed_loop_orbit(
                    geometry,
                    SCALE,
                    phase_sample_count=41,
                    relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
                )
            ),
        }
        for geometry in geometries
    ]


def _impossible_geometry_control() -> list[dict[str, object]]:
    cases = (
        PlanarClosedGeometry(0.75, 0.78, 2.00),
        PlanarClosedGeometry(0.0, 0.78, 0.25),
    )
    results = []
    for geometry in cases:
        try:
            feasible_closed_loop_configuration(geometry, phase_rad=0.0, branch=1)
        except ValueError as error:
            results.append(
                {
                    "geometry_m": asdict(geometry),
                    "rejected": True,
                    "reason": str(error),
                }
            )
        else:
            results.append(
                {"geometry_m": asdict(geometry), "rejected": False, "reason": None}
            )
    return results


def _manufactured_matrix_control() -> dict[str, object]:
    matrix, _ = _jacobian(GEOMETRY, phase_rad=0.4)
    regular = audit_scaled_planar_closure_jacobian(
        matrix,
        (1.0, 0.75),
        relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
    )
    dependent = matrix.copy()
    dependent[3] = dependent[2]
    adverse = audit_scaled_planar_closure_jacobian(
        dependent,
        (1.0, 0.75),
        relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
    )
    return {
        "adverse_fixture": "fourth row replaced by third row",
        "adverse_rank": adverse.rank,
        "adverse_nullity": adverse.nullity,
        "regular_rank": regular.rank,
        "regular_nullity": regular.nullity,
    }


def build_report() -> dict[str, object]:
    """Build deterministic, scale-qualified exact-closure evidence."""

    nominal = audit_closed_loop_orbit(
        GEOMETRY,
        SCALE,
        phase_sample_count=PHASE_SAMPLE_COUNT,
        relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
    )
    degeneracies = audit_triangle_degeneracies(
        GEOMETRY,
        SCALE,
        relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
    )
    report = {
        "classification": "analytical_planar_exact_position_closure",
        "coordinate_scale_contract": asdict(SCALE),
        "exact_triangle_degeneracies": {
            "lower_geometry_m": asdict(degeneracies.lower_geometry),
            "lower_position_closure_residual_m": (
                degeneracies.lower_position_closure_residual_m
            ),
            "lower_rank_audit": _audit_payload(degeneracies.lower),
            "upper_geometry_m": asdict(degeneracies.upper_geometry),
            "upper_position_closure_residual_m": (
                degeneracies.upper_position_closure_residual_m
            ),
            "upper_rank_audit": _audit_payload(degeneracies.upper),
        },
        "falsifiers": [
            "Any registered orbit sample violates the closure tolerance.",
            "A global phase rotation changes the singular spectrum beyond tolerance.",
            "Equivalent length units change rank, nullity, or condition number.",
            "Either exact triangle degeneracy fails to add one velocity null mode.",
            "An impossible triangle geometry passes the constructor.",
            "A manufactured row dependency fails to reduce rank.",
        ],
        "geometry_controls": _geometry_control(),
        "inference_boundary": (
            "This exact same-origin planar kinematic triangle does not establish "
            "anatomical shoulder closure, dynamics, constraint forces or multipliers, "
            "muscle action, passive torque, human occurrence, or coaching guidance."
        ),
        "impossible_geometry_controls": _impossible_geometry_control(),
        "issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9113",
        "manufactured_matrix_killswitch": _manufactured_matrix_control(),
        "near_lower_boundary_sweep": _near_boundary_control(),
        "nominal_geometry_m": asdict(GEOMETRY),
        "nominal_orbit": _orbit_payload(nominal),
        "parent_issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9027",
        "phase_resolution_controls": [
            {
                "phase_sample_count_per_branch": count,
                "orbit": _orbit_payload(
                    audit_closed_loop_orbit(
                        GEOMETRY,
                        SCALE,
                        phase_sample_count=count,
                        relative_tolerance=NOMINAL_RELATIVE_TOLERANCE,
                    )
                ),
            }
            for count in (17, 61, PHASE_SAMPLE_COUNT)
        ],
        "position_closure_tolerance_m": POSITION_CLOSURE_TOLERANCE_M,
        "relative_rank_tolerance": NOMINAL_RELATIVE_TOLERANCE,
        "roundoff_diagnostic_reporting": ROUND_OFF_DIAGNOSTIC_REPORTING,
        "scale_controls": _scale_control(),
        "schema_version": SCHEMA_VERSION,
        "unit_equivalence_control": _equivalent_unit_control(),
    }
    canonical = canonicalize_published_numbers(
        report,
        context="published closed-loop singular-margin evidence",
    )
    if not isinstance(canonical, dict):
        raise TypeError("canonical report must remain a mapping")
    return canonical


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed on lost closure, rank, controls, or inference boundaries."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    nominal = report.get("nominal_orbit", {})
    if nominal.get("minimum_rank") != 4 or nominal.get("maximum_rank") != 4:
        raise ValueError("nominal orbit must retain rank four")
    if nominal.get("minimum_nullity") != 1 or nominal.get("maximum_nullity") != 1:
        raise ValueError("nominal orbit must retain nullity one")
    if nominal.get("maximum_closure_residual_m", float("inf")) > (
        POSITION_CLOSURE_TOLERANCE_M
    ):
        raise ValueError("nominal orbit violates position closure tolerance")
    if nominal.get("maximum_scaled_singular_value_spread_m", float("inf")) > 1e-12:
        raise ValueError("global phase symmetry lost singular-spectrum invariance")
    exact = report.get("exact_triangle_degeneracies", {})
    for boundary in ("lower", "upper"):
        audit = exact.get(f"{boundary}_rank_audit", {})
        if audit.get("rank") != 3 or audit.get("nullity") != 2:
            raise ValueError(
                f"{boundary} degeneracy must retain rank three/nullity two"
            )
    unit = report.get("unit_equivalence_control", {})
    if unit.get("rank_before") != unit.get("rank_after"):
        raise ValueError("equivalent length units changed rank")
    if (
        abs(
            unit.get("condition_number_before", 0.0)
            - unit.get("condition_number_after", 1.0)
        )
        > 1e-12
    ):
        raise ValueError("equivalent length units changed condition number")
    if not all(
        case.get("rejected") for case in report.get("impossible_geometry_controls", [])
    ):
        raise ValueError("an impossible geometry did not fail closed")
    killswitch = report.get("manufactured_matrix_killswitch", {})
    if killswitch.get("regular_rank") != 4 or killswitch.get("adverse_rank") != 3:
        raise ValueError("manufactured rank killswitch did not fire")
    if "does not establish" not in report.get("inference_boundary", ""):
        raise ValueError("inference boundary was promoted or removed")
    if report != build_report():
        raise ValueError("registered report differs from deterministic recomputation")
    return {
        "near_boundary_case_count": len(report["near_lower_boundary_sweep"]),
        "nominal_orbit_sample_count": int(nominal["sample_count"]),
        "phase_resolution_case_count": len(report["phase_resolution_controls"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Write or validate the registered evidence artifact."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    args = parser.parse_args(argv)
    if args.action == "write":
        REPORT_PATH.write_text(
            json.dumps(build_report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(REPORT_PATH)
        return 0
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_report(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
