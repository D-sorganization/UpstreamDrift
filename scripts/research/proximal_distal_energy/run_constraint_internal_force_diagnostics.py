"""Generate or validate constraint and internal-force diagnostics."""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.bilateral_wrench_identifiability import (
    LinearMapAudit,
    audit_linear_map,
    internal_axial_measurement,
)
from scripts.research.proximal_distal_energy.constraint_internal_force_diagnostics import (
    PlanarClosedLoopAudit,
    normalized_full_hand_wrench_map,
    normalized_point_force_wrench_map,
    planar_closed_loop_audit,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REPORT_PATH = ARTICLE / "data/constraint_internal_force_diagnostics.json"
SCHEMA_VERSION = "proximal-distal-constraint-internal-force-diagnostics/v2"
RELATIVE_TOLERANCE = 1e-12
WRENCH_REFERENCE_LENGTH_M = 0.10
ANGULAR_COORDINATE_SCALE_RAD = 1.0
TRANSLATION_COORDINATE_SCALE_M = 0.75
NEAR_COINCIDENT_SPAN_M = 1e-6
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/mechanism_ladder_study.json",
    "docs/research/proximal_distal_energy_transfer/data/"
    "bilateral_wrench_identifiability_study.json",
    "docs/research/proximal_distal_energy_transfer/data/"
    "subject_scaled_spatial_geometry.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _contacts(span_m: float) -> np.ndarray:
    return np.array(((-0.5 * span_m, 0.0, 0.0), (0.5 * span_m, 0.0, 0.0)))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _linear_payload(audit: LinearMapAudit, matrix: np.ndarray) -> dict[str, object]:
    residual = (
        float(np.max(np.abs(matrix @ audit.right_null_basis)))
        if audit.right_null_basis.size
        else 0.0
    )
    return {
        "matrix_shape": list(audit.matrix_shape),
        "maximum_normalized_nullspace_residual": residual,
        "minimum_nonzero_normalized_singular_value": (
            audit.minimum_nonzero_singular_value
        ),
        "normalized_nonzero_condition_number": _finite_or_none(
            audit.nonzero_condition_number
        ),
        "nullity": audit.nullity,
        "rank": audit.rank,
        "normalized_singular_values": [float(value) for value in audit.singular_values],
    }


def _planar_payload(audit: PlanarClosedLoopAudit) -> dict[str, object]:
    return {
        "angular_coordinate_scale_rad": audit.angular_coordinate_scale_rad,
        "maximum_scaled_nullspace_residual_m": (
            audit.maximum_scaled_nullspace_residual_m
        ),
        "nullity": audit.nullity,
        "rank": audit.rank,
        "scaled_condition_number": _finite_or_none(audit.scaled_condition_number),
        "scaled_singular_values_m": list(audit.scaled_singular_values_m),
        "smallest_scaled_singular_value_m": (audit.smallest_scaled_singular_value_m),
        "translation_coordinate_scale_m": audit.translation_coordinate_scale_m,
    }


def _planar_audit(
    grip_angle_rad: float,
    *,
    translation_scale_m: float = TRANSLATION_COORDINATE_SCALE_M,
) -> PlanarClosedLoopAudit:
    return planar_closed_loop_audit(
        lead_angle_rad=0.0,
        trail_angle_rad=0.0,
        grip_angle_rad=grip_angle_rad,
        angular_coordinate_scale_rad=ANGULAR_COORDINATE_SCALE_RAD,
        translation_coordinate_scale_m=translation_scale_m,
    )


def _source_authority() -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    loaded = {path: _load_source(path) for path in SOURCE_PATHS}
    authority = {
        path: {
            "schema_version": loaded[path]["schema_version"],
            "sha256": _sha256(ROOT / path),
        }
        for path in SOURCE_PATHS
    }
    return authority, loaded


def _planar_report(mechanism: dict[str, Any]) -> dict[str, object]:
    registered = mechanism["closed_loop_diagnostics"]
    if not (
        registered["minimum_rank"] == registered["maximum_rank"] == 4
        and registered["minimum_nullspace_dimension"]
        == registered["maximum_nullspace_dimension"]
        == 1
    ):
        raise ValueError("registered planar closure authority lost rank contract")
    regular = _planar_audit(0.0)
    singular = _planar_audit(np.pi / 2.0)
    near_cases = [
        {
            "grip_angle_offset_rad": offset,
            **_planar_payload(_planar_audit(np.pi / 2.0 - offset)),
        }
        for offset in (1e-2, 1e-4, 1e-6)
    ]
    scale_cases = [
        {
            "grip_angle_offset_rad": 1e-3,
            **_planar_payload(
                _planar_audit(
                    np.pi / 2.0 - 1e-3,
                    translation_scale_m=translation_scale,
                )
            ),
        }
        for translation_scale in (0.25, 0.75, 1.5)
    ]
    return {
        "coordinate_scale_contract": {
            "angular_coordinate_scale_rad": ANGULAR_COORDINATE_SCALE_RAD,
            "translation_coordinate_scale_m": TRANSLATION_COORDINATE_SCALE_M,
        },
        "exact_singular_case": {
            "geometry": {
                "grip_angle_rad": float(np.pi / 2.0),
                "lead_angle_rad": 0.0,
                "trail_angle_rad": 0.0,
            },
            "geometry_status": (
                "analytical_jacobian_alignment_only; position closure, anatomical "
                "feasibility, and human occurrence are not established"
            ),
            **_planar_payload(singular),
        },
        "inference_boundary": (
            "The right nullspace contains locally feasible normalized generalized "
            "velocity directions. Kinematic rank does not determine constraint "
            "multipliers, individual hand forces, force sign, or human strategy. "
            "Conditioning is meaningful only under the recorded coordinate scales."
        ),
        "near_singular_cases": near_cases,
        "regular_case": {
            "geometry": {
                "grip_angle_rad": 0.0,
                "lead_angle_rad": 0.0,
                "trail_angle_rad": 0.0,
            },
            **_planar_payload(regular),
        },
        "registered_201_geometry_authority": {
            "sample_count": registered["sample_count"],
            "minimum_rank": registered["minimum_rank"],
            "maximum_rank": registered["maximum_rank"],
            "minimum_nullspace_dimension": registered["minimum_nullspace_dimension"],
            "maximum_nullspace_dimension": registered["maximum_nullspace_dimension"],
            "conditioning_status": (
                "legacy raw-coordinate condition values are not interpreted here"
            ),
        },
        "scale_sensitivity_cases": scale_cases,
    }


def _bilateral_report(bilateral: dict[str, Any]) -> dict[str, object]:
    raw = bilateral["point_force_map"]
    if raw["rank"] != 5 or raw["nullity"] != 1:
        raise ValueError(
            "registered bilateral point-force authority lost rank contract"
        )
    span_cases = []
    for span in (0.0, 1e-6, 1e-4, 1e-2, 0.06, 0.12, 0.20, 0.30):
        matrix = normalized_point_force_wrench_map(
            _contacts(span),
            reference_length_m=WRENCH_REFERENCE_LENGTH_M,
        )
        audit = audit_linear_map(matrix, relative_tolerance=RELATIVE_TOLERANCE)
        span_cases.append({"span_m": span, **_linear_payload(audit, matrix)})
    near_matrix = normalized_point_force_wrench_map(
        _contacts(NEAR_COINCIDENT_SPAN_M),
        reference_length_m=WRENCH_REFERENCE_LENGTH_M,
    )
    threshold_cases = [
        {
            "relative_tolerance": tolerance,
            **_linear_payload(
                audit_linear_map(near_matrix, relative_tolerance=tolerance),
                near_matrix,
            ),
        }
        for tolerance in (1e-12, 1e-8, 1e-6)
    ]
    contacts = _contacts(0.20)
    point_matrix = normalized_point_force_wrench_map(
        contacts,
        reference_length_m=WRENCH_REFERENCE_LENGTH_M,
    )
    point_audit = audit_linear_map(
        point_matrix,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    full_matrix = normalized_full_hand_wrench_map(
        contacts,
        reference_length_m=WRENCH_REFERENCE_LENGTH_M,
    )
    full_audit = audit_linear_map(
        full_matrix,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    augmented_matrix = np.vstack((point_matrix, internal_axial_measurement(contacts)))
    augmented_audit = audit_linear_map(
        augmented_matrix,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    return {
        "augmented_axial_measurement_case": _linear_payload(
            augmented_audit,
            augmented_matrix,
        ),
        "coincident_contact_case": span_cases[0],
        "full_bilateral_six_axis_input_map": _linear_payload(
            full_audit,
            full_matrix,
        ),
        "inference_boundary": (
            "A net wrench does not identify individual point forces or bilateral "
            "six-axis hand wrenches. Null modes are sensing/allocation ambiguities, "
            "not evidence for a produced axial push-pull or free couple."
        ),
        "near_coincident_threshold_sensitivity": threshold_cases,
        "normalization": {
            "moment_rows_divided_by_reference_length_m": (WRENCH_REFERENCE_LENGTH_M),
            "six_axis_input_moment_scale": (
                "common_force_scale_times_reference_length"
            ),
            "purpose": (
                "remove force-versus-moment unit mixing; conditioning remains "
                "conditional on the declared length"
            ),
        },
        "registered_raw_map_authority": raw,
        "registered_span_case": {
            "span_m": 0.20,
            **_linear_payload(point_audit, point_matrix),
        },
        "span_cases": span_cases,
    }


def _spatial_report(spatial: dict[str, Any]) -> dict[str, object]:
    geometry = spatial["geometry_tests"]
    closure = spatial["closure_tests"]
    if geometry["constraint_jacobian_rank_values"] != [6]:
        raise ValueError("spatial prescribed-state constraint rank authority changed")
    return {
        "closure_tests": closure,
        "constraint_jacobian_rank_values": geometry["constraint_jacobian_rank_values"],
        "conditioning_status": (
            "source condition values are retained in the source artifact but are "
            "not compared here because its coordinate scale contract is absent"
        ),
        "inference_boundary": (
            "The prescribed spatial states fail the anatomical contact-closure "
            "gate. Full local row rank at an open state is not a feasible closed "
            "contact trajectory, an identified contact force, or human evidence."
        ),
        "model_tier": spatial["model_tier"],
    }


def build_report() -> dict[str, object]:
    """Build deterministic cross-tier nullspace and singular-geometry evidence."""

    authority, sources = _source_authority()
    return {
        "classification": "cross_tier_scaled_linear_map_and_singular_geometry_audit",
        "falsifiers": [
            "The regular planar closure map loses rank four or nullity one.",
            "The analytical singular planar geometry does not gain a velocity mode.",
            "Coincident point contacts retain moment observability above rank three.",
            "A separated point-force map loses its axial allocation null mode.",
            "A nullspace is promoted to observed individual-hand force.",
        ],
        "inference_status": {
            "constraint_multiplier_or_force_from_kinematics": "not_identified",
            "human_strategy": "untested",
            "individual_hand_force_from_net_wrench": "structurally_unidentifiable",
            "singular_geometry_mechanism": (
                "established_for_declared_scaled_analytical_maps"
            ),
        },
        "issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9027",
        "nullspace_semantics": {
            "full_hand_wrench_measurement_nullspace": {
                "domain": "two six-axis hand wrenches",
                "identifies_individual_hand_force": False,
                "meaning": "bilateral allocations invisible to one net club wrench",
            },
            "kinematic_velocity_nullspace": {
                "domain": "normalized generalized velocities satisfying J_c S zdot = 0",
                "identifies_individual_hand_force": False,
                "meaning": "locally feasible closed-loop motion directions",
            },
            "point_force_measurement_nullspace": {
                "domain": "two three-axis point forces mapped to one net club wrench",
                "identifies_individual_hand_force": False,
                "meaning": "equal-and-opposite axial allocation invisible to net wrench",
            },
        },
        "planar_closed_loop": _planar_report(sources[SOURCE_PATHS[0]]),
        "bilateral_point_force": _bilateral_report(sources[SOURCE_PATHS[1]]),
        "relative_rank_tolerance": RELATIVE_TOLERANCE,
        "schema_version": SCHEMA_VERSION,
        "source_authority": authority,
        "spatial_prescribed_state": _spatial_report(sources[SOURCE_PATHS[2]]),
    }


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Fail closed on scale drift, semantic conflation, or lost adverse cases."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    semantics = report.get("nullspace_semantics", {})
    if (
        semantics.get("kinematic_velocity_nullspace", {}).get(
            "identifies_individual_hand_force"
        )
        is not False
    ):
        raise ValueError("kinematic nullspace must not identify individual hand force")
    if report.get("inference_status", {}).get("human_strategy") != "untested":
        raise ValueError("human strategy must remain untested")
    planar = report.get("planar_closed_loop", {})
    expected_scale = {
        "angular_coordinate_scale_rad": ANGULAR_COORDINATE_SCALE_RAD,
        "translation_coordinate_scale_m": TRANSLATION_COORDINATE_SCALE_M,
    }
    if planar.get("coordinate_scale_contract") != expected_scale:
        raise ValueError("planar coordinate scale contract changed")
    if (
        planar.get("regular_case", {}).get("rank") != 4
        or planar.get("regular_case", {}).get("nullity") != 1
    ):
        raise ValueError("regular planar case must retain rank four/nullity one")
    if (
        planar.get("exact_singular_case", {}).get("rank") != 3
        or planar.get("exact_singular_case", {}).get("nullity") != 2
    ):
        raise ValueError("singular planar case must retain rank three/nullity two")
    if any(case.get("rank") != 4 for case in planar.get("scale_sensitivity_cases", [])):
        raise ValueError("positive coordinate scales must retain planar rank four")
    bilateral = report.get("bilateral_point_force", {})
    if (
        bilateral.get("coincident_contact_case", {}).get("rank") != 3
        or bilateral.get("coincident_contact_case", {}).get("nullity") != 3
    ):
        raise ValueError("coincident-contact case must retain rank three/nullity three")
    if (
        bilateral.get("registered_span_case", {}).get("rank") != 5
        or bilateral.get("registered_span_case", {}).get("nullity") != 1
    ):
        raise ValueError("separated-contact case must retain rank five/nullity one")
    if report != build_report():
        raise ValueError("registered report differs from deterministic recomputation")
    return {
        "planar_singular_rank": int(planar["exact_singular_case"]["rank"]),
        "source_authority_count": len(report["source_authority"]),
        "wrench_coincident_rank": int(bilateral["coincident_contact_case"]["rank"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Write or validate the registered diagnostic artifact."""

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
