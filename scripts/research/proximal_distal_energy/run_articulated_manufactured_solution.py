"""Generate independent articulated manufactured-solution evidence (#8910)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import sys
import tempfile
from typing import Any, Literal

import numpy as np

from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    require_robotics_pinocchio,
)
from scripts.research.proximal_distal_energy.articulated_manufactured_solution import (
    evaluate_manufactured_constrained_motion,
    evaluate_manufactured_free_body,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data"
OUTPUT = DATA / "articulated_manufactured_solution.json"
AUTHORITY_LOCK = (
    Path(__file__).resolve().parent
    / "requirements"
    / "articulated-authority-py311.lock"
)
AUTHORITY_PYTHON_VERSION = "3.11.15"
AUTHORITY_PROFILE = "articulated-manufactured-authority-py311-v1"
ROLLING_PROFILE = "articulated-manufactured-rolling-native-v1"
RecordProfile = Literal["authority", "rolling"]
_AUTHORITY_DISTRIBUTIONS = {
    "mujoco": "3.8.0",
    "numpy": "2.3.5",
    "pin": "3.8.0",
    "scipy": "1.15.3",
}
_CROSS_ENGINE_TOLERANCE = 1.0e-8
_POSITION_TOLERANCE_M = 1.0e-10
_VELOCITY_TOLERANCE_M_S = 1.0e-10
_VIRTUAL_POWER_TOLERANCE_W = 1.0e-9
_ROLLING_COMPATIBILITY_ABSOLUTE_TOLERANCE_BY_FIELD = {
    "free_body.inverse_dynamics_relative_error.lagrange_mujoco": 1.0e-8,
    "free_body.inverse_dynamics_relative_error.lagrange_pinocchio": 1.0e-8,
    "free_body.inverse_dynamics_relative_error.mujoco_pinocchio": 1.0e-8,
    "free_body.inverse_dynamics_relative_error.maximum": 1.0e-8,
    "free_body.integration_step_error_rad.0.0005": 1.0e-8,
    "free_body.integration_step_error_rad.0.001": 1.0e-8,
    "free_body.integration_step_error_rad.0.002": 1.0e-8,
    "free_body.richardson_orders.0": 1.0e-3,
    "free_body.richardson_orders.1": 1.0e-3,
    "free_body.gravity_free_zero_torque_relative_drift.linear_momentum": 1.0e-8,
    "free_body.gravity_free_zero_torque_relative_drift.angular_momentum": 1.0e-8,
    "free_body.gravity_free_zero_torque_relative_drift.kinetic_energy": 1.0e-8,
    "constrained_motion.position_residual_m": _POSITION_TOLERANCE_M,
    "constrained_motion.velocity_residual_m_s": _VELOCITY_TOLERANCE_M_S,
    "constrained_motion.virtual_power_residual_w": _VIRTUAL_POWER_TOLERANCE_W,
    "constrained_motion.multiplier_relative_residual": 1.0e-8,
    "constrained_motion.cross_engine_multiplier_relative_residual": 1.0e-8,
    "constrained_motion.equilibrium_relative_residual": 1.0e-8,
}
_REQUIRED_GATE_TOLERANCE_FIELDS = (
    "inverse_dynamics_relative_tolerance",
    "conservation_relative_tolerance",
    "cross_engine_relative_tolerance",
    "constraint_position_tolerance_m",
    "constraint_velocity_tolerance_m_s",
    "constraint_virtual_power_tolerance_w",
)
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py",
    "scripts/research/proximal_distal_energy/articulated_manufactured_solution.py",
    "scripts/research/proximal_distal_energy/requirements/articulated-authority-py311.in",
    "scripts/research/proximal_distal_energy/run_articulated_manufactured_solution.py",
    "scripts/research/proximal_distal_energy/register_articulated_manufactured_solution_claims.py",
    "scripts/research/proximal_distal_energy/spatial_full_body.py",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
    "tests/research/test_articulated_manufactured_solution.py",
    "tests/research/test_articulated_manufactured_hybrid_authority_red.py",
    "tests/research/test_articulated_manufactured_hybrid_policy_completeness_red.py",
    "tests/research/test_articulated_manufactured_hybrid_semantics_red.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_record_bytes(record: dict[str, Any]) -> bytes:
    """Return canonical UTF-8/LF JSON and reject every non-finite number."""

    try:
        text = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except ValueError as error:
        raise ValueError("record contains a non-finite numeric value") from error
    return f"{text}\n".encode()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required distribution is unavailable: {name}") from error


def validate_authority_environment() -> None:
    """Fail unless execution is the exact governed CPython/Linux stack."""

    actual_platform = (platform.system(), platform.machine().lower())
    if actual_platform != ("Linux", "x86_64"):
        raise RuntimeError("authority requires Linux x86_64")
    if platform.python_implementation() != "CPython":
        raise RuntimeError("authority requires CPython")
    expected_python = tuple(int(part) for part in AUTHORITY_PYTHON_VERSION.split("."))
    if sys.version_info[:3] != expected_python:
        raise RuntimeError(
            f"authority requires exact Python patch {AUTHORITY_PYTHON_VERSION}"
        )
    mismatches: dict[str, tuple[str, str]] = {}
    for name, expected in _AUTHORITY_DISTRIBUTIONS.items():
        actual = _distribution_version(name)
        if actual != expected:
            mismatches[name] = (expected, actual)
    if mismatches:
        raise RuntimeError(
            f"authority dependency versions do not match lock: {mismatches}"
        )


def _execution_profile(profile: RecordProfile) -> dict[str, Any]:
    if profile == "authority":
        validate_authority_environment()
        return {
            "id": AUTHORITY_PROFILE,
            "publication_authority": "authoritative",
            "publication_eligible": True,
            "python_minor": "3.11",
            "platform": "linux-x86_64",
            "dependency_lock": {
                "path": AUTHORITY_LOCK.relative_to(ROOT).as_posix(),
                "sha256": _sha256(AUTHORITY_LOCK),
            },
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": _distribution_version("numpy"),
                "mujoco": _distribution_version("mujoco"),
                "pinocchio": _distribution_version("pin"),
            },
        }
    if profile == "rolling":
        return {
            "id": ROLLING_PROFILE,
            "publication_authority": "non_authoritative_compatibility_only",
            "publication_eligible": False,
            "runtime_versions": {
                "python": sys.version.split()[0],
                "numpy": _distribution_version("numpy"),
                "mujoco": _distribution_version("mujoco"),
                "pin": _distribution_version("pin"),
            },
        }
    raise ValueError(f"unsupported execution profile: {profile!r}")


def _free_record(result: Any) -> dict[str, Any]:
    return {
        "all_gates_pass": result.closed_form_check_passed,
        "independent_engine_difference_detected": (
            result.independent_engine_difference_detected
        ),
        "inverse_dynamics_relative_error": {
            "lagrange_mujoco": result.lagrange_mujoco_relative_error,
            "lagrange_pinocchio": result.lagrange_pinocchio_relative_error,
            "mujoco_pinocchio": result.mujoco_pinocchio_relative_error,
            "maximum": result.inverse_dynamics_residual,
        },
        "integration_step_error_rad": {
            str(step): error
            for step, error in sorted(result.integration_step_errors.items())
        },
        "richardson_orders": list(result.richardson_orders),
        "gravity_free_zero_torque_relative_drift": {
            "linear_momentum": result.linear_momentum_conservation_error,
            "angular_momentum": result.angular_momentum_conservation_error,
            "kinetic_energy": result.mechanical_energy_conservation_error,
        },
    }


def _constrained_record(result: Any) -> dict[str, Any]:
    return {
        "all_gates_pass": result.closed_form_check_passed,
        "independent_engine_difference_detected": (
            result.independent_engine_difference_detected
        ),
        "position_residual_m": result.constraint_residual,
        "velocity_residual_m_s": result.constraint_velocity_residual,
        "virtual_power_residual_w": result.constraint_virtual_power_w,
        "multiplier_relative_residual": result.lagrange_multiplier_residual,
        "cross_engine_multiplier_relative_residual": (
            result.action_reaction_residual_n
        ),
        "equilibrium_relative_residual": result.equilibrium_residual,
    }


def build_record(profile: RecordProfile = "authority") -> dict[str, Any]:
    """Execute the registered controls and return a release record."""

    execution_profile = _execution_profile(profile)
    import mujoco
    import pinocchio as pin

    pinocchio_version = require_robotics_pinocchio(pin)
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    free = evaluate_manufactured_free_body(
        model, q, duration_s=0.01, time_steps_s=(0.002, 0.001, 0.0005)
    )
    constrained = evaluate_manufactured_constrained_motion(
        model,
        q,
        duration_s=0.01,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
    )
    return {
        "schema_version": "1.1.0",
        "study_id": "articulated-manufactured-solution-independent-v1",
        "classification": "synthetic_numerical_verification_not_human_evidence",
        "execution_profile": execution_profile,
        "model": {
            "canonical_sha256": model.canonical_hash,
            "coordinate_count": model.nq,
            "profile": default_synthetic_profiles()[0].profile_id,
            "closed_state_index": [0, 6],
        },
        "engines": {
            "analytical": "lagrange_christoffel_finite_difference_mass_gradient",
            "mujoco": str(mujoco.__version__),
            "pinocchio": pinocchio_version,
        },
        "design": {
            "duration_s": 0.01,
            "time_steps_s": [0.002, 0.001, 0.0005],
            "registered_richardson_order_interval": [0.9, 1.1],
            "inverse_dynamics_relative_tolerance": 0.05,
            "conservation_relative_tolerance": 0.02,
            "cross_engine_relative_tolerance": _CROSS_ENGINE_TOLERANCE,
            "constraint_position_tolerance_m": _POSITION_TOLERANCE_M,
            "constraint_velocity_tolerance_m_s": _VELOCITY_TOLERANCE_M_S,
            "constraint_virtual_power_tolerance_w": _VIRTUAL_POWER_TOLERANCE_W,
            "rolling_compatibility_absolute_tolerance_by_field": (
                dict(_ROLLING_COMPATIBILITY_ABSOLUTE_TOLERANCE_BY_FIELD)
            ),
            "conservation_scope": "gravity_free_zero_torque_free_floating_club_subtree",
            "killswitch": "add_10_nm_to_mujoco_inverse_and_require_gate_failure",
        },
        "free_body": _free_record(free),
        "constrained_motion": _constrained_record(constrained),
        "all_gates_pass": bool(
            free.closed_form_check_passed and constrained.closed_form_check_passed
        ),
        "limitations": [
            "The trajectories are manufactured and are not measured golf swings.",
            "The conservation rollout isolates the free club subtree because the pelvis tree is world-supported.",
            "Agreement verifies the declared model and operators, not anatomy or coaching strategy.",
        ],
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCE_PATHS},
    }


def _registered_gates(record: dict[str, Any]) -> tuple[bool, bool, bool]:
    free_body = record.get("free_body")
    constrained = record.get("constrained_motion")
    if not isinstance(free_body, dict) or not isinstance(constrained, dict):
        raise ValueError("semantic record is missing registered gate sections")
    return (
        record.get("all_gates_pass") is True,
        free_body.get("all_gates_pass") is True,
        constrained.get("all_gates_pass") is True,
    )


def _mapping(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"semantic record is missing mapping: {field}")
    return value


def _finite(mapping: dict[str, Any], field: str) -> float:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"semantic field is not numeric: {field}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"semantic field is not finite: {field}")
    return numeric


def _finite_value(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"semantic field is not numeric: {field}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"semantic field is not finite: {field}")
    return numeric


def _positive_limit(design: dict[str, Any], field: str) -> float:
    if field not in design:
        raise ValueError(f"required semantic tolerance is missing: {field}")
    value = _finite(design, field)
    if value <= 0.0:
        raise ValueError(f"semantic tolerance must be positive: {field}")
    return value


def _nonnegative_values(mapping: dict[str, Any], prefix: str) -> None:
    for field, value in mapping.items():
        if _finite_value(value, f"{prefix}.{field}") < 0.0:
            raise ValueError(f"semantic field must be nonnegative: {prefix}.{field}")


def _integration_errors(free: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    raw = _mapping(free, "integration_step_error_rad")
    pairs: list[tuple[float, float]] = []
    for step, error in raw.items():
        try:
            numeric_step = float(step)
        except (TypeError, ValueError) as exception:
            raise ValueError("semantic integration step is not numeric") from exception
        numeric_error = _finite_value(error, f"integration_step_error_rad.{step}")
        if numeric_step <= 0.0 or numeric_error < 0.0:
            raise ValueError("semantic integration evidence must be nonnegative")
        pairs.append((numeric_step, numeric_error))
    if not pairs:
        raise ValueError("semantic record has no integration evidence")
    ordered = tuple(sorted(pairs))
    if any(
        finer_error > coarser_error
        for (_, finer_error), (_, coarser_error) in zip(
            ordered, ordered[1:], strict=False
        )
    ):
        raise ValueError("semantic integration errors are not monotonic")
    return ordered


def _validate_numeric_gates(
    record: dict[str, Any], *, authority_maximum: float | None = None
) -> float:
    design = _mapping(record, "design")
    free = _mapping(record, "free_body")
    inverse = _mapping(free, "inverse_dynamics_relative_error")
    drift = _mapping(free, "gravity_free_zero_torque_relative_drift")
    constrained = _mapping(record, "constrained_motion")
    limits = {
        field: _positive_limit(design, field)
        for field in _REQUIRED_GATE_TOLERANCE_FIELDS
    }
    inverse_limit = limits["inverse_dynamics_relative_tolerance"]
    conservation_limit = limits["conservation_relative_tolerance"]
    cross_engine_limit = limits["cross_engine_relative_tolerance"]
    position_limit = limits["constraint_position_tolerance_m"]
    velocity_limit = limits["constraint_velocity_tolerance_m_s"]
    virtual_power_limit = limits["constraint_virtual_power_tolerance_w"]
    order_bounds = design.get("registered_richardson_order_interval")
    orders = free.get("richardson_orders")
    if not isinstance(order_bounds, list) or len(order_bounds) != 2:
        raise ValueError("semantic record has invalid Richardson bounds")
    if not isinstance(orders, list) or not orders:
        raise ValueError("semantic record has no Richardson evidence")
    inverse_components = {
        name: _finite(inverse, name) for name in inverse if name != "maximum"
    }
    if not inverse_components or "maximum" not in inverse:
        raise ValueError("semantic inverse-dynamics maximum is missing")
    _nonnegative_values(inverse, "inverse_dynamics_relative_error")
    maximum = _finite(inverse, "maximum")
    computed_maximum = max(inverse_components.values())
    if maximum != computed_maximum and maximum != authority_maximum:
        raise ValueError("semantic inverse-dynamics maximum is inconsistent")
    _nonnegative_values(drift, "gravity_free_zero_torque_relative_drift")
    drift_values = tuple(_finite(drift, name) for name in drift)
    order_values = tuple(
        _finite_value(value, f"richardson_orders.{index}")
        for index, value in enumerate(orders)
    )
    if any(value < 0.0 for value in order_values):
        raise ValueError("semantic Richardson order must be nonnegative")
    _integration_errors(free)
    constrained_numeric = {
        name: value
        for name, value in constrained.items()
        if name not in {"all_gates_pass", "independent_engine_difference_detected"}
    }
    _nonnegative_values(constrained_numeric, "constrained_motion")
    constrained_values = (
        _finite(constrained, "multiplier_relative_residual"),
        _finite(constrained, "equilibrium_relative_residual"),
    )
    passed = (
        free.get("independent_engine_difference_detected") is True
        and constrained.get("independent_engine_difference_detected") is True
        and maximum < inverse_limit
        and max(drift_values) < conservation_limit
        and all(
            float(order_bounds[0]) <= value <= float(order_bounds[1])
            for value in order_values
        )
        and max(constrained_values) < inverse_limit
        and _finite(constrained, "cross_engine_multiplier_relative_residual")
        < cross_engine_limit
        and _finite(constrained, "position_residual_m") < position_limit
        and _finite(constrained, "velocity_residual_m_s") < velocity_limit
        and _finite(constrained, "virtual_power_residual_w") < virtual_power_limit
    )
    if not passed:
        raise ValueError("semantic numeric gate comparison failed")
    return maximum


def _require_profile(record: dict[str, Any], profile_id: str, authority: str) -> None:
    profile = record.get("execution_profile")
    if not isinstance(profile, dict):
        raise ValueError("semantic record is missing execution provenance")
    if profile.get("id") != profile_id:
        raise ValueError(f"unexpected execution profile: {profile.get('id')}")
    if profile.get("publication_authority") != authority:
        raise ValueError("execution profile has invalid publication authority")
    expected_eligibility = authority == "authoritative"
    if profile.get("publication_eligible") is not expected_eligibility:
        raise ValueError("execution profile has invalid publication eligibility")


def _compatibility_tolerances(record: dict[str, Any]) -> dict[str, float]:
    design = _mapping(record, "design")
    field = "rolling_compatibility_absolute_tolerance_by_field"
    if field not in design:
        raise ValueError("required semantic compatibility policy is missing")
    raw = design[field]
    if not isinstance(raw, dict):
        raise ValueError("semantic compatibility tolerance policy is malformed")
    required = set(_ROLLING_COMPATIBILITY_ABSOLUTE_TOLERANCE_BY_FIELD)
    actual = set(raw)
    if actual != required:
        missing = sorted(required - actual)
        unknown = sorted(str(item) for item in actual - required)
        raise ValueError(
            "semantic compatibility policy is incomplete: "
            f"missing={missing}, unknown={unknown}"
        )
    tolerances: dict[str, float] = {}
    for field, value in raw.items():
        if not isinstance(field, str) or not field:
            raise ValueError("semantic compatibility tolerance path is malformed")
        numeric = _finite_value(value, f"compatibility tolerance {field}")
        if numeric <= 0.0:
            raise ValueError("semantic compatibility tolerance must be positive")
        tolerances[field] = numeric
    return tolerances


def _compare_values(
    authority: object,
    rolling: object,
    path: str,
    tolerances: dict[str, float],
) -> None:
    if isinstance(authority, dict) and isinstance(rolling, dict):
        if authority.keys() != rolling.keys():
            raise ValueError(f"semantic compatibility fields differ: {path}")
        for field in authority:
            child = f"{path}.{field}" if path else str(field)
            _compare_values(authority[field], rolling[field], child, tolerances)
        return
    if isinstance(authority, list) and isinstance(rolling, list):
        if len(authority) != len(rolling):
            raise ValueError(f"semantic compatibility sequence differs: {path}")
        for index, authority_value in enumerate(authority):
            child = f"{path}.{index}"
            _compare_values(authority_value, rolling[index], child, tolerances)
        return
    numeric_types = (int, float)
    authority_is_numeric = isinstance(authority, numeric_types) and not isinstance(
        authority, bool
    )
    rolling_is_numeric = isinstance(rolling, numeric_types) and not isinstance(
        rolling, bool
    )
    if authority_is_numeric or rolling_is_numeric:
        authority_value = _finite_value(authority, path)
        rolling_value = _finite_value(rolling, path)
        tolerance = tolerances.get(path, 0.0)
        if abs(rolling_value - authority_value) > tolerance:
            raise ValueError(
                f"semantic compatibility tolerance exceeded at {path}: "
                f"authority={authority_value}, rolling={rolling_value}, "
                f"tolerance={tolerance}"
            )
        return
    if authority != rolling:
        raise ValueError(f"semantic compatibility value differs: {path}")


def compare_semantic_evidence(
    authority: dict[str, Any], rolling: dict[str, Any]
) -> dict[str, Any]:
    """Compare scientific identity and gates while excluding runtime provenance."""

    _require_profile(authority, AUTHORITY_PROFILE, "authoritative")
    _require_profile(
        rolling,
        ROLLING_PROFILE,
        "non_authoritative_compatibility_only",
    )
    identity_fields = (
        "schema_version",
        "study_id",
        "classification",
        "model",
        "design",
        "limitations",
        "source_sha256",
    )
    mismatches = [
        field for field in identity_fields if authority.get(field) != rolling.get(field)
    ]
    if mismatches:
        raise ValueError(f"semantic identity mismatch: {mismatches}")
    authority_gates = _registered_gates(authority)
    rolling_gates = _registered_gates(rolling)
    if not all(authority_gates) or not all(rolling_gates):
        raise ValueError("semantic gate comparison failed")
    tolerances = _compatibility_tolerances(authority)
    if tolerances != _compatibility_tolerances(rolling):
        raise ValueError("semantic compatibility tolerance policies differ")
    authority_maximum = _validate_numeric_gates(authority)
    _validate_numeric_gates(rolling, authority_maximum=authority_maximum)
    for field in ("free_body", "constrained_motion"):
        authority_section = authority[field]
        rolling_section = rolling[field]
        if not isinstance(authority_section, dict) or not isinstance(
            rolling_section, dict
        ):
            raise ValueError(f"semantic section is malformed: {field}")
        _compare_values(authority_section, rolling_section, field, tolerances)
    return {
        "all_registered_gates_pass": True,
        "scientific_identity_matches": True,
        "authority_profile": authority.get("execution_profile"),
        "rolling_profile": rolling.get("execution_profile"),
    }


def write_record(path: Path = OUTPUT, *, profile: RecordProfile = "authority") -> Path:
    """Write the deterministic evidence record."""

    if profile not in ("authority", "rolling"):
        raise ValueError(f"unsupported execution profile: {profile!r}")
    if profile == "rolling" and path.resolve() == OUTPUT.resolve():
        raise ValueError("rolling evidence cannot replace the authoritative record")
    record = build_record(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_record_bytes(record)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--profile", choices=("authority", "rolling"), default="authority"
    )
    parser.add_argument("--compare-committed", action="store_true")
    arguments = parser.parse_args()
    if arguments.compare_committed and arguments.output.resolve() == OUTPUT.resolve():
        parser.error("--compare-committed requires a temporary --output path")
    output = write_record(arguments.output, profile=arguments.profile)
    if arguments.compare_committed:
        generated = json.loads(output.read_text(encoding="utf-8"))
        committed = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if (
            arguments.profile == "authority"
            and output.read_bytes() != OUTPUT.read_bytes()
        ):
            raise SystemExit("authoritative bytes differ from committed record")
        compare_semantic_evidence(committed, generated)
    print(output)


if __name__ == "__main__":
    main()
