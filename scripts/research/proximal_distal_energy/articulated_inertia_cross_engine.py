"""Qualify the closed subject-scaled articulated inertia in two native engines.

This module is a common-state dynamics gate between the kinematic closure atlas
and forward bilateral contact.  It does not advance contact, infer muscle
action, or establish anatomy.  MuJoCo and Pinocchio independently assemble the
same immutable 20-coordinate tree and evaluate mass, bias, and inverse dynamics
at every closed state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    mujoco_inverse_dynamics,
    mujoco_mass_matrix_and_bias,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

FloatArray = NDArray[np.float64]
_MINIMUM_PINOCCHIO_VERSION = (2, 6)
_PINOCCHIO_REQUIRED_API = (
    "Model",
    "SE3",
    "Inertia",
    "crba",
    "nonLinearEffects",
    "rnea",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py",
    "scripts/research/proximal_distal_energy/run_articulated_inertia_cross_engine.py",
    "scripts/research/proximal_distal_energy/spatial_full_body.py",
    "scripts/research/proximal_distal_energy/subject_scaled_spatial_geometry.py",
)


def require_robotics_pinocchio(pin: Any) -> str:
    """Return the qualified engine version or reject a module-name collision.

    Postcondition: the returned version identifies robotics Pinocchio >= 2.6 and
    the supplied module exposes the native dynamics API used by this program.
    """

    version = getattr(pin, "__version__", None)
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:[+.-].*)?", str(version))
    if match is None or tuple(map(int, match.groups(default="0")[:2])) < (
        _MINIMUM_PINOCCHIO_VERSION
    ):
        raise RuntimeError(
            "robotics Pinocchio >= 2.6 is required; install the 'pin' package, "
            "not the unrelated PyPI 'pinocchio' package"
        )
    missing = [name for name in _PINOCCHIO_REQUIRED_API if not hasattr(pin, name)]
    if missing:
        raise RuntimeError(
            "robotics Pinocchio is missing required robotics API: " + ", ".join(missing)
        )
    return str(version)


@dataclass(frozen=True, slots=True)
class ArticulatedInertiaConfig:
    """Preoutcome equivalence and positive-definiteness gates."""

    mass_matrix_relative_tolerance: float = 1.0e-9
    bias_relative_tolerance: float = 1.0e-9
    inverse_dynamics_relative_tolerance: float = 1.0e-9
    symmetry_absolute_tolerance: float = 1.0e-10
    minimum_eigenvalue_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        for name in (
            "mass_matrix_relative_tolerance",
            "bias_relative_tolerance",
            "inverse_dynamics_relative_tolerance",
            "symmetry_absolute_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.minimum_eigenvalue_tolerance)
            or self.minimum_eigenvalue_tolerance <= 0.0
        ):
            raise ValueError("minimum_eigenvalue_tolerance must be finite and positive")


def finite_difference_kinematics(
    position: FloatArray, time_s: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return second-order finite-difference velocity and acceleration."""

    position = np.asarray(position, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if position.ndim != 2 or position.shape[0] < 3:
        raise ValueError("position must have shape (at least 3, coordinates)")
    if time_s.shape != (position.shape[0],) or np.any(~np.isfinite(time_s)):
        raise ValueError("time_s must be finite and match the position samples")
    if np.any(~np.isfinite(position)) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("position must be finite and time_s strictly increasing")
    velocity = np.gradient(position, time_s, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, time_s, axis=0, edge_order=2)
    return np.asarray(velocity), np.asarray(acceleration)


def _pinocchio_joint_model(pin: Any, kind: str, axis: FloatArray) -> Any:
    key = (kind, tuple(np.asarray(axis, dtype=float)))
    constructors = {
        ("revolute", (1.0, 0.0, 0.0)): pin.JointModelRX,
        ("revolute", (0.0, 1.0, 0.0)): pin.JointModelRY,
        ("revolute", (0.0, 0.0, 1.0)): pin.JointModelRZ,
        ("prismatic", (1.0, 0.0, 0.0)): pin.JointModelPX,
        ("prismatic", (0.0, 1.0, 0.0)): pin.JointModelPY,
        ("prismatic", (0.0, 0.0, 1.0)): pin.JointModelPZ,
    }
    try:
        return constructors[key]()
    except KeyError as error:
        raise ValueError(f"unsupported articulated joint convention: {key}") from error


def build_pinocchio_articulated_model(pin: Any, model: SpatialModel) -> Any:
    """Build the canonical scalar-joint tree in robotics Pinocchio."""

    require_robotics_pinocchio(pin)
    native = pin.Model()
    native.gravity.linear = np.array([0.0, 0.0, -9.80665])
    joint_ids: list[int] = []
    for joint in model.joints:
        parent = 0 if joint.parent < 0 else joint_ids[joint.parent]
        placement = pin.SE3(np.eye(3), np.asarray(joint.offset_m, dtype=float))
        joint_id = native.addJoint(
            parent,
            _pinocchio_joint_model(pin, joint.kind, joint.axis),
            placement,
            joint.name,
        )
        joint_ids.append(joint_id)
    for body in model.bodies:
        inertia = pin.Inertia(
            body.mass_kg,
            np.asarray(body.com_offset_m, dtype=float),
            np.eye(3) * (0.4 * body.mass_kg * body.radius_m**2),
        )
        native.appendBodyToJoint(joint_ids[body.joint], inertia, pin.SE3.Identity())
    if native.nq != model.nq or native.nv != model.nq:
        raise RuntimeError("Pinocchio coordinate dimensions do not match the contract")
    return native


def _relative_error(left: FloatArray, right: FloatArray) -> tuple[float, float]:
    absolute = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    scale = max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return absolute, absolute / scale


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_closed_state_authority() -> tuple[NDArray[Any], ...]:
    """Load and qualify the immutable closed-state input authority."""

    with np.load(DATA_DIR / "subject_scaled_closed_contact.npz") as source:
        time_s = np.asarray(source["time_s"], dtype=float)
        profile_index = np.asarray(source["case_profile_index"], dtype=int)
        grip_span_m = np.asarray(source["case_grip_span_m"], dtype=float)
        solution_q = np.asarray(source["solution_q"], dtype=float)
        feasible = np.asarray(source["feasible"], dtype=bool)
    if solution_q.shape != (18, 13, 20) or not np.all(feasible):
        raise RuntimeError("the closed-state authority is incomplete or infeasible")
    return time_s, profile_index, grip_span_m, solution_q


def _evaluate_native_operators(
    pin: Any,
    time_s: NDArray[Any],
    profile_index: NDArray[Any],
    solution_q: NDArray[Any],
) -> tuple[dict[str, NDArray[Any]], list[str], list[str]]:
    """Evaluate both engines and return atomic error arrays and model identity."""

    shape = solution_q.shape[:2]
    metrics = {
        "mass_abs": np.empty(shape),
        "mass_rel": np.empty(shape),
        "bias_abs": np.empty(shape),
        "bias_rel": np.empty(shape),
        "inverse_abs": np.empty(shape),
        "inverse_rel": np.empty(shape),
        "symmetry": np.empty((*shape, 2)),
        "eigenvalue": np.empty((*shape, 2)),
    }
    profiles = default_synthetic_profiles()
    model_hashes: list[str] = []
    joint_sequence: list[str] = []
    nonlinear_effects = pin.nonLinearEffects  # type: ignore[attr-defined]

    for case in range(shape[0]):
        model, _ = build_subject_scaled_model(profiles[profile_index[case]])
        native = build_pinocchio_articulated_model(pin, model)
        native_data = native.createData()
        model_hashes.append(model.canonical_hash)
        joint_sequence = [joint.name for joint in model.joints]
        velocity, acceleration = finite_difference_kinematics(solution_q[case], time_s)
        for sample in range(shape[1]):
            q = solution_q[case, sample]
            qd = velocity[sample]
            qdd = acceleration[sample]
            matrix_m, bias_m = mujoco_mass_matrix_and_bias(model, q, qd)
            matrix_p = np.asarray(pin.crba(native, native_data, q)).copy()
            bias_p = np.asarray(nonlinear_effects(native, native_data, q, qd)).copy()
            inverse_m = mujoco_inverse_dynamics(model, q, qd, qdd, np.zeros(model.nq))
            inverse_p = np.asarray(pin.rnea(native, native_data, q, qd, qdd)).copy()
            metrics["mass_abs"][case, sample], metrics["mass_rel"][case, sample] = (
                _relative_error(matrix_m, matrix_p)
            )
            metrics["bias_abs"][case, sample], metrics["bias_rel"][case, sample] = (
                _relative_error(bias_m, bias_p)
            )
            (
                metrics["inverse_abs"][case, sample],
                metrics["inverse_rel"][case, sample],
            ) = _relative_error(inverse_m, inverse_p)
            metrics["symmetry"][case, sample] = (
                np.max(np.abs(matrix_m - matrix_m.T)),
                np.max(np.abs(matrix_p - matrix_p.T)),
            )
            metrics["eigenvalue"][case, sample] = (
                np.min(np.linalg.eigvalsh(matrix_m)),
                np.min(np.linalg.eigvalsh(matrix_p)),
            )
    return metrics, model_hashes, joint_sequence


def _build_inertia_record(
    config: ArticulatedInertiaConfig,
    *,
    engine_versions: dict[str, str],
    solution_q: NDArray[Any],
    grip_span_m: NDArray[Any],
    metrics: dict[str, NDArray[Any]],
    gates: NDArray[Any],
    model_hashes: list[str],
    joint_sequence: list[str],
) -> dict[str, Any]:
    """Build the governed record without recomputing native operators."""

    shape = solution_q.shape[:2]
    return {
        "schema_version": "articulated-inertia-cross-engine/v1",
        "study_id": "subject-scaled-closed-state-articulated-inertia-parity",
        "model_tier": "common_state_articulated_rigid_body_dynamics",
        "design": {
            "profile_count": len(default_synthetic_profiles()),
            "grip_span_count": int(np.unique(grip_span_m).size),
            "case_count": int(shape[0]),
            "samples_per_case": int(shape[1]),
            "state_count": int(np.prod(shape)),
            "coordinate_count": int(solution_q.shape[2]),
            "kinematic_derivatives": (
                "second-order finite differences on the 13-sample closed-state path"
            ),
            "common_state_not_forward_trajectory": True,
        },
        "engines": {
            "mujoco": {
                "version": engine_versions["mujoco"],
                "operators": ["mj_fullM", "qfrc_bias", "mj_inverse"],
            },
            "pinocchio": {
                "version": engine_versions["pinocchio"],
                "operators": ["crba", "nonLinearEffects", "rnea"],
            },
        },
        "coordinate_contract": {
            "model_hash_count": len(set(model_hashes)),
            "case_model_hashes": model_hashes,
            "joint_sequence": joint_sequence,
            "gravity_m_s2": [0.0, 0.0, -9.80665],
            "spherical_body_inertia": "2/5 mass radius^2 in every principal axis",
        },
        "tolerances": asdict(config),
        "results": {
            "maximum_mass_matrix_absolute_error": float(np.max(metrics["mass_abs"])),
            "maximum_mass_matrix_relative_error": float(np.max(metrics["mass_rel"])),
            "maximum_bias_absolute_error": float(np.max(metrics["bias_abs"])),
            "maximum_bias_relative_error": float(np.max(metrics["bias_rel"])),
            "maximum_inverse_dynamics_absolute_error": float(
                np.max(metrics["inverse_abs"])
            ),
            "maximum_inverse_dynamics_relative_error": float(
                np.max(metrics["inverse_rel"])
            ),
            "maximum_symmetry_residual": float(np.max(metrics["symmetry"])),
            "minimum_mass_matrix_eigenvalue": float(np.min(metrics["eigenvalue"])),
            "failed_state_count": int(np.count_nonzero(~gates)),
            "all_registered_gates_passed": bool(np.all(gates)),
        },
        "claim_boundary": {
            "supported": (
                "native articulated mass, bias, and inverse-dynamics operators "
                "agree on the declared closed common states"
            ),
            "forward_contact": "not_established",
            "scapulothoracic_anatomy": "not_established",
            "distributed_grip_or_shaft": "not_established",
            "muscle_action": "not_inferred_from_net_joint_dynamics",
            "human_strategy": "untested",
        },
        "next_gate": (
            "apply bilateral compliant contact to this articulated tree, then "
            "repeat convergence, contact-loss, adverse-load, power, and energy gates"
        ),
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in SOURCE_PATHS},
    }


def run_articulated_inertia_atlas(
    config: ArticulatedInertiaConfig = ArticulatedInertiaConfig(),
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Evaluate every closed state in native MuJoCo and Pinocchio."""

    if not isinstance(config, ArticulatedInertiaConfig):
        raise TypeError("config must be an ArticulatedInertiaConfig")
    try:
        import mujoco
        import pinocchio as pin
    except ImportError as error:  # pragma: no cover - native runtime gate
        raise RuntimeError("MuJoCo and robotics Pinocchio are required") from error
    pin_version = require_robotics_pinocchio(pin)
    time_s, profile_index, grip_span_m, solution_q = _load_closed_state_authority()
    metrics, model_hashes, joint_sequence = _evaluate_native_operators(
        pin, time_s, profile_index, solution_q
    )
    gates = (metrics["mass_rel"] <= config.mass_matrix_relative_tolerance) & (
        metrics["bias_rel"] <= config.bias_relative_tolerance
    )
    gates &= metrics["inverse_rel"] <= config.inverse_dynamics_relative_tolerance
    gates &= np.all(metrics["symmetry"] <= config.symmetry_absolute_tolerance, axis=2)
    gates &= np.all(
        metrics["eigenvalue"] >= config.minimum_eigenvalue_tolerance, axis=2
    )
    arrays: dict[str, NDArray[Any]] = {
        "time_s": time_s,
        "case_profile_index": profile_index,
        "case_grip_span_m": grip_span_m,
        "mass_matrix_absolute_error": metrics["mass_abs"],
        "mass_matrix_relative_error": metrics["mass_rel"],
        "bias_absolute_error": metrics["bias_abs"],
        "bias_relative_error": metrics["bias_rel"],
        "inverse_dynamics_absolute_error": metrics["inverse_abs"],
        "inverse_dynamics_relative_error": metrics["inverse_rel"],
        "mass_matrix_symmetry_residual": metrics["symmetry"],
        "minimum_mass_matrix_eigenvalue": metrics["eigenvalue"],
        "all_gates_passed": gates,
        "engine_names": np.asarray(["mujoco", "pinocchio"]),
    }
    record = _build_inertia_record(
        config,
        engine_versions={"mujoco": str(mujoco.__version__), "pinocchio": pin_version},
        solution_q=solution_q,
        grip_span_m=grip_span_m,
        metrics=metrics,
        gates=gates,
        model_hashes=model_hashes,
        joint_sequence=joint_sequence,
    )
    return record, arrays


__all__ = [
    "ArticulatedInertiaConfig",
    "build_pinocchio_articulated_model",
    "finite_difference_kinematics",
    "run_articulated_inertia_atlas",
]
