"""Explicit additional BUET upper/trunk plus Hamner distal-leg source assembly.

No donor source field is edited. This is a native mechanical candidate, not an
anatomy, passive-force, contact or capture-matching qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.native_mtp_reduction import (
    _body_velocity,
    _constraint_residual,
    _native_mass,
    _native_total_generalized_force,
)

_DISTAL_BODIES = tuple(
    f"{body}_{side}"
    for side in ("r", "l")
    for body in ("tibia", "talus", "calcn", "toes")
)
_DISTAL_JOINTS = tuple(
    f"{joint}_{side}"
    for side in ("r", "l")
    for joint in ("knee", "ankle", "subtalar", "mtp")
)
_EXCLUDED_HAMNER_MUSCLES = frozenset(
    f"{name}_{side}"
    for side in ("r", "l")
    for name in ("ercspn", "intobl", "extobl", "psoas")
)
_HIP_POSES = ((0.0, 0.0, 0.0), (0.2, -0.1, 0.15), (-0.15, 0.08, -0.12))
_CHECK_TOL = 1e-10
_REVIEWED_BUET_SHA256 = (
    "b66a31e08087327f8756d587f471b0494a3d9d6f7f6d510de66beba4a9e79bf6"
)
_REVIEWED_HAMNER_SHA256 = (
    "349ce7a44f1541794f2283cab73bb954270228f2f50808de549ec5097a1687cf"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_binary_sha(module_path: str | None) -> str:
    if module_path is None:
        raise ValueError("native module has no source artifact")
    return _sha(Path(module_path))


@dataclass(frozen=True)
class BuetHamnerAssemblyRequest:
    """Exact local donors and a fresh output for version-one regional ownership."""

    buet_source: Path
    buet_sha256: str
    hamner_source: Path
    hamner_sha256: str
    derived_output: Path

    def __post_init__(self) -> None:
        if not all(
            isinstance(path, Path)
            for path in (self.buet_source, self.hamner_source, self.derived_output)
        ):
            raise TypeError("native source/output paths must be Path objects")
        if not all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (self.buet_sha256, self.hamner_sha256)
        ):
            raise ValueError("exact lowercase source SHA-256 declarations required")
        if (self.buet_sha256, self.hamner_sha256) != (
            _REVIEWED_BUET_SHA256,
            _REVIEWED_HAMNER_SHA256,
        ):
            raise ValueError("reviewed source hashes required by assembly recipe v1")
        resolved = {
            path.resolve()
            for path in (self.buet_source, self.hamner_source, self.derived_output)
        }
        if len(resolved) != 3:
            raise ValueError("source and output paths must be distinct")


@dataclass(frozen=True)
class BuetHamnerAssemblyReceipt:
    """Native fresh-reload sampled mechanical comparison, not qualification."""

    buet_source_sha256: str
    hamner_source_sha256: str
    derived_sha256: str
    assembler_sha256: str
    native_version: str
    native_simulation_sha256: str
    native_simbody_sha256: str
    native_actuators_sha256: str
    buet_muscle_count: int
    hamner_muscle_count: int
    observed_pose_count: int
    max_shared_body_transform_error: float
    max_shared_body_velocity_error: float
    max_owned_body_transform_error: float
    max_owned_body_velocity_error: float
    max_owned_muscle_length_error: float
    max_owned_muscle_speed_error: float
    minimum_native_mass_eigenvalue: float
    native_applied_force_norm: float
    max_native_constraint_residual: float
    fresh_native_reload: bool


@dataclass(frozen=True)
class _NativeSample:
    body_transform_error: float
    body_velocity_error: float
    owned_body_transform_error: float
    owned_body_velocity_error: float
    muscle_length_error: float
    muscle_speed_error: float
    minimum_mass_eigenvalue: float
    native_force_norm: float
    constraint_residual: float


def _native_names(collection: Any) -> tuple[str, ...]:
    return tuple(collection.get(i).getName() for i in range(collection.getSize()))


def _physical_body_signature(body: Any) -> tuple[float, ...]:
    inertia = body.getInertia()
    return (
        float(body.getMass()),
        *(float(body.getMassCenter().get(i)) for i in range(3)),
        *(float(inertia.getMoments().get(i)) for i in range(3)),
        *(float(inertia.getProducts().get(i)) for i in range(3)),
    )


def _check_source_inventory(buet: Any, hamner: Any) -> tuple[str, ...]:
    import opensim as osim

    buet_bodies = set(_native_names(buet.getBodySet()))
    hamner_bodies = set(_native_names(hamner.getBodySet()))
    if not set(_DISTAL_BODIES).issubset(hamner_bodies):
        raise ValueError("Hamner distal body inventory is incomplete")
    if buet_bodies.intersection(_DISTAL_BODIES):
        raise ValueError("BUET already owns a declared distal body")
    if not {"pelvis", "femur_r", "femur_l"}.issubset(buet_bodies & hamner_bodies):
        raise ValueError("shared pelvis/femur seam is incomplete")
    for name in ("pelvis", "femur_r", "femur_l"):
        before = _physical_body_signature(buet.getBodySet().get(name))
        after = _physical_body_signature(hamner.getBodySet().get(name))
        if before != after or not np.all(np.isfinite(before)):
            raise ValueError("source shared-body physical properties disagree")
    for name in ("hip_r", "hip_l"):
        buet_hip = osim.CustomJoint.safeDownCast(buet.getJointSet().get(name))
        hamner_hip = osim.CustomJoint.safeDownCast(hamner.getJointSet().get(name))
        if buet_hip is None or hamner_hip is None:
            raise ValueError("source shared-hip joint class changed")
        first = buet_hip.getSpatialTransform().dump()
        second = hamner_hip.getSpatialTransform().dump()
        if first != second:
            raise ValueError("source shared-hip mechanics disagree")
    if not set(_DISTAL_JOINTS).issubset(_native_names(hamner.getJointSet())):
        raise ValueError("Hamner distal joint inventory is incomplete")
    buet_muscles = _native_names(buet.getMuscles())
    hamner_muscles = _native_names(hamner.getMuscles())
    if len(buet_muscles) != 473 or len(hamner_muscles) != 92:
        raise ValueError("reviewed BUET473/Hamner92 source inventory required")
    if not _EXCLUDED_HAMNER_MUSCLES.issubset(hamner_muscles):
        raise ValueError("declared trunk and psoas ownership cannot be resolved")
    owned = tuple(
        name for name in hamner_muscles if name not in _EXCLUDED_HAMNER_MUSCLES
    )
    if len(owned) != 84 or set(owned) & set(buet_muscles):
        raise ValueError("exact nonoverlapping 84-muscle lower partition required")
    return owned


def _append_owned_components(derived: Any, hamner: Any, owned: tuple[str, ...]) -> None:
    for name in _DISTAL_BODIES:
        derived.addBody(hamner.getBodySet().get(name).clone())
    for name in _DISTAL_JOINTS:
        derived.addJoint(hamner.getJointSet().get(name).clone())
    for name in owned:
        derived.addForce(hamner.getMuscles().get(name).clone())
    derived.finalizeConnections()


def _verify_exact_components(
    buet: Any, hamner: Any, fresh: Any, owned: tuple[str, ...]
) -> None:
    """Compare native serialized properties, including paths and wrap sockets."""
    groups = (
        (buet.getBodySet(), fresh.getBodySet(), _native_names(buet.getBodySet())),
        (buet.getJointSet(), fresh.getJointSet(), _native_names(buet.getJointSet())),
        (buet.getMuscles(), fresh.getMuscles(), _native_names(buet.getMuscles())),
        (
            buet.getConstraintSet(),
            fresh.getConstraintSet(),
            _native_names(buet.getConstraintSet()),
        ),
        (hamner.getBodySet(), fresh.getBodySet(), _DISTAL_BODIES),
        (hamner.getJointSet(), fresh.getJointSet(), _DISTAL_JOINTS),
        (hamner.getMuscles(), fresh.getMuscles(), owned),
    )
    for source_set, derived_set, names in groups:
        for name in names:
            if source_set.get(name).dump() != derived_set.get(name).dump():
                raise ValueError(f"native regional property/path changed: {name}")


def _transform_array(body: Any, state: Any) -> np.ndarray:
    transform = body.getTransformInGround(state)
    return np.array(
        [transform.R().get(i, j) for i in range(3) for j in range(3)]
        + [transform.p().get(i) for i in range(3)],
        dtype=float,
    )


def _set_common_hips(model: Any, state: Any, values: tuple[float, ...]) -> None:
    for side in ("r", "l"):
        for name, value in zip(
            ("hip_flexion", "hip_adduction", "hip_rotation"), values, strict=True
        ):
            model.getCoordinateSet().get(f"{name}_{side}").setValue(state, value, False)


def _set_common_hip_speeds(model: Any, state: Any) -> None:
    for side in ("r", "l"):
        for name, speed in zip(
            ("hip_flexion", "hip_adduction", "hip_rotation"),
            (0.07, -0.03, 0.02),
            strict=True,
        ):
            model.getCoordinateSet().get(f"{name}_{side}").setSpeedValue(state, speed)


def _body_errors(
    source: Any,
    source_state: Any,
    derived: Any,
    derived_state: Any,
    names: tuple[str, ...],
) -> tuple[float, float]:
    max_transform = max_velocity = 0.0
    for name in names:
        original = _transform_array(source.getBodySet().get(name), source_state)
        other = _transform_array(derived.getBodySet().get(name), derived_state)
        if not np.all(np.isfinite(original)) or not np.all(np.isfinite(other)):
            raise ValueError("nonfinite regional body observation")
        max_transform = max(max_transform, float(np.max(np.abs(original - other))))
        source_velocity = _body_velocity(source.getBodySet().get(name), source_state)
        derived_velocity = _body_velocity(derived.getBodySet().get(name), derived_state)
        max_velocity = max(
            max_velocity, float(np.max(np.abs(source_velocity - derived_velocity)))
        )
    return max_transform, max_velocity


def _sample_errors(
    buet: Any, hamner: Any, derived: Any, owned: tuple[str, ...]
) -> _NativeSample:
    buet_state = buet.initSystem()
    hamner_state = hamner.initSystem()
    derived_state = derived.initSystem()
    max_body = max_velocity = max_owned_body = max_owned_velocity = 0.0
    max_muscle = max_speed = max_residual = 0.0
    min_mass = float("inf")
    max_force = 0.0
    for pose in _HIP_POSES:
        for model, state in (
            (buet, buet_state),
            (hamner, hamner_state),
            (derived, derived_state),
        ):
            _set_common_hips(model, state, pose)
            _set_common_hip_speeds(model, state)
            model.realizeVelocity(state)
        body, velocity = _body_errors(
            buet, buet_state, derived, derived_state, ("pelvis", "femur_r", "femur_l")
        )
        max_body, max_velocity = max(max_body, body), max(max_velocity, velocity)
        body, velocity = _body_errors(
            hamner, hamner_state, derived, derived_state, _DISTAL_BODIES
        )
        max_owned_body = max(max_owned_body, body)
        max_owned_velocity = max(max_owned_velocity, velocity)
        for name in _native_names(buet.getMuscles()):
            before = float(buet.getMuscles().get(name).getLength(buet_state))
            after = float(derived.getMuscles().get(name).getLength(derived_state))
            if not np.isfinite(before) or not np.isfinite(after):
                raise ValueError("nonfinite BUET muscle path observation")
            max_muscle = max(max_muscle, abs(before - after))
            source_speed = float(
                buet.getMuscles().get(name).getLengtheningSpeed(buet_state)
            )
            derived_speed = float(
                derived.getMuscles().get(name).getLengtheningSpeed(derived_state)
            )
            if not np.isfinite(source_speed) or not np.isfinite(derived_speed):
                raise ValueError("nonfinite BUET muscle path speed")
            max_speed = max(max_speed, abs(source_speed - derived_speed))
        for name in owned:
            before = float(hamner.getMuscles().get(name).getLength(hamner_state))
            after = float(derived.getMuscles().get(name).getLength(derived_state))
            if not np.isfinite(before) or not np.isfinite(after):
                raise ValueError("nonfinite Hamner muscle path observation")
            max_muscle = max(max_muscle, abs(before - after))
            source_speed = float(
                hamner.getMuscles().get(name).getLengtheningSpeed(hamner_state)
            )
            derived_speed = float(
                derived.getMuscles().get(name).getLengtheningSpeed(derived_state)
            )
            if not np.isfinite(source_speed) or not np.isfinite(derived_speed):
                raise ValueError("nonfinite Hamner muscle path speed")
            max_speed = max(max_speed, abs(source_speed - derived_speed))
        mass = _native_mass(derived, derived_state)
        min_mass = min(min_mass, float(np.linalg.eigvalsh(mass).min()))
        force = _native_total_generalized_force(derived, derived_state)
        max_force = max(max_force, float(np.linalg.norm(force)))
        max_residual = max(max_residual, _constraint_residual(derived_state))
    if not np.all(np.isfinite((min_mass, max_force))) or min_mass <= 0:
        raise ValueError("native candidate mass or force observation is invalid")
    return _NativeSample(
        max_body,
        max_velocity,
        max_owned_body,
        max_owned_velocity,
        max_muscle,
        max_speed,
        min_mass,
        max_force,
        max_residual,
    )


def assemble_buet_hamner(
    request: BuetHamnerAssemblyRequest,
) -> BuetHamnerAssemblyReceipt:
    """Build the explicit source-owned candidate and verify fresh native paths."""
    import opensim as osim
    import opensim._actuators as native_actuators
    import opensim._simbody as native_simbody
    import opensim._simulation as native_simulation

    if (
        not request.buet_source.is_file()
        or _sha(request.buet_source) != request.buet_sha256
    ):
        raise ValueError("BUET source SHA-256 mismatch or missing source")
    if (
        not request.hamner_source.is_file()
        or _sha(request.hamner_source) != request.hamner_sha256
    ):
        raise ValueError("Hamner source SHA-256 mismatch or missing source")
    if request.derived_output.exists():
        raise FileExistsError(request.derived_output)
    output_parent = request.derived_output.parent
    if not output_parent.is_dir():
        raise ValueError("derived output parent directory missing")
    buet = osim.Model(str(request.buet_source))
    hamner = osim.Model(str(request.hamner_source))
    buet.initSystem()
    hamner.initSystem()
    owned = _check_source_inventory(buet, hamner)
    derived = osim.Model(buet)
    _append_owned_components(derived, hamner, owned)
    derived.printToXML(str(request.derived_output))
    try:
        fresh = osim.Model(str(request.derived_output))
        fresh.initSystem()
        if fresh.getBodySet().getSize() != buet.getBodySet().getSize() + 8:
            raise ValueError("fresh assembled body inventory changed")
        if fresh.getMuscles().getSize() != buet.getMuscles().getSize() + 84:
            raise ValueError("fresh assembled muscle inventory changed")
        _verify_exact_components(buet, hamner, fresh, owned)
        sample = _sample_errors(buet, hamner, fresh, owned)
        if (
            max(
                sample.body_transform_error,
                sample.body_velocity_error,
                sample.owned_body_transform_error,
                sample.owned_body_velocity_error,
                sample.muscle_length_error,
                sample.muscle_speed_error,
            )
            > _CHECK_TOL
        ):
            raise ValueError("fresh regional path or shared-body mechanics changed")
        if (_sha(request.buet_source), _sha(request.hamner_source)) != (
            request.buet_sha256,
            request.hamner_sha256,
        ):
            raise ValueError("native source changed during assembly")
    except Exception:
        request.derived_output.unlink(missing_ok=True)
        raise
    return BuetHamnerAssemblyReceipt(
        buet_source_sha256=request.buet_sha256,
        hamner_source_sha256=request.hamner_sha256,
        derived_sha256=_sha(request.derived_output),
        assembler_sha256=_sha(Path(__file__)),
        native_version=osim.GetVersionAndDate(),
        native_simulation_sha256=_native_binary_sha(native_simulation.__file__),
        native_simbody_sha256=_native_binary_sha(native_simbody.__file__),
        native_actuators_sha256=_native_binary_sha(native_actuators.__file__),
        buet_muscle_count=buet.getMuscles().getSize(),
        hamner_muscle_count=len(owned),
        observed_pose_count=len(_HIP_POSES),
        max_shared_body_transform_error=sample.body_transform_error,
        max_shared_body_velocity_error=sample.body_velocity_error,
        max_owned_body_transform_error=sample.owned_body_transform_error,
        max_owned_body_velocity_error=sample.owned_body_velocity_error,
        max_owned_muscle_length_error=sample.muscle_length_error,
        max_owned_muscle_speed_error=sample.muscle_speed_error,
        minimum_native_mass_eigenvalue=sample.minimum_mass_eigenvalue,
        native_applied_force_norm=sample.native_force_norm,
        max_native_constraint_residual=sample.constraint_residual,
        fresh_native_reload=True,
    )
