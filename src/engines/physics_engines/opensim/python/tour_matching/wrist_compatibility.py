"""Pinned native wrist geometry observations, not anatomical qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import import_module
import math
from pathlib import Path
import sys
from typing import Any

from src.engines.physics_engines.opensim.python.tour_matching.full_swing_tracking import (
    validate_model_checkpoint,
)
from src.engines.physics_engines.opensim.python.tour_matching.muscle_qualification import (
    compute_path_length_finite_difference_moment_arm,
)


@dataclass(frozen=True)
class WristKinematicContract:
    """Explicit radian poses and numerical tolerances; none is a measured limit."""

    independent_coordinates: tuple[str, ...]
    dependent_coordinates: tuple[str, ...]
    muscle_paths: tuple[str, ...]
    required_component_paths: tuple[str, ...]
    anchor_frame: str
    hand_frame: str
    poses_rad: tuple[tuple[float, ...], ...]
    difference_steps_rad: tuple[float, ...]
    coordinate_tolerance_rad: float
    translation_tolerance_m: float
    constraint_tolerance: float
    moment_arm_tolerance_m: float
    declared_coupled_radian_coordinates: tuple[str, ...] = ()
    assembly_accuracy: float | None = None


def _validate_contract(contract: WristKinematicContract) -> None:
    accuracy = contract.assembly_accuracy
    if accuracy is not None and (not math.isfinite(accuracy) or not 0 < accuracy < 1):
        raise ValueError("assembly accuracy must be finite and between zero and one")
    for name in ("independent_coordinates", "muscle_paths", "required_component_paths"):
        if not getattr(contract, name):
            raise ValueError(f"{name} must be nonempty")
    coordinate_paths = contract.independent_coordinates + contract.dependent_coordinates
    declared = contract.declared_coupled_radian_coordinates
    if len(set(declared)) != len(declared) or not set(declared) <= set(
        coordinate_paths
    ):
        raise ValueError(
            "coupled radian declarations must name unique requested coordinates"
        )
    for paths in (
        coordinate_paths,
        contract.muscle_paths,
        contract.required_component_paths,
    ):
        if len(set(paths)) != len(paths) or any(not p.startswith("/") for p in paths):
            raise ValueError("component paths must be unique absolute native paths")
    if contract.anchor_frame == contract.hand_frame:
        raise ValueError("anchor and hand frames must differ")
    steps = contract.difference_steps_rad
    if len(steps) < 2 or any(not math.isfinite(h) or h <= 0 for h in steps):
        raise ValueError("at least two positive finite difference steps are required")
    if any(a <= b for a, b in zip(steps, steps[1:], strict=False)):
        raise ValueError("difference steps must be strictly decreasing")
    for name in (
        "coordinate_tolerance_rad",
        "translation_tolerance_m",
        "constraint_tolerance",
        "moment_arm_tolerance_m",
    ):
        value = getattr(contract, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if min(steps) <= contract.coordinate_tolerance_rad:
        raise ValueError("difference steps must exceed coordinate tolerance")
    if not contract.poses_rad or any(
        len(pose) != len(contract.independent_coordinates)
        or any(not math.isfinite(q) for q in pose)
        for pose in contract.poses_rad
    ):
        raise ValueError("poses must be finite and match the independent coordinates")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("native observation is non-finite")
    return result


def _resolve(
    osim: Any, model: Any, state: Any, contract: WristKinematicContract
) -> dict[str, Any]:
    components = {c.getAbsolutePathString(): c for c in model.getComponentsList()}
    coordinates = {}
    for path in contract.independent_coordinates + contract.dependent_coordinates:
        coordinate = (
            osim.Coordinate.safeDownCast(components.get(path))
            if path in components
            else None
        )
        if coordinate is None:
            raise ValueError(f"coordinate component is missing: {path}")
        motion = coordinate.getMotionType()
        if motion == osim.Coordinate.Coupled:
            if path not in contract.declared_coupled_radian_coordinates:
                raise ValueError(
                    f"coupled coordinate requires explicit radian declaration: {path}"
                )
        elif motion != osim.Coordinate.Rotational:
            raise ValueError(f"wrist coordinate is not rotational: {path}")
        elif path in contract.declared_coupled_radian_coordinates:
            raise ValueError(
                f"declared coupled coordinate is natively rotational: {path}"
            )
        dependent = coordinate.isDependent(state)
        if path in contract.independent_coordinates:
            if coordinate.getLocked(state):
                raise ValueError(f"independent coordinate is locked: {path}")
            if coordinate.isPrescribed(state) or dependent:
                raise ValueError(
                    f"independent coordinate is prescribed or dependent: {path}"
                )
        elif not dependent:
            raise ValueError(
                f"declared dependent coordinate is not natively dependent: {path}"
            )
        coordinates[path] = coordinate
    muscles = {}
    for path in contract.muscle_paths:
        muscle = (
            osim.Muscle.safeDownCast(components.get(path))
            if path in components
            else None
        )
        if muscle is None:
            raise ValueError(f"muscle component is missing: {path}")
        muscles[path] = muscle
    for path in contract.required_component_paths:
        if path not in components:
            raise ValueError(f"required component is missing: {path}")
        constraint = osim.Constraint.safeDownCast(components[path])
        if constraint is not None and not constraint.isEnforced(state):
            raise ValueError(f"required constraint is disabled: {path}")
    frames = []
    for path in (contract.anchor_frame, contract.hand_frame):
        frame = (
            osim.PhysicalFrame.safeDownCast(components.get(path))
            if path in components
            else None
        )
        if frame is None:
            raise ValueError(f"physical frame component is missing: {path}")
        frames.append(frame)
    return {
        "components": components,
        "coordinates": coordinates,
        "muscles": muscles,
        "frames": frames,
    }


def _coordinate_values(model: Any, state: Any) -> dict[str, float]:
    coordinates = model.getCoordinateSet()
    return {
        coordinates.get(i).getAbsolutePathString(): _finite(
            coordinates.get(i).getValue(state)
        )
        for i in range(coordinates.getSize())
    }


def _check_held_coordinates(
    osim: Any,
    model: Any,
    baseline: Any,
    state: Any,
    contract: WristKinematicContract,
) -> None:
    coordinates = model.getCoordinateSet()
    for index in range(coordinates.getSize()):
        coordinate = coordinates.get(index)
        path = coordinate.getAbsolutePathString()
        if path in contract.independent_coordinates or coordinate.isDependent(state):
            continue
        motion = coordinate.getMotionType()
        if motion == osim.Coordinate.Rotational:
            tolerance = contract.coordinate_tolerance_rad
        elif motion == osim.Coordinate.Translational:
            tolerance = contract.translation_tolerance_m
        else:
            raise ValueError(
                f"held coordinate has unsupported native motion type: {path}"
            )
        actual = _finite(coordinate.getValue(state))
        original = _finite(coordinate.getValue(baseline))
        if abs(actual - original) > tolerance:
            raise ValueError(
                f"unrequested independent coordinate moved during assembly: {path}; "
                f"baseline={original:.17g}, actual={actual:.17g}, tolerance={tolerance:.17g}"
            )


def _assemble_pose(
    osim: Any,
    model: Any,
    baseline: Any,
    resolved: dict[str, Any],
    contract: WristKinematicContract,
    requested: tuple[float, ...],
) -> tuple[Any, dict[str, float], list[float]]:
    state = osim.State(baseline)
    coordinates = resolved["coordinates"]
    for path, value in zip(contract.independent_coordinates, requested, strict=True):
        coordinate = coordinates[path]
        if value < coordinate.getRangeMin() or value > coordinate.getRangeMax():
            raise ValueError(f"pose or perturbation is outside native range: {path}")
        coordinate.setValue(state, value, False)
    model.assemble(state)
    model.realizePosition(state)
    _check_held_coordinates(osim, model, baseline, state, contract)
    values = _coordinate_values(model, state)
    for path, value in zip(contract.independent_coordinates, requested, strict=True):
        if abs(values[path] - value) > contract.coordinate_tolerance_rad:
            raise ValueError(
                f"requested coordinate was not achieved by native assembly: {path}"
            )
    errors = state.getQErr()
    residuals = [_finite(errors.get(i)) for i in range(errors.size())]
    if any(abs(error) > contract.constraint_tolerance for error in residuals):
        raise ValueError("native position constraint error exceeds declared tolerance")
    return state, values, residuals


def _hand_transform(state: Any, frames: list[Any]) -> list[list[float]]:
    anchor, hand = frames
    transform = hand.findTransformBetween(state, anchor)
    rotation, translation = transform.R(), transform.p()
    return [
        [_finite(rotation.get(i, j)) for j in range(3)] + [_finite(translation.get(i))]
        for i in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _moment_arm_rows(
    osim: Any,
    model: Any,
    baseline: Any,
    resolved: dict[str, Any],
    contract: WristKinematicContract,
    pose: tuple[float, ...],
    nominal: Any,
) -> list[dict[str, Any]]:
    rows = []
    for index, path in enumerate(contract.independent_coordinates):
        coordinate = resolved["coordinates"][path]
        for step in contract.difference_steps_rad:
            samples: list[dict[str, Any]] = []
            for direction in (-1, 1):
                perturbed = list(pose)
                perturbed[index] += direction * step
                state, values, residuals = _assemble_pose(
                    osim, model, baseline, resolved, contract, tuple(perturbed)
                )
                length_map = {
                    p: _finite(m.getLength(state))
                    for p, m in resolved["muscles"].items()
                }
                samples.append(
                    {
                        "all_native_coordinate_values": values,
                        "native_position_constraint_errors": residuals,
                        "muscle_lengths_m": length_map,
                    }
                )
            delta = (
                samples[1]["all_native_coordinate_values"][path]
                - samples[0]["all_native_coordinate_values"][path]
            )
            if (
                delta <= 0
                or abs(delta - 2 * step) > 2 * contract.coordinate_tolerance_rad
            ):
                raise ValueError(
                    "native perturbation did not achieve its declared coordinate span"
                )
            for muscle_path, muscle in resolved["muscles"].items():
                native = _finite(muscle.computeMomentArm(nominal, coordinate))
                lengths: list[float] = [
                    sample["muscle_lengths_m"][muscle_path] for sample in samples
                ]
                # Reuse the public central-difference provider with achieved span.
                derivative = _pair_derivative(lengths, delta)
                difference = _finite(abs(native - derivative))
                rows.append(
                    {
                        "coordinate_path": path,
                        "muscle_path": muscle_path,
                        "requested_step_rad": step,
                        "actual_coordinate_delta_rad": delta,
                        "native_moment_arm_m": native,
                        "finite_difference_m": derivative,
                        "absolute_difference_m": difference,
                        "within_declared_tolerance": difference
                        <= contract.moment_arm_tolerance_m,
                        "perturbed_samples": samples,
                    }
                )
    return rows


def _pair_derivative(lengths: list[float], delta: float) -> float:
    def length_at(offset: float) -> float:
        return lengths[1 if offset > 0 else 0]

    return compute_path_length_finite_difference_moment_arm(length_at, 0.0, delta / 2)


def _runtime_identity(osim: Any) -> dict[str, Any]:
    binaries = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (
            name.startswith("opensim.")
            and filename
            and Path(filename).suffix in {".pyd", ".so"}
        ):
            binaries[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return {
        "version": osim.GetVersionAndDate(),
        "loaded_extension_sha256": binaries,
        "transitive_library_closure": "unverified",
    }


def observe_native_wrist(
    source: Path | str,
    expected_sha256: str,
    contract: WristKinematicContract,
) -> dict[str, Any]:
    """Observe native constrained wrist kinematics without editing the source.

    Raises on invalid or unachieved pose requests. Derivative disagreements stay
    in the returned evidence. Never equilibrates, integrates, or qualifies anatomy.
    External model inventory, license clearance and dependency closure remain required.
    """
    _validate_contract(contract)
    digest = validate_model_checkpoint(source, expected_sha256)
    osim = import_module("opensim")
    model = osim.Model(str(source))
    model.finalizeConnections()
    source_accuracy = model.get_assembly_accuracy()
    pre_policy_digest = _digest(model.dump())
    if contract.assembly_accuracy is not None:
        model.set_assembly_accuracy(contract.assembly_accuracy)
    baseline = model.initSystem()
    loaded_digest = _digest(model.dump())
    resolved = _resolve(osim, model, baseline, contract)
    poses = []
    for pose in contract.poses_rad:
        state, values, residuals = _assemble_pose(
            osim, model, baseline, resolved, contract, pose
        )
        rows = _moment_arm_rows(osim, model, baseline, resolved, contract, pose, state)
        poses.append(
            {
                "requested_coordinates_rad": dict(
                    zip(contract.independent_coordinates, pose, strict=True)
                ),
                "coordinates_rad": {p: values[p] for p in resolved["coordinates"]},
                "all_native_coordinate_values": values,
                "native_position_constraint_errors": residuals,
                "hand_in_anchor_transform": _hand_transform(state, resolved["frames"]),
                "muscle_lengths_m": {
                    p: _finite(m.getLength(state))
                    for p, m in resolved["muscles"].items()
                },
                "moment_arm_checks": rows,
            }
        )
    validate_model_checkpoint(source, digest)
    if _digest(model.dump()) != loaded_digest:
        raise ValueError("native source parameters changed during observation")
    components = resolved["components"]
    return {
        "schema_version": "opensim-wrist-kinematics/1",
        "scientific_status": "unqualified",
        "status": "native-kinematics-observed",
        "source_sha256": digest,
        "loaded_serialization_sha256": loaded_digest,
        "assembly_policy": {
            "method": "Model.assemble after setting requested coordinates without enforcement",
            "source_accuracy": source_accuracy,
            "requested_accuracy": contract.assembly_accuracy,
            "effective_accuracy": model.get_assembly_accuracy(),
            "pre_policy_initialization_serialization_sha256": pre_policy_digest,
        },
        "runtime": _runtime_identity(osim),
        "provider_source_sha256": {
            function.__module__: hashlib.sha256(
                Path(function.__code__.co_filename).read_bytes()
            ).hexdigest()
            for function in (
                observe_native_wrist,
                validate_model_checkpoint,
                compute_path_length_finite_difference_moment_arm,
            )
        },
        "contract": asdict(contract),
        "native_coordinate_motion_types": {
            path: int(coordinate.getMotionType())
            for path, coordinate in resolved["coordinates"].items()
        },
        "poses": poses,
        "required_components": [
            {
                "path": p,
                "concrete_class": components[p].getConcreteClassName(),
                "native_serialization_sha256": _digest(components[p].dump()),
                "native_serialization": components[p].dump(),
            }
            for p in contract.required_component_paths
        ],
        "source_file_unchanged": True,
        "prepared_parameters_unchanged": True,
        "required_evidence": [
            "licensed-anatomical-registration",
            "path-and-wrap-validation",
            "muscle-capacity-and-state-policy",
            "contact-and-grip",
            "full-state-independent-replay",
        ],
    }
