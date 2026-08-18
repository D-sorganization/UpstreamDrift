"""Versioned scientific contracts for interaction-transfer evidence.

This module provides the engine-neutral evidence boundary used by the
proximal-to-distal model ladder. It deliberately separates preregistered
predictions from numerical trajectories and keeps every spatial wrench tied to
an interface, action direction, frame, reference point, and compatible twist.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from .drift_control_transfer import JointTransferTrajectory

FloatArray: TypeAlias = npt.NDArray[np.float64]
SplitName: TypeAlias = Literal["total", "drift", "control", "zvcf"]

SCHEMA_VERSION = "proximal-distal-evidence-v2"
_CLOSURE_RTOL = 1e-9
_CLOSURE_ATOL = 1e-10
_PREDICTION_STATUSES = frozenset(
    {"untested", "supported", "contradicted", "inconclusive"}
)


def _nonempty(name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _nonempty_tuple(name: str, value: Iterable[object]) -> tuple[str, ...]:
    items = tuple(_nonempty(name, item) for item in value)
    if not items:
        raise ValueError(f"{name} must be non-empty")
    return items


def _finite_array(name: str, value: object, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass(frozen=True, slots=True)
class InterfaceDescriptor:
    """Unambiguous ownership and coordinate metadata for one interface."""

    name: str
    proximal_body: str
    distal_body: str
    frame: str
    reference_point: str
    action_direction: str

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(
                self, field.name, _nonempty(field.name, getattr(self, field.name))
            )


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """A preregistered, falsifiable mechanism prediction."""

    prediction_id: str
    hypothesis_id: str
    statement: str
    estimand: str
    intervention: str
    expected_result: str
    falsifier: str
    competing_explanations: tuple[str, ...]
    negative_controls: tuple[str, ...]
    required_model_tiers: tuple[str, ...]
    tolerance_id: str
    status: str = "untested"

    def __post_init__(self) -> None:
        for name in (
            "prediction_id",
            "hypothesis_id",
            "statement",
            "estimand",
            "intervention",
            "expected_result",
            "falsifier",
            "tolerance_id",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        for name in (
            "competing_explanations",
            "negative_controls",
            "required_model_tiers",
        ):
            object.__setattr__(self, name, _nonempty_tuple(name, getattr(self, name)))
        if self.status not in _PREDICTION_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_PREDICTION_STATUSES)}, got {self.status!r}"
            )

    def as_record(self) -> dict[str, object]:
        """Return a JSON-safe record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NumericalTolerance:
    """Scale-aware tolerance calibrated before a prediction is evaluated."""

    tolerance_id: str
    absolute: float
    relative: float
    calibration_method: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tolerance_id", _nonempty("tolerance_id", self.tolerance_id)
        )
        object.__setattr__(
            self,
            "calibration_method",
            _nonempty("calibration_method", self.calibration_method),
        )
        object.__setattr__(self, "source", _nonempty("source", self.source))
        if not np.isfinite(self.absolute) or self.absolute < 0.0:
            raise ValueError("absolute tolerance must be finite and non-negative")
        if not np.isfinite(self.relative) or self.relative < 0.0:
            raise ValueError("relative tolerance must be finite and non-negative")
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError("at least one tolerance component must be positive")

    def as_record(self) -> dict[str, object]:
        """Return a JSON-safe record."""
        return asdict(self)


def calibrate_convergence_tolerance(
    *,
    tolerance_id: str,
    step_sizes: object,
    observed_values: object,
    safety_factor: float,
    source: str,
) -> NumericalTolerance:
    """Calibrate a tolerance from the two finest pre-outcome discretizations.

    Preconditions:
        At least two positive step sizes are supplied in strictly decreasing
        order, and values are finite with matching length.
    Postconditions:
        The absolute bound equals ``safety_factor`` times the finest-pair
        difference; the relative bound uses the finest result as its scale.
    """
    steps = np.asarray(step_sizes, dtype=float).reshape(-1)
    values = np.asarray(observed_values, dtype=float).reshape(-1)
    if steps.size < 2 or steps.shape != values.shape:
        raise ValueError("step_sizes and observed_values must have equal length >= 2")
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("step_sizes must be positive and finite")
    if np.any(np.diff(steps) >= 0.0):
        raise ValueError("step_sizes must be strictly decreasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("observed_values must be finite")
    if not np.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("safety_factor must be finite and at least one")
    absolute = float(safety_factor * abs(values[-1] - values[-2]))
    scale = max(abs(float(values[-1])), np.finfo(float).eps)
    if absolute == 0.0:
        absolute = float(safety_factor * np.finfo(float).eps * scale)
    return NumericalTolerance(
        tolerance_id=tolerance_id,
        absolute=absolute,
        relative=absolute / scale,
        calibration_method="finest-pair-difference-times-safety-factor",
        source=source,
    )


@dataclass(frozen=True, slots=True)
class EvidenceManifest:
    """Prediction and tolerance registry for one scientific program."""

    study_id: str
    predictions: tuple[PredictionRecord, ...]
    tolerances: tuple[NumericalTolerance, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _nonempty("study_id", self.study_id))
        object.__setattr__(
            self, "schema_version", _nonempty("schema_version", self.schema_version)
        )
        predictions = tuple(self.predictions)
        tolerances = tuple(self.tolerances)
        if not predictions:
            raise ValueError("predictions must be non-empty")
        prediction_ids = [item.prediction_id for item in predictions]
        tolerance_ids = [item.tolerance_id for item in tolerances]
        if len(set(prediction_ids)) != len(prediction_ids):
            raise ValueError("prediction IDs must be unique")
        if len(set(tolerance_ids)) != len(tolerance_ids):
            raise ValueError("tolerance IDs must be unique")
        known_tolerances = set(tolerance_ids)
        missing = sorted({item.tolerance_id for item in predictions} - known_tolerances)
        if missing:
            raise ValueError(f"predictions reference unknown tolerance IDs: {missing}")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "tolerances", tolerances)

    def as_record(self) -> dict[str, object]:
        """Return a JSON-safe manifest."""
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "predictions": [item.as_record() for item in self.predictions],
            "tolerances": [item.as_record() for item in self.tolerances],
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EvidenceManifest:
        """Build and validate a manifest decoded from JSON."""
        return cls(
            schema_version=record["schema_version"],
            study_id=record["study_id"],
            predictions=tuple(
                PredictionRecord(**item) for item in record["predictions"]
            ),
            tolerances=tuple(
                NumericalTolerance(**item) for item in record["tolerances"]
            ),
        )


def load_evidence_manifest(path: str | Path) -> EvidenceManifest:
    """Load a UTF-8 JSON evidence manifest and validate every contract."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict):
        raise ValueError("evidence manifest root must be an object")
    return EvidenceManifest.from_record(record)


@dataclass(frozen=True, slots=True)
class SpatialWrenchTrajectory:
    """Named 3-D interface wrenches and compatible twists at matched states.

    Wrench layout is ``[Fx, Fy, Fz, Mx, My, Mz]`` and twist layout is
    ``[vx, vy, vz, wx, wy, wz]``. Arrays have shape ``(T, I, 6)``.
    """

    time: FloatArray
    interfaces: tuple[InterfaceDescriptor, ...]
    reference_position_m: FloatArray
    wrench_total: FloatArray
    wrench_drift: FloatArray
    wrench_control: FloatArray
    twist: FloatArray
    model_tier: str
    wrench_zvcf: FloatArray | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float).reshape(-1)
        if time.size < 2 or not np.all(np.isfinite(time)):
            raise ValueError("time must contain at least two finite samples")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing")
        interfaces = tuple(self.interfaces)
        if not interfaces:
            raise ValueError("interfaces must be non-empty")
        names = [item.name for item in interfaces]
        if len(set(names)) != len(names):
            raise ValueError("interface names must be unique")
        samples = time.size
        count = len(interfaces)
        positions = _finite_array(
            "reference_position_m", self.reference_position_m, (samples, count, 3)
        )
        arrays = {
            name: _finite_array(name, getattr(self, name), (samples, count, 6))
            for name in (
                "wrench_total",
                "wrench_drift",
                "wrench_control",
                "twist",
            )
        }
        if not np.allclose(
            arrays["wrench_total"],
            arrays["wrench_drift"] + arrays["wrench_control"],
            rtol=_CLOSURE_RTOL,
            atol=_CLOSURE_ATOL,
        ):
            residual = np.max(
                np.abs(
                    arrays["wrench_total"]
                    - arrays["wrench_drift"]
                    - arrays["wrench_control"]
                )
            )
            raise ValueError(
                f"wrench_total must equal drift + control; residual={residual:.3e}"
            )
        zvcf = None
        if self.wrench_zvcf is not None:
            zvcf = _finite_array("wrench_zvcf", self.wrench_zvcf, (samples, count, 6))
        object.__setattr__(self, "time", time.copy())
        object.__setattr__(self, "interfaces", interfaces)
        object.__setattr__(self, "reference_position_m", positions)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "wrench_zvcf", zvcf)
        object.__setattr__(self, "model_tier", _nonempty("model_tier", self.model_tier))
        object.__setattr__(
            self, "schema_version", _nonempty("schema_version", self.schema_version)
        )

    @property
    def sample_count(self) -> int:
        """Return the number of samples."""
        return int(self.time.size)

    @property
    def interface_count(self) -> int:
        """Return the number of named interfaces."""
        return len(self.interfaces)

    def as_init_dict(self) -> dict[str, Any]:
        """Return constructor fields for deterministic rebuilding."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def _wrench(self, split: SplitName) -> FloatArray:
        if split == "zvcf":
            if self.wrench_zvcf is None:
                raise ValueError("zvcf wrench is unavailable")
            return self.wrench_zvcf
        return getattr(self, f"wrench_{split}")

    def power(self, split: SplitName) -> FloatArray:
        """Return compatible wrench–twist power with shape ``(T, I)``."""
        wrench = self._wrench(split)
        return np.einsum(
            "tid,tid->ti", wrench[..., :3], self.twist[..., :3]
        ) + np.einsum("tid,tid->ti", wrench[..., 3:], self.twist[..., 3:])

    def transport(
        self, new_reference_position_m: object, *, reference_point: str
    ) -> SpatialWrenchTrajectory:
        """Transport every wrench/twist together, preserving total power."""
        new_positions = _finite_array(
            "new_reference_position_m",
            new_reference_position_m,
            self.reference_position_m.shape,
        )
        offset = new_positions - self.reference_position_m

        def move_required_wrench(wrench: FloatArray) -> FloatArray:
            moved = wrench.copy()
            moved[..., 3:] = wrench[..., 3:] - np.cross(offset, wrench[..., :3])
            return moved

        def move_optional_wrench(wrench: FloatArray | None) -> FloatArray | None:
            if wrench is None:
                return None
            return move_required_wrench(wrench)

        moved_twist = self.twist.copy()
        moved_twist[..., :3] = self.twist[..., :3] + np.cross(
            self.twist[..., 3:], offset
        )
        interfaces = tuple(
            replace(item, reference_point=_nonempty("reference_point", reference_point))
            for item in self.interfaces
        )
        return replace(
            self,
            interfaces=interfaces,
            reference_position_m=new_positions,
            wrench_total=move_required_wrench(self.wrench_total),
            wrench_drift=move_required_wrench(self.wrench_drift),
            wrench_control=move_required_wrench(self.wrench_control),
            wrench_zvcf=move_optional_wrench(self.wrench_zvcf),
            twist=moved_twist,
        )

    def rotate(self, rotation: object, *, frame: str) -> SpatialWrenchTrajectory:
        """Express all vector fields in a proper rotated Cartesian frame."""
        matrix = np.asarray(rotation, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("rotation must have shape (3, 3) with finite values")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12) or not np.isclose(
            np.linalg.det(matrix), 1.0, atol=1e-12
        ):
            raise ValueError("rotation must be a proper rotation")

        def rotate_vectors(values: FloatArray) -> FloatArray:
            result = values.copy()
            result[..., :3] = np.einsum("ab,tib->tia", matrix, values[..., :3])
            result[..., 3:] = np.einsum("ab,tib->tia", matrix, values[..., 3:])
            return result

        zvcf = None if self.wrench_zvcf is None else rotate_vectors(self.wrench_zvcf)
        return replace(
            self,
            interfaces=tuple(
                replace(item, frame=_nonempty("frame", frame))
                for item in self.interfaces
            ),
            reference_position_m=np.einsum(
                "ab,tib->tia", matrix, self.reference_position_m
            ),
            wrench_total=rotate_vectors(self.wrench_total),
            wrench_drift=rotate_vectors(self.wrench_drift),
            wrench_control=rotate_vectors(self.wrench_control),
            wrench_zvcf=zvcf,
            twist=rotate_vectors(self.twist),
        )


def spatial_from_planar(
    trajectory: JointTransferTrajectory,
    *,
    body_pairs: tuple[tuple[str, str], ...],
) -> SpatialWrenchTrajectory:
    """Embed a planar joint-transfer trajectory in the spatial v2 contract.

    The mapping is lossless: planar x/y vectors occupy the first two spatial
    axes, while scalar couple and angular velocity occupy the z rotational
    axis. Callers must declare the proximal/distal body pair for every joint.
    """
    joint_names = tuple(trajectory.joint_names)
    if len(body_pairs) != len(joint_names):
        raise ValueError("body_pairs must provide one pair for every planar joint")
    interfaces = tuple(
        InterfaceDescriptor(
            name=joint_name,
            proximal_body=body_pair[0],
            distal_body=body_pair[1],
            frame=trajectory.frame,
            reference_point=trajectory.reference_point,
            action_direction=trajectory.force_direction,
        )
        for joint_name, body_pair in zip(joint_names, body_pairs, strict=True)
    )
    sample_count = int(trajectory.sample_count)
    joint_count = len(joint_names)

    def vector3(values: object) -> FloatArray:
        result = np.zeros((sample_count, joint_count, 3), dtype=float)
        result[..., :2] = np.asarray(values, dtype=float)
        return result

    def wrench(force: object, couple: object) -> FloatArray:
        result = np.zeros((sample_count, joint_count, 6), dtype=float)
        result[..., :2] = np.asarray(force, dtype=float)
        result[..., 5] = np.asarray(couple, dtype=float)
        return result

    twist = np.zeros((sample_count, joint_count, 6), dtype=float)
    twist[..., :2] = np.asarray(trajectory.velocity, dtype=float)
    twist[..., 5] = np.asarray(trajectory.angular_velocity, dtype=float)
    return SpatialWrenchTrajectory(
        time=trajectory.time,
        interfaces=interfaces,
        reference_position_m=vector3(trajectory.position),
        wrench_total=wrench(
            trajectory.force_total,
            trajectory.couple_total,
        ),
        wrench_drift=wrench(
            trajectory.force_drift,
            trajectory.couple_drift,
        ),
        wrench_control=wrench(
            trajectory.force_control,
            trajectory.couple_control,
        ),
        twist=twist,
        model_tier=trajectory.model_tier,
    )


__all__ = [
    "SCHEMA_VERSION",
    "EvidenceManifest",
    "InterfaceDescriptor",
    "NumericalTolerance",
    "PredictionRecord",
    "SpatialWrenchTrajectory",
    "calibrate_convergence_tolerance",
    "load_evidence_manifest",
    "spatial_from_planar",
]
