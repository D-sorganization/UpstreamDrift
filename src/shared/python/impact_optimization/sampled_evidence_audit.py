"""Fail-closed audit of supplied, sampled rollout evidence.

This module checks sample values against a :class:`PhysiologicalEnvelope`.
It does not interpolate between samples or certify anatomy/continuous motion.
Scalar joint-axis rates are only comparable to an envelope when the supplied
mapping is explicitly identified and documented; a one-to-one Euler axis may
not represent an anatomical joint degree of freedom.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)

Array = NDArray[np.float64]
Violation = tuple[str, str, int, float]
_TIME_ATOL_S = 1e-12


class JointMappingKind(StrEnum):
    """Mapping classes understood by this sampled audit.

    The supported value asserts direct scalar generalized-joint channels after
    the caller has mapped units and signs. It excludes coupled coordinates,
    motor transmissions, and actuator-side torque that has not been converted
    to joint generalized torque.
    """

    DIRECT_SCALAR_GENERALIZED_JOINT = "direct_scalar_generalized_joint"
    COUPLED = "coupled"
    UNKNOWN = "unknown"


class RateEvidenceKind(StrEnum):
    """Rate sources the audit can distinguish explicitly."""

    DIRECT_RECORDED_RATE = "direct_recorded_rate"
    FINITE_DIFFERENCE = "finite_difference"
    UNKNOWN = "unknown"


class TorqueEvidenceStage(StrEnum):
    """Applied-torque stages recognized by the audit."""

    TOTAL_POST_SATURATION = "total_post_saturation"
    COMMAND_OR_DELTA = "command_or_delta"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class JointSignalBinding:
    """Explicit one-to-one binding from an envelope joint to sampled channels."""

    envelope_joint: str
    coordinate_channel: str
    rate_channel: str
    actuator_channel: str
    mapping_kind: JointMappingKind
    provenance: str


@dataclass(frozen=True)
class SampledMotionEvidence:
    """Actual stage records on one declared sample grid.

    Rates are consumed directly from ``rate_deg_s``. Torque must be the total
    actuator value after saturation, never an optimizer delta or command.
    Channel arrays are ``(samples, channels)`` in their corresponding name
    order. Channels must already be in the envelope's units and signs;
    torque channels must be total post-saturation generalized joint torques
    in N m. ``expected_actuator_names`` comes from the simulator's actuator
    inventory, independently of the physiological envelope. Provenance and
    expected schedules are caller assertions; this audit cannot authenticate
    their sources or establish that the caller enumerated every actual stage.
    """

    times_s: Array
    expected_times_s: Array
    stage_ids: tuple[str, ...]
    expected_stage_ids: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    coordinate_deg: Array
    rate_names: tuple[str, ...]
    rate_deg_s: Array
    actuator_names: tuple[str, ...]
    total_applied_actuator_torque_nm: Array
    expected_actuator_names: tuple[str, ...]
    rate_kind: RateEvidenceKind
    rate_provenance: str
    torque_stage: TorqueEvidenceStage
    mapping_provenance: str
    torque_provenance: str
    limit_provenance: dict[str, str]
    bindings: tuple[JointSignalBinding, ...]


@dataclass(frozen=True)
class SampledEvidenceAudit:
    """Outcome over the supplied sample grid, with no continuous guarantee."""

    status: Literal["pass", "violation", "indeterminate"]
    violations: tuple[Violation, ...] = ()
    reasons: tuple[str, ...] = ()
    assumed_limits: tuple[tuple[str, str], ...] = ()
    sampled_only: bool = True
    continuous_certified: bool = False
    assumption_limited: bool = False
    provenance_is_caller_assertion: bool = True
    mapping_provenance: str = ""
    rate_provenance: str = ""
    torque_provenance: str = ""
    limit_provenance: tuple[tuple[str, str], ...] = ()
    scope_note: str = (
        "Sampled values only; schedule completeness is relative to caller-declared "
        "times and stage IDs; their completeness is not independently verified. "
        "Provenance is caller-asserted. Channels must already be mapped to "
        "envelope units/signs and generalized joint torque. No between-sample "
        "or anatomical certification; one-to-one Euler axes may not represent anatomy."
    )


def _names_are_unique(names: tuple[str, ...]) -> bool:
    return (
        bool(names)
        and all(isinstance(name, str) and name for name in names)
        and len(set(names)) == len(names)
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return value
    return ()


def _has_provenance(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _numeric_array(value: object) -> Array | None:
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _assumed_limits(envelope: PhysiologicalEnvelope) -> tuple[tuple[str, str], ...]:
    assumed: set[tuple[str, str]] = set()
    assumed.update((name, "torque") for name in envelope.assumed_torque_slots)
    assumed.update((name, "velocity") for name in envelope.assumed_velocity_slots)
    assumed.update((name, "rom") for name in envelope.soft_rom_slots)
    assumed.update((name, f"rom:{side}") for name, side in envelope.assumed_rom_sides)
    return tuple(sorted(assumed))


def audit_sampled_evidence(
    evidence: SampledMotionEvidence, envelope: PhysiologicalEnvelope
) -> SampledEvidenceAudit:
    """Compare every expected sample against ROM, rate, and signed torque limits.

    Structural, mapping, provenance, or sampling gaps return ``indeterminate``.
    A ``pass`` means only that no listed limit was exceeded at the requested
    samples. Assumed envelope values remain explicitly flagged in the report.
    """
    if not isinstance(evidence, SampledMotionEvidence):
        return SampledEvidenceAudit(
            status="indeterminate",
            reasons=("sampled evidence record has an unsupported type",),
        )
    reasons: list[str] = []
    limit_sources = evidence.limit_provenance
    if not isinstance(limit_sources, Mapping):
        reasons.append("per-limit provenance mapping is required")
        limit_sources = {}
    elif any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in limit_sources.items()
    ):
        reasons.append("limit provenance entries must be named strings")
    if not _names_are_unique(envelope.names):
        reasons.append("envelope joint names must be nonempty and unique")
    times = _numeric_array(evidence.times_s)
    expected_times = _numeric_array(evidence.expected_times_s)
    if times is None or expected_times is None:
        reasons.append("sample times must be numeric arrays")
        times = np.array([], dtype=float) if times is None else times
        expected_times = (
            np.array([], dtype=float) if expected_times is None else expected_times
        )
    stage_ids_valid = isinstance(evidence.stage_ids, tuple) and _names_are_unique(
        evidence.stage_ids
    )
    expected_ids_valid = isinstance(evidence.expected_stage_ids, tuple) and (
        _names_are_unique(evidence.expected_stage_ids)
    )
    if (
        times.ndim != 1
        or expected_times.ndim != 1
        or times.size == 0
        or not np.isfinite(times).all()
        or not np.isfinite(expected_times).all()
        or np.any(np.diff(times) < 0.0)
        or np.any(np.diff(expected_times) < 0.0)
        or times.shape != expected_times.shape
        or not np.allclose(times, expected_times, rtol=0.0, atol=_TIME_ATOL_S)
        or not stage_ids_valid
        or not expected_ids_valid
        or len(evidence.stage_ids) != times.size
        or evidence.stage_ids != evidence.expected_stage_ids
    ):
        reasons.append(
            "sample times and ordered stage IDs do not match the caller-declared schedule"
        )

    coordinate_names = _string_tuple(evidence.coordinate_names)
    rate_names = _string_tuple(evidence.rate_names)
    actuator_names = _string_tuple(evidence.actuator_names)
    expected_actuator_names = _string_tuple(evidence.expected_actuator_names)
    domains = (
        ("coordinate", coordinate_names, evidence.coordinate_deg),
        ("rate", rate_names, evidence.rate_deg_s),
        (
            "actuator",
            actuator_names,
            evidence.total_applied_actuator_torque_nm,
        ),
    )
    for label, names, raw_values in domains:
        values = _numeric_array(raw_values)
        if not _names_are_unique(names):
            reasons.append(f"{label} channel names must be nonempty and unique")
        if (
            values is None
            or values.ndim != 2
            or values.shape != (times.size, len(names))
            or not np.isfinite(values).all()
        ):
            reasons.append(f"{label} stage records are partial or nonfinite")

    if evidence.torque_stage is not TorqueEvidenceStage.TOTAL_POST_SATURATION:
        reasons.append("torque evidence is not total post-saturation applied torque")
    if evidence.rate_kind is not RateEvidenceKind.DIRECT_RECORDED_RATE:
        reasons.append("rate evidence must be a directly supplied recorded rate")
    if not _has_provenance(evidence.rate_provenance):
        reasons.append("direct rate provenance is required")
    if not _has_provenance(evidence.mapping_provenance) or not _has_provenance(
        evidence.torque_provenance
    ):
        reasons.append("mapping and applied-torque provenance are required")
    if not _names_are_unique(expected_actuator_names):
        reasons.append("expected actuator inventory must be nonempty and unique")
    if set(actuator_names) != set(expected_actuator_names):
        reasons.append(
            "actual actuator records do not cover the declared actuator inventory"
        )

    bindings = evidence.bindings
    if not isinstance(bindings, tuple) or any(
        not isinstance(binding, JointSignalBinding) for binding in bindings
    ):
        reasons.append("joint bindings must be a tuple of validated binding records")
        bindings = ()
    elif any(
        not all(
            isinstance(value, str)
            for value in (
                binding.envelope_joint,
                binding.coordinate_channel,
                binding.rate_channel,
                binding.actuator_channel,
            )
        )
        for binding in bindings
    ):
        reasons.append("binding channel names must be strings")
        bindings = ()
    by_joint = {binding.envelope_joint: binding for binding in bindings}
    if _names_are_unique(envelope.names) and (
        len(by_joint) != len(bindings) or set(by_joint) != set(envelope.names)
    ):
        reasons.append("bindings must cover every envelope joint exactly once")
    if set(coordinate_names) != {b.coordinate_channel for b in bindings} or set(
        rate_names
    ) != {b.rate_channel for b in bindings}:
        reasons.append(
            "coordinate and rate records contain missing or unclassified joints"
        )
    if any(
        binding.mapping_kind is not JointMappingKind.DIRECT_SCALAR_GENERALIZED_JOINT
        for binding in bindings
    ):
        reasons.append("unknown, coupled, or unsupported joint mapping")
    if any(not _has_provenance(binding.provenance) for binding in bindings):
        reasons.append("each joint binding requires provenance")
    if any(
        binding.coordinate_channel not in coordinate_names
        or binding.rate_channel not in rate_names
        or binding.actuator_channel not in actuator_names
        for binding in bindings
    ):
        reasons.append("a mapped coordinate, rate, or actuator channel is missing")
    if (
        len({b.coordinate_channel for b in bindings}) != len(bindings)
        or len({b.rate_channel for b in bindings}) != len(bindings)
        or len({b.actuator_channel for b in bindings}) != len(bindings)
    ):
        reasons.append("joint mappings must be one-to-one in every signal domain")
    if {b.actuator_channel for b in bindings} != set(expected_actuator_names):
        reasons.append(
            "envelope mappings do not cover every expected actuator exactly once"
        )

    for joint in envelope.names:
        for kind in ("rom", "velocity", "torque"):
            if not _has_provenance(limit_sources.get(f"{joint}:{kind}", "")):
                reasons.append(f"missing provenance for {joint} {kind} limit")
        idx = envelope.names.index(joint)
        if not np.isfinite(envelope.rom_deg[idx]).all():
            reasons.append(f"missing finite ROM bounds for {joint}")

    assumed = _assumed_limits(envelope)
    assumption_limited = bool(assumed)
    if reasons:
        return SampledEvidenceAudit(
            status="indeterminate",
            reasons=tuple(dict.fromkeys(reasons)),
            assumed_limits=assumed,
            assumption_limited=assumption_limited,
            mapping_provenance=(
                evidence.mapping_provenance
                if isinstance(evidence.mapping_provenance, str)
                else ""
            ),
            rate_provenance=(
                evidence.rate_provenance
                if isinstance(evidence.rate_provenance, str)
                else ""
            ),
            torque_provenance=(
                evidence.torque_provenance
                if isinstance(evidence.torque_provenance, str)
                else ""
            ),
            limit_provenance=tuple(
                sorted(
                    (k, v)
                    for k, v in limit_sources.items()
                    if isinstance(k, str) and isinstance(v, str)
                )
            ),
        )

    coordinate_idx = {name: i for i, name in enumerate(coordinate_names)}
    rate_idx = {name: i for i, name in enumerate(rate_names)}
    actuator_idx = {name: i for i, name in enumerate(actuator_names)}
    violations: list[Violation] = []
    for joint in envelope.names:
        binding = by_joint[joint]
        j = envelope.names.index(joint)
        coordinate = np.asarray(evidence.coordinate_deg)[
            :, coordinate_idx[binding.coordinate_channel]
        ]
        rate = np.asarray(evidence.rate_deg_s)[:, rate_idx[binding.rate_channel]]
        torque = np.asarray(evidence.total_applied_actuator_torque_nm)[
            :, actuator_idx[binding.actuator_channel]
        ]
        lower, upper = envelope.rom_deg[j]
        for sample in np.flatnonzero(coordinate < lower):
            violations.append(
                (joint, "rom_lower", int(sample), float(lower - coordinate[sample]))
            )
        for sample in np.flatnonzero(coordinate > upper):
            violations.append(
                (joint, "rom_upper", int(sample), float(coordinate[sample] - upper))
            )
        for sample in np.flatnonzero(np.abs(rate) > envelope.vel_max_deg_s[j]):
            violations.append(
                (
                    joint,
                    "velocity",
                    int(sample),
                    float(abs(rate[sample]) - envelope.vel_max_deg_s[j]),
                )
            )
        cap = np.where(
            torque >= 0.0,
            envelope.tau_max_positive_nm[j],
            envelope.tau_max_negative_nm[j],
        )
        for sample in np.flatnonzero(np.abs(torque) > cap):
            violations.append(
                (joint, "torque", int(sample), float(abs(torque[sample]) - cap[sample]))
            )
    return SampledEvidenceAudit(
        status="violation" if violations else "pass",
        violations=tuple(violations),
        assumed_limits=assumed,
        assumption_limited=assumption_limited,
        mapping_provenance=evidence.mapping_provenance,
        rate_provenance=evidence.rate_provenance,
        torque_provenance=evidence.torque_provenance,
        limit_provenance=tuple(sorted(limit_sources.items())),
    )
