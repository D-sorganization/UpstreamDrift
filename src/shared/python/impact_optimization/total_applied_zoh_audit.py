"""Caller-attested total-input audit under explicit left-ZOH assumptions.

Work assumes linear velocity between independently supplied endpoints of each
hold. The inputs, mapping and provenance are caller assertions; neither this
quadrature nor the update-rate check certifies the intervening physical motion.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.analysis.power_work_metrics import _split_trapezoid
from src.shared.python.impact_optimization.option_b_prototype import _finite

Array = NDArray[np.float64]


class AppliedEvidenceKind(StrEnum):
    """Declared origin of the supplied rotational generalized inputs."""

    TOTAL_APPLIED = "total_applied"
    IDEAL_DIRECT_OBSERVED = "ideal_direct_observed"
    COMMAND = "command"
    DELTA = "delta"


@dataclass(frozen=True)
class AppliedZOH:
    """N left-held inputs and independent start/end rates on N+1 boundaries.

    Provenance and mapping are caller-attested, not authenticated. This API
    cannot prove an input was applied merely because it was declared so.
    """

    hold_boundaries_s: Array
    channel_names: tuple[str, ...]
    total_applied_nm: Array
    velocity_start_rad_s: Array
    velocity_end_rad_s: Array
    evidence_kind: AppliedEvidenceKind
    provenance: str
    mapping_provenance: str
    effort_units: str
    velocity_units: str
    hold_semantics: str


@dataclass(frozen=True)
class ZOHLimits:
    """Declared limits; update rate is not continuous slew under ZOH."""

    lower_nm: Array
    upper_nm: Array
    max_update_rate_nm_s: Array | None
    max_l2_effort_nm2_s: float | None
    max_positive_work_j: float | None
    max_abs_signed_work_j: float | None


@dataclass(frozen=True)
class ZOHWorkAudit:
    """Result for this declared input schedule and interpolated velocities."""

    status: Literal["pass", "violation", "indeterminate"]
    violations: tuple[str, ...]
    missing_limits: tuple[str, ...]
    signed_work_j: float
    positive_work_j: float
    negative_work_j: float
    l2_effort_nm2_s: float
    peak_update_rate_nm_s: Array | None
    evidence_kind: AppliedEvidenceKind
    assumption_limited: bool
    continuous_certified: bool = False
    provenance_is_caller_asserted: bool = True
    initial_update_unobserved: bool = True
    scope_note: str = (
        "Pass covers declared holds and update transitions only. Prehistory and "
        "between-hold motion are not certified; rotational mapping, units, "
        "limits and applied-input provenance are caller assertions. This is "
        "not a physical or anatomical feasibility certificate."
    )


def audit_total_applied_zoh(evidence: AppliedZOH, limits: ZOHLimits) -> ZOHWorkAudit:
    """Audit declared total applied torque without clipping or inferring it.

    Positive work is summed per joint before aggregation, preventing
    simultaneous generation and absorption from cancelling. Velocity is
    linearly interpolated within each hold; signed power is split exactly at
    linear zero crossings by the existing per-joint work utility. The slew
    quantity is |delta torque| / time between updates. ZOH torque jumps have
    no finite classical continuous slew, and the prehistory is not observed.
    """
    if not isinstance(evidence, AppliedZOH) or not isinstance(limits, ZOHLimits):
        raise ValueError("applied evidence and limits records are required")
    if not isinstance(
        evidence.evidence_kind, AppliedEvidenceKind
    ) or evidence.evidence_kind not in (
        AppliedEvidenceKind.TOTAL_APPLIED,
        AppliedEvidenceKind.IDEAL_DIRECT_OBSERVED,
    ):
        raise ValueError("total applied input evidence is required")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (evidence.provenance, evidence.mapping_provenance)
    ):
        raise ValueError("input and coordinate mapping provenance are required")
    if (evidence.effort_units, evidence.velocity_units, evidence.hold_semantics) != (
        "N m",
        "rad/s",
        "left_zoh",
    ):
        raise ValueError("rotational N m, rad/s and left_zoh conventions are required")
    names = evidence.channel_names
    if (
        not isinstance(names, tuple)
        or not names
        or any(not isinstance(n, str) or not n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("channel names must be nonempty and unique")
    times = _finite(evidence.hold_boundaries_s, "hold boundaries", 1)
    torque = _finite(evidence.total_applied_nm, "total applied torque", 2)
    v_start = _finite(evidence.velocity_start_rad_s, "start velocity", 2)
    v_end = _finite(evidence.velocity_end_rad_s, "end velocity", 2)
    lower = _finite(limits.lower_nm, "lower torque limit", 1)
    upper = _finite(limits.upper_nm, "upper torque limit", 1)
    slew = (
        None
        if limits.max_update_rate_nm_s is None
        else _finite(limits.max_update_rate_nm_s, "update-rate limit", 1)
    )
    if (
        len(times) < 2
        or not np.isfinite(np.diff(times)).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("hold boundaries must be finite and strictly increasing")
    count, channels = len(times) - 1, len(names)
    if any(x.shape != (count, channels) for x in (torque, v_start, v_end)):
        raise ValueError("each hold requires total torque and both endpoint velocities")
    if any(x.shape != (channels,) for x in (lower, upper)) or (
        slew is not None and slew.shape != (channels,)
    ):
        raise ValueError("limit vectors must match the channel inventory")
    if np.any(lower > upper) or (slew is not None and np.any(slew < 0.0)):
        raise ValueError("torque bounds and update-rate limits must be ordered")
    scalars = (
        limits.max_l2_effort_nm2_s,
        limits.max_positive_work_j,
        limits.max_abs_signed_work_j,
    )
    if any(
        value is not None and (not np.isfinite(value) or value < 0.0)
        for value in scalars
    ):
        raise ValueError("work and effort limits must be finite and nonnegative")

    durations = np.diff(times)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        start_power, end_power = torque * v_start, torque * v_end
        effort = float(np.sum(torque * torque * durations[:, None]))
        rates = np.abs(np.diff(torque, axis=0)) / np.diff(times[:-1])[:, None]
    if not all(
        np.isfinite(x).all() for x in (start_power, end_power, rates)
    ) or not np.isfinite(effort):
        raise ValueError("work or update-rate arithmetic is nonfinite")
    positive, negative = 0.0, 0.0
    for row, dt in enumerate(durations):
        for channel in range(channels):
            pos, neg = _split_trapezoid(
                np.array([start_power[row, channel], end_power[row, channel]]),
                float(dt),
            )
            positive += pos
            negative += neg
    signed = positive + negative
    if not np.isfinite([positive, negative, signed]).all():
        raise ValueError("work integral is nonfinite")
    peak = np.max(rates, axis=0) if count > 1 else None
    violations = []
    if np.any((torque < lower) | (torque > upper)):
        violations.append("torque")
    if peak is not None and slew is not None and np.any(peak > slew):
        violations.append("update_rate")
    if limits.max_l2_effort_nm2_s is not None and effort > limits.max_l2_effort_nm2_s:
        violations.append("l2_effort")
    if limits.max_positive_work_j is not None and positive > limits.max_positive_work_j:
        violations.append("positive_work")
    if (
        limits.max_abs_signed_work_j is not None
        and abs(signed) > limits.max_abs_signed_work_j
    ):
        violations.append("signed_work")
    missing = tuple(
        name
        for name, value in (
            ("update_rate", slew),
            ("l2_effort", limits.max_l2_effort_nm2_s),
            ("positive_work", limits.max_positive_work_j),
            ("signed_work", limits.max_abs_signed_work_j),
        )
        if value is None
    )
    return ZOHWorkAudit(
        status=(
            "violation"
            if violations
            else (
                "indeterminate"
                if peak is None
                or missing
                or evidence.evidence_kind is AppliedEvidenceKind.IDEAL_DIRECT_OBSERVED
                else "pass"
            )
        ),
        violations=tuple(violations),
        missing_limits=missing,
        signed_work_j=signed,
        positive_work_j=positive,
        negative_work_j=negative,
        l2_effort_nm2_s=effort,
        peak_update_rate_nm_s=peak,
        evidence_kind=evidence.evidence_kind,
        assumption_limited=evidence.evidence_kind
        is AppliedEvidenceKind.IDEAL_DIRECT_OBSERVED,
    )
