"""Canonical immutable domain contracts and simulator ports.

Follows Design by Contract (DbC), Law of Demeter (LoD), and DRY.
All boundary validation runs regardless of python -O or DBC_LEVEL settings.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1,)


def _validate_not_bool(value: Any, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} cannot be a boolean value, got {value!r}")


def _validate_finite_float(value: Any, name: str) -> float:
    _validate_not_bool(value, name)
    try:
        f_val = float(value)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"{name} must be a real number: {exc}") from exc
    if not math.isfinite(f_val):
        raise ValueError(f"{name} must be finite, got {f_val}")
    return f_val


def _validate_3vector(vec: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(vec, (tuple, list, Sequence)) or len(vec) != 3:
        raise ValueError(f"{name} must be a 3-sequence, got {type(vec)}")
    return (
        _validate_finite_float(vec[0], f"{name}[0]"),
        _validate_finite_float(vec[1], f"{name}[1]"),
        _validate_finite_float(vec[2], f"{name}[2]"),
    )


class SourceKind(str, Enum):
    """Source that produced the shot data."""

    MANUAL = "manual"
    DEMO_PEAK_SPEED = "demo_peak_speed"
    MODEL_CONTACT = "model_contact"
    IMPORTED = "imported"


class ContactStatus(str, Enum):
    """Contact qualification status."""

    NOT_APPLICABLE = "not_applicable"
    UNVERIFIED = "unverified"
    QUALIFIED = "qualified"
    DEMO_ONLY = "demo_only"


class NumericalStatus(str, Enum):
    """Numerical solver convergence status."""

    NOT_APPLICABLE = "not_applicable"
    UNVERIFIED = "unverified"
    CONVERGED = "converged"
    ESTIMATED = "estimated"
    FAILED = "failed"


class ScientificStatus(str, Enum):
    """Scientific validity qualification status."""

    NOT_APPLICABLE = "not_applicable"
    UNVERIFIED = "unverified"
    BENCHMARKED = "benchmarked"
    PEAK_SPEED_HEURISTIC = "peak_speed_heuristic"


@dataclass(frozen=True)
class ShotQualification:
    """Multi-axis qualification state preventing misleading single booleans."""

    contact: ContactStatus
    numerical: NumericalStatus
    scientific: ScientificStatus
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_refs, tuple):
            object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))


@dataclass(frozen=True)
class AimContext:
    """Orientation and alignment context between source and target frames.

    ``source_to_target_rotation`` is a 3x3 orthonormal rotation matrix (RᵀR = I, det R = +1).
    """

    source_to_target_rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    revision: int = 1
    provenance: str = "default_identity"

    def __post_init__(self) -> None:
        _validate_not_bool(self.revision, "revision")
        if not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError(
                f"revision must be a non-negative integer, got {self.revision!r}"
            )
        if not self.provenance or not self.provenance.strip():
            raise ValueError("provenance must be a non-empty string")

        mat = self.source_to_target_rotation
        if len(mat) != 3:
            raise ValueError(
                f"source_to_target_rotation must have 3 rows, got {len(mat)}"
            )
        validated_rows = []
        for i, row in enumerate(mat):
            validated_rows.append(
                _validate_3vector(row, f"source_to_target_rotation[{i}]")
            )
        r = validated_rows

        # Check R * R^T = I
        for i in range(3):
            for j in range(3):
                dot = sum(r[i][k] * r[j][k] for k in range(3))
                expected = 1.0 if i == j else 0.0
                if abs(dot - expected) > 1e-4:
                    raise ValueError(
                        f"source_to_target_rotation must be a proper rotation (orthogonal with det=+1); dot({i},{j})={dot} != {expected}"
                    )

        # Check det(R) = +1 (proper rotation, no reflection)
        det = (
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
        )
        if abs(det - 1.0) > 1e-4:
            raise ValueError(
                f"source_to_target_rotation must be a proper rotation with det(R)=+1, got det={det}"
            )
        object.__setattr__(self, "source_to_target_rotation", tuple(validated_rows))


@dataclass(frozen=True)
class ClubData:
    """Optional club delivery kinematics at impact."""

    club_speed_m_s: float | None = None
    attack_angle_rad: float | None = None
    club_path_rad: float | None = None
    face_to_target_rad: float | None = None
    face_to_path_rad: float | None = None
    dynamic_loft_rad: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "club_speed_m_s",
            "attack_angle_rad",
            "club_path_rad",
            "face_to_target_rad",
            "face_to_path_rad",
            "dynamic_loft_rad",
        ):
            val = getattr(self, name)
            if val is not None:
                _validate_finite_float(val, name)


@dataclass(frozen=True)
class ShotEnvelope:
    """Canonical immutable shot contract across UpstreamDrift simulator ports.

    All physical vectors are in SI units in the right-handed launch frame (+x forward, +y left, +z up).
    """

    schema_version: int
    shot_id: str
    session_id: str
    source_kind: SourceKind
    qualification: ShotQualification
    ball_velocity_m_s: tuple[float, float, float]
    ball_angular_velocity_rad_s: tuple[float, float, float]
    aim_context: AimContext
    created_at_utc: str
    frame: str = "target_local_xyz"
    model_run_id: str | None = None
    trace_digest: str | None = None
    impact_id: str | None = None
    impact_time_s: float | None = None
    club_data: ClubData | None = None

    def __post_init__(self) -> None:
        _validate_not_bool(self.schema_version, "schema_version")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema_version {self.schema_version} is not in supported versions {SUPPORTED_SCHEMA_VERSIONS}"
            )
        if not self.shot_id or not str(self.shot_id).strip():
            raise ValueError("shot_id must be a non-empty string")
        if not self.session_id or not str(self.session_id).strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.source_kind, SourceKind):
            raise TypeError(
                f"source_kind must be a SourceKind enum, got {type(self.source_kind)}"
            )
        if not isinstance(self.qualification, ShotQualification):
            raise TypeError("qualification must be a ShotQualification instance")

        # Validate velocity vector
        vel = _validate_3vector(self.ball_velocity_m_s, "ball_velocity_m_s")
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        if speed <= 0.0:
            raise ValueError(
                "zero-speed event is no strike and cannot be submitted as a shot"
            )
        object.__setattr__(self, "ball_velocity_m_s", vel)

        # Validate angular velocity vector
        spin = _validate_3vector(
            self.ball_angular_velocity_rad_s, "ball_angular_velocity_rad_s"
        )
        object.__setattr__(self, "ball_angular_velocity_rad_s", spin)

        if not isinstance(self.aim_context, AimContext):
            raise TypeError("aim_context must be an AimContext instance")
        if not self.created_at_utc or not str(self.created_at_utc).strip():
            raise ValueError("created_at_utc must be a non-empty ISO timestamp string")

        # Provenance invariants for model-generated shots
        if self.source_kind in (SourceKind.MODEL_CONTACT, SourceKind.DEMO_PEAK_SPEED):
            if not self.model_run_id or not str(self.model_run_id).strip():
                raise ValueError(
                    "model_run_id and trace_digest are required for model sources"
                )
            if not self.trace_digest or not str(self.trace_digest).strip():
                raise ValueError(
                    "model_run_id and trace_digest are required for model sources"
                )
            if not self.impact_id or not str(self.impact_id).strip():
                raise ValueError("impact_id is required for model sources")
            if self.impact_time_s is None:
                raise ValueError("impact_time_s is required for model sources")
            _validate_finite_float(self.impact_time_s, "impact_time_s")
            if self.impact_time_s < 0.0:
                raise ValueError(
                    f"impact_time_s must be non-negative, got {self.impact_time_s}"
                )
        elif self.impact_time_s is not None:
            _validate_finite_float(self.impact_time_s, "impact_time_s")


@dataclass(frozen=True)
class ShotMetadata:
    """Metadata and provenance bundle for creating ShotEnvelope instances."""

    shot_id: str
    session_id: str
    aim_context: AimContext
    created_at_utc: str
    source_kind: SourceKind = SourceKind.MANUAL
    qualification: ShotQualification | None = None
    club_data: ClubData | None = None
    model_run_id: str | None = None
    trace_digest: str | None = None
    impact_id: str | None = None
    impact_time_s: float | None = None


class CapabilityState(str, Enum):
    """Honest capability state without hopeful booleans."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Describes support for a specific simulator capability."""

    state: CapabilityState
    evidence: str
    version: str | None = None


@dataclass(frozen=True)
class SimulatorCapabilities:
    """Declared capabilities of a simulator destination."""

    shot_input: CapabilityDescriptor
    club_data: CapabilityDescriptor
    native_avatar_animation: CapabilityDescriptor
    course_state_feedback: CapabilityDescriptor
    local_trajectory_return: CapabilityDescriptor = field(
        default_factory=lambda: CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="Simulator owns displayed flight; does not return integrated trajectory",
        )
    )
    aim_control: CapabilityDescriptor = field(
        default_factory=lambda: CapabilityDescriptor(
            state=CapabilityState.UNSUPPORTED,
            evidence="Aim control not supported by open connect v1",
        )
    )


class UnsupportedCapabilityError(Exception):
    """Raised when an operation requires a capability that is unsupported or unverified."""

    def __init__(
        self,
        capability_name: str,
        state: CapabilityState,
        evidence: str,
        message: str | None = None,
    ) -> None:
        self.capability_name = capability_name
        self.state = state
        self.evidence = evidence
        msg = message or f"Capability '{capability_name}' is {state.value}: {evidence}"
        super().__init__(msg)


def assert_capability_supported(
    capabilities: SimulatorCapabilities,
    capability_name: str,
) -> CapabilityDescriptor:
    """Assert that a specific simulator capability is SUPPORTED.

    Raises:
        AttributeError: If capability_name is not a recognized capability descriptor.
        UnsupportedCapabilityError: If capability is UNSUPPORTED or UNVERIFIED.
    """
    if not hasattr(capabilities, capability_name):
        raise AttributeError(
            f"'{type(capabilities).__name__}' has no capability descriptor named '{capability_name}'"
        )
    descriptor: CapabilityDescriptor = getattr(capabilities, capability_name)
    if descriptor.state != CapabilityState.SUPPORTED:
        raise UnsupportedCapabilityError(
            capability_name=capability_name,
            state=descriptor.state,
            evidence=descriptor.evidence,
        )
    return descriptor


class ConnectionState(str, Enum):
    """Simulator connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass(frozen=True)
class ConnectionStatus:
    """Status of a simulator connection."""

    state: ConnectionState
    endpoint: str
    message: str = ""


class SubmissionState(str, Enum):
    """Outcome state of a shot submission attempt."""

    CONFIRMED_ACCEPTED = "confirmed_accepted"
    REJECTED = "rejected"
    UNKNOWN_AMBIGUOUS = "unknown_ambiguous"
    FAILED_BEFORE_SEND = "failed_before_send"


@dataclass(frozen=True)
class SubmissionReceipt:
    """Audit receipt for a shot submission."""

    shot_id: str
    session_id: str
    state: SubmissionState
    destination_id: str
    attempt_id: str
    timestamp_utc: str
    raw_response: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class SimulatorEvent:
    """Asynchronous event received from a simulator peer."""

    event_type: str
    payload: dict[str, Any]
    timestamp_utc: str


@runtime_checkable
class SimulatorAdapter(Protocol):
    """Port for simulator destinations (GSPro, Local reference, etc.)."""

    def capabilities(self) -> SimulatorCapabilities:
        """Return the declared capabilities of this adapter."""
        ...

    async def connect(self, config: dict[str, Any]) -> ConnectionStatus:
        """Establish connection to simulator."""
        ...

    async def submit(self, shot: ShotEnvelope) -> SubmissionReceipt:
        """Submit a shot to the simulator."""
        ...

    def events(self) -> AsyncIterator[SimulatorEvent]:
        """Stream asynchronous events from the simulator."""
        ...

    async def disconnect(self) -> None:
        """Cleanly terminate the simulator connection."""
        ...


class SessionState(str, Enum):
    """Lifecycle state of the GolfSessionService."""

    IDLE = "idle"
    PREPARED = "prepared"
    ARMED = "armed"
    SUBMITTING = "submitting"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class PreparedShot:
    """A prepared shot ready for validation, arming, and impact submission."""

    prepared_shot_id: str
    shot: ShotEnvelope
    context_revision: int
    created_at_utc: str
    arm_token: str | None = None
    is_armed: bool = False

    def __post_init__(self) -> None:
        if not self.prepared_shot_id or not str(self.prepared_shot_id).strip():
            raise ValueError("prepared_shot_id must be a non-empty string")
        if not isinstance(self.shot, ShotEnvelope):
            raise TypeError("shot must be a ShotEnvelope instance")
        _validate_not_bool(self.context_revision, "context_revision")
        if not isinstance(self.context_revision, int) or self.context_revision < 0:
            raise ValueError(
                f"context_revision must be a non-negative integer, got {self.context_revision!r}"
            )
        if not self.created_at_utc or not str(self.created_at_utc).strip():
            raise ValueError("created_at_utc must be a non-empty ISO timestamp string")


class ReplayPlaybackState(str, Enum):
    """Playback state of the presentation replay clock."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    SCRUBBING = "scrubbing"


@dataclass(frozen=True)
class ReplayTimingRecord:
    """Latency metrics and timestamps for a single-impact submission."""

    shot_id: str
    impact_clock_time_s: float
    send_attempt_timestamp_utc: str
    response_timestamp_utc: str
    impact_to_send_latency_ms: float
    send_to_response_latency_ms: float
    observed_onset_timestamp_utc: str | None = None

    def __post_init__(self) -> None:
        if not self.shot_id or not str(self.shot_id).strip():
            raise ValueError("shot_id must be a non-empty string")
        _validate_finite_float(self.impact_clock_time_s, "impact_clock_time_s")
        if self.impact_clock_time_s < 0.0:
            raise ValueError("impact_clock_time_s must be non-negative")
        if (
            not self.send_attempt_timestamp_utc
            or not str(self.send_attempt_timestamp_utc).strip()
        ):
            raise ValueError("send_attempt_timestamp_utc must be a non-empty string")
        if (
            not self.response_timestamp_utc
            or not str(self.response_timestamp_utc).strip()
        ):
            raise ValueError("response_timestamp_utc must be a non-empty string")
        _validate_finite_float(
            self.impact_to_send_latency_ms, "impact_to_send_latency_ms"
        )
        if self.impact_to_send_latency_ms < 0.0:
            raise ValueError("impact_to_send_latency_ms must be non-negative")
        _validate_finite_float(
            self.send_to_response_latency_ms, "send_to_response_latency_ms"
        )
        if self.send_to_response_latency_ms < 0.0:
            raise ValueError("send_to_response_latency_ms must be non-negative")


@dataclass(frozen=True)
class ReplayFrame:
    """Presentation frame record capturing model replay status and pose timestamp."""

    timestamp_s: float
    frame_index: int
    model_run_id: str
    is_impact_frame: bool = False
    qualification: ShotQualification | None = None

    def __post_init__(self) -> None:
        _validate_finite_float(self.timestamp_s, "timestamp_s")
        if self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be non-negative")
        _validate_not_bool(self.frame_index, "frame_index")
        if not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not self.model_run_id or not str(self.model_run_id).strip():
            raise ValueError("model_run_id must be a non-empty string")
