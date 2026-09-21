"""Local reference simulator adapter wrapping existing ball-flight physics.

Follows TDD, DbC, Law of Demeter, and DRY.
Distinctly separates numerical trajectory simulation from submission receipt auditing.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.shared.python.golf_simulator.contracts import (
    CapabilityDescriptor,
    CapabilityState,
    ConnectionState,
    ConnectionStatus,
    ShotEnvelope,
    SimulatorAdapter,
    SimulatorCapabilities,
    SimulatorEvent,
    SubmissionReceipt,
    SubmissionState,
)
from src.shared.python.golf_simulator.launch_bridge import (
    shot_envelope_to_launch_conditions,
)

if TYPE_CHECKING:
    from src.shared.python.physics.swing_ball_flight_pipeline import (
        FlightSimulatorProtocol,
    )


@dataclass(frozen=True)
class TrajectoryRecord:
    """Audit record for a simulated local trajectory with provenance."""

    shot_id: str
    points: list[Any]
    simulated_at_utc: str
    provenance: str

    def __post_init__(self) -> None:
        if not self.shot_id or not str(self.shot_id).strip():
            raise ValueError("shot_id must be a non-empty string")
        if not self.simulated_at_utc or not str(self.simulated_at_utc).strip():
            raise ValueError("simulated_at_utc must be a non-empty ISO timestamp")
        if not self.provenance or not str(self.provenance).strip():
            raise ValueError("provenance must be a non-empty string")


class LocalReferenceAdapter:
    """Local reference destination adapter satisfying SimulatorAdapter.

    Uses existing UpstreamDrift ball-flight models (e.g. BallFlightSimulator)
    to compute local trajectories with full provenance labeling.
    """

    def __init__(
        self,
        flight_simulator: FlightSimulatorProtocol | None = None,
    ) -> None:
        if flight_simulator is None:
            # Lazy import to prevent importing full physics stack unnecessarily
            from src.shared.python.physics.rust_kernel import is_rust_available

            if is_rust_available():
                from src.shared.python.physics.ball_simulator import (
                    BallFlightSimulator,
                )

                self._flight_simulator: FlightSimulatorProtocol = BallFlightSimulator()
            else:
                from src.shared.python.physics.ball_enhanced_simulator import (
                    EnhancedBallFlightSimulator,
                )

                self._flight_simulator = EnhancedBallFlightSimulator()
        else:
            self._flight_simulator = flight_simulator

        self._connected: bool = False
        self._trajectories: dict[str, TrajectoryRecord] = {}

    def capabilities(self) -> SimulatorCapabilities:
        """Return the declared capabilities of this local reference adapter."""
        return SimulatorCapabilities(
            shot_input=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Local reference flight simulation via FlightSimulatorProtocol",
                version="1.0.0",
            ),
            club_data=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Local reference accepts club delivery parameters",
                version="1.0.0",
            ),
            local_trajectory_return=CapabilityDescriptor(
                state=CapabilityState.SUPPORTED,
                evidence="Returns simulated RK4 trajectory points",
                version="1.0.0",
            ),
            native_avatar_animation=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Avatar animation is viewport responsibility",
            ),
            course_state_feedback=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Local range destination without course feedback",
            ),
            aim_control=CapabilityDescriptor(
                state=CapabilityState.UNSUPPORTED,
                evidence="Local range fixed target frame",
            ),
        )

    async def connect(self, config: dict[str, Any]) -> ConnectionStatus:
        """Connect to the local reference destination."""
        self._connected = True
        return ConnectionStatus(
            state=ConnectionState.CONNECTED,
            endpoint="local_in_memory",
            message="Local reference simulator ready",
        )

    async def submit(self, shot: ShotEnvelope) -> SubmissionReceipt:
        """Simulate trajectory locally and return an audit receipt."""
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        attempt_id = str(uuid.uuid4())

        if not self._connected:
            return SubmissionReceipt(
                shot_id=shot.shot_id,
                session_id=shot.session_id,
                state=SubmissionState.FAILED_BEFORE_SEND,
                destination_id="local_reference",
                attempt_id=attempt_id,
                timestamp_utc=now_utc,
                detail="Adapter not connected",
            )

        # Convert shot envelope to LaunchConditions
        launch = shot_envelope_to_launch_conditions(shot)

        # Simulate trajectory using the underlying flight simulator
        points = self._flight_simulator.simulate_trajectory(launch)

        provenance = (
            f"local_reference:{type(self._flight_simulator).__name__}:"
            f"source={shot.source_kind.value}:schema={shot.schema_version}"
        )

        record = TrajectoryRecord(
            shot_id=shot.shot_id,
            points=list(points),
            simulated_at_utc=now_utc,
            provenance=provenance,
        )
        self._trajectories[shot.shot_id] = record

        return SubmissionReceipt(
            shot_id=shot.shot_id,
            session_id=shot.session_id,
            state=SubmissionState.CONFIRMED_ACCEPTED,
            destination_id="local_reference",
            attempt_id=attempt_id,
            timestamp_utc=now_utc,
            detail=f"Local simulation completed ({len(points)} points)",
        )

    def get_last_trajectory(self, shot_id: str) -> list[Any] | None:
        """Retrieve simulated trajectory points for a shot if available."""
        record = self._trajectories.get(shot_id)
        if record is None:
            return None
        return list(record.points)

    def get_trajectory_record(self, shot_id: str) -> TrajectoryRecord | None:
        """Retrieve the complete trajectory record including provenance."""
        return self._trajectories.get(shot_id)

    async def events(self) -> AsyncIterator[SimulatorEvent]:
        """Stream asynchronous events (empty for local reference)."""
        if False:
            yield  # pragma: no cover

    async def disconnect(self) -> None:
        """Cleanly disconnect the local reference adapter."""
        self._connected = False
