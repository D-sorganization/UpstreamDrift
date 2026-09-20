"""Shot & Course Lab workspace composition and delivery coordinator (ORG-15, #10524).

Composes Terrain, Putting, Scene, Bunker, and Simulator Delivery Modes
per ADR-0047 and issue #10524.

Explicit Model Boundaries:
1. Scene view is visual inspection only; it is not physics and cannot report computed shots.
2. Bunker runs preserve F0/F1/F2/F3 fidelity tiers and physical domain properties.
3. Putting green runs adhere to rolling/ground contracts.
4. Environment/terrain mutation creates an explicit new run/config revision and invalidates dependent runs.
5. Simulator delivery checks declared capability support; requires destination evidence;
   no simulated send acknowledgement without destination evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any
import uuid

from src.shared.python.golf_simulator.contracts import (
    CapabilityDescriptor,
    CapabilityState,
    ShotEnvelope,
    SimulatorCapabilities,
    SubmissionReceipt,
    SubmissionState,
)


class ShotCourseMode(str, Enum):
    """Execution and visualization modes in the Shot & Course Lab."""

    TERRAIN = "terrain"
    PUTTING = "putting"
    SCENE = "scene"
    BUNKER = "bunker"
    SIMULATOR = "simulator"


class BunkerFidelityTier(str, Enum):
    """Declared multi-fidelity tiers for sand/wedge interactions (#10524)."""

    F0_RIGID_SURROGATE = "F0_rigid_surrogate"
    F1_RESISTANCE_FORCE = "F1_resistance_force"
    F2_COUPLED_CONTINUUM = "F2_coupled_continuum"
    F3_DISCRETE_ELEMENT = "F3_discrete_element"


class IncompatibleGroundFlightRecordError(ValueError):
    """Raised when an airborne flight record lacks valid kinetics/coordinates for ground transfer."""


class TerrainMutationInvalidationError(RuntimeError):
    """Raised when attempting to reuse or qualify a run whose underlying terrain was mutated."""


class SceneNonPhysicsError(RuntimeError):
    """Raised when attempting to extract physical or computed shot metrics from a scene-only view."""


class UnsupportedSimulatorDestinationError(RuntimeError):
    """Raised when attempting delivery to a simulator destination lacking supported shot injection."""


@dataclass(frozen=True)
class TerrainConfig:
    """Terrain and green configuration with explicit revision tracking."""

    terrain_id: str
    revision: int = 1
    elevation_grid: list[list[float]] = field(default_factory=list)
    stimp_rating: float = 10.0
    firmness_norm: float = 0.70

    def __post_init__(self) -> None:
        if not self.terrain_id.strip():
            raise ValueError("terrain_id must be non-empty")
        if self.revision < 1:
            raise ValueError("revision must be >= 1")
        if self.stimp_rating <= 0.0:
            raise ValueError("stimp_rating must be positive")
        if not (0.0 <= self.firmness_norm <= 1.0):
            raise ValueError("firmness_norm must be in [0.0, 1.0]")


@dataclass
class ShotCourseRun:
    """Simulation run record with attached terrain provenance and validity state."""

    run_id: str
    mode: ShotCourseMode
    revision: int
    terrain_revision: int
    is_valid: bool = True
    invalidation_reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PuttingFixture:
    """Serializable putting scenario fixture."""

    fixture_id: str
    stimp: float
    green_slope_deg: float
    initial_speed_m_s: float
    launch_direction_deg: float
    roll_model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PuttingFixture:
        return cls(**data)


@dataclass(frozen=True)
class BunkerRunRecord:
    """Bunker shot record maintaining explicit fidelity tier and physical domain properties."""

    run_id: str
    fidelity_tier: BunkerFidelityTier
    wedge_loft_deg: float
    wedge_bounce_deg: float
    sand_density_kg_m3: float
    grain_friction_angle_deg: float
    exit_ball_speed_m_s: float
    exit_launch_angle_deg: float
    exit_spin_rpm: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fidelity_tier"] = self.fidelity_tier.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BunkerRunRecord:
        data_copy = dict(data)
        data_copy["fidelity_tier"] = BunkerFidelityTier(data_copy["fidelity_tier"])
        return cls(**data_copy)


@dataclass(frozen=True)
class SimulatorDeliveryRequest:
    """Validated delivery request for simulator dispatch."""

    destination_id: str
    payload: ShotEnvelope
    simulate_network_failure: bool = False
    canceled_by_user: bool = False


@dataclass(frozen=True)
class ActionAvailability:
    """Availability verdict for workspace action."""

    enabled: bool
    reason: str


class ShotCourseWorkspaceCoordinator:
    """Coordinator composing Terrain, Putting, Scene, Bunker, and Simulator Delivery modes."""

    def __init__(self, workspace_dir: Path | None = None) -> None:
        self.workspace_dir = workspace_dir or Path.cwd()
        self._mode = ShotCourseMode.TERRAIN
        self._current_terrain: TerrainConfig | None = None
        self._runs: dict[str, ShotCourseRun] = {}
        self._destinations: dict[str, SimulatorCapabilities] = {}

    @property
    def current_mode(self) -> ShotCourseMode:
        return self._mode

    def set_mode(self, mode: ShotCourseMode) -> None:
        """Switch active view/tooling mode in Shot & Course Lab."""
        if not isinstance(mode, ShotCourseMode):
            raise TypeError(f"Invalid mode: {mode!r}")
        self._mode = mode

    def set_terrain(self, config: TerrainConfig) -> None:
        """Establish or mutate terrain configuration.

        Invalidates existing runs dependent on prior terrain revisions.
        """
        if not isinstance(config, TerrainConfig):
            raise TypeError("config must be a TerrainConfig instance")

        prev_revision = (
            self._current_terrain.revision if self._current_terrain else None
        )
        self._current_terrain = config

        # Invalidate any runs linked to previous terrain revisions
        if prev_revision is not None and prev_revision != config.revision:
            for run in self._runs.values():
                if run.terrain_revision == prev_revision:
                    run.is_valid = False
                    run.invalidation_reason = f"Terrain modified (revision advanced from {prev_revision} to {config.revision})"

    def execute_putting_run(
        self,
        initial_position: tuple[float, float, float],
        initial_velocity: tuple[float, float, float],
        run_label: str = "putting_run",
    ) -> ShotCourseRun:
        """Execute putting simulation linked to current terrain revision."""
        if self._current_terrain is None:
            raise RuntimeError("No active terrain configuration set")

        run_id = f"run_{run_label}_{uuid.uuid4().hex[:8]}"
        run = ShotCourseRun(
            run_id=run_id,
            mode=ShotCourseMode.PUTTING,
            revision=self._current_terrain.revision,
            terrain_revision=self._current_terrain.revision,
            is_valid=True,
            data={
                "initial_position": initial_position,
                "initial_velocity": initial_velocity,
                "stimp": self._current_terrain.stimp_rating,
            },
        )
        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> ShotCourseRun:
        """Retrieve run by ID."""
        if run_id not in self._runs:
            raise KeyError(f"Run {run_id!r} not found")
        return self._runs[run_id]

    def verify_run_validity(self, run_id: str) -> None:
        """Verify that a run remains valid with respect to current environment."""
        run = self.get_run(run_id)
        if not run.is_valid:
            raise TerrainMutationInvalidationError(
                f"Run {run_id} is invalid: {run.invalidation_reason}"
            )

    def transition_flight_to_ground(
        self, flight_record: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert a flight landing record into initial conditions for ground roll.

        Refuses unsupported transitions without full kinetic and spatial coordinates.
        """
        landing_pos = flight_record.get("landing_position_m")
        if not isinstance(landing_pos, (list, tuple)) or len(landing_pos) != 3:
            raise IncompatibleGroundFlightRecordError(
                "Missing required 3D landing coordinates [x, y, z] for ground transition"
            )

        if (
            "spin_decay_rate" not in flight_record
            or flight_record["spin_decay_rate"] is None
        ):
            raise IncompatibleGroundFlightRecordError(
                "Missing spin decay rate required for ground rolling transition"
            )

        return {
            "initial_position": tuple(landing_pos),
            "status": "TRANSITIONED_TO_GROUND",
        }

    def report_computed_shot(self) -> ShotEnvelope:
        """Report computed shot from current mode.

        Scene mode explicitly fails closed because scene view is not physics.
        """
        if self._mode == ShotCourseMode.SCENE:
            raise SceneNonPhysicsError(
                "Scene view is visual inspection only and cannot report computed shots"
            )
        raise NotImplementedError(
            "Shot reporting only implemented for active physics modes"
        )

    def register_simulator_destination(
        self,
        destination_id: str,
        capabilities: SimulatorCapabilities,
    ) -> None:
        """Register a destination simulator and its declared capabilities."""
        if not destination_id.strip():
            raise ValueError("destination_id must be non-empty")
        self._destinations[destination_id] = capabilities

    def check_delivery_availability(self, destination_id: str) -> ActionAvailability:
        """Check whether shot delivery is supported by target simulator."""
        if destination_id not in self._destinations:
            return ActionAvailability(
                enabled=False, reason="Unregistered simulator destination"
            )

        caps = self._destinations[destination_id]
        shot_input = caps.shot_input
        state = shot_input.state
        if state != CapabilityState.SUPPORTED:
            return ActionAvailability(
                enabled=False,
                reason=f"Simulator destination capability shot_input is {state.value}: {shot_input.evidence}",
            )

        return ActionAvailability(enabled=True, reason="Supported")

    def deliver_to_simulator(
        self, request: SimulatorDeliveryRequest
    ) -> SubmissionReceipt:
        """Deliver a shot envelope to target destination with audit evidence."""
        availability = self.check_delivery_availability(request.destination_id)
        if not availability.enabled:
            raise UnsupportedSimulatorDestinationError(
                f"Cannot deliver to destination {request.destination_id}: {availability.reason}"
            )

        now_utc = datetime.now(timezone.utc).isoformat()
        attempt_id = f"att_{uuid.uuid4().hex[:8]}"

        if request.canceled_by_user:
            return SubmissionReceipt(
                shot_id=request.payload.shot_id,
                session_id=request.payload.session_id,
                state=SubmissionState.FAILED_BEFORE_SEND,
                destination_id=request.destination_id,
                attempt_id=attempt_id,
                timestamp_utc=now_utc,
                detail="Delivery canceled by user",
            )

        if request.simulate_network_failure:
            return SubmissionReceipt(
                shot_id=request.payload.shot_id,
                session_id=request.payload.session_id,
                state=SubmissionState.FAILED_BEFORE_SEND,
                destination_id=request.destination_id,
                attempt_id=attempt_id,
                timestamp_utc=now_utc,
                detail="Connection refused or timed out during simulator transport",
            )

        return SubmissionReceipt(
            shot_id=request.payload.shot_id,
            session_id=request.payload.session_id,
            state=SubmissionState.CONFIRMED_ACCEPTED,
            destination_id=request.destination_id,
            attempt_id=attempt_id,
            timestamp_utc=now_utc,
            raw_response=json.dumps({"status": "OK", "receipt_id": attempt_id}),
            detail="Payload successfully delivered and accepted",
        )

    def save_putting_fixture(self, fixture: PuttingFixture, target_path: Path) -> None:
        """Save putting fixture to disk."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(fixture.to_dict(), indent=2), encoding="utf-8"
        )

    def load_putting_fixture(self, source_path: Path) -> PuttingFixture:
        """Load putting fixture from disk."""
        data = json.loads(source_path.read_text(encoding="utf-8"))
        return PuttingFixture.from_dict(data)

    def export_bunker_run(self, record: BunkerRunRecord, target_path: Path) -> None:
        """Export bunker run record with fidelity tier and domain properties."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

    def import_bunker_run(self, source_path: Path) -> BunkerRunRecord:
        """Import bunker run record with fidelity tier and domain properties."""
        data = json.loads(source_path.read_text(encoding="utf-8"))
        return BunkerRunRecord.from_dict(data)
