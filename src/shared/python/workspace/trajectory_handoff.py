"""Shot Trajectory Handoff Coordinator (ORG-14, #10523).

Connects Swing, Impact, Flight, and Preserved Trajectory Viewers.
Enforces ADR-0047:
- Preserved viewer identities across Shot Tracer (Qt), web BallFlight, and Impact Explorer (ROC).
- Wire format: swing_sim.ball_flight_trajectory/1 (SI units, declared frame FLIGHT_FRAME_ID).
- Honest engine sourcing: delegates to validated swing state providers (manual, MuJoCo);
  refuses unsupported engine sourcing (drake, pinocchio) or unvalidated full-body runs fail-closed.
- Carries environmental and launch conditions across session context.
- Provides atomic transaction and rollback semantics so retained samples survive interchange unaltered.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.api.routes._ball_flight_trajectory_import import (
    ImportedBallFlightTrajectory,
    import_trajectory_record as import_web_trajectory_record,
)
from src.launchers._shot_tracer_trajectory_import import (
    ImportedTrajectoryCurve,
    import_trajectory_record as import_qt_trajectory_record,
)
from src.shared.python.contracts import ensure, require
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.ledger import find_repo_root
from src.shared.python.physics.ball_launch_conditions import (
    EnvironmentalConditions,
    LaunchConditions,
)
from src.shared.python.physics.flight_trajectory_export import (
    BALL_FLIGHT_TRAJECTORY_FORMAT,
    FLIGHT_FRAME_ID,
    pipeline_result_to_trajectory_record,
    trajectory_record_to_json,
)
from src.shared.python.physics.swing_ball_flight_pipeline import (
    PipelineResult,
    SwingBallFlightPipeline,
    SwingState,
)
from src.shared.python.physics.swing_state_providers import (
    SwingStateConfig,
    available_swing_state_providers,
)
from src.shared.python.workspace.results_workspace import (
    ResultArtifactItem,
    ResultCategory,
)

logger = get_logger(__name__)

__all__ = [
    "ExtractionAdapterError",
    "FrameUnitMismatchError",
    "InvalidTrajectoryHashError",
    "ShotTrajectoryHandoffCoordinator",
    "UnsupportedEngineSourceError",
]

SUPPORTED_ENGINES = ("manual", "mujoco")
SUPPORTED_FRAMES = (FLIGHT_FRAME_ID,)


class UnsupportedEngineSourceError(ValueError):
    """Diagnostic error raised when attempting to source swing state from an unsupported or unimplemented engine."""


class InvalidTrajectoryHashError(ValueError):
    """Diagnostic error raised when a trajectory artifact's cryptographic sha256 hash does not match disk bytes."""


class FrameUnitMismatchError(ValueError):
    """Diagnostic error raised when trajectory wire data uses an unsupported or mismatched reference frame or units."""


class ExtractionAdapterError(ValueError):
    """Diagnostic error raised when arbitrary full-body run data lacks a validated extraction adapter for clubhead state."""


class ShotTrajectoryHandoffCoordinator:
    """Coordinates trajectory generation, export, validation, and handoff to specialized viewers."""

    def __init__(self, repo_root: Path | str | None = None) -> None:
        if repo_root is not None:
            self._repo_root = Path(repo_root).resolve()
        else:
            self._repo_root = find_repo_root()
        self._environment: EnvironmentalConditions = EnvironmentalConditions()
        self._active_trajectory: Path | None = None
        self._staged_trajectory: Path | None = None
        self._in_transaction: bool = False
        self._artifacts: dict[str, ResultArtifactItem] = {}

    def set_environment_conditions(self, env: EnvironmentalConditions) -> None:
        """Update the active session environmental conditions."""
        require(
            isinstance(env, EnvironmentalConditions),
            "env must be an instance of EnvironmentalConditions",
        )
        self._environment = env

    def get_environment_conditions(self) -> EnvironmentalConditions:
        """Get the active session environmental conditions."""
        return self._environment

    def get_swing_state(self, engine: str, config: SwingStateConfig) -> SwingState:
        """Produce a SwingState from an honest engine provider."""
        require(bool(engine), "engine name must not be empty")
        require(
            isinstance(config, SwingStateConfig), "config must be a SwingStateConfig"
        )

        engine_normalized = engine.strip().lower()
        providers = {p.provider_id: p for p in available_swing_state_providers()}
        if engine_normalized not in providers:
            raise UnsupportedEngineSourceError(
                f"Engine '{engine}' is not supported. Supported engines are: {sorted(providers.keys())}."
            )
        provider = providers[engine_normalized]
        if not provider.is_available():
            raise UnsupportedEngineSourceError(
                f"Engine source '{engine}' is not supported ({provider.availability_reason()})."
            )
        state = provider.get_swing_state(config)
        ensure(
            state.engine_name == provider.provider_id,
            "engine_name must match provider_id",
        )
        return state

    def extract_swing_state_from_run(
        self, engine: str, run_data: dict[str, Any]
    ) -> SwingState:
        """Extract a validated SwingState from full-body run data.

        Refuses unsupported engines (drake, pinocchio) and unvalidated run data.
        """
        engine_normalized = engine.strip().lower()
        if engine_normalized in ("drake", "pinocchio"):
            raise UnsupportedEngineSourceError(
                f"Full-body engine extraction for '{engine}' is not supported or not implemented. "
                "Arbitrary full-body handoff requires an explicit validated extraction adapter."
            )
        if engine_normalized not in SUPPORTED_ENGINES:
            raise UnsupportedEngineSourceError(
                f"Engine '{engine}' is not supported. Supported engines are: {list(SUPPORTED_ENGINES)}."
            )

        if "unvalidated_raw_dump" in run_data:
            raise ExtractionAdapterError(
                f"Missing required validated extraction adapter for engine '{engine}'. "
                "Arbitrary full-body runs cannot guess clubhead impact state."
            )

        # Check for validated kinematics adapter keys
        if "validated_kinematics" in run_data:
            kin = run_data["validated_kinematics"]
            config = SwingStateConfig(
                clubhead_speed_ms=float(kin.get("clubhead_speed_ms", 45.0)),
                loft_deg=float(kin.get("loft_deg", 10.5)),
                clubhead_mass_kg=float(kin.get("clubhead_mass_kg", 0.200)),
            )
            return self.get_swing_state(engine_normalized, config)

        if "clubhead_velocity" in run_data:
            vel = np.array(run_data["clubhead_velocity"], dtype=float)
            speed = float(np.linalg.norm(vel))
            loft = float(run_data.get("loft_deg", 10.5))
            mass = float(run_data.get("clubhead_mass_kg", 0.200))
            config = SwingStateConfig(
                clubhead_speed_ms=speed,
                loft_deg=loft,
                clubhead_mass_kg=mass,
            )
            return self.get_swing_state(engine_normalized, config)

        if "clubhead_speed_ms" in run_data:
            config = SwingStateConfig(
                clubhead_speed_ms=float(run_data["clubhead_speed_ms"]),
                loft_deg=float(run_data.get("loft_deg", 10.5)),
                clubhead_mass_kg=float(run_data.get("clubhead_mass_kg", 0.200)),
            )
            return self.get_swing_state(engine_normalized, config)

        raise ExtractionAdapterError(
            f"Missing required validated extraction adapter kinematics for engine '{engine}'."
        )

    def _adapt_swing_to_pipeline(self, swing: SwingState) -> SwingState:
        """Align swing kinematics to canonical forward flight frame [x=fwd, y=left, z=up]."""
        if (
            swing.clubhead_velocity[0] > 0
            and abs(swing.clubhead_velocity[1]) < 1e-6
            and abs(swing.clubhead_velocity[2]) < 1e-6
            and swing.clubhead_orientation[0] > 0
        ):
            return swing

        speed = float(np.linalg.norm(swing.clubhead_velocity))
        loft_rad = math.radians(swing.clubhead_loft_deg)
        face_normal = np.array([math.cos(loft_rad), 0.0, math.sin(loft_rad)])

        meta = dict(swing.metadata)
        meta["coordinate_conversion"] = "engine_frame_to_pipeline_strike_frame"
        meta["raw_clubhead_velocity"] = [float(v) for v in swing.clubhead_velocity]
        meta["raw_face_normal"] = [float(n) for n in swing.clubhead_orientation]

        return SwingState(
            clubhead_velocity=np.array([speed, 0.0, 0.0]),
            clubhead_angular_velocity=swing.clubhead_angular_velocity.copy(),
            clubhead_orientation=face_normal,
            clubhead_mass=swing.clubhead_mass,
            clubhead_loft_deg=swing.clubhead_loft_deg,
            clubhead_moi=swing.clubhead_moi,
            impact_offset=(
                swing.impact_offset.copy() if swing.impact_offset is not None else None
            ),
            engine_name=swing.engine_name,
            metadata=meta,
        )

    def simulate_and_export_trajectory(
        self,
        swing_state: SwingState,
        trajectory_name: str,
    ) -> tuple[PipelineResult, Path]:
        """Simulate swing -> impact -> ball flight pipeline and export wire trajectory record."""
        adapted_swing = self._adapt_swing_to_pipeline(swing_state)
        pipeline = SwingBallFlightPipeline(environment=self._environment)
        pipe_result = pipeline.run(adapted_swing)
        record = pipeline_result_to_trajectory_record(
            pipe_result, source_id=trajectory_name
        )
        out_path = self._repo_root / f"{trajectory_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_json = trajectory_record_to_json(record)
        out_path.write_text(raw_json, encoding="utf-8")
        self.validate_and_register_trajectory(out_path, run_id=trajectory_name)
        return pipe_result, out_path

    def simulate_and_export_trajectory_from_run(
        self,
        engine: str,
        run_data: dict[str, Any],
        trajectory_name: str,
    ) -> tuple[PipelineResult, Path]:
        """Extract swing state from run data, then simulate and export."""
        swing_state = self.extract_swing_state_from_run(engine, run_data)
        return self.simulate_and_export_trajectory(swing_state, trajectory_name)

    def load_into_shot_tracer(self, path: Path) -> ImportedTrajectoryCurve:
        """Import wire trajectory record into Shot Tracer (Qt viewer)."""
        target_path = path.resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Trajectory file does not exist: {target_path}")
        return import_qt_trajectory_record(target_path)

    def load_into_ball_flight_web(self, path: Path) -> ImportedBallFlightTrajectory:
        """Import wire trajectory record into BallFlight web route/page model."""
        target_path = path.resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Trajectory file does not exist: {target_path}")
        raw_bytes = target_path.read_bytes()
        try:
            record = json.loads(raw_bytes.decode("utf-8"))
        except Exception as err:
            raise ValueError(
                f"Invalid trajectory JSON in {target_path}: {err}"
            ) from err
        return import_web_trajectory_record(record)

    def load_into_impact_explorer(self, path: Path) -> dict[str, Any]:
        """Import wire trajectory record into Impact Explorer 3D rate-of-closure playback."""
        target_path = path.resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Trajectory file does not exist: {target_path}")
        raw_bytes = target_path.read_bytes()
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as err:
            raise ValueError(
                f"Invalid trajectory JSON in {target_path}: {err}"
            ) from err

        frame_id = data.get("frame_id")
        if frame_id not in SUPPORTED_FRAMES:
            raise FrameUnitMismatchError(
                f"Unsupported frame '{frame_id}'. Expected {FLIGHT_FRAME_ID}."
            )

        prov = data.get("provenance", {})
        return {
            "source_id": data.get("source_id"),
            "model_family": prov.get("model_family"),
            "model_name": prov.get("model_name"),
            "frame_id": frame_id,
            "samples": copy.deepcopy(data.get("samples", [])),
        }

    def validate_and_register_trajectory(
        self, path: Path, run_id: str | None = None
    ) -> ResultArtifactItem:
        """Validate wire format and frame integrity, registering the artifact."""
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file does not exist: {path}")

        raw_bytes = path.read_bytes()
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as err:
            raise ValueError(f"Invalid trajectory JSON in {path}: {err}") from err

        frame_id = data.get("frame_id")
        if frame_id not in SUPPORTED_FRAMES:
            raise FrameUnitMismatchError(
                f"Unsupported frame '{frame_id}'. Expected one of {SUPPORTED_FRAMES}."
            )

        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
        prov = data.get("provenance", {})

        try:
            rel_path = str(path.relative_to(self._repo_root))
        except ValueError:
            rel_path = str(path)

        item = ResultArtifactItem(
            run_id=run_id or path.stem,
            category=ResultCategory.FLIGHT_TRAJECTORY,
            path=rel_path,
            engine=prov.get("model_family"),
            provenance=prov,
            sha256=sha256_hash,
            size_bytes=len(raw_bytes),
        )
        self._artifacts[item.run_id] = item
        return item

    def verify_artifact_integrity(self, item: ResultArtifactItem) -> bool:
        """Verify artifact on disk matches its registered cryptographic sha256 hash."""
        target_path = item.resolve_path(self._repo_root)
        if not target_path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {target_path}")

        current_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
        if item.sha256 is not None and current_hash != item.sha256:
            raise InvalidTrajectoryHashError(
                f"Cryptographic hash mismatch for artifact '{item.run_id}': "
                f"expected {item.sha256}, got {current_hash}."
            )
        return True

    def set_active_trajectory(self, path: Path) -> None:
        """Set the active session trajectory path."""
        self._active_trajectory = path

    def get_active_trajectory(self) -> Path | None:
        """Get the active session trajectory path."""
        return self._active_trajectory

    def begin_session_transaction(self) -> None:
        """Begin an atomic session transaction."""
        self._in_transaction = True
        self._staged_trajectory = None

    def stage_tentative_trajectory(self, path: Path) -> None:
        """Stage a tentative trajectory within an active transaction."""
        if not self._in_transaction:
            raise RuntimeError(
                "Cannot stage trajectory without an active session transaction"
            )
        self._staged_trajectory = path

    def commit_session_transaction(self) -> None:
        """Commit staged trajectory to active session state."""
        if not self._in_transaction:
            raise RuntimeError("No active session transaction to commit")
        if self._staged_trajectory is not None:
            self._active_trajectory = self._staged_trajectory
        self._staged_trajectory = None
        self._in_transaction = False

    def cancel_session_transaction(self) -> None:
        """Cancel the active session transaction, rolling back to earlier trajectory."""
        self._staged_trajectory = None
        self._in_transaction = False
