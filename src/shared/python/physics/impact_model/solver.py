import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from .models import create_impact_model
from .types import (
    ImpactEvent,
    ImpactModelType,
    ImpactParameters,
    PostImpactState,
    PreImpactState,
)
from .utils import compute_gear_effect_spin, validate_energy_balance

logger = get_logger(__name__)


class ImpactRecorder:
    """Records impact events during simulation.

    Issue #758: Surfaces pre-impact and post-impact states in recorder outputs.
    Provides energy balance checks for each impact.
    """

    def __init__(self) -> None:
        """Initialize impact recorder."""
        self.events: list[ImpactEvent] = []
        self._impact_counter = 0

    def record_impact(
        self,
        timestamp: float,
        pre_state: PreImpactState,
        post_state: PostImpactState,
        params: ImpactParameters,
        model_type: ImpactModelType = ImpactModelType.RIGID_BODY,
    ) -> ImpactEvent:
        """Record an impact event.

        Args:
            timestamp: Simulation time [s]
            pre_state: Pre-impact state
            post_state: Post-impact state
            params: Impact parameters used
            model_type: Type of impact model used

        Returns:
            Recorded ImpactEvent
        """
        if timestamp is None:
            raise ValueError("timestamp must be provided")
        energy_balance = validate_energy_balance(pre_state, post_state, params)

        event = ImpactEvent(
            timestamp=timestamp,
            pre_state=pre_state,
            post_state=post_state,
            energy_balance=energy_balance,
            impact_id=self._impact_counter,
            model_type=model_type,
        )

        self.events.append(event)
        self._impact_counter += 1

        logger.info(
            f"Impact #{event.impact_id} recorded at t={timestamp:.4f}s, "
            f"ball speed: {energy_balance['ball_launch_speed']:.1f} m/s, "
            f"energy loss: {energy_balance['energy_loss_ratio']:.1%}"
        )

        return event

    def get_all_events(self) -> list[ImpactEvent]:
        """Get all recorded impact events."""
        return self.events.copy()

    def export_to_dict(self) -> dict:
        """Export all events as dictionary for JSON serialization.

        Returns:
            Dictionary with all impact events and summary
        """
        events_data = [
            {
                "impact_id": event.impact_id,
                "timestamp": event.timestamp,
                "model_type": event.model_type.name,
                "pre_impact": {
                    "clubhead_velocity": event.pre_state.clubhead_velocity.tolist(),
                    "ball_velocity": event.pre_state.ball_velocity.tolist(),
                    "ball_spin": event.pre_state.ball_angular_velocity.tolist(),
                },
                "post_impact": {
                    "ball_velocity": event.post_state.ball_velocity.tolist(),
                    "ball_spin": event.post_state.ball_angular_velocity.tolist(),
                    "clubhead_velocity": event.post_state.clubhead_velocity.tolist(),
                    "contact_duration": event.post_state.contact_duration,
                    "energy_transfer": event.post_state.energy_transfer,
                },
                "energy_balance": event.energy_balance,
            }
            for event in self.events
        ]

        return {
            "num_impacts": len(self.events),
            "events": events_data,
            "summary": self.get_summary(),
        }

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics for all impacts.

        Returns:
            Dictionary with summary statistics
        """
        if not self.events:
            return {"num_impacts": 0}

        speeds = [e.energy_balance["ball_launch_speed"] for e in self.events]
        losses = [e.energy_balance["energy_loss_ratio"] for e in self.events]

        return {
            "num_impacts": len(self.events),
            "mean_ball_speed": float(np.mean(speeds)),
            "max_ball_speed": float(np.max(speeds)),
            "mean_energy_loss_ratio": float(np.mean(losses)),
        }

    def reset(self) -> None:
        """Clear all recorded events."""
        self.events.clear()
        self._impact_counter = 0


class ImpactSolverAPI:
    """Engine-agnostic API for impact solving.

    Issue #758: Impact solver callable from simulation workflow.
    Provides unified interface for different physics engines.
    """

    def __init__(
        self,
        model_type: ImpactModelType = ImpactModelType.RIGID_BODY,
        params: ImpactParameters | None = None,
    ) -> None:
        """Initialize impact solver.

        Args:
            model_type: Type of impact model to use
            params: Impact parameters (uses defaults if None)
        """
        if model_type is None:
            raise ValueError("model_type must be provided")
        self.model_type = model_type
        self.model = create_impact_model(model_type)
        self.params = params or ImpactParameters()
        self.recorder = ImpactRecorder()

    # fmt: off
    @precondition(  # fmt: skip
        lambda self, timestamp, clubhead_velocity, clubhead_orientation, ball_velocity=None, ball_angular_velocity=None, clubhead_mass=0.200, record=True: (
            clubhead_mass > 0
        ),
        "Clubhead mass must be positive",
    )
    @precondition(  # fmt: skip
        lambda self, timestamp, clubhead_velocity, clubhead_orientation, ball_velocity=None, ball_angular_velocity=None, clubhead_mass=0.200, record=True: (
            timestamp >= 0
        ),
        "Timestamp must be non-negative",
    )
    def solve_impact(
        self,
        timestamp: float,
        clubhead_velocity: np.ndarray,
        clubhead_orientation: np.ndarray,
        ball_velocity: np.ndarray | None = None,
        ball_angular_velocity: np.ndarray | None = None,
        clubhead_mass: float = 0.200,
        record: bool = True,
    ) -> PostImpactState:
    # fmt: on
        """Solve impact and optionally record event.

        Simplified API for common use case.

        Args:
            timestamp: Current simulation time [s]
            clubhead_velocity: Clubhead velocity [m/s] (3,)
            clubhead_orientation: Clubface normal [unitless] (3,)
            ball_velocity: Ball velocity (default: stationary) [m/s] (3,)
            ball_angular_velocity: Ball spin (default: zero) [rad/s] (3,)
            clubhead_mass: Clubhead mass [kg]
            record: Whether to record this impact

        Returns:
            Post-impact state
        """
        if timestamp is None:
            raise ValueError("timestamp must be provided")
        if ball_velocity is None:
            ball_velocity = np.zeros(3)
        if ball_angular_velocity is None:
            ball_angular_velocity = np.zeros(3)

        pre_state = PreImpactState(
            clubhead_velocity=np.asarray(clubhead_velocity),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.asarray(clubhead_orientation),
            ball_position=np.zeros(3),
            ball_velocity=np.asarray(ball_velocity),
            ball_angular_velocity=np.asarray(ball_angular_velocity),
            clubhead_mass=clubhead_mass,
        )

        post_state = self.model.solve(pre_state, self.params)

        if record:
            self.recorder.record_impact(
                timestamp, pre_state, post_state, self.params, self.model_type
            )

        return post_state

    def solve_with_gear_effect(
        self,
        timestamp: float,
        clubhead_velocity: np.ndarray,
        clubhead_orientation: np.ndarray,
        impact_offset: np.ndarray,
        ball_velocity: np.ndarray | None = None,
        clubhead_mass: float = 0.200,
        record: bool = True,
    ) -> PostImpactState:
        """Solve impact with gear effect spin from offset impact.

        Args:
            timestamp: Current simulation time [s]
            clubhead_velocity: Clubhead velocity [m/s] (3,)
            clubhead_orientation: Clubface normal [unitless] (3,)
            impact_offset: Offset from face center [m] (2,) [horizontal, vertical]
            ball_velocity: Ball velocity [m/s] (3,)
            clubhead_mass: Clubhead mass [kg]
            record: Whether to record this impact

        Returns:
            Post-impact state with gear effect spin added
        """
        # Solve base impact
        if timestamp is None:
            raise ValueError("timestamp must be provided")
        post_state = self.solve_impact(
            timestamp,
            clubhead_velocity,
            clubhead_orientation,
            ball_velocity,
            None,
            clubhead_mass,
            record=False,  # Record after adding gear effect
        )

        # Add gear effect spin
        gear_spin = compute_gear_effect_spin(
            impact_offset=np.asarray(impact_offset),
            clubhead_velocity=np.asarray(clubhead_velocity),
            clubface_normal=np.asarray(clubhead_orientation),
            gear_factor=self.params.gear_effect_factor,
            h_scale=self.params.gear_effect_h_scale,
            v_scale=self.params.gear_effect_v_scale,
        )

        # Create modified post-state with gear effect
        modified_post = PostImpactState(
            ball_velocity=post_state.ball_velocity,
            ball_angular_velocity=post_state.ball_angular_velocity + gear_spin,
            clubhead_velocity=post_state.clubhead_velocity,
            clubhead_angular_velocity=post_state.clubhead_angular_velocity,
            contact_duration=post_state.contact_duration,
            energy_transfer=post_state.energy_transfer,
            impact_location=np.asarray(impact_offset),
        )

        if record:
            pre_state = PreImpactState(
                clubhead_velocity=np.asarray(clubhead_velocity),
                clubhead_angular_velocity=np.zeros(3),
                clubhead_orientation=np.asarray(clubhead_orientation),
                ball_position=np.zeros(3),
                ball_velocity=(
                    np.asarray(ball_velocity)
                    if ball_velocity is not None
                    else np.zeros(3)
                ),
                ball_angular_velocity=np.zeros(3),
                clubhead_mass=clubhead_mass,
            )
            self.recorder.record_impact(
                timestamp, pre_state, modified_post, self.params, self.model_type
            )

        return modified_post

    def get_energy_report(self) -> dict:
        """Get energy balance report for all recorded impacts.

        Issue #758: Energy balance checks reported for each impact.

        Returns:
            Dictionary with energy analysis for all impacts
        """
        if not self.recorder.events:
            return {"error": "No impacts recorded"}

        reports = [
            {
                "impact_id": event.impact_id,
                "timestamp": event.timestamp,
                "ke_pre": event.energy_balance["total_ke_pre"],
                "ke_post": event.energy_balance["total_ke_post"],
                "energy_lost": event.energy_balance["energy_lost"],
                "loss_ratio": event.energy_balance["energy_loss_ratio"],
                "ball_speed": event.energy_balance["ball_launch_speed"],
            }
            for event in self.recorder.events
        ]

        # Aggregate statistics
        total_ke_pre = sum(r["ke_pre"] for r in reports)
        total_ke_post = sum(r["ke_post"] for r in reports)

        return {
            "impacts": reports,
            "total_ke_pre": total_ke_pre,
            "total_ke_post": total_ke_post,
            "total_energy_lost": total_ke_pre - total_ke_post,
            "overall_loss_ratio": (
                (total_ke_pre - total_ke_post) / total_ke_pre if total_ke_pre > 0 else 0
            ),
        }

    def validate_cor_behavior(
        self, tolerance: float = 0.05
    ) -> dict[str, bool | float | str | int]:
        """Validate COR behavior across recorded impacts.

        Issue #758: Tests validate COR and spin behavior within tolerance.

        Args:
            tolerance: Acceptable deviation from expected COR

        Returns:
            Validation result with pass/fail and details
        """
        if tolerance is None:
            raise ValueError("tolerance must be provided")
        if not self.recorder.events:
            return {"valid": False, "error": "No impacts recorded"}

        expected_cor = self.params.cor
        measured_cors = []

        for event in self.recorder.events:
            # Compute effective COR from velocities
            v_club_pre = np.linalg.norm(event.pre_state.clubhead_velocity)
            v_ball_pre = np.linalg.norm(event.pre_state.ball_velocity)
            v_club_post = np.linalg.norm(event.post_state.clubhead_velocity)
            v_ball_post = np.linalg.norm(event.post_state.ball_velocity)

            # COR = (v_ball_post - v_club_post) / (v_club_pre - v_ball_pre)
            approach = v_club_pre - v_ball_pre
            if approach > 0.1:  # Avoid division by small number
                separation = v_ball_post - v_club_post
                measured_cor = separation / approach
                measured_cors.append(measured_cor)

        if not measured_cors:
            return {"valid": False, "error": "Could not compute COR"}

        mean_cor = float(np.mean(measured_cors))
        deviation = abs(mean_cor - expected_cor)

        return {
            "valid": deviation <= tolerance,
            "expected_cor": expected_cor,
            "measured_cor_mean": mean_cor,
            "deviation": deviation,
            "tolerance": tolerance,
            "num_samples": len(measured_cors),
        }

    def validate_spin_behavior(
        self, max_spin_rpm: float = 10000
    ) -> dict[str, bool | float | str | int]:
        """Validate spin behavior is within physical limits.

        Issue #758: Tests validate COR and spin behavior within tolerance.

        Args:
            max_spin_rpm: Maximum acceptable spin rate [RPM]

        Returns:
            Validation result with pass/fail and details
        """
        if max_spin_rpm is None:
            raise ValueError("max_spin_rpm must be provided")
        if not self.recorder.events:
            return {"valid": False, "error": "No impacts recorded"}

        max_spin_rad = max_spin_rpm * 2 * np.pi / 60  # Convert to rad/s

        spins = [
            np.linalg.norm(event.post_state.ball_angular_velocity)
            for event in self.recorder.events
        ]

        max_observed = float(np.max(spins))
        max_observed_rpm = max_observed * 60 / (2 * np.pi)

        return {
            "valid": max_observed <= max_spin_rad,
            "max_observed_rpm": max_observed_rpm,
            "max_allowed_rpm": max_spin_rpm,
            "num_samples": len(spins),
        }

    def reset(self) -> None:
        """Reset solver state and clear recorded impacts."""
        self.recorder.reset()
