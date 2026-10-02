"""Data extraction, validation, and analysis routines for simulation service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.shared.python.dashboard.recorder import GenericPhysicsRecorder

logger = get_logger(__name__)


def validate_simulation_data(
    simulation_data: dict[str, Any],
    expected_frames: int,
    has_controls: bool = False,
    is_mock: bool = False,
) -> None:
    """Validate required channels, non-empty arrays, and length alignment.

    Raises:
        ValueError: If required channels are missing, empty, or misaligned.
    """
    required_channels = (
        "times",
        "joint_positions",
        "joint_velocities",
        "joint_accelerations",
    )
    for channel in required_channels:
        if channel not in simulation_data or len(simulation_data[channel]) == 0:
            raise ValueError(
                f"Simulation failed required channel validation: '{channel}' is missing or empty"
            )

    n_times = len(simulation_data["times"])
    if n_times == 0:
        raise ValueError("Simulation produced zero recorded samples")

    if not is_mock and n_times != expected_frames:
        raise ValueError(
            f"Retained sample count ({n_times}) does not match expected frame count ({expected_frames})"
        )

    for channel in ("joint_positions", "joint_velocities", "joint_accelerations"):
        if channel in simulation_data and len(simulation_data[channel]) != n_times:
            raise ValueError(
                f"Channel '{channel}' length ({len(simulation_data[channel])}) "
                f"does not match times length ({n_times})"
            )

    if has_controls:
        if (
            "control_inputs" not in simulation_data
            or len(simulation_data["control_inputs"]) == 0
        ):
            raise ValueError(
                "Simulation with commanded control inputs produced no recorded control data"
            )
        if len(simulation_data["control_inputs"]) != n_times:
            raise ValueError(
                f"Channel 'control_inputs' length ({len(simulation_data['control_inputs'])}) "
                f"does not match times length ({n_times})"
            )


def extract_simulation_data(
    recorder: GenericPhysicsRecorder,
) -> dict[str, Any]:
    """Extract simulation data from recorder.

    Args:
        recorder: Physics recorder with simulation data

    Returns:
        Dictionary containing simulation data
    """
    if not (recorder is not None):
        raise ValueError("recorder must be provided")
    data: dict[str, Any] = {}

    try:
        times, positions = recorder.get_time_series("joint_positions")
        data["times"] = times.tolist() if hasattr(times, "tolist") else times
        data["joint_positions"] = (
            positions.tolist() if hasattr(positions, "tolist") else positions
        )

        times, velocities = recorder.get_time_series("joint_velocities")
        data["joint_velocities"] = (
            velocities.tolist() if hasattr(velocities, "tolist") else velocities
        )

        times, accelerations = recorder.get_time_series("joint_accelerations")
        data["joint_accelerations"] = (
            accelerations.tolist()
            if hasattr(accelerations, "tolist")
            else accelerations
        )

        try:
            times, controls = recorder.get_time_series("control_inputs")
            if len(controls) == 0:
                times, controls = recorder.get_time_series("joint_torques")
            if len(controls) > 0:
                data["control_inputs"] = (
                    controls.tolist() if hasattr(controls, "tolist") else controls
                )
        except (KeyError, ValueError, AttributeError) as e:
            logger.debug("Control inputs not available: %s", e)

    except (KeyError, ValueError, AttributeError, TypeError) as e:
        logger.warning("Error extracting simulation data: %s", e)

    return data


def perform_simulation_analysis(
    recorder: GenericPhysicsRecorder,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Perform analysis on simulation data with explicit channel availability status (R09).

    Args:
        recorder: Physics recorder with simulation data
        config: Analysis configuration

    Returns:
        Analysis results dict including '_channel_status' and '_status'.
    """
    if not (recorder is not None):
        raise ValueError("recorder must be provided")
    results: dict[str, Any] = {}
    channel_status: dict[str, str] = {}
    requested_count = 0
    success_count = 0

    if config.get("ztcf", False):
        requested_count += 1
        try:
            times, ztcf = recorder.get_time_series("ztcf_accel")
            results["ztcf_acceleration"] = (
                ztcf.tolist() if hasattr(ztcf, "tolist") else ztcf
            )
            channel_status["ztcf_acceleration"] = "available"
            success_count += 1
        except (KeyError, ValueError, AttributeError, TypeError, RuntimeError) as e:
            logger.warning("Error performing ztcf analysis: %s", e)
            channel_status["ztcf_acceleration"] = f"unavailable: {e}"

    if config.get("zvcf", False):
        requested_count += 1
        try:
            times, zvcf = recorder.get_time_series("zvcf_accel")
            results["zvcf_acceleration"] = (
                zvcf.tolist() if hasattr(zvcf, "tolist") else zvcf
            )
            channel_status["zvcf_acceleration"] = "available"
            success_count += 1
        except (KeyError, ValueError, AttributeError, TypeError, RuntimeError) as e:
            logger.warning("Error performing zvcf analysis: %s", e)
            channel_status["zvcf_acceleration"] = f"unavailable: {e}"

    if config.get("track_drift", False):
        requested_count += 1
        try:
            times, drift = recorder.get_time_series("drift_accel")
            results["drift_acceleration"] = (
                drift.tolist() if hasattr(drift, "tolist") else drift
            )
            channel_status["drift_acceleration"] = "available"
            success_count += 1
        except (KeyError, ValueError, AttributeError, TypeError, RuntimeError) as e:
            logger.warning("Error performing drift analysis: %s", e)
            channel_status["drift_acceleration"] = f"unavailable: {e}"

    if requested_count > 0:
        results["_channel_status"] = channel_status
        if success_count == requested_count:
            results["_status"] = "completed"
        elif success_count > 0:
            results["_status"] = "partial"
        else:
            results["_status"] = "failed"
    else:
        results["_status"] = "not_requested"

    return results
