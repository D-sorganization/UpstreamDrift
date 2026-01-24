import numpy as np

from src.shared.python.engine_availability import (
    PINOCCHIO_AVAILABLE,
    skip_if_unavailable,
)

# Skip test module if pinocchio is not installed
pytestmark = skip_if_unavailable("pinocchio")

if PINOCCHIO_AVAILABLE:
    from src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui import (
        PinocchioRecorder,
    )


def test_pinocchio_recorder_basic():
    recorder = PinocchioRecorder()
    assert recorder.get_num_frames() == 0

    recorder.start_recording()

    # Record a frame
    q = np.zeros(7)
    v = np.zeros(6)
    tau = np.zeros(6)

    recorder.record_frame(
        time=0.1, q=q, v=v, tau=tau, kinetic_energy=10.0, potential_energy=5.0
    )

    assert recorder.get_num_frames() == 1

    recorder.stop_recording()
    assert recorder.is_recording is False

    # Test data retrieval
    times, positions = recorder.get_time_series("joint_positions")
    assert len(times) == 1
    assert times[0] == 0.1
    assert np.allclose(positions[0], q)

    # Test derived/default fields
    times, speeds = recorder.get_time_series("club_head_speed")
    assert len(speeds) == 1
    assert speeds[0] == 0.0


def test_pinocchio_recorder_with_club_data():
    recorder = PinocchioRecorder()
    recorder.start_recording()

    q = np.zeros(7)
    v = np.zeros(6)
    club_pos = np.array([1.0, 2.0, 3.0])
    club_vel = np.array([0.1, 0.0, 0.0])

    recorder.record_frame(
        time=0.1, q=q, v=v, club_head_position=club_pos, club_head_velocity=club_vel
    )

    times, pos = recorder.get_time_series("club_head_position")
    assert np.allclose(pos[0], club_pos)

    times, speed = recorder.get_time_series("club_head_speed")
    assert speed[0] == 0.1


def test_recorder_empty():
    recorder = PinocchioRecorder()
    t, v = recorder.get_time_series("joint_positions")
    assert len(t) == 0
    assert len(v) == 0
