"""
Test the common IO schema for the BunkerShot3D results.
"""

import numpy as np
from pathlib import Path

from bunkershot3d.io.schema import BunkerShotResultWriter, BunkerShotResultReader


def test_result_schema_creation_and_reading(tmp_path: Path) -> None:
    """Test that we can create a result file with the correct schema and read it back."""
    filepath = tmp_path / "test_result.h5"

    # Write dummy data
    writer = BunkerShotResultWriter(filepath)

    # 1. Clubhead pose (quaternion) and position at t=0
    writer.write_clubhead_state(
        time=0.0,
        position=np.array([0.0, 0.0, 0.0]),
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # 2. Contact wrench
    writer.write_contact_wrench(
        time=0.0, force=np.array([10.0, 0.0, 0.0]), torque=np.array([0.0, 5.0, 0.0])
    )

    # 3. Downsampled grain state
    positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    velocities = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    writer.write_grain_state(time=0.0, positions=positions, velocities=velocities)

    writer.close()

    # Read the data back
    reader = BunkerShotResultReader(filepath)

    times, positions, quats = reader.read_clubhead_states()
    assert len(times) == 1
    assert times[0] == 0.0
    assert np.allclose(positions[0], [0.0, 0.0, 0.0])
    assert np.allclose(quats[0], [1.0, 0.0, 0.0, 0.0])

    times, forces, torques = reader.read_contact_wrenches()
    assert len(times) == 1
    assert times[0] == 0.0
    assert np.allclose(forces[0], [10.0, 0.0, 0.0])
    assert np.allclose(torques[0], [0.0, 5.0, 0.0])

    grain_times, grain_pos, grain_vel = reader.read_grain_states()
    assert len(grain_times) == 1
    assert grain_times[0] == 0.0
    assert np.allclose(grain_pos[0], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert np.allclose(grain_vel[0], [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])

    reader.close()
