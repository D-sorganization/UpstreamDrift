import numpy as np
from pathlib import Path
from bunkershot3d.kinematics.trajectory import (
    SwingTrajectory,
    generate_reference_trajectory,
)


def test_trajectory_generation_and_loading(tmp_path: Path) -> None:
    csv_path = tmp_path / "reference_swing.csv"
    generate_reference_trajectory(csv_path)
    assert csv_path.exists()

    traj = SwingTrajectory.from_csv(csv_path)
    assert len(traj.time) == 100
    assert traj.positions.shape == (100, 3)

    # Test interpolation at midpoint
    pos, quat, lvel, avel = traj.interpolate(0.05)
    assert np.allclose(pos, [0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(lvel[1], 0.0)  # Y velocity is zero
