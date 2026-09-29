"""Shared fixtures for wave7 bunkershot3d coverage tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from _bunker_fixtures_8612 import write_config, write_straight_trajectory

#: Prescribed swing speed for the mocked Chrono runs (m/s). Deliberately slow:
#: the Rayleigh-limited timestep makes a tour-speed swing cost ~10^6 steps,
#: which the driver now refuses (#8612, B30).
SWING_SPEED = 1.0


@pytest.fixture
def bunker_config_path(tmp_path: Path) -> Path:
    """A valid, *runnable* BunkerShotConfig YAML written to disk.

    Quartz stiffness (#8612: 1e7 Pa gives 47 % grain interpenetration at tour
    speed) and a resolvable trajectory, since the backends no longer substitute
    a nominal impact velocity when the swing file is missing.
    """
    write_straight_trajectory(
        tmp_path / "swing_data.csv", speed=SWING_SPEED, duration=2.0e-4, n_samples=11
    )
    return write_config(
        tmp_path / "bunker.yaml",
        grain_count=100,
        diameter_mean=0.01,
        diameter_sigma_log=0.1,
        duration=1.0e-4,
        rate_hz=1.0e5,
        trajectory_file="swing_data.csv",
        length_x=2.0,
        width_y=1.0,
        depth_z=0.5,
    )
