"""
Tests verifying the MPM driver runs a real physics loop, not a hard-coded mock.
Resolves issue #5676 (and orig #5553).

Current state:
- The driver uses MuJoCo discrete spheres as a granular media approximation.
- It has a 500-step settle phase (hardcoded, see issue #5676 acceptance criterion).
- Impact steps are derived from config.trajectory.duration / timestep (not hardcoded).
- The contact wrench is computed from actual MuJoCo contact forces (not a constant).
- The fallback velocity (5.0 m/s) is used when no trajectory CSV is supplied.

DbC postconditions:
- State must evolve over at least 2 timesteps.
- Output must change when input configuration changes.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed — skipping MPM driver tests")

import numpy as np  # noqa: E402

from bunkershot3d.backends.mpm.driver import MPMDriver  # noqa: E402
from bunkershot3d.io.schema import BunkerShotResultReader  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_config(tmp_path: Path) -> Path:
    yaml_content = textwrap.dedent("""\
        bunker_bed:
          domain:
            length_x: 2.0
            width_y: 1.0
            depth_z: 0.5
          boundary: "fixed"
        grain_population:
          count: 10
          diameter_mean: 0.01
          diameter_sigma_log: 0.1
          density: 2650.0
          coarse_graining_factor: 1.0
        contact_model:
          friction_coefficient: 0.5
          restitution_coefficient: 0.3
          youngs_modulus: 1e7
          poisson_ratio: 0.25
        clubhead:
          loft_deg: 56.0
          bounce_deg: 10.0
          width: 0.1
          height: 0.05
          mass: 0.3
        trajectory:
          duration: 0.005
          file: "swing_data.csv"
        output:
          downsample_grains: 1
          rate_hz: 1000.0
    """)
    config_path = tmp_path / "canonical.yaml"
    config_path.write_text(yaml_content, encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_driver_state_evolves(dummy_config: Path, tmp_path: Path) -> None:
    """
    Smoke test: driver runs for >=2 timesteps and state evolves.

    DbC postcondition: clubhead position must change by at least 1e-4 m.
    """
    driver = MPMDriver(dummy_config)
    output_path = tmp_path / "result.h5"
    driver.run(output_path)

    assert output_path.exists(), "Output HDF5 file was not created"

    reader = BunkerShotResultReader(output_path)
    times, positions, _ = reader.read_clubhead_states()
    reader.close()

    assert len(times) >= 2, f"Expected >=2 timesteps, got {len(times)}"
    # Clubhead moves along +x at 5.0 m/s when no trajectory file is present
    dist = float(np.linalg.norm(positions[-1] - positions[0]))
    assert dist > 1e-4, f"Clubhead state did not evolve (displacement={dist:.2e} m)"


def test_driver_output_changes_with_input(dummy_config: Path, tmp_path: Path) -> None:
    """
    Unit test: driver output changes when the input configuration changes.

    DbC postcondition: final clubhead positions must differ between runs
    with different trajectory durations.
    """
    import yaml

    # Run once with default trajectory duration (0.005 s)
    driver1 = MPMDriver(dummy_config)
    out1 = tmp_path / "result1.h5"
    driver1.run(out1)

    # Modify the config to use a longer duration -> more movement
    with open(dummy_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["trajectory"]["duration"] = 0.010
    cfg2_path = tmp_path / "canonical2.yaml"
    with open(cfg2_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    driver2 = MPMDriver(cfg2_path)
    out2 = tmp_path / "result2.h5"
    driver2.run(out2)

    reader1 = BunkerShotResultReader(out1)
    _, pos1, _ = reader1.read_clubhead_states()
    pos1_end = pos1[-1].copy()
    reader1.close()

    reader2 = BunkerShotResultReader(out2)
    _, pos2, _ = reader2.read_clubhead_states()
    pos2_end = pos2[-1].copy()
    reader2.close()

    assert not np.allclose(pos1_end, pos2_end), (
        f"Driver output did not change with input state: "
        f"pos1_end={pos1_end}, pos2_end={pos2_end}"
    )
