import importlib
import sys

import pytest
import mujoco
from pathlib import Path
from bunkershot3d.backends.chrono.driver import BackendNotImplementedError, ChronoDriver
from bunkershot3d.backends.liggghts.driver import LiggghtsDriver
from bunkershot3d.backends.mpm.driver import MPMDriver

# Detect whether pychrono is available in the test environment
try:
    import pychrono  # noqa: F401

    _PYCHRONO_AVAILABLE = True
except ImportError:
    _PYCHRONO_AVAILABLE = False


@pytest.fixture
def dummy_config(tmp_path: Path) -> Path:
    yaml_content = """
    bunker_bed:
      domain:
        length_x: 2.0
        width_y: 1.0
        depth_z: 0.5
      boundary: "fixed"
    grain_population:
      count: 1000
      diameter_mean: 0.002
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
      file: "swing_data.csv"
    output:
      downsample_grains: 10
      rate_hz: 500.0
    """
    config_path = tmp_path / "canonical.yaml"
    with open(config_path, "w") as f:
        f.write(yaml_content)
    return config_path


def test_chrono_driver_init(dummy_config: Path) -> None:
    driver = ChronoDriver(dummy_config)
    assert driver.config is not None
    assert driver.config.bunker_bed.domain.length_x == 2.0


def test_chrono_driver_raises_without_pychrono(
    monkeypatch: pytest.MonkeyPatch, dummy_config: Path
) -> None:
    """Without pychrono, setup() must raise BackendNotImplementedError, not ImportError."""
    import bunkershot3d.backends.chrono.driver as mod

    # Simulate pychrono being absent by patching the module-level flag
    monkeypatch.setattr(mod, "_HAS_CHRONO", False)

    driver = ChronoDriver(dummy_config)
    with pytest.raises(BackendNotImplementedError, match="pychrono is not installed"):
        driver.setup()


def test_chrono_driver_run_raises_without_setup(
    monkeypatch: pytest.MonkeyPatch, dummy_config: Path
) -> None:
    """run() without setup() must raise BackendNotImplementedError."""
    import bunkershot3d.backends.chrono.driver as mod

    monkeypatch.setattr(mod, "_HAS_CHRONO", False)

    driver = ChronoDriver(dummy_config)
    with pytest.raises(BackendNotImplementedError):
        driver.run("/dev/null")


@pytest.mark.integration
@pytest.mark.skipif(not _PYCHRONO_AVAILABLE, reason="pychrono not installed")
def test_chrono_driver_short_shot(dummy_config: Path, tmp_path: Path) -> None:
    """Integration: run a short simulation with pychrono when available."""
    driver = ChronoDriver(dummy_config)
    driver.setup()
    output_path = tmp_path / "chrono_result.h5"
    driver.run(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_liggghts_driver_init(dummy_config: Path) -> None:
    driver = LiggghtsDriver(dummy_config)
    assert driver.config is not None


def test_mpm_driver_init(dummy_config: Path) -> None:
    driver = MPMDriver(dummy_config)
    assert driver.config is not None


def test_mpm_driver_execution(dummy_config: Path, tmp_path: Path) -> None:
    driver = MPMDriver(dummy_config)
    output_path = tmp_path / "result.h5"
    driver.run(output_path)
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Tests for issue #5553 fixes
# ---------------------------------------------------------------------------


def test_mpm_step_count_from_trajectory_duration(
    dummy_config: Path, tmp_path: Path
) -> None:
    """Step count must equal trajectory.duration / model.opt.timestep."""
    import yaml
    import mujoco

    # Write a config with a known duration
    with open(dummy_config) as f:
        cfg = yaml.safe_load(f)
    cfg["trajectory"]["duration"] = 0.05  # 50 ms
    patched = tmp_path / "patched.yaml"
    with open(patched, "w") as f:
        yaml.dump(cfg, f)

    driver = MPMDriver(patched)
    driver.setup()
    assert driver.model is not None

    dt = driver.model.opt.timestep
    expected_steps = int(round(0.05 / dt))
    n_steps = int(round(driver.config.trajectory.duration / dt))
    assert n_steps == expected_steps
    # Sanity: for dt=0.001 and duration=0.05 we expect 50 steps
    assert n_steps == 50


def test_mpm_contact_wrench_shape(dummy_config: Path, tmp_path: Path) -> None:
    """Contact wrench extraction returns arrays of shape (3,), not the old mock."""
    driver = MPMDriver(dummy_config)
    driver.setup()
    assert driver.model is not None and driver.data is not None

    clubhead_id = mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_BODY, "clubhead")
    force, torque = driver._extract_contact_wrench(clubhead_id)
    assert force.shape == (3,), f"Expected (3,) force, got {force.shape}"
    assert torque.shape == (3,), f"Expected (3,) torque, got {torque.shape}"
