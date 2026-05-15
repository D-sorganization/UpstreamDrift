import pytest
from pathlib import Path
from bunkershot3d.backends.chrono.driver import ChronoDriver
from bunkershot3d.backends.liggghts.driver import LiggghtsDriver
from bunkershot3d.backends.mpm.driver import MPMDriver


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
