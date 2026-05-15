import pytest
from pathlib import Path
from bunkershot3d.backends.chrono.driver import ChronoDriver
from bunkershot3d.backends.liggghts.driver import LiggghtsDriver
from bunkershot3d.backends.mpm.driver import MPMDriver


@pytest.fixture
def dummy_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "canonical.yaml"
    with open(config_path, "w") as f:
        f.write("bunker_bed:\n  domain:\n    length_x: 0.4\n")
    return config_path


def test_chrono_driver_init(dummy_config: Path) -> None:
    driver = ChronoDriver(dummy_config)
    assert driver.config is not None
    assert driver.config["bunker_bed"]["domain"]["length_x"] == 0.4


def test_liggghts_driver_init(dummy_config: Path) -> None:
    driver = LiggghtsDriver(dummy_config)
    assert driver.config is not None


def test_mpm_driver_init(dummy_config: Path) -> None:
    driver = MPMDriver(dummy_config)
    assert driver.config is not None
