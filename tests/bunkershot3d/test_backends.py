import pytest

mujoco = pytest.importorskip("mujoco")
from pathlib import Path

from _bunker_fixtures_8612 import write_config, write_straight_trajectory
from bunkershot3d.backends.chrono.driver import BackendNotImplementedError, ChronoDriver
from bunkershot3d.backends.liggghts.driver import LiggghtsDriver
from bunkershot3d.backends.mpm.driver import MPMDriver

# Detect whether pychrono is available in the test environment
try:
    import pychrono  # noqa: F401

    _PYCHRONO_AVAILABLE = True
except ImportError:
    _PYCHRONO_AVAILABLE = False

_SPEED = 1.0


@pytest.fixture
def dummy_config(tmp_path: Path) -> Path:
    """A runnable config: quartz stiffness (#8612) plus a resolvable swing."""
    write_straight_trajectory(tmp_path / "swing_data.csv", speed=_SPEED, duration=0.02)
    return write_config(
        tmp_path / "canonical.yaml",
        grain_count=200,
        diameter_mean=0.002,
        diameter_sigma_log=0.1,
        duration=0.005,
        rate_hz=500.0,
        trajectory_file="swing_data.csv",
    )


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
# Tests for issue #5553 fixes, updated for #8612
# ---------------------------------------------------------------------------


def test_mpm_step_count_from_trajectory_duration(
    dummy_config: Path, tmp_path: Path
) -> None:
    """Step count follows trajectory.duration and the *derived* timestep.

    Before #8612 the timestep was the value authored in the MJCF (1 ms). It is
    now the Courant-stable step for the actual swing speed, so the step count
    is ``duration / dt`` with ``dt = 0.1 d / v``.
    """
    from bunkershot3d.backends.prescribed_motion import load_trajectory

    driver = MPMDriver(dummy_config)
    driver.setup()
    assert driver.model is not None

    trajectory = load_trajectory(driver.config_path, driver.config)
    plan = driver._plan(trajectory, 200_000)

    expected_dt = 0.1 * driver.config.grain_diameter_mean / _SPEED
    assert plan.dt <= expected_dt * 1.001
    assert plan.n_steps == int(round(driver.config.trajectory_duration / plan.dt))


def test_mpm_contact_wrench_shape(dummy_config: Path, tmp_path: Path) -> None:
    """Contact wrench extraction returns arrays of shape (3,), not the old mock."""
    driver = MPMDriver(dummy_config)
    driver.setup()
    assert driver.model is not None and driver.data is not None

    clubhead_id = mujoco.mj_name2id(driver.model, mujoco.mjtObj.mjOBJ_BODY, "clubhead")
    force, torque = driver._extract_contact_wrench(clubhead_id)
    assert force.shape == (3,), f"Expected (3,) force, got {force.shape}"
    assert torque.shape == (3,), f"Expected (3,) torque, got {torque.shape}"
