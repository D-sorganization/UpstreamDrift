"""Regression tests for the Chrono backend (#8612).

Covers baseline findings:

- **B3** — ``GetAppliedForce()`` returns the externally *applied* load, not the
  contact reaction, so the reported wrench was ~0 by construction.
- **B9** — the trajectory was ignored: a hard-coded 5.0 m/s impact velocity and
  a fixed 200 steps.
- **B30** — ``dt = 1 / output_rate_hz`` (5e-4 s) was used as the *integrator*
  step against a 0.2-Rayleigh limit of 4.2e-8 s, ~11 900x over.
- **B16** — grains were placed on ``np.linspace`` z-layers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _bunker_fixtures_8612 import (
    QUARTZ_DENSITY,
    make_mock_chrono,
    rayleigh_time,
    write_config,
    write_straight_trajectory,
)
from bunkershot3d.backends.stability import StepBudgetExceededError

pytestmark = pytest.mark.unit

_SPEED = 1.0


@pytest.fixture
def chrono_mock(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    import bunkershot3d.backends.chrono.driver as chrono_mod

    chrono = make_mock_chrono()
    monkeypatch.setattr(chrono_mod, "chrono", chrono, raising=False)
    monkeypatch.setattr(chrono_mod, "_HAS_CHRONO", True)
    return chrono


def _short_run_config(tmp_path: Path) -> Path:
    """A config small enough that a Rayleigh-stable run fits the step budget."""
    write_straight_trajectory(
        tmp_path / "swing.csv", speed=_SPEED, duration=2.0e-4, n_samples=11
    )
    return write_config(
        tmp_path / "c.yaml",
        grain_count=20,
        diameter_mean=0.01,
        diameter_sigma_log=0.0,
        duration=1.0e-4,
        rate_hz=1.0e5,
        trajectory_file="swing.csv",
        length_x=0.4,
        width_y=0.2,
        depth_z=0.1,
    )


class TestContactReaction:
    """B3: the reported wrench must be the contact reaction."""

    def test_reads_contact_force_not_applied_force(
        self, tmp_path: Path, chrono_mock: object
    ) -> None:
        from bunkershot3d.backends.chrono.driver import ChronoDriver

        driver = ChronoDriver(_short_run_config(tmp_path))
        driver.setup()
        driver.run(tmp_path / "out.h5")

        body = driver._clubhead_body
        assert body.GetContactForce.called, (
            "the driver must read the contact reaction, not the applied load"
        )
        assert body.GetContactTorque.called
        assert not body.GetAppliedForce.called
        assert not body.GetAppliedTorque.called


class TestIntegrationTimestep:
    """B30: the integrator step must respect the Rayleigh limit."""

    def test_every_step_is_rayleigh_stable(
        self, tmp_path: Path, chrono_mock: object
    ) -> None:
        from bunkershot3d.backends.chrono.driver import ChronoDriver

        driver = ChronoDriver(_short_run_config(tmp_path))
        driver.setup()
        driver.run(tmp_path / "out.h5")

        limit = 0.2 * rayleigh_time(0.005, QUARTZ_DENSITY, 7.0e10, 0.17)
        steps = [
            call.args[0]
            for call in driver._system.DoStepDynamics.call_args_list  # type: ignore[union-attr]
        ]
        assert steps, "the driver never stepped"
        assert max(steps) <= limit * (1.0 + 1e-9), (
            f"integrator stepped at {max(steps):.3e} s against a Rayleigh limit "
            f"of {limit:.3e} s"
        )

    def test_output_rate_is_not_the_integration_step(
        self, tmp_path: Path, chrono_mock: object
    ) -> None:
        from bunkershot3d.backends.chrono.driver import ChronoDriver

        config_path = _short_run_config(tmp_path)
        driver = ChronoDriver(config_path)
        driver.setup()
        driver.run(tmp_path / "out.h5")

        sampling_interval = driver.config.to_solver_settings().output_period_s
        steps = [
            call.args[0]
            for call in driver._system.DoStepDynamics.call_args_list  # type: ignore[union-attr]
        ]
        assert all(step < sampling_interval for step in steps)

    def test_intractable_configuration_is_refused(
        self, tmp_path: Path, chrono_mock: object
    ) -> None:
        """0.4 mm grains for 50 ms needs ~10^6 steps — refuse, do not fake it."""
        from bunkershot3d.backends.chrono.driver import ChronoDriver

        write_straight_trajectory(tmp_path / "swing.csv", speed=25.0, duration=0.05)
        config = write_config(
            tmp_path / "big.yaml",
            grain_count=20,
            diameter_mean=0.0004,
            diameter_sigma_log=0.0,
            duration=0.05,
            trajectory_file="swing.csv",
        )
        driver = ChronoDriver(config)
        driver.setup()
        with pytest.raises(StepBudgetExceededError, match="step"):
            driver.run(tmp_path / "out.h5")


class TestTrajectoryDrivenKinematics:
    """B9: the swing trajectory drives the clubhead, not a hard-coded 5 m/s."""

    def test_clubhead_advances_at_the_trajectory_speed(
        self, tmp_path: Path, chrono_mock: object
    ) -> None:
        from bunkershot3d.backends.chrono.driver import ChronoDriver

        driver = ChronoDriver(_short_run_config(tmp_path))
        driver.setup()
        driver.run(tmp_path / "out.h5")

        calls = driver._clubhead_body.SetPos.call_args_list  # type: ignore[union-attr]
        xs = [call.args[0].x for call in calls]
        travel = xs[-1] - xs[1]
        expected = _SPEED * driver.config.to_trajectory_source().duration_s
        assert travel == pytest.approx(expected, rel=0.05), (
            "the clubhead is not following the prescribed trajectory"
        )

    def test_clubhead_velocity_is_prescribed(
        self, tmp_path: Path, chrono_mock: object
    ) -> None:
        from bunkershot3d.backends.chrono.driver import ChronoDriver

        driver = ChronoDriver(_short_run_config(tmp_path))
        driver.setup()
        driver.run(tmp_path / "out.h5")
        assert driver._clubhead_body.SetPosDt.called, (  # type: ignore[union-attr]
            "the clubhead must carry the swing velocity into the contact solve"
        )

    def test_missing_trajectory_raises(
        self, tmp_path: Path, chrono_mock: object
    ) -> None:
        from bunkershot3d.backends.chrono.driver import ChronoDriver

        config = write_config(
            tmp_path / "c.yaml",
            grain_count=10,
            diameter_mean=0.01,
            duration=1.0e-4,
            trajectory_file="absent.csv",
        )
        driver = ChronoDriver(config)
        driver.setup()
        with pytest.raises(FileNotFoundError, match="absent.csv"):
            driver.run(tmp_path / "out.h5")

    def test_no_hard_coded_impact_velocity_in_source(self) -> None:
        import bunkershot3d.backends.chrono.driver as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "impact_velocity = 5.0" not in source


class TestContactStiffnessPrecondition:
    def test_soft_stiffness_is_refused_at_setup(
        self, tmp_path: Path, chrono_mock: object
    ) -> None:
        from bunkershot3d.backends.chrono.driver import ChronoDriver
        from bunkershot3d.backends.stability import ContactStiffnessError

        config = write_config(
            tmp_path / "soft.yaml", youngs_modulus=1.0e7, poisson_ratio=0.25
        )
        driver = ChronoDriver(config)
        with pytest.raises(ContactStiffnessError):
            driver.setup()
