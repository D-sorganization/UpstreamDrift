"""Tests for BunkerShot3D stub backend guards (Issue #5486).

Verifies that unimplemented backends raise NotImplementedError rather than  # tracked: #5486
silently returning mock data.

Note: TestLiggghtsDriverStub has been removed because the LIGGGHTS backend
is now implemented (#5552). See tests/bunkershot3d/test_liggghts_driver.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ChronoDriver stub guard
# ---------------------------------------------------------------------------


class TestChronoDriverStub:
    """ChronoDriver must raise NotImplementedError on setup() and run()."""  # tracked: #5486

    def _make_driver(self) -> object:
        """Create a ChronoDriver with a mocked config."""
        from src.bunkershot3d.backends.chrono.driver import ChronoDriver

        mock_config = MagicMock()
        with patch(
            "src.bunkershot3d.backends.chrono.driver.BunkerShotConfig.from_yaml",
            return_value=mock_config,
        ):
            return ChronoDriver("fake_config.yaml")

    def test_setup_raises_not_implemented(self) -> None:
        driver = self._make_driver()
        with pytest.raises(NotImplementedError):
            driver.setup()

    def test_run_raises_not_implemented(self) -> None:
        driver = self._make_driver()
        with pytest.raises(NotImplementedError):
            driver.run("output.h5")

    def test_setup_error_message_mentions_mpm(self) -> None:
        driver = self._make_driver()
        with pytest.raises(NotImplementedError, match="MPM"):
            driver.setup()


# ---------------------------------------------------------------------------
# Angle of repose calibration stub guard
# ---------------------------------------------------------------------------


class TestAngleOfReposeStub:
    """AngleOfReposeExperiment.run_simulation must raise NotImplementedError."""  # tracked: #5486

    def test_run_simulation_raises_not_implemented(self) -> None:
        from src.bunkershot3d.calibration.angle_of_repose import (
            AngleOfReposeExperiment,
        )

        experiment = AngleOfReposeExperiment()
        with pytest.raises(NotImplementedError):
            experiment.run_simulation({"friction_coefficient": 0.5})

    def test_run_simulation_error_message_references_issue(self) -> None:
        from src.bunkershot3d.calibration.angle_of_repose import (
            AngleOfReposeExperiment,
        )

        experiment = AngleOfReposeExperiment()
        with pytest.raises(NotImplementedError, match="5486"):
            experiment.run_simulation({"friction_coefficient": 0.5})

    def test_compute_angle_of_repose_raises_not_implemented(self) -> None:
        """Module-level convenience function must also raise NotImplementedError."""  # tracked: #5486
        from src.bunkershot3d.calibration.angle_of_repose import (
            compute_angle_of_repose,
        )

        with pytest.raises(NotImplementedError):
            compute_angle_of_repose(friction=0.5)
