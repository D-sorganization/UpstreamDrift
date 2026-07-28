"""Regression tests for issue #7999.

Before the fix:

- ``DrainedShearCellExperiment.run_simulation`` was a hand-written line
  (``phi_peak = 20 + 30 * friction``) with no ``use_mock`` flag and an unused
  ``backend`` attribute, so ``calibrate_backend(..., use_mock=False)`` still
  received fabricated shear-cell data;
- ``CalibrationOptimizer`` searched ``[friction, restitution]`` although no
  experiment reads ``restitution_coefficient``: the objective was exactly flat
  in that dimension, so repeated runs returned uniform random draws (0.21-0.87)
  alongside an ``error`` of ~1e-26;
- ``calibrate_all.__main__`` hardcoded ``use_mock=True`` inside an
  ``except Exception`` that swallowed every failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from bunkershot3d.calibration.angle_of_repose import AngleOfReposeExperiment
from bunkershot3d.calibration.drained_shear_cell import DrainedShearCellExperiment
from bunkershot3d.calibration.optimizer import (
    CalibrationOptimizer,
    InertParameterError,
)
from bunkershot3d.exceptions import BackendNotImplementedError

pytestmark = pytest.mark.unit

_CONFIG_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "bunkershot3d"
    / "calibration"
    / "configs"
)


class TestShearCellRequiresExplicitMock:
    @pytest.mark.parametrize("backend", ["chrono", "mpm", "liggghts"])
    def test_real_backend_raises(self, backend: str) -> None:
        """No DEM shear cell exists; asking for one must fail loudly."""
        with pytest.raises(BackendNotImplementedError):
            DrainedShearCellExperiment(backend=backend)

    def test_explicit_use_mock_is_allowed(self) -> None:
        exp = DrainedShearCellExperiment(backend="chrono", use_mock=True)
        phi_peak, phi_res = exp.run_simulation({"friction_coefficient": 0.5})
        assert phi_peak == pytest.approx(35.0)
        assert phi_res == pytest.approx(30.0)

    def test_mock_backend_still_works(self) -> None:
        exp = DrainedShearCellExperiment(backend="mock")
        assert exp.run_simulation({"friction_coefficient": 0.1})[0] == pytest.approx(
            23.0
        )

    def test_calibrate_does_not_invent_a_restitution(self) -> None:
        """It used to return a hardcoded restitution_coefficient of 0.3."""
        best = DrainedShearCellExperiment(backend="mock").calibrate()
        assert "friction_coefficient" in best
        assert "restitution_coefficient" not in best


class TestOptimizerRejectsInertParameters:
    def test_declared_parameters_are_friction_only(self) -> None:
        for experiment in (
            AngleOfReposeExperiment(backend="mock"),
            DrainedShearCellExperiment(backend="mock"),
        ):
            assert experiment.calibrated_parameters == ("friction_coefficient",)

    def test_result_has_no_restitution(self) -> None:
        result = CalibrationOptimizer(
            AngleOfReposeExperiment(backend="mock")
        ).optimize()
        assert set(result) == {"friction_coefficient", "error"}

    def test_repeated_runs_agree(self) -> None:
        """Restitution used to vary 4x across identical runs; friction did not."""
        runs = [
            CalibrationOptimizer(AngleOfReposeExperiment(backend="mock")).optimize()
            for _ in range(3)
        ]
        frictions = [r["friction_coefficient"] for r in runs]
        assert np.allclose(frictions, 0.5, atol=1e-6)
        for run in runs:
            assert len(run) == 2

    def test_inert_parameter_is_rejected(self) -> None:
        """Declaring a parameter the objective ignores must raise."""

        class _InertExperiment:
            calibrated_parameters = ("friction_coefficient", "restitution_coefficient")
            target_angle = 32.0

            def run_simulation(self, params: dict) -> float:
                return 20.0 + params["friction_coefficient"] * 24.0

        optimizer = CalibrationOptimizer(_InertExperiment())
        with pytest.raises(InertParameterError, match="restitution_coefficient"):
            optimizer.optimize()

    def test_sensitivity_reports_positive_effect_for_friction(self) -> None:
        sensitivities = CalibrationOptimizer(
            AngleOfReposeExperiment(backend="mock")
        ).check_sensitivity()
        assert sensitivities["friction_coefficient"] > 0.0


class TestCalibrateAllProvenance:
    def test_written_config_records_how_it_was_produced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import bunkershot3d.calibration.calibrate_all as mod

        monkeypatch.setattr(
            mod,
            "__file__",
            str(tmp_path / "src" / "bunkershot3d" / "calibration" / "calibrate_all.py"),
        )
        path = mod.calibrate_backend("mock", use_mock=True)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert data["provenance"]["method"] == "analytical-mock"
        assert data["provenance"]["calibrated"] == ["friction_coefficient"]
        assert any(
            "restitution_coefficient" in entry
            for entry in data["provenance"]["not_calibrated"]
        )

    def test_non_mock_run_fails_instead_of_fabricating(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import bunkershot3d.calibration.calibrate_all as mod

        monkeypatch.setattr(
            mod,
            "__file__",
            str(tmp_path / "src" / "bunkershot3d" / "calibration" / "calibrate_all.py"),
        )
        with pytest.raises(BackendNotImplementedError):
            mod.calibrate_backend("chrono", use_mock=False)

    def test_main_does_not_hardcode_use_mock(self) -> None:
        """__main__ used to force use_mock=True and swallow every exception."""
        import bunkershot3d.calibration.calibrate_all as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "use_mock=True" not in source
        assert "except Exception" not in source


class TestFabricatedConfigsRemoved:
    @pytest.mark.parametrize("backend", ["chrono", "mpm", "liggghts"])
    def test_optimizer_noise_configs_are_gone(self, backend: str) -> None:
        """The committed restitution values were uniform random draws (#7999)."""
        assert not (_CONFIG_DIR / f"sand_{backend}.yaml").exists()

    def test_configs_directory_explains_the_removal(self) -> None:
        readme = _CONFIG_DIR / "README.md"
        assert readme.exists()
        assert "7999" in readme.read_text(encoding="utf-8")
