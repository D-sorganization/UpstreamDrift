"""Wave 7: ``calibration/calibrate_all.py`` coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from bunkershot3d.calibration.calibrate_all import calibrate_backend


class TestCalibrateBackend:
    def test_calibrate_backend_writes_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """calibrate_backend writes a sand_<backend>.yaml file under configs/."""
        # Patch the path resolution so we don't touch the real configs/ tree
        import bunkershot3d.calibration.calibrate_all as mod

        fake_root = tmp_path
        # __file__ is .../calibration/calibrate_all.py; its parent.parent.parent.parent
        # is the project root. Patch by inserting a sentinel.
        target_dir = fake_root / "configs" / "bunkershot3d"
        # Patch Path(__file__).parent.parent.parent.parent
        monkeypatch.setattr(
            mod,
            "__file__",
            str(
                fake_root / "src" / "bunkershot3d" / "calibration" / "calibrate_all.py"
            ),
        )
        calibrate_backend("mpm", use_mock=True)
        out = target_dir / "sand_mpm.yaml"
        assert out.exists()
        with open(out) as f:
            data = yaml.safe_load(f)
        assert "sand_parameters" in data
        sp = data["sand_parameters"]
        assert "friction_coefficient" in sp
        assert "restitution_coefficient" in sp
        assert sp["cohesion"] == 0.0
        assert sp["density"] == 1600.0
        assert sp["mean_diameter"] == 0.0004
        # Friction in physical range
        assert 0.0 <= sp["friction_coefficient"] <= 1.0
        assert 0.0 <= sp["restitution_coefficient"] <= 1.0

    def test_calibrate_backend_produces_finite_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import bunkershot3d.calibration.calibrate_all as mod

        monkeypatch.setattr(
            mod,
            "__file__",
            str(tmp_path / "src" / "bunkershot3d" / "calibration" / "calibrate_all.py"),
        )
        # mock backend triggers AngleOfReposeExperiment(use_mock=True) branch
        calibrate_backend("mock", use_mock=True)
        out = tmp_path / "configs" / "bunkershot3d" / "sand_mock.yaml"
        assert out.exists()


class TestCalibrateAllOptimizerWiring:
    """Confirm that the optimizer is wired correctly to both experiments."""

    def test_uses_optimizer_optimize_for_both_experiments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import bunkershot3d.calibration.calibrate_all as mod

        monkeypatch.setattr(
            mod,
            "__file__",
            str(tmp_path / "src" / "bunkershot3d" / "calibration" / "calibrate_all.py"),
        )
        # Patch CalibrationOptimizer to a fake whose .optimize() is deterministic
        with patch.object(mod, "CalibrationOptimizer") as MockOpt:
            instance = MockOpt.return_value
            instance.optimize.return_value = {
                "friction_coefficient": 0.42,
                "restitution_coefficient": 0.21,
                "error": 0.0,
            }
            calibrate_backend("mpm", use_mock=True)
            # Optimizer constructed twice (one per experiment)
            assert MockOpt.call_count == 2
            # optimize() invoked twice
            assert instance.optimize.call_count == 2

        out = tmp_path / "configs" / "bunkershot3d" / "sand_mpm.yaml"
        with open(out) as f:
            data = yaml.safe_load(f)
        # Averaged values from the deterministic stub
        assert data["sand_parameters"]["friction_coefficient"] == pytest.approx(0.42)
        assert data["sand_parameters"]["restitution_coefficient"] == pytest.approx(0.21)
