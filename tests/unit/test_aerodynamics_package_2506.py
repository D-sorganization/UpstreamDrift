"""Contract tests for issue #2506: aerodynamics.py split to package.

These tests define acceptance criteria — they run red before the split
and green after.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
AERO_PKG = REPO / "src/shared/python/physics/aerodynamics"
LOC_BUDGET = 300


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestAerodynamicsPackageStructure:
    """aerodynamics/ package files must exist after the split."""

    @pytest.mark.unit
    def test_package_dir_exists(self) -> None:
        assert AERO_PKG.is_dir(), f"Package dir not found: {AERO_PKG}"

    @pytest.mark.unit
    def test_init_exists(self) -> None:
        assert (AERO_PKG / "__init__.py").exists()

    @pytest.mark.unit
    def test_config_module_exists(self) -> None:
        assert (AERO_PKG / "_config.py").exists()

    @pytest.mark.unit
    def test_aerodynamics_package_2506_models_module_exists(self) -> None:
        assert (AERO_PKG / "_models.py").exists()

    @pytest.mark.unit
    def test_wind_module_exists(self) -> None:
        assert (AERO_PKG / "_wind.py").exists()

    @pytest.mark.unit
    def test_environment_module_exists(self) -> None:
        assert (AERO_PKG / "_environment.py").exists()

    @pytest.mark.unit
    def test_engine_module_exists(self) -> None:
        assert (AERO_PKG / "_engine.py").exists()


class TestAerodynamicsFileSizes:
    """Each module in the aerodynamics package must be <= 300 LOC."""

    @pytest.mark.unit
    def test_init_loc(self) -> None:
        loc = _count_lines(AERO_PKG / "__init__.py")
        assert loc <= LOC_BUDGET, f"__init__.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_config_loc(self) -> None:
        loc = _count_lines(AERO_PKG / "_config.py")
        assert loc <= LOC_BUDGET, f"_config.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_aerodynamics_package_2506_models_loc(self) -> None:
        loc = _count_lines(AERO_PKG / "_models.py")
        assert loc <= LOC_BUDGET, f"_models.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_wind_loc(self) -> None:
        loc = _count_lines(AERO_PKG / "_wind.py")
        assert loc <= LOC_BUDGET, f"_wind.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_environment_loc(self) -> None:
        loc = _count_lines(AERO_PKG / "_environment.py")
        assert loc <= LOC_BUDGET, f"_environment.py has {loc} LOC; budget {LOC_BUDGET}"

    @pytest.mark.unit
    def test_engine_loc(self) -> None:
        loc = _count_lines(AERO_PKG / "_engine.py")
        assert loc <= LOC_BUDGET, f"_engine.py has {loc} LOC; budget {LOC_BUDGET}"


class TestAerodynamicsPublicAPI:
    """Public API must be importable from the package root (backward compat)."""

    @pytest.mark.unit
    def test_import_aerodynamics_config(self) -> None:
        from src.shared.python.physics.aerodynamics import AerodynamicsConfig

        assert AerodynamicsConfig is not None

    @pytest.mark.unit
    def test_import_wind_config(self) -> None:
        from src.shared.python.physics.aerodynamics import WindConfig

        assert WindConfig is not None

    @pytest.mark.unit
    def test_import_randomization_config(self) -> None:
        from src.shared.python.physics.aerodynamics import RandomizationConfig

        assert RandomizationConfig is not None

    @pytest.mark.unit
    def test_import_drag_model(self) -> None:
        from src.shared.python.physics.aerodynamics import DragModel

        assert DragModel is not None

    @pytest.mark.unit
    def test_import_lift_model(self) -> None:
        from src.shared.python.physics.aerodynamics import LiftModel

        assert LiftModel is not None

    @pytest.mark.unit
    def test_import_magnus_model(self) -> None:
        from src.shared.python.physics.aerodynamics import MagnusModel

        assert MagnusModel is not None

    @pytest.mark.unit
    def test_import_wind_gust(self) -> None:
        from src.shared.python.physics.aerodynamics import WindGust

        assert WindGust is not None

    @pytest.mark.unit
    def test_import_turbulence_model(self) -> None:
        from src.shared.python.physics.aerodynamics import TurbulenceModel

        assert TurbulenceModel is not None

    @pytest.mark.unit
    def test_import_wind_model(self) -> None:
        from src.shared.python.physics.aerodynamics import WindModel

        assert WindModel is not None

    @pytest.mark.unit
    def test_import_environment_snapshot(self) -> None:
        from src.shared.python.physics.aerodynamics import EnvironmentSnapshot

        assert EnvironmentSnapshot is not None

    @pytest.mark.unit
    def test_import_environment_randomizer(self) -> None:
        from src.shared.python.physics.aerodynamics import EnvironmentRandomizer

        assert EnvironmentRandomizer is not None

    @pytest.mark.unit
    def test_import_aerodynamics_engine(self) -> None:
        from src.shared.python.physics.aerodynamics import AerodynamicsEngine

        assert AerodynamicsEngine is not None
