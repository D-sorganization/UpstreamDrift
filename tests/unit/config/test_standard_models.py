"""Tests for src.shared.python.config.standard_models (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.shared.python.config.standard_models import StandardModelManager

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestStandardModelManagerInit:
    def test_construct_with_tmp_root(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert mgr.suite_root == tmp_path

    def test_construct_default_suite_root(self) -> None:
        mgr = StandardModelManager()
        assert isinstance(mgr.suite_root, Path)

    def test_config_is_dict(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert isinstance(mgr.config, dict)

    def test_config_has_standard_humanoid(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert "standard_humanoid" in mgr.config

    def test_config_has_golf_clubs(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert "golf_clubs" in mgr.config


# ---------------------------------------------------------------------------
# list_available_models
# ---------------------------------------------------------------------------


class TestListAvailableModels:
    def test_standard_models_returns_dict(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        result = mgr.list_available_models()
        assert isinstance(result, dict)

    def test_humanoid_in_list(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        result = mgr.list_available_models()
        assert "humanoid" in result

    def test_golf_clubs_in_list(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        result = mgr.list_available_models()
        assert "golf_clubs" in result


# ---------------------------------------------------------------------------
# get_golf_club_path
# ---------------------------------------------------------------------------


class TestGetGolfClubPath:
    def test_driver_returns_path(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("driver")
        assert isinstance(path, Path)

    def test_driver_path_ends_with_urdf(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("driver")
        assert path.suffix == ".urdf"

    def test_iron7_returns_path(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("iron_7")
        assert isinstance(path, Path)


# ---------------------------------------------------------------------------
# get_standard_humanoid_path
# ---------------------------------------------------------------------------


class TestGetStandardHumanoidPath:
    def test_raises_when_model_absent(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        mgr = StandardModelManager(suite_root=tmp_path)
        # Prevent download attempt, just test the error path
        with patch.object(mgr, "download_standard_humanoid", return_value=False):
            from src.shared.python.data_io.common_utils import GolfModelingError

            with pytest.raises(GolfModelingError):
                mgr.get_standard_humanoid_path()
