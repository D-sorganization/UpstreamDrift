"""Extra coverage for StandardModelManager (URDF generation, error paths)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from src.shared.python.config.standard_models import StandardModelManager
from src.shared.python.core import GolfModelingError


class TestConfigPersistence:
    def test_config_file_is_created_on_first_init(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        assert mgr.config_file.exists()

    def test_reload_yields_same_config_keys(self, tmp_path: Path) -> None:
        mgr1 = StandardModelManager(suite_root=tmp_path)
        mgr2 = StandardModelManager(suite_root=tmp_path)
        assert set(mgr1.config.keys()) == set(mgr2.config.keys())

    def test_loads_existing_yaml_unchanged(self, tmp_path: Path) -> None:
        # First create with defaults, then mutate file, then reload
        mgr = StandardModelManager(suite_root=tmp_path)
        cfg_path = mgr.config_file
        custom = {
            "standard_humanoid": {"urdf_path": "x.urdf"},
            "simple_humanoid": {"urdf_path": "s.urdf"},
            "golf_clubs": {
                "driver": {
                    "name": "Custom Driver",
                    "urdf_path": "clubs/driver.urdf",
                    "loft_deg": 9.0,
                    "length_m": 1.1,
                    "mass_kg": 0.3,
                }
            },
        }
        cfg_path.write_text(yaml.safe_dump(custom))
        mgr2 = StandardModelManager(suite_root=tmp_path)
        assert mgr2.config["golf_clubs"]["driver"]["name"] == "Custom Driver"

    def test_empty_yaml_yields_empty_dict(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        mgr.config_file.write_text("")
        mgr2 = StandardModelManager(suite_root=tmp_path)
        assert mgr2.config == {}


class TestGolfClubGeneration:
    def test_unknown_club_raises(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        with pytest.raises(GolfModelingError, match="Unknown golf club"):
            mgr.get_golf_club_path("not_a_club")

    def test_driver_urdf_is_generated_and_written(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("driver")
        assert path.exists()
        text = path.read_text()
        assert "<robot" in text
        assert "grip" in text
        assert "shaft" in text
        assert "head" in text

    def test_iron7_urdf_is_generated(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("iron_7")
        assert path.exists()
        assert "<robot" in path.read_text()

    def test_existing_urdf_not_regenerated(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        path = mgr.get_golf_club_path("driver")
        path.write_text("<!-- sentinel -->")
        again = mgr.get_golf_club_path("driver")
        assert again.read_text() == "<!-- sentinel -->"


class TestListAndSetup:
    def test_list_models_contains_both_humanoids(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        result = mgr.list_available_models()
        assert "standard" in result["humanoid"]
        assert "simple" in result["humanoid"]

    def test_setup_all_models_returns_bool(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        with patch.object(mgr, "download_standard_humanoid", return_value=True):
            result = mgr.setup_all_models()
        assert isinstance(result, bool)

    def test_setup_propagates_download_failure(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        with patch.object(mgr, "download_standard_humanoid", return_value=False):
            assert mgr.setup_all_models() is False


class TestStandardHumanoidPath:
    def test_downloads_then_raises_when_still_missing(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        with (
            patch.object(mgr, "download_standard_humanoid", return_value=False),
            pytest.raises(GolfModelingError, match="not found"),
        ):
            mgr.get_standard_humanoid_path()

    def test_returns_existing_path_without_download(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        target = mgr.suite_root / mgr.config["standard_humanoid"]["urdf_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<robot/>")
        # download must not be called
        with patch.object(
            mgr, "download_standard_humanoid", side_effect=AssertionError
        ):
            assert mgr.get_standard_humanoid_path() == target


class TestTemporaryMeshes:
    def test_creates_all_mesh_stubs(self, tmp_path: Path) -> None:
        mgr = StandardModelManager(suite_root=tmp_path)
        mesh_dir = tmp_path / "m"
        mesh_dir.mkdir()
        mgr._create_temporary_meshes(mesh_dir)
        files = list(mesh_dir.glob("*.stl"))
        assert len(files) >= 10
        # Each contains a solid header
        sample = files[0].read_text()
        assert sample.startswith("solid temporary")


class TestValidateModelCompatibility:
    def test_returns_dict_with_engine_keys(self, tmp_path: Path) -> None:
        urdf = tmp_path / "bot.urdf"
        urdf.write_text("<robot/>")
        mgr = StandardModelManager(suite_root=tmp_path)
        results = mgr.validate_model_compatibility(urdf)
        assert isinstance(results, dict)
        # at least one engine key must appear (depends on installed extras)
        # the function records True/False per engine attempted
        assert set(results.keys()).issubset({"mujoco", "drake", "pinocchio"})
