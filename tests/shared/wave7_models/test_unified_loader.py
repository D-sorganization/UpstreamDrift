"""Tests for model_generation.library.unified_loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from model_generation.library.unified_loader import (
    LoadResult,
    ModelFormat,
    UnifiedModelLoader,
    UserPreferences,
    detect_format,
)


class TestUserPreferences:
    def test_defaults(self) -> None:
        p = UserPreferences()
        assert p.default_model_id == "mujoco_humanoid"
        assert p.recent_models == []
        assert p.max_recent == 10
        assert p.show_segments is True

    def test_add_recent_moves_to_front(self) -> None:
        p = UserPreferences()
        p.add_recent("a")
        p.add_recent("b")
        p.add_recent("a")
        assert p.recent_models == ["a", "b"]

    def test_add_recent_trims_to_max(self) -> None:
        p = UserPreferences(max_recent=3)
        for i in range(5):
            p.add_recent(f"m{i}")
        assert len(p.recent_models) == 3
        assert p.recent_models[0] == "m4"

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            UserPreferences().add_recent(None)  # type: ignore[arg-type]

    def test_dict_roundtrip(self) -> None:
        p = UserPreferences(
            default_model_id="x",
            recent_models=["a", "b"],
            show_frames=True,
        )
        d = p.to_dict()
        p2 = UserPreferences.from_dict(d)
        assert p2.default_model_id == "x"
        assert p2.recent_models == ["a", "b"]
        assert p2.show_frames is True

    def test_from_dict_with_missing_keys(self) -> None:
        p = UserPreferences.from_dict({})
        assert p.default_model_id == "mujoco_humanoid"


class TestLoadResult:
    def test_name_from_model(self) -> None:
        class FakeModel:
            name = "robot1"

        r = LoadResult(model=FakeModel())
        assert r.name == "robot1"

    def test_name_from_path(self, tmp_path: Path) -> None:
        p = tmp_path / "myrobot.urdf"
        r = LoadResult(source_path=p)
        assert r.name == "myrobot"

    def test_name_unknown(self) -> None:
        assert LoadResult().name == "unknown"


class TestDetectFormat:
    def test_urdf_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "x.urdf"
        p.write_text("")
        assert detect_format(p) is ModelFormat.URDF

    def test_xacro(self, tmp_path: Path) -> None:
        p = tmp_path / "x.xacro"
        p.write_text("")
        assert detect_format(p) is ModelFormat.URDF

    def test_mjcf_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "x.mjcf"
        p.write_text("")
        assert detect_format(p) is ModelFormat.MJCF

    def test_xml_with_robot_content(self, tmp_path: Path) -> None:
        p = tmp_path / "x.xml"
        p.write_text("<?xml version='1.0'?><robot name='r'/>")
        assert detect_format(p) is ModelFormat.URDF

    def test_xml_with_mujoco_content(self, tmp_path: Path) -> None:
        p = tmp_path / "x.xml"
        p.write_text("<mujoco/>")
        assert detect_format(p) is ModelFormat.MJCF

    def test_xml_unknown_content_defaults_mjcf(self, tmp_path: Path) -> None:
        # By extension map, .xml maps to MJCF when content is ambiguous
        p = tmp_path / "x.xml"
        p.write_text("<other/>")
        assert detect_format(p) is ModelFormat.MJCF

    def test_xml_nonexistent_defaults(self, tmp_path: Path) -> None:
        # Still maps by extension even if file doesn't exist
        p = tmp_path / "missing.xml"
        assert detect_format(p) is ModelFormat.MJCF

    def test_unknown_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "x.txt"
        p.write_text("")
        assert detect_format(p) is ModelFormat.UNKNOWN


class TestUnifiedLoaderPreferences:
    def test_prefs_dir_created(self, tmp_path: Path) -> None:
        d = tmp_path / "prefs"
        loader = UnifiedModelLoader(prefs_dir=d)
        assert d.exists()
        assert loader.preferences.default_model_id == "mujoco_humanoid"

    def test_load_corrupt_prefs(self, tmp_path: Path) -> None:
        d = tmp_path / "prefs"
        d.mkdir()
        (d / "model_explorer_prefs.json").write_text("{garbage")
        loader = UnifiedModelLoader(prefs_dir=d)
        assert loader.preferences.default_model_id == "mujoco_humanoid"

    def test_save_and_reload_prefs(self, tmp_path: Path) -> None:
        d = tmp_path / "prefs"
        loader = UnifiedModelLoader(prefs_dir=d)
        loader.set_default_model("custom_id")
        # New loader picks up persisted change
        loader2 = UnifiedModelLoader(prefs_dir=d)
        assert loader2.preferences.default_model_id == "custom_id"

    def test_set_default_none_raises(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path / "p")
        with pytest.raises(ValueError):
            loader.set_default_model(None)  # type: ignore[arg-type]


class TestUnifiedLoaderBundled:
    def test_get_bundled_model_info_none_raises(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        with pytest.raises(ValueError):
            loader.get_bundled_model_info(None)  # type: ignore[arg-type]

    def test_list_bundled_with_patched_manifest(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        # Patch the cached manifest directly
        loader._bundled_manifest = {
            "models": [{"id": "x", "name": "X", "file": "x.urdf"}]
        }
        items = loader.list_bundled_models()
        assert items == [{"id": "x", "name": "X", "file": "x.urdf"}]

    def test_get_bundled_model_info_present_and_absent(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        loader._bundled_manifest = {
            "models": [{"id": "x", "name": "X", "file": "x.urdf"}]
        }
        assert loader.get_bundled_model_info("x")["name"] == "X"
        assert loader.get_bundled_model_info("missing") is None

    def test_load_bundled_unknown_id(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        loader._bundled_manifest = {"models": []}
        r = loader.load_bundled("nope")
        assert r.success is False
        assert "not found" in (r.error or "")

    def test_load_bundled_file_missing(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        loader._bundled_manifest = {
            "models": [{"id": "x", "name": "X", "file": "missing.urdf"}]
        }
        r = loader.load_bundled("x")
        assert r.success is False
        assert "missing" in (r.error or "").lower()

    def test_corrupt_manifest_logged(self, tmp_path: Path, monkeypatch) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        bundled_dir = tmp_path / "bundled"
        bundled_dir.mkdir()
        (bundled_dir / "manifest.json").write_text("{not json")
        monkeypatch.setattr(loader, "_get_bundled_dir", lambda: bundled_dir)
        assert loader._get_manifest() == {"models": []}


class TestUnifiedLoaderFileLoading:
    def test_load_file_missing(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        r = loader.load_file(tmp_path / "no.urdf")
        assert r.success is False
        assert "not found" in r.error.lower()

    def test_load_file_urdf(self, simple_urdf: Path, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        r = loader.load_file(simple_urdf)
        assert r.success is True
        assert r.source_format is ModelFormat.URDF
        assert r.model is not None
        assert len(r.model.links) == 2

    def test_load_file_none_raises(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        with pytest.raises(ValueError):
            loader.load_file(None)  # type: ignore[arg-type]

    def test_load_default_falls_back(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        loader._bundled_manifest = {"models": []}
        loader._preferences.default_model_id = "custom_missing"
        r = loader.load_default()
        # Falls back to mujoco_humanoid which is also missing — but the
        # function should still return a LoadResult (failed).
        assert isinstance(r, LoadResult)
        assert r.success is False

    def test_convert_to_urdf_failure(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        with patch.object(
            loader._mjcf_converter,
            "mjcf_to_urdf",
            side_effect=ValueError("bad"),
        ):
            assert loader.convert_to_urdf("anything") is None

    def test_convert_to_mjcf_failure(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        with patch.object(
            loader._mjcf_converter,
            "urdf_to_mjcf",
            side_effect=ValueError("bad"),
        ):
            assert loader.convert_to_mjcf("anything") is None

    def test_convert_to_urdf_success(self, tmp_path: Path) -> None:
        loader = UnifiedModelLoader(prefs_dir=tmp_path)
        with patch.object(
            loader._mjcf_converter, "mjcf_to_urdf", return_value="<robot/>"
        ):
            assert loader.convert_to_urdf("x") == "<robot/>"
