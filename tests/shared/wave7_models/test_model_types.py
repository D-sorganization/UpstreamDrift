"""Tests for model_generation.library._model_types."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_generation.library._model_types import (
    LibraryConfig,
    ModelCategory,
    ModelEntry,
    ModelFormat,
    RepositorySource,
)


class TestEnums:
    def test_model_format_values(self) -> None:
        assert ModelFormat.URDF.value == "urdf"
        assert ModelFormat.MJCF.value == "mjcf"

    def test_model_category_values(self) -> None:
        assert ModelCategory.HUMANOID.value == "humanoid"
        assert ModelCategory.OTHER.value == "other"
        # All members map to lowercase strings
        for cat in ModelCategory:
            assert isinstance(cat.value, str) and cat.value == cat.value.lower()

    def test_repository_source_values(self) -> None:
        assert RepositorySource.LOCAL.value == "local"
        assert RepositorySource.BUNDLED.value == "bundled"
        assert {s.value for s in RepositorySource} >= {
            "local",
            "github",
            "gitlab",
            "url",
            "bundled",
        }


class TestModelEntry:
    def test_defaults(self) -> None:
        e = ModelEntry(id="x", name="X")
        assert e.id == "x"
        assert e.name == "X"
        assert e.description == ""
        assert e.category is ModelCategory.OTHER
        assert e.source is RepositorySource.LOCAL
        assert e.model_format is ModelFormat.URDF
        assert e.tags == []
        assert e.is_read_only is True

    def test_to_dict_roundtrip(self) -> None:
        e = ModelEntry(
            id="m1",
            name="Model1",
            description="d",
            category=ModelCategory.HUMANOID,
            source=RepositorySource.GITHUB,
            source_url="https://example.com/m.urdf",
            model_format=ModelFormat.MJCF,
            urdf_path=Path("/tmp/m.urdf"),
            mesh_dir=Path("/tmp/meshes"),
            tags=["a", "b"],
            link_count=2,
            joint_count=1,
            dof_count=1,
            is_cached=True,
            is_read_only=False,
        )
        d = e.to_dict()
        assert d["category"] == "humanoid"
        assert d["model_format"] == "mjcf"
        assert d["source"] == "github"
        assert d["urdf_path"] == str(Path("/tmp/m.urdf"))
        assert d["mesh_dir"] == str(Path("/tmp/meshes"))
        assert d["tags"] == ["a", "b"]

        e2 = ModelEntry.from_dict(d)
        assert e2.id == e.id
        assert e2.category is ModelCategory.HUMANOID
        assert e2.model_format is ModelFormat.MJCF
        assert e2.source is RepositorySource.GITHUB
        assert e2.urdf_path == Path("/tmp/m.urdf")
        assert e2.mesh_dir == Path("/tmp/meshes")
        assert e2.tags == ["a", "b"]

    def test_to_dict_none_paths(self) -> None:
        e = ModelEntry(id="x", name="X")
        d = e.to_dict()
        assert d["urdf_path"] is None
        assert d["mesh_dir"] is None

    def test_from_dict_minimal(self) -> None:
        e = ModelEntry.from_dict({"id": "x", "name": "X"})
        assert e.id == "x"
        assert e.category is ModelCategory.OTHER
        assert e.source is RepositorySource.LOCAL
        assert e.model_format is ModelFormat.URDF
        assert e.urdf_path is None

    def test_from_dict_invalid_format_falls_back(self) -> None:
        e = ModelEntry.from_dict({"id": "x", "name": "X", "model_format": "junk"})
        assert e.model_format is ModelFormat.URDF

    def test_from_dict_none_data_raises(self) -> None:
        with pytest.raises(ValueError, match="data must be provided"):
            ModelEntry.from_dict(None)  # type: ignore[arg-type]


class TestLibraryConfig:
    def test_defaults_are_under_home(self) -> None:
        cfg = LibraryConfig()
        assert ".model_generation" in str(cfg.cache_dir)
        assert cfg.index_file.name == "index.json"
        assert cfg.auto_cache is True
        assert cfg.cache_meshes is True
        assert cfg.verify_checksums is True
        assert cfg.default_repositories == []

    def test_custom(self, tmp_path: Path) -> None:
        cfg = LibraryConfig(
            cache_dir=tmp_path / "c",
            index_file=tmp_path / "i.json",
            auto_cache=False,
        )
        assert cfg.cache_dir == tmp_path / "c"
        assert cfg.auto_cache is False
