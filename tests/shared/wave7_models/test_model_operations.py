"""Tests for model_generation.library._model_operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_generation.converters.urdf_parser import URDFParser
from model_generation.library._model_operations import (
    add_local_model,
    create_editable_copy,
    remove_model,
)
from model_generation.library._model_types import (
    LibraryConfig,
    ModelCategory,
    ModelEntry,
    RepositorySource,
)


class TestAddLocalModel:
    def test_adds_with_metadata(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        entries: dict[str, ModelEntry] = {}
        parser = URDFParser()
        entry = add_local_model(
            entries,
            parser,
            lib_config,
            simple_urdf,
            name="My Model",
            category=ModelCategory.ROBOT_ARM,
            description="d",
            tags=["t1"],
        )
        assert entry.id == "simple"
        assert entry.name == "My Model"
        assert entry.category is ModelCategory.ROBOT_ARM
        assert entry.tags == ["t1"]
        assert entry.link_count == 2
        assert entry.joint_count == 1
        assert entry.dof_count == 1
        assert entry.is_read_only is False
        assert entries["simple"] is entry

    def test_default_name_from_filename(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        entries: dict[str, ModelEntry] = {}
        e = add_local_model(entries, URDFParser(), lib_config, simple_urdf)
        assert e.name == "simple"

    def test_collision_id_disambiguation(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        entries = {"simple": ModelEntry(id="simple", name="existing")}
        e = add_local_model(entries, URDFParser(), lib_config, simple_urdf)
        assert e.id == "simple_1"

    def test_missing_file_raises(
        self, tmp_path: Path, lib_config: LibraryConfig
    ) -> None:
        with pytest.raises(FileNotFoundError):
            add_local_model({}, URDFParser(), lib_config, tmp_path / "no.urdf")

    def test_unparseable_urdf_still_added_with_zero_counts(
        self, tmp_path: Path, lib_config: LibraryConfig
    ) -> None:
        bad = tmp_path / "bad.urdf"
        bad.write_text("<not><valid")
        entries: dict[str, ModelEntry] = {}
        e = add_local_model(entries, URDFParser(), lib_config, bad)
        assert e.link_count == 0
        assert e.joint_count == 0
        assert e.dof_count == 0

    def test_copy_to_library_copies_files(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        # Add a mesh dir alongside the URDF
        mesh_dir = simple_urdf.parent / "meshes"
        mesh_dir.mkdir()
        (mesh_dir / "m.stl").write_bytes(b"stub")

        entries: dict[str, ModelEntry] = {}
        e = add_local_model(
            entries, URDFParser(), lib_config, simple_urdf, copy_to_library=True
        )
        assert e.urdf_path is not None
        assert e.urdf_path.exists()
        # Now under cache_dir/<id>/
        assert lib_config.cache_dir in e.urdf_path.parents
        copied_mesh = e.urdf_path.parent / "meshes" / "m.stl"
        assert copied_mesh.exists()

    def test_persists_index(self, simple_urdf: Path, lib_config: LibraryConfig) -> None:
        entries: dict[str, ModelEntry] = {}
        add_local_model(entries, URDFParser(), lib_config, simple_urdf)
        assert lib_config.index_file.exists()


class TestCreateEditableCopy:
    def test_missing_model_returns_none(self, lib_config: LibraryConfig) -> None:
        assert create_editable_copy({}, lib_config, "missing") is None

    def test_no_urdf_path_returns_none(self, lib_config: LibraryConfig) -> None:
        entries = {
            "x": ModelEntry(
                id="x", name="X", source=RepositorySource.LOCAL, is_cached=True
            )
        }
        assert create_editable_copy(entries, lib_config, "x") is None

    def test_copy_creates_new_entry(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        entries: dict[str, ModelEntry] = {}
        e = add_local_model(entries, URDFParser(), lib_config, simple_urdf, name="Orig")
        new_entry = create_editable_copy(entries, lib_config, e.id, new_name="My Copy")
        assert new_entry is not None
        assert new_entry.id == "my_copy"
        assert new_entry.is_read_only is False
        assert new_entry.urdf_path is not None
        assert new_entry.urdf_path.exists()
        # Default destination under cache_dir/editable
        assert "editable" in str(new_entry.urdf_path)

    def test_copy_to_custom_destination(
        self, simple_urdf: Path, lib_config: LibraryConfig, tmp_path: Path
    ) -> None:
        entries: dict[str, ModelEntry] = {}
        e = add_local_model(entries, URDFParser(), lib_config, simple_urdf)
        dest = tmp_path / "mydest"
        new_entry = create_editable_copy(entries, lib_config, e.id, destination=dest)
        assert new_entry is not None
        assert dest in new_entry.urdf_path.parents

    def test_copy_includes_meshes(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        mesh_dir = simple_urdf.parent / "meshes"
        mesh_dir.mkdir()
        (mesh_dir / "m.stl").write_bytes(b"stub")
        entries: dict[str, ModelEntry] = {}
        e = add_local_model(
            entries, URDFParser(), lib_config, simple_urdf, copy_to_library=True
        )
        new_entry = create_editable_copy(entries, lib_config, e.id)
        assert new_entry is not None
        assert new_entry.mesh_dir is not None
        assert new_entry.mesh_dir.exists()


class TestRemoveModel:
    def test_remove_present(self, simple_urdf: Path, lib_config: LibraryConfig) -> None:
        entries: dict[str, ModelEntry] = {}
        e = add_local_model(entries, URDFParser(), lib_config, simple_urdf)
        assert remove_model(entries, lib_config, e.id) is True
        assert e.id not in entries

    def test_remove_missing(self, lib_config: LibraryConfig) -> None:
        assert remove_model({}, lib_config, "nope") is False

    def test_remove_with_delete_files(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        entries: dict[str, ModelEntry] = {}
        e = add_local_model(
            entries, URDFParser(), lib_config, simple_urdf, copy_to_library=True
        )
        urdf_dir = e.urdf_path.parent
        assert urdf_dir.exists()
        remove_model(entries, lib_config, e.id, delete_files=True)
        assert not urdf_dir.exists()

    def test_delete_files_skipped_outside_cache(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        entries: dict[str, ModelEntry] = {}
        # Without copy_to_library, urdf_path is outside cache_dir
        e = add_local_model(entries, URDFParser(), lib_config, simple_urdf)
        assert remove_model(entries, lib_config, e.id, delete_files=True) is True
        # Original file should still exist (was outside cache)
        assert simple_urdf.exists()

    def test_none_id_raises(self, lib_config: LibraryConfig) -> None:
        with pytest.raises(ValueError):
            remove_model({}, lib_config, None)  # type: ignore[arg-type]
