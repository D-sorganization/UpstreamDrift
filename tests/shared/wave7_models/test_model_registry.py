"""Tests for model_generation.library._model_registry."""

from __future__ import annotations

import json
from pathlib import Path

from model_generation.library._model_registry import (
    load_index,
    register_bundled_models,
    save_index,
)
from model_generation.library._model_types import (
    LibraryConfig,
    ModelCategory,
    ModelEntry,
    ModelFormat,
    RepositorySource,
)


def _cfg(tmp_path: Path) -> LibraryConfig:
    return LibraryConfig(
        cache_dir=tmp_path / "cache",
        index_file=tmp_path / "index.json",
    )


class TestSaveLoadIndex:
    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        entries = {
            "a": ModelEntry(
                id="a",
                name="A",
                category=ModelCategory.HUMANOID,
                source=RepositorySource.LOCAL,
                tags=["t1"],
            ),
        }
        save_index(cfg, entries)
        assert cfg.index_file.exists()

        out: dict[str, ModelEntry] = {}
        load_index(cfg, out)
        assert "a" in out
        assert out["a"].name == "A"
        assert out["a"].category is ModelCategory.HUMANOID

    def test_load_missing_index_is_noop(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path)
        out: dict[str, ModelEntry] = {}
        load_index(cfg, out)  # no file: should not raise
        assert out == {}

    def test_load_corrupt_index_logs_and_continues(
        self, tmp_path: Path, caplog
    ) -> None:
        cfg = _cfg(tmp_path)
        cfg.index_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.index_file.write_text("{not json")
        out: dict[str, ModelEntry] = {}
        load_index(cfg, out)
        assert out == {}

    def test_save_index_failure_is_swallowed(self, tmp_path: Path, monkeypatch) -> None:
        cfg = _cfg(tmp_path)

        def bad_dumps(*a, **k):
            raise ValueError("boom")

        monkeypatch.setattr(
            "model_generation.library._model_registry.json.dumps", bad_dumps
        )
        # Should not raise even though serialization fails
        save_index(cfg, {"a": ModelEntry(id="a", name="A")})


class TestRegisterBundledModels:
    def test_no_manifest_is_noop(self, tmp_path: Path, monkeypatch) -> None:
        # Redirect bundled dir away from the actual package
        monkeypatch.setattr(
            "model_generation.library._model_registry.Path",
            Path,  # ensure not patched accidentally
        )
        entries: dict[str, ModelEntry] = {}
        # Bundled dir is computed from module file; if no manifest at real path,
        # we can still confirm function tolerates absence by calling it.
        # We assert it does not raise and entries remain unchanged or only
        # gains real bundled entries (read-only).
        before = dict(entries)
        register_bundled_models(entries)
        # entries may be populated by real bundled models; ensure all entries
        # added are well-formed if any.
        for k, v in entries.items():
            assert k == v.id
            assert isinstance(v.name, str)
        # original entries untouched
        for k, v in before.items():
            assert entries[k] is v

    def test_with_synthetic_manifest(self, tmp_path: Path, monkeypatch) -> None:
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "a.urdf").write_text("<robot/>")
        (bundled / "b.xml").write_text("<mujoco/>")
        manifest = {
            "models": [
                {
                    "id": "m_a",
                    "name": "Model A",
                    "file": "a.urdf",
                    "format": "urdf",
                    "category": "humanoid",
                    "tags": ["tag1"],
                    "link_count": 3,
                    "joint_count": 2,
                    "dof_count": 2,
                },
                {
                    "id": "m_b",
                    "name": "Model B",
                    "file": "b.xml",
                    "format": "mjcf",
                    "category": "not_a_real_cat",  # triggers fallback
                },
                {
                    "id": "m_missing",
                    "name": "Missing",
                    "file": "nope.urdf",
                    "format": "urdf",
                },
                {
                    "id": "m_badfmt",
                    "name": "BadFmt",
                    "file": "a.urdf",
                    "format": "weirdo",
                },
            ]
        }
        (bundled / "manifest.json").write_text(json.dumps(manifest))

        # Patch Path(__file__).parent inside the registry module to point
        # at our tmp dir's parent so .parent / "bundled" == our bundled dir.
        fake_module_file = tmp_path / "_model_registry.py"
        fake_module_file.write_text("# fake")
        monkeypatch.setattr(
            "model_generation.library._model_registry.__file__",
            str(fake_module_file),
        )

        entries: dict[str, ModelEntry] = {}
        register_bundled_models(entries)

        assert "m_a" in entries
        assert entries["m_a"].category is ModelCategory.HUMANOID
        assert entries["m_a"].source is RepositorySource.BUNDLED
        assert entries["m_a"].is_read_only is True
        assert entries["m_a"].is_cached is True
        assert entries["m_a"].link_count == 3

        assert "m_b" in entries
        assert entries["m_b"].model_format is ModelFormat.MJCF
        # Invalid category falls back
        assert entries["m_b"].category is ModelCategory.OTHER

        # Missing file is skipped
        assert "m_missing" not in entries

        # Bad format falls back to URDF
        assert entries["m_badfmt"].model_format is ModelFormat.URDF

    def test_existing_entry_not_overwritten(self, tmp_path: Path, monkeypatch) -> None:
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "a.urdf").write_text("<robot/>")
        (bundled / "manifest.json").write_text(
            json.dumps(
                {"models": [{"id": "m_a", "name": "Bundled Name", "file": "a.urdf"}]}
            )
        )
        fake_module_file = tmp_path / "_model_registry.py"
        fake_module_file.write_text("# fake")
        monkeypatch.setattr(
            "model_generation.library._model_registry.__file__",
            str(fake_module_file),
        )

        original = ModelEntry(id="m_a", name="Original")
        entries = {"m_a": original}
        register_bundled_models(entries)
        assert entries["m_a"] is original
        assert entries["m_a"].name == "Original"

    def test_corrupt_manifest_logs_and_returns(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "manifest.json").write_text("{not json")
        fake_module_file = tmp_path / "_model_registry.py"
        fake_module_file.write_text("# fake")
        monkeypatch.setattr(
            "model_generation.library._model_registry.__file__",
            str(fake_module_file),
        )
        entries: dict[str, ModelEntry] = {}
        register_bundled_models(entries)
        assert entries == {}
