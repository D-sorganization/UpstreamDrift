"""Tests for model_generation.library._model_loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from model_generation.converters.urdf_parser import URDFParser
from model_generation.library._model_loader import download_model, load_model
from model_generation.library._model_types import (
    LibraryConfig,
    ModelEntry,
    ModelFormat,
    RepositorySource,
)


class TestLoadModel:
    def test_unknown_id_returns_none(self, lib_config: LibraryConfig) -> None:
        assert load_model({}, URDFParser(), lib_config, "missing") is None

    def test_none_id_raises(self, lib_config: LibraryConfig) -> None:
        with pytest.raises(ValueError):
            load_model({}, URDFParser(), lib_config, None)  # type: ignore[arg-type]

    def test_local_model_loads(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        entries = {
            "m": ModelEntry(
                id="m",
                name="M",
                source=RepositorySource.LOCAL,
                urdf_path=simple_urdf,
                is_cached=True,
            )
        }
        result = load_model(entries, URDFParser(), lib_config, "m")
        assert result is not None
        assert len(result.links) == 2

    def test_missing_path_returns_none(self, lib_config: LibraryConfig) -> None:
        entries = {
            "m": ModelEntry(
                id="m",
                name="M",
                source=RepositorySource.LOCAL,
                urdf_path=Path("/no/such/path.urdf"),
                is_cached=True,
            )
        }
        assert load_model(entries, URDFParser(), lib_config, "m") is None

    def test_parser_error_returns_none(
        self, tmp_path: Path, lib_config: LibraryConfig
    ) -> None:
        bad = tmp_path / "bad.urdf"
        bad.write_text("<not><valid")
        entries = {
            "m": ModelEntry(
                id="m",
                name="M",
                source=RepositorySource.LOCAL,
                urdf_path=bad,
                is_cached=True,
            )
        }
        assert load_model(entries, URDFParser(), lib_config, "m") is None

    def test_mjcf_path(self, tmp_path: Path, lib_config: LibraryConfig) -> None:
        mjcf = tmp_path / "m.xml"
        mjcf.write_text("<mujoco><worldbody/></mujoco>")
        entries = {
            "m": ModelEntry(
                id="m",
                name="M",
                source=RepositorySource.LOCAL,
                urdf_path=mjcf,
                model_format=ModelFormat.MJCF,
                is_cached=True,
            )
        }
        # Should not crash; result is converter-dependent but function must
        # tolerate either ParsedModel or None.
        result = load_model(entries, URDFParser(), lib_config, "m")
        assert result is None or hasattr(result, "links")

    def test_force_download_calls_download(
        self, simple_urdf: Path, lib_config: LibraryConfig
    ) -> None:
        entry = ModelEntry(
            id="m",
            name="M",
            source=RepositorySource.GITHUB,
            source_url="https://example.com/m.urdf",
            urdf_path=simple_urdf,
            is_cached=True,
        )
        entries = {"m": entry}
        with patch(
            "model_generation.library._model_loader.download_model",
            return_value=True,
        ) as mock_dl:
            load_model(entries, URDFParser(), lib_config, "m", force_download=True)
            mock_dl.assert_called_once()


class TestDownloadModel:
    def test_no_url_returns_false(self, lib_config: LibraryConfig) -> None:
        e = ModelEntry(id="m", name="M", source=RepositorySource.GITHUB)
        assert download_model(e, lib_config, {"m": e}) is False

    def test_success(
        self, tmp_path: Path, lib_config: LibraryConfig, monkeypatch
    ) -> None:
        # Set up a fake urlretrieve that writes a file
        def fake_urlretrieve(url, dest):
            Path(dest).write_text("downloaded")
            return dest, None

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
        e = ModelEntry(
            id="m",
            name="M",
            source=RepositorySource.URL,
            source_url="https://example.com/file.urdf",
        )
        entries = {"m": e}
        assert download_model(e, lib_config, entries) is True
        assert e.is_cached is True
        assert e.urdf_path is not None
        assert e.urdf_path.exists()

    def test_oserror_returns_false(
        self, lib_config: LibraryConfig, monkeypatch
    ) -> None:
        def boom(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlretrieve", boom)
        e = ModelEntry(
            id="m",
            name="M",
            source=RepositorySource.URL,
            source_url="https://example.com/file.urdf",
        )
        assert download_model(e, lib_config, {"m": e}) is False

    def test_id_with_slash_sanitized(
        self, lib_config: LibraryConfig, monkeypatch
    ) -> None:
        captured = {}

        def fake(url, dest):
            captured["dest"] = Path(dest)
            Path(dest).write_text("x")
            return dest, None

        monkeypatch.setattr("urllib.request.urlretrieve", fake)
        e = ModelEntry(
            id="org/name",
            name="N",
            source=RepositorySource.GITHUB,
            source_url="https://example.com/n.urdf",
        )
        assert download_model(e, lib_config, {"org/name": e}) is True
        # The cache directory part should not contain a literal slash from
        # the id (the sanitized form replaces '/' with '_')
        assert "org_name" in str(captured["dest"])
