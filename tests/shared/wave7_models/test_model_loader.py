"""Tests for model_generation.library._model_loader."""

from __future__ import annotations

from io import BytesIO
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
from model_generation.library.model_library import ModelLibrary


class _UrlopenResponse:
    def __init__(self, payload: bytes = b"downloaded") -> None:
        self._body = BytesIO(payload)

    def __enter__(self) -> BytesIO:
        return self._body

    def __exit__(self, *args: object) -> None:
        self._body.close()


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

    @pytest.mark.parametrize(
        "source_url",
        [
            "file:///tmp/secret.urdf",
            "http://example.com/model.urdf",
            "ftp://example.com/model.urdf",
        ],
    )
    def test_model_library_force_download_rejects_non_https_before_urlopen(
        self, lib_config: LibraryConfig, source_url: str
    ) -> None:
        library = ModelLibrary(lib_config)
        library._entries = {
            "m": ModelEntry(
                id="m",
                name="M",
                source=RepositorySource.URL,
                source_url=source_url,
                is_cached=False,
            )
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            result = library.load_model("m", force_download=True)

        assert result is None
        assert not (lib_config.cache_dir / "m").exists()
        mock_urlopen.assert_not_called()


class TestDownloadModel:
    def test_no_url_returns_false(self, lib_config: LibraryConfig) -> None:
        e = ModelEntry(id="m", name="M", source=RepositorySource.GITHUB)
        assert download_model(e, lib_config, {"m": e}) is False

    def test_success(
        self, tmp_path: Path, lib_config: LibraryConfig, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *args, **kwargs: _UrlopenResponse(),
        )
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

        monkeypatch.setattr("urllib.request.urlopen", boom)
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

        def fake_urlopen(request, **kwargs):
            captured["url"] = request.full_url
            return _UrlopenResponse(b"x")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        e = ModelEntry(
            id="org/name",
            name="N",
            source=RepositorySource.GITHUB,
            source_url="https://example.com/n.urdf",
        )
        assert download_model(e, lib_config, {"org/name": e}) is True
        # The cache directory part should not contain a literal slash from
        # the id (the sanitized form replaces '/' with '_')
        assert e.urdf_path is not None
        assert "org_name" in str(e.urdf_path)
        assert captured["url"] == "https://example.com/n.urdf"
