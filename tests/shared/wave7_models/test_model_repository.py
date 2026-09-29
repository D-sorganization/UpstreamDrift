"""Tests for model_generation.library._model_repository."""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from model_generation.library._model_repository import (
    KNOWN_REPOSITORIES,
    _fetch_github_models,
    _fetch_repository_models,
    _fetch_url_models,
    add_repository,
    refresh_repository,
)
from model_generation.library._model_types import (
    LibraryConfig,
    ModelEntry,
    ModelFormat,
    RepositorySource,
)


class TestAddRepository:
    def test_minimal(self) -> None:
        repos: dict[str, Any] = {}
        add_repository(repos, "test")
        assert "test" in repos
        assert repos["test"]["type"] == "github"
        assert repos["test"]["branch"] == "main"

    def test_full(self) -> None:
        repos: dict[str, Any] = {}
        add_repository(
            repos,
            "r",
            repo_type="url",
            owner="o",
            repo="rr",
            branch="dev",
            path="sub",
            url="https://x",
        )
        assert repos["r"]["type"] == "url"
        assert repos["r"]["owner"] == "o"
        assert repos["r"]["url"] == "https://x"
        assert repos["r"]["branch"] == "dev"


class TestKnownRepositories:
    def test_contains_expected(self) -> None:
        for k in (
            "human_gazebo",
            "robot_descriptions",
            "pybullet_data",
            "mujoco_menagerie",
        ):
            assert k in KNOWN_REPOSITORIES


class TestFetchUrlModels:
    def test_url_urdf(self) -> None:
        models = _fetch_url_models("r", {"url": "https://x/a.urdf"})
        assert len(models) == 1
        assert models[0].source is RepositorySource.URL
        assert models[0].source_url == "https://x/a.urdf"

    def test_no_url(self) -> None:
        assert _fetch_url_models("r", {}) == []

    def test_non_urdf_url(self) -> None:
        assert _fetch_url_models("r", {"url": "https://x/a.zip"}) == []

    def test_none_repo_name_raises(self) -> None:
        with pytest.raises(ValueError):
            _fetch_url_models(None, {"url": "https://x.urdf"})  # type: ignore[arg-type]


def _mock_urlopen_response(payload):
    """Build a context-manager-style mock for urlopen returning JSON."""
    body = json.dumps(payload).encode()
    cm = MagicMock()
    cm.__enter__.return_value = io.BytesIO(body)
    cm.__exit__.return_value = False
    return cm


class TestFetchGithubModels:
    def test_no_owner(self) -> None:
        assert _fetch_github_models("r", {"repo": "x"}) == []

    def test_files_at_top_level(self) -> None:
        listing = [
            {
                "type": "file",
                "name": "a.urdf",
                "download_url": "https://x/a.urdf",
            },
            {
                "type": "file",
                "name": "b.xml",
                "download_url": "https://x/b.xml",
            },
            {
                "type": "file",
                "name": "ignore.txt",
                "download_url": "https://x/ignore.txt",
            },
        ]
        with patch(
            "urllib.request.urlopen", return_value=_mock_urlopen_response(listing)
        ):
            models = _fetch_github_models(
                "myrepo", {"owner": "o", "repo": "r", "branch": "main"}
            )
        names = {m.name for m in models}
        assert names == {"a", "b"}
        formats = {m.model_format for m in models}
        assert ModelFormat.URDF in formats
        assert ModelFormat.MJCF in formats
        for m in models:
            assert m.source is RepositorySource.GITHUB
            assert m.is_cached is False

    def test_subdir_traversal(self) -> None:
        top = [
            {
                "type": "dir",
                "name": "robot1",
                "url": "https://api.x/contents/robot1",
            },
        ]
        sub = [
            {
                "type": "file",
                "name": "model.urdf",
                "download_url": "https://x/model.urdf",
            },
        ]
        responses = iter([_mock_urlopen_response(top), _mock_urlopen_response(sub)])

        def side(*a, **k):
            return next(responses)

        with patch("urllib.request.urlopen", side_effect=side):
            models = _fetch_github_models("rn", {"owner": "o", "repo": "r"})
        assert len(models) == 1
        assert models[0].id == "rn/robot1"

    def test_network_error_returns_empty(self) -> None:
        with patch("urllib.request.urlopen", side_effect=OSError("boom")):
            assert _fetch_github_models("r", {"owner": "o", "repo": "rr"}) == []


class TestRefreshRepository:
    def test_unknown_repo_raises(self) -> None:
        cfg = LibraryConfig()
        with pytest.raises(ValueError, match="Unknown repository"):
            refresh_repository("nope", {}, {}, cfg)

    def test_uses_known_repository(self, tmp_path) -> None:
        cfg = LibraryConfig(cache_dir=tmp_path / "c", index_file=tmp_path / "i.json")
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen_response([]),
        ):
            models = refresh_repository("human_gazebo", {}, {}, cfg)
        assert models == []

    def test_uses_user_repository(self, tmp_path) -> None:
        cfg = LibraryConfig(cache_dir=tmp_path / "c", index_file=tmp_path / "i.json")
        repos = {
            "mine": {
                "type": "url",
                "url": "https://example.com/x.urdf",
            }
        }
        entries: dict[str, ModelEntry] = {}
        models = refresh_repository("mine", repos, entries, cfg)
        assert len(models) == 1
        assert "mine/model" in entries
        assert cfg.index_file.exists()


class TestFetchRepositoryDispatch:
    def test_dispatches_url(self) -> None:
        models = _fetch_repository_models("r", {"type": "url", "url": "https://x.urdf"})
        assert len(models) == 1

    def test_unknown_type_empty(self) -> None:
        assert _fetch_repository_models("r", {"type": "weird"}) == []
