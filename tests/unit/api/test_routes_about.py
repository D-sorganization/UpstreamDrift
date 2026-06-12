"""Tests for the /about route and shared version resolution (issue #7459).

Covers:
    - The version resolution chain (VERSION file -> importlib.metadata ->
      fallback) shared between the desktop About dialog and the API route.
    - Git commit reading from .git (direct hash, symbolic ref, absence).
    - GET /about response shape and GET /about/onboarding card copy.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.about import router
from src.shared.python import version_info

pytestmark = pytest.mark.unit


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


class TestVersionResolutionChain:
    def test_version_file_wins(self, tmp_path: Path) -> None:
        (tmp_path / "VERSION").write_text("9.9.9\nextra\n", encoding="utf-8")
        assert version_info.read_version_file(tmp_path) == "9.9.9"
        assert version_info.resolve_app_version(tmp_path) == "9.9.9"

    def test_missing_version_file_returns_none(self, tmp_path: Path) -> None:
        assert version_info.read_version_file(tmp_path) is None

    def test_empty_version_file_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "VERSION").write_text("   \n", encoding="utf-8")
        assert version_info.read_version_file(tmp_path) is None

    def test_metadata_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda name: f"{name}-7.7.7")
        assert version_info.resolve_app_version(tmp_path) == "upstream-drift-7.7.7"

    def test_hardcoded_fallback_when_nothing_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _not_found(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", _not_found)
        assert (
            version_info.resolve_app_version(tmp_path) == version_info.FALLBACK_VERSION
        )

    def test_repo_version_file_is_picked_up(self) -> None:
        # The real repo has a VERSION file; the chain must return its content.
        expected = (version_info.get_repo_root() / "VERSION").read_text(
            encoding="utf-8"
        )
        assert version_info.resolve_app_version() == expected.strip().splitlines()[0]


class TestSafeVersionHelpers:
    def test_safe_module_version_missing(self) -> None:
        assert version_info.safe_module_version("definitely_not_a_module_xyz") == (
            "not installed"
        )

    def test_safe_module_version_numpy(self) -> None:
        v = version_info.safe_module_version("numpy")
        assert v not in ("not installed", "unknown")

    def test_installed_dist_version_missing(self) -> None:
        assert version_info.installed_dist_version("definitely-not-a-dist-xyz") == (
            "not installed"
        )


class TestReadGitCommit:
    def test_no_git_dir_returns_none(self, tmp_path: Path) -> None:
        assert version_info.read_git_commit(tmp_path) is None

    def test_detached_head_direct_hash(self, tmp_path: Path) -> None:
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")
        assert version_info.read_git_commit(tmp_path) == "a" * 40

    def test_symbolic_ref(self, tmp_path: Path) -> None:
        git = tmp_path / ".git"
        ref_dir = git / "refs" / "heads"
        ref_dir.mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (ref_dir / "main").write_text("b" * 40 + "\n", encoding="utf-8")
        assert version_info.read_git_commit(tmp_path) == "b" * 40

    def test_packed_refs(self, tmp_path: Path) -> None:
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted\n"
            f"{'c' * 40} refs/heads/main\n",
            encoding="utf-8",
        )
        assert version_info.read_git_commit(tmp_path) == "c" * 40

    def test_never_raises_on_garbage(self, tmp_path: Path) -> None:
        (tmp_path / ".git").write_text("not a gitdir pointer", encoding="utf-8")
        assert version_info.read_git_commit(tmp_path) is None


class TestAboutEndpoint:
    def test_about_shape(self, client: TestClient) -> None:
        resp = client.get("/api/v1/about")
        assert resp.status_code == 200
        body = resp.json()
        assert body["app_name"] == "UpstreamDrift"
        assert body["app_version"] == version_info.resolve_app_version()
        assert body["python_version"].count(".") == 2
        assert "numpy" in body["dependencies"]
        assert body["links"]["report_bug"] == version_info.ISSUES_URL
        assert body["links"]["repository"] == version_info.REPO_URL
        assert body["links"]["user_guide"] == version_info.USER_GUIDE_URL
        # git_commit is either None (no .git in deploy) or a hash string.
        assert body["git_commit"] is None or isinstance(body["git_commit"], str)

    def test_onboarding_copy(self, client: TestClient) -> None:
        resp = client.get("/api/v1/about/onboarding")
        assert resp.status_code == 200
        body = resp.json()
        assert body["header"] == "Welcome to UpstreamDrift"
        assert isinstance(body["cards"], list)
        assert len(body["cards"]) >= 2
        for card in body["cards"]:
            assert card["title"]
            assert card["body"]
            assert card["link_url"].startswith("https://")

    def test_about_routes_registered_on_local_app(self) -> None:
        from src.api.local_server import create_local_app

        local_client = TestClient(create_local_app())

        versioned = local_client.get("/api/v1/about")
        legacy = local_client.get("/api/about")

        assert versioned.status_code == 200
        assert legacy.status_code == 200


class TestSharedResolutionWithDesktopDialog:
    def test_about_dialog_delegates_to_shared_helper(self) -> None:
        """The desktop dialog must use the same implementation (one impl)."""
        import importlib.util

        spec = importlib.util.find_spec("PyQt6")
        if spec is None:
            pytest.skip("PyQt6 not installed")
        from src.launchers import about_dialog

        assert about_dialog._resolve_app_version is version_info.resolve_app_version
        assert about_dialog._safe_version is version_info.safe_module_version
        assert about_dialog.REPO_URL == version_info.REPO_URL
        assert about_dialog.ISSUES_URL == version_info.ISSUES_URL


class TestOnboardingCopySingleSourced:
    def test_qt_dialog_loads_shared_json(self) -> None:
        """Qt onboarding copy loader reads the shared JSON file."""
        import importlib.util

        spec = importlib.util.find_spec("PyQt6")
        if spec is None:
            pytest.skip("PyQt6 not installed")
        from src.launchers.onboarding_dialog import (
            ONBOARDING_CARDS_PATH,
            load_onboarding_copy,
        )

        assert ONBOARDING_CARDS_PATH.exists()
        copy = load_onboarding_copy()
        assert copy["header"] == "Welcome to UpstreamDrift"
        assert len(copy["cards"]) >= 2
