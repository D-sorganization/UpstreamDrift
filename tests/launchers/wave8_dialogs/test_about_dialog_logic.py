"""Tests for non-GUI logic in src.launchers.about_dialog."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.launchers import about_dialog as ad


class TestReadVersionFile:
    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        with patch.object(
            ad, "__file__", str(tmp_path / "nonexistent" / "about_dialog.py")
        ):
            assert ad._read_version_file() is None

    def test_reads_first_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Layout: tmp/repo/src/launchers/about_dialog.py -> VERSION at parents[2]
        repo = tmp_path / "repo"
        (repo / "src" / "launchers").mkdir(parents=True)
        fake_file = repo / "src" / "launchers" / "about_dialog.py"
        fake_file.write_text("# stub")
        (repo / "VERSION").write_text("9.9.9\nignored\n")

        monkeypatch.setattr(ad, "__file__", str(fake_file))
        assert ad._read_version_file() == "9.9.9"

    def test_skips_empty_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        (repo / "src" / "launchers").mkdir(parents=True)
        fake_file = repo / "src" / "launchers" / "about_dialog.py"
        fake_file.write_text("# stub")
        (repo / "VERSION").write_text("   \n")

        monkeypatch.setattr(ad, "__file__", str(fake_file))
        assert ad._read_version_file() is None

    def test_oserror_handled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Patch Path.exists to raise OSError on read_text path
        repo = tmp_path / "repo"
        (repo / "src" / "launchers").mkdir(parents=True)
        fake_file = repo / "src" / "launchers" / "about_dialog.py"
        fake_file.write_text("# stub")
        monkeypatch.setattr(ad, "__file__", str(fake_file))

        original_read = Path.read_text

        def boom(self: Path, *a: object, **kw: object) -> str:
            raise OSError("denied")

        monkeypatch.setattr(Path, "read_text", boom)
        # Should not raise; just return None
        (repo / "VERSION").write_text("1.2.3")
        assert ad._read_version_file() is None
        monkeypatch.setattr(Path, "read_text", original_read)


class TestResolveAppVersion:
    def test_uses_version_file_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ad, "_read_version_file", lambda: "7.0.0")
        assert ad._resolve_app_version() == "7.0.0"

    def test_fallback_constant_when_no_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ad, "_read_version_file", lambda: None)
        from importlib.metadata import PackageNotFoundError

        def fake_version(name: str) -> str:
            raise PackageNotFoundError(name)

        import importlib.metadata as m

        monkeypatch.setattr(m, "version", fake_version)
        assert ad._resolve_app_version() == "1.0.0-beta"

    def test_uses_importlib_metadata_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ad, "_read_version_file", lambda: None)
        import importlib.metadata as m

        def fake_version(name: str) -> str:
            if name == "upstream-drift":
                return "3.2.1"
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError(name)

        monkeypatch.setattr(m, "version", fake_version)
        assert ad._resolve_app_version() == "3.2.1"


class TestSafeVersion:
    def test_returns_version_string(self) -> None:
        result = ad._safe_version("sys")
        # sys has no __version__; should return "unknown"
        assert result == "unknown"

    def test_known_module_with_version(self) -> None:
        # platform doesn't expose __version__ either; test with a stub.
        fake = type("M", (), {"__version__": "4.5.6"})()
        with patch.dict(sys.modules, {"_fakepkg_xyz": fake}):
            assert ad._safe_version("_fakepkg_xyz") == "4.5.6"

    def test_missing_module_returns_not_installed(self) -> None:
        assert ad._safe_version("nope_definitely_missing_pkg_x9z") == "not installed"


class TestGatherVersionInfo:
    def test_returns_required_keys(self) -> None:
        info = ad.gather_version_info()
        for key in ("app", "python", "qt", "numpy", "ezc3d", "platform"):
            assert key in info
            assert isinstance(info[key], str)

    def test_python_version_format(self) -> None:
        info = ad.gather_version_info()
        # Should look like a version, contain digits and dots
        assert any(c.isdigit() for c in info["python"])


class TestBuildAboutHtml:
    def test_uses_provided_info(self) -> None:
        info = {
            "app": "9.9.9",
            "python": "3.11.0",
            "qt": "6.5.0",
            "numpy": "1.26",
            "ezc3d": "not installed",
            "platform": "Linux 6.0",
        }
        html = ad.build_about_html(info)
        assert "9.9.9" in html
        assert "3.11.0" in html
        assert "ezc3d not installed" in html or "not installed" in html
        assert "UpstreamDrift" in html

    def test_gathers_when_none(self) -> None:
        html = ad.build_about_html(None)
        assert "<h2>UpstreamDrift</h2>" in html

    def test_includes_urls(self) -> None:
        html = ad.build_about_html(
            {
                "app": "x",
                "python": "x",
                "qt": "x",
                "numpy": "x",
                "ezc3d": "x",
                "platform": "x",
            }
        )
        assert ad.REPO_URL in html
        assert ad.ISSUES_URL in html


class TestUrlHelpers:
    def test_issues_url_derives_from_repo(self) -> None:
        assert f"{ad.REPO_URL}/issues" == ad.ISSUES_URL

    def test_open_issues_page_calls_qdesktop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[object] = []
        monkeypatch.setattr(
            ad.QDesktopServices, "openUrl", lambda url: calls.append(url) or True
        )
        ad.open_issues_page()
        assert len(calls) == 1
        assert ad.ISSUES_URL in calls[0].toString()


class TestOpenUserGuide:
    def test_falls_back_to_repo_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Point __file__ to empty tmp area so no docs exist.
        fake_file = tmp_path / "src" / "launchers" / "about_dialog.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.write_text("# stub")
        monkeypatch.setattr(ad, "__file__", str(fake_file))

        calls: list[object] = []
        monkeypatch.setattr(
            ad.QDesktopServices, "openUrl", lambda url: calls.append(url) or True
        )
        ad.open_user_guide()
        assert any(ad.REPO_URL in c.toString() for c in calls)

    def test_opens_local_doc_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        launchers_dir = repo / "src" / "launchers"
        launchers_dir.mkdir(parents=True)
        fake_file = launchers_dir / "about_dialog.py"
        fake_file.write_text("# stub")
        (repo / "docs").mkdir()
        manual = repo / "docs" / "USER_MANUAL.md"
        manual.write_text("# manual")

        monkeypatch.setattr(ad, "__file__", str(fake_file))

        called: list[Path] = []

        # Inject a fake document_reader.
        import types

        fake_mod = types.ModuleType("src.shared.python.ui.qt.widgets.document_reader")
        fake_mod.show_document = lambda p: called.append(p)  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules,
            "src.shared.python.ui.qt.widgets.document_reader",
            fake_mod,
        )

        ad.open_user_guide()
        assert called and called[0] == manual


class TestOpenMotionMatchLoadersDoc:
    def test_falls_back_to_user_guide(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_file = tmp_path / "src" / "launchers" / "about_dialog.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.write_text("# stub")
        monkeypatch.setattr(ad, "__file__", str(fake_file))

        called: list[str] = []
        monkeypatch.setattr(ad, "open_user_guide", lambda: called.append("ug"))
        ad.open_motion_match_loaders_doc()
        assert called == ["ug"]
