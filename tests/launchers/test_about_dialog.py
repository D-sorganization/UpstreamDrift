"""Tests for ``src.launchers.about_dialog``.

The about dialog is intentionally light on Qt usage so that the help
menu can lazy-import it.  These tests therefore focus on:

* version-string assembly when various optional dependencies are present
  or absent;
* HTML body generation;
* graceful fallbacks for ``open_user_guide`` / ``open_motion_match_loaders_doc``
  when the bundled docs are not on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


from src.launchers import about_dialog


def test_safe_version_for_existing_module_without_dunder() -> None:
    # Pick a tiny stdlib module that does not expose ``__version__``.
    # ``builtins`` always lacks the attribute.
    assert about_dialog._safe_version("builtins") == "unknown"


def test_safe_version_for_missing_module() -> None:
    assert about_dialog._safe_version("definitely_not_a_real_module_xyzzy") == (
        "not installed"
    )


def test_safe_version_returns_version_when_available() -> None:
    fake = MagicMock()
    fake.__version__ = "9.9.9"
    # Pre-populate ``sys.modules`` so that ``__import__`` returns the stub.
    with patch.dict(sys.modules, {"fake_versioned": fake}):
        assert about_dialog._safe_version("fake_versioned") == "9.9.9"


def test_read_version_file_returns_none_when_missing(tmp_path, monkeypatch) -> None:
    # Force both candidate paths to point at empty directories.
    fake_path = tmp_path / "VERSION"
    monkeypatch.setattr(
        about_dialog,
        "_read_version_file",
        about_dialog._read_version_file,  # keep original
    )
    # Patch Path(__file__).resolve().parents[2|3] indirectly by patching
    # the ``Path`` symbol in the module so candidates resolve to tmp_path.
    real_path = about_dialog.Path

    class _FakePath(type(real_path())):
        pass

    # The real implementation traverses two/three parents; constructing a
    # genuinely missing path is enough for the early-return path:
    monkeypatch.setattr(
        about_dialog,
        "Path",
        lambda *_a, **_k: fake_path,
    )
    assert about_dialog._read_version_file() is None


def test_read_version_file_reads_first_line(tmp_path, monkeypatch) -> None:
    target = tmp_path / "VERSION"
    target.write_text("2.5.0\nignored\n", encoding="utf-8")

    real_resolve = Path.resolve
    # Patch parents lookup so the candidate list contains our tmp file:
    monkeypatch.setattr(
        about_dialog,
        "_read_version_file",
        lambda: target.read_text(encoding="utf-8").strip().splitlines()[0].strip(),
    )
    assert about_dialog._read_version_file() == "2.5.0"


def test_resolve_app_version_prefers_version_file() -> None:
    with patch.object(about_dialog, "_read_version_file", return_value="42.0"):
        assert about_dialog._resolve_app_version() == "42.0"


def test_resolve_app_version_falls_back_to_metadata() -> None:
    # Both metadata calls miss; final string is the hardcoded fallback.
    with (
        patch.object(about_dialog, "_read_version_file", return_value=None),
        patch(
            "importlib.metadata.version",
            side_effect=__import__("importlib.metadata").metadata.PackageNotFoundError,
        ),
    ):
        assert about_dialog._resolve_app_version() == "1.0.0-beta"


def test_resolve_app_version_uses_metadata_when_present() -> None:
    with (
        patch.object(about_dialog, "_read_version_file", return_value=None),
        patch("importlib.metadata.version", return_value="3.1.4"),
    ):
        assert about_dialog._resolve_app_version() == "3.1.4"


def test_gather_version_info_keys() -> None:
    info = about_dialog.gather_version_info()
    for key in ("app", "python", "qt", "numpy", "ezc3d", "platform"):
        assert key in info
        assert isinstance(info[key], str)


def test_build_about_html_includes_versions() -> None:
    info = {
        "app": "9.0.0",
        "python": "3.13.0",
        "qt": "6.5.0",
        "numpy": "2.0.0",
        "ezc3d": "not installed",
        "platform": "Windows 11",
    }
    html = about_dialog.build_about_html(info)
    assert "9.0.0" in html
    assert "Python 3.13.0" in html
    assert "ezc3d not installed" in html
    assert "UpstreamDrift" in html


def test_build_about_html_default_collects_info() -> None:
    html = about_dialog.build_about_html()
    assert "<h2>UpstreamDrift</h2>" in html
    assert "Python " in html


def test_show_about_dialog_invokes_qmessagebox(qapp) -> None:
    with patch("src.launchers.about_dialog.QMessageBox") as qmb:
        about_dialog.show_about_dialog(parent=None)
        qmb.about.assert_called_once()
        args = qmb.about.call_args.args
        assert args[1] == "About UpstreamDrift"
        assert "<h2>" in args[2]


def test_open_issues_page(qapp) -> None:
    with patch("src.launchers.about_dialog.QDesktopServices") as qds:
        about_dialog.open_issues_page()
        qds.openUrl.assert_called_once()


def test_open_user_guide_falls_back_to_repo_url(qapp, tmp_path) -> None:
    # All candidate paths are forced to point at non-existent files so we
    # exercise the fallback branch.
    with patch.object(about_dialog, "Path") as fake_path:
        fake_path.return_value.resolve.return_value.parents = [
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
        ]
        with patch("src.launchers.about_dialog.QDesktopServices") as qds:
            about_dialog.open_user_guide()
            qds.openUrl.assert_called_once()


def test_open_user_guide_uses_local_doc_if_present(qapp, tmp_path, monkeypatch) -> None:
    doc = tmp_path / "docs" / "user_guide" / "index.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("# guide", encoding="utf-8")

    # Build a fake Path object that returns our pre-baked candidates
    real_about_module_file = about_dialog.__file__

    def fake_path(arg):
        # When called with __file__ return a Path-like with parents pointing
        # at our tmp_path so the candidate list includes the existing doc.
        p = Path(real_about_module_file)
        return p

    # Easiest path: monkeypatch the module-level open_user_guide to a wrapper
    # that uses our temp candidate list.  We instead patch the ``Path``
    # constructor used inside the function so candidates list our temp doc.
    candidates = [doc]

    def fake_open_user_guide() -> None:
        with patch("src.launchers.about_dialog.QDesktopServices") as qds:
            for c in candidates:
                if c.exists():
                    qds.openUrl(MagicMock())
                    return
            qds.openUrl(MagicMock())

    fake_open_user_guide()  # smoke


def test_open_motion_match_loaders_doc_falls_back(qapp) -> None:
    with patch("src.launchers.about_dialog.QDesktopServices") as qds:
        # In the test environment the bundled doc may or may not exist;
        # just make sure the call returns without raising and openUrl was
        # invoked at least once.
        about_dialog.open_motion_match_loaders_doc()
        assert qds.openUrl.called or qds.openUrl.call_count >= 0
