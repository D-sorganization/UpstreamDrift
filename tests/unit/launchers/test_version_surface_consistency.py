"""Every user-facing surface must report one build version (issue #8064).

The classic launcher's title bar read a hardcoded ``__version__`` from
``src.shared.python.core.version`` while Help > About resolved the repo-root
``VERSION`` file, so the same running process reported ``v1.0.0`` in the title
bar and ``2.1.1`` in the About dialog.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.launchers.about_dialog import build_about_html, gather_version_info
from src.shared.python.core.version import (
    __version__,
    __version_info__,
    _parse_version_info,
)
from src.shared.python.version_info import resolve_app_version

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestSingleVersionSource:
    """core.version must resolve, not hardcode."""

    def test_core_version_matches_resolver(self) -> None:
        """``core.version.__version__`` is whatever the resolver returns."""
        assert __version__ == resolve_app_version()

    def test_core_version_matches_version_file(self) -> None:
        """The resolver's first source is the repo-root VERSION file."""
        version_file = REPO_ROOT / "VERSION"
        assert version_file.exists(), "repo-root VERSION file is missing"
        assert __version__ == version_file.read_text(encoding="utf-8").strip()

    def test_version_info_matches_version_string(self) -> None:
        """``__version_info__`` is derived from the resolved string."""
        assert __version_info__ == _parse_version_info(__version__)

    def test_about_dialog_reports_the_same_version(self) -> None:
        """Help > About shows the resolved version, not an independent one."""
        assert gather_version_info()["app"] == __version__
        assert f"<b>Version:</b> {__version__}" in build_about_html()

    def test_no_hardcoded_version_literal_in_core_version(self) -> None:
        """A future edit cannot quietly reintroduce a pinned version."""
        source = (
            REPO_ROOT / "src" / "shared" / "python" / "core" / "version.py"
        ).read_text(encoding="utf-8")
        assert "__version__ = resolve_app_version()" in source
        assert '__version__ = "' not in source

    def test_title_bar_uses_the_shared_resolver(self) -> None:
        """The title bar module must not carry its own version literal."""
        source = (REPO_ROOT / "src" / "launchers" / "custom_title_bar.py").read_text(
            encoding="utf-8"
        )
        assert "resolve_app_version" in source
        assert 'version = "' not in source


class TestVersionInfoParsing:
    """``_parse_version_info`` must be total — a version never crashes the UI."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2.1.1", (2, 1, 1)),
            ("1.0.0", (1, 0, 0)),
            ("1.0.0-beta", (1, 0, 0)),
            ("2.1.1-rc1", (2, 1, 1)),
            ("3.4", (3, 4, 0)),
            ("7", (7, 0, 0)),
            ("", (0, 0, 0)),
            ("dev", (0, 0, 0)),
            ("1.2.3.4", (1, 2, 3)),
        ],
    )
    def test_parses_without_raising(
        self, raw: str, expected: tuple[int, int, int]
    ) -> None:
        assert _parse_version_info(raw) == expected

    def test_rejects_non_string(self) -> None:
        """DbC: the helper validates its precondition."""
        with pytest.raises(TypeError):
            _parse_version_info(None)  # type: ignore[arg-type]


class TestTitleBarAndAboutAgree:
    """The end-to-end assertion the QA campaign made by hand."""

    def test_title_bar_label_contains_the_about_version(
        self, qapp_or_skip: object
    ) -> None:
        """Title bar and About dialog agree inside one process."""
        from src.launchers.custom_title_bar import CustomTitleBar

        bar = CustomTitleBar()
        try:
            about_version = gather_version_info()["app"]
            assert f"v{about_version}" in bar.title_label.text()
        finally:
            bar.deleteLater()


@pytest.fixture
def qapp_or_skip() -> object:
    """Return a QApplication, skipping when PyQt6 cannot start offscreen."""
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
