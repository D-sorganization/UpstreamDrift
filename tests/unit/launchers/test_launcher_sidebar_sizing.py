"""Tests for issue #5505 — proportional sidebar sizing.

Verifies:
- Styles.SIDEBAR_MIN_WIDTH, Styles.SPACING_LG, Styles.MARGIN_PAGE constants exist.
- Launcher uses Styles.SIDEBAR_MIN_WIDTH for sidebar.setMinimumWidth().
- Launcher replaces hard-coded setSizes([85, 1200]) with 1:5 stretch factors.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch


# ---------------------------------------------------------------------------
# Helpers — work whether or not PyQt6 is importable
# ---------------------------------------------------------------------------


def _styles():
    """Import Styles (safe; module has no top-level Qt calls)."""
    from src.shared.python.theme.style_constants import Styles

    return Styles


def _make_launcher():
    """Return a launcher instance with stubs so no Qt display is needed.

    We patch QSplitter at the launcher module level so the real
    setStretchFactor / setSizes are replaced with tracking mocks before
    _setup_main_layout runs.
    """
    stretch_calls: list[tuple[int, int]] = []
    sizes_calls: list[list[int]] = []

    class TrackingSplitter:
        """Minimal QSplitter stand-in that records calls."""

        def __init__(self, *args, **kwargs):
            pass

        def addWidget(self, w):
            pass

        def setHandleWidth(self, v):
            pass

        def setProperty(self, k, v):
            pass

        def style(self):
            return None

        def setStretchFactor(self, idx: int, factor: int):
            stretch_calls.append((idx, factor))

        def setSizes(self, sizes):
            sizes_calls.append(list(sizes))

    # ---------- build a minimal launcher stub ----------

    class _FakeSizePolicy:
        Policy = MagicMock()

    qt_stub = MagicMock()
    qt_stub.QSplitter = TrackingSplitter
    qt_stub.QSizePolicy = _FakeSizePolicy

    class DummyLauncher:
        """Stub that exercises _setup_main_layout in isolation."""

        def __init__(self):
            self._setup_main_layout()

        # -- minimal plumbing expected by _setup_main_layout --
        def setLayout(self, layout_):
            pass

        def apply_styles(self):
            pass

        def _setup_search_shortcuts(self):
            pass

        def _init_overlay(self):
            pass

        def _setup_top_bar(self):
            return MagicMock()

        def _setup_bottom_bar(self):
            return MagicMock()

        def _setup_grid_area(self, layout):
            pass

        def _setup_ai_panel(self):
            pass

        def _setup_global_sidebar(self):
            w = MagicMock()
            w.minimumWidth.return_value = 0
            w.setMinimumWidth = MagicMock()
            w.setSizePolicy = MagicMock()
            return w

    # Inject at module level so that QSplitter references in
    # launcher_ui_setup resolve to TrackingSplitter.
    with patch.dict(
        "sys.modules",
        {
            "PyQt6": MagicMock(),
            "PyQt6.QtWidgets": qt_stub,
            "PyQt6.QtCore": MagicMock(),
            "PyQt6.QtGui": MagicMock(),
        },
    ):
        import importlib

        from src.shared.python.theme.style_constants import Styles  # noqa: F401
        import src.launchers.launcher_ui_setup as lus

        importlib.reload(lus)

        class PatchedLauncher(DummyLauncher, lus.LauncherUISetupMixin):
            pass

        launcher = PatchedLauncher.__new__(PatchedLauncher)
        launcher._setup_main_layout = lambda: None  # skip real layout

    return launcher, stretch_calls, sizes_calls


# ---------------------------------------------------------------------------
# Tests — Styles constants
# ---------------------------------------------------------------------------


def test_sidebar_minimum_width_constant_exists():
    """Styles.SIDEBAR_MIN_WIDTH must exist and be >= 120 px (issue #5505)."""
    Styles = _styles()
    assert hasattr(Styles, "SIDEBAR_MIN_WIDTH"), "Styles.SIDEBAR_MIN_WIDTH not found"
    assert isinstance(Styles.SIDEBAR_MIN_WIDTH, int)
    assert Styles.SIDEBAR_MIN_WIDTH >= 120, (
        f"Expected SIDEBAR_MIN_WIDTH >= 120, got {Styles.SIDEBAR_MIN_WIDTH}"
    )


def test_spacing_constant_exists():
    """Styles.SPACING_LG must equal 20 (issue #5505)."""
    Styles = _styles()
    assert hasattr(Styles, "SPACING_LG"), "Styles.SPACING_LG not found"
    assert Styles.SPACING_LG == 20, (
        f"Expected SPACING_LG == 20, got {Styles.SPACING_LG}"
    )


def test_margin_page_constant_exists():
    """Styles.MARGIN_PAGE must equal 30 (issue #5505)."""
    Styles = _styles()
    assert hasattr(Styles, "MARGIN_PAGE"), "Styles.MARGIN_PAGE not found"
    assert Styles.MARGIN_PAGE == 30, (
        f"Expected MARGIN_PAGE == 30, got {Styles.MARGIN_PAGE}"
    )


# ---------------------------------------------------------------------------
# Tests — launcher_ui_setup source text
# ---------------------------------------------------------------------------


def test_sidebar_widget_minimum_width_uses_constant():
    """_setup_global_sidebar must call setMinimumWidth(Styles.SIDEBAR_MIN_WIDTH)."""
    from src.shared.python.theme.style_constants import Styles

    source_path = "src/launchers/launcher_ui_setup.py"
    with open(source_path, encoding="utf-8") as fh:
        source = fh.read()

    assert "Styles.SIDEBAR_MIN_WIDTH" in source, (
        "launcher_ui_setup.py must use Styles.SIDEBAR_MIN_WIDTH "
        "instead of a hard-coded pixel value"
    )
    # The hard-coded 85 should no longer appear as the argument to setMinimumWidth.
    import re

    assert not re.search(r"setMinimumWidth\(85\)", source), (
        "Hard-coded setMinimumWidth(85) still present; replace with Styles.SIDEBAR_MIN_WIDTH"
    )


def test_main_splitter_uses_stretch_factors_not_fixed_sizes():
    """_setup_main_layout must use 1:5 stretch factors and drop setSizes([85,…])."""
    source_path = "src/launchers/launcher_ui_setup.py"
    with open(source_path, encoding="utf-8") as fh:
        source = fh.read()

    import re

    # Must NOT contain the old hard-coded sizes call.
    assert not re.search(r"setSizes\(\s*\[85", source), (
        "Hard-coded setSizes([85, …]) still present; replace with setStretchFactor"
    )

    # Must contain setStretchFactor calls with 1 and 5.
    stretch_matches = re.findall(r"setStretchFactor\((\d+),\s*(\d+)\)", source)
    factors = {int(f) for _, f in stretch_matches}
    assert 1 in factors and 5 in factors, (
        f"Expected setStretchFactor calls with factors 1 and 5; found factors {factors}"
    )
