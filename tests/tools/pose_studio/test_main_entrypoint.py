"""Tests for ``src.tools.pose_studio.__main__``.

Covers the two branches of :func:`main`:

* the happy path delegates to :func:`gui.main` and returns its result;
* an :class:`ImportError` from the GUI dependencies prints help text and
  returns ``1``.

Also covers :func:`get_dockable_ui` which simply re-exports the GUI
helper for the unified launcher.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.tools.pose_studio import __main__ as main_mod

pytestmark = pytest.mark.unit


def test_main_returns_gui_result() -> None:
    fake_gui = MagicMock()
    fake_gui.main.return_value = 42
    with patch.dict(sys.modules, {"src.tools.pose_studio.gui": fake_gui}):
        assert main_mod.main() == 42
    fake_gui.main.assert_called_once_with()


def test_main_handles_import_error(capsys: pytest.CaptureFixture[str]) -> None:
    # Force the import inside main() to fail.
    with patch.dict(sys.modules, {"src.tools.pose_studio.gui": None}):
        rc = main_mod.main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "Pose Studio" in err
    assert "gui-tools" in err


def test_get_dockable_ui_delegates() -> None:
    fake_gui = MagicMock()
    sentinel = object()
    fake_gui.get_dockable_ui.return_value = sentinel
    with patch.dict(sys.modules, {"src.tools.pose_studio.gui": fake_gui}):
        assert main_mod.get_dockable_ui() is sentinel
    fake_gui.get_dockable_ui.assert_called_once_with()
