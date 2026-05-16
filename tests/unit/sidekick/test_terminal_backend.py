"""Tests for OS terminal backend and widget.

Verifies the PTY-backed terminal backend, the Python REPL rename,
and default tab registration.

Issue #5617: real OS terminal tab with PTY backend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_os_terminal():
    """Import os_terminal module classes."""
    from src.shared.python.upstream_drift_tools.ui.tools_sidebar.os_terminal import (
        SidekickOsTerminalWidget,
        create_terminal_backend,
    )

    return SidekickOsTerminalWidget, create_terminal_backend


def _import_runtime_tabs():
    """Import runtime_tabs module via importlib to bypass package __init__.py.

    runtime_tabs.py uses relative imports (``from . import design_tokens``),
    which only resolve when the package is loaded from the vendor path.  We
    load it directly so we can inspect names without constructing Qt widgets.
    """
    import importlib.util
    import pathlib

    runtime_tabs_path = (
        pathlib.Path(__file__).parents[3]
        / "src"
        / "shared"
        / "python"
        / "upstream_drift_tools"
        / "ui"
        / "tools_sidebar"
        / "runtime_tabs.py"
    )
    # We only need to check module-level names; patch heavy deps so the
    # file can be loaded without a running Qt application.
    from unittest.mock import MagicMock

    fake_theme = MagicMock()
    fake_registry = MagicMock()
    fake_qt_compat = MagicMock()
    fake_help = MagicMock()
    fake_calculator_assist = MagicMock()
    fake_calculator_runtime = MagicMock()
    fake_calculator_startup = MagicMock()

    stub_modules = {
        # relative-import targets resolved as if loaded from the vendor package
        "upstream_drift_tools.ui.tools_sidebar.design_tokens": fake_theme,
        "upstream_drift_tools.ui.tools_sidebar.calculator_assist": fake_calculator_assist,
        "upstream_drift_tools.ui.tools_sidebar.calculator_runtime": fake_calculator_runtime,
        "upstream_drift_tools.ui.tools_sidebar.calculator_startup": fake_calculator_startup,
        "upstream_drift_tools.ui.tools_sidebar.help_content": fake_help,
        "upstream_drift_tools.ui.tools_sidebar.qt_compat": fake_qt_compat,
        "upstream_drift_tools.ui.tools_sidebar.registry": fake_registry,
    }

    with patch.dict("sys.modules", stub_modules):
        spec = importlib.util.spec_from_file_location(
            "upstream_drift_tools.ui.tools_sidebar.runtime_tabs",
            runtime_tabs_path,
        )
        mod = importlib.util.module_from_spec(spec)
        # Set the package so relative imports resolve to our stubs
        mod.__package__ = "upstream_drift_tools.ui.tools_sidebar"
        spec.loader.exec_module(mod)

    return mod


def _import_sidebar():
    """Import sidebar module."""
    from src.shared.python.upstream_drift_tools.ui.tools_sidebar.sidebar import (
        UnifiedToolsSidebar,
    )

    return UnifiedToolsSidebar


# ---------------------------------------------------------------------------
# Backend tests
# ---------------------------------------------------------------------------


def _safe_shell() -> str:
    """Return a shell path that exists on the current platform."""
    import shutil
    import sys

    if sys.platform == "win32":
        for candidate in ("pwsh", "powershell", "cmd"):
            found = shutil.which(candidate)
            if found:
                return found
        return "cmd"  # always present on Windows
    for candidate in ("bash", "sh"):
        found = shutil.which(candidate)
        if found:
            return found
    return "/bin/sh"


@pytest.mark.unit
def test_fallback_backend_used_when_pty_unavailable() -> None:
    """When pywinpty/ptyprocess are not installed, a fallback backend is used."""
    import subprocess
    import sys
    from unittest.mock import MagicMock

    _, create_terminal_backend = _import_os_terminal()

    # Patch subprocess.Popen so the fallback doesn't actually spawn a process
    dummy_proc = MagicMock()
    dummy_proc.poll.return_value = None
    dummy_proc.stdin = MagicMock()
    dummy_proc.stdout = iter([])  # empty iterable so reader thread exits

    with (
        patch.dict("sys.modules", {"winpty": None, "ptyprocess": None}),
        patch(
            "src.shared.python.upstream_drift_tools.ui.tools_sidebar.os_terminal"
            ".subprocess.Popen",
            return_value=dummy_proc,
        ),
    ):
        if sys.platform == "win32":
            # On Windows, ptyprocess path is not attempted; only winpty
            backend = create_terminal_backend(_safe_shell())
        else:
            # On POSIX, both winpty (KeyError) and ptyprocess fail
            backend = create_terminal_backend(_safe_shell())

    assert backend is not None, "Expected a fallback backend, not None"


@pytest.mark.unit
def test_create_terminal_backend_returns_object_with_write_method() -> None:
    """Terminal backends expose a write() method."""
    import subprocess
    from unittest.mock import MagicMock

    _, create_terminal_backend = _import_os_terminal()

    dummy_proc = MagicMock()
    dummy_proc.poll.return_value = None
    dummy_proc.stdin = MagicMock()
    dummy_proc.stdout = iter([])

    with (
        patch.dict("sys.modules", {"winpty": None, "ptyprocess": None}),
        patch(
            "src.shared.python.upstream_drift_tools.ui.tools_sidebar.os_terminal"
            ".subprocess.Popen",
            return_value=dummy_proc,
        ),
    ):
        backend = create_terminal_backend(_safe_shell())

    assert hasattr(backend, "write"), "Backend must expose write()"


@pytest.mark.unit
def test_create_terminal_backend_returns_object_with_close_method() -> None:
    """Terminal backends expose a close() method."""
    import subprocess
    from unittest.mock import MagicMock

    _, create_terminal_backend = _import_os_terminal()

    dummy_proc = MagicMock()
    dummy_proc.poll.return_value = None
    dummy_proc.stdin = MagicMock()
    dummy_proc.stdout = iter([])

    with (
        patch.dict("sys.modules", {"winpty": None, "ptyprocess": None}),
        patch(
            "src.shared.python.upstream_drift_tools.ui.tools_sidebar.os_terminal"
            ".subprocess.Popen",
            return_value=dummy_proc,
        ),
    ):
        backend = create_terminal_backend(_safe_shell())

    assert hasattr(backend, "close"), "Backend must expose close()"


# ---------------------------------------------------------------------------
# Widget existence tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_os_terminal_widget_exists() -> None:
    """SidekickOsTerminalWidget must be importable from os_terminal module."""
    SidekickOsTerminalWidget, _ = _import_os_terminal()

    assert SidekickOsTerminalWidget is not None


@pytest.mark.unit
def test_python_repl_widget_still_exists_under_new_name() -> None:
    """SidekickTerminalWidget is renamed to SidekickPythonReplWidget."""
    runtime_tabs = _import_runtime_tabs()

    assert hasattr(runtime_tabs, "SidekickPythonReplWidget"), (
        "SidekickPythonReplWidget must exist in runtime_tabs"
    )
    assert runtime_tabs.SidekickPythonReplWidget is not None


@pytest.mark.unit
def test_old_sidekick_terminal_widget_name_preserved_as_alias() -> None:
    """SidekickTerminalWidget alias is preserved for backward compatibility."""
    runtime_tabs = _import_runtime_tabs()

    assert hasattr(runtime_tabs, "SidekickTerminalWidget"), (
        "SidekickTerminalWidget backward-compat alias must remain"
    )
    assert runtime_tabs.SidekickTerminalWidget is runtime_tabs.SidekickPythonReplWidget


# ---------------------------------------------------------------------------
# Default tab registration tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_tabs_registers_terminal_and_python_repl() -> None:
    """Sidebar default tabs include both 'terminal' (OS) and 'python-repl'."""
    UnifiedToolsSidebar = _import_sidebar()

    # Instantiate with mocked Qt
    sidebar = UnifiedToolsSidebar.__new__(UnifiedToolsSidebar)
    sidebar._tab_definitions = []
    sidebar._active_widgets = {}

    definitions = sidebar._default_tab_definitions()
    tab_ids = [t.tab_id for t in definitions]

    assert "terminal" in tab_ids, f"'terminal' tab missing from {tab_ids}"
    has_python_repl = "python-repl" in tab_ids or "python_repl" in tab_ids
    assert has_python_repl, f"'python-repl' tab missing from {tab_ids}"


@pytest.mark.unit
def test_terminal_tab_listed_before_python_repl() -> None:
    """The OS 'terminal' tab appears before 'python-repl' in default order."""
    UnifiedToolsSidebar = _import_sidebar()

    sidebar = UnifiedToolsSidebar.__new__(UnifiedToolsSidebar)
    sidebar._tab_definitions = []
    sidebar._active_widgets = {}

    definitions = sidebar._default_tab_definitions()
    tab_ids = [t.tab_id for t in definitions]

    terminal_idx = tab_ids.index("terminal")
    python_repl_idx = next(
        (i for i, tid in enumerate(tab_ids) if tid in ("python-repl", "python_repl")),
        None,
    )

    assert python_repl_idx is not None
    assert terminal_idx < python_repl_idx, (
        "'terminal' must appear before 'python-repl' in default tab order"
    )
