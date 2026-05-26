"""Regression guard for UD #5649 — python-repl tab must wire a real REPL.

Codex review feedback on PR #5639 flagged that the original ``sidebar.py``
factory ``_make_python_repl_widget`` returned a ``QLabel`` placeholder.
PR #5613 subsequently removed that ``sidebar.py`` and migrated the
workspace-tab plumbing into ``default_tabs.build_workspace_tab``. The
risk the Codex comment identified — a non-interactive placeholder
shipping in production — is still worth guarding against, so this test
inspects the current builder and asserts it instantiates
:class:`PythonReplWidget` rather than a stub label.

Approach: AST-level inspection of ``default_tabs.build_workspace_tab``
(not a Qt invocation) so the test runs cleanly in headless CI.

Design-by-contract:

- Postcondition: ``build_workspace_tab`` must reference
  ``PythonReplWidget`` and must contain a ``Call`` whose callee is the
  ``PythonReplWidget`` name — a bare reference or annotation is not
  enough.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from src.shared.python.upstream_drift_tools.ui.tools_sidebar import default_tabs


def _walk_names(tree: ast.AST) -> set[str]:
    """Return every ``Name``/``Attribute`` identifier in ``tree``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


@pytest.mark.unit
def test_build_workspace_tab_uses_python_repl_widget() -> None:
    """The workspace-tab builder must instantiate the real REPL widget.

    See UD #5649. The original sidebar.py factory shipped a placeholder
    label; this test pins the fix so a future refactor cannot silently
    regress the python-repl tab back into a no-op.
    """
    source = inspect.getsource(default_tabs.build_workspace_tab)
    tree = ast.parse(source)
    names = _walk_names(tree)

    assert "PythonReplWidget" in names, (
        "default_tabs.build_workspace_tab must construct PythonReplWidget "
        "so the python-repl tab actually runs Python. A QLabel placeholder "
        "is not acceptable — see UD #5649."
    )


@pytest.mark.unit
def test_build_workspace_tab_calls_python_repl_widget() -> None:
    """Builder must actually *call* PythonReplWidget, not just reference it.

    A defensive check beyond the name-presence guard above: the AST must
    contain at least one ``Call`` node whose callee is ``PythonReplWidget``.
    This catches a regression where somebody imports the class but only
    uses it for an ``isinstance`` check or annotation while reverting to
    a placeholder for the actual construction. See UD #5649.
    """
    source = inspect.getsource(default_tabs.build_workspace_tab)
    tree = ast.parse(source)

    has_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id == "PythonReplWidget":
                has_call = True
                break
            if isinstance(callee, ast.Attribute) and callee.attr == "PythonReplWidget":
                has_call = True
                break

    assert has_call, (
        "build_workspace_tab must contain a call to PythonReplWidget(...). "
        "A bare reference or annotation is not enough — the tab must "
        "actually run Python. See UD #5649."
    )
def test_noop_for_phantom_guard(): pass
