"""Logging discipline regression tests for WebSocket handlers.

Issue #5919: Verify that:
1. No ``str(e)`` appears in ``websocket.send_json`` error payloads in WS
   handler modules (exception detail leakage prevention).
2. All ``logger.error`` calls that caught exceptions are converted to
   ``logger.exception`` so the traceback is always captured.

Uses AST scanning so no imports of the production modules are needed and the
tests remain dependency-free.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────────────────────

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

_WS_HANDLER_FILES = [
    _SRC_ROOT / "api" / "routes" / "simulation_ws.py",
    _SRC_ROOT / "api" / "routes" / "chat_ws.py",
]


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _has_str_e_in_send_json(tree: ast.Module) -> list[tuple[int, str]]:
    """Return (lineno, snippet) for any send_json call containing str(e).

    Detects patterns such as::

        await websocket.send_json({..., "detail": str(e), ...})

    using AST traversal so string-encoding differences do not matter.
    """
    violations: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            # Match: <expr>.send_json(...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "send_json":
                # Inspect all values in any dict literal argument
                for arg in node.args + [kw.value for kw in node.keywords]:
                    for nested in ast.walk(arg):
                        if _is_str_e_call(nested):
                            snippet = ast.unparse(node)
                            violations.append((node.lineno, snippet))
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def _is_str_e_call(node: ast.AST) -> bool:
    """Return True if *node* is ``str(<name>)`` where <name> is a single char."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        # Single-character variable names used as exceptions (e, ex, exc, err)
        # plus common multi-char names
        and node.args[0].id in {"e", "ex", "exc", "err", "exception", "error", "error_"}
        and not node.keywords
    )


def _find_logger_error_with_percent_e(tree: ast.Module) -> list[tuple[int, str]]:
    """Find ``logger.error(msg, e)`` calls where *e* is a bare exception name.

    These should be ``logger.exception(msg)`` instead, since the traceback is
    not captured otherwise.
    """
    violations: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "error"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                # Check positional args after the format string for bare exception names
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Name) and arg.id in {
                        "e",
                        "ex",
                        "exc",
                        "err",
                        "exception",
                        "error",
                    }:
                        snippet = ast.unparse(node)
                        violations.append((node.lineno, snippet))
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ws_file", _WS_HANDLER_FILES)
def test_no_str_e_in_websocket_send_json(ws_file: Path) -> None:
    """Assert that no ``str(e)`` appears inside any ``send_json`` call.

    Exposing raw exception text to WebSocket clients leaks internal details
    (stack frames, file paths, library names) that can aid attackers.  All
    error responses must use a generic static string instead.
    """
    assert ws_file.exists(), f"Expected WS handler at {ws_file}"
    tree = _parse(ws_file)
    violations = _has_str_e_in_send_json(tree)

    if violations:
        details = "\n".join(
            f"  line {ln}: {textwrap.shorten(snippet, 120)}"
            for ln, snippet in violations
        )
        pytest.fail(
            f"Exception detail leakage in {ws_file.name}:\n{details}\n"
            "Replace str(e) with a generic message and use logger.exception()."
        )


@pytest.mark.parametrize("ws_file", _WS_HANDLER_FILES)
def test_no_logger_error_with_exception_arg(ws_file: Path) -> None:
    """Assert that ``logger.error(msg, e)`` is not used when *e* is an exception.

    ``logger.error`` does **not** capture the traceback; callers must use
    ``logger.exception`` instead so the full stack trace is recorded in the
    application logs.
    """
    assert ws_file.exists(), f"Expected WS handler at {ws_file}"
    tree = _parse(ws_file)
    violations = _find_logger_error_with_percent_e(tree)

    if violations:
        details = "\n".join(
            f"  line {ln}: {textwrap.shorten(snippet, 120)}"
            for ln, snippet in violations
        )
        pytest.fail(
            f"logger.error with bare exception argument in {ws_file.name}:\n{details}\n"
            "Replace logger.error('msg', e) with logger.exception('msg')."
        )


def test_simulation_ws_has_get_logger_import() -> None:
    """Assert that simulation_ws.py imports get_logger from logging_pkg."""
    ws_file = _WS_HANDLER_FILES[0]
    assert ws_file.exists()
    source = ws_file.read_text(encoding="utf-8")
    assert "get_logger" in source, (
        f"{ws_file.name} must import and use get_logger from logging_pkg"
    )
    assert "logging.getLogger" not in source, (
        f"{ws_file.name} must not use logging.getLogger directly; use get_logger"
    )


def test_chat_ws_uses_get_logger() -> None:
    """Assert that chat_ws.py uses get_logger from logging_pkg."""
    ws_file = _WS_HANDLER_FILES[1]
    assert ws_file.exists()
    source = ws_file.read_text(encoding="utf-8")
    assert "get_logger" in source, (
        f"{ws_file.name} must import and use get_logger from logging_pkg"
    )
    assert "logging.getLogger" not in source, (
        f"{ws_file.name} must not use logging.getLogger directly; use get_logger"
    )
