"""The chat child copy must track the pinned Tools tree (UpstreamDrift#12803).

Tools#4897 split ``ChatDockWidget.__init__`` into config objects. The UD child
copy never received it, which left a 16-parameter constructor behind an expiring
architecture-budget exception. These tests pin the contract so the copy cannot
silently fall behind again:

* every vendored ``chat`` file has a UD counterpart that is byte-identical
  modulo the ``src.shared.python`` / ``shared.python`` import-prefix spelling
  (or the UD path is retired and re-exports the pinned module), and
* the dock's constructor and config dataclasses have the canonical shape.

``_chat_dock_widget_qt.py`` is now byte-identical to Tools (#12803); the former
UD-only ``_qt/runtime.py`` extraction was retired with the re-sync.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.helpers.seam_guards import require_vendor_path

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UD_CHAT = _REPO_ROOT / "src" / "shared" / "python" / "chat"
_TOOLS_CHAT = _REPO_ROOT / "vendor/ud-tools/src/shared/python/chat"
_DOCK = "_chat_dock_widget_qt.py"

# Paths compared by contract rather than by text, with the reason.
_TEXT_WAIVERS = {
    _DOCK: "Tools copy exceeds the 1,200-line budget; compared by contract below.",
    "_qt/ui_builder.py": "Routed to pinned Tools via UdSharedImportAliasFinder (#13724); local copy retired in #13720.",
    # Chat tests pin UD-local drift hashes and are not a Tools-synced artefact.
    "tests/test_chat_drift.py": "UD-local baseline hashes.",
}
_CONFIG_CLASSES = (
    "ChatConnectionConfig",
    "ChatPresentationConfig",
    "ChatIntegrationHooks",
)
_MAX_PARAMS = 8


def test_chat_ui_builder_implementation_is_retired_from_ud_copy() -> None:
    """The pinned provider owns the public builder implementation."""
    builder_path = _UD_CHAT / "_qt" / "ui_builder.py"
    assert not _has_top_level_function(builder_path, "build_chat_dock_ui"), (
        "build_chat_dock_ui must come from pinned Tools"
    )


def test_pinned_chat_ui_builder_defines_the_public_implementation() -> None:
    """The exact pinned provider owns all three public builder symbols."""
    provider = require_vendor_path(_TOOLS_CHAT / "_qt" / "ui_builder.py")
    tree = ast.parse(provider.read_text(encoding="utf-8"))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "ChatDockView" in classes
    assert {"mirror_chat_dock_view", "build_chat_dock_ui"} <= functions


def _has_top_level_function(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in tree.body
    )


def _norm(text: str) -> str:
    text = text.replace("\r\n", "\n")
    # Allowed differences: import-prefix spelling, and the one expression that
    # UD's ruff (SIM210) rewrites in chat_dock_widget.py.
    text = text.replace("return token if token else None", "return token or None")
    return text.replace("src.shared.python", "shared.python")


def _tools_files() -> list[Path]:
    root = require_vendor_path(_TOOLS_CHAT)
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _init(path: Path) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in _class(tree, "ChatDockWidget").body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    raise AssertionError("ChatDockWidget.__init__ not found")


def test_chat_child_copy_matches_pinned_tools_tree() -> None:
    drifted = []
    for tools_file in _tools_files():
        rel = tools_file.relative_to(_TOOLS_CHAT).as_posix()
        if rel in _TEXT_WAIVERS:
            continue
        ud_file = _UD_CHAT / rel
        if rel == "_qt/ui_builder.py":
            assert not _has_top_level_function(ud_file, "build_chat_dock_ui"), (
                "the pinned-provider contract owns this public export"
            )
            continue  # the alias suite pins origin, exports, and module identity
        if not ud_file.exists():
            continue  # retired path: served by the pinned tree via the seam finder
        if _norm(ud_file.read_text(encoding="utf-8")) != _norm(
            tools_file.read_text(encoding="utf-8")
        ):
            drifted.append(rel)
    assert not drifted, f"chat child copy drifted from pinned Tools: {drifted}"


def test_dock_constructor_matches_tools_signature() -> None:
    ud = _init(_UD_CHAT / _DOCK)
    tools = _init(require_vendor_path(_TOOLS_CHAT / _DOCK))
    assert ast.dump(ud.args) == ast.dump(tools.args)


def test_dock_config_dataclasses_match_tools() -> None:
    ud_tree = ast.parse((_UD_CHAT / _DOCK).read_text(encoding="utf-8"))
    tools_tree = ast.parse(
        require_vendor_path(_TOOLS_CHAT / _DOCK).read_text(encoding="utf-8")
    )
    for name in _CONFIG_CLASSES:
        assert ast.dump(_class(ud_tree, name)) == ast.dump(_class(tools_tree, name))


def test_dock_constructor_fits_architecture_budget() -> None:
    init = _init(_UD_CHAT / _DOCK)
    a = init.args
    params = [
        p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs) if p.arg != "self"
    ]
    if a.vararg:
        params.append(a.vararg.arg)
    if a.kwarg:
        params.append(a.kwarg.arg)
    assert len(params) <= _MAX_PARAMS
    # Constructor length is owned by scripts/ci/check_architecture_budget.py: the
    # Tools-owned copy is 112 lines and carries a tracked function-lines exception.


def test_dock_rejects_legacy_keyword_arguments() -> None:
    init = _init(_UD_CHAT / _DOCK)
    names = {p.arg for p in init.args.args + init.args.kwonlyargs}
    assert {"connection", "presentation", "integrations", "parent"} <= names
    assert "app_context" not in names
    assert init.args.kwarg is None


def test_architecture_budget_has_no_chat_dock_parameters_exception() -> None:
    import json

    budget = json.loads(
        (_REPO_ROOT / "scripts/config/architecture_budget.json").read_text("utf-8")
    )
    entries = budget.get("exceptions", [])
    assert not [
        e
        for e in entries
        if e.get("symbol") == "ChatDockWidget.__init__"
        and e.get("rule") == "parameters"
    ]
