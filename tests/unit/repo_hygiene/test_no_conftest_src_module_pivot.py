"""Guard against a conftest pivoting ``sys.modules["src"]`` globally (UD #9409).

Why this rule exists
--------------------
``sys.modules`` is process-global state.  A ``conftest.py`` that rebinds
``sys.modules["src"]`` (the engine conftest pivot pattern) does not only
affect its own directory: the rebind is visible to every test that collects
or runs while it is installed, and a pivot installed without save/restore
evicts the real repo ``src`` package for the remainder of the session.
That failure class broke ``tests/unit/utils/test_path_validation.py`` when
the C3D-viewer conftest first installed its pivot (issue #9402, PR #9404),
and it motivated the per-directory scoping now carried by
``tests.helpers.engine_src_pivot.EngineSrcPivot``.

The two legitimate engine pivots (``tests/unit/c3d_viewer/ui`` and
``tests/unit/engines/simscape/three_d_gui``) go through that shared,
fully-restoring helper and therefore never write ``sys.modules["src"]``
directly.  The narrow rule enforced here is: **a conftest.py must never
bind, delete, pop, or setdefault the ``"src"`` key of ``sys.modules``**.
Directory-scoped shadowing that genuinely needs the pivot belongs in
``EngineSrcPivot`` (or must be added there), so that paired enter/exit
hooks -- not conftest ad-hoc state -- own the global namespace.

This guard complements ``test_no_permanent_src_module_shadow.py``, which
exercises the two known pivot conftests' restore behaviour; this file is
the static, repo-wide tripwire that fails the moment a *new* conftest
installs its own pivot.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Directories never imported by pytest's conftest loader; excluded from the
# scan so vendored / third-party trees cannot fail the report.
_SKIPPED_DIRS = frozenset({".git", "node_modules", "vendor"})

# This ledger only ever ratchets down.  Adding an entry requires an owner and
# a tracking issue; the guard above explains why a conftest-owned ``src``
# pivot is a session-poisoning defect rather than a style preference.
_PREEXISTING_SRC_PIVOT_CONFTESTS: frozenset[str] = frozenset()


def _iter_conftest_sources() -> Iterator[tuple[Path, ast.Module]]:
    for path in sorted(_REPO_ROOT.rglob("conftest.py")):
        parts = path.relative_to(_REPO_ROOT).parts
        if any(part.startswith(".") for part in parts[:-1]):
            continue
        if _SKIPPED_DIRS.intersection(parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        yield path, tree


def _is_sys_modules(node: ast.expr) -> bool:
    """Whether *node* is the ``sys.modules`` attribute expression."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _is_src_key(node: ast.expr) -> bool:
    """Whether *node* is the literal ``"src"`` subscript key."""
    return isinstance(node, ast.Constant) and node.value == "src"


def _assigned_targets(node: ast.AST) -> list[ast.expr]:
    """Return the written/delete targets of an assignment-like node."""
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    if isinstance(node, ast.AugAssign):
        return [node.target]
    if isinstance(node, ast.Delete):
        return list(node.targets)
    return []


def _find_src_pivot_lines(tree: ast.Module) -> list[int]:
    """Line numbers in *tree* that write the ``"src"`` key of ``sys.modules``.

    Covers every write shape a pivot can use: plain/augmented/annotated
    assignment, ``del``, and ``sys.modules.pop("src", ...)`` /
    ``sys.modules.setdefault("src", ...)``.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        for target in _assigned_targets(node):
            if (
                isinstance(target, ast.Subscript)
                and _is_sys_modules(target.value)
                and _is_src_key(target.slice)
            ):
                lines.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"pop", "setdefault"}
            and _is_sys_modules(node.func.value)
            and node.args
            and _is_src_key(node.args[0])
        ):
            lines.append(node.lineno)
    return sorted(lines)


def test_conftests_do_not_pivot_the_src_module_globally() -> None:
    """No conftest.py may write ``sys.modules["src"]`` directly.

    DbC: the engine ``src`` shadow is process-global state and belongs to
    the paired enter/exit hooks of ``tests.helpers.engine_src_pivot``;
    a conftest-owned rebind leaks to every other test in the session.
    """
    offenders: list[str] = []
    for path, tree in _iter_conftest_sources():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _PREEXISTING_SRC_PIVOT_CONFTESTS:
            continue
        lines = _find_src_pivot_lines(tree)
        if lines:
            offenders.append(f"{rel}: sys.modules['src'] written at lines {lines}")

    assert not offenders, (
        "conftest.py files may not pivot sys.modules['src'] directly "
        "(use tests.helpers.engine_src_pivot.EngineSrcPivot):\n" + "\n".join(offenders)
    )


def test_ledger_only_lists_conftests_that_still_exist() -> None:
    """The allowlist must ratchet down: stale entries are a failure."""
    tracked = {
        path.relative_to(_REPO_ROOT).as_posix()
        for path, _tree in _iter_conftest_sources()
    }
    stale = sorted(_PREEXISTING_SRC_PIVOT_CONFTESTS - tracked)
    assert not stale, f"ledger lists conftests that no longer exist: {stale}"
