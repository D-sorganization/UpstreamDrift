"""sidekick state_manager must not reach across the tool-tree boundary (#3333).

state_manager previously imported ``safe_read_json``/``safe_write_json`` from a
foreign application tree via the bare name ``utils.file_utils`` and ``UTC`` from
the bare ``compatibility`` module. Both resolved only by sys.path accident and
broke in installed deployments. These tests pin the new self-contained behaviour.
"""

from __future__ import annotations

import ast
import datetime
from pathlib import Path

import pytest

_STATE_MANAGER = Path(__file__).resolve().parents[1] / "utils" / "state_manager.py"
_WGS_REACTOR = (
    Path(__file__).resolve().parents[1]
    / "process_calculators"
    / "wgs_reactor_calculator.py"
)


@pytest.mark.unit
def test_state_manager_has_no_cross_tree_imports() -> None:
    """No ``from utils...`` or ``from compatibility...`` imports remain."""
    tree = ast.parse(_STATE_MANAGER.read_text(encoding="utf-8"))
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in {"utils", "compatibility"}:
                offending.append(node.module)
    assert offending == [], (
        f"state_manager still imports across the tool-tree boundary: {offending}"
    )


@pytest.mark.unit
def test_wgs_reactor_reads_json_from_sidekick_json_io() -> None:
    """WGS should not reach through state_manager for JSON helper functions."""
    tree = ast.parse(_WGS_REACTOR.read_text(encoding="utf-8"))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "sidekick.utils.state_manager"
    ]

    assert imports == [], "WGS still imports JSON helpers through state_manager"


@pytest.mark.unit
def test_state_manager_utc_is_stdlib_utc() -> None:
    from sidekick.utils.state_manager import UTC

    assert UTC is datetime.timezone.utc  # noqa: UP017 - Python 3.10 lacks datetime.UTC.


@pytest.mark.unit
def test_json_io_round_trip(tmp_path: Path) -> None:
    from sidekick.utils.json_io import safe_read_json, safe_write_json

    target = tmp_path / "nested" / "data.json"
    payload = {"k": [1, 2, 3], "s": "v"}
    assert safe_write_json(target, payload) is True
    assert safe_read_json(target) == payload


@pytest.mark.unit
def test_json_io_missing_returns_default(tmp_path: Path) -> None:
    from sidekick.utils.json_io import safe_read_json

    assert safe_read_json(tmp_path / "absent.json", default="DEF") == "DEF"


@pytest.mark.unit
def test_json_io_none_path_raises() -> None:
    from sidekick.utils.json_io import safe_read_json, safe_write_json

    with pytest.raises(ValueError, match="file_path must be provided"):
        safe_read_json(None)
    with pytest.raises(ValueError, match="file_path must be provided"):
        safe_write_json(None, {})


@pytest.mark.unit
def test_json_io_non_serializable_returns_false(tmp_path: Path) -> None:
    from sidekick.utils.json_io import safe_write_json

    # object() is not JSON-serialisable and there is no default= serialiser.
    assert safe_write_json(tmp_path / "bad.json", {"x": object()}) is False
