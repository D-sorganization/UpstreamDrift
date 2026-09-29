"""Unit tests for anti-phantom-merge Rule 3 path extraction (UD #11124).

`_issue_referenced_paths` must recognize ``scripts/`` and
``.github/workflows/`` references and must not latch onto prose
parentheticals (e.g. ``engines/api/core/shared/robotics``) as path
references.
"""

from __future__ import annotations

import pytest

from scripts.ci.check_phantom_guard_paths import (
    ISSUE_PATH_PATTERN,
    _issue_referenced_paths,
)

pytestmark = pytest.mark.unit


def test_extracts_inline_code_scripts_path() -> None:
    body = (
        "The budget lives in `scripts/config/mypy_exclusion_budget.json` and is "
        "checked by `scripts/check_mypy_exclusion_budget.py`."
    )
    assert _issue_referenced_paths(body) == [
        "scripts/check_mypy_exclusion_budget.py",
        "scripts/config/mypy_exclusion_budget.json",
    ]


def test_recognizes_workflow_paths() -> None:
    body = "Update `.github/workflows/anti-phantom-merge.yml` to use the new check."
    assert _issue_referenced_paths(body) == [
        ".github/workflows/anti-phantom-merge.yml",
    ]


def test_parenthetical_prose_is_not_a_path_reference() -> None:
    body = (
        "PR for #9411 M-1 moves `ratchet_on` via "
        "`scripts/check_mypy_exclusion_budget.py` "
        "(engines/api/core/shared/robotics)."
    )
    assert _issue_referenced_paths(body) == [
        "scripts/check_mypy_exclusion_budget.py",
    ]


def test_parenthetical_split_on_commas_yields_no_fragment_paths() -> None:
    body = (
        "`src/foo.py` covers gates (api/core, rust_core/x, scripts/one)"
    )
    assert _issue_referenced_paths(body) == ["src/foo.py"]


def test_pattern_extended() -> None:
    assert ISSUE_PATH_PATTERN.search("scripts/foo.py")
    assert ISSUE_PATH_PATTERN.search(".github/workflows/ci.yml")


def test_existing_api_path_still_matched() -> None:
    assert _issue_referenced_paths("touch `api/routes/main.py`") == [
        "api/routes/main.py",
    ]


def test_original_10965_body_yields_scripts_paths() -> None:
    """Reproduce the original false positive with issue #10965's prose."""
    body = (
        "`scripts/config/mypy_exclusion_budget.json` declares six "
        "`coverage_gates`: api-routes, data-io, execution-checkpointing, "
        "deployment, optimization and engine-adapters.\n"
        "\n"
        "- `scripts/check_coverage_gates.py` has its own separate hard-coded "
        "gates (engines/api/core/shared/robotics).\n"
        "- No workflow invokes it (`grep -rn check_coverage_gates .github` "
        "finds nothing).\n"
    )
    referenced = _issue_referenced_paths(body)
    assert "scripts/check_coverage_gates.py" in referenced
    assert "scripts/config/mypy_exclusion_budget.json" in referenced
    assert "api/core/shared/robotics" not in referenced