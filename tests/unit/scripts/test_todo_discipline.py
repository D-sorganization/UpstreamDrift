"""Tests for the TODO/FIXME discipline checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import check_todo_discipline as mod

pytestmark = pytest.mark.unit


def test_python_comment_todo_requires_issue_reference(tmp_path: Path) -> None:
    source = tmp_path / "src" / "pkg" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "# TODO: untracked cleanup",
                "# FIXME(#1234): tracked cleanup",
                "value = 'TODO in strings is ignored'",
            ]
        ),
        encoding="utf-8",
    )

    assert mod._scan_file(source) == [(1, "# TODO: untracked cleanup")]


def test_workflow_comment_todo_requires_issue_reference(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "nightly.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "\n".join(
            [
                "name: nightly",
                "jobs:",
                "  test:",
                "    steps:",
                "      # TODO: generate dashboard",
                "      # Future: upload dashboard #8220",
            ]
        ),
        encoding="utf-8",
    )

    assert mod._scan_file(workflow) == [(5, "      # TODO: generate dashboard")]


def test_workflow_shell_placeholder_requires_issue_reference(tmp_path: Path) -> None:
    workflow = tmp_path / ".github" / "workflows" / "nightly.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "\n".join(
            [
                "name: nightly",
                "jobs:",
                "  test:",
                "    steps:",
                '      - run: echo "Dashboard generation placeholder"',
                '      - run: echo "Tracked placeholder #8220"',
            ]
        ),
        encoding="utf-8",
    )

    assert mod._scan_file(workflow) == [
        (5, '      - run: echo "Dashboard generation placeholder"')
    ]


def test_iter_scanned_files_includes_src_tests_scripts_and_workflows(
    tmp_path: Path,
) -> None:
    expected_paths = [
        tmp_path / "src" / "pkg" / "module.py",
        tmp_path / "tests" / "unit" / "test_module.py",
        tmp_path / "scripts" / "tool.py",
        tmp_path / ".github" / "workflows" / "ci.yml",
    ]
    for path in expected_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    scanned = {
        path.relative_to(tmp_path).as_posix()
        for path in mod._iter_scanned_files(tmp_path)
    }

    assert scanned == {
        "src/pkg/module.py",
        "tests/unit/test_module.py",
        "scripts/tool.py",
        ".github/workflows/ci.yml",
    }
