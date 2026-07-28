from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.check_optional_stack_skip_policy import (
    OptionalStackPolicyError,
    check_optional_stack_skip_policy,
    main,
    summarize_junit,
)

pytestmark = pytest.mark.unit


def _write_junit(path: Path, body: str) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f"<testsuites><testsuite>{body}</testsuite></testsuites>\n",
        encoding="utf-8",
    )


def test_available_dependency_accepts_at_least_one_passed_case(tmp_path: Path) -> None:
    junit = tmp_path / "pinocchio.xml"
    _write_junit(
        junit,
        '<testcase name="test_recorder_basic"/>'
        '<testcase name="test_optional_gui"><skipped message="headless"/></testcase>',
    )

    counts = check_optional_stack_skip_policy(
        junit_path=junit,
        dependency_available=True,
        pytest_exit_code=0,
    )

    assert counts["passed"] == 1
    assert counts["skipped"] == 1


def test_available_dependency_rejects_all_skipped_cases(tmp_path: Path) -> None:
    junit = tmp_path / "pinocchio.xml"
    _write_junit(
        junit,
        '<testcase name="test_recorder_basic"><skipped message="missing gui"/></testcase>',
    )

    with pytest.raises(OptionalStackPolicyError, match="no ecosystem testcase passed"):
        check_optional_stack_skip_policy(
            junit_path=junit,
            dependency_available=True,
            pytest_exit_code=0,
        )


def test_unavailable_dependency_allows_all_skipped_cases(tmp_path: Path) -> None:
    junit = tmp_path / "pinocchio.xml"
    _write_junit(
        junit,
        '<testcase name="test_recorder_basic"><skipped message="missing pinocchio"/></testcase>',
    )

    counts = check_optional_stack_skip_policy(
        junit_path=junit,
        dependency_available=False,
        pytest_exit_code=0,
    )

    assert counts["passed"] == 0
    assert counts["skipped"] == 1


def test_available_dependency_rejects_pytest_collection_exit_code_5(
    tmp_path: Path,
) -> None:
    with pytest.raises(OptionalStackPolicyError, match="collected zero"):
        check_optional_stack_skip_policy(
            junit_path=tmp_path / "missing.xml",
            dependency_available=True,
            pytest_exit_code=5,
        )


def test_unavailable_dependency_allows_pytest_collection_exit_code_5(
    tmp_path: Path,
) -> None:
    counts = check_optional_stack_skip_policy(
        junit_path=tmp_path / "missing.xml",
        dependency_available=False,
        pytest_exit_code=5,
    )

    assert counts == {"passed": 0, "skipped": 0, "failed": 0, "error": 0, "matched": 0}


def test_junit_summary_counts_failures_and_errors(tmp_path: Path) -> None:
    junit = tmp_path / "pinocchio.xml"
    _write_junit(
        junit,
        '<testcase name="test_passed"/>'
        '<testcase name="test_skipped"><skipped/></testcase>'
        '<testcase name="test_failed"><failure/></testcase>'
        '<testcase name="test_error"><error/></testcase>',
    )

    assert summarize_junit(junit) == {
        "passed": 1,
        "skipped": 1,
        "failed": 1,
        "error": 1,
        "matched": 4,
    }


def test_main_writes_summary_file(tmp_path: Path) -> None:
    junit = tmp_path / "pinocchio.xml"
    summary = tmp_path / "summary.md"
    _write_junit(junit, '<testcase name="test_recorder_basic"/>')

    assert (
        main(
            [
                "--junit",
                str(junit),
                "--available",
                "true",
                "--pytest-exit-code",
                "0",
                "--summary-file",
                str(summary),
            ]
        )
        == 0
    )
    assert "Passed: 1 | Failed: 0 | Skipped: 0 | Matched: 1" in summary.read_text(
        encoding="utf-8"
    )
