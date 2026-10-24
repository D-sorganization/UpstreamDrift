"""Tests for pytest quarantine marker standard, audit scanner, and expiry helper.

RM-CI-030 / Issue #2225: Fleet standard for quarantining flaky tests without
hiding defects.
"""

from __future__ import annotations

import datetime as dt
import json
import textwrap
from pathlib import Path

import pytest

from scripts.pytest_quarantine import (
    QuarantineRecord,
    QuarantineRunnerStats,
    audit_quarantine,
    execute_with_quarantine_retry,
    format_quarantine_report,
    main,
    parse_quarantine_ast,
)

pytestmark = pytest.mark.unit

# --------------------------------------------------------------------------- #
# QuarantineRecord Contract & Validation                                      #
# --------------------------------------------------------------------------- #


def test_quarantine_record_valid() -> None:
    record = QuarantineRecord(
        test_id="tests/test_sample.py::test_flaky",
        file_path="tests/test_sample.py",
        issue="2225",
        owner="gemini",
        expires="2026-11-01",
    )
    assert record.test_id == "tests/test_sample.py::test_flaky"
    assert record.issue == "2225"
    assert record.owner == "gemini"
    assert record.expires_date == dt.date(2026, 11, 1)

    today_before = dt.date(2026, 10, 21)
    today_after = dt.date(2026, 11, 2)
    assert not record.is_expired(as_of=today_before)
    assert record.is_expired(as_of=today_after)
    assert record.days_until_expiry(as_of=today_before) == 11
    assert record.days_until_expiry(as_of=today_after) == -1


def test_quarantine_record_invalid_date() -> None:
    with pytest.raises(ValueError, match="Invalid ISO expiry date"):
        QuarantineRecord(
            test_id="test_foo",
            file_path="test_foo.py",
            issue="100",
            owner="gemini",
            expires="not-a-date",
        )


def test_quarantine_record_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="issue must be a non-empty string"):
        QuarantineRecord(
            test_id="test_foo",
            file_path="test_foo.py",
            issue="",
            owner="gemini",
            expires="2026-11-01",
        )

    with pytest.raises(ValueError, match="owner must be a non-empty string"):
        QuarantineRecord(
            test_id="test_foo",
            file_path="test_foo.py",
            issue="100",
            owner="",
            expires="2026-11-01",
        )


# --------------------------------------------------------------------------- #
# AST Marker Extraction                                                       #
# --------------------------------------------------------------------------- #


def test_parse_quarantine_ast(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import pytest

        @pytest.mark.quarantine(issue="2225", owner="gemini", expires="2026-11-05")
        def test_flaky_a():
            pass

        @pytest.mark.unit
        def test_stable():
            pass

        class TestClass:
            @pytest.mark.quarantine(
                issue="https://github.com/org/repo/issues/10",
                owner="codex",
                expires="2026-10-30",
            )
            def test_flaky_method(self):
                pass
        """
    )
    test_file = tmp_path / "test_module.py"
    test_file.write_text(code, encoding="utf-8")

    records = parse_quarantine_ast(test_file)
    assert len(records) == 2

    rec0 = records[0]
    assert rec0.test_id == f"{test_file.as_posix()}::test_flaky_a"
    assert rec0.issue == "2225"
    assert rec0.owner == "gemini"
    assert rec0.expires == "2026-11-05"

    rec1 = records[1]
    assert rec1.test_id == f"{test_file.as_posix()}::TestClass::test_flaky_method"
    assert rec1.issue == "https://github.com/org/repo/issues/10"
    assert rec1.owner == "codex"
    assert rec1.expires == "2026-10-30"


def test_parse_quarantine_ast_class_level(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import pytest

        @pytest.mark.quarantine(issue="class-123", owner="cursor", expires="2026-11-20")
        class TestQuarantinedSuite:
            def test_method_one(self):
                pass

            def test_method_two(self):
                pass

            def helper_method(self):
                pass
        """
    )
    test_file = tmp_path / "test_class_quarantine.py"
    test_file.write_text(code, encoding="utf-8")

    records = parse_quarantine_ast(test_file)
    assert len(records) == 2
    test_ids = [r.test_id for r in records]
    assert f"{test_file.as_posix()}::TestQuarantinedSuite::test_method_one" in test_ids
    assert f"{test_file.as_posix()}::TestQuarantinedSuite::test_method_two" in test_ids
    for r in records:
        assert r.issue == "class-123"
        assert r.owner == "cursor"
        assert r.expires == "2026-11-20"


def test_parse_quarantine_ast_missing_argument(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import pytest

        @pytest.mark.quarantine(owner="gemini", expires="2026-11-05")
        def test_missing_issue():
            pass
        """
    )
    test_file = tmp_path / "test_broken.py"
    test_file.write_text(code, encoding="utf-8")

    with pytest.raises(ValueError, match="missing required 'issue'"):
        parse_quarantine_ast(test_file)


# --------------------------------------------------------------------------- #
# Quarantine Audit & Policy Gates                                             #
# --------------------------------------------------------------------------- #


def test_audit_quarantine_within_cap_and_valid(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import pytest

        @pytest.mark.quarantine(issue="2225", owner="gemini", expires="2026-11-10")
        def test_flaky_one():
            pass
        """
    )
    (tmp_path / "test_one.py").write_text(code, encoding="utf-8")

    result = audit_quarantine(
        search_paths=[tmp_path],
        max_quarantine=5,
        as_of=dt.date(2026, 10, 21),
    )

    assert len(result.records) == 1
    assert not result.is_over_cap
    assert len(result.expired_records) == 0
    assert result.max_allowed == 5
    assert result.total_count == 1


def test_audit_quarantine_exceeds_cap(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import pytest

        @pytest.mark.quarantine(issue="1", owner="gemini", expires="2026-11-10")
        def test_1(): pass

        @pytest.mark.quarantine(issue="2", owner="gemini", expires="2026-11-10")
        def test_2(): pass
        """
    )
    (tmp_path / "test_cap.py").write_text(code, encoding="utf-8")

    result = audit_quarantine(
        search_paths=[tmp_path],
        max_quarantine=1,
        as_of=dt.date(2026, 10, 21),
    )

    assert result.total_count == 2
    assert result.is_over_cap
    assert result.max_allowed == 1


def test_audit_quarantine_identifies_expired_tests(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import pytest

        @pytest.mark.quarantine(issue="1", owner="gemini", expires="2026-10-15")
        def test_expired(): pass

        @pytest.mark.quarantine(issue="2", owner="gemini", expires="2026-10-25")
        def test_active(): pass
        """
    )
    (tmp_path / "test_exp.py").write_text(code, encoding="utf-8")

    result = audit_quarantine(
        search_paths=[tmp_path],
        max_quarantine=5,
        as_of=dt.date(2026, 10, 21),
    )

    assert result.total_count == 2
    assert len(result.expired_records) == 1
    assert result.expired_records[0].test_id.endswith("test_expired")


# --------------------------------------------------------------------------- #
# Rerun Isolation Scope                                                       #
# --------------------------------------------------------------------------- #


def test_quarantine_rerun_isolation() -> None:
    attempts: dict[str, int] = {"flaky_quarantined": 0, "unmarked_flaky": 0}

    def flaky_quarantined_callable() -> bool:
        attempts["flaky_quarantined"] += 1
        return attempts["flaky_quarantined"] >= 2

    def unmarked_flaky_callable() -> bool:
        attempts["unmarked_flaky"] += 1
        return attempts["unmarked_flaky"] >= 2

    stats = QuarantineRunnerStats()

    # 1. Unmarked test should NEVER be retried; fails on attempt 1
    unmarked_success = execute_with_quarantine_retry(
        test_id="test_unmarked",
        is_quarantined=False,
        test_callable=unmarked_flaky_callable,
        max_retries=2,
        stats=stats,
    )
    assert not unmarked_success
    assert attempts["unmarked_flaky"] == 1
    assert stats.unmarked_failures == 1
    assert stats.passed_after_rerun == 0

    # 2. Quarantined test CAN be retried; succeeds on attempt 2
    quarantined_success = execute_with_quarantine_retry(
        test_id="test_quarantined",
        is_quarantined=True,
        test_callable=flaky_quarantined_callable,
        max_retries=2,
        stats=stats,
    )
    assert quarantined_success
    assert attempts["flaky_quarantined"] == 2
    assert stats.passed_after_rerun == 1
    assert stats.passed_first_attempt == 0


# --------------------------------------------------------------------------- #
# Formatting & JSON Serialization                                             #
# --------------------------------------------------------------------------- #


def test_format_quarantine_report(tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import pytest

        @pytest.mark.quarantine(issue="2225", owner="gemini", expires="2026-10-10")
        def test_expired(): pass
        """
    )
    (tmp_path / "test_report.py").write_text(code, encoding="utf-8")

    result = audit_quarantine(
        search_paths=[tmp_path],
        max_quarantine=5,
        as_of=dt.date(2026, 10, 21),
    )

    report_md = format_quarantine_report(result, as_markdown=True)
    assert "Quarantine Flake Audit" in report_md
    assert "**Expired Quarantines**: 1" in report_md
    assert "test_report.py::test_expired" in report_md

    raw_dict = result.to_dict()
    assert raw_dict["total_count"] == 1
    assert len(raw_dict["expired"]) == 1
    assert raw_dict["max_allowed"] == 5


def test_cli_main_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = textwrap.dedent(
        """
        import pytest

        @pytest.mark.quarantine(issue="2225", owner="gemini", expires="2026-11-20")
        def test_cli(): pass
        """
    )
    (tmp_path / "test_cli.py").write_text(code, encoding="utf-8")
    report_file = tmp_path / "report.json"

    exit_code = main(
        [
            "--paths",
            str(tmp_path),
            "--as-of",
            "2026-10-21",
            "--format",
            "json",
            "--report-file",
            str(report_file),
        ]
    )
    assert exit_code == 0
    assert report_file.exists()
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["total_count"] == 1
    assert not data["is_over_cap"]
