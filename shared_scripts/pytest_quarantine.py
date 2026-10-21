"""Fleet standard for quarantining flaky tests without hiding defects.

RM-CI-030 / Issue #2225 (Phase 2):
Defines the marker contract, AST scanner, expiry audit, and rerun isolation
helpers for @pytest.mark.quarantine.

Key contracts:
1. Marker contract: @pytest.mark.quarantine(issue=..., owner=..., expires=...)
2. Per-repo maximum quarantine count (ratchet).
3. Expiry is enforced in a nightly/scheduled non-blocking job that escalates the
   linked issue, NOT as a collection-time failure in required jobs (Fable 3.2).
4. Reruns are strictly scoped to quarantine-marked tests only; pass-after-rerun
   counts are tracked separately from the required-check failure rate.
5. Evidence rule: Same-commit replay showing inconsistent outcomes. Infra
   failures and reproducible races cannot be quarantined.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_QUARANTINE_PER_REPO: int = 10
ISO_DATE_FORMAT: str = "%Y-%m-%d"


# --------------------------------------------------------------------------- #
# Data types & validation                                                     #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class QuarantineRecord:
    """Metadata recorded for a quarantined flaky test.

    DbC Preconditions:
    - test_id: non-empty string identifying the test
    - file_path: non-empty string path to the test module
    - issue: non-empty string referencing the tracking issue
    - owner: non-empty string indicating the owning agent/engineer
    - expires: ISO-8601 date string 'YYYY-MM-DD'
    """

    test_id: str
    file_path: str
    issue: str
    owner: str
    expires: str

    def __post_init__(self) -> None:
        if not self.test_id or not self.test_id.strip():
            raise ValueError("test_id must be a non-empty string")
        if not self.file_path or not self.file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        if not self.issue or not self.issue.strip():
            raise ValueError("issue must be a non-empty string")
        if not self.owner or not self.owner.strip():
            raise ValueError("owner must be a non-empty string")

        try:
            dt.datetime.strptime(self.expires, ISO_DATE_FORMAT)
        except (ValueError, TypeError) as err:
            msg = (
                f"Invalid ISO expiry date {self.expires!r}; "
                f"expected format 'YYYY-MM-DD'"
            )
            raise ValueError(msg) from err

    @property
    def expires_date(self) -> dt.date:
        """Parse expiry string to datetime.date."""
        return dt.datetime.strptime(self.expires, ISO_DATE_FORMAT).date()

    def is_expired(self, as_of: dt.date | None = None) -> bool:
        """True if the expiry date is strictly before the reference date."""
        ref = as_of or dt.date.today()
        return self.expires_date < ref

    def days_until_expiry(self, as_of: dt.date | None = None) -> int:
        """Number of days until expiry (negative if already expired)."""
        ref = as_of or dt.date.today()
        return (self.expires_date - ref).days

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return {
            "test_id": self.test_id,
            "file_path": self.file_path,
            "issue": self.issue,
            "owner": self.owner,
            "expires": self.expires,
            "days_until_expiry": self.days_until_expiry(),
            "expired": self.is_expired(),
        }


@dataclasses.dataclass(frozen=True)
class QuarantineAuditResult:
    """Summary of a quarantine audit pass across a repository or directory."""

    records: tuple[QuarantineRecord, ...]
    max_allowed: int
    as_of: dt.date

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def is_over_cap(self) -> bool:
        return self.total_count > self.max_allowed

    @property
    def expired_records(self) -> tuple[QuarantineRecord, ...]:
        return tuple(r for r in self.records if r.is_expired(as_of=self.as_of))

    @property
    def active_records(self) -> tuple[QuarantineRecord, ...]:
        return tuple(r for r in self.records if not r.is_expired(as_of=self.as_of))

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.strftime(ISO_DATE_FORMAT),
            "total_count": self.total_count,
            "max_allowed": self.max_allowed,
            "is_over_cap": self.is_over_cap,
            "expired_count": len(self.expired_records),
            "active_count": len(self.active_records),
            "records": [r.to_dict() for r in self.records],
            "expired": [r.to_dict() for r in self.expired_records],
        }


@dataclasses.dataclass
class QuarantineRunnerStats:
    """Runtime execution statistics with isolated rerun accounting."""

    total_executed: int = 0
    passed_first_attempt: int = 0
    passed_after_rerun: int = 0
    quarantined_failures: int = 0
    unmarked_failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_executed": self.total_executed,
            "passed_first_attempt": self.passed_first_attempt,
            "passed_after_rerun": self.passed_after_rerun,
            "quarantined_failures": self.quarantined_failures,
            "unmarked_failures": self.unmarked_failures,
        }


# --------------------------------------------------------------------------- #
# AST parsing for @pytest.mark.quarantine                                    #
# --------------------------------------------------------------------------- #


def _extract_constant_value(node: ast.AST) -> str | None:
    """Extract string value from an AST Constant or Name node."""
    if isinstance(node, ast.Constant):
        return str(node.value)
    return None


def parse_quarantine_ast(file_path: Path | str) -> list[QuarantineRecord]:
    """Parse a Python test file and extract all @pytest.mark.quarantine records.

    Pre: file_path points to an existing Python file.
    Post: list of valid QuarantineRecord instances found on functions/methods.
    Raises ValueError on syntax errors or invalid quarantine decorator metadata.
    """
    path = Path(file_path).resolve()
    posix_path = path.as_posix()
    content = path.read_text(encoding="utf-8")

    tree = ast.parse(content, filename=posix_path)
    records: list[QuarantineRecord] = []

    def process_function(
        fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
        class_prefix: str = "",
    ) -> None:
        for decorator in fn_node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue

            call_func = decorator.func
            is_quarantine = False
            if isinstance(call_func, ast.Attribute) and call_func.attr == "quarantine":
                attr_val = call_func.value
                if isinstance(attr_val, ast.Attribute) and attr_val.attr == "mark":
                    if (
                        isinstance(attr_val.value, ast.Name)
                        and attr_val.value.id == "pytest"
                    ):
                        is_quarantine = True

            if not is_quarantine:
                continue

            kwargs: dict[str, str] = {}
            for kw in decorator.keywords:
                if kw.arg:
                    str_val = _extract_constant_value(kw.value)
                    if str_val is not None:
                        kwargs[kw.arg] = str_val

            missing: list[str] = [
                field for field in ("issue", "owner", "expires") if field not in kwargs
            ]
            if missing:
                missing_str = ", ".join(repr(m) for m in missing)
                msg = (
                    f"Quarantine marker in {posix_path}:{fn_node.lineno} "
                    f"missing required {missing_str}"
                )
                raise ValueError(msg)

            test_name = (
                f"{class_prefix}::{fn_node.name}" if class_prefix else fn_node.name
            )
            node_id = f"{posix_path}::{test_name}"

            record = QuarantineRecord(
                test_id=node_id,
                file_path=posix_path,
                issue=kwargs["issue"],
                owner=kwargs["owner"],
                expires=kwargs["expires"],
            )
            records.append(record)

    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            process_function(item)
        elif isinstance(item, ast.ClassDef):
            for class_item in item.body:
                if isinstance(class_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    process_function(class_item, class_prefix=item.name)

    return records


def scan_quarantine_records(
    search_paths: Sequence[Path | str],
) -> list[QuarantineRecord]:
    """Recursively scan directories or files for quarantine markers."""
    all_records: list[QuarantineRecord] = []

    for raw_path in search_paths:
        p = Path(raw_path)
        if p.is_file() and p.suffix == ".py":
            all_records.extend(parse_quarantine_ast(p))
        elif p.is_dir():
            for child in p.rglob("*.py"):
                if child.name.startswith("test_") or child.name.endswith("_test.py"):
                    try:
                        all_records.extend(parse_quarantine_ast(child))
                    except Exception as err:
                        logger.warning(
                            "Failed to parse quarantine markers from %s: %s", child, err
                        )
                        raise

    return all_records


def audit_quarantine(
    search_paths: Sequence[Path | str],
    max_quarantine: int = DEFAULT_MAX_QUARANTINE_PER_REPO,
    as_of: dt.date | None = None,
) -> QuarantineAuditResult:
    """Run an audit over search_paths and return a QuarantineAuditResult."""
    ref_date = as_of or dt.date.today()
    records = scan_quarantine_records(search_paths)
    return QuarantineAuditResult(
        records=tuple(records),
        max_allowed=max_quarantine,
        as_of=ref_date,
    )


# --------------------------------------------------------------------------- #
# Rerun isolation helper                                                      #
# --------------------------------------------------------------------------- #


def execute_with_quarantine_retry(
    test_id: str,
    is_quarantined: bool,
    test_callable: Callable[[], bool],
    max_retries: int = 2,
    stats: QuarantineRunnerStats | None = None,
) -> bool:
    """Execute a test callable with retries strictly isolated to quarantined tests.

    Pre:
    - test_callable returns True for pass, False for failure.
    - If is_quarantined is False, max_retries is ignored and test is executed once.

    Post:
    - If unmarked and fails: marked as unmarked_failures, returns False immediately.
    - If quarantined and fails initially but passes on retry: marked as
      passed_after_rerun, returns True.
    - If quarantined and fails all attempts: marked as quarantined_failures,
      returns False.
    """
    stats_sink = stats or QuarantineRunnerStats()
    stats_sink.total_executed += 1

    first_attempt_ok = test_callable()
    if first_attempt_ok:
        stats_sink.passed_first_attempt += 1
        return True

    # If test is not quarantined, NEVER retry!
    if not is_quarantined:
        stats_sink.unmarked_failures += 1
        return False

    # Quarantined test: attempt retries
    for _ in range(max_retries):
        if test_callable():
            stats_sink.passed_after_rerun += 1
            return True

    stats_sink.quarantined_failures += 1
    return False


# --------------------------------------------------------------------------- #
# Report formatting                                                           #
# --------------------------------------------------------------------------- #


def format_quarantine_report(
    result: QuarantineAuditResult,
    as_markdown: bool = False,
) -> str:
    """Format audit results for terminal or markdown issue escalation."""
    as_of_str = result.as_of.strftime(ISO_DATE_FORMAT)

    if as_markdown:
        cap_status = "YES (FAIL)" if result.is_over_cap else "No (OK)"
        lines: list[str] = [
            f"# Quarantine Flake Audit ({as_of_str})",
            "",
            f"- **Total Quarantined**: {result.total_count} / {result.max_allowed} max",
            f"- **Over Cap**: {cap_status}",
            f"- **Expired Quarantines**: {len(result.expired_records)}",
            "",
        ]

        if result.expired_records:
            lines.append("## Expired Quarantines (Action Required)")
            lines.append("| Test | Issue | Owner | Expired On | Days Overdue |")
            lines.append("| --- | --- | --- | --- | --- |")
            for rec in result.expired_records:
                days_over = abs(rec.days_until_expiry(as_of=result.as_of))
                lines.append(
                    f"| `{rec.test_id}` | #{rec.issue} | @{rec.owner} "
                    f"| {rec.expires} | +{days_over}d |"
                )
            lines.append("")

        if result.active_records:
            lines.append("## Active Quarantines")
            lines.append("| Test | Issue | Owner | Expires | Days Remaining |")
            lines.append("| --- | --- | --- | --- | --- |")
            for rec in result.active_records:
                days_left = rec.days_until_expiry(as_of=result.as_of)
                lines.append(
                    f"| `{rec.test_id}` | #{rec.issue} | @{rec.owner} "
                    f"| {rec.expires} | {days_left}d |"
                )
            lines.append("")

        return "\n".join(lines)

    # Plain text format
    lines = [
        f"Quarantine Flake Audit (As of {as_of_str}):",
        f"  Total: {result.total_count} (Cap: {result.max_allowed})",
        f"  Over Cap: {'YES' if result.is_over_cap else 'NO'}",
        f"  Expired: {len(result.expired_records)}",
    ]
    for rec in result.expired_records:
        lines.append(
            f"  [EXPIRED] {rec.test_id} (issue: {rec.issue}, "
            f"owner: {rec.owner}, expired: {rec.expires})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI Entrypoint                                                              #
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit pytest quarantine markers and expiry."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["tests"],
        help="Directories or files to scan (default: tests)",
    )
    parser.add_argument(
        "--max-quarantine",
        type=int,
        default=DEFAULT_MAX_QUARANTINE_PER_REPO,
        help=(
            f"Maximum allowed quarantined tests per repo "
            f"(default: {DEFAULT_MAX_QUARANTINE_PER_REPO})"
        ),
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Evaluation date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--fail-on-expired",
        action="store_true",
        help="Exit non-zero if any test has expired (useful for nightly escalation)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output report format",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Optional path to write report output",
    )

    args = parser.parse_args(argv)

    ref_date = (
        dt.datetime.strptime(args.as_of, ISO_DATE_FORMAT).date()
        if args.as_of
        else dt.date.today()
    )

    result = audit_quarantine(
        search_paths=[Path(p) for p in args.paths],
        max_quarantine=args.max_quarantine,
        as_of=ref_date,
    )

    if args.format == "json":
        output = json.dumps(result.to_dict(), indent=2)
    elif args.format == "markdown":
        output = format_quarantine_report(result, as_markdown=True)
    else:
        output = format_quarantine_report(result, as_markdown=False)

    if args.report_file:
        Path(args.report_file).write_text(output, encoding="utf-8")
    else:
        print(output)

    if result.is_over_cap:
        logger.error(
            "Quarantine cap exceeded: %d tests in quarantine, max allowed is %d",
            result.total_count,
            result.max_allowed,
        )
        return 1

    if args.fail_on_expired and result.expired_records:
        logger.error(
            "Found %d expired quarantined tests requiring escalation",
            len(result.expired_records),
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
