#!/usr/bin/env python3
"""Reject silent-failure patterns in security CI workflows."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WORKFLOWS = (
    Path(".github/workflows/ci-standard.yml"),
    Path(".github/workflows/Jules-Sentinel.yml"),
)
CONTINUE_ON_ERROR = "continue-on-error"
INLINE_PIP_AUDIT_WAIVER = "inline-pip-audit-waiver"
MASKED_SCANNER_FAILURE = "masked-scanner-failure"
PIP_AUDIT_WAIVER_HELPER = "scripts/ci/check_pip_audit_waivers.py"
MASKED_SCANNER_RE = re.compile(
    r"\b(?:pip-audit|pip_audit|npm\s+audit|bandit|semgrep)\b"
    r"[^\n]*\|\|\s*(?:echo|true)\b"
)
SECURITY_STEP_RE = re.compile(r"name:\s*.*(?:audit|scanner|security)", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    """A silent-failure workflow violation."""

    path: Path
    line: int
    pattern: str
    detail: str


def _iter_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"workflow file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _uses_waiver_helper(lines: list[str]) -> bool:
    return any(PIP_AUDIT_WAIVER_HELPER in line for line in lines)


def _line_violation(path: Path, line_number: int, pattern: str, line: str) -> Violation:
    return Violation(path=path, line=line_number, pattern=pattern, detail=line.strip())


def _find_file_violations(path: Path) -> list[Violation]:
    lines = _iter_lines(path)
    uses_helper = _uses_waiver_helper(lines)
    violations: list[Violation] = []
    in_security_step = False
    for line_number, line in enumerate(lines, start=1):
        if re.match(r"\s*-\s+name:", line):
            in_security_step = SECURITY_STEP_RE.search(line) is not None
        if "continue-on-error: true" in line and in_security_step:
            violations.append(
                _line_violation(path, line_number, CONTINUE_ON_ERROR, line)
            )
        if "--ignore-vuln" in line and not uses_helper:
            violations.append(
                _line_violation(path, line_number, INLINE_PIP_AUDIT_WAIVER, line)
            )
        if MASKED_SCANNER_RE.search(line):
            violations.append(
                _line_violation(path, line_number, MASKED_SCANNER_FAILURE, line)
            )
    return violations


def find_violations(paths: list[Path]) -> list[Violation]:
    """Return silent-failure violations for the provided workflow paths."""
    if not paths:
        raise ValueError("at least one workflow path is required")
    violations: list[Violation] = []
    for path in paths:
        violations.extend(_find_file_violations(path))
    return violations


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflows",
        nargs="*",
        type=Path,
        default=list(DEFAULT_WORKFLOWS),
        help="Workflow files to scan.",
    )
    return parser.parse_args()


def main() -> int:
    """Print violations and return non-zero when security CI can fail silently."""
    args = parse_args()
    violations = find_violations(args.workflows)
    for violation in violations:
        print(
            f"{violation.path}:{violation.line}: {violation.pattern}: "
            f"{violation.detail}"
        )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
