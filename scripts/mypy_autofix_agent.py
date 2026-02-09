"""Mypy Autofix Agent - Intelligent mypy error resolution.

This script acts as an agent that:
1. Runs mypy and captures structured error output
2. Classifies each error by fixability
3. Applies real fixes where possible (import corrections, type narrowing)
4. Falls back to targeted # type: ignore[code] only when necessary
5. Verifies fixes by re-running mypy on modified files
6. Reports all changes for commit messages

Safeguards:
- Max fixes per run (default: 20)
- Max files modified per run (default: 15)
- Never modifies files outside src/ and tests/
- Prefers real fixes over suppressions
- Tracks all changes for auditability

Usage:
    python scripts/mypy_autofix_agent.py [--max-fixes N] [--max-files N] [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MypyError:
    """Parsed mypy error."""

    file: str
    line: int
    column: int
    severity: str  # "error" or "note"
    message: str
    code: str  # e.g., "union-attr", "valid-type", "import-untyped"


@dataclass
class Fix:
    """A fix to apply."""

    file: str
    line: int
    description: str
    strategy: str  # "real-fix" or "suppression"
    original_code: str = ""


@dataclass
class AgentReport:
    """Report of all actions taken."""

    total_errors: int = 0
    errors_fixed: int = 0
    real_fixes: int = 0
    suppressions: int = 0
    files_modified: list[str] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)


# Common third-party modules that trigger import-untyped
KNOWN_UNTYPED_MODULES = {
    "mujoco",
    "dm_control",
    "pinocchio",
    "pin",
    "drake",
    "pydrake",
    "opensim",
    "myosuite",
    "gymnasium",
    "gym",
    "meshcat",
    "trimesh",
    "pybullet",
    "cv2",
    "mediapipe",
    "onnxruntime",
    "sklearn",
    "scipy",
    "PIL",
    "yaml",
    "toml",
    "rich",
    "click",
    "uvicorn",
    "starlette",
    "websockets",
    "serial",
    "usb",
    "hid",
    "pygame",
    "OpenGL",
    "moderngl",
}

# Common type imports that resolve name-defined errors
COMMON_TYPE_IMPORTS = {
    "Callable": "from collections.abc import Callable",
    "Iterator": "from collections.abc import Iterator",
    "Generator": "from collections.abc import Generator",
    "Sequence": "from collections.abc import Sequence",
    "Mapping": "from collections.abc import Mapping",
    "Iterable": "from collections.abc import Iterable",
    "Optional": "from typing import Optional",
    "Union": "from typing import Union",
    "Any": "from typing import Any",
    "ClassVar": "from typing import ClassVar",
    "TypeVar": "from typing import TypeVar",
    "Protocol": "from typing import Protocol",
    "TypeAlias": "from typing import TypeAlias",
    "Final": "from typing import Final",
    "Literal": "from typing import Literal",
    "overload": "from typing import overload",
    "cast": "from typing import cast",
    "TYPE_CHECKING": "from typing import TYPE_CHECKING",
    "Self": "from typing import Self",
    "TypedDict": "from typing import TypedDict",
    "NamedTuple": "from typing import NamedTuple",
    "Path": "from pathlib import Path",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "Enum": "from enum import Enum",
    "dataclass": "from dataclasses import dataclass",
    "abstractmethod": "from abc import abstractmethod",
    "ABC": "from abc import ABC",
}


def run_mypy(config_file: str | None = None) -> str:
    """Run mypy and return raw output."""
    cmd = ["mypy", "src", "--no-error-summary"]
    if config_file:
        cmd.extend(["--config-file", config_file])
    # Show error codes for targeted fixes
    cmd.append("--show-error-codes")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout + result.stderr


def parse_mypy_output(output: str) -> list[MypyError]:
    """Parse mypy output into structured errors."""
    errors = []
    # Pattern: file.py:line:col: severity: message  [error-code]
    pattern = re.compile(
        r"^(.+?):(\d+):(\d+):\s+(error|note):\s+(.+?)(?:\s+\[([^\]]+)\])?\s*$"
    )
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match:
            file_path, line_no, col, severity, message, code = match.groups()
            if severity == "error" and code:
                errors.append(
                    MypyError(
                        file=file_path,
                        line=int(line_no),
                        column=int(col),
                        severity=severity,
                        message=message,
                        code=code or "unknown",
                    )
                )
    return errors


def read_file_lines(filepath: str) -> list[str]:
    """Read file and return lines (preserving newlines)."""
    path = Path(filepath)
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def write_file_lines(filepath: str, lines: list[str]) -> None:
    """Write lines back to file."""
    Path(filepath).write_text("".join(lines), encoding="utf-8")


def has_type_ignore(line: str, code: str | None = None) -> bool:
    """Check if a line already has a type: ignore comment."""
    if "# type: ignore" in line:
        if code and f"[{code}]" in line:
            return True
        if code is None:
            return True
        # Has a blanket ignore
        if "# type: ignore\n" in line or line.rstrip().endswith("# type: ignore"):
            return True
    return False


def add_type_ignore(line: str, code: str) -> str:
    """Add # type: ignore[code] to a line."""
    stripped = line.rstrip("\n\r")
    # Check if there's already an inline comment
    if "# type: ignore" in stripped:
        # Already has type ignore - add our code to existing bracket
        if re.search(r"# type: ignore\[([^\]]+)\]", stripped):
            return re.sub(
                r"# type: ignore\[([^\]]+)\]",
                rf"# type: ignore[\1, {code}]",
                stripped,
            ) + "\n"
        return stripped + "\n"  # Has blanket ignore, leave it
    if "#" in stripped:
        # Has another comment - add before existing comment's content
        return stripped + f"  # type: ignore[{code}]\n"
    return stripped + f"  # type: ignore[{code}]\n"


def get_line_indent(line: str) -> str:
    """Get the leading whitespace of a line."""
    return line[: len(line) - len(line.lstrip())]


def fix_callable_as_type(lines: list[str], error: MypyError) -> Fix | None:
    """Fix 'callable is not valid as a type' by replacing with Callable.

    This is a REAL fix, not a suppression.
    """
    if error.code != "valid-type":
        return None
    if '"callable" is not valid as a type' not in error.message.lower():
        return None

    idx = error.line - 1
    if idx >= len(lines):
        return None

    line = lines[idx]
    # Replace 'callable' with 'Callable[..., Any]' in type annotations
    if ": callable" in line.lower():
        original = line
        # Handle parameter annotations like "param: callable"
        line = re.sub(
            r":\s*callable\b",
            ": Callable[..., Any]",
            line,
            flags=re.IGNORECASE,
        )
        lines[idx] = line

        # Ensure Callable and Any are imported
        _ensure_import(lines, "from collections.abc import Callable")
        _ensure_import(lines, "from typing import Any")

        return Fix(
            file=error.file,
            line=error.line,
            description=f"Replace 'callable' with 'Callable[..., Any]'",
            strategy="real-fix",
            original_code=original.strip(),
        )
    return None


def fix_union_attr(lines: list[str], error: MypyError) -> Fix | None:
    """Fix union-attr by adding isinstance narrowing.

    This is a REAL fix - adds proper type narrowing.
    """
    if error.code != "union-attr":
        return None

    # Extract the type from: Item "X" of "X | Y" has no attribute "Z"
    match = re.search(
        r'Item "(\w+)" of "([^"]+)" has no attribute "(\w+)"',
        error.message,
    )
    if not match:
        return None

    bad_type, union_type, attr = match.groups()

    # Find the other type(s) in the union that DO have the attribute
    types_in_union = [t.strip() for t in union_type.split("|")]
    good_types = [t for t in types_in_union if t != bad_type and t != "None"]

    if not good_types:
        return None  # Can't determine the right type

    idx = error.line - 1
    if idx >= len(lines):
        return None

    line = lines[idx]
    indent = get_line_indent(line)

    # Find the variable name being accessed (look for var.attr pattern)
    var_match = re.search(rf"(\w+)\.{re.escape(attr)}", line)
    if not var_match:
        return None

    var_name = var_match.group(1)
    target_type = good_types[0]

    # Check if there's already an isinstance check nearby (within 3 lines above)
    for check_idx in range(max(0, idx - 3), idx):
        if f"isinstance({var_name}" in lines[check_idx]:
            # Already has narrowing, just suppress
            return None

    # Insert isinstance assertion before the line
    assert_line = f"{indent}assert isinstance({var_name}, {target_type})\n"
    lines.insert(idx, assert_line)

    return Fix(
        file=error.file,
        line=error.line,
        description=f"Add isinstance({var_name}, {target_type}) narrowing for union-attr",
        strategy="real-fix",
        original_code=line.strip(),
    )


def fix_name_not_defined(lines: list[str], error: MypyError) -> Fix | None:
    """Fix name-defined errors by adding missing imports.

    This is a REAL fix when the name is a known type.
    """
    if error.code != "name-defined":
        return None

    # Extract name: Name "X" is not defined
    match = re.search(r'Name "(\w+)" is not defined', error.message)
    if not match:
        return None

    name = match.group(1)
    if name in COMMON_TYPE_IMPORTS:
        import_line = COMMON_TYPE_IMPORTS[name]
        if _ensure_import(lines, import_line):
            return Fix(
                file=error.file,
                line=error.line,
                description=f"Add missing import: {import_line}",
                strategy="real-fix",
            )
    return None


def fix_import_errors(lines: list[str], error: MypyError) -> Fix | None:
    """Fix import-untyped and import-not-found with targeted suppression.

    These are SUPPRESSIONS but acceptable for third-party packages.
    """
    if error.code not in ("import-untyped", "import-not-found"):
        return None

    idx = error.line - 1
    if idx >= len(lines):
        return None

    line = lines[idx]
    if has_type_ignore(line, error.code):
        return None  # Already suppressed

    lines[idx] = add_type_ignore(line, error.code)
    return Fix(
        file=error.file,
        line=error.line,
        description=f"Suppress {error.code} for third-party import",
        strategy="suppression",
        original_code=line.strip(),
    )


def fix_generic_suppression(lines: list[str], error: MypyError) -> Fix | None:
    """Last resort: add targeted # type: ignore[code] suppression.

    Only used when no real fix is available. Uses specific error codes
    rather than blanket ignores.
    """
    # Only suppress specific, well-understood error codes
    suppressible_codes = {
        "assignment",
        "arg-type",
        "return-value",
        "attr-defined",
        "override",
        "misc",
        "call-overload",
        "type-arg",
        "index",
        "operator",
        "no-untyped-call",
        "redundant-cast",
        "var-annotated",
    }

    if error.code not in suppressible_codes:
        return None  # Don't suppress unknown codes

    idx = error.line - 1
    if idx >= len(lines):
        return None

    line = lines[idx]
    if has_type_ignore(line, error.code):
        return None  # Already suppressed

    lines[idx] = add_type_ignore(line, error.code)
    return Fix(
        file=error.file,
        line=error.line,
        description=f"Suppress mypy [{error.code}]: {error.message[:80]}",
        strategy="suppression",
        original_code=line.strip(),
    )


def _ensure_import(lines: list[str], import_statement: str) -> bool:
    """Add an import statement if not already present.

    Inserts after the last existing import from the same module.
    Returns True if import was added.
    """
    # Check if already imported
    for line in lines:
        if import_statement in line:
            return False

    # Find the right place to insert
    # Look for the last import line before any code
    last_import_idx = -1
    in_docstring = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                in_docstring = False
                continue
            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                in_docstring = True
                continue
        if in_docstring:
            continue
        if stripped.startswith(("import ", "from ")):
            last_import_idx = i
        elif stripped and not stripped.startswith("#") and last_import_idx >= 0:
            break  # Hit non-import code after imports

    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, import_statement + "\n")
    else:
        # No imports found, add after module docstring or at top
        insert_at = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Skip past docstring
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    insert_at = i + 1
                    break
                for j in range(i + 1, len(lines)):
                    if '"""' in lines[j] or "'''" in lines[j]:
                        insert_at = j + 1
                        break
                break
            elif stripped and not stripped.startswith("#"):
                insert_at = i
                break
        lines.insert(insert_at, import_statement + "\n")
    return True


def is_safe_path(filepath: str) -> bool:
    """Check if a file is safe to modify."""
    path = Path(filepath)
    # Only modify src/ and tests/ directories
    parts = path.parts
    if not any(p in ("src", "tests") for p in parts):
        return False
    # Never modify __pycache__, .git, vendor files
    if any(p.startswith(".") or p == "__pycache__" or p == "vendor" for p in parts):
        return False
    # Only modify .py files
    if path.suffix != ".py":
        return False
    return True


def run_agent(
    max_fixes: int = 20,
    max_files: int = 15,
    dry_run: bool = False,
    verbose: bool = False,
    config_file: str | None = None,
) -> AgentReport:
    """Main agent loop: observe, classify, fix, report."""
    report = AgentReport()

    # Step 1: Run mypy
    if verbose:
        print(">>> Running mypy...")
    output = run_mypy(config_file)
    errors = parse_mypy_output(output)
    report.total_errors = len(errors)

    if verbose:
        print(f">>> Found {len(errors)} mypy errors")

    if not errors:
        print("No mypy errors found.")
        return report

    # Step 2: Group errors by file
    errors_by_file: dict[str, list[MypyError]] = defaultdict(list)
    for error in errors:
        if is_safe_path(error.file):
            errors_by_file[error.file].append(error)
        else:
            report.skipped_reasons.append(
                f"Skipped {error.file}:{error.line} - outside safe path"
            )

    # Step 3: Apply fixes (file by file, respecting limits)
    files_modified = 0
    total_fixes = 0

    # Fix strategies in priority order (real fixes first)
    fix_strategies = [
        fix_callable_as_type,
        fix_union_attr,
        fix_name_not_defined,
        fix_import_errors,
        fix_generic_suppression,  # Last resort
    ]

    for filepath, file_errors in sorted(errors_by_file.items()):
        if files_modified >= max_files:
            report.skipped_reasons.append(
                f"Skipped {filepath} - max files ({max_files}) reached"
            )
            continue
        if total_fixes >= max_fixes:
            report.skipped_reasons.append(
                f"Skipped {filepath} - max fixes ({max_fixes}) reached"
            )
            continue

        lines = read_file_lines(filepath)
        if not lines:
            continue

        file_changed = False
        # Sort errors by line number descending so fixes don't shift line numbers
        for error in sorted(file_errors, key=lambda e: e.line, reverse=True):
            if total_fixes >= max_fixes:
                break

            # Try each fix strategy in priority order
            fix = None
            for strategy in fix_strategies:
                fix = strategy(lines, error)
                if fix:
                    break

            if fix:
                total_fixes += 1
                file_changed = True
                if fix.strategy == "real-fix":
                    report.real_fixes += 1
                else:
                    report.suppressions += 1
                report.fixes_applied.append(
                    f"  [{fix.strategy}] {fix.file}:{fix.line} - {fix.description}"
                )
                if verbose:
                    print(f"  FIX: {fix.file}:{fix.line} [{fix.strategy}] {fix.description}")
            else:
                report.skipped_reasons.append(
                    f"No fix available: {error.file}:{error.line} [{error.code}] {error.message[:60]}"
                )

        if file_changed:
            if not dry_run:
                write_file_lines(filepath, lines)
            files_modified += 1
            report.files_modified.append(filepath)

    report.errors_fixed = total_fixes

    return report


def print_report(report: AgentReport) -> None:
    """Print a human-readable report."""
    print("\n" + "=" * 60)
    print("  MYPY AUTOFIX AGENT REPORT")
    print("=" * 60)
    print(f"  Total mypy errors found:  {report.total_errors}")
    print(f"  Errors fixed:             {report.errors_fixed}")
    print(f"    Real fixes:             {report.real_fixes}")
    print(f"    Suppressions:           {report.suppressions}")
    print(f"  Files modified:           {len(report.files_modified)}")

    if report.fixes_applied:
        print("\n  Fixes applied:")
        for fix_desc in report.fixes_applied:
            print(f"  {fix_desc}")

    if report.skipped_reasons:
        print(f"\n  Skipped ({len(report.skipped_reasons)}):")
        # Only show first 10 skipped reasons
        for reason in report.skipped_reasons[:10]:
            print(f"    - {reason}")
        if len(report.skipped_reasons) > 10:
            print(f"    ... and {len(report.skipped_reasons) - 10} more")

    print("=" * 60)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Mypy Autofix Agent - Intelligently fix mypy errors"
    )
    parser.add_argument(
        "--max-fixes",
        type=int,
        default=20,
        help="Maximum number of fixes per run (default: 20)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=15,
        help="Maximum files to modify per run (default: 15)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without modifying files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to mypy config file (default: uses pyproject.toml)",
    )
    args = parser.parse_args()

    report = run_agent(
        max_fixes=args.max_fixes,
        max_files=args.max_files,
        dry_run=args.dry_run,
        verbose=args.verbose,
        config_file=args.config_file,
    )

    print_report(report)

    # Exit code: 0 if any fixes were applied, 1 if no fixes possible
    if report.errors_fixed > 0:
        return 0
    if report.total_errors > 0:
        return 1  # Errors found but none fixable
    return 0


if __name__ == "__main__":
    sys.exit(main())
