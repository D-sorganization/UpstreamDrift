#!/usr/bin/env python3
"""Run full-src mypy against an accountable checked-in error baseline.

Preconditions:
    ``mypy`` must be installed in the active Python environment. The baseline
    file is JSON with ``schema_version``, ``command``, and ``errors`` keys.
Postconditions:
    Returns 0 only when mypy exits cleanly, or when every reported mypy error
    is already present in the baseline and no baseline error disappeared.
    Any new error, stale baseline entry, or malformed baseline returns 1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path("scripts/config/full_src_mypy_baseline.json")
DEFAULT_TARGET = "src"
DEFAULT_CONFIG = "pyproject.toml"
SCHEMA_VERSION = 1
SUMMARY_PREFIXES = (
    "Found ",
    "Success: ",
)


@dataclass(frozen=True)
class MypyError:
    """Normalized mypy diagnostic used for baseline comparison."""

    path: str
    line: int | None
    column: int | None
    severity: str
    message: str
    code: str | None

    @property
    def key(self) -> str:
        return json.dumps(
            {
                "path": self.path,
                "line": self.line,
                "column": self.column,
                "severity": self.severity,
                "message": self.message,
                "code": self.code,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "message": self.message,
            "code": self.code,
        }


def normalize_path(path: str) -> str:
    """Normalize diagnostic paths for cross-platform baseline stability."""
    return path.strip().replace("\\", "/")


def parse_mypy_line(line: str) -> MypyError | None:
    """Parse one mypy diagnostic line into a stable baseline key."""
    stripped = line.strip()
    if not stripped or stripped.startswith(SUMMARY_PREFIXES):
        return None

    parts = stripped.split(":")
    if len(parts) < 3:
        return None

    path = normalize_path(parts[0])
    line_number = _parse_positive_int(parts[1])
    if line_number is None:
        return None

    index = 2
    column: int | None = None
    parsed_column = _parse_positive_int(parts[index])
    if parsed_column is not None:
        column = parsed_column
        index += 1

    if index >= len(parts):
        return None

    severity = parts[index].strip()
    if severity not in {"error", "note"}:
        return None
    message = ":".join(parts[index + 1 :]).strip()
    if not message:
        return None

    code = None
    if message.endswith("]") and " [" in message:
        message, raw_code = message.rsplit(" [", 1)
        message = message.strip()
        code = raw_code[:-1]

    return MypyError(
        path=path,
        line=line_number,
        column=column,
        severity=severity,
        message=message,
        code=code,
    )


def _parse_positive_int(value: str) -> int | None:
    with contextlib.suppress(ValueError):
        parsed = int(value.strip())
        if parsed > 0:
            return parsed
    return None


def parse_mypy_output(output: str) -> list[MypyError]:
    """Return stable mypy diagnostics parsed from stdout and stderr."""
    diagnostics: list[MypyError] = []
    for line in output.splitlines():
        diagnostic = parse_mypy_line(line)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return diagnostics


def load_baseline(path: Path) -> list[MypyError]:
    """Load the checked-in mypy baseline."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("baseline must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    raw_errors = data.get("errors")
    if not isinstance(raw_errors, list):
        raise ValueError("baseline errors must be a list")
    return [_parse_baseline_error(raw_error) for raw_error in raw_errors]


def _parse_baseline_error(raw_error: Any) -> MypyError:
    if not isinstance(raw_error, dict):
        raise ValueError("each baseline error must be an object")
    path = raw_error.get("path")
    severity = raw_error.get("severity")
    message = raw_error.get("message")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("baseline error path must be a non-empty string")
    if severity not in {"error", "note"}:
        raise ValueError("baseline error severity must be error or note")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("baseline error message must be a non-empty string")
    return MypyError(
        path=normalize_path(path),
        line=_optional_int(raw_error.get("line"), "line"),
        column=_optional_int(raw_error.get("column"), "column"),
        severity=severity,
        message=message.strip(),
        code=_optional_str(raw_error.get("code"), "code"),
    )


def _optional_int(raw_value: Any, field_name: str) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, int) and raw_value > 0:
        return raw_value
    raise ValueError(f"baseline error {field_name} must be a positive integer or null")


def _optional_str(raw_value: Any, field_name: str) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    raise ValueError(f"baseline error {field_name} must be a non-empty string or null")


def compare_to_baseline(
    current_errors: list[MypyError], baseline_errors: list[MypyError]
) -> tuple[list[MypyError], list[MypyError]]:
    """Return ``(new_errors, stale_baseline_errors)``."""
    current_by_key = {error.key: error for error in current_errors}
    baseline_by_key = {error.key: error for error in baseline_errors}
    new_errors = [
        current_by_key[key]
        for key in sorted(set(current_by_key) - set(baseline_by_key))
    ]
    stale_errors = [
        baseline_by_key[key]
        for key in sorted(set(baseline_by_key) - set(current_by_key))
    ]
    return new_errors, stale_errors


def run_mypy(target: str, config_file: str) -> tuple[int, str]:
    """Run mypy and return its exit code plus combined output."""
    command = [
        sys.executable,
        "-m",
        "mypy",
        target,
        "--config-file",
        config_file,
    ]
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.returncode, result.stdout


def write_baseline(path: Path, command: str, errors: list[MypyError]) -> None:
    """Write a sorted baseline snapshot for deliberate review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "errors": [
            error.to_json() for error in sorted(errors, key=lambda item: item.key)
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline from current mypy output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run full-src mypy and enforce the checked-in baseline."""
    args = _build_parser().parse_args(argv)
    baseline_path = Path(args.baseline)
    command = f"python -m mypy {args.target} --config-file {args.config_file}"
    return_code, output = run_mypy(args.target, args.config_file)
    current_errors = parse_mypy_output(output)

    if args.update_baseline:
        write_baseline(baseline_path, command, current_errors)
        print(f"wrote {len(current_errors)} mypy baseline errors to {baseline_path}")
        return 0

    if return_code == 0 and not current_errors:
        print("full-src mypy passed with no baseline debt")
        return 0

    try:
        baseline_errors = load_baseline(baseline_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"full-src mypy baseline failed: {exc}\n")
        return 1

    new_errors, stale_errors = compare_to_baseline(current_errors, baseline_errors)
    if new_errors or stale_errors:
        sys.stderr.write("full-src mypy baseline failed:\n")
        _write_error_section("new errors", new_errors)
        _write_error_section("stale baseline errors", stale_errors)
        return 1

    print(f"full-src mypy matched baseline ({len(baseline_errors)} errors)")
    return 0


def _write_error_section(title: str, errors: list[MypyError]) -> None:
    if not errors:
        return
    sys.stderr.write(f"  {title} ({len(errors)}):\n")
    for error in errors[:25]:
        line = error.line if error.line is not None else "?"
        code = f" [{error.code}]" if error.code else ""
        sys.stderr.write(
            f"    {error.path}:{line}: {error.severity}: {error.message}{code}\n"
        )
    if len(errors) > 25:
        sys.stderr.write(f"    ... {len(errors) - 25} more\n")


if __name__ == "__main__":
    raise SystemExit(main())
