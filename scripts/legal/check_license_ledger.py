"""Advisory checks for the third-party license ledger.

The ledger is intentionally human-maintained: commercial readiness needs
reviewer judgment, not just package metadata. This script provides the CI-sized
contract that every direct dependency declaration has a row and that known
commercialization gates remain visibly flagged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib


_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")
_LEDGER_PACKAGE_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|")
_OPENPOSE_ROW_RE = re.compile(r"^\|\s*`openpose`\s*\|", re.IGNORECASE)


def normalize_package_name(name: str) -> str:
    """Return the normalized PyPA package key used by the ledger."""
    if not name:
        msg = "package name must not be empty"
        raise ValueError(msg)
    return name.lower().replace("_", "-")


def dependency_name(requirement: str) -> str:
    """Extract the package name from a PEP 508-style dependency string."""
    candidate = requirement.split(";", 1)[0].strip().split("[", 1)[0]
    match = _NAME_RE.match(candidate)
    if match is None:
        msg = f"could not parse dependency name from {requirement!r}"
        raise ValueError(msg)
    return normalize_package_name(match.group(0))


def declared_dependency_names(pyproject_path: Path) -> set[str]:
    """Return normalized direct dependency names declared in pyproject.toml."""
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]

    names = {dependency_name(dep) for dep in project.get("dependencies", [])}
    optional = project.get("optional-dependencies", {})
    for dependencies in optional.values():
        names.update(dependency_name(dep) for dep in dependencies)
    return names


def ledger_package_names(ledger_path: Path) -> set[str]:
    """Return normalized package names from markdown table rows."""
    rows: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        match = _LEDGER_PACKAGE_RE.match(line)
        if match is not None and match.group(1).lower() != "package":
            rows.add(normalize_package_name(match.group(1)))
    return rows


def _openpose_row_status(ledger_text: str) -> str | None:
    """Return the stripped Status column of the openpose table row, or None.

    Parses the specific openpose row rather than searching the whole file, so
    legend text that contains 'Non-commercial' / 'Opt-in' as definition prose
    does not produce a false-positive gate result.
    """
    for line in ledger_text.splitlines():
        if _OPENPOSE_ROW_RE.match(line):
            # Pipe-split yields: ['', package, scope, version, license, status, notes, '']
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 7:  # noqa: PLR2004
                return cols[5]
    return None


def validate_license_ledger(pyproject_path: Path, ledger_path: Path) -> list[str]:
    """Return validation errors for the dependency ledger."""
    declared = declared_dependency_names(pyproject_path)
    ledgered = ledger_package_names(ledger_path)

    errors = [
        f"missing license ledger row for {name}" for name in sorted(declared - ledgered)
    ]

    ledger_text = ledger_path.read_text(encoding="utf-8")
    if "`openpose`" not in ledger_text:
        errors.append("missing OpenPose commercialization-gate row")
    else:
        status = _openpose_row_status(ledger_text)
        if status is None or "Non-commercial" not in status or "Opt-in" not in status:
            errors.append("OpenPose row must state Non-commercial and Opt-in")

    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the advisory ledger check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("docs") / "legal" / "licenses.md",
    )
    args = parser.parse_args(argv)

    errors = validate_license_ledger(args.pyproject, args.ledger)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("OK: license ledger covers declared direct dependencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
