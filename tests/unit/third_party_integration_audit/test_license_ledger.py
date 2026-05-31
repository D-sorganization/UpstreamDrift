"""License-ledger coverage checks for direct dependency declarations."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_LEDGER_SCRIPT = ROOT / "scripts" / "legal" / "check_license_ledger.py"

_spec = importlib.util.spec_from_file_location("check_license_ledger", _LEDGER_SCRIPT)
assert _spec is not None
assert _spec.loader is not None
_ledger_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ledger_module
_spec.loader.exec_module(_ledger_module)

declared_dependency_names = _ledger_module.declared_dependency_names
ledger_package_names = _ledger_module.ledger_package_names
validate_license_ledger = _ledger_module.validate_license_ledger


def test_license_ledger_covers_declared_dependencies() -> None:
    """Every direct dependency in pyproject.toml must have a ledger row."""
    declared = declared_dependency_names(ROOT / "pyproject.toml")
    ledgered = ledger_package_names(ROOT / "docs" / "legal" / "licenses.md")

    assert declared <= ledgered


def test_license_ledger_flags_openpose_as_non_commercial_opt_in() -> None:
    """OpenPose must stay visibly fenced for commercial builds."""
    errors = validate_license_ledger(
        pyproject_path=ROOT / "pyproject.toml",
        ledger_path=ROOT / "docs" / "legal" / "licenses.md",
    )

    assert "missing OpenPose commercialization-gate row" not in errors
    assert "OpenPose row must state Non-commercial and Opt-in" not in errors


def test_license_ledger_validates_openpose_row_cells(tmp_path: Path) -> None:
    """OpenPose status and notes must be validated on its own table row."""
    ledger = (ROOT / "docs" / "legal" / "licenses.md").read_text(encoding="utf-8")
    bad_lines = []
    for line in ledger.splitlines():
        if line.startswith("| `openpose`"):
            bad_lines.append(
                "| `openpose` | external tool | not packaged in `pyproject.toml` | "
                "Commercial | Default | Approved for default commercial ingestion. |"
            )
        else:
            bad_lines.append(line)
    bad_ledger = "\n".join(bad_lines) + "\n"
    ledger_path = tmp_path / "licenses.md"
    ledger_path.write_text(bad_ledger, encoding="utf-8")

    errors = validate_license_ledger(
        pyproject_path=ROOT / "pyproject.toml",
        ledger_path=ledger_path,
    )

    assert "OpenPose row must state Non-commercial and Opt-in" in errors


def test_license_ledger_script_reports_clean() -> None:
    """The advisory script must share the same contract as CI/tests."""
    errors = validate_license_ledger(
        pyproject_path=ROOT / "pyproject.toml",
        ledger_path=ROOT / "docs" / "legal" / "licenses.md",
    )

    assert errors == []
