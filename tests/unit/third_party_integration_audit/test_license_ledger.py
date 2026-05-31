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
    ledger = (ROOT / "docs" / "legal" / "licenses.md").read_text(encoding="utf-8")

    assert "`openpose`" in ledger
    assert "Non-commercial" in ledger
    assert "Opt-in" in ledger


def test_license_ledger_script_reports_clean() -> None:
    """The advisory script must share the same contract as CI/tests."""
    errors = validate_license_ledger(
        pyproject_path=ROOT / "pyproject.toml",
        ledger_path=ROOT / "docs" / "legal" / "licenses.md",
    )

    assert errors == []
