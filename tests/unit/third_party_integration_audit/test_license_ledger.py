"""License-ledger coverage checks for direct dependency declarations."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
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
_openpose_row_status = _ledger_module._openpose_row_status


def test_license_ledger_covers_declared_dependencies() -> None:
    """Every direct dependency in pyproject.toml must have a ledger row."""
    declared = declared_dependency_names(ROOT / "pyproject.toml")
    ledgered = ledger_package_names(ROOT / "docs" / "legal" / "licenses.md")

    assert declared <= ledgered


def test_license_ledger_flags_openpose_as_non_commercial_opt_in() -> None:
    """OpenPose must stay visibly fenced for commercial builds."""
    ledger_text = (ROOT / "docs" / "legal" / "licenses.md").read_text(encoding="utf-8")

    status = _openpose_row_status(ledger_text)
    assert status is not None, "openpose row not found in ledger"
    assert "Non-commercial" in status, (
        f"openpose status missing 'Non-commercial': {status!r}"
    )
    assert "Opt-in" in status, f"openpose status missing 'Opt-in': {status!r}"


def test_openpose_gate_validates_row_not_legend() -> None:
    """validate_license_ledger must catch a downgraded openpose row status.

    The legend section always contains 'Non-commercial' and 'Opt-in' as definition
    text. If the check searches the whole file instead of the specific row, it
    passes even when the openpose Status cell is changed to 'Commercial-OK'.
    """
    ledger_path = ROOT / "docs" / "legal" / "licenses.md"
    real_text = ledger_path.read_text(encoding="utf-8")

    # Tamper: replace the openpose row's status cell with "Commercial-OK"
    tampered = re.sub(
        r"(?m)^(\|\s*`openpose`(?:[^|]*\|){4})\s*Non-commercial, Opt-in\s*(\|)",
        r"\1 Commercial-OK \2",
        real_text,
    )
    assert tampered != real_text, "tamper regex did not match — test fixture is broken"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(tampered)
        tmp_path = Path(tmp_file.name)

    try:
        errors = validate_license_ledger(ROOT / "pyproject.toml", tmp_path)
        assert errors, (
            "expected an error when openpose row status is 'Commercial-OK' "
            "but 'Non-commercial'/'Opt-in' still appear in the legend"
        )
    finally:
        tmp_path.unlink()


def test_license_ledger_script_reports_clean() -> None:
    """The advisory script must share the same contract as CI/tests."""
    errors = validate_license_ledger(
        pyproject_path=ROOT / "pyproject.toml",
        ledger_path=ROOT / "docs" / "legal" / "licenses.md",
    )

    assert errors == []
