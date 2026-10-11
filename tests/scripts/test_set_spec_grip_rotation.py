"""The set_spec_grip_rotation script patches a document idempotently (#12221)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.set_spec_grip_rotation import main

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "development" / "full_body_models"


def _receipt(path: Path, lead: list[float], trail: list[float]) -> Path:
    path.write_text(
        json.dumps(
            {
                "sides": {
                    "lead": {"total_deg": lead},
                    "trail": {"total_deg": trail},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_apply_changes_once_then_is_byte_identical(tmp_path: Path) -> None:
    document = tmp_path / "driver.json"
    shutil.copy(DOCS / "full_body_spec_anthro_driver.json", document)
    receipt = _receipt(tmp_path / "fit.json", [-80.0, 40.0, 5.0], [-30.0, 50.0, 60.0])
    original = document.read_bytes()

    assert main(["apply", str(receipt), str(document)]) == 0
    patched = document.read_bytes()
    assert patched != original
    data = json.loads(patched)
    assert data["subject"]["grip_rotation_deg"] == {
        "L": [-80.0, 40.0, 5.0],
        "R": [-30.0, 50.0, 60.0],
    }
    assert "#12221" in data["provenance"]
    assert "fit.json" in data["provenance"]

    assert main(["apply", str(receipt), str(document)]) == 0
    assert document.read_bytes() == patched


def test_current_values_leave_the_file_alone(tmp_path: Path) -> None:
    document = tmp_path / "driver.json"
    shutil.copy(DOCS / "full_body_spec_anthro_driver.json", document)
    current = json.loads(document.read_text())["subject"]["grip_rotation_deg"]
    receipt = _receipt(tmp_path / "fit.json", current["L"], current["R"])
    before = document.read_bytes()
    assert main(["apply", str(receipt), str(document)]) == 0
    assert document.read_bytes() == before


def test_receipt_without_totals_raises(tmp_path: Path) -> None:
    document = tmp_path / "driver.json"
    shutil.copy(DOCS / "full_body_spec_anthro_driver.json", document)
    receipt = tmp_path / "fit.json"
    receipt.write_text(json.dumps({"sides": {"lead": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="total_deg"):
        main(["apply", str(receipt), str(document)])
