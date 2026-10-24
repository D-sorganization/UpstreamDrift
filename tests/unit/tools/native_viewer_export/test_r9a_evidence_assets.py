"""R9a (#12688) committed evidence assets and reproduction receipt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = (
    ROOT
    / "docs/development/full_body_models/evidence/appearance/engines/drake_pinocchio"
)
MUJOCO_STILLS = (
    "r9a_driver_mujoco_face_on_address.png",
    "r9a_driver_mujoco_down_the_line_address.png",
    "r9a_driver_mujoco_face_on_impact.png",
    "r9a_driver_mujoco_down_the_line_impact.png",
)


def test_r9a_render_script_and_receipt_exist() -> None:
    script = EVIDENCE / "render_drake_pinocchio_appearance.py"
    receipt_path = EVIDENCE / "r9a_appearance_receipt.json"
    assert script.is_file()
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["issue"] == "#12688"
    assert receipt["swing"]["capture"] == "capture-A"
    assert set(receipt["swing"]["frames"]) == {"address", "impact"}
    assert receipt["settings"]["views"] == ["face_on", "down_the_line"]


def test_r9a_mujoco_reference_stills_are_present() -> None:
    for name in MUJOCO_STILLS:
        path = EVIDENCE / name
        assert path.is_file(), name
        assert path.stat().st_size > 10_000, name
