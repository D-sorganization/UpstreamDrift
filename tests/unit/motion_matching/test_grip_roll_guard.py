"""SHOW-A13 (#14738): the fleet grip-roll guard, red-first.

The convention tests (back of the wrist on top of the shaft = 0 deg, + toward the
target, the old A2 frames about 75 deg) fail on a tree that still uses the old
zero and pass once SHOW-A7 (#14731) lands. The repo-level
audit passes only when every spec and binding is measured in band or allowlisted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.shared.python.motion_matching.grip_roll import (
    GripPoseState,
    address_grip_roll_deg,
    reference_axes,
)
from src.shared.python.motion_matching.grip_roll_guard import (
    audit,
    discover_bindings,
    discover_specs,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
BINDINGS = ROOT / "src/config/grip_bindings"
GUARD = ROOT / "src/config/grip_roll_guard.json"
TARGET_DEG = 15.0
SHAFT = np.array([-0.718, 0.028, -0.696]) / np.linalg.norm([-0.718, 0.028, -0.696])


def _club_rotation() -> np.ndarray:
    """A proper rotation whose club-frame y axis is the shaft."""
    y = SHAFT
    x = np.cross(y, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, y, np.cross(x, y)])


def _state_with_back_of_hand(world_back: np.ndarray) -> GripPoseState:
    """Both hands with the back of the hand (-dorsal) along ``world_back``."""
    rot = _club_rotation()
    dorsal_club = -(rot.T @ world_back)
    return GripPoseState(rot, dorsal_club, dorsal_club, TargetFrame())


def _axes() -> tuple[np.ndarray, np.ndarray]:
    return reference_axes(SHAFT, TargetFrame())


def test_back_of_the_wrist_on_top_of_the_shaft_reads_zero() -> None:
    _, e_v = _axes()
    lead, trail = address_grip_roll_deg(_state_with_back_of_hand(e_v))
    assert lead == pytest.approx(0.0, abs=1e-6)
    assert trail == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("deg", [15.0, 45.0, 90.0])
def test_rotation_toward_the_target_is_positive(deg: float) -> None:
    e_h, e_v = _axes()
    back = np.cos(np.radians(deg)) * e_v + np.sin(np.radians(deg)) * e_h
    lead, _ = address_grip_roll_deg(_state_with_back_of_hand(back))
    assert lead == pytest.approx(deg, abs=1e-6)


def test_old_a2_frames_read_about_75_degrees() -> None:
    """The A2 grip read 15 deg on the old zero, which is 90 - 15 = 75 deg here."""
    e_h, e_v = _axes()
    old = np.radians(15.0)
    back = np.sin(old) * e_v + np.cos(old) * e_h  # old: n = cos(old) e_h + sin e_v
    lead, _ = address_grip_roll_deg(_state_with_back_of_hand(back))
    assert lead == pytest.approx(75.0, abs=1e-6)


def _guard() -> dict:
    return json.loads(GUARD.read_text(encoding="utf-8"))


def test_every_spec_and_binding_is_accounted_for_in_band_or_allowlisted() -> None:
    found = audit(
        models_dir=MODELS,
        bindings_dir=BINDINGS,
        guard=_guard(),
        target_deg=TARGET_DEG,
    )
    assert not found, "\n".join(f"{v.item}: {v.reason}" for v in found)


def _write_spec(path: Path, dorsal: list[float]) -> None:
    joint = {"axis": dorsal, "type": "hinge"}
    del joint  # shape comes from a real spec below; kept for clarity
    real = json.loads(
        (MODELS / "full_body_spec_anthro_driver.json").read_text(encoding="utf-8")
    )
    path.write_text(json.dumps(real), encoding="utf-8")


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    models, bindings = tmp_path / "models", tmp_path / "bindings"
    models.mkdir()
    bindings.mkdir()
    _write_spec(models / "full_body_spec_x.json", [0.0, 0.0, 1.0])
    (bindings / "b.json").write_text("{}", encoding="utf-8")
    return models, bindings


def _rot() -> list[list[float]]:
    return Rotation.identity().as_matrix().tolist()


def test_an_item_with_neither_a_record_nor_an_allowlist_entry_fails(
    tmp_path: Path,
) -> None:
    models, bindings = _tree(tmp_path)
    found = audit(
        models_dir=models, bindings_dir=bindings, guard={}, target_deg=TARGET_DEG
    )
    assert {v.item for v in found} == {"spec/full_body_spec_x.json", "binding/b.json"}


def test_allowlist_entries_need_a_reason_and_an_owner_issue(tmp_path: Path) -> None:
    models, bindings = _tree(tmp_path)
    guard = {
        "allowlist": {
            "spec/full_body_spec_x.json": {"reason": "", "issue": 1},
            "binding/b.json": {"reason": "pending", "issue": "A11"},
        }
    }
    found = audit(
        models_dir=models, bindings_dir=bindings, guard=guard, target_deg=TARGET_DEG
    )
    assert len(found) == 2
    assert {v.item for v in found} == set(guard["allowlist"])


def test_an_allowlist_entry_for_a_missing_item_fails(tmp_path: Path) -> None:
    models, bindings = _tree(tmp_path)
    guard = {
        "allowlist": {
            "spec/full_body_spec_x.json": {"reason": "r", "issue": 1},
            "binding/b.json": {"reason": "r", "issue": 1},
            "binding/gone.json": {"reason": "r", "issue": 1},
        }
    }
    found = audit(
        models_dir=models, bindings_dir=bindings, guard=guard, target_deg=TARGET_DEG
    )
    assert [v.item for v in found] == ["binding/gone.json"]


def test_a_measured_spec_outside_the_band_fails_unless_allowlisted(
    tmp_path: Path,
) -> None:
    models, bindings = _tree(tmp_path)
    rec = {"spec/full_body_spec_x.json": {"club_rotation": _club_rotation().tolist()}}
    base = {"binding/b.json": {"reason": "r", "issue": 1}}
    out = audit(
        models_dir=models,
        bindings_dir=bindings,
        guard={"records": rec, "allowlist": base},
        target_deg=TARGET_DEG,
        tolerance_deg=1e-9,
    )
    assert [v.item for v in out] == ["spec/full_body_spec_x.json"]
    allowed = {**base, "spec/full_body_spec_x.json": {"reason": "r", "issue": 1}}
    ok = audit(
        models_dir=models,
        bindings_dir=bindings,
        guard={"records": rec, "allowlist": allowed},
        target_deg=TARGET_DEG,
        tolerance_deg=1e-9,
    )
    assert ok == []


def test_an_allowlisted_spec_that_reads_in_band_is_stale(tmp_path: Path) -> None:
    models, bindings = _tree(tmp_path)
    rec = {"spec/full_body_spec_x.json": {"club_rotation": _club_rotation().tolist()}}
    allowed = {
        "binding/b.json": {"reason": "r", "issue": 1},
        "spec/full_body_spec_x.json": {"reason": "r", "issue": 1},
    }
    out = audit(
        models_dir=models,
        bindings_dir=bindings,
        guard={"records": rec, "allowlist": allowed},
        target_deg=TARGET_DEG,
        tolerance_deg=180.0,
    )
    assert [v.item for v in out] == ["spec/full_body_spec_x.json"]
    assert "remove it from the allowlist" in out[0].reason


def test_discovery_lists_real_specs_and_all_bindings() -> None:
    specs = discover_specs(MODELS)
    assert "full_body_spec_anthro_driver.json" in specs
    assert "full_body_spec_v1.appearance.json" not in specs
    assert len(discover_bindings(BINDINGS)) >= 19


LEGACY = re.compile(
    r"palm plane (is )?vertical|0 ?deg = (the )?(hand's |hand )?(palm plane )?"
    r"(is )?vertical|0 = vertical|0 deg vertical",
    re.IGNORECASE,
)
SCANNED = (".py", ".md")
SKIP_PARTS = ("vendor", "evidence", "archive", "node_modules", ".git")
HISTORY = {"DEVELOPMENT_LOG.md"}  # dated state-table history, never reworded


def test_the_old_zero_wording_is_gone_everywhere() -> None:
    """A grep for the old "0 = palm plane vertical" wording must come back empty."""
    this = Path(__file__).resolve()
    hits = []
    for base in (ROOT / "src", ROOT / "docs", ROOT / "tests"):
        for path in base.rglob("*"):
            if path.suffix not in SCANNED or path.resolve() == this:
                continue
            if path.name in HISTORY:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            for n, line in enumerate(path.read_text("utf-8", "ignore").splitlines(), 1):
                if LEGACY.search(line):
                    hits.append(f"{path.relative_to(ROOT)}:{n}")
    assert not hits, hits
