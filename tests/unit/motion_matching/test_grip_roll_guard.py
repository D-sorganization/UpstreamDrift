"""SHOW-A13 (#14738): the fleet grip-roll guard."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

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
    x = np.cross(SHAFT, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, SHAFT, np.cross(x, SHAFT)])


def _state_with_back_of_hand(world_back: np.ndarray) -> GripPoseState:
    rot = _club_rotation()
    dorsal_club = -(rot.T @ world_back)
    return GripPoseState(rot, dorsal_club, dorsal_club, TargetFrame())


def test_convention_contract_zeros_signs_and_old_a2_frames() -> None:
    e_h, e_v = reference_axes(SHAFT, TargetFrame())
    lead, trail = address_grip_roll_deg(_state_with_back_of_hand(e_v))
    assert lead == pytest.approx(0.0, abs=1e-6)
    assert trail == pytest.approx(0.0, abs=1e-6)
    for deg in (15.0, 45.0, 90.0):
        b = np.cos(np.radians(deg)) * e_v + np.sin(np.radians(deg)) * e_h
        assert address_grip_roll_deg(_state_with_back_of_hand(b))[0] == pytest.approx(
            deg, abs=1e-6
        )
    old = np.radians(15.0)
    b_old = np.sin(old) * e_v + np.cos(old) * e_h
    assert address_grip_roll_deg(_state_with_back_of_hand(b_old))[0] == pytest.approx(
        75.0, abs=1e-6
    )


def _guard() -> dict:
    return json.loads(GUARD.read_text(encoding="utf-8"))


def test_every_spec_and_binding_is_accounted_for_in_band_or_allowlisted() -> None:
    found = audit(
        models_dir=MODELS, bindings_dir=BINDINGS, guard=_guard(), target_deg=TARGET_DEG
    )
    assert not found, "\n".join(f"{v.item}: {v.reason}" for v in found)


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    models, bindings = tmp_path / "models", tmp_path / "bindings"
    models.mkdir()
    bindings.mkdir()
    real = json.loads(
        (MODELS / "full_body_spec_anthro_driver.json").read_text(encoding="utf-8")
    )
    (models / "full_body_spec_x.json").write_text(json.dumps(real), encoding="utf-8")
    (bindings / "b.json").write_text("{}", encoding="utf-8")
    return models, bindings


def test_allowlist_validation_rules(tmp_path: Path) -> None:
    models, bindings = _tree(tmp_path)
    assert {
        v.item
        for v in audit(
            models_dir=models, bindings_dir=bindings, guard={}, target_deg=TARGET_DEG
        )
    } == {"spec/full_body_spec_x.json", "binding/b.json"}
    guard = {
        "allowlist": {
            "spec/full_body_spec_x.json": {"reason": "", "issue": 1},
            "binding/b.json": {"reason": "pending", "issue": "A11"},
            "binding/gone.json": {"reason": "r", "issue": 1},
        }
    }
    found = audit(
        models_dir=models, bindings_dir=bindings, guard=guard, target_deg=TARGET_DEG
    )
    assert len(found) == 3 and {v.item for v in found} == set(guard["allowlist"])


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
    assert (
        audit(
            models_dir=models,
            bindings_dir=bindings,
            guard={"records": rec, "allowlist": allowed},
            target_deg=TARGET_DEG,
            tolerance_deg=1e-9,
        )
        == []
    )


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
    r"\bpalm plane (is )?vertical|0 ?deg = (the )?(hand's |hand )?(palm plane )?"
    r"(is )?vertical|0 = vertical|\b0 deg vertical",
    re.IGNORECASE,
)
SCANNED = (".py", ".md")
SKIP_PARTS = ("vendor", "evidence", "archive", "node_modules", ".git")
HISTORY = {"DEVELOPMENT_LOG.md"}


def test_the_old_zero_wording_is_gone_everywhere() -> None:
    this = Path(__file__).resolve()
    hits = []
    try:
        res = subprocess.run(
            ["git", "ls-files", "src", "docs", "tests"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        paths = [ROOT / p for p in res.stdout.splitlines() if p.endswith(SCANNED)]
    except (subprocess.SubprocessError, OSError):
        paths = [
            p
            for b in (ROOT / "src", ROOT / "docs", ROOT / "tests")
            for p in b.rglob("*")
            if p.suffix in SCANNED
        ]

    for path in paths:
        if (
            path.resolve() == this
            or path.name in HISTORY
            or any(part in SKIP_PARTS for part in path.parts)
        ):
            continue
        for n, line in enumerate(path.read_text("utf-8", "ignore").splitlines(), 1):
            if LEGACY.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{n}")
    assert not hits, hits
