"""Native MuJoCo load/step of MS-51 golfer scenes (#10344).

Uses ``mujoco.MjModel.from_xml_path`` — the MyoSuite package wheel is not
required for MJCF composition smoke. Marked ``requires_myosuite`` so CI
lanes that lack musculoskeletal assets can skip cleanly when the pin is
absent; MuJoCo remains the load backend (Law of Demeter: SDK import here).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_myosuite,
    pytest.mark.requires_mujoco,
]

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "shared" / "models" / "myosuite" / "golf" / "body"
PENETRATION_LIMIT_M = 0.005


@pytest.fixture(scope="module")
def generated_scenes() -> Path:
    myo_body = ROOT / "shared/models/myosuite/myo_sim/body/myobody_simpleupper.xml"
    if not myo_body.is_file():
        pytest.skip("myo_sim not initialized; run scripts/setup_myosuite_models.ps1")
    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        generate_golfer_scene,
    )

    paths = generate_golfer_scene(repo_root=ROOT, output_root=MODELS)
    assert paths.driver.is_file()
    return MODELS


@pytest.mark.requires_mujoco
def test_driver_scene_loads_and_forwards(generated_scenes: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml = generated_scenes / "golfer_myobody_driver.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nq > 0
    assert np.isfinite(data.qpos).all()
    # Address pose: no contact penetration beyond 5 mm budget.
    if data.ncon:
        depths = [float(data.contact[i].dist) for i in range(data.ncon)]
        deepest = min(depths)
        assert deepest > -PENETRATION_LIMIT_M, deepest


@pytest.mark.requires_mujoco
def test_iron_scene_loads(generated_scenes: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml = generated_scenes / "golfer_myobody_iron.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nq > 0


@pytest.mark.requires_mujoco
def test_receipt_records_pin_and_hashes(generated_scenes: Path) -> None:
    receipt_path = generated_scenes / "golfer_myobody_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["myo_sim_pin"].startswith("33f3ded")
    assert receipt["qualification"]["parity_budget_qualified"] is False
    assert "driver" in receipt["clubs"] and "iron" in receipt["clubs"]
