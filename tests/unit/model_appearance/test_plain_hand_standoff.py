"""Plain (no appearance flag) exports draw one hand per grip, not two (HRG-G1).

The club's grip-hand meshes replace the ``RHandStandoff`` skeleton capsule in
the MuJoCo and MeshCat plain paths, through the one shared predicate
``library.is_grip_hand_standoff``. Visual only: physics is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import library
from src.tools.native_viewer_export.backends._meshcat_scene import (
    build_scene_visuals,
)

pytestmark = pytest.mark.unit
SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def test_predicate_matches_only_the_standoff_body() -> None:
    assert library.is_grip_hand_standoff("solid_reference:Kinetic/Grip/RHandStandoff")
    assert not library.is_grip_hand_standoff("x/Clubface Vector")
    assert not library.is_grip_hand_standoff("x/RHand")


def test_meshcat_plain_has_no_standoff_capsule(tmp_path: Path) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    scene = build_scene_visuals(spec, tmp_path, None)
    assert not [i for i in scene.items if library.is_grip_hand_standoff(i.body)]
    assert any(i.label.endswith("_hand") for i in scene.items)  # hands still drawn


def test_mujoco_plain_visual_export_has_no_standoff_capsule_and_same_physics() -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import (
        full_body_mjcf as exporter,
    )

    raw = SPEC.read_bytes()
    plain = mujoco.MjModel.from_xml_string(exporter.export_full_body_mjcf(raw)[0])
    xml, _ = exporter.export_full_body_mjcf(raw, visual=True)
    visual = mujoco.MjModel.from_xml_string(xml)
    names = [visual.geom(i).name for i in range(visual.ngeom)]
    assert not [n for n in names if "capsule" in n and "RHandStandoff" in n]
    assert any("club_lead_hand" in n for n in names)
    assert visual.nbody == plain.nbody and visual.njnt == plain.njnt
    np.testing.assert_allclose(visual.body_mass, plain.body_mass)
    np.testing.assert_allclose(visual.body_inertia, plain.body_inertia)
