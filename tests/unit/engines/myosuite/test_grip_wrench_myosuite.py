"""MyoSuite scene grip extraction reuses the shared MuJoCo efc helper (GCV-8, #11714)."""

from __future__ import annotations

import pytest

from src.engines.physics_engines.myosuite.python import golfer_scene
from src.engines.physics_engines.mujoco.python.grip_efc import DEFAULT_GRIP_WELDS

pytestmark = pytest.mark.unit


def test_scene_weld_names_match_the_shared_extraction_convention() -> None:
    fragment = golfer_scene._equality_fragment()
    for name in DEFAULT_GRIP_WELDS.values():
        assert f'name="{name}"' in fragment
    # Hand site is object 1 and the club site object 2 (the club is the second).
    assert 'site1="grip_site_hand_r"' in fragment
    assert 'site2="grip_site_club_r"' in fragment


def test_generated_scene_emits_per_hand_grip_when_the_pinned_assets_exist() -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.grip_efc import (
        grip_analysis_from_efc,
    )

    scene = golfer_scene.resolve_golfer_scene()
    if scene.is_placeholder:
        pytest.skip("pinned myo_sim assets unavailable: placeholder scene")
    try:
        model = mujoco.MjModel.from_xml_path(str(scene.xml_path))
    except ValueError as exc:  # missing included myo_sim files
        pytest.skip(f"myo_sim scene not loadable: {exc}")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grip = grip_analysis_from_efc(model, data)
    assert grip.split_method == "efc_force"
    assert grip.left is not None and grip.right is not None
