"""Historical frame priors retain source uncertainty and native realizability."""

import json

import numpy as np
import pytest

from src.shared.python.model_appearance.player_models import frame_prior, player_profile

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240)]


@pytest.mark.parametrize(
    "player, expected",
    [
        ("mac_ogrady", -0.635),
        ("tom_watson", -0.391),
        ("greg_norman_1986", -0.452),
        ("george_knudson", -0.501),
        ("john_schlee", -0.858),
        ("johnny_miller", -0.762),
        ("lee_trevino", 0.248),
    ],
)
def test_sourced_historical_frame_prior(player, expected):
    """Resolve the selected roster source and report a bounded physical estimate."""
    profile = player_profile(player)
    prior = frame_prior(profile)
    assert profile["frame_prior"] == {"method": "height_mass", "source_index": 0}
    assert prior["source"] == profile["sources"][0]
    assert prior["source"]["era"] and prior["source"]["values"]
    assert prior["value"] == pytest.approx(expected, abs=0.001)
    assert "not measured" in prior["assumption"]
    assert -1 <= prior["value"] <= 1


def test_moe_norman_weight_is_unknown():
    """Do not present the provisional mass as a measured source-backed frame."""
    profile = player_profile("moe_norman")
    assert "frame_prior" in profile and profile["frame_prior"] is None
    assert frame_prior(profile) == {"method": "unknown", "value": 0.0}
    assert "unmeasured" in profile["limitations"][0]


@pytest.mark.parametrize("player", ["john_schlee", "lee_trevino"])
def test_native_frame_prior_preserves_mass_com_and_realizable_inertia(player):
    """Apply two differing priors at fixed size and verify the native mass matrix."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance.player_models import build_player_spec
    from src.shared.python.model_appearance.player_stature import (
        build_stature_calibrated,
    )

    profile = player_profile(player)
    base, _ = build_stature_calibrated(profile["stature_m"], profile["mass_kg"])
    sized = build_player_spec(player)
    for original, changed in zip(base["bodies"], sized["bodies"], strict=True):
        for a, b in zip(original["solids"], changed["solids"], strict=True):
            assert a["mass_kg"] == b["mass_kg"]
            assert a["com_m"] == b["com_m"]
            tensor = np.asarray(b["inertia_com_kg_m2"])
            assert np.linalg.eigvalsh(tensor).min() > 0
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(json.dumps(s).encode())[0])
        for s in (base, sized)
    ]
    np.testing.assert_array_equal(models[0].body_mass, models[1].body_mass)
    states = []
    for model in models:
        state = mujoco.MjData(model)
        mujoco.mj_forward(model, state)
        matrix = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, matrix, state.qM)
        assert np.linalg.eigvalsh(matrix).min() > 0
        states.append(state)
    assert not np.array_equal(states[0].qM, states[1].qM)


def test_historical_reference_skin_retains_fitted_plant(tmp_path):
    """Export a sourced player on a base and compare its native dynamics exactly."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance.player_models import export_player
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    spec = reference_spec("driver")
    source = tmp_path / "base.json"
    source.write_text(json.dumps(spec))
    out = export_player("lee_trevino", tmp_path / "skin", base_spec=source)
    assert json.loads((out.parent / "model.json").read_text()) == spec
    receipt = json.loads((out.parent / "receipt.json").read_text())
    assert "frame_prior" not in receipt
    base = mujoco.MjModel.from_xml_string(
        export_full_body_mjcf(json.dumps(spec).encode())[0]
    )
    skin = mujoco.MjModel.from_xml_path(str(out))
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "jnt_type",
        "jnt_qposadr",
        "jnt_range",
    ):
        np.testing.assert_array_equal(getattr(base, field), getattr(skin, field))
    a, b = mujoco.MjData(base), mujoco.MjData(skin)
    for phase in (0.0, 0.03, -0.04):
        for model, state in ((base, a), (skin, b)):
            state.qpos[:] = model.qpos0 + phase
            state.qvel[:] = phase
            mujoco.mj_forward(model, state)
        for field in ("qM", "qfrc_bias", "qacc"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
