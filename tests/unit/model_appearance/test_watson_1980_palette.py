"""Watson's 1980 appearance remains separate from physical frame priors."""

import copy
import json

import numpy as np
import pytest

from src.shared.python.model_appearance import player_models

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240)]


def test_watson_visual_palette_and_source():
    """Use a cited 1980 color reference and an explicit stylized lean build."""
    profile = player_models.player_profile("tom_watson")
    appearance = profile["appearance"]
    assert appearance["body_build"] == -0.25
    assert appearance["materials"]["polo_white"]["base_color"] == [
        0.25,
        0.43,
        0.65,
        1.0,
    ]
    assert appearance["materials"]["shoe_leather"]["base_color"] == [
        0.9,
        0.9,
        0.87,
        1.0,
    ]
    assert any(
        "360029" in source["url"] and "1980" in source["era"]
        for source in profile["sources"]
    )
    assert any(
        "sweater" in entry.lower() and "short" in entry.lower()
        for entry in profile["limitations"]
    )
    assert profile["frame_prior"] == {"method": "height_mass", "source_index": 0}


def test_visual_build_does_not_change_sized_frame_prior(monkeypatch):
    """Confirm the original sourced physical frame input survives a visual change."""
    from src.shared.python.model_appearance import player_stature

    profile = player_models.player_profile("tom_watson")
    expected = player_models.frame_prior(profile)["value"]
    seen = []
    monkeypatch.setattr(
        player_models, "player_profile", lambda _: copy.deepcopy(profile)
    )

    def fake_build(height, mass, club, *, frame_build):
        """Capture the physical argument without rerunning calibration."""
        seen.append(frame_build)
        return {"subject": {}}, {}

    monkeypatch.setattr(player_stature, "build_stature_calibrated", fake_build)
    player_models.build_player_spec("tom_watson")
    profile["appearance"]["body_build"] = 0
    player_models.build_player_spec("tom_watson")
    assert seen == [expected, expected]


def test_watson_skin_preserves_native_fitted_plant():
    """Compare physics arrays and dynamics with and without Watson's appearance."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    raw = json.dumps(reference_spec("driver")).encode()
    docs = (None, player_models.player_appearance("tom_watson"))
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=d)[0])
        for d in docs
    ]
    for name in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_pos",
        "jnt_qposadr",
        "jnt_range",
    ):
        np.testing.assert_array_equal(
            getattr(models[0], name), getattr(models[1], name)
        )
    for phase in (0.0, 0.03, -0.04):
        states = []
        for model in models:
            state = mujoco.MjData(model)
            state.qpos[:] = model.qpos0 + phase
            state.qvel[:] = phase
            mujoco.mj_forward(model, state)
            states.append(state)
        for name in ("qM", "qfrc_bias", "qacc"):
            np.testing.assert_array_equal(
                getattr(states[0], name), getattr(states[1], name)
            )
