"""Sized-candidate torso inertia estimates from a bounded frame prior."""

import copy
import json

import numpy as np
import pytest

from src.shared.python.model_appearance.player_frame_inertia import apply_frame_inertia
from src.shared.python.model_appearance.player_stature import build_stature_calibrated

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240)]


@pytest.fixture(scope="module")
def sized():
    """Return one canonical sized candidate for controlled comparisons."""
    return build_stature_calibrated(1.8, 80.0)[0]


def torso(spec):
    """Select the three documented torso solids by their stable names."""
    return {
        s["name"].split("/")[-1]: s
        for b in spec["bodies"]
        for s in b["solids"]
        if s["name"].split("/")[-1] in {"pelvis", "middle_trunk", "upper_trunk"}
    }


def test_default_identity_validation_and_no_mutation(sized):
    """Keep default bytes exact and reject invalid inputs without mutation."""
    original = copy.deepcopy(sized)
    assert apply_frame_inertia(sized, 0) == sized
    for invalid in (-1.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            apply_frame_inertia(sized, invalid)
    assert sized == original
    bad = copy.deepcopy(sized)
    torso(bad)["pelvis"]["placement"][0][0] = -1
    with pytest.raises(ValueError, match="placement"):
        apply_frame_inertia(bad, 1)


def test_mass_com_and_realizable_inertia(sized):
    """Preserve mass/COM and second-moment positivity throughout the range."""
    from src.shared.python.motion_matching.full_body_spec import (
        canonical_sha256,
        upper_body_slice,
    )

    for build in (-1.0, 0.5, 1.0):
        changed = apply_frame_inertia(sized, build)
        assert changed["upper_body_sha256"] == canonical_sha256(
            upper_body_slice(changed)
        )
        for name, source in torso(sized).items():
            result = torso(changed)[name]
            assert source["mass_kg"] == result["mass_kg"]
            assert source["com_m"] == result["com_m"]
            a = np.asarray(result["inertia_com_kg_m2"])
            moment = np.trace(a) / 2 * np.eye(3) - a
            assert np.linalg.eigvalsh(moment).min() > 0
            assert np.linalg.eigvalsh(a).min() > 0
        assert changed["coordinate_order"] == sized["coordinate_order"]
        assert (
            torso(changed)["pelvis"]["inertia_com_kg_m2"]
            != torso(sized)["pelvis"]["inertia_com_kg_m2"]
        )


def test_native_mass_matrix_changes_at_fixed_mass(sized):
    """Compile both frame estimates and compare qM at the same position."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    models = [
        mujoco.MjModel.from_xml_string(
            export_full_body_mjcf(json.dumps(apply_frame_inertia(sized, b)).encode())[0]
        )
        for b in (0.0, 1.0)
    ]
    np.testing.assert_array_equal(models[0].body_mass, models[1].body_mass)
    states = []
    for model in models:
        data = mujoco.MjData(model)
        data.qpos[:] += 0.03
        mujoco.mj_forward(model, data)
        dense = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, dense, data.qM)
        assert np.linalg.eigvalsh(dense).min() > 0
        states.append(data)
    assert not np.array_equal(states[0].qM, states[1].qM)


def test_catalog_build_forwarding(monkeypatch):
    """Forward catalog build only when making a newly sized candidate."""
    from src.shared.python.model_appearance import player_models, player_stature

    profile = {"stature_m": 1.8, "mass_kg": 80.0, "appearance": {"body_build": 0.6}}
    monkeypatch.setattr(player_models, "player_profile", lambda _: profile)
    seen = {}

    def build(*args, **kwargs):
        """Capture the named frame input without running the calibration solver."""
        seen.update(kwargs)
        return {"subject": {}}, {}

    monkeypatch.setattr(player_stature, "build_stature_calibrated", build)
    player_models.build_player_spec("test")
    assert seen["frame_build"] == 0.6


def test_skin_only_export_never_calls_sized_builder(tmp_path, monkeypatch, sized):
    """Leave the supplied C3D plant unchanged when a profile has a build setting."""
    from src.engines.physics_engines.mujoco.python import full_body_mjcf
    from src.shared.python.model_appearance import player_models

    original_profile = player_models.player_profile("tiger_woods")
    original_profile["appearance"]["body_build"] = 0.6
    monkeypatch.setattr(player_models, "player_profile", lambda _: original_profile)
    monkeypatch.setattr(
        player_models,
        "player_appearance",
        lambda _: player_models.document_from_dict({"schema_version": "appearance-v1"}),
    )
    monkeypatch.setattr(
        player_models,
        "build_player_spec",
        lambda *a, **kw: pytest.fail("unexpected sized build"),
    )
    monkeypatch.setattr(
        full_body_mjcf,
        "export_full_body_mjcf",
        lambda raw, appearance=None: ("<mujoco/>", {}),
    )
    source = tmp_path / "base.json"
    source.write_text(json.dumps(sized))
    result = player_models.export_player(
        "tiger_woods", tmp_path / "out", base_spec=source
    )
    assert json.loads((result.parent / "model.json").read_text()) == sized


def test_calibration_applies_estimate_to_sized_candidate(sized):
    """Calibrate equal-size candidates and verify the requested prior is recorded."""
    actual, receipt = build_stature_calibrated(1.8, 80.0, frame_build=0.6)
    expected = apply_frame_inertia(sized, 0.6)
    for name, solid in torso(expected).items():
        np.testing.assert_allclose(
            torso(actual)[name]["inertia_com_kg_m2"],
            solid["inertia_com_kg_m2"],
            rtol=0,
            atol=1e-13,
        )
    assert receipt["frame_inertia"]["frame_build"] == 0.6
    assert receipt["frame_inertia"]["transverse_scale"] == pytest.approx(1.108)
