"""Frame-sensitive limb inertias preserve anatomy and native realizability."""

import copy
import json

import numpy as np
import pytest

from src.shared.python.model_appearance.player_frame_inertia import (
    LIMB_SOLIDS,
    apply_frame_inertia,
)
from src.shared.python.model_appearance.player_stature import build_stature_calibrated

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240)]


@pytest.fixture(scope="module")
def sized():
    """Build a canonical sized candidate for controlled limb comparisons."""
    return build_stature_calibrated(1.8, 80.0)[0]


def limb_map(spec):
    """Find the ten expected limb solids by their explicit body identities."""
    bodies = {body["name"]: body for body in spec["bodies"]}
    return {name: bodies[name]["solids"][0] for name in LIMB_SOLIDS}


def test_limb_frame_prior_preserves_mass_com_and_realizability(sized):
    """A bounded frame input changes all limb inertias without mutating the base."""
    from src.shared.python.motion_matching.anthropometric_candidate import (
        _segment_axis_in_body,
    )

    original = copy.deepcopy(sized)
    for value in (-1.0, 0.5, 1.0):
        changed = apply_frame_inertia(sized, value)
        for name, solid in limb_map(sized).items():
            after = limb_map(changed)[name]
            assert solid["mass_kg"] == after["mass_kg"]
            assert solid["com_m"] == after["com_m"]
            assert solid["inertia_com_kg_m2"] != after["inertia_com_kg_m2"]
            inertia = np.asarray(after["inertia_com_kg_m2"])
            second = 0.5 * np.trace(inertia) * np.eye(3) - inertia
            old = np.asarray(solid["inertia_com_kg_m2"])
            old_second = 0.5 * np.trace(old) * np.eye(3) - old
            axis, _, _ = _segment_axis_in_body(sized, name)
            rotation = np.asarray(solid["placement"])[:3, :3]
            direction = rotation.T @ axis
            assert direction @ second @ direction == pytest.approx(
                direction @ old_second @ direction, abs=1e-12
            )
            assert np.linalg.eigvalsh(second).min() > 0
            assert np.linalg.eigvalsh(inertia).min() > 0
    assert sized == original
    assert apply_frame_inertia(sized, 0) == sized


def test_rejects_ambiguous_joints_and_bad_frames(sized):
    """Fail closed on ambiguous endpoint mappings and malformed solid poses."""
    body = next(iter(LIMB_SOLIDS))
    bad = copy.deepcopy(sized)
    outgoing = next(j for j in bad["joints"] if j["parent"] == body)
    bad["joints"].append(copy.deepcopy(outgoing))
    with pytest.raises(ValueError, match="joint"):
        apply_frame_inertia(bad, 0.5)
    bad = copy.deepcopy(sized)
    incoming = next(j for j in bad["joints"] if j["child"] == body)
    incoming["parent"] = incoming["parent"].replace("HubtoLS", "HubtoRS")
    with pytest.raises(ValueError, match="joint"):
        apply_frame_inertia(bad, 0.5)
    bad = copy.deepcopy(sized)
    limb_map(bad)[body]["placement"][0][0] = 2
    with pytest.raises(ValueError, match="rigid"):
        apply_frame_inertia(bad, 0.5)
    bad = copy.deepcopy(sized)
    limb_map(bad)[body]["inertia_com_kg_m2"] = np.diag([10, 1, 1]).tolist()
    with pytest.raises(ValueError, match="unrealizable"):
        apply_frame_inertia(bad, 0.5)


def test_rotated_solid_frame_has_same_physical_estimate(sized):
    """Rotate a solid's coordinates and verify the transformed result rotates back."""
    body = next(iter(LIMB_SOLIDS))
    before = limb_map(sized)[body]
    rotated = copy.deepcopy(sized)
    solid = limb_map(rotated)[body]
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    solid["placement"] = np.block(
        [[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]
    ).tolist()
    solid["com_m"] = (R.T @ np.asarray(before["com_m"])).tolist()
    solid["inertia_com_kg_m2"] = (
        R.T @ np.asarray(before["inertia_com_kg_m2"]) @ R
    ).tolist()
    direct = np.asarray(
        limb_map(apply_frame_inertia(sized, 0.7))[body]["inertia_com_kg_m2"]
    )
    rotated_result = np.asarray(
        limb_map(apply_frame_inertia(rotated, 0.7))[body]["inertia_com_kg_m2"]
    )
    np.testing.assert_allclose(R @ rotated_result @ R.T, direct, rtol=0, atol=1e-12)


def test_native_limb_sensitivity_with_fixed_masses(sized):
    """Confirm positive native mass matrices and changed qM at representative poses."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.full_body_spec import (
        canonical_sha256,
        upper_body_slice,
    )

    limbs_changed = apply_frame_inertia(sized, 1.0)
    torso_only = copy.deepcopy(limbs_changed)
    for name, source in limb_map(sized).items():
        limb_map(torso_only)[name]["inertia_com_kg_m2"] = source["inertia_com_kg_m2"]
    torso_only["upper_body_sha256"] = canonical_sha256(upper_body_slice(torso_only))
    models = [
        mujoco.MjModel.from_xml_string(
            export_full_body_mjcf(json.dumps(spec).encode())[0]
        )
        for spec in (torso_only, limbs_changed)
    ]
    np.testing.assert_array_equal(models[0].body_mass, models[1].body_mass)
    for phase in (0.0, 0.03, -0.04):
        data = []
        for model in models:
            state = mujoco.MjData(model)
            state.qpos[:] += phase
            mujoco.mj_forward(model, state)
            dense = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, dense, state.qM)
            assert np.linalg.eigvalsh(dense).min() > 0
            data.append(state)
        assert not np.array_equal(data[0].qM, data[1].qM)


def test_sized_builder_applies_limb_prior(sized):
    """Verify the catalog-facing builder applies each estimated limb inertia."""
    actual, receipt = build_stature_calibrated(1.8, 80.0, frame_build=0.5)
    expected = apply_frame_inertia(sized, 0.5)
    for name, solid in limb_map(expected).items():
        np.testing.assert_allclose(
            limb_map(actual)[name]["inertia_com_kg_m2"],
            solid["inertia_com_kg_m2"],
            rtol=0,
            atol=1e-12,
        )
    assert "limb" in receipt["frame_inertia"]["method"]
