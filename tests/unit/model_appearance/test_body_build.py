"""Appearance-only body build controls preserve the CC0 skin and plant."""

import dataclasses

import numpy as np
import pytest

from src.shared.python.model_appearance import body_skin_fit as fit
from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    document_from_dict,
    document_to_dict,
)
from tests.unit.model_appearance.test_body_skin_fit import _subject

pytestmark = pytest.mark.unit


def test_build_schema_roundtrip_and_bounds():
    """Validate finite bounded values and omit the historical default."""
    assert "body_build" not in document_to_dict(AppearanceDocument())
    for value in (-1.0, 0.0, 1.0):
        doc = AppearanceDocument(body_build=value)
        assert document_from_dict(document_to_dict(doc)).body_build == value
    for value in (-1.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            AppearanceDocument(body_build=value)


def test_body_build_warp_is_bounded_and_keeps_topology():
    """Check distinct waist widths, identity, fixed joints, and oriented faces."""
    skin = fit.load_body_skin()
    fitted = fit.fit_body_skin(skin, _subject(), fit.load_fit_settings())
    original = fitted.vertices.copy()
    np.testing.assert_array_equal(fit.shape_body_build(fitted, 0).vertices, original)
    shaped = [fit.shape_body_build(fitted, v) for v in (-1.0, 1.0)]
    assert (
        np.isfinite(shaped[0].vertices).all() and np.isfinite(shaped[1].vertices).all()
    )
    for body in shaped:
        for joint in fitted.joints:
            np.testing.assert_array_equal(body.joints[joint], fitted.joints[joint])
        p, q = original[skin.faces], body.vertices[skin.faces]
        a = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        b = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        assert np.all(np.einsum("ij,ij->i", a, b) > 0)
    waist = (abs(original[:, 2] - 0.2) < 0.02) & (abs(original[:, 1]) < 0.3)
    assert np.ptp(shaped[0].vertices[waist, 1]) < np.ptp(original[waist, 1])
    assert np.ptp(original[waist, 1]) < np.ptp(shaped[1].vertices[waist, 1])


@pytest.mark.timeout(300)
def test_native_build_changes_skin_but_not_physics():
    """Compare MuJoCo plant arrays and dynamics at three states for both extremes."""
    import json

    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    spec = reference_spec("driver")
    spec.setdefault("subject", {})["body_visual_backend"] = "skinned_cc0"
    raw = json.dumps(spec).encode()
    docs = [AppearanceDocument(body_build=value) for value in (0.0, -1.0, 1.0)]
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=doc)[0])
        for doc in docs
    ]
    base = models[0]
    for other in models[1:]:
        assert not np.array_equal(base.skin_vert, other.skin_vert)
        for name in (
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "jnt_type",
            "jnt_qposadr",
            "jnt_dofadr",
            "jnt_range",
            "qpos0",
        ):
            np.testing.assert_array_equal(getattr(base, name), getattr(other, name))
        for phase in (0.0, 0.03, -0.04):
            states = []
            for model in (base, other):
                data = mujoco.MjData(model)
                data.qpos[:] += phase
                data.qvel[:] = phase
                mujoco.mj_forward(model, data)
                states.append(data)
            for name in ("qM", "qfrc_bias", "qacc"):
                np.testing.assert_array_equal(
                    getattr(states[0], name), getattr(states[1], name)
                )
