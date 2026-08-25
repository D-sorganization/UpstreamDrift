"""Tests for the MuJoCo swing source facade (issue #8975, EPIC #8965/WS2).

Covers:
* extraction contract — finite, correctly-shaped world-frame kinematics;
* the scripted forward-dynamics reference swing and its honest metadata;
* a golden-fixture replay so CI catches structural regressions (wrong body,
  wrong frame, wrong sign conventions) without asserting bitwise physics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.physics.mujoco_swing_source import (  # noqa: E402
    DEFAULT_CLUBHEAD_BODY,
    REFERENCE_TORQUE_SCALE,
    ClubheadKinematics,
    extract_clubhead_state,
    load_golf_swing_model,
    run_reference_swing,
    run_scripted_swing,
)

pytestmark = pytest.mark.unit

GOLDEN_FIXTURE = Path(__file__).parent / "fixtures" / "mujoco_swing_golden.json"


@pytest.fixture(scope="module")
def model():
    return load_golf_swing_model()


# ---------------------------------------------------------------------------
# Extraction contract
# ---------------------------------------------------------------------------


class TestExtractClubheadState:
    def test_shapes_and_finiteness(self, model):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        state = extract_clubhead_state(model, data)
        assert state.velocity.shape == (3,)
        assert state.angular_velocity.shape == (3,)
        assert state.orientation_quat.shape == (4,)
        assert state.face_normal.shape == (3,)
        assert state.inertia_diagonal.shape == (3,)
        for arr in (
            state.velocity,
            state.angular_velocity,
            state.orientation_quat,
            state.face_normal,
            state.inertia_diagonal,
        ):
            assert np.isfinite(arr).all()

    def test_face_normal_is_unit_vector(self, model):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        state = extract_clubhead_state(model, data)
        assert np.linalg.norm(state.face_normal) == pytest.approx(1.0, abs=1e-9)

    def test_mass_and_inertia_come_from_model(self, model):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        state = extract_clubhead_state(model, data)
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, DEFAULT_CLUBHEAD_BODY
        )
        assert state.mass == pytest.approx(float(model.body_mass[body_id]))
        assert state.mass > 0.0
        assert (state.inertia_diagonal > 0.0).all()

    def test_unknown_body_rejected(self, model):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with pytest.raises(ValueError, match="not found"):
            extract_clubhead_state(model, data, body_name="no_such_body")


# ---------------------------------------------------------------------------
# Scripted swing (full forward dynamics)
# ---------------------------------------------------------------------------


class TestRunScriptedSwing:
    def test_produces_moving_clubhead(self, model):
        state = run_scripted_swing(REFERENCE_TORQUE_SCALE, model=model)
        assert isinstance(state, ClubheadKinematics)
        assert state.speed > 1.0  # a swing, not numerical noise
        assert state.sim_time > 0.0

    def test_speed_grows_with_torque_scale(self, model):
        slow = run_scripted_swing(0.05, model=model)
        fast = run_scripted_swing(0.3, model=model)
        assert fast.speed > slow.speed

    @pytest.mark.parametrize("bad_scale", [0.0, -0.1, 1.5])
    def test_invalid_torque_scale_rejected(self, bad_scale, model):
        with pytest.raises(ValueError):
            run_scripted_swing(bad_scale, model=model)


# ---------------------------------------------------------------------------
# Reference swing (speed-calibrated) + honest metadata
# ---------------------------------------------------------------------------


class TestRunReferenceSwing:
    def test_calibrates_toward_target_speed(self):
        target = 45.0
        state, metadata = run_reference_swing(target)
        assert state.speed == pytest.approx(target, rel=0.10)
        assert metadata["achieved_speed_ms"] == pytest.approx(state.speed)
        assert metadata["target_speed_ms"] == target

    def test_metadata_is_honest_about_method(self):
        _, metadata = run_reference_swing(40.0)
        assert metadata["method"] == "mujoco_forward_dynamics"
        assert metadata["model_name"] == "upper_body_golf_swing"
        assert "model_asset" in metadata
        assert metadata["timestep_s"] > 0.0
        # Residual reported, never fabricated to zero-match the request.
        expected_residual = (
            metadata["achieved_speed_ms"] - metadata["target_speed_ms"]
        ) / metadata["target_speed_ms"]
        assert metadata["speed_residual_rel"] == pytest.approx(expected_residual)

    def test_invalid_target_rejected(self):
        with pytest.raises(ValueError):
            run_reference_swing(0.0)
        with pytest.raises(ValueError):
            run_reference_swing(float("nan"))


# ---------------------------------------------------------------------------
# Golden fixture — recorded reference extraction replayed with tolerances
# ---------------------------------------------------------------------------


class TestGoldenFixture:
    def test_reference_extraction_matches_recorded_fixture(self, model):
        """Replay the recorded scripted swing; loose tolerances absorb
        platform floating-point drift while catching structural bugs
        (wrong body, wrong frame, dropped mass/inertia)."""
        golden = json.loads(GOLDEN_FIXTURE.read_text())
        state = run_scripted_swing(golden["torque_scale"], model=model)

        assert state.speed == pytest.approx(golden["speed"], rel=0.10)
        np.testing.assert_allclose(
            state.velocity, golden["velocity"], rtol=0.15, atol=1.0
        )
        np.testing.assert_allclose(
            state.angular_velocity,
            golden["angular_velocity"],
            rtol=0.15,
            atol=2.0,
        )
        # Model constants must match exactly-ish: they are read from MJCF,
        # not integrated, so any drift means the wrong body was sampled.
        assert state.mass == pytest.approx(golden["mass"], rel=1e-6)
        np.testing.assert_allclose(
            state.inertia_diagonal, golden["inertia_diagonal"], rtol=1e-6
        )
        assert np.linalg.norm(state.face_normal) == pytest.approx(1.0, abs=1e-9)
