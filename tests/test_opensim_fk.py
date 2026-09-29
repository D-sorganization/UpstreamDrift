"""Tests for OpenSim forward-kinematics extraction (issue #4116).

The pure-NumPy quaternion conversion is exercised without the OpenSim
wheel installed. The end-to-end FK tests load the shipped
``golf_humanoid.osim`` and are gated behind ``@pytest.mark.requires_opensim``
so default CI passes on hosts without the OpenSim bindings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.motion_matching.forward_kinematics import (
    CANONICAL_LANDMARKS,
    _rotation_matrix_to_quat_wxyz,
    extract_clubhead_pose,
    extract_full_pose,
    extract_grip_pose,
)

# ---------------------------------------------------------------------------
# Pure-NumPy unit tests (run on every CI; no OpenSim required).
# ---------------------------------------------------------------------------


class TestRotationToQuat:
    """Unit tests for ``_rotation_matrix_to_quat_wxyz``."""

    def test_identity_returns_unit_w(self) -> None:
        q = _rotation_matrix_to_quat_wxyz(np.eye(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_180_about_x_returns_x_unit(self) -> None:
        # Rotation 180 deg about X: diag(1, -1, -1)
        rot = np.diag([1.0, -1.0, -1.0])
        q = _rotation_matrix_to_quat_wxyz(rot)
        # 180 about x -> q = (0, 1, 0, 0); sign-canonicalised w >= 0.
        # When w == 0 either sign is valid; just assert magnitude.
        assert abs(q[0]) < 1e-12
        np.testing.assert_allclose(np.abs(q[1:]), [1.0, 0.0, 0.0], atol=1e-12)

    def test_90_about_z_correct(self) -> None:
        # 90 deg about Z: cos=0, sin=1
        rot = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        q = _rotation_matrix_to_quat_wxyz(rot)
        expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        np.testing.assert_allclose(q, expected, atol=1e-12)

    def test_unit_norm(self) -> None:
        # Random small rotation
        rng = np.random.default_rng(42)
        for _ in range(10):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(-np.pi, np.pi)
            # Rodrigues
            k = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            rot = np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
            q = _rotation_matrix_to_quat_wxyz(rot)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-12
            assert q[0] >= 0.0  # w sign-canonicalised

    def test_bad_shape_raises(self) -> None:
        # #6938: now delegates to the canonical opensim_golf.fk helper,
        # whose message names the parameter ``rot_matrix``.
        with pytest.raises(ValueError, match="rot_matrix must be"):
            _rotation_matrix_to_quat_wxyz(np.eye(4))


class TestExtractWithMockModel:
    """Verify extraction logic with a mocked OpenSim model.

    These tests do not need the OpenSim wheel because we replace the
    SWIG calls with ``MagicMock`` instances that return scripted values.
    They confirm the canonical-frame contract (shape, dtype, ordering).
    """

    @staticmethod
    def _make_mock_model(
        translations: dict[str, tuple[float, float, float]],
    ) -> MagicMock:
        """Build a model whose components return scripted positions.

        Each frame returns identity rotation and the supplied translation,
        so we can sanity-check that pos/quat plumbing maps frames -> output
        keys correctly.
        """
        model = MagicMock()

        def get_component(path: str) -> MagicMock:
            xyz = translations[path]
            transform = MagicMock()

            p_vec = MagicMock()
            p_vec.get.side_effect = lambda i, _xyz=xyz: _xyz[i]
            transform.p.return_value = p_vec

            rot = MagicMock()
            rot.get.side_effect = lambda i, j: 1.0 if i == j else 0.0
            transform.R.return_value = rot

            frame = MagicMock()
            frame.getTransformInGround.return_value = transform
            return frame

        model.getComponent.side_effect = get_component
        return model

    def test_extract_grip_pose_returns_canonical_shapes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bypass the opensim import probe.
        monkeypatch.setattr(
            "src.engines.physics_engines.opensim.python.opensim_golf."
            "fk._require_opensim",
            lambda: None,
        )
        translations = {
            CANONICAL_LANDMARKS["grip"]: (0.10, 1.20, -0.05),
            CANONICAL_LANDMARKS["clubhead"]: (0.30, 0.10, -0.04),
        }
        model = self._make_mock_model(translations)
        state = MagicMock()

        pos, quat = extract_grip_pose(state, model)

        assert pos.shape == (3,)
        assert quat.shape == (4,)
        assert pos.dtype == np.float64
        assert quat.dtype == np.float64
        np.testing.assert_allclose(pos, [0.10, 1.20, -0.05], atol=1e-12)
        # Identity rotation -> quat = (1, 0, 0, 0).
        np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_extract_clubhead_pose_uses_correct_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.engines.physics_engines.opensim.python.opensim_golf."
            "fk._require_opensim",
            lambda: None,
        )
        translations = {
            CANONICAL_LANDMARKS["grip"]: (0.10, 1.20, -0.05),
            CANONICAL_LANDMARKS["clubhead"]: (0.30, 0.10, -0.04),
        }
        model = self._make_mock_model(translations)
        state = MagicMock()

        pos, quat = extract_clubhead_pose(state, model)
        np.testing.assert_allclose(pos, [0.30, 0.10, -0.04], atol=1e-12)
        np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_extract_full_pose_returns_all_landmarks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.engines.physics_engines.opensim.python.opensim_golf."
            "fk._require_opensim",
            lambda: None,
        )
        translations = {
            CANONICAL_LANDMARKS["grip"]: (0.10, 1.20, -0.05),
            CANONICAL_LANDMARKS["clubhead"]: (0.30, 0.10, -0.04),
        }
        model = self._make_mock_model(translations)
        state = MagicMock()

        full = extract_full_pose(state, model)
        # Each landmark must produce two keys: <name>_pos, <name>_quat.
        for landmark in CANONICAL_LANDMARKS:
            assert f"{landmark}_pos" in full
            assert f"{landmark}_quat" in full
            assert full[f"{landmark}_pos"].shape == (3,)
            assert full[f"{landmark}_quat"].shape == (4,)
        np.testing.assert_allclose(full["grip_pos"], [0.10, 1.20, -0.05], atol=1e-12)
        np.testing.assert_allclose(
            full["clubhead_pos"], [0.30, 0.10, -0.04], atol=1e-12
        )

    def test_none_inputs_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            extract_grip_pose(None, MagicMock())
        with pytest.raises(ValueError):
            extract_clubhead_pose(MagicMock(), None)
        with pytest.raises(ValueError):
            extract_full_pose(None, None)


# ---------------------------------------------------------------------------
# End-to-end tests against the shipped golf_humanoid.osim.
# Skipped on hosts without the OpenSim Python bindings.
# ---------------------------------------------------------------------------


_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid.osim"
)


@pytest.fixture(scope="module")
def opensim_module() -> object:
    """Import opensim or skip the module if unavailable."""
    pytest.importorskip("opensim", reason="OpenSim Python bindings not installed")
    import opensim  # type: ignore[import-untyped]

    return opensim


@pytest.fixture(scope="module")
def golf_humanoid(opensim_module: object) -> tuple[object, object]:
    """Load the MVP golf-humanoid model and a freshly-realised state."""
    if not _MODEL_PATH.exists():
        pytest.skip(f"Model file missing: {_MODEL_PATH}")
    model = opensim_module.Model(str(_MODEL_PATH))  # type: ignore[attr-defined]
    state = model.initSystem()
    return model, state


@pytest.mark.requires_opensim
class TestForwardKinematicsLive:
    """End-to-end FK extraction against the shipped .osim model."""

    def test_neutral_pose_grip_is_finite_and_plausible(
        self, golf_humanoid: tuple[object, object]
    ) -> None:
        """Neutral pose: grip is finite, non-zero, within a plausible range."""
        model, state = golf_humanoid
        pos, quat = extract_grip_pose(state, model)
        assert pos.shape == (3,) and quat.shape == (4,)
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(quat))
        # Non-zero: at least one component above 1 cm magnitude.
        assert np.linalg.norm(pos) > 1e-2
        # Plausibility: a ~1.8 m human's grip cannot be more than 3 m
        # from origin in any direction.
        assert np.all(np.abs(pos) < 3.0)
        # Unit quaternion.
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-9
        assert quat[0] >= 0.0

    def test_neutral_pose_clubhead_is_finite_and_plausible(
        self, golf_humanoid: tuple[object, object]
    ) -> None:
        """Neutral pose: clubhead is finite, non-zero, within plausible range."""
        model, state = golf_humanoid
        pos, quat = extract_clubhead_pose(state, model)
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(quat))
        assert np.linalg.norm(pos) > 1e-2
        assert np.all(np.abs(pos) < 3.5)
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-9

    def test_clubhead_below_grip_in_world(
        self, golf_humanoid: tuple[object, object]
    ) -> None:
        """In the neutral hanging pose the clubhead sits below the grip.

        OpenSim is Y-up; ``club_head_offset`` is at ``Club + (0, -1.14, 0)``
        so the clubhead Y-coord must be lower than the grip Y-coord.
        """
        model, state = golf_humanoid
        grip_pos, _ = extract_grip_pose(state, model)
        head_pos, _ = extract_clubhead_pose(state, model)
        assert head_pos[1] < grip_pos[1], (
            f"Expected clubhead Y < grip Y; got grip={grip_pos}, clubhead={head_pos}"
        )

    def test_extract_full_pose_contains_all_landmarks(
        self, golf_humanoid: tuple[object, object]
    ) -> None:
        model, state = golf_humanoid
        full = extract_full_pose(state, model)
        for landmark in CANONICAL_LANDMARKS:
            assert f"{landmark}_pos" in full
            assert f"{landmark}_quat" in full
            assert np.all(np.isfinite(full[f"{landmark}_pos"]))
            assert np.all(np.isfinite(full[f"{landmark}_quat"]))

    def test_round_trip_set_q_recover_grip(
        self,
        opensim_module: object,
        golf_humanoid: tuple[object, object],
    ) -> None:
        """Round-trip: set ``q`` via the model's coordinate handles,
        re-realise position, and confirm extraction returns a different
        (but still finite) grip pose than the neutral one.

        This is the FK side of the coord_map round-trip described in
        issue #4114; the actual coord_map module is tracked separately.
        """
        model, state = golf_humanoid
        neutral_grip, _ = extract_grip_pose(state, model)

        # Perturb every coordinate by a small angle (1 deg = 0.0174 rad).
        coord_set = model.getCoordinateSet()
        n_coords = coord_set.getSize()
        assert n_coords > 0, "Model must have at least one coordinate"

        for i in range(n_coords):
            coord = coord_set.get(i)
            try:
                value = coord.getValue(state)
                coord.setValue(state, value + 0.0174)
            except Exception:  # noqa: BLE001 — locked coords raise; skip them
                continue

        moved_grip, moved_quat = extract_grip_pose(state, model)
        assert np.all(np.isfinite(moved_grip))
        assert np.all(np.isfinite(moved_quat))
        # The grip should have moved at least 1 mm somewhere.
        delta = float(np.linalg.norm(moved_grip - neutral_grip))
        assert delta > 1e-3, (
            f"Grip did not move after perturbing every coordinate; "
            f"neutral={neutral_grip}, moved={moved_grip}"
        )
