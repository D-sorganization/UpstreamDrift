"""Articulated forward kinematics engine (issue #9711): hand-built chain tests.

Every test builds a small chain whose geometry has a closed-form answer, so a
regression cannot hide behind a re-derivation of the same maths: the planar
arm against cos/sin, the spherical joint against an explicitly composed Euler
product, and the Jacobian against central finite differences on a tree with
mixed 1-, 2- and 3-DOF joints. DbC contracts are exercised by feeding specs
and query arrays that must be rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.model.kinematics import (
    DOF,
    FKResult,
    Joint,
    Landmark,
    ModelSpec,
    Segment,
    forward_kinematics,
    jacobian,
)
from src.shared.python.core.contracts import PreconditionError

pytestmark = pytest.mark.unit

Z_AXIS = (0.0, 0.0, 1.0)
X_AXIS = (1.0, 0.0, 0.0)


def _planar_arm() -> ModelSpec:
    """Root hinge about z, elbow hinge about z, one landmark at the tip."""
    return ModelSpec.create(
        segments={
            "upper": Segment(
                name="upper",
                parent=None,
                joint=Joint("shoulder", (DOF(Z_AXIS, -np.pi, np.pi),)),
                offset=(0.0, 0.0, 0.0),
            ),
            "fore": Segment(
                name="fore",
                parent="upper",
                joint=Joint("elbow", (DOF(Z_AXIS, -np.pi, np.pi),)),
                offset=(1.0, 0.0, 0.0),
            ),
        },
        landmarks={"tip": Landmark("tip", "fore", (1.0, 0.0, 0.0))},
    )


def _mixed_tree() -> ModelSpec:
    """A 1-DOF root, a 2-DOF middle and a 3-DOF leaf, two landmarks."""
    return ModelSpec.create(
        segments={
            "root": Segment(
                name="root",
                parent=None,
                joint=Joint("root_y", (DOF(Z_AXIS, -1.0, 1.0),)),
                offset=(0.0, 0.0, 0.0),
            ),
            "mid": Segment(
                name="mid",
                parent="root",
                joint=Joint(
                    "universal",
                    (
                        DOF(X_AXIS, -0.5, 0.5),
                        DOF(Z_AXIS, -0.5, 0.5),
                    ),
                ),
                offset=(0.3, 0.1, 0.0),
            ),
            "leaf": Segment(
                name="leaf",
                parent="mid",
                joint=Joint(
                    "spherical",
                    (
                        DOF(X_AXIS, -0.3, 0.3),
                        DOF(Z_AXIS, -0.3, 0.3),
                        DOF((0.0, 1.0, 0.0), -0.3, 0.3),
                    ),
                ),
                offset=(0.2, 0.0, 0.1),
            ),
        },
        landmarks={
            "mid_point": Landmark("mid_point", "mid", (0.1, 0.2, 0.3)),
            "leaf_point": Landmark("leaf_point", "leaf", (0.4, 0.0, 0.0)),
        },
    )


def test_planar_fk_matches_analytic_two_link_pose() -> None:
    spec = _planar_arm()
    theta1, theta2 = 0.4, -0.7
    result = forward_kinematics(spec, np.array([[theta1, theta2]]))
    expected = np.array(
        [
            np.cos(theta1) + np.cos(theta1 + theta2),
            np.sin(theta1) + np.sin(theta1 + theta2),
            0.0,
        ]
    )
    assert isinstance(result, FKResult)
    np.testing.assert_allclose(result.landmarks[0, 0], expected, atol=1e-12)


def test_fk_is_vectorised_over_frames() -> None:
    spec = _mixed_tree()
    q = np.linspace(-0.9, 0.9, 7 * spec.n_dof).reshape(7, spec.n_dof)
    batched = forward_kinematics(spec, q)
    for t in range(7):
        single = forward_kinematics(spec, q[t : t + 1])
        np.testing.assert_allclose(batched.landmarks[t], single.landmarks[0])


def test_spherical_joint_composes_euler_rotation_in_declared_order() -> None:
    spec = _mixed_tree()
    a, b, c = 0.21, -0.34, 0.15
    q = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, a, b, c]])  # root and mid at home
    result = forward_kinematics(spec, q)

    def rot(axis: np.ndarray, angle: float) -> np.ndarray:
        k = np.asarray(axis, dtype=float)
        cross = np.array(
            [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
        )
        return (
            np.eye(3)
            + np.sin(angle) * cross
            + (1.0 - np.cos(angle)) * (cross @ cross)
        )

    composed = (
        rot(np.array(X_AXIS), a) @ rot(np.array(Z_AXIS), b) @ rot(np.array([0.0, 1.0, 0.0]), c)
    )
    offset = np.array([0.4, 0.0, 0.0])
    leaf_origin = np.array([0.3, 0.1, 0.0]) + np.array([0.2, 0.0, 0.1])
    expected = leaf_origin + composed @ offset
    np.testing.assert_allclose(result.landmarks[0, 1], expected, atol=1e-12)


def test_jacobian_matches_central_finite_differences() -> None:
    spec = _mixed_tree()
    rng = np.random.default_rng(7)
    q = rng.uniform(-0.25, 0.25, size=(5, spec.n_dof))
    jac = jacobian(spec, q)
    assert jac.shape == (5, 2 * 3, spec.n_dof)
    eps = 1e-6
    for t in range(5):
        for d in range(spec.n_dof):
            qp, qm = q[t].copy(), q[t].copy()
            qp[d] += eps
            qm[d] -= eps
            plus = forward_kinematics(
                spec, qp[None, :], enforce_limits=False
            ).landmarks[0]
            minus = forward_kinematics(
                spec, qm[None, :], enforce_limits=False
            ).landmarks[0]
            fd = ((plus - minus) / (2 * eps)).reshape(-1)
            np.testing.assert_allclose(jac[t, :, d], fd, atol=1e-8, rtol=1e-6)


def test_dof_downstream_of_landmark_has_zero_jacobian_column() -> None:
    spec = ModelSpec.create(
        segments={
            "arm": Segment(
                name="arm",
                parent=None,
                joint=Joint("shoulder", (DOF(Z_AXIS, -np.pi, np.pi),)),
                offset=(0.0, 0.0, 0.0),
            ),
            "hand": Segment(
                name="hand",
                parent="arm",
                joint=Joint("wrist", (DOF(X_AXIS, -1.0, 1.0),)),
                offset=(0.5, 0.0, 0.0),
            ),
        },
        landmarks={"shoulder_mark": Landmark("shoulder_mark", "arm", (0.05, 0.0, 0.0))},
    )
    q = np.array([[0.3, 0.4]])
    jac = jacobian(spec, q)
    np.testing.assert_allclose(jac[0, :, 1], 0.0, atol=1e-15)
    assert np.abs(jac[0, :, 0]).max() > 1e-6


def test_fk_rejects_out_of_limit_angles_and_can_relax() -> None:
    spec = _planar_arm()
    bad = np.array([[np.pi + 0.1, 0.0]])
    with pytest.raises(PreconditionError):
        forward_kinematics(spec, bad)
    relaxed = forward_kinematics(spec, bad, enforce_limits=False)
    assert relaxed.landmarks.shape == (1, 1, 3)


def test_fk_rejects_malformed_query_arrays() -> None:
    spec = _planar_arm()
    with pytest.raises(PreconditionError):
        forward_kinematics(spec, np.zeros((2, spec.n_dof + 1)))
    with pytest.raises(PreconditionError):
        forward_kinematics(spec, np.zeros(spec.n_dof))
    with pytest.raises(PreconditionError):
        forward_kinematics(spec, np.array([[np.nan, 0.0]]))


@pytest.mark.parametrize(
    ("segments", "landmarks", "message"),
    [
        pytest.param({}, {"m": Landmark("m", "a", (1, 0, 0))}, "empty", id="empty"),
        pytest.param(
            {
                "a": Segment("a", "ghost", Joint("j", (DOF(Z_AXIS, 0, 1),)), (0, 0, 0)),
            },
            {},
            "parent",
            id="unknown-parent",
        ),
        pytest.param(
            {
                "a": Segment("a", "b", Joint("j", (DOF(Z_AXIS, 0, 1),)), (0, 0, 0)),
                "b": Segment("b", "a", Joint("k", (DOF(X_AXIS, 0, 1),)), (0, 0, 0)),
            },
            {},
            "cycle",
            id="cycle",
        ),
        pytest.param(
            {
                "a": Segment(
                    "a", None, Joint("j", (DOF((0, 0, 2), -1, 1),)), (0, 0, 0)
                ),
            },
            {},
            "axis",
            id="non-unit-axis",
        ),
        pytest.param(
            {"a": Segment("a", None, Joint("j", (DOF(Z_AXIS, 1, -1),)), (0, 0, 0))},
            {},
            "limits",
            id="inverted-limits",
        ),
        pytest.param(
            {
                "a": Segment("a", None, Joint("j", (DOF(Z_AXIS, 0, 1),)), (0, 0, 0)),
            },
            {"m": Landmark("m", "ghost", (1, 0, 0))},
            "landmark",
            id="unknown-landmark-segment",
        ),
    ],
)
def test_spec_validation_rejects_broken_models(
    segments: dict, landmarks: dict, message: str
) -> None:
    with pytest.raises(PreconditionError, match=".*"):
        ModelSpec.create(segments, landmarks)


def test_joint_without_dofs_is_rejected() -> None:
    with pytest.raises(PreconditionError):
        ModelSpec.create(
            segments={"a": Segment("a", None, Joint("j", ()), (0, 0, 0))},
            landmarks={},
        )


def test_fk_reports_landmark_and_segment_series_with_shapes() -> None:
    spec = _mixed_tree()
    q = np.zeros((3, spec.n_dof))
    result = forward_kinematics(spec, q)
    assert result.landmarks.shape == (3, 2, 3)
    assert result.positions.shape == (3, 3, 3)
    assert result.orientations.shape == (3, 3, 3, 3)
    np.testing.assert_allclose(result.orientations[:, 0], np.eye(3)[None, :, :])