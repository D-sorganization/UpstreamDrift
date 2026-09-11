"""Rigid attachment residuals must relax dynamics without hiding deformation."""

import numpy as np
import pytest

from src.shared.python.motion_matching.rigidity import rigid_attachment_residuals

pytestmark = pytest.mark.unit


def test_independent_rigid_body_motion_has_zero_floor() -> None:
    offsets = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 3.0, 4.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = offsets @ rotation.T + [3.0, -2.0, 1.0]
    target[-1] = [-10.0, 20.0, 30.0]
    errors = rigid_attachment_residuals(offsets, target[None], ["a", "a", "a", "b"])
    np.testing.assert_allclose(errors, 0.0, atol=1e-12)


def test_stretched_pair_has_known_nonzero_floor() -> None:
    offsets = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.array([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    np.testing.assert_allclose(
        rigid_attachment_residuals(offsets, target, ["a", "a"]), [[1.0, 1.0]]
    )


def test_reflection_is_not_accepted_as_rigid_rotation() -> None:
    offsets = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    target = offsets.copy()
    target[:, 0] *= -1
    assert (
        np.linalg.norm(rigid_attachment_residuals(offsets, target[None], ["a"] * 4))
        > 0.1
    )


def test_missing_markers_remain_missing_without_biasing_visible_points() -> None:
    offsets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = np.array([[[np.nan] * 3, [4.0, 5.0, 6.0]], [[np.nan] * 3, [np.nan] * 3]])
    result = rigid_attachment_residuals(offsets, target, ["a", "a"])
    assert result[0, 1] == 0
    assert np.isnan(result[0, 0]) and np.isnan(result[1]).all()


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_nonfinite_observed_coordinates_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        rigid_attachment_residuals(
            np.zeros((1, 3)), np.array([[[bad, 0.0, 0.0]]]), ["a"]
        )


def test_partial_point_and_invalid_body_contracts_are_rejected() -> None:
    with pytest.raises(ValueError):
        rigid_attachment_residuals(
            np.zeros((1, 3)), np.array([[[np.nan, 0.0, 0.0]]]), ["a"]
        )
    with pytest.raises(ValueError):
        rigid_attachment_residuals(np.zeros((1, 3)), np.zeros((1, 1, 3)), [])
    with pytest.raises(ValueError):
        rigid_attachment_residuals(np.zeros((1, 3)), np.zeros((1, 1, 3)), [""])
