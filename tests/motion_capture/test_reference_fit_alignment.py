"""A fixed transform recovers placement without erasing motion differences."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.motion_capture.reference.fit_alignment import estimate_reference_transform

pytestmark = pytest.mark.unit


def test_recover_similarity_and_keep_later_motion_difference() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]], float)
    rotation = Rotation.from_euler("y", 0.6).as_matrix()
    target = 1.2 * source @ rotation.T + [3, 1, 2]
    result = estimate_reference_transform(source, target, scale=True)
    np.testing.assert_allclose(result.apply(source), target, atol=1e-12)
    assert result.scale == pytest.approx(1.2)
    assert result.body_size_normalized
    assert not result.is_calibrated
    later = source.copy()
    later[1, 0] += 0.1
    assert np.linalg.norm(result.apply(later)[1] - target[1]) == pytest.approx(0.12)


def test_alignment_masks_missing_pairs_but_rejects_degenerate_layout() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan] * 3])
    result = estimate_reference_transform(source, source + 1)
    np.testing.assert_allclose(result.translation_m, [1, 1, 1])
    with pytest.raises(ValueError, match="non-collinear"):
        estimate_reference_transform(np.zeros((4, 3)), np.ones((4, 3)))


def test_rigid_alignment_does_not_change_scale() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    result = estimate_reference_transform(source, 2 * source)
    assert result.scale == 1
