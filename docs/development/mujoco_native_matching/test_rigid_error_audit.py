"""Counterfactual alignment must retain observation masks and articulation error."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from audit_rigid_marker_errors import decompose

POINTS = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
)


def test_translation_removes_only_common_offset() -> None:
    offset = np.array([0.01, -0.02, 0.03])
    result = decompose(POINTS, POINTS + offset, np.ones(4, dtype=bool))
    np.testing.assert_allclose(result["translation_correction_m"], offset, atol=1e-15)
    assert result["translation_rms_m"] < 1e-14
    assert result["original_rms_m"] == pytest.approx(np.linalg.norm(offset))


def test_rotation_convention_and_nonrigid_residual() -> None:
    rotation = Rotation.from_euler("z", 0.2).as_matrix()
    result = decompose(
        POINTS, POINTS @ rotation.T + [0.1, 0.2, 0.3], np.ones(4, dtype=bool)
    )
    np.testing.assert_allclose(
        result["rotation_prediction_to_target"], rotation, atol=1e-14
    )
    assert result["rigid_rms_m"] < 1e-14
    scaled = decompose(POINTS, 2 * POINTS, np.ones(4, dtype=bool))
    assert scaled["rigid_rms_m"] == pytest.approx(np.sqrt(3))


def test_mask_ignores_missing_target_and_rejects_insufficient_geometry() -> None:
    target = POINTS.copy()
    target[0] = np.nan
    result = decompose(POINTS, target, np.array([False, True, True, True]))
    assert result["observed_count"] == 3
    assert result["original_rms_m"] == 0
    with pytest.raises(ValueError):
        decompose(POINTS, target, np.array([False, False, True, True]))
