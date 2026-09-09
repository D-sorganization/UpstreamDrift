"""DLT triangulation: exact on clean data, honest about rejected views."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct import PinholeCamera, look_at
from src.motion_capture.reconstruct.cameras import intrinsics_from_fov
from src.motion_capture.reconstruct.geometry import (
    projection_matrix,
    reprojection_residuals,
    triangulate,
)

pytestmark = pytest.mark.unit


def _rig() -> list[PinholeCamera]:
    k = intrinsics_from_fov(1920, 1200, 70.0)
    target = np.array([0.0, 1.0, 0.0])
    spots = {
        "a": np.array([0.0, 1.2, 4.0]),
        "b": np.array([-4.0, 1.2, 0.0]),
        "c": np.array([2.5, 3.0, 3.0]),
    }
    return [
        PinholeCamera(n, k, look_at(p, target), p, (1920, 1200))
        for n, p in spots.items()
    ]


def _pixels(cams, point):
    return np.vstack([cam.project(point[None, :])[0][0] for cam in cams])


def test_projection_matrix_matches_camera_projection() -> None:
    cam = _rig()[0]
    x = np.array([0.3, 1.1, -0.2])
    hom = projection_matrix(cam) @ np.append(x, 1.0)
    assert np.allclose(hom[:2] / hom[2], cam.project(x[None, :])[0][0])


def test_triangulation_is_exact_without_noise() -> None:
    cams = _rig()
    truth = np.array([0.25, 0.9, -0.1])
    result = triangulate(cams, _pixels(cams, truth))
    assert (
        result.ok and result.contributing == ("a", "b", "c") and result.rejected == ()
    )
    assert np.allclose(result.point_m, truth, atol=1e-6)
    assert result.rms_px is not None and result.rms_px < 1e-4
    assert result.covariance_m2 is not None and result.covariance_m2.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(result.covariance_m2) > 0)


def test_noise_gives_millimetre_error_and_covariance_of_the_right_size() -> None:
    cams = _rig()
    truth = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(50):
        px = _pixels(cams, truth) + rng.normal(0, 1.0, (3, 2))
        result = triangulate(cams, px, sigma_px=1.0)
        errors.append(np.linalg.norm(result.point_m - truth))
    assert np.median(errors) < 0.01  # 1 px at 4 m with a 70 deg lens is ~3 mm
    predicted = np.sqrt(np.trace(result.covariance_m2))
    assert 0.3 * predicted < np.median(errors) < 3.0 * predicted


def test_gross_view_is_rejected_and_named() -> None:
    cams = _rig()
    truth = np.array([0.1, 1.2, 0.05])
    px = _pixels(cams, truth)
    px[2] += [150.0, -90.0]  # camera c is wrong
    result = triangulate(cams, px, gate_px=8.0)
    assert result.rejected == ("c",) and result.contributing == ("a", "b")
    assert np.allclose(result.point_m, truth, atol=1e-6)
    assert result.residuals_px["c"] > 50.0


def test_two_views_cannot_reject_and_zero_confidence_is_unused() -> None:
    cams = _rig()
    truth = np.array([0.0, 1.0, 0.0])
    px = _pixels(cams, truth)
    result = triangulate(cams, px, confidence=np.array([1.0, 1.0, 0.0]))
    assert result.contributing == ("a", "b") and result.rejected == ("c",)
    # With two views an error along the epipolar line is absorbed by depth:
    # the point moves, the residuals stay small, and nothing can be blamed.
    # That is why the acceptance program demands a third useful view.
    px[1] += [200.0, 0.0]
    fooled = triangulate(cams, px, confidence=np.array([1.0, 1.0, 0.0]), gate_px=8.0)
    assert fooled.ok and fooled.rejected == ("c",)
    assert np.linalg.norm(fooled.point_m - truth) > 0.1
    # The third view exposes it.
    exposed = triangulate(cams, px, gate_px=8.0)
    assert exposed.rejected == ("b",) and np.allclose(exposed.point_m, truth, atol=1e-6)


def test_contracts() -> None:
    cams = _rig()
    with pytest.raises(Exception, match="two views"):
        triangulate(cams[:1], np.zeros((1, 2)))
    with pytest.raises(Exception, match="align"):
        triangulate(cams, np.zeros((2, 2)))
    with pytest.raises(Exception, match="gate_px"):
        triangulate(cams, np.zeros((3, 2)), gate_px=0)
    res = reprojection_residuals(cams, np.array([0.0, 1.0, 10.0]), np.zeros((3, 2)))
    assert np.isnan(res[0])  # behind camera a, which sits at z = 4 looking at -z
