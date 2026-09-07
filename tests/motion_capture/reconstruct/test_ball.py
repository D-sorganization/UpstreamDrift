"""Ball detection: found with a score, or not found with a reason."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.ball import (
    BallDetectorOptions,
    ball_at_rest,
    detect_ball,
)

pytest.importorskip("cv2")
pytestmark = pytest.mark.unit


def _scene(ball=(300, 400), radius=12, distractor=True, noise=8) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(0)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 120, 40)  # green turf (BGR)
    img[:200] = (60, 60, 60)  # dark wall
    if distractor:
        cv2.rectangle(img, (500, 380), (560, 420), (240, 240, 240), -1)  # white box
        cv2.circle(img, (120, 420), 12, (30, 30, 220), -1)  # red ball-sized blob
    if ball is not None:
        cv2.circle(img, ball, radius, (245, 245, 245), -1)
    noisy = img.astype(int) + rng.integers(-noise, noise + 1, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def test_ball_is_found_and_distractors_ranked_out() -> None:
    det = detect_ball(_scene())
    assert det.found and det.best is not None
    assert det.best.center_px == pytest.approx((300.0, 400.0), abs=1.5)
    assert 10.0 <= det.best.radius_px <= 14.0
    assert det.best.circularity > 0.8
    # the white box fails circularity and the red blob fails saturation
    assert all(c.radius_px <= 14.5 for c in det.candidates)


def test_no_ball_reports_a_reason_not_a_guess() -> None:
    det = detect_ball(_scene(ball=None))
    assert not det.found and det.best is None
    assert det.reason is not None and "no bright" in det.reason


def test_hint_steers_between_two_plausible_balls() -> None:
    import cv2

    img = _scene(distractor=False)
    cv2.circle(img, (500, 300), 12, (245, 245, 245), -1)  # a second white ball
    far = detect_ball(img, BallDetectorOptions(hint_px=(500.0, 300.0)))
    near = detect_ball(img, BallDetectorOptions(hint_px=(300.0, 400.0)))
    assert far.best is not None and far.best.center_px[0] == pytest.approx(
        500.0, abs=1.5
    )
    assert near.best is not None and near.best.center_px[0] == pytest.approx(
        300.0, abs=1.5
    )
    assert len(far.candidates) == 2


def test_radius_gate_and_contracts() -> None:
    det = detect_ball(_scene(radius=60))
    assert not det.found
    with pytest.raises(Exception, match="radius range"):
        BallDetectorOptions(min_radius_px=10, max_radius_px=5)
    with pytest.raises(Exception, match="uint8"):
        detect_ball(np.zeros((4, 4, 3), dtype=float))


def test_ball_at_rest_needs_a_stable_run() -> None:
    steady = [detect_ball(_scene()) for _ in range(6)]
    assert ball_at_rest(steady) == pytest.approx((300.0, 400.0), abs=1.5)
    moving = [detect_ball(_scene(ball=(300 + 5 * k, 400))) for k in range(6)]
    assert ball_at_rest(moving) is None
    gaps = [detect_ball(_scene(ball=None))] * 3 + steady[:3]
    assert ball_at_rest(gaps, min_frames=5) is None
