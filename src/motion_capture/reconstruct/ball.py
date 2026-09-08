"""Golf-ball detection in one view (issue #9621).

The ball at address is the one scene point every camera sees on every take,
so it anchors extrinsic self-calibration and fixes translation between takes.
This detector is deliberately classical and explainable: a bright,
low-saturation, near-circular blob whose radius lies in the expected range.
It returns *candidates ranked by a score* and never invents a ball — when no
candidate passes the gates the result says so with the reason, and an
operator-provided hint (``plan`` ``ball_px``) can steer the choice.

Assumptions (stated, not hidden): a white or near-white ball, a background
darker or more saturated than the ball, and a known plausible radius range
in pixels for the camera's distance. A ball on a white mat will not be found
by this detector; that case is reported, not guessed.

Real-frame evidence (take 2, 2026-09-07, down-the-line view at address): the
spare balls on the mat are found at 4 px radius once the value gate suits the
bay lighting; the addressed ball sits under the club head in that frame and
is not visible, which is exactly why the at-rest tracker over the address
phase and the operator hint exist. Multi-view agreement (C3) is what finally
disambiguates several balls on one mat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BallDetectorOptions:
    """Gates and weights; all validated."""

    min_radius_px: float = 4.0
    max_radius_px: float = 40.0
    # Defaults come from the lab bay on 2026-09-07 (1280x720, bay lights on):
    # the balls on the mat read V 150-190 and S < 90, so a brighter gate misses
    # them; a lit outdoor scene can raise min_value again.
    min_value: int = 120  # V channel (0-255): the ball is bright
    max_saturation: int = 90  # S channel (0-255): the ball is not coloured
    min_circularity: float = 0.8  # 4*pi*area / perimeter^2; a square is 0.785
    expected_radius_px: float | None = None  # candidates near this radius win
    hint_px: tuple[float, float] | None = None  # operator hint, pixel coordinates
    hint_radius_px: float = 80.0  # candidates farther than this from the hint lose

    def __post_init__(self) -> None:
        require(0 < self.min_radius_px < self.max_radius_px, "radius range invalid")
        require(0 <= self.min_value <= 255, "min_value must be 0..255")
        require(0 <= self.max_saturation <= 255, "max_saturation must be 0..255")
        require(0 < self.min_circularity <= 1.0, "min_circularity must be in (0, 1]")
        require(self.hint_radius_px > 0, "hint_radius_px must be positive")
        require(
            self.expected_radius_px is None or self.expected_radius_px > 0,
            "expected_radius_px must be positive when set",
        )


class BallCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    center_px: tuple[float, float]
    radius_px: float
    circularity: float
    mean_value: float
    score: float


class BallDetection(BaseModel):
    """What was found, or why nothing was."""

    model_config = ConfigDict(frozen=True)

    found: bool
    best: BallCandidate | None = None
    candidates: tuple[BallCandidate, ...] = ()
    reason: str | None = None


def _mask(image_bgr: Array, options: BallDetectorOptions) -> npt.NDArray[np.uint8]:
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2] >= options.min_value
    pale = hsv[:, :, 1] <= options.max_saturation
    mask = (bright & pale).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return np.asarray(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), dtype=np.uint8)


def _score(c: BallCandidate, options: BallDetectorOptions) -> float:
    score = c.circularity * min(1.0, c.mean_value / 255.0)
    if options.expected_radius_px is not None:
        ratio = np.log(c.radius_px / options.expected_radius_px)
        score *= float(np.exp(-((ratio / 0.5) ** 2)))  # half-decade tolerance
    if options.hint_px is not None:
        dist = float(
            np.hypot(
                c.center_px[0] - options.hint_px[0], c.center_px[1] - options.hint_px[1]
            )
        )
        score *= float(np.exp(-((dist / options.hint_radius_px) ** 2)))
    return score


def detect_ball(
    image_bgr: Array, options: BallDetectorOptions = BallDetectorOptions()
) -> BallDetection:
    """Rank bright round blobs in one BGR frame; ``found`` only when one passes.

    Precondition: ``image_bgr`` is ``(H, W, 3)`` uint8. Postcondition: the
    returned candidates are sorted by descending score and every one satisfies
    the radius and circularity gates.
    """
    import cv2

    require(
        image_bgr.ndim == 3 and image_bgr.shape[2] == 3 and image_bgr.dtype == np.uint8,
        "image must be HxWx3 uint8",
    )
    mask = _mask(image_bgr, options)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    candidates: list[BallCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0 or perimeter <= 0:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not (options.min_radius_px <= radius <= options.max_radius_px):
            continue
        circularity = 4.0 * np.pi * area / (perimeter**2)
        if circularity < options.min_circularity:
            continue
        blob = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        mean_value = float(value[blob > 0].mean())
        candidate = BallCandidate(
            center_px=(float(cx), float(cy)),
            radius_px=float(radius),
            circularity=float(circularity),
            mean_value=mean_value,
            score=0.0,
        )
        candidates.append(
            candidate.model_copy(update={"score": _score(candidate, options)})
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    if not candidates:
        return BallDetection(
            found=False,
            reason=(
                "no bright, pale, near-circular blob within "
                f"{options.min_radius_px:.0f}-{options.max_radius_px:.0f} px radius"
            ),
        )
    return BallDetection(found=True, best=candidates[0], candidates=tuple(candidates))


def ball_at_rest(
    detections: list[BallDetection], *, max_drift_px: float = 2.0, min_frames: int = 5
) -> tuple[float, float] | None:
    """The ball position while it is stationary, or None if never stable.

    Takes per-frame detections (address phase), requires at least ``min_frames``
    consecutive found frames whose centres stay within ``max_drift_px`` of their
    median, and returns that median. Postcondition: None means no stable run.
    """
    require(max_drift_px > 0 and min_frames >= 2, "invalid rest criteria")
    run: list[tuple[float, float]] = []
    for det in detections:
        if not det.found or det.best is None:
            run = []
            continue
        run.append(det.best.center_px)
        if len(run) >= min_frames:
            pts = np.array(run[-min_frames:])
            med = np.median(pts, axis=0)
            if np.all(np.linalg.norm(pts - med, axis=1) <= max_drift_px):
                return float(med[0]), float(med[1])
    return None
