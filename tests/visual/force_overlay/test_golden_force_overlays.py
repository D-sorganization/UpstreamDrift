"""Golden image visual regression tests for force overlay renderers (FTO-30, #11315).

Renders small 320x240 fixtures from FTO-21 through each renderer headlessly and
compares against committed golden references. Tolerance is Mean Absolute Difference
(MAD) <= 2.0 / 255 (defined once as MAX_MEAN_ABS_DIFF).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pytest

from tests.integration.cross_engine.force_overlay_fixtures import STANDARD

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

# Perceptual tolerance defined once: mean absolute difference <= 2.0 on [0, 255]
MAX_MEAN_ABS_DIFF: Final[float] = 2.0

GOLDENS_DIR: Final[Path] = Path(__file__).parent / "goldens"


def compute_mean_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float:
    """Compute mean absolute difference between two BGR images."""
    if actual.shape != expected.shape:
        raise ValueError(
            f"Image shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        )
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return float(np.mean(diff))


def compare_against_golden(
    name: str, actual_bgr: np.ndarray, regen: bool = False
) -> None:
    """Compare an actual rendered image against the golden reference file."""
    assert actual_bgr.shape == (
        240,
        320,
        3,
    ), f"Expected 320x240 BGR image, got {actual_bgr.shape}"
    golden_path = GOLDENS_DIR / f"{name}.png"

    if regen or os.environ.get("FTO_REGEN_GOLDENS") == "1":
        GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(golden_path), actual_bgr)
        return

    if not golden_path.exists():
        pytest.fail(
            f"Golden image not found: {golden_path}. "
            f"Set FTO_REGEN_GOLDENS=1 to generate initial reference."
        )

    expected_bgr = cv2.imread(str(golden_path))
    assert expected_bgr is not None, f"Failed to read golden image {golden_path}"

    mad = compute_mean_abs_diff(actual_bgr, expected_bgr)
    assert mad <= MAX_MEAN_ABS_DIFF, (
        f"Visual difference for '{name}' exceeded perceptual threshold: "
        f"MAD={mad:.4f} > {MAX_MEAN_ABS_DIFF:.4f}"
    )


def test_matplotlib_golden_render() -> None:
    """Matplotlib 3D glyph renderer matches golden image within perceptual tolerance."""
    from scripts.render_force_overlay_gallery import render_matplotlib_snapshot

    img = render_matplotlib_snapshot(STANDARD, width=320, height=240)
    compare_against_golden("matplotlib_pendulum_320x240", img)


def test_opencv_golden_render() -> None:
    """OpenCV calibrated 2D projected glyphs match golden image within perceptual tolerance."""
    from scripts.render_force_overlay_gallery import render_opencv_snapshot

    img = render_opencv_snapshot(STANDARD, width=320, height=240)
    compare_against_golden("opencv_pendulum_320x240", img)


@pytest.mark.requires_mujoco
@pytest.mark.requires_gl
def test_mujoco_golden_render() -> None:
    """MuJoCo offscreen render matches golden image within perceptual tolerance."""
    mujoco = pytest.importorskip("mujoco")
    from scripts.render_force_overlay_gallery import render_mujoco_snapshot

    img = render_mujoco_snapshot(STANDARD, width=320, height=240)
    compare_against_golden("mujoco_pendulum_320x240", img)
