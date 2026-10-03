"""Golden-image regression tests for force/torque overlay renderers (FTO-30, #11315).

Compares small (320x240) renders of the shared FTO-21 fixtures across headlessly
supported renderers (Matplotlib, OpenCV, and MuJoCo) against committed golden PNGs.
Perceptual tolerance: SSIM >= 0.98 OR mean absolute difference (MAD) <= 2/255.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.integration.cross_engine.force_overlay_fixtures import STANDARD
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ForceGlyphStyle,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.matplotlib_glyphs import (
    draw_glyphs_3d,
    equalize_3d_axes,
)
from src.shared.python.force_overlay.renderers.opencv_glyphs import (
    PinholeProjector,
    draw_glyphs_on_frame,
)

GOLDENS_DIR = Path(__file__).parent / "goldens"
PERCEPTUAL_SSIM_THRESHOLD: float = 0.98
PERCEPTUAL_MAD_THRESHOLD: float = 2.0 / 255.0  # Mean absolute difference per channel

pytestmark = [pytest.mark.unit]


def compute_image_difference(img_a: np.ndarray, img_b: np.ndarray) -> dict[str, float]:
    """Compute mean absolute difference (MAD) and SSIM between two BGR/RGB images.

    Defined exactly once for all force overlay visual regression tests.
    """
    if img_a.shape != img_b.shape:
        raise ValueError(f"Image shape mismatch: {img_a.shape} vs {img_b.shape}")

    a = img_a.astype(np.float64) / 255.0
    b = img_b.astype(np.float64) / 255.0

    mad = float(np.mean(np.abs(a - b)))

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    # SSIM on 2D slice or multi-channel
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a_sq = mu_a**2
    mu_b_sq = mu_b**2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a**2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b**2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    )
    ssim = float(np.mean(ssim_map))

    return {"mad": mad, "ssim": ssim}


def assert_matches_golden(rendered_bgr: np.ndarray, golden_name: str) -> None:
    """Assert rendered BGR array matches the committed golden PNG within tolerance."""
    golden_path = GOLDENS_DIR / golden_name
    assert golden_path.is_file(), (
        f"Golden image not found at {golden_path}. "
        "Goldens must be generated and committed."
    )

    golden_bgr = cv2.imread(str(golden_path), cv2.IMREAD_COLOR)
    assert golden_bgr is not None, f"Failed to read golden image {golden_path}"

    diff = compute_image_difference(rendered_bgr, golden_bgr)
    passed = (
        diff["ssim"] >= PERCEPTUAL_SSIM_THRESHOLD
        or diff["mad"] <= PERCEPTUAL_MAD_THRESHOLD
    )
    assert passed, (
        f"Visual regression failure for {golden_name}: "
        f"SSIM={diff['ssim']:.4f} (threshold >= {PERCEPTUAL_SSIM_THRESHOLD}), "
        f"MAD={diff['mad']:.6f} (threshold <= {PERCEPTUAL_MAD_THRESHOLD:.6f})"
    )


def test_perceptual_metric_identical_images() -> None:
    """Verify perceptual comparison gives exact zero difference on identical arrays."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[50:150, 50:150] = (255, 128, 64)
    diff = compute_image_difference(img, img)
    assert diff["mad"] == 0.0
    assert diff["ssim"] >= 0.9999


def test_matplotlib_hanging_golden() -> None:
    """Render FTO-21 hanging pendulum fixture through Matplotlib 3D and compare with golden."""
    wrench = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="joint_reaction:pivot",
        body="link",
        point_m=STANDARD.pivot,
        force_n=(0.0, 0.0, STANDARD.weight),
        source="fto21_synthetic",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="matplotlib", wrenches=(wrench,))
    glyphs = build_glyphs(frame, ForceGlyphStyle(shaft_radius_m=0.015))

    fig = plt.figure(figsize=(3.2, 2.4), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=15, azim=-60)
    draw_glyphs_3d(ax, glyphs)
    equalize_3d_axes(ax, np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.2]]))
    fig.tight_layout(pad=0.1)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    rgba = np.asarray(buf)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    assert bgr.shape == (240, 320, 3)
    assert_matches_golden(bgr, "matplotlib_hanging.png")


def test_opencv_hanging_golden() -> None:
    """Render FTO-21 hanging pendulum fixture through OpenCV and compare with golden."""
    wrench = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="joint_reaction:pivot",
        body="link",
        point_m=STANDARD.pivot,
        force_n=(0.0, 0.0, STANDARD.weight),
        source="fto21_synthetic",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="opencv", wrenches=(wrench,))
    glyphs = build_glyphs(frame, ForceGlyphStyle(shaft_radius_m=0.02))

    class SyntheticProjector:
        @property
        def world_frame(self) -> str:
            return "adr0041"

        def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pts = np.asarray(points_world, dtype=np.float64)
            # Simple synthetic orthographic projection to 320x240 image
            # Center at (160, 120), scale 100 px/meter
            u = 160.0 + pts[..., 0] * 100.0
            v = 200.0 - pts[..., 2] * 100.0
            uv = np.stack([u, v], axis=-1)
            valid = np.ones(pts.shape[:-1], dtype=bool)
            return uv, valid

    canvas = np.full((240, 320, 3), 30, dtype=np.uint8)
    draw_glyphs_on_frame(canvas, glyphs, SyntheticProjector())

    assert canvas.shape == (240, 320, 3)
    assert_matches_golden(canvas, "opencv_hanging.png")


def test_mujoco_hanging_golden() -> None:
    """Render FTO-21 hanging pendulum fixture through MuJoCo offscreen and compare with golden."""
    try:
        import mujoco
    except ImportError:
        pytest.skip("mujoco not installed")

    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
        add_glyphs_to_scene,
    )

    xml = f"""
    <mujoco model="hanging_pendulum_fixture">
      <visual>
        <global offwidth="320" offheight="240"/>
      </visual>
      <worldbody>
        <light pos="0 -1 2" dir="0 1 -1"/>
        <body name="link" pos="0 0 {STANDARD.pivot_height}">
          <geom name="pivot_geom" type="sphere" size="0.04" rgba="0.5 0.5 0.5 1"/>
          <geom name="rod_geom" type="cylinder" fromto="0 0 0 0 0 -{STANDARD.length}" size="0.02" rgba="0.8 0.8 0.8 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 240, 320)
    renderer.update_scene(data)

    wrench = OverlayWrench(
        kind=WrenchKind.JOINT_REACTION,
        label="joint_reaction:pivot",
        body="link",
        point_m=STANDARD.pivot,
        force_n=(0.0, 0.0, STANDARD.weight),
        source="fto21_synthetic",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="mujoco", wrenches=(wrench,))
    glyphs = build_glyphs(frame, ForceGlyphStyle(shaft_radius_m=0.015))

    receipt = add_glyphs_to_scene(renderer.scene, glyphs)
    assert receipt.added > 0

    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    assert bgr.shape == (240, 320, 3)
    assert_matches_golden(bgr, "mujoco_hanging.png")
