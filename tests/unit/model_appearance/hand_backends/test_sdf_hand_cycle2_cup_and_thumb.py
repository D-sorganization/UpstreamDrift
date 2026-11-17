"""TDD Cycle 2 tests for sdf_hand: cupped palm, opposed thumb, and thumb-side camera."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.contract import GripArticulation
from src.shared.python.model_appearance.hand_backends.sdf_pose import (
    pose_hand_skeleton,
)
from src.shared.python.model_appearance.hand_backends.sdf_render import _camera_for
from src.shared.python.model_appearance.hand_backends.sdf_settings import (
    load_sdf_hand_settings,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("side", ("lead", "trail"))
def test_palm_metacarpals_cup_around_grip_without_flat_paddle(side: str) -> None:
    """Metacarpal bases curve around the grip instead of jutting ~85 mm radially out."""
    settings = load_sdf_hand_settings()
    spec = GripSpec.legacy()
    skeleton = pose_hand_skeleton(
        spec,
        side,  # type: ignore[arg-type]
        GripArticulation.from_grip_spec(spec, side),  # type: ignore[arg-type]
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    assert skeleton.hand_length_m == pytest.approx(
        settings.hand_length_m * skeleton.scale, abs=1e-6
    )
    by_name = {prim.name: prim for prim in skeleton.primitives}
    max_allowed_r = settings.default_grip_radius_m + 0.056 * skeleton.scale
    for digit in ("index", "middle", "ring", "little"):
        base = np.asarray(by_name[f"{digit}_metacarpal_base"].start, dtype=float)
        rho = float(np.hypot(base[0], base[1]))
        assert rho <= max_allowed_r, (side, digit, rho, max_allowed_r)


@pytest.mark.parametrize("side", ("lead", "trail"))
def test_thumb_has_opposed_sweep_and_reaches_past_index(side: str) -> None:
    """Thumb stays distinct and sweeps obliquely while preserving axis[2] > 0.95."""
    settings = load_sdf_hand_settings()
    spec = GripSpec.legacy()
    skeleton = pose_hand_skeleton(
        spec,
        side,  # type: ignore[arg-type]
        GripArticulation.from_grip_spec(spec, side),  # type: ignore[arg-type]
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    by_name = {prim.name: prim for prim in skeleton.primitives}
    th_mc = by_name["thumb_metacarpal"]
    th_prox = by_name["thumb_proximal"]
    th_dist = by_name["thumb_distal"]
    idx_prox = by_name["index_proximal"]

    dist_vec = np.asarray(th_dist.end, dtype=float) - np.asarray(
        th_dist.start, dtype=float
    )
    dist_axis = dist_vec / max(float(np.linalg.norm(dist_vec)), 1e-12)
    assert float(dist_axis[2]) > 0.95
    assert float(th_dist.end[2]) >= float(idx_prox.start[2]) + 0.012 * skeleton.scale

    rad = np.array([th_prox.start[0], th_prox.start[1], 0.0], dtype=float)
    rad /= max(float(np.linalg.norm(rad)), 1e-12)
    tan = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), rad)
    sweep = abs(
        float(
            np.dot(
                np.asarray(th_dist.end, dtype=float)
                - np.asarray(th_mc.start, dtype=float),
                tan,
            )
        )
    )
    assert sweep >= 0.005 * skeleton.scale, (side, sweep)


@pytest.mark.parametrize("side", ("lead", "trail"))
def test_oblique_camera_prefers_thumb_side_and_keeps_70pct_fill(side: str) -> None:
    """Oblique camera shows thumb web from the radial side while keeping frame_fill=0.7."""
    settings = load_sdf_hand_settings()
    assert settings.frame_fill == pytest.approx(0.70)
    spec = GripSpec.legacy()
    skeleton = pose_hand_skeleton(
        spec,
        side,  # type: ignore[arg-type]
        GripArticulation.from_grip_spec(spec, side),  # type: ignore[arg-type]
        grip_radius_m=settings.default_grip_radius_m,
        settings=settings,
    )
    by_name = {prim.name: prim for prim in skeleton.primitives}
    th = np.asarray(by_name["thumb_proximal"].start, dtype=float)
    thumb_xy = np.array([th[0], th[1], 0.0], dtype=float)
    thumb_xy /= max(float(np.linalg.norm(thumb_xy)), 1e-12)

    cam_dir, _cam_up = _camera_for(
        "oblique",
        np.asarray(skeleton.dorsal_hint, dtype=float),
        thumb_dir=thumb_xy,
    )
    assert float(np.dot(cam_dir, thumb_xy)) > 0.15
    assert float(cam_dir[2]) > 0.15
