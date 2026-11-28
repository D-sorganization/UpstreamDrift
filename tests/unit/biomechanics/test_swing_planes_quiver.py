"""Unit tests for Biomechanics Slice 3: swing planes and WSCG quiver overlay (#15664)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import numpy as np
import pytest

from src.shared.python.biomechanics.swing_planes_quiver import (
    SwingPlaneResult,
    WscgQuiverOverlay,
    build_wscg_quiver_overlay,
    compute_hogan_glass_plane,
    compute_mackenzie_net_force_plane,
    compute_trackman_swing_plane,
    fit_kwon_fsp_plane,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_BAD = [
    [[0, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    [[0.5, 0.5, 0.5]] * 3,
    [[0, 0, 0], [1, 0, 0], [0, math.nan, 0]],
]


def _angle_rad(u: tuple[float, float, float], v: np.ndarray) -> float:
    """Return unsigned angle in radians between two 3D vectors."""
    ua, va = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    dot = abs(float(np.dot(ua / np.linalg.norm(ua), va / np.linalg.norm(va))))
    return float(math.acos(min(1.0, max(-1.0, dot))))


def test_fit_kwon_fsp_plane_recovers_exact_normal_and_rmse() -> None:
    """Kwon FSP SVD fit recovers exact unit normal (< 1e-6 rad) and zero RMSE on planar arc."""
    n = np.array([0.0, -math.sin(math.radians(52.0)), math.cos(math.radians(52.0))])
    u = np.cross(n, np.array([1.0, 0.0, 0.0]))
    u /= np.linalg.norm(u)
    v, orig = np.cross(n, u), np.array([0.1, -0.35, 0.85])
    ts = np.linspace(-0.6, 0.6, 41)
    c, s = math.cos, math.sin
    arc = np.array([orig + 1.15 * (c(a) * u + s(a) * v) for a in ts])
    res = fit_kwon_fsp_plane(arc)
    assert isinstance(res, SwingPlaneResult) and res.plane_kind == "kwon_fsp"
    assert np.linalg.norm(res.normal) == pytest.approx(1.0, abs=1e-12)
    assert _angle_rad(res.normal, n) < 1e-6 and res.plane_rmse_m < 1e-9
    assert res.normal[2] > 0.0 and res.sign_convention
    with pytest.raises(FrozenInstanceError):
        res.plane_rmse_m = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("bad", _BAD)
def test_fit_kwon_fsp_plane_rejects_degenerate_points(bad: list[list[float]]) -> None:
    """Kwon FSP rejects fewer than 3 points, collinear, coincident, or non-finite sets."""
    with pytest.raises(ValueError):
        fit_kwon_fsp_plane(np.array(bad, dtype=float))


def test_mackenzie_trackman_and_hogan_planes_contracts_and_orientation() -> None:
    """MacKenzie, TrackMan, and Hogan planes return frozen SwingPlaneResult with consistent orientation."""
    g, h = np.array([0.0, -0.2, 1.0]), np.array([0.0, -0.9, 0.2])
    f, v = np.array([120.0, 35.0, 80.0]), np.array([32.0, 2.0, -4.0])
    mk_fn = compute_mackenzie_net_force_plane
    tm_fn = compute_trackman_swing_plane
    hg_fn = compute_hogan_glass_plane
    p_sh, p_ball = [0.0, -0.45, 1.45], [0.0, 0.0, 0.0]
    mk, tm, hg = mk_fn(g, h, f), tm_fn(v, g, h), hg_fn(p_sh, p_ball)
    for r, kind in (
        (mk, "mackenzie_net_force"),
        (tm, "trackman_swing_plane"),
        (hg, "hogan_glass_plane"),
    ):
        assert isinstance(r, SwingPlaneResult) and r.plane_kind == kind
        assert np.linalg.norm(r.normal) == pytest.approx(1.0, abs=1e-12)
        assert abs(float(np.dot(r.normal, r.in_plane_x))) < 1e-12
        assert abs(float(np.dot(r.normal, r.in_plane_y))) < 1e-12
        assert r.plane_rmse_m == pytest.approx(0.0, abs=1e-12)
        assert r.normal[2] >= 0.0 and r.sign_convention
        with pytest.raises(FrozenInstanceError):
            r.slope_deg = 0.0  # type: ignore[misc]
    rp = hg_fn(p_sh, p_ball, reference_normal=(0, 1, 1))
    rn = hg_fn(p_sh, p_ball, reference_normal=(0, -1, -1))
    assert np.dot(rp.normal, rn.normal) == pytest.approx(-1.0, abs=1e-12)
    for fn, args in (
        (tm_fn, (h - g, g, h)),
        (mk_fn, (g, h, h - g)),
        (hg_fn, ([1, 0, 0], p_ball)),
    ):
        with pytest.raises(ValueError):
            fn(*args)
    ms, ts = np.asarray(mk.in_plane_x) * 0.2, np.asarray(tm.in_plane_x) * 0.2
    mk_m = mk_fn([g, g + ms], [h, h + ms], [f, 0.5 * f])
    tm_m = tm_fn([v, 0.8 * v], [g, g + ts], [h, h + ts])
    hg_m = hg_fn([p_sh, [0.3, -0.45, 1.45]], p_ball)
    assert max(mk_m.plane_rmse_m, tm_m.plane_rmse_m, hg_m.plane_rmse_m) < 1e-9


def test_build_wscg_quiver_overlay_physics_and_validation() -> None:
    """WSCG quiver overlay computes KE, work-rate, WSCG vectors, and applies magnitude floors."""
    pos = [[0.0, 0.0, 0.9], [0.0, 0.0, 1.3], [0.2, -0.5, 0.4]]
    vel = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    frc = [[10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [1e-9, 0.0, 0.0]]
    fn = build_wscg_quiver_overlay
    segs, m, rke = ("pelvis", "torso", "club"), [10.0, 20.0, 0.4], [1.0, 2.0, 0.5]
    ov = fn(segs, pos, vel, frc, m, rke, 1e-4, 0.5)
    assert isinstance(ov, WscgQuiverOverlay)
    assert ov.segment_kinetic_energy_j == pytest.approx((6.0, 42.0, 0.5))
    assert ov.total_kinetic_energy_j == pytest.approx(48.5)
    assert ov.segment_work_rates_w == pytest.approx((10.0, 0.0, 0.0))
    assert ov.active_mask == (True, True, False)
    assert ov.quiver_vectors[0] == pytest.approx((5.0, 0.0, 0.0))
    assert ov.quiver_vectors[2] == (0.0, 0.0, 0.0)
    assert ov.work_rate_vectors[0] == pytest.approx((5.0, 0.0, 0.0))
    assert ov.work_rate_vectors[1] == (0.0, 0.0, 0.0)
    assert ov.wscg_position_m[2] == pytest.approx((9.0 + 26.0 + 0.16) / 30.4)
    with pytest.raises(ValueError):
        fn(("p",), pos[:1], vel[:1], frc[:1], [-1.0])
    with pytest.raises(ValueError):
        fn(("a", "b"), pos[:1], vel[:2], frc[:2], [1.0, 1.0])
