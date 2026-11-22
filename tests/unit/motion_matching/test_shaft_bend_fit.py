"""Kinematic shaft-bend fit (#15150 C10): a synthetic driver chain bent by a
known tip-force history; six markers made from it; the fit gives it back."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    build_shaft_chain,
    clamped_shaft_document,
)
from src.shared.python.motion_matching.shaft_bend_fit import (
    ChainGeometry,
    club_frame_targets,
    fit_shaft_bend,
    load_bend_settings,
    marker_positions,
    observability,
    shape_basis,
)

pytestmark = pytest.mark.unit

#: Tip-slope recovery tolerances: exact markers 0.05 deg, 1 mm noise 0.5 deg.
EXACT_TOL_DEG = 0.05
NOISY_TOL_DEG = 0.5
NOISE_M = 0.001
PEAK_SLOPE_DEG = (10.0, -4.0)  # club x / z planes, inside the 20 deg bound
#: Head and shaft markers in the club-body frame (shaft along -y, axis z=0.064).
OFFSETS = np.array(
    [
        [0.020, 0.040, 0.085],
        [-0.025, 0.110, 0.125],
        [0.025, 0.120, 0.135],
        [0.0, -0.667, 0.064],
        [0.025, -0.740, 0.087],
        [0.0, -0.788, 0.064],
    ]
)


@pytest.fixture(scope="module")
def geometry() -> ChainGeometry:
    return ChainGeometry.from_chain(build_shaft_chain(DRIVER))


def _history(geometry: ChainGeometry, frames: int = 60) -> np.ndarray:
    """Smooth tip-force bend (frames, joints, 2) peaking at PEAK_SLOPE_DEG."""
    pulse = np.sin(np.linspace(0.0, math.pi, frames)) ** 2
    weights = shape_basis(geometry, 1)[0]
    peaks = np.radians(PEAK_SLOPE_DEG)
    return pulse[:, None, None] * weights[None, :, None] * peaks[None, None, :]


def _tip_slopes_deg(bend: np.ndarray) -> np.ndarray:
    return np.degrees(bend.sum(axis=1))


def test_shape_basis_sums_to_unit_tip_slope(geometry: ChainGeometry) -> None:
    basis = shape_basis(geometry, 2)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0)
    assert np.all(np.diff(basis[0]) < 0)  # tip force: largest moment at the grip


def test_positive_bend_x_raises_tip_along_z(geometry: ChainGeometry) -> None:
    bend = np.zeros((geometry.n_joints, 2))
    np.testing.assert_allclose(marker_positions(geometry, bend, OFFSETS), OFFSETS)
    bend[-1, 0] = 0.1
    moved = marker_positions(geometry, bend, OFFSETS)
    assert moved[0, 2] > OFFSETS[0, 2]
    np.testing.assert_allclose(moved[3:], OFFSETS[3:], atol=1e-12)


def test_identifiability(geometry: ChainGeometry) -> None:
    """Head: slope + offset per plane; shaft markers add joint 0; 2 of 8 hidden."""
    settings = load_bend_settings()
    head = observability(geometry, OFFSETS[:3], settings)
    assert head.strong_rank == 4
    both = observability(geometry, OFFSETS, settings)
    assert both.strong_rank == 6
    assert both.singular_values[-2:].max() < 1e-12 * both.singular_values[0]


@pytest.mark.parametrize(
    ("noise_m", "tol_deg"), [(0.0, EXACT_TOL_DEG), (NOISE_M, NOISY_TOL_DEG)]
)
def test_recovery(geometry: ChainGeometry, noise_m: float, tol_deg: float) -> None:
    truth = _history(geometry)
    markers = np.stack([marker_positions(geometry, b, OFFSETS) for b in truth])
    markers = markers + np.random.default_rng(7).normal(0.0, noise_m, markers.shape)
    markers[20:23, 1] = np.nan  # a head marker drops out
    fit = fit_shaft_bend(geometry, markers, OFFSETS, load_bend_settings())
    error = np.abs(_tip_slopes_deg(fit.bend) - _tip_slopes_deg(truth))
    assert error.max() < tol_deg


def test_break_lets_the_bend_drop_at_contact(geometry: ChainGeometry) -> None:
    truth = _history(geometry)
    truth[40:] = 0.0  # contact at frame 40 unloads the shaft at once
    markers = np.stack([marker_positions(geometry, b, OFFSETS) for b in truth])
    cut = fit_shaft_bend(geometry, markers, OFFSETS, load_bend_settings(), (40,))
    error = np.abs(_tip_slopes_deg(cut.bend) - _tip_slopes_deg(truth))
    assert error[:40].max() < EXACT_TOL_DEG


def test_club_frame_targets_remove_rigid_motion(geometry: ChainGeometry) -> None:
    bend = _history(geometry)[30]
    local = marker_positions(geometry, bend, OFFSETS)
    c, s = math.cos(0.7), math.sin(0.7)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = local @ rot.T + np.array([0.3, -1.2, 0.9])
    back = club_frame_targets(world[None], local[None], (3, 4, 5))[0]
    np.testing.assert_allclose(back, local, atol=1e-9)
    straight = club_frame_targets(world[None], OFFSETS, (3, 4, 5))[0]
    np.testing.assert_allclose(straight[3:], OFFSETS[3:], atol=5e-3)


def test_kinematics_match_mujoco_chain(geometry: ChainGeometry) -> None:
    """The numpy chain equals the emitted MJCF chain (primitive order Rx, Rz)."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import shaft_chain_mjcf as mjcf

    document = clamped_shaft_document(build_shaft_chain(DRIVER))
    model = mujoco.MjModel.from_xml_string(mjcf.export_clamped_shaft_mjcf(document))
    data = mujoco.MjData(model)
    bend = np.radians([[3.0, -2.0], [4.0, 1.5], [-2.5, 3.0], [5.0, -4.0]])
    data.qpos[:] = bend.reshape(-1)
    mujoco.mj_kinematics(model, data)
    ours = marker_positions(geometry, bend, np.array([document["tip_point_m"]]))[0]
    np.testing.assert_allclose(ours, data.site(mjcf.TIP_SITE).xpos, atol=1e-12)


def test_contracts(geometry: ChainGeometry) -> None:
    with pytest.raises(ValueError, match="shapes"):
        shape_basis(geometry, 3)
    settings = load_bend_settings()
    with pytest.raises(ValueError, match="markers"):
        fit_shaft_bend(geometry, np.zeros((4, 5, 3)), OFFSETS, settings)
    with pytest.raises(ValueError, match="breaks"):
        fit_shaft_bend(geometry, np.zeros((4, 6, 3)), OFFSETS, settings, (0,))


def _two_shape_fit(
    geometry: ChainGeometry, x_amplitudes_deg: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit three identical frames bent by two x-plane shapes; max slope 20 deg."""
    base = load_bend_settings()
    settings = type(base)(
        shapes_per_plane=2,
        smoothness_weight_m_per_rad=0.0,
        ridge_weight_m_per_rad=0.0,
        max_tip_slope_deg=20.0,
        rank_relative_tolerance=base.rank_relative_tolerance,
        solver_max_nfev=1000,
        solver_tolerance=1e-12,
    )
    basis = shape_basis(geometry, 2)
    amps = np.radians(x_amplitudes_deg)
    bend = np.zeros((geometry.n_joints, 2))
    bend[:, 0] = amps[0] * basis[0] + amps[1] * basis[1]
    markers = np.tile(marker_positions(geometry, bend, OFFSETS), (3, 1, 1))
    fit = fit_shaft_bend(geometry, markers, OFFSETS, settings)
    return np.degrees(fit.amplitudes), np.degrees(fit.bend.sum(axis=1))


def test_same_sign_amplitudes_clamp_at_combined_tip_slope(
    geometry: ChainGeometry,
) -> None:
    """Two 15 deg shapes sum to 30 deg: the 20 deg maximum is on the sum."""
    amplitudes, slopes = _two_shape_fit(geometry, (15.0, 15.0))
    assert np.all(np.abs(slopes[:, 0]) <= 20.0 + 1e-9)
    assert np.all(np.abs(amplitudes[:, :2].sum(axis=1)) <= 20.0 + 1e-9)
    np.testing.assert_allclose(slopes[:, 0], 20.0, atol=1e-6)


def test_opposing_amplitudes_with_small_sum_stay_allowed(
    geometry: ChainGeometry,
) -> None:
    """25 and -10 deg individually exceed 20 deg but sum to an allowed 15 deg."""
    amplitudes, slopes = _two_shape_fit(geometry, (25.0, -10.0))
    np.testing.assert_allclose(slopes[:, 0], 15.0, atol=1e-6)
    np.testing.assert_allclose(amplitudes[:, 0], 25.0, atol=1e-3)
    np.testing.assert_allclose(amplitudes[:, 1], -10.0, atol=1e-3)
