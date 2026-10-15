"""NSX-16: grip-frame split coordinates, grasp map and kernel (#12561)."""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_wrench import SPLIT_METHODS
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    GripFrame,
    SpatialInertia,
)
from src.shared.python.loop_nullspace.fixtures import planar_golfer_snapshot
from src.shared.python.loop_nullspace.grip_split import (
    SPLIT_COLUMN_NAMES,
    closing_hand_map,
    equal_split_inverse,
    grasp_map,
    hand_wrenches,
    net_club_wrench,
    scale_split,
    split_basis,
    split_coordinates,
    split_functional,
)

pytestmark = pytest.mark.unit

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "loop_nullspace"
    / "goldens"
    / "grasp_map_worked_example.json"
)
TOL = 1e-12


def _frac(x):
    if isinstance(x, list):
        return np.array([_frac(v) for v in x], dtype=np.float64)
    return float(Fraction(x))


@pytest.fixture(scope="module")
def gold():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def example(gold):
    ex = gold["worked_example"]
    fr = ex["frame"]
    frame = GripFrame(
        rotation=_frac(fr["R_world_from_frame"]), origin=_frac(fr["origin_world"])
    )
    return ex, frame, _frac(fr["r_L_world"]), _frac(fr["r_R_world"])


def _club_ne(ex, frame):
    """World-axis Newton-Euler keywords from the golden's frame-axis inputs."""
    bc = ex["bare_club"]
    rot = frame.rotation
    inertia = _frac(bc["inertia_about_com_frame_kg_m2"])
    r = rot @ _frac(bc["com_from_origin_frame_m"])
    w = rot @ _frac(bc["omega_frame_rad_s"])
    return {
        "mass_kg": _frac(bc["mass_kg"]),
        "gravity_m_s2": _frac(bc["gravity_world_m_s2"]),
        "com_m": frame.origin + r,
        "com_acceleration_m_s2": _frac(bc["a_com_world_m_s2"]),
        "inertia_world_kg_m2": rot @ inertia @ rot.T,
        "angular_velocity_rad_s": w,
        "angular_acceleration_rad_s2": rot @ _frac(bc["alpha_frame_rad_s2"]),
    }


@pytest.fixture(scope="module")
def snapshot(example):
    ex, frame, _, _ = example
    snap = planar_golfer_snapshot().snapshot
    bare = SpatialInertia(
        mass=_frac(ex["bare_club"]["mass_kg"]),
        com=_frac(ex["bare_club"]["com_from_origin_frame_m"]),
        inertia=_frac(ex["bare_club"]["inertia_about_com_frame_kg_m2"]),
    )
    return dataclasses.replace(snap, grip_frame=frame, bare_club=bare)


def test_grasp_map_matches_golden_exactly(gold, example) -> None:
    _, frame, r_l, r_r = example
    g = grasp_map(frame, r_l, r_r)
    assert g.shape == (6, 12)
    np.testing.assert_allclose(g, _frac(gold["axial_matrices"]["G"]), atol=TOL)


def test_kernel_matches_golden_and_is_exact(gold, example) -> None:
    _, frame, r_l, r_r = example
    g = grasp_map(frame, r_l, r_r)
    k = split_basis(g)
    assert k.shape == (12, 6)
    np.testing.assert_allclose(k, _frac(gold["axial_matrices"]["K"]), atol=TOL)
    np.testing.assert_allclose(g @ k, np.zeros((6, 6)), atol=TOL)
    assert np.linalg.matrix_rank(k) == 6
    assert np.linalg.matrix_rank(g) == 6
    assert len(SPLIT_COLUMN_NAMES) == 6
    assert list(SPLIT_COLUMN_NAMES) == gold["axial_matrices"]["column_names"]


def test_phi_and_right_inverse_identities(example) -> None:
    _, frame, r_l, r_r = example
    g = grasp_map(frame, r_l, r_r)
    k = split_basis(g)
    phi, gs = split_functional(), equal_split_inverse()
    np.testing.assert_allclose(phi @ k, np.eye(6), atol=TOL)
    np.testing.assert_allclose(g @ gs, np.eye(6), atol=TOL)
    np.testing.assert_allclose(phi @ gs, np.zeros((6, 6)), atol=TOL)
    stacked = np.vstack([g, phi])
    np.testing.assert_allclose(np.linalg.inv(stacked), np.hstack([gs, k]), atol=1e-11)


def test_hand_loads_match_golden_and_round_trip(gold, example) -> None:
    ex, frame, r_l, r_r = example
    g = grasp_map(frame, r_l, r_r)
    net = _frac(ex["net_wrench_recovered"])
    s = _frac(ex["split_coordinates"]["s"])
    w_l, w_r = hand_wrenches(net, s, g)
    loads = ex["hand_loads_frame"]
    np.testing.assert_allclose(w_l[:3], _frac(loads["f_L"]), atol=TOL)
    np.testing.assert_allclose(w_l[3:], _frac(loads["c_L"]), atol=TOL)
    np.testing.assert_allclose(w_r[:3], _frac(loads["f_R"]), atol=TOL)
    np.testing.assert_allclose(w_r[3:], _frac(loads["c_R"]), atol=TOL)
    np.testing.assert_allclose(split_coordinates(w_l, w_r), s, atol=TOL)
    np.testing.assert_allclose(g @ np.concatenate([w_l, w_r]), net, atol=1e-11)


def test_round_trip_random_split(example) -> None:
    _, frame, r_l, r_r = example
    g = grasp_map(frame, r_l, r_r)
    rng = np.random.default_rng(3)
    for _ in range(10):
        net, s = rng.normal(size=6), rng.normal(size=6)
        w_l, w_r = hand_wrenches(net, s, g)
        np.testing.assert_allclose(split_coordinates(w_l, w_r), s, atol=TOL)
        np.testing.assert_allclose(g @ np.concatenate([w_l, w_r]), net, atol=TOL)


def test_net_wrench_matches_bare_club_newton_euler_golden(example, snapshot) -> None:
    ex, frame, _, _ = example
    rec = net_club_wrench(snapshot, _club_ne(ex, frame))
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert rec.units == "N,N,N,N m,N m,N m"
    np.testing.assert_allclose(
        rec.values, _frac(ex["net_wrench_recovered"]), rtol=1e-12, atol=1e-12
    )


def test_net_wrench_rejects_mass_mismatch_and_missing_keys(example, snapshot) -> None:
    ex, frame, _, _ = example
    ne = _club_ne(ex, frame)
    with pytest.raises(ValueError, match="mass"):
        net_club_wrench(snapshot, {**ne, "mass_kg": 2.0})
    bad = dict(ne)
    bad.pop("com_m")
    with pytest.raises(ValueError, match="club keywords"):
        net_club_wrench(snapshot, bad)


def test_kernel_and_split_independent_of_origin(example) -> None:
    ex, frame, r_l, r_r = example
    g = grasp_map(frame, r_l, r_r)
    k = split_basis(g)
    t = np.array([0.03, -0.02, 0.05])
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    transport = np.eye(6)
    transport[3:, :3] = -skew
    shifted = transport @ g
    np.testing.assert_allclose(shifted @ k, np.zeros((6, 6)), atol=TOL)
    # Split coordinates depend only on the hand loads, not on the origin.
    s = _frac(ex["split_coordinates"]["s"])
    w_l, w_r = hand_wrenches(_frac(ex["net_wrench_recovered"]), s, g)
    np.testing.assert_allclose(split_coordinates(w_l, w_r), s, atol=TOL)


def test_scaling_and_units(example) -> None:
    ex, *_ = example
    s = _frac(ex["split_coordinates"]["s"])
    np.testing.assert_allclose(
        scale_split(s), _frac(ex["split_coordinates"]["s_scaled"]), atol=TOL
    )
    with pytest.raises(ValueError):
        scale_split(np.ones(5))


@pytest.mark.parametrize("hand", ["R", "L"])
def test_closing_hand_map(gold, example, hand) -> None:
    ex, frame, r_l, r_r = example
    g = grasp_map(frame, r_l, r_r)
    net = _frac(ex["net_wrench_recovered"])
    s = _frac(ex["split_coordinates"]["s"])
    w_l, w_r = hand_wrenches(net, s, g)
    d = np.array([_frac(gold["axial_matrices"]["delta_m"]), 0.0, 0.0])
    a_mat, s0 = closing_hand_map(hand, d, net)
    lam = -(w_r if hand == "R" else w_l)
    np.testing.assert_allclose(a_mat @ lam + s0, s, atol=1e-11)
    if hand == "R":
        cl = ex["closing_hand"]
        np.testing.assert_allclose(a_mat, -_frac(cl["Q"]), atol=TOL)
        np.testing.assert_allclose(s0, _frac(cl["s0"]), atol=TOL)
        np.testing.assert_allclose(lam, _frac(cl["lambda_frame"]), atol=TOL)


def test_validation_errors(example) -> None:
    _, frame, r_l, r_r = example
    with pytest.raises(ValueError, match="midpoint"):
        grasp_map(frame, r_l, r_r + 0.01)
    with pytest.raises(ValueError, match="distinct"):
        grasp_map(frame, r_l, r_l)
    with pytest.raises(ValueError):
        split_basis(np.ones((6, 12)))
    g = grasp_map(frame, r_l, r_r)
    with pytest.raises(ValueError):
        hand_wrenches(np.ones(5), np.ones(6), g)
    with pytest.raises(ValueError):
        split_coordinates(np.ones(5), np.ones(6))
    with pytest.raises(ValueError, match="hand"):
        closing_hand_map("X", np.zeros(3), np.zeros(6))


def test_loop_nullspace_family_is_a_declared_split_method() -> None:
    assert "loop_nullspace_family" in SPLIT_METHODS
