"""Hand-split envelope quantities under a restricted reaction set (NSX-26).

The split coordinates are the affine closing-hand map of NSX-16 evaluated on
the closure multiplier.  The multiplier lives at the closure point ``P``; the
hand grip points lie on the shaft at signed coordinates ``x_L``, ``x_R`` from
``P``.  A dropped closure row is a multiplier fixed at zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import ClosureKind
from src.shared.python.loop_nullspace.decomposition import decompose
from src.shared.python.loop_nullspace.grip_split import (
    SPLIT_COLUMN_NAMES,
    grasp_map,
    split_functional,
)
from src.shared.python.loop_nullspace.evidence import GripFrame
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.registry import HypothesisSet
from src.shared.python.loop_nullspace.restriction import restrict
from src.shared.python.loop_nullspace.restriction_hand_split import (
    HandSplitFrame,
    hand_split_quantities,
    hand_split_values,
)
from tests.unit.loop_nullspace.restriction_oracles import identity_closure_snapshot

pytestmark = pytest.mark.unit

W6, S5, B3 = ClosureKind.WELD6, ClosureKind.SHAFT_FREE5, ClosureKind.BALL3


def _hands(x_left: float, x_right: float, y_left, y_right):
    """Net wrench about ``P`` and the multiplier at ``P`` from known hand loads."""
    p_to = lambda x: np.array([x, 0.0, 0.0])  # noqa: E731 - hand grip point from P
    f_l, c_l = y_left[:3], y_left[3:]
    f_r, c_r = y_right[:3], y_right[3:]
    force = f_l + f_r
    moment_p = c_l + c_r + np.cross(p_to(x_left), f_l) + np.cross(p_to(x_right), f_r)
    # wrench of the right hand about P; lambda is minus it (+J^T lambda acts on the tree)
    lam_p = -np.concatenate([f_r, c_r + np.cross(p_to(x_right), f_r)])
    return np.concatenate([force, moment_p]), lam_p


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("x_l", "x_r"), [(-0.1, 0.1), (0.05, 0.3), (-0.2, 0.0)])
def test_split_values_equal_phi_y_for_known_hand_loads(seed, x_l, x_r) -> None:
    rng = np.random.default_rng(seed)
    y_l, y_r = rng.normal(size=6), rng.normal(size=6)
    net_p, lam_p = _hands(x_l, x_r, y_l, y_r)
    frame = HandSplitFrame(x_l, x_r, net_p)
    expected = split_functional() @ np.concatenate([y_l, y_r])
    np.testing.assert_allclose(hand_split_values(frame, lam_p), expected, atol=1e-12)


def test_left_closing_hand_is_supported() -> None:
    rng = np.random.default_rng(5)
    y_l, y_r = rng.normal(size=6), rng.normal(size=6)
    x_l, x_r = -0.12, 0.08
    net_p, _ = _hands(x_l, x_r, y_l, y_r)
    lam_p = -np.concatenate([y_l[:3], y_l[3:] + np.cross([x_l, 0, 0], y_l[:3])])
    frame = HandSplitFrame(x_l, x_r, net_p, closing_hand="L")
    expected = split_functional() @ np.concatenate([y_l, y_r])
    np.testing.assert_allclose(hand_split_values(frame, lam_p), expected, atol=1e-12)


def test_split_satisfies_the_grasp_map_identities() -> None:
    """``G y`` is the net wrench about the grip midpoint ``o``; ``Phi y = s``."""
    rng = np.random.default_rng(7)
    y_l, y_r = rng.normal(size=6), rng.normal(size=6)
    x_l, x_r = -0.1, 0.3
    net_p, lam_p = _hands(x_l, x_r, y_l, y_r)
    s = hand_split_values(HandSplitFrame(x_l, x_r, net_p), lam_p)
    origin = np.array([(x_l + x_r) / 2, 0.0, 0.0])
    g_map = grasp_map(GripFrame(np.eye(3), origin), [x_l, 0, 0], [x_r, 0, 0])
    y = np.concatenate([y_l, y_r])
    net_o = np.concatenate([net_p[:3], net_p[3:] - np.cross(origin, net_p[:3])])
    np.testing.assert_allclose(g_map @ y, net_o, atol=1e-12)
    np.testing.assert_allclose(s, split_functional() @ y, atol=1e-12)


def test_closure_point_on_the_closing_hand_makes_the_transport_the_identity() -> None:
    rng = np.random.default_rng(3)
    lam = rng.normal(size=6)
    net = rng.normal(size=6)
    frame = HandSplitFrame(-0.1, 0.0, net)  # x_R = 0: the closure point is the grip
    plain = hand_split_values(HandSplitFrame(-0.1, 0.0, net), lam)
    np.testing.assert_allclose(hand_split_values(frame, lam), plain)


# ------------------------------------------------------ quantities on snapshots
def _quantities(kind: ClosureKind, frame: HandSplitFrame):
    base = identity_closure_snapshot(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    snap = restrict(base, kind)
    dec = decompose(snap, weights=np.eye(6))
    fs = FeasibleSet.from_hypotheses(dec, HypothesisSet("t", "T", (), ()))
    return snap, dec, hand_split_quantities(lambda _s: frame)(fs, snap)


@pytest.mark.parametrize("kind", [W6, S5, B3])
def test_quantities_are_the_split_of_the_embedded_multiplier(kind) -> None:
    rng = np.random.default_rng(11)
    frame = HandSplitFrame(-0.1, 0.1, rng.normal(size=6))
    snap, dec, qs = _quantities(kind, frame)
    assert [q.name for q in qs] == list(SPLIT_COLUMN_NAMES)
    rows = {W6: range(6), S5: (0, 1, 2, 4, 5), B3: (0, 1, 2)}[kind]
    for _ in range(4):
        z = rng.normal(size=dec.nullity)
        x = dec.x(z)
        lam_full = np.zeros(6)
        lam_full[list(rows)] = x[dec.n_actuated : dec.n_actuated + dec.m]
        expected = hand_split_values(frame, lam_full)
        got = np.array([q.const + q.coeff @ z for q in qs])
        np.testing.assert_allclose(got, expected, atol=1e-12)
    assert [q.units for q in qs] == ["N", "N", "N", "N m", "N m", "N m"]


def test_quantity_factory_rejects_coincident_hands() -> None:
    with pytest.raises(ValueError, match="distinct"):
        HandSplitFrame(0.1, 0.1, np.zeros(6))
    with pytest.raises(ValueError, match="net_wrench"):
        HandSplitFrame(-0.1, 0.1, np.zeros(5))
    with pytest.raises(ValueError, match="closing_hand"):
        HandSplitFrame(-0.1, 0.1, np.zeros(6), closing_hand="X")
