"""Pin Featherstone passive X_J convention for jcalc (issue #15233)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.spatial_algebra.joints import jcalc
from src.shared.python.spatial_algebra.transforms import xlt, xrot

pytestmark = pytest.mark.unit

# RBDA / spatial_v2 passive rotation E = R^T (Featherstone coordinate transforms).


def _featherstone_e_rot(jtype: str, q: float) -> np.ndarray:
    c = float(np.cos(q))
    s = float(np.sin(q))
    if jtype == "Rx":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=np.float64)
    if jtype == "Ry":
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)
    if jtype == "Rz":
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    msg = f"unsupported revolute type: {jtype}"
    raise ValueError(msg)


@pytest.mark.parametrize("jtype", ["Rx", "Ry", "Rz"])
def test_revolute_matches_featherstone_xrot(jtype: str) -> None:
    q = np.pi / 3
    xj, _, _ = jcalc(jtype, q)
    expected = xrot(_featherstone_e_rot(jtype, q))
    np.testing.assert_allclose(xj, expected, atol=1e-14)


def test_rz_pi_half_known_answer_first_column() -> None:
    """RBDA spatial_v2: Rz(pi/2) yields E[:, 0] = [0, -1, 0]."""
    xj, _, _ = jcalc("Rz", np.pi / 2)
    e_block = xj[0:3, 0:3]
    np.testing.assert_allclose(e_block[:, 0], np.array([0.0, -1.0, 0.0]), atol=1e-14)


@pytest.mark.parametrize(
    "jtype,axis",
    [
        ("Px", np.array([1.0, 0.0, 0.0])),
        ("Py", np.array([0.0, 1.0, 0.0])),
        ("Pz", np.array([0.0, 0.0, 1.0])),
    ],
)
def test_prismatic_matches_xlt(jtype: str, axis: np.ndarray) -> None:
    q = 0.37
    xj, _, _ = jcalc(jtype, q)
    expected = xlt(q * axis)
    np.testing.assert_allclose(xj, expected, atol=1e-14)


def test_two_link_rz_px_forward_kinematics_against_independent() -> None:
    """Chain jcalc transforms the way xrot/xlt compose for a 2-DoF arm."""
    q_rot = 0.6
    q_slide = 0.25
    link_length = 1.2

    x_j1, _, _ = jcalc("Rz", q_rot)
    x_tree = xlt(np.array([link_length, 0.0, 0.0]))
    x_j2, _, _ = jcalc("Px", q_slide)

    x_from_jcalc = x_j2 @ x_tree @ x_j1

    x_ref = (
        xlt(np.array([q_slide, 0.0, 0.0]))
        @ xlt(np.array([link_length, 0.0, 0.0]))
        @ xrot(_featherstone_e_rot("Rz", q_rot))
    )
    np.testing.assert_allclose(x_from_jcalc, x_ref, atol=1e-14)

    # End-effector x-axis direction in base coordinates (independent SE(3) check).
    e_ref = _featherstone_e_rot("Rz", q_rot)
    axis_in_base = e_ref @ np.array([1.0, 0.0, 0.0])
    # Unit motion along child x after rotation+translations has same direction.
    motion_unit_x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    transformed = x_from_jcalc @ motion_unit_x
    np.testing.assert_allclose(transformed[3:6], axis_in_base, atol=1e-14)
    np.testing.assert_allclose(transformed[0:3], 0.0, atol=1e-14)