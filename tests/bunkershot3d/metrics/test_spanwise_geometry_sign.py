"""The spanwise sign, pinned against the real lofted mesh (issue #8699).

``test_spanwise`` checks the arithmetic on synthetic soles: *given* that the
toe half carries less load, the balance goes negative. That leaves one link in
the chain unpinned -- that grinding **toe** relief into a real head is what
makes the toe half carry less -- and it is the link where a sign error would
actually hide, because it depends on which way body ``+y`` runs in the mesh
:mod:`bunkershot3d.geometry.lofting` builds.

So this module lofts the head. Four wedges differing **only** in their relief
fractions are meshed, their sole elements taken, and each element loaded at one
uniform pressure, so the load is the sole's own area distribution and nothing
else. Under that load the metric must report:

======================= ==========================================
heel / toe relief       spanwise balance
======================= ==========================================
0.0 / 0.0               ~0 -- an unrelieved sole is symmetric
0.0 / 0.3               **negative** -- toe relief loads the heel
0.3 / 0.0               **positive** -- heel relief loads the toe
0.3 / 0.3               ~0 again -- the two cancel
======================= ==========================================

No solver runs here: a uniform pressure is not a strike, and this module claims
nothing about one. It claims only that the *geometry* and the *metric* agree on
which end of the blade is which. Under F0's real, non-uniform sand loading the
same four heads read -0.055, -0.162, -0.013 and -0.120 -- same directions,
larger magnitudes, and an unrelieved sole that is *not* balanced once it is
delivered face-open into sand.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from bunkershot3d.geometry import CamberFit, WedgeGeometry, get_preset, loft_wedge
from bunkershot3d.metrics import SoleLoadTrace, spanwise_load
from bunkershot3d.solvers import FidelityTier, SurfaceElements

pytestmark = pytest.mark.unit

#: Grind the reliefs are varied against. Any preset would do; this one is a
#: published retail wedge rather than a patent example.
BASE_PRESET = "sm9_58_m"

#: Loft resolution. Coarse on purpose -- the sole still resolves ~830 elements
#: across 17 spanwise stations, which is far more than the metric's floor, and
#: each loft solves a camber segment by root-finding per station.
N_PROFILE_POINTS = 25
N_STATIONS = 17

#: Spanwise bins. 17 stations support 8 at two stations per bin.
N_BINS = 8

#: The uniform pressure every sole element is loaded at [Pa]. Its value is
#: irrelevant: the balance is a ratio, so any positive constant gives the same
#: answer. Stated rather than hidden so nobody reads it as a sand pressure.
UNIFORM_PRESSURE_PA = 1.0e5

#: How close to zero an unrelieved sole's balance has to read. Two orders of
#: magnitude below the shift 0.3 of relief produces, so "symmetric" and
#: "relieved" cannot be confused.
SYMMETRY_TOLERANCE = 1.0e-3


def _balance(geometry: WedgeGeometry) -> float:
    """Return the spanwise balance of one head under uniform sole pressure.

    Args:
        geometry: The design vector to loft.

    Returns:
        The signed heel/toe balance; negative is toward the heel.
    """
    mesh = loft_wedge(
        geometry,
        n_profile_points=N_PROFILE_POINTS,
        n_stations=N_STATIONS,
        camber_fit=CamberFit.NEAREST,
    ).mesh
    elements = SurfaceElements.from_mesh(mesh)
    sole = elements.normals[:, 2] < 0.0
    areas = elements.areas_m2[sole]
    load = SoleLoadTrace(
        time_s=np.array([0.0, 0.01]),
        element_centroid_body_m=elements.centroids_m[sole],
        element_area_m2=areas,
        element_normal_force_N=np.tile(UNIFORM_PRESSURE_PA * areas, (2, 1)),
    )
    result = spanwise_load(load, n_bins=N_BINS, fidelity_tier=FidelityTier.F0)
    return result.heel_toe_balance


@pytest.fixture(scope="module")
def balances() -> dict[tuple[float, float], float]:
    """Balance of the four relief combinations, lofted once for the module."""
    base = get_preset(BASE_PRESET).geometry
    return {
        (heel, toe): _balance(
            dataclasses.replace(
                base, heel_relief_fraction=heel, toe_relief_fraction=toe
            )
        )
        for heel, toe in ((0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (0.3, 0.3))
    }


class TestSpanwiseSignAgainstTheMesh:
    """Which end of the real blade the metric calls the heel."""

    def test_an_unrelieved_sole_is_symmetric(self, balances) -> None:
        """No relief, uniform pressure: the lofted sole balances about zero."""
        assert abs(balances[0.0, 0.0]) < SYMMETRY_TOLERANCE

    def test_toe_relief_moves_the_load_toward_the_heel(self, balances) -> None:
        """The sign that matters. Toe relief removes toe sole area."""
        assert balances[0.0, 0.3] < -SYMMETRY_TOLERANCE

    def test_heel_relief_moves_the_load_toward_the_toe(self, balances) -> None:
        """The mirror image, so the result is a direction and not an offset."""
        assert balances[0.3, 0.0] > SYMMETRY_TOLERANCE

    def test_the_two_reliefs_are_opposite_and_comparable(self, balances) -> None:
        """Equal relief at both ends is symmetric again: they cancel."""
        assert balances[0.0, 0.3] == pytest.approx(-balances[0.3, 0.0], rel=0.05)
        assert abs(balances[0.3, 0.3]) < SYMMETRY_TOLERANCE

    def test_relief_moves_the_balance_further_than_symmetry_tolerates(
        self, balances
    ) -> None:
        """The effect is real, not numerical dust: an order above the floor."""
        assert abs(balances[0.0, 0.3]) > 10.0 * abs(balances[0.0, 0.0])
