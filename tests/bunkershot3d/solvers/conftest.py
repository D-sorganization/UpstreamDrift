"""Fixtures for the F0 solver tests (issue #8611).

Two mesh builders, for two different jobs:

* :func:`box_mesh` takes arbitrary SI dimensions and is used by the
  physics tests, where a 20 x 80 mm sole has to be a 20 x 80 mm sole.
* :func:`dyadic_box_mesh` snaps every coordinate to a multiple of
  ``2 ** -10 m`` (about 0.98 mm).  Dyadic vertices plus dyadic
  power-of-two translations means the transformed coordinates are
  **exactly representable**, so a translation changes no bit of any edge
  difference, normal or area.  That is what lets the metamorphic tests
  assert to 1e-14 instead of to a hand-waved tolerance: the only error
  left is the solver's own arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from bunkershot3d.geometry.mesh import TriangleMesh
from bunkershot3d.sand import PlayingCondition, SandState, playing_condition
from bunkershot3d.solvers import (
    DRFTSolver,
    MaterialResponse,
    RefusalPolicy,
    SurfaceElements,
)

DYADIC_UNIT_M = 2.0**-10
"""Grid the metamorphic fixtures snap to: 2^-10 m, about 0.98 mm."""

_BOX_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int64,
)


def box_mesh(
    size_x_m: float,
    size_y_m: float,
    size_z_m: float,
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> TriangleMesh:
    """A closed, outward-wound axis-aligned box."""
    half = np.array([size_x_m, size_y_m, size_z_m], dtype=np.float64) / 2.0
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return TriangleMesh(signs * half + np.asarray(centre_m), _BOX_FACES)


def dyadic_box_mesh(
    half_x_units: int = 10,
    half_y_units: int = 40,
    half_z_units: int = 2,
    centre_units: tuple[int, int, int] = (0, 0, -40),
) -> TriangleMesh:
    """A box whose every vertex coordinate is an exact multiple of 2^-10 m.

    Args:
        half_x_units: Half-extent along x, in units of ``2 ** -10 m``.
        half_y_units: Half-extent along y.
        half_z_units: Half-extent along z.
        centre_units: Box centre, in the same units.

    Returns:
        A watertight box with exactly representable coordinates.
    """
    return box_mesh(
        2 * half_x_units * DYADIC_UNIT_M,
        2 * half_y_units * DYADIC_UNIT_M,
        2 * half_z_units * DYADIC_UNIT_M,
        centre_m=tuple(unit * DYADIC_UNIT_M for unit in centre_units),  # type: ignore[arg-type]
    )


def reflected_elements(
    elements: SurfaceElements, mirror: NDArray[np.float64]
) -> SurfaceElements:
    """Mirror a body, keeping its normals outward.

    Deriving normals from mirrored triangles would flip them inward,
    because a mirror has determinant -1.  Mirroring the element arrays
    directly keeps ``n -> M n``, which is the physically correct image of
    an outward normal.
    """
    return SurfaceElements(
        elements.centroids_m @ mirror.T,
        elements.normals @ mirror.T,
        elements.areas_m2,
    )


@pytest.fixture
def firm_sand() -> SandState:
    """A firm USGA-spec bunker, the default design condition."""
    return playing_condition(PlayingCondition.FIRM)


@pytest.fixture
def material(firm_sand: SandState) -> MaterialResponse:
    """The F0 material response for :func:`firm_sand`."""
    return MaterialResponse.from_sand_state(firm_sand)


@pytest.fixture
def solver(material: MaterialResponse) -> DRFTSolver:
    """A reporting solver: verdicts are inspected rather than raised.

    Bunker-shot speeds are far outside the published envelope by design,
    so the tests that are *about* the physics use the reporting policy
    and the tests that are about refusal set the strict policy
    explicitly.
    """
    return DRFTSolver(material=material, refusal_policy=RefusalPolicy.REPORT)


@pytest.fixture
def sole_elements() -> SurfaceElements:
    """A 20 x 80 x 4 mm sole plate, buried 40 mm deep."""
    return SurfaceElements.from_mesh(
        box_mesh(0.020, 0.080, 0.004, centre_m=(0.0, 0.0, -0.040))
    )


@pytest.fixture
def dyadic_elements() -> SurfaceElements:
    """A dyadic plate for the bit-exact metamorphic transforms."""
    return SurfaceElements.from_mesh(dyadic_box_mesh())
