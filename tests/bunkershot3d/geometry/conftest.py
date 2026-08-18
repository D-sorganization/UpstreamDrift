"""Shared fixtures for the parametric wedge-geometry tests (issue #8609)."""

from __future__ import annotations

import pytest

from bunkershot3d.geometry.bounce import GeometricBounce
from bunkershot3d.geometry.mesh import TriangleMesh
from bunkershot3d.geometry.wedge import WedgeGeometry


def build_reference_wedge(**overrides: object) -> WedgeGeometry:
    """A mid-bounce 56 deg wedge sitting inside the patent's preferred bands."""
    params: dict[str, object] = {
        "loft_deg": 56.0,
        "lie_deg": 64.0,
        "geometric_bounce": GeometricBounce(21.0),
        "sole_width_mm": 21.0,
        "entry_height_mm": 3.5,
        "leading_edge_radius_mm": 7.5,
        "trailing_edge_radius_mm": 42.0,
        "sole_camber_area_mm2": 55.0,
        "centre_rocker_radius_mm": 250.0,
        "heel_rocker_radius_mm": 90.0,
        "toe_rocker_radius_mm": 130.0,
        "trailing_relief_fraction": 0.15,
        "heel_relief_fraction": 0.20,
        "toe_relief_fraction": 0.10,
        "face_progression_mm": 2.0,
        "blade_length_mm": 78.0,
        "face_height_mm": 38.0,
        "topline_width_mm": 4.0,
        "head_mass_g": 304.0,
    }
    params.update(overrides)
    return WedgeGeometry.from_millimetres(**params)  # type: ignore[arg-type]


@pytest.fixture
def wedge() -> WedgeGeometry:
    return build_reference_wedge()


@pytest.fixture(scope="module")
def wedge_mesh() -> TriangleMesh:
    from bunkershot3d.geometry.lofting import build_wedge_mesh

    return build_wedge_mesh(build_reference_wedge())
