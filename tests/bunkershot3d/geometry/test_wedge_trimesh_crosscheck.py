"""Independent cross-check of the native mass properties against trimesh.

`trimesh` is deliberately NOT a dependency of this repo (ADR-0032: an OEM
tool must be able to verify its own numbers), so this module skips
cleanly when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.geometry.lofting import build_wedge_mesh
from bunkershot3d.geometry.mass_properties import compute_mass_properties
from bunkershot3d.geometry.solids import box_mesh, cylinder_mesh, icosphere_mesh

from .conftest import build_reference_wedge

pytestmark = pytest.mark.unit

trimesh = pytest.importorskip("trimesh", reason="trimesh is an optional cross-check")


def _as_trimesh(mesh):  # type: ignore[no-untyped-def]
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )


def _cases():  # type: ignore[no-untyped-def]
    return {
        "box": box_mesh(0.03, 0.02, 0.01, centre=np.array([0.05, -0.02, 0.01])),
        "sphere": icosphere_mesh(0.02, subdivisions=3),
        "cylinder": cylinder_mesh(0.015, 0.04, n_segments=48),
        "wedge": build_wedge_mesh(build_reference_wedge()),
    }


@pytest.mark.parametrize("name", ["box", "sphere", "cylinder", "wedge"])
def test_volume_centroid_and_inertia_match_trimesh(name: str) -> None:
    mesh = _cases()[name]
    reference = _as_trimesh(mesh)
    assert reference.is_watertight

    density = 7800.0
    reference.density = density
    props = compute_mass_properties(mesh, density_kg_m3=density)

    assert props.volume_m3 == pytest.approx(float(reference.volume), rel=1e-10)
    np.testing.assert_allclose(
        props.centroid_m, np.asarray(reference.center_mass), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        props.inertia_kg_m2,
        np.asarray(reference.moment_inertia),
        rtol=1e-8,
        atol=1e-14,
    )
