"""Closed grip-cylinder mesh and its validator (issue #11739, OSV-7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.grip_contact.grip_mesh import (
    capped_cylinder_mesh,
    validate_closed_mesh,
    write_obj,
)

pytestmark = pytest.mark.unit

X, Y, Z = np.eye(3)


def _mesh(**kw):
    args = {
        "radius_m": 0.0127,
        "axis_point_m": np.array([0.1, 0.0, 0.0]),
        "axis": X,
        "radial": Y,
        "axial_range_m": (-0.05, 0.05),
        "segments": 32,
        "ring_pitch_m": 5e-3,
        **kw,
    }
    return capped_cylinder_mesh(**args)


def test_capped_cylinder_is_closed_with_cylinder_volume() -> None:
    v, f = _mesh()
    validate_closed_mesh(v, f)
    a, b, c = (v[f[:, k]] for k in range(3))
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0
    # polygonal prism: area (n/2) r^2 sin(2 pi / n) times the length
    expected = 0.5 * 32 * 0.0127**2 * math.sin(2 * math.pi / 32) * 0.1
    assert volume == pytest.approx(expected, rel=1e-9)


def test_mesh_follows_axis_point_and_range() -> None:
    v, _ = _mesh()
    assert v[:, 0].min() == pytest.approx(0.05)
    assert v[:, 0].max() == pytest.approx(0.15)
    radial = np.linalg.norm(v[:-2, 1:], axis=1)
    np.testing.assert_allclose(radial, 0.0127, atol=1e-12)


def test_open_tube_is_rejected_with_clear_message() -> None:
    v, f = _mesh()
    sides = f[: -2 * 32]  # drop both cap fans -> uncapped tube
    with pytest.raises(ValueError, match="not closed"):
        validate_closed_mesh(v, sides)


def test_inward_normals_are_rejected() -> None:
    v, f = _mesh()
    with pytest.raises(ValueError, match="inward"):
        validate_closed_mesh(v, f[:, ::-1])


def test_non_manifold_and_bad_indices_are_rejected() -> None:
    v, f = _mesh()
    with pytest.raises(ValueError, match="out of range"):
        validate_closed_mesh(v, f + 10_000)
    with pytest.raises(ValueError, match="directed edge"):
        validate_closed_mesh(v, np.vstack([f, f[:1]]))
    bad = f.copy()
    bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError, match="degenerate"):
        validate_closed_mesh(v, bad)
    with pytest.raises(ValueError, match="shape"):
        validate_closed_mesh(v[:, :2], f)


@pytest.mark.parametrize(
    "kw",
    [
        {"radius_m": 0.0},
        {"axial_range_m": (0.1, 0.1)},
        {"segments": 4},
        {"ring_pitch_m": -1.0},
        {"radial": X},
    ],
)
def test_builder_preconditions(kw) -> None:
    with pytest.raises(ValueError):
        _mesh(**kw)


def test_write_obj_round_trip_and_refuses_open_mesh(tmp_path) -> None:
    v, f = _mesh()
    path = tmp_path / "grip.obj"
    write_obj(path, v, f)
    lines = path.read_text().splitlines()
    assert sum(s.startswith("v ") for s in lines) == len(v)
    assert sum(s.startswith("f ") for s in lines) == len(f)
    with pytest.raises(ValueError, match="not closed"):
        write_obj(tmp_path / "open.obj", v, f[: -2 * 32])
    assert not (tmp_path / "open.obj").exists()
