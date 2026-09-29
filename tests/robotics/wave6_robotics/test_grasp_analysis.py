"""Tests for grasp_analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.contact.grasp_analysis import (
    check_force_closure,
    compute_contact_wrench_cone,
    compute_grasp_matrix,
    compute_grasp_quality,
    required_contact_forces,
)
from src.robotics.core.types import ContactState


def _contact(idx: int, pos, normal=None, mu=0.5) -> ContactState:
    return ContactState(
        contact_id=idx,
        body_a="finger",
        body_b="object",
        position=np.asarray(pos, dtype=float),
        normal=np.asarray(normal if normal is not None else [0, 0, 1.0], dtype=float),
        normal_force=1.0,
        friction_coefficient=mu,
    )


def _antipodal_contacts() -> list[ContactState]:
    return [
        _contact(0, [1.0, 0.0, 0.0], normal=[-1.0, 0.0, 0.0]),
        _contact(1, [-1.0, 0.0, 0.0], normal=[1.0, 0.0, 0.0]),
        _contact(2, [0.0, 1.0, 0.0], normal=[0.0, -1.0, 0.0]),
        _contact(3, [0.0, -1.0, 0.0], normal=[0.0, 1.0, 0.0]),
    ]


def test_compute_grasp_matrix_shape() -> None:
    cs = _antipodal_contacts()
    G = compute_grasp_matrix(cs)
    assert G.shape == (6, 3 * len(cs))


def test_compute_grasp_matrix_object_frame() -> None:
    cs = _antipodal_contacts()
    G = compute_grasp_matrix(cs, object_frame=np.zeros(3))
    assert G.shape[0] == 6


def test_compute_grasp_matrix_empty_raises() -> None:
    with pytest.raises(ValueError, match="At least one"):
        compute_grasp_matrix([])


def test_check_force_closure_one_contact_raises() -> None:
    with pytest.raises(ValueError, match="At least 2"):
        check_force_closure([_contact(0, [0, 0, 0])])


def test_check_force_closure_antipodal() -> None:
    cs = _antipodal_contacts()
    has, margin = check_force_closure(cs, num_cone_faces=8)
    assert isinstance(has, bool)
    assert isinstance(margin, float)


def test_compute_grasp_quality_min_singular() -> None:
    cs = _antipodal_contacts()
    q = compute_grasp_quality(cs, metric="min_singular_value")
    assert q >= 0


def test_compute_grasp_quality_volume() -> None:
    cs = _antipodal_contacts()
    q = compute_grasp_quality(cs, metric="volume")
    assert q >= 0


def test_compute_grasp_quality_isotropy() -> None:
    cs = _antipodal_contacts()
    q = compute_grasp_quality(cs, metric="isotropy")
    assert 0 <= q <= 1


def test_compute_grasp_quality_unknown_metric() -> None:
    cs = _antipodal_contacts()
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_grasp_quality(cs, metric="nope")


def test_compute_grasp_quality_empty_contacts() -> None:
    with pytest.raises(ValueError, match="At least one"):
        compute_grasp_quality([])


def test_compute_grasp_quality_degenerate_single_contact() -> None:
    q = compute_grasp_quality([_contact(0, [0, 0, 0])], metric="min_singular_value")
    assert q >= 0


def test_compute_contact_wrench_cone() -> None:
    cs = _antipodal_contacts()
    W = compute_contact_wrench_cone(cs, num_faces=4)
    assert W.shape[0] == 6
    assert W.shape[1] == len(cs) * 4


def test_required_contact_forces_returns_array_or_none() -> None:
    cs = _antipodal_contacts()
    desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    f = required_contact_forces(cs, desired)
    # Should be either valid (3*n,) vector or None
    assert f is None or f.shape == (3 * len(cs),)
