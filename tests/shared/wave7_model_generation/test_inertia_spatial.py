"""Tests for spatial inertia utilities."""

from __future__ import annotations

import numpy as np
import pytest
from model_generation.inertia.spatial import (
    composite_rigid_body_inertia,
    mcI,
    spatial_inertia_to_urdf,
    spatial_transform,
    transform_spatial_inertia,
    urdf_to_spatial_inertia,
)


def test_mcI_zero_com_block_structure() -> None:
    inertia = np.diag([1.0, 2.0, 3.0])
    M = mcI(5.0, np.zeros(3), inertia)
    assert M.shape == (6, 6)
    # Upper-left equals inertia when COM is zero
    assert np.allclose(M[:3, :3], inertia)
    # Lower-right is m*I3
    assert np.allclose(M[3:, 3:], 5.0 * np.eye(3))
    # Off-diagonal blocks zero
    assert np.allclose(M[:3, 3:], 0.0)
    assert np.allclose(M[3:, :3], 0.0)


def test_mcI_wrong_com_shape() -> None:
    with pytest.raises(ValueError):
        mcI(1.0, np.array([1.0, 2.0]), np.eye(3))


def test_mcI_wrong_inertia_shape() -> None:
    with pytest.raises(ValueError):
        mcI(1.0, np.zeros(3), np.eye(4))


def test_mcI_symmetrizes_input() -> None:
    # Slightly non-symmetric input gets symmetrized
    inertia = np.array([[1.0, 0.1, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    M = mcI(1.0, np.zeros(3), inertia)
    expected_sym = 0.5 * (inertia + inertia.T)
    assert np.allclose(M[:3, :3], expected_sym)


def test_spatial_to_urdf_roundtrip() -> None:
    mass = 2.5
    com = (0.1, -0.2, 0.3)
    ixx, iyy, izz = 1.0, 2.0, 3.0
    ixy, ixz, iyz = 0.05, 0.0, -0.02
    sI = urdf_to_spatial_inertia(mass, com, ixx, iyy, izz, ixy, ixz, iyz)
    out = spatial_inertia_to_urdf(sI)
    assert out["mass"] == pytest.approx(mass)
    assert out["com"][0] == pytest.approx(com[0])
    assert out["com"][1] == pytest.approx(com[1])
    assert out["com"][2] == pytest.approx(com[2])
    assert out["ixx"] == pytest.approx(ixx)
    assert out["iyy"] == pytest.approx(iyy)
    assert out["izz"] == pytest.approx(izz)
    assert out["ixy"] == pytest.approx(ixy)
    assert out["iyz"] == pytest.approx(iyz)


def test_spatial_to_urdf_bad_shape() -> None:
    with pytest.raises(ValueError):
        spatial_inertia_to_urdf(np.eye(4))


def test_spatial_to_urdf_invalid_mass() -> None:
    bad = np.zeros((6, 6))
    with pytest.raises(ValueError):
        spatial_inertia_to_urdf(bad)


def test_spatial_transform_identity() -> None:
    X = spatial_transform(np.eye(3), np.zeros(3))
    assert np.allclose(X[:3, :3], np.eye(3))
    assert np.allclose(X[3:, 3:], np.eye(3))
    assert np.allclose(X[3:, :3], 0.0)


def test_spatial_transform_bad_rotation() -> None:
    with pytest.raises(ValueError):
        spatial_transform(np.eye(4), np.zeros(3))


def test_spatial_transform_bad_translation() -> None:
    with pytest.raises(ValueError):
        spatial_transform(np.eye(3), np.zeros(4))


def test_transform_spatial_inertia_identity() -> None:
    inertia = np.diag([1, 2, 3, 4, 5, 6]).astype(np.float64)
    X = np.eye(6)
    out = transform_spatial_inertia(inertia, X)
    assert np.allclose(out, inertia)


def test_transform_spatial_inertia_bad_shape() -> None:
    with pytest.raises(ValueError):
        transform_spatial_inertia(np.eye(5), np.eye(6))


def test_composite_rigid_body_inertia_sum() -> None:
    inertia = mcI(1.0, np.zeros(3), np.eye(3))
    result = composite_rigid_body_inertia([(inertia, np.eye(6)), (inertia, np.eye(6))])
    assert np.allclose(result, 2 * inertia)


def test_composite_rigid_body_inertia_empty() -> None:
    result = composite_rigid_body_inertia([])
    assert np.allclose(result, np.zeros((6, 6)))
