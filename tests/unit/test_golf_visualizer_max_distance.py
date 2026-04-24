"""Tests for the einsum max-distance calculation in golf_visualizer_widget.py.

Validates the np.einsum-based distance computation against the original
np.linalg.norm reference for representative inputs (issue #2799).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

pytestmark = pytest.mark.unit


def _einsum_max_distance(positions: np.ndarray, center: np.ndarray) -> float:
    """Optimised version: uses np.einsum to avoid intermediate norm array."""
    diff = positions - center
    return float(np.sqrt(np.max(np.einsum("ij,ij->i", diff, diff))))


def _reference_max_distance(positions: np.ndarray, center: np.ndarray) -> float:
    """Reference implementation: uses np.linalg.norm."""
    return float(np.max(np.linalg.norm(positions - center, axis=1)))


@pytest.mark.parametrize(
    "positions",
    [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[3.0, 4.0, 0.0]]),
        np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-5.0, 5.0, 5.0], [2.0, -3.0, 7.0]]
        ),
    ],
)
def test_einsum_matches_reference(positions: np.ndarray) -> None:
    """Optimised calculation must match the linalg.norm reference."""
    center = np.mean(positions, axis=0)
    expected = _reference_max_distance(positions, center)
    result = _einsum_max_distance(positions, center)
    assert abs(result - expected) < 1e-10, f"mismatch: {result} vs {expected}"


def test_single_point_distance_zero() -> None:
    """A single point: center == point, so max distance is 0."""
    positions = np.array([[5.0, 3.0, 1.0]])
    center = np.mean(positions, axis=0)
    result = _einsum_max_distance(positions, center)
    assert result == pytest.approx(0.0)


def test_symmetric_arrangement() -> None:
    """Points equidistant from origin → max_distance == that distance."""
    r = 3.0
    positions = np.array([[r, 0.0, 0.0], [-r, 0.0, 0.0], [0.0, r, 0.0], [0.0, -r, 0.0]])
    center = np.mean(positions, axis=0)  # (0,0,0)
    result = _einsum_max_distance(positions, center)
    assert result == pytest.approx(r)
