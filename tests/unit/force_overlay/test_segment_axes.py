"""Tests for the pure-numpy segment axis helper (FTO-11, #11296)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.force_overlay.segment_axes import (
    SegmentAxis,
    segment_axes_from_joint_tree,
)

pytestmark = pytest.mark.unit


def test_chain_produces_axes_and_skips_leaf() -> None:
    origins = {
        "j1": np.array([0.0, 0.0, 0.0]),
        "j2": np.array([0.0, 0.0, -1.0]),
        "j3": np.array([0.0, 0.0, -2.0]),
    }
    body_of_joint = {"j1": "upper", "j2": "lower", "j3": "hand"}
    child_joint_of = {"upper": "j2", "lower": "j3"}
    axes, skipped = segment_axes_from_joint_tree(origins, child_joint_of, body_of_joint)
    assert sorted(a.body for a in axes) == ["lower", "upper"]
    assert isinstance(axes[0], SegmentAxis)
    upper = next(a for a in axes if a.body == "upper")
    np.testing.assert_allclose(upper.proximal_m, (0, 0, 0))
    np.testing.assert_allclose(upper.distal_m, (0, 0, -1))
    assert skipped == ("hand",)


def test_branching_body_is_skipped() -> None:
    origins = {k: np.zeros(3) + i for i, k in enumerate(("a", "b", "c"))}
    axes, skipped = segment_axes_from_joint_tree(
        origins, {"root": ["b", "c"]}, {"a": "root", "b": "x", "c": "y"}
    )
    assert "root" in skipped
    assert all(ax.body != "root" for ax in axes)


def test_missing_joint_origin_is_skipped() -> None:
    origins = {"j1": np.zeros(3)}
    axes, skipped = segment_axes_from_joint_tree(
        origins, {"upper": "j2"}, {"j1": "upper", "j2": "lower"}
    )
    assert axes == ()
    assert set(skipped) == {"upper", "lower"}


def test_invalid_origin_raises() -> None:
    with pytest.raises(ValueError):
        segment_axes_from_joint_tree({"j1": np.array([np.nan, 0, 0])}, {}, {"j1": "b"})
    with pytest.raises(TypeError):
        segment_axes_from_joint_tree([], {}, {})  # type: ignore[arg-type]
