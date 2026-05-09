"""Unit tests for :class:`ProcrustesAnisotropicFitter`."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.shared.python.body_part_viz import (
    BindingKind,
    MarkerBinding,
    ShapeFitter,
)
from src.shared.python.body_part_viz.fitters import ProcrustesAnisotropicFitter

from ._stubs import StubShape


def _binding(n_markers: int = 4) -> MarkerBinding:
    names = tuple(f"m{j}" for j in range(n_markers))
    return MarkerBinding(kind=BindingKind.CLUSTER, marker_names=names)


def test_implements_shape_fitter_protocol() -> None:
    assert isinstance(ProcrustesAnisotropicFitter(), ShapeFitter)


def test_recovers_anisotropic_scale_within_tolerance() -> None:
    # Axis-aligned rest cluster: principal axes = world axes, so the
    # Kabsch solution under anisotropic scaling is the identity. That
    # decouples rotation and scale and lets us recover s exactly.
    rest = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    scale_known = np.array([1.0, 2.0, 0.5])
    scaled = rest * scale_known

    cluster = np.stack([rest, scaled], axis=0)  # (T=2, N=6, 3)
    markers = {f"m{j}": cluster[:, j, :] for j in range(6)}
    binding = _binding(n_markers=6)

    fitted = ProcrustesAnisotropicFitter().fit(StubShape(), binding, markers)

    # Frame 0 = rest → unit scale, identity rotation.
    assert np.allclose(fitted.scale[0], 1.0, atol=1e-6)
    assert np.allclose(fitted.rotation_matrix[0], np.eye(3), atol=1e-9)
    # Frame 1 = anisotropically scaled → recover s.
    assert np.allclose(fitted.scale[1], scale_known, atol=1e-6)


def test_too_few_markers_logs_warning_and_falls_back_to_kabsch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rest = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cluster = np.stack([rest, rest * 2.0], axis=0)
    markers = {f"m{j}": cluster[:, j, :] for j in range(3)}
    binding = _binding(n_markers=3)

    with caplog.at_level(
        logging.WARNING,
        logger=("src.shared.python.body_part_viz.fitters.procrustes_anisotropic"),
    ):
        fitted = ProcrustesAnisotropicFitter().fit(StubShape(), binding, markers)

    assert any("falling back" in rec.message for rec in caplog.records)
    # Fallback ⇒ unit scale even though the cluster was scaled by 2.
    assert np.allclose(fitted.scale, 1.0)


def test_wrong_binding_kind_raises_type_error() -> None:
    binding = MarkerBinding(kind=BindingKind.BETWEEN_TWO, marker_names=("a", "b"))
    with pytest.raises(TypeError, match="CLUSTER"):
        ProcrustesAnisotropicFitter().fit(
            StubShape(),
            binding,
            {"a": np.zeros((1, 3)), "b": np.zeros((1, 3))},
        )


def test_nan_frame_marked_invalid() -> None:
    rest = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
        ]
    )
    cluster = np.stack([rest, rest, rest], axis=0)
    markers = {f"m{j}": cluster[:, j, :] for j in range(4)}
    markers["m2"][1] = np.nan

    fitted = ProcrustesAnisotropicFitter().fit(StubShape(), _binding(), markers)

    assert bool(fitted.valid_mask[0])
    assert not bool(fitted.valid_mask[1])
    assert bool(fitted.valid_mask[2])


def test_all_invalid_frames_returns_default() -> None:
    n = 3
    markers = {f"m{j}": np.full((n, 3), np.nan) for j in range(4)}
    fitted = ProcrustesAnisotropicFitter().fit(StubShape(), _binding(), markers)
    assert not bool(fitted.valid_mask.any())


def test_collinear_axis_keeps_unit_scale_for_zero_norm_axis() -> None:
    # Cluster lying in the xy-plane → centred z-component is identically zero.
    rest = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    scale_known = np.array([2.0, 0.5, 1.0])
    scaled = rest * scale_known
    cluster = np.stack([rest, scaled], axis=0)
    markers = {f"m{j}": cluster[:, j, :] for j in range(4)}

    fitted = ProcrustesAnisotropicFitter().fit(StubShape(), _binding(), markers)

    assert np.allclose(fitted.scale[1, :2], scale_known[:2], atol=1e-6)
    # Z-axis is degenerate (zero variance) — fitter must keep it positive.
    assert fitted.scale[1, 2] > 0.0
