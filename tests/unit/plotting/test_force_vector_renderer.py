"""Tests for force vector visualization renderer.

Covers:
- plot_joint_force_vectors (total forces at joints)
- plot_ztcf_force_vectors (ZTCF passive component)
- plot_force_delta_vectors (active control component)
- plot_force_decomposition (side-by-side total/ZTCF/delta)
- Precondition enforcement and edge cases
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.shared.python.core.contracts.exceptions import PreconditionError
from src.shared.python.plotting.renderers.force_vectors import (
    ForceVectorRenderer,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_data_manager() -> MagicMock:
    """Create a mock DataManager with force data."""
    dm = MagicMock()
    dm.joint_names = ["Shoulder", "Elbow", "Wrist"]
    dm.get_joint_name = lambda idx: (
        ["Shoulder", "Elbow", "Wrist"][idx] if idx < 3 else f"Joint {idx}"
    )

    times = np.linspace(0, 1, 50)
    # Joint positions: (50, 3, 3) — 50 frames, 3 joints, 3D
    positions = np.random.default_rng(42).uniform(-1, 1, (50, 3, 3))
    # Force vectors: (50, 3, 3) — 50 frames, 3 joints, 3D
    forces = np.random.default_rng(42).uniform(-100, 100, (50, 3, 3))

    def mock_get_series(field_name: str):
        if field_name == "joint_world_positions":
            return times, positions
        if field_name == "joint_forces":
            return times, forces
        if field_name == "ztcf_joint_forces":
            return times, forces * 0.3  # ZTCF is a fraction of total
        return np.array([]), np.array([])

    dm.get_series = mock_get_series
    return dm


@pytest.fixture
def renderer(mock_data_manager: MagicMock) -> ForceVectorRenderer:
    return ForceVectorRenderer(mock_data_manager)


@pytest.fixture
def fig() -> Figure:
    return Figure(figsize=(10, 8))


# ============================================================================
# plot_joint_force_vectors
# ============================================================================


class TestPlotJointForceVectors:
    """Tests for total joint force vector plotting."""

    def test_creates_axes_on_figure(
        self, renderer: ForceVectorRenderer, fig: Figure
    ) -> None:
        renderer.plot_joint_force_vectors(fig, frame_idx=0)
        assert len(fig.get_axes()) > 0

    def test_with_explicit_data(
        self, renderer: ForceVectorRenderer, fig: Figure
    ) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        forces = np.array([[10, 0, 0], [0, 20, 0], [0, 0, 30]], dtype=float)
        renderer.plot_joint_force_vectors(fig, positions=positions, forces=forces)
        axes = fig.get_axes()
        assert len(axes) == 1

    def test_scale_parameter_affects_arrows(
        self, renderer: ForceVectorRenderer, fig: Figure
    ) -> None:
        positions = np.array([[0, 0, 0]], dtype=float)
        forces = np.array([[100, 0, 0]], dtype=float)
        # Should not raise with different scales
        renderer.plot_joint_force_vectors(
            fig, positions=positions, forces=forces, scale=0.001
        )
        assert len(fig.get_axes()) > 0

    def test_none_figure_raises(self, renderer: ForceVectorRenderer) -> None:
        with pytest.raises(AssertionError):
            renderer.plot_joint_force_vectors(None, frame_idx=0)  # type: ignore[arg-type]


# ============================================================================
# plot_ztcf_force_vectors
# ============================================================================


class TestPlotZTCFForceVectors:
    """Tests for ZTCF force vector plotting."""

    def test_creates_axes(self, renderer: ForceVectorRenderer, fig: Figure) -> None:
        renderer.plot_ztcf_force_vectors(fig, frame_idx=0)
        assert len(fig.get_axes()) > 0

    def test_with_explicit_data(
        self, renderer: ForceVectorRenderer, fig: Figure
    ) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        ztcf_forces = np.array([[5, 3, 0], [2, 8, 0]], dtype=float)
        renderer.plot_ztcf_force_vectors(
            fig, positions=positions, ztcf_forces=ztcf_forces
        )
        assert len(fig.get_axes()) == 1


# ============================================================================
# plot_force_delta_vectors
# ============================================================================


class TestPlotForceDeltaVectors:
    """Tests for delta (total - ZTCF) force vector plotting."""

    def test_creates_axes(self, renderer: ForceVectorRenderer, fig: Figure) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        total = np.array([[10, 20, 0], [30, 40, 0]], dtype=float)
        ztcf = np.array([[3, 5, 0], [10, 15, 0]], dtype=float)
        renderer.plot_force_delta_vectors(
            fig, positions=positions, total_forces=total, ztcf_forces=ztcf
        )
        assert len(fig.get_axes()) > 0

    def test_mismatched_shapes_handled(
        self, renderer: ForceVectorRenderer, fig: Figure
    ) -> None:
        """Mismatched total/ztcf shapes should raise or show error."""
        positions = np.array([[0, 0, 0]], dtype=float)
        total = np.array([[10, 20, 0]], dtype=float)
        ztcf = np.array([[3, 5, 0], [1, 2, 0]], dtype=float)
        with pytest.raises((PreconditionError, ValueError, AssertionError)):
            renderer.plot_force_delta_vectors(
                fig, positions=positions, total_forces=total, ztcf_forces=ztcf
            )


# ============================================================================
# plot_force_decomposition
# ============================================================================


class TestPlotForceDecomposition:
    """Tests for combined total/ZTCF/delta decomposition subplot."""

    def test_creates_three_subplots(
        self, renderer: ForceVectorRenderer, fig: Figure
    ) -> None:
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        total = np.array([[10, 20, 0], [30, 40, 0]], dtype=float)
        ztcf = np.array([[3, 5, 0], [10, 15, 0]], dtype=float)
        renderer.plot_force_decomposition(
            fig, positions=positions, total_forces=total, ztcf_forces=ztcf
        )
        # Should have 3 subplots: total, ztcf, delta
        assert len(fig.get_axes()) == 3

    def test_empty_data_shows_message(
        self, renderer: ForceVectorRenderer, fig: Figure
    ) -> None:
        renderer.plot_force_decomposition(
            fig,
            positions=np.empty((0, 3)),
            total_forces=np.empty((0, 3)),
            ztcf_forces=np.empty((0, 3)),
        )
        # Should still create axes (with "no data" message)
        assert len(fig.get_axes()) > 0
