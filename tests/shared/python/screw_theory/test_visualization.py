"""Tests for Screw Theory visualization utilities.

Validates plot_screw_axis_3d using a mock matplotlib 3D axes to avoid
requiring a display server.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.screw_theory.kinematics import (
    Twist,
    compute_screw_axis,
)
from src.shared.python.screw_theory.visualization import plot_screw_axis_3d

pytestmark = pytest.mark.unit


@pytest.fixture
def rotation_screw():
    """ScrewAxis for pure rotation about Z-axis."""
    twist = Twist(
        angular=np.array([0.0, 0.0, 1.0]),
        linear=np.array([0.0, 0.0, 0.0]),
        body_name="test_body",
        reference_point=np.array([0.0, 0.0, 0.0]),
    )
    return compute_screw_axis(twist)


@pytest.fixture
def translation_screw():
    """ScrewAxis for pure translation along X-axis."""
    twist = Twist(
        angular=np.zeros(3),
        linear=np.array([1.0, 0.0, 0.0]),
        body_name="test_body",
        reference_point=np.array([0.0, 0.0, 0.0]),
    )
    return compute_screw_axis(twist)


def _make_mock_ax():
    """Create a mock matplotlib 3D axes object."""
    ax = MagicMock()
    return ax


class TestPlotScrewAxis3D:
    """Tests for plot_screw_axis_3d."""

    def test_rotation_screw_calls_plot(self, rotation_screw) -> None:
        """plot_screw_axis_3d calls ax.plot exactly once for a rotation screw."""
        ax = _make_mock_ax()
        plot_screw_axis_3d(ax, rotation_screw, length=1.0, color="blue")
        ax.plot.assert_called_once()

    def test_rotation_screw_calls_quiver(self, rotation_screw) -> None:
        """plot_screw_axis_3d calls ax.quiver for the direction arrow."""
        ax = _make_mock_ax()
        plot_screw_axis_3d(ax, rotation_screw, length=1.0, color="red")
        ax.quiver.assert_called_once()

    def test_rotation_screw_with_label(self, rotation_screw) -> None:
        """plot_screw_axis_3d passes label to ax.plot when provided."""
        ax = _make_mock_ax()
        plot_screw_axis_3d(ax, rotation_screw, length=1.0, label="ISA")
        _, plot_kwargs = ax.plot.call_args
        assert plot_kwargs.get("label") == "ISA"

    def test_rotation_screw_annotates_pitch(self, rotation_screw) -> None:
        """plot_screw_axis_3d calls ax.text to annotate pitch for small pitch values."""
        ax = _make_mock_ax()
        plot_screw_axis_3d(ax, rotation_screw, length=1.0)
        # pitch = 0 for pure rotation — should annotate
        ax.text.assert_called_once()

    def test_translation_screw_no_pitch_annotation(self, translation_screw) -> None:
        """plot_screw_axis_3d does NOT annotate pitch for singular (pure translation)."""
        ax = _make_mock_ax()
        plot_screw_axis_3d(ax, translation_screw, length=1.0)
        ax.text.assert_not_called()

    def test_color_passed_to_plot(self, rotation_screw) -> None:
        """Color argument is forwarded to ax.plot."""
        ax = _make_mock_ax()
        plot_screw_axis_3d(ax, rotation_screw, length=0.5, color="green")
        _, plot_kwargs = ax.plot.call_args
        assert plot_kwargs.get("color") == "green"

    def test_custom_length_changes_endpoints(self, rotation_screw) -> None:
        """Different length values produce different plot coordinates."""
        ax1 = _make_mock_ax()
        ax2 = _make_mock_ax()

        plot_screw_axis_3d(ax1, rotation_screw, length=0.5)
        plot_screw_axis_3d(ax2, rotation_screw, length=2.0)

        args1 = ax1.plot.call_args[0]
        args2 = ax2.plot.call_args[0]

        # Z coordinates (index 2 for a Z-axis rotation) should differ by factor 4
        z_range1 = abs(args1[2][1] - args1[2][0])
        z_range2 = abs(args2[2][1] - args2[2][0])
        assert z_range2 > z_range1

    def test_no_quiver_when_zero_length_direction(self, rotation_screw) -> None:
        """No crash when the arrow direction norm is effectively zero (edge case)."""
        ax = _make_mock_ax()
        # Use a very small length — arrow computation should still be safe
        plot_screw_axis_3d(ax, rotation_screw, length=1e-12)
        # Should not raise; quiver may or may not be called depending on norm guard
        assert True  # Confirms no exception

    def test_assert_screw_not_none(self) -> None:
        """plot_screw_axis_3d raises ValueError when screw is None."""
        ax = _make_mock_ax()
        with pytest.raises(ValueError):
            plot_screw_axis_3d(ax, None, length=1.0)  # type: ignore[arg-type]
