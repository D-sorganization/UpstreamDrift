"""Unit tests for polynomial_generator.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.polynomial_generator import (
    PolynomialGeneratorWidget,
)


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.fixture
def generator_widget(qapp):
    widget = PolynomialGeneratorWidget()
    yield widget


def test_widget_initialization(generator_widget):
    """Test the widget initializes correctly."""
    assert generator_widget.windowTitle() == "Polynomial Function Generator"
    assert generator_widget.mode == "add_points"
    assert generator_widget.order_spin.value() == 6


def test_set_joints(generator_widget):
    """Test setting joints."""
    joints = ["joint_a", "joint_b"]
    generator_widget.set_joints(joints)

    assert generator_widget.joint_names == joints
    assert generator_widget.joint_combo.count() == 2
    assert generator_widget.joint_combo.itemText(0) == "joint_a"


def test_calculate_poly_fit_success(generator_widget):
    """Test polynomial fitting logic."""
    # Set enough points for a 2nd order fit
    generator_widget.order_spin.setValue(2)
    generator_widget.current_points = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]

    success = generator_widget._calculate_poly_fit()

    assert success is True
    assert generator_widget.polynomial_coeffs is not None
    # y = x^2, so coeffs should be roughly [1, 0, 0]
    assert np.allclose(generator_widget.polynomial_coeffs, [1.0, 0.0, 0.0], atol=1e-5)


def test_calculate_poly_fit_insufficient_points(generator_widget):
    """Test fitting with too few points."""
    generator_widget.order_spin.setValue(2)
    generator_widget.current_points = [(0.0, 0.0), (1.0, 1.0)]

    success = generator_widget._calculate_poly_fit()
    assert success is False
    assert generator_widget.polynomial_coeffs is None


@patch("PyQt6.QtWidgets.QMessageBox.warning")
def test_fit_polynomial_with_ui(mock_warning, generator_widget):
    """Test fit polynomial via UI action."""
    generator_widget.order_spin.setValue(2)
    generator_widget.current_points = [(0.0, 0.0), (1.0, 1.0)]

    generator_widget._fit_polynomial()
    mock_warning.assert_called_once()


def test_generate_from_equation(generator_widget):
    """Test generating points from an equation."""
    generator_widget.equation_input.setText("x**2")
    generator_widget.x_min_spin.setValue(0.0)
    generator_widget.x_max_spin.setValue(2.0)

    # We should intercept the GUI operations
    with (
        patch.object(generator_widget, "_update_plot"),
        patch.object(generator_widget, "_fit_polynomial"),
    ):
        generator_widget._generate_from_equation()

        assert len(generator_widget.current_points) == 20
        # The points should follow y = x^2
        x, y = generator_widget.current_points[-1]
        assert np.isclose(x, 2.0)
        assert np.isclose(y, 4.0)


def test_set_mode(generator_widget):
    """Test changing modes."""
    generator_widget._set_mode("equation", True)

    assert generator_widget.mode == "equation"
    assert generator_widget.equation_input.isEnabled() is True
    assert generator_widget.generate_eq_btn.isEnabled() is True


def test_clear_data(generator_widget):
    """Test clearing data."""
    generator_widget.current_points = [(1.0, 1.0)]
    generator_widget.polynomial_coeffs = np.array([1.0, 0.0])

    generator_widget._clear_data()

    assert len(generator_widget.current_points) == 0
    assert generator_widget.polynomial_coeffs is None
