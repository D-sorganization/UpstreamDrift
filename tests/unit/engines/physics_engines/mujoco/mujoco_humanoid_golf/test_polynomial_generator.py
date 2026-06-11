"""Unit tests for polynomial_generator.py."""

from unittest.mock import patch

import numpy as np
import pytest
from PyQt6 import QtWidgets

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.polynomial_generator import (
    PolynomialGeneratorWidget,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.control_system import (
    ControlSystem,
    ControlType,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.actuator_detail_dialog import (
    ActuatorDetailDialog,
)
from src.shared.python.signal_toolkit.polynomial_generator import (
    PolynomialFitError,
    PolynomialGeneratorWidget as CanonicalPolynomialGeneratorWidget,
)

pytestmark = pytest.mark.unit


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


def test_mujoco_polynomial_generator_is_canonical_widget():
    """The MuJoCo import path is a compatibility shim over signal_toolkit."""
    assert PolynomialGeneratorWidget is CanonicalPolynomialGeneratorWidget
    assert PolynomialFitError.__name__ == "PolynomialFitError"


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


def test_fit_polynomial_with_ui(qapp):
    """Test fit polynomial via UI action."""
    errors: list[tuple[str, str]] = []
    generator_widget = PolynomialGeneratorWidget(
        error_handler=lambda *args: errors.append(args)
    )
    generator_widget.order_spin.setValue(2)
    generator_widget.current_points = [(0.0, 0.0), (1.0, 1.0)]

    generator_widget._fit_polynomial()
    assert errors == [("Fit Error", "Need at least 3 points for a 2th order fit.")]


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


def test_actuator_detail_dialog_embeds_canonical_polynomial_widget(qapp):
    """Smoke-test the polynomial actuator path and signal wiring."""
    control_system = ControlSystem(1)
    control_system.set_control_type(0, ControlType.POLYNOMIAL)

    dialog = ActuatorDetailDialog(
        control_system=control_system,
        actuator_index=0,
        actuator_name="hip",
        slider_sync=None,
    )

    assert isinstance(dialog.poly_widget, CanonicalPolynomialGeneratorWidget)

    coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    dialog.poly_widget.polynomial_generated.emit("hip", coeffs)

    assert control_system.get_control_type(0) is ControlType.POLYNOMIAL
    np.testing.assert_allclose(
        control_system.get_actuator_control(0).polynomial_coeffs,
        np.array(coeffs),
    )
