"""Unit tests for Pinocchio Torque Fitting Tool."""

import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Test fit_torque_poly
from engines.physics_engines.pinocchio.python.pinocchio_golf.torque_fitting import (
    fit_torque_poly,
    evaluate_torque_poly,
    main,
)

class TestTorqueFitting:
    """Test suite for torque fitting utilities."""

    def test_fit_torque_poly_exact(self):
        """Test fitting a polynomial to exact data."""
        # y = 2x^2 + 3x + 1
        t = np.linspace(0, 10, 20)
        tau = 2*t**2 + 3*t + 1

        coeffs = fit_torque_poly(t, tau, degree=2)

        # Coefficients should be [2, 3, 1] (highest power first)
        np.testing.assert_allclose(coeffs, [2, 3, 1], atol=1e-10)

    def test_evaluate_torque_poly(self):
        """Test evaluating a polynomial."""
        coeffs = np.array([2.0, 3.0, 1.0]) # 2x^2 + 3x + 1
        t = np.array([0, 1, 2])

        expected = 2*t**2 + 3*t + 1
        result = evaluate_torque_poly(coeffs, t)

        np.testing.assert_allclose(result, expected)

    def test_fit_shape_mismatch(self):
        """Test error on shape mismatch."""
        t = np.array([1, 2, 3])
        tau = np.array([1, 2]) # Mismatch

        with pytest.raises(ValueError, match="t and tau must have same shape"):
            fit_torque_poly(t, tau)

    def test_main(self):
        """Test main function via mocking."""
        # Create a dummy CSV file content
        csv_content = "t,tau\n0,1\n1,6\n2,15\n" # y = 2x^2 + 3x + 1

        with patch("argparse.ArgumentParser.parse_args") as mock_args, \
             patch("numpy.loadtxt") as mock_loadtxt, \
             patch("matplotlib.pyplot.show") as mock_show, \
             patch("matplotlib.pyplot.figure"), \
             patch("matplotlib.pyplot.plot"), \
             patch("numpy.save") as mock_save:

            # Setup args
            mock_args.return_value = MagicMock(csv="data.csv", degree=2, out="out.npy")

            # Setup loadtxt to return our data
            # Data from csv_content: [[0,1], [1,6], [2,15]]
            mock_loadtxt.return_value = np.array([[0, 1], [1, 6], [2, 15]])

            # Run main
            main()

            # Verify numpy.save was called
            mock_save.assert_called_once()
            args, _ = mock_save.call_args
            assert args[0] == "out.npy"
            coeffs = args[1]
            # Should match [2, 3, 1] approximately
            np.testing.assert_allclose(coeffs, [2, 3, 1], atol=1e-8)
