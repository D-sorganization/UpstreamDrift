"""
Heavy Integration Contracts — SciPy / SymPy
=============================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: SciPy and SymPy provide the numerical and symbolic capabilities
the project depends on (ODE integration, optimization, Lagrangian mechanics).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.live_simulation
class TestSciPyContracts:
    """Contract: SciPy provides ODE solvers, optimization, and linalg."""

    def test_scipy_version(self) -> None:
        """SciPy is importable and meets minimum version."""
        import scipy

        major, minor = scipy.__version__.split(".")[:2]
        assert int(major) >= 1, f"SciPy >= 1.0 required, got {scipy.__version__}"

    def test_scipy_ode_integration(self) -> None:
        """SciPy solve_ivp integrates a pendulum ODE (used by simulation_core)."""
        from scipy.integrate import solve_ivp

        # Simple pendulum: theta'' = -g/L * sin(theta)
        g, L = 9.80665, 1.0

        def pendulum(t: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], -g / L * np.sin(y[0])])

        sol = solve_ivp(pendulum, [0, 2.0], [0.3, 0.0], max_step=0.01)

        assert sol.solver_status == "success"
        assert sol.t[-1] >= 1.99
        assert not np.any(np.isnan(sol.y))

    def test_scipy_optimize_minimize(self) -> None:
        """SciPy minimize works (used by swing_optimizer)."""
        from scipy.optimize import minimize

        # Rosenbrock function — classic test
        def rosenbrock(x: np.ndarray) -> float:
            return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

        result = minimize(rosenbrock, [0.0, 0.0], method="L-BFGS-B")
        assert result.solver_status == "success" or result.fun < 1e-4
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=0.1)

    def test_scipy_linalg_eigh(self) -> None:
        """SciPy linalg.eigh for mass matrix eigenanalysis."""
        from scipy.linalg import eigh

        # Symmetric positive definite matrix (like a mass matrix)
        M = np.array([[2.0, 0.5], [0.5, 1.0]])
        eigenvalues, eigenvectors = eigh(M)

        assert all(ev > 0 for ev in eigenvalues)
        assert eigenvectors.shape == (2, 2)

    def test_scipy_interpolate(self) -> None:
        """SciPy interpolation (used for terrain/green surface)."""
        from scipy.interpolate import RegularGridInterpolator

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        data = np.random.default_rng(42).random((10, 10))

        interp = RegularGridInterpolator((x, y), data)
        val = interp(np.array([[0.5, 0.5]]))
        assert val.shape == (1,)
        assert not np.isnan(val[0])

    def test_scipy_signal_filtering(self) -> None:
        """SciPy signal filtering (used by signal_toolkit)."""
        from scipy.signal import butter, sosfilt

        # 4th-order Butterworth low-pass at 10 Hz / 100 Hz sample rate
        sos = butter(4, 10, btype="low", fs=100, output="sos")

        # Filter a noisy signal
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
        filtered = sosfilt(sos, signal)

        assert len(filtered) == len(signal)
        # High-frequency component should be attenuated
        assert np.std(filtered) < np.std(signal)


@pytest.mark.live_simulation
class TestSymPyContracts:
    """Contract: SymPy provides symbolic computation for Lagrangian mechanics."""

    def test_sympy_version(self) -> None:
        """SymPy is importable."""
        import sympy

        assert hasattr(sympy, "__version__")

    def test_sympy_symbolic_differentiation(self) -> None:
        """SymPy can differentiate symbolic expressions."""
        import sympy as sp

        t = sp.Symbol("t")
        theta = sp.Function("theta")(t)

        # Kinetic energy of a pendulum: T = 0.5 * m * L^2 * theta_dot^2
        m, L, g = sp.symbols("m L g", positive=True)
        theta_dot = sp.diff(theta, t)

        T = sp.Rational(1, 2) * m * L**2 * theta_dot**2
        V = -m * g * L * sp.cos(theta)
        Lagrangian = T - V

        # Euler-Lagrange equation
        dL_dtheta = sp.diff(Lagrangian, theta)
        dL_dthetadot = sp.diff(Lagrangian, theta_dot)
        EL = sp.diff(dL_dthetadot, t) - dL_dtheta

        # Should simplify to m*L^2*theta'' + m*g*L*sin(theta)
        simplified = sp.simplify(EL)
        assert simplified != 0, "Euler-Lagrange equation should be non-trivial"

    def test_sympy_matrix_operations(self) -> None:
        """SymPy can compute symbolic matrix inverse and determinant."""
        import sympy as sp

        a, b, c, d = sp.symbols("a b c d")
        M = sp.Matrix([[a, b], [c, d]])

        det = M.det()
        assert det == a * d - b * c

        M_inv = M.inv()
        identity = sp.simplify(M * M_inv)
        assert identity == sp.eye(2)

    def test_sympy_lambdify(self) -> None:
        """SymPy lambdify converts symbolic expressions to numpy functions."""
        import sympy as sp

        x = sp.Symbol("x")
        expr = sp.sin(x) ** 2 + sp.cos(x) ** 2

        f = sp.lambdify(x, expr, "numpy")

        # sin²(x) + cos²(x) = 1 for all x
        x_vals = np.linspace(0, 2 * np.pi, 100)
        result = f(x_vals)
        np.testing.assert_allclose(result, 1.0, atol=1e-14)


pytestmark = pytest.mark.live_simulation
