"""Taylor and Maclaurin series expansion module.

This module provides functionality for computing Taylor and Maclaurin series
expansions of functions, including:
- Taylor series computation at any point
- Maclaurin series (Taylor at x=0)
- Pre-defined common function series (sin, cos, exp, ln, etc.)
- Convergence analysis
- Error bounds estimation

Following pragmatic programming and Design by Contract principles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class SeriesResult:
    """Result of a series expansion computation.

    Attributes:
        coefficients: Array of Taylor/Maclaurin coefficients [c0, c1, c2, ...]
        n_terms: Number of terms in the expansion
        center: Center point of the expansion (a in Taylor series)
        function: Callable that evaluates the series approximation
        radius_of_convergence: Estimated radius of convergence (None if unknown)
    """

    coefficients: NDArray[np.floating]
    n_terms: int
    center: float
    function: Callable[[ArrayLike], float | NDArray[np.floating]]
    radius_of_convergence: float | None = None


class SeriesExpansion:
    """Taylor and Maclaurin series expansion calculator.

    This class provides methods for computing Taylor and Maclaurin series
    expansions of functions using numerical differentiation.

    Attributes:
        max_terms: Maximum number of terms allowed in series expansion
        h: Step size for numerical differentiation
    """

    def __init__(self, max_terms: int = 50, h: float = 1e-5) -> None:
        """Initialize the series expansion calculator.

        Args:
            max_terms: Maximum number of terms in series expansions.
                       Must be positive.
            h: Step size for numerical differentiation

        Raises:
            ValueError: If max_terms is not positive
        """
        if max_terms <= 0:
            raise ValueError(f"max_terms must be positive, got {max_terms}")

        self.max_terms = max_terms
        self.h = h

    def taylor_series(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        center: float,
        n_terms: int,
    ) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
        """Compute Taylor series expansion of a function.

        The Taylor series of f(x) around center a is:
            f(x) = Σ f^(n)(a)/n! * (x-a)^n

        Args:
            f: Function to expand. Must be callable and accept numeric input.
            center: Center point (a) of the expansion
            n_terms: Number of terms to include in the series

        Returns:
            Callable that evaluates the Taylor series approximation

        Raises:
            TypeError: If f is not callable
            ValueError: If n_terms is not positive
        """
        if not callable(f):
            raise TypeError(f"f must be callable, got {type(f)}")
        if n_terms <= 0:
            raise ValueError(f"n_terms must be positive, got {n_terms}")

        n_terms = min(n_terms, self.max_terms)
        coefficients = self.get_coefficients(f, center, n_terms)

        def taylor_func(x: ArrayLike) -> float | NDArray[np.floating]:
            """Evaluate the Taylor series at x."""
            x_arr = np.asarray(x)
            dx = x_arr - center
            result = np.zeros_like(x_arr, dtype=np.float64)

            for n, coeff in enumerate(coefficients):
                result = result + coeff * (dx**n)

            # Return scalar if input was scalar
            if np.ndim(x) == 0:
                return float(result)
            return result

        return taylor_func

    def maclaurin_series(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        n_terms: int,
    ) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
        """Compute Maclaurin series expansion of a function.

        The Maclaurin series is a Taylor series centered at x=0:
            f(x) = Σ f^(n)(0)/n! * x^n

        Args:
            f: Function to expand. Must be callable and accept numeric input.
            n_terms: Number of terms to include in the series

        Returns:
            Callable that evaluates the Maclaurin series approximation

        Raises:
            TypeError: If f is not callable
            ValueError: If n_terms is not positive
        """
        return self.taylor_series(f, center=0, n_terms=n_terms)

    def get_coefficients(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        center: float,
        n_terms: int,
    ) -> NDArray[np.floating]:
        """Compute Taylor series coefficients.

        Uses polynomial fitting for numerical stability. This approach samples
        the function at multiple points near the center and fits a polynomial,
        which is much more stable than repeated numerical differentiation.

        Args:
            f: Function to expand
            center: Center point of the expansion
            n_terms: Number of coefficients to compute

        Returns:
            Array of coefficients [c0, c1, c2, ..., c_{n-1}]
        """
        if not (f is not None):
            raise ValueError("f must be provided")
        if not (f is not None):
            raise ValueError("f must be provided")
        n_terms = min(n_terms, self.max_terms)

        # Use polynomial fitting for stability
        # Sample the function at 2*n_terms points around center
        num_samples = max(2 * n_terms + 1, 21)
        dx_range = min(1.0, n_terms * 0.1)  # Adaptive range based on terms

        # Create sample points centered at 'center'
        dx_values = np.linspace(-dx_range, dx_range, num_samples)
        x_samples = center + dx_values
        y_samples = np.array([float(f(float(x))) for x in x_samples])  # type: ignore[arg-type]

        # Fit polynomial of degree n_terms-1 in terms of (x - center)
        # This gives us Taylor coefficients directly
        try:
            # Use numpy's polyfit with the shifted variable
            coeffs_reversed = np.polyfit(dx_values, y_samples, n_terms - 1)
            # polyfit returns highest degree first, we need lowest first
            coefficients = coeffs_reversed[::-1]
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to numerical differentiation
            coefficients = np.zeros(n_terms)
            for n in range(n_terms):
                derivative = self._nth_derivative(f, center, n)
                coefficients[n] = derivative / self._factorial(n)

        return coefficients

    def get_series_result(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        center: float,
        n_terms: int,
    ) -> SeriesResult:
        """Compute complete series result with metadata.

        Args:
            f: Function to expand
            center: Center point of the expansion
            n_terms: Number of terms to include

        Returns:
            SeriesResult dataclass with coefficients, function, and metadata
        """
        if not (f is not None):
            raise ValueError("f must be provided")
        if not (f is not None):
            raise ValueError("f must be provided")
        coefficients = self.get_coefficients(f, center, n_terms)
        series_func = self.taylor_series(f, center, n_terms)

        # Estimate radius of convergence if possible
        roc = self._estimate_radius_of_convergence(coefficients)

        return SeriesResult(
            coefficients=coefficients,
            n_terms=n_terms,
            center=center,
            function=series_func,
            radius_of_convergence=roc,
        )

    def analyze_convergence(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        center: float,
        x_test: float,
        tolerance: float = 1e-10,
    ) -> dict:
        """Analyze convergence of Taylor series at a test point.

        Args:
            f: Function to expand
            center: Center point of the expansion
            x_test: Point at which to test convergence
            tolerance: Convergence tolerance

        Returns:
            Dictionary with convergence analysis results:
            - convergent: Whether the series converges at x_test
            - terms_for_convergence: Number of terms needed for convergence
            - final_error: Error at max_terms
            - errors_by_term: List of errors for each number of terms
        """
        if not (f is not None):
            raise ValueError("f must be provided")
        if not (f is not None):
            raise ValueError("f must be provided")
        try:
            exact_value = float(f(x_test))  # type: ignore[arg-type]
        except (ValueError, RuntimeError, FloatingPointError):
            return {
                "convergent": False,
                "terms_for_convergence": None,
                "final_error": float("inf"),
                "errors_by_term": [],
            }

        errors_by_term = []
        convergent = False
        terms_for_convergence = None
        prev_approx = None

        for n in range(1, self.max_terms + 1):
            taylor_func = self.taylor_series(f, center, n)
            approx = float(taylor_func(x_test))  # type: ignore[arg-type]
            error = abs(approx - exact_value)
            errors_by_term.append(error)

            if error < tolerance and not convergent:
                convergent = True
                terms_for_convergence = n

            # Check for divergence (error growing)
            if prev_approx is not None and (
                abs(approx) > 1e15 or np.isnan(approx) or np.isinf(approx)
            ):
                return {
                    "convergent": False,
                    "terms_for_convergence": None,
                    "final_error": float("inf"),
                    "errors_by_term": errors_by_term,
                }
            prev_approx = approx

        return {
            "convergent": convergent,
            "terms_for_convergence": terms_for_convergence,
            "final_error": errors_by_term[-1] if errors_by_term else float("inf"),
            "errors_by_term": errors_by_term,
        }

    def estimate_error_bound(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        center: float,
        x_test: float,
        n_terms: int,
    ) -> float:
        """Estimate error bound for Taylor series truncation.

        Uses the Lagrange remainder formula approximation:
        |R_n(x)| ≤ M * |x - center|^(n+1) / (n+1)!

        where M is an upper bound on |f^(n+1)| in the interval.

        Args:
            f: Function to expand
            center: Center point of the expansion
            x_test: Point at which to estimate error
            n_terms: Number of terms in the truncated series

        Returns:
            Estimated upper bound on the error
        """
        if not (f is not None):
            raise ValueError("f must be provided")
        if not (f is not None):
            raise ValueError("f must be provided")
        if n_terms <= 0:
            return float("inf")

        # Estimate (n+1)th derivative magnitude
        try:
            deriv = abs(self._nth_derivative(f, center, n_terms))
            # Add safety factor for estimation uncertainty
            M = deriv * 2.0
        except (ValueError, RuntimeError, FloatingPointError):
            return float("inf")

        dx = abs(x_test - center)
        factorial_np1 = self._factorial(n_terms + 1)

        if factorial_np1 == 0:
            return float("inf")

        bound = M * (dx ** (n_terms + 1)) / factorial_np1
        return float(bound)

    def _nth_derivative(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        x: float,
        n: int,
    ) -> float:
        """Compute nth derivative of f at x using numerical differentiation.

        Uses Richardson extrapolation for improved accuracy and stability.

        Args:
            f: Function to differentiate
            x: Point at which to compute derivative
            n: Order of derivative (0 returns f(x))

        Returns:
            Approximate value of f^(n)(x)
        """
        if not (f is not None):
            raise ValueError("f must be provided")
        if not (f is not None):
            raise ValueError("f must be provided")
        if n == 0:
            return float(f(x))  # type: ignore[arg-type]

        # Use Richardson extrapolation for better accuracy
        return self._richardson_derivative(f, x, n)

    def _richardson_derivative(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        x: float,
        n: int,
        num_terms: int = 6,
    ) -> float:
        """Compute nth derivative using Richardson extrapolation.

        Richardson extrapolation improves accuracy by combining estimates
        at different step sizes to cancel leading error terms.

        Args:
            f: Function to differentiate
            x: Point at which to compute derivative
            n: Order of derivative
            num_terms: Number of extrapolation terms

        Returns:
            Approximate value of f^(n)(x)
        """
        # Compute derivatives at decreasing step sizes
        if not (f is not None):
            raise ValueError("f must be provided")
        if not (f is not None):
            raise ValueError("f must be provided")
        h0 = 0.5  # Initial step size (larger for stability)
        estimates = []

        for i in range(num_terms):
            h = h0 / (2**i)
            est = self._central_difference_deriv(f, x, n, h)
            estimates.append(est)

        # Apply Richardson extrapolation
        # Each level cancels O(h^2) error terms
        for level in range(1, num_terms):
            new_estimates = []
            factor = 4**level  # Factor for second-order error
            for i in range(len(estimates) - 1):
                improved = (factor * estimates[i + 1] - estimates[i]) / (factor - 1)
                new_estimates.append(improved)
            estimates = new_estimates

        return estimates[0] if estimates else 0.0

    def _central_difference_deriv(
        self,
        f: Callable[[ArrayLike], ArrayLike],
        x: float,
        n: int,
        h: float,
    ) -> float:
        """Compute nth derivative using central difference formula.

        Uses the formula for central differences:
        f^(n)(x) ≈ (1/h^n) * Σ_{k=0}^{n} (-1)^k * C(n,k) * f(x + (n/2-k)*h)

        Args:
            f: Function to differentiate
            x: Point at which to compute derivative
            n: Order of derivative
            h: Step size

        Returns:
            Approximate derivative value
        """
        if not (f is not None):
            raise ValueError("f must be provided")
        if not (f is not None):
            raise ValueError("f must be provided")
        result = 0.0
        for k in range(n + 1):
            coeff = ((-1) ** k) * self._binomial(n, k)
            point = x + (n / 2 - k) * h
            try:
                val = float(f(point))  # type: ignore[arg-type]
                if np.isfinite(val):
                    result += coeff * val
            except (ValueError, RuntimeError, FloatingPointError):
                pass

        return result / (h**n)

    @staticmethod
    def _factorial(n: int) -> int:
        """Compute factorial of n."""
        if n < 0:
            raise ValueError(f"Factorial undefined for negative numbers: {n}")
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def _binomial(n: int, k: int) -> int:
        """Compute binomial coefficient C(n, k)."""
        if not (n is not None):
            raise ValueError("n must be provided")
        if not (n is not None):
            raise ValueError("n must be provided")
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    @staticmethod
    def _estimate_radius_of_convergence(
        coefficients: NDArray[np.floating],
    ) -> float | None:
        """Estimate radius of convergence from coefficients.

        Uses the ratio test: R = lim |a_n / a_{n+1}|

        Returns None if estimation fails.
        """
        if len(coefficients) < 5:
            return None

        ratios = []
        for i in range(len(coefficients) - 1):
            if abs(coefficients[i + 1]) > 1e-15:
                ratio = abs(coefficients[i] / coefficients[i + 1])
                if not np.isinf(ratio) and not np.isnan(ratio):
                    ratios.append(ratio)

        if len(ratios) < 3:
            return None

        # Use last few ratios for estimate
        return float(np.median(ratios[-5:]))


# =============================================================================
# Pre-defined Common Series Functions
# =============================================================================


def exp_series(
    n_terms: int = 20,
) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
    """Create exponential series approximation: e^x = Σ x^n/n!

    Args:
        n_terms: Number of terms in the series

    Returns:
        Callable that computes the exponential series
    """

    def exp_func(x: ArrayLike) -> float | NDArray[np.floating]:
        """Compute the exponential series approximation for *x*."""
        x_arr = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x_arr)
        term = np.ones_like(x_arr)

        for n in range(n_terms):
            result = result + term
            term = term * x_arr / (n + 1)

        if np.ndim(x) == 0:
            return float(result)
        return result

    return exp_func


def sin_series(
    n_terms: int = 20,
) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
    """Create sine series approximation: sin(x) = Σ (-1)^n * x^(2n+1)/(2n+1)!

    Args:
        n_terms: Number of terms in the series

    Returns:
        Callable that computes the sine series
    """

    def sin_func(x: ArrayLike) -> float | NDArray[np.floating]:
        """Compute the sine series approximation for *x*."""
        x_arr = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x_arr)
        term = x_arr.copy()  # First term is x

        for n in range(n_terms):
            result = result + term
            # Next term: multiply by -x^2 / ((2n+2)(2n+3))
            term = -term * x_arr * x_arr / ((2 * n + 2) * (2 * n + 3))

        if np.ndim(x) == 0:
            return float(result)
        return result

    return sin_func


def cos_series(
    n_terms: int = 20,
) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
    """Create cosine series approximation: cos(x) = Σ (-1)^n * x^(2n)/(2n)!

    Args:
        n_terms: Number of terms in the series

    Returns:
        Callable that computes the cosine series
    """

    def cos_func(x: ArrayLike) -> float | NDArray[np.floating]:
        """Compute the cosine series approximation for *x*."""
        x_arr = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x_arr)
        term = np.ones_like(x_arr)  # First term is 1

        for n in range(n_terms):
            result = result + term
            # Next term: multiply by -x^2 / ((2n+1)(2n+2))
            term = -term * x_arr * x_arr / ((2 * n + 1) * (2 * n + 2))

        if np.ndim(x) == 0:
            return float(result)
        return result

    return cos_func


def ln_series(
    n_terms: int = 50,
) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
    """Create natural log series approximation: ln(1+x) = Σ (-1)^(n+1) * x^n/n

    Valid for |x| < 1.

    Args:
        n_terms: Number of terms in the series

    Returns:
        Callable that computes the ln(1+x) series
    """

    def ln_func(x: ArrayLike) -> float | NDArray[np.floating]:
        """Compute the natural logarithm series ln(1+x) for *x*."""
        x_arr = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x_arr)
        term = x_arr.copy()  # First term is x

        for n in range(1, n_terms + 1):
            sign = 1 if n % 2 == 1 else -1
            result = result + sign * term / n
            term = term * x_arr

        if np.ndim(x) == 0:
            return float(result)
        return result

    return ln_func


def geometric_series(
    n_terms: int = 50,
) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
    """Create geometric series approximation: 1/(1-x) = Σ x^n

    Valid for |x| < 1.

    Args:
        n_terms: Number of terms in the series

    Returns:
        Callable that computes the geometric series
    """

    def geo_func(x: ArrayLike) -> float | NDArray[np.floating]:
        """Compute the geometric series 1/(1-x) for *x*."""
        x_arr = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x_arr)
        term = np.ones_like(x_arr)

        for _ in range(n_terms):
            result = result + term
            term = term * x_arr

        if np.ndim(x) == 0:
            return float(result)
        return result

    return geo_func


def arctan_series(
    n_terms: int = 50,
) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
    """Create arctan series approximation: arctan(x) = Σ (-1)^n * x^(2n+1)/(2n+1)

    Valid for |x| <= 1.

    Args:
        n_terms: Number of terms in the series

    Returns:
        Callable that computes the arctan series
    """

    def arctan_func(x: ArrayLike) -> float | NDArray[np.floating]:
        """Compute the arctangent series approximation for *x*."""
        x_arr = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x_arr)
        term = x_arr.copy()  # First term is x
        x_squared = x_arr * x_arr

        for n in range(n_terms):
            sign = 1 if n % 2 == 0 else -1
            result = result + sign * term / (2 * n + 1)
            term = term * x_squared

        if np.ndim(x) == 0:
            return float(result)
        return result

    return arctan_func


def sinh_series(
    n_terms: int = 20,
) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
    """Create sinh series approximation: sinh(x) = Σ x^(2n+1)/(2n+1)!

    Args:
        n_terms: Number of terms in the series

    Returns:
        Callable that computes the sinh series
    """

    def sinh_func(x: ArrayLike) -> float | NDArray[np.floating]:
        """Compute the hyperbolic sine series approximation for *x*."""
        x_arr = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x_arr)
        term = x_arr.copy()  # First term is x

        for n in range(n_terms):
            result = result + term
            # Next term: multiply by x^2 / ((2n+2)(2n+3))
            term = term * x_arr * x_arr / ((2 * n + 2) * (2 * n + 3))

        if np.ndim(x) == 0:
            return float(result)
        return result

    return sinh_func


def cosh_series(
    n_terms: int = 20,
) -> Callable[[ArrayLike], float | NDArray[np.floating]]:
    """Create cosh series approximation: cosh(x) = Σ x^(2n)/(2n)!

    Args:
        n_terms: Number of terms in the series

    Returns:
        Callable that computes the cosh series
    """

    def cosh_func(x: ArrayLike) -> float | NDArray[np.floating]:
        """Compute the hyperbolic cosine series approximation for *x*."""
        x_arr = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x_arr)
        term = np.ones_like(x_arr)  # First term is 1

        for n in range(n_terms):
            result = result + term
            # Next term: multiply by x^2 / ((2n+1)(2n+2))
            term = term * x_arr * x_arr / ((2 * n + 1) * (2 * n + 2))

        if np.ndim(x) == 0:
            return float(result)
        return result

    return cosh_func
