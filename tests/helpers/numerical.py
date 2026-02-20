"""Shared numerical test helpers for UpstreamDrift.

Provides reusable assertion functions for physics simulations,
conservation laws, and numerical analysis. Designed for use across
all engine test suites (Drake, Pinocchio, MuJoCo, Simscape).

Design by Contract:
    - All public functions validate preconditions and raise TypeError/ValueError
      for invalid inputs before performing any assertion logic.
    - Assertion failures raise AssertionError with diagnostic messages.
"""

import math
from typing import List, Optional, Sequence, Union

Number = Union[int, float]


def _validate_number(value: object, name: str) -> float:
    """Validate that value is a finite number.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If value is not int or float.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def is_finite(value: object) -> bool:
    """Check whether *value* is a finite number (not NaN, not Inf).

    Args:
        value: Any object to test.

    Returns:
        True if value is a finite int or float, False otherwise.

    Examples:
        >>> is_finite(1.0)
        True
        >>> is_finite(float('nan'))
        False
        >>> is_finite(float('inf'))
        False
        >>> is_finite("hello")
        False
    """
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def assert_close(
    actual: Number,
    expected: Number,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """Assert that *actual* is close to *expected* within tolerances.

    Uses the same formula as ``numpy.isclose``:
        ``|actual - expected| <= atol + rtol * |expected|``

    Args:
        actual: The computed value.
        expected: The reference value.
        rtol: Relative tolerance (default 1e-7).
        atol: Absolute tolerance (default 0).

    Raises:
        TypeError: If actual or expected is not a number.
        ValueError: If rtol or atol is negative.
        AssertionError: If values are not close, with a diagnostic message.
    """
    actual_f = _validate_number(actual, "actual")
    expected_f = _validate_number(expected, "expected")
    rtol_f = _validate_number(rtol, "rtol")
    atol_f = _validate_number(atol, "atol")

    if rtol_f < 0:
        raise ValueError(f"rtol must be non-negative, got {rtol_f}")
    if atol_f < 0:
        raise ValueError(f"atol must be non-negative, got {atol_f}")

    diff = abs(actual_f - expected_f)
    threshold = atol_f + rtol_f * abs(expected_f)

    if not (diff <= threshold):
        rel_err = diff / abs(expected_f) if expected_f != 0 else float("inf")
        raise AssertionError(
            f"Values not close: actual={actual_f}, expected={expected_f}, "
            f"diff={diff:.2e}, rel_err={rel_err:.2e}, "
            f"threshold={threshold:.2e} (rtol={rtol_f}, atol={atol_f})"
        )


def assert_conserved(
    before: Number,
    after: Number,
    quantity_name: str = "quantity",
    rtol: float = 1e-6,
) -> None:
    """Assert that a conserved quantity has not changed beyond tolerance.

    Useful for verifying mass conservation, energy conservation,
    element balance (C/H/O tracking), etc.

    Args:
        before: Value before the operation.
        after: Value after the operation.
        quantity_name: Human-readable name for error messages.
        rtol: Relative tolerance (default 1e-6).

    Raises:
        TypeError: If before or after is not a number.
        ValueError: If rtol is negative.
        AssertionError: If the quantity changed beyond tolerance.
    """
    before_f = _validate_number(before, "before")
    after_f = _validate_number(after, "after")
    rtol_f = _validate_number(rtol, "rtol")

    if rtol_f < 0:
        raise ValueError(f"rtol must be non-negative, got {rtol_f}")

    if before_f == 0.0 and after_f == 0.0:
        return  # Both zero — trivially conserved

    diff = abs(after_f - before_f)
    scale = max(abs(before_f), abs(after_f))
    rel_change = diff / scale if scale != 0 else float("inf")

    if rel_change > rtol_f:
        raise AssertionError(
            f"{quantity_name} not conserved: before={before_f}, after={after_f}, "
            f"relative change={rel_change:.2e}, tolerance={rtol_f:.2e}"
        )


def assert_monotonic(
    values: Sequence[Number],
    increasing: bool = True,
    strict: bool = False,
    label: str = "sequence",
) -> None:
    """Assert that a sequence of values is monotonically ordered.

    Args:
        values: Sequence of numbers to check.
        increasing: If True, check non-decreasing; if False, check non-increasing.
        strict: If True, require strictly increasing/decreasing (no equal neighbours).
        label: Human-readable name for error messages.

    Raises:
        TypeError: If values is not a sequence or contains non-numbers.
        ValueError: If the sequence has fewer than 2 elements.
        AssertionError: If monotonicity is violated, identifying the first violation.
    """
    if not hasattr(values, "__len__") or not hasattr(values, "__getitem__"):
        raise TypeError(f"values must be a sequence, got {type(values).__name__}")

    if len(values) < 2:
        raise ValueError(
            f"Need at least 2 values to check monotonicity, got {len(values)}"
        )

    for i, v in enumerate(values):
        _validate_number(v, f"values[{i}]")

    direction = "increasing" if increasing else "decreasing"
    strictness = "strictly " if strict else ""

    for i in range(len(values) - 1):
        a, b = float(values[i]), float(values[i + 1])
        if increasing:
            violation = (b < a) if not strict else (b <= a)
        else:
            violation = (b > a) if not strict else (b >= a)

        if violation:
            raise AssertionError(
                f"{label} is not {strictness}{direction}: "
                f"values[{i}]={a} vs values[{i+1}]={b}"
            )


def assert_physics_state(
    position: Sequence[Number],
    velocity: Sequence[Number],
    acceleration: Optional[Sequence[Number]] = None,
) -> None:
    """Validate that physics state vectors are well-formed.

    Checks that all components are finite and that vectors have
    consistent shapes (same length).

    Args:
        position: Position vector (e.g. [x, y, z]).
        velocity: Velocity vector (same dimensionality as position).
        acceleration: Optional acceleration vector (same dimensionality).

    Raises:
        TypeError: If inputs are not sequences of numbers.
        ValueError: If vectors have inconsistent lengths or contain NaN/Inf.
    """
    def _check_vector(vec: Sequence[Number], name: str) -> int:
        if not hasattr(vec, "__len__") or not hasattr(vec, "__getitem__"):
            raise TypeError(f"{name} must be a sequence, got {type(vec).__name__}")
        if len(vec) == 0:
            raise ValueError(f"{name} must not be empty")
        for i, v in enumerate(vec):
            _validate_number(v, f"{name}[{i}]")
            if not is_finite(v):
                raise ValueError(
                    f"{name}[{i}] is not finite: {v}"
                )
        return len(vec)

    n_pos = _check_vector(position, "position")
    n_vel = _check_vector(velocity, "velocity")

    if n_pos != n_vel:
        raise ValueError(
            f"position and velocity must have same length: "
            f"len(position)={n_pos}, len(velocity)={n_vel}"
        )

    if acceleration is not None:
        n_acc = _check_vector(acceleration, "acceleration")
        if n_acc != n_pos:
            raise ValueError(
                f"acceleration must have same length as position: "
                f"len(acceleration)={n_acc}, len(position)={n_pos}"
            )


def assert_jacobian_symmetry(
    J: Sequence[Sequence[Number]],
    rtol: float = 1e-6,
) -> None:
    """Assert that a Jacobian (or any square matrix) is symmetric.

    Checks that ``J[i][j]`` is close to ``J[j][i]`` for all i, j.

    Args:
        J: A square matrix represented as a list of lists (rows).
        rtol: Relative tolerance for element-wise comparison.

    Raises:
        TypeError: If J is not a sequence of sequences of numbers.
        ValueError: If J is not square or is empty.
        AssertionError: If any pair (i,j) violates symmetry beyond rtol.
    """
    if not hasattr(J, "__len__") or not hasattr(J, "__getitem__"):
        raise TypeError(f"J must be a sequence of sequences, got {type(J).__name__}")

    n = len(J)
    if n == 0:
        raise ValueError("Jacobian must not be empty")

    for i, row in enumerate(J):
        if not hasattr(row, "__len__") or not hasattr(row, "__getitem__"):
            raise TypeError(
                f"J[{i}] must be a sequence, got {type(row).__name__}"
            )
        if len(row) != n:
            raise ValueError(
                f"Jacobian must be square: row {i} has length {len(row)}, "
                f"expected {n}"
            )
        for j, v in enumerate(row):
            _validate_number(v, f"J[{i}][{j}]")

    rtol_f = _validate_number(rtol, "rtol")
    if rtol_f < 0:
        raise ValueError(f"rtol must be non-negative, got {rtol_f}")

    for i in range(n):
        for j in range(i + 1, n):
            a = float(J[i][j])
            b = float(J[j][i])
            diff = abs(a - b)
            scale = max(abs(a), abs(b), 1e-15)
            if diff / scale > rtol_f:
                raise AssertionError(
                    f"Jacobian not symmetric at ({i},{j}): "
                    f"J[{i}][{j}]={a}, J[{j}][{i}]={b}, "
                    f"rel_diff={diff/scale:.2e}, rtol={rtol_f}"
                )
