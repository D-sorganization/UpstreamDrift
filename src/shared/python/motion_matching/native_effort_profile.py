"""Explicit conversion from native absolute-time sextics to primitive efforts."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque,
)


class NativeEffortProfile:
    """Own native highest-power-first coefficients and a world-to-base rotation.

    The caller must supply the native inventory with three translational force
    inputs first. Remaining inputs are joint-conjugate torques. No horizon
    normalization or time shift is applied; evaluate always takes seconds.
    """

    def __init__(
        self,
        coordinate_names: Sequence[str],
        native_coefficients: ArrayLike,
        world_to_base: ArrayLike,
    ) -> None:
        names = tuple(coordinate_names)
        if (
            len(names) < 3
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name.strip() for name in names)
        ):
            raise ValueError("Provide unique nonempty native coordinate names")
        coefficients = np.array(native_coefficients, dtype=float, copy=True)
        if coefficients.shape != (len(names), 7) or not np.isfinite(coefficients).all():
            raise ValueError(
                "Native coefficients must be finite and coordinate-by-seven"
            )
        rotation = np.array(world_to_base, dtype=float, copy=True)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0)
        ):
            raise ValueError("Provide a proper world-to-base rotation")
        # Convert once to the existing shared evaluator's documented layout.
        self._coefficients = coefficients[:, ::-1].copy()
        self._coefficients.setflags(write=False)
        rotation.setflags(write=False)
        self._rotation = rotation
        self._names = names

    def evaluate(self, time_s: float) -> dict[str, float]:
        """Return finite native primitive forces/torques at absolute seconds."""
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Native profile time must be finite nonnegative seconds")
        with np.errstate(over="ignore", invalid="ignore"):
            values = evaluate_polynomial_torque(self._coefficients, time_s)
            values[:3] = self._rotation @ values[:3]
        if not np.isfinite(values).all():
            raise ValueError("Native effort evaluation overflowed")
        return dict(zip(self._names, map(float, values), strict=True))
