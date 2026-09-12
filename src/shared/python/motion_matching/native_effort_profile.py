"""Explicit conversion from native absolute-time sextics to primitive efforts."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque,
)
from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape


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
        self._bernstein = bernstein_to_simscape(np.eye(7), duration_s=1.0)[:, ::-1]
        self._force_map = np.eye(len(names))
        self._force_map[:3, :3] = rotation

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

    def bernstein_control_jacobian(
        self, time_s: float, *, basis_duration_s: float, first_control: int = 4
    ) -> NDArray[np.float64]:
        """Differentiate primitive efforts by physical Bernstein input controls.

        Columns are input-coordinate-major then ascending control index through
        six. Force columns include the world-to-base map; torque columns retain
        primitive conjugacy. These are physical controls, without optimizer scale.
        """
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Native profile time must be finite nonnegative seconds")
        if not np.isfinite(basis_duration_s) or basis_duration_s <= 0:
            raise ValueError("Basis duration must be finite and positive")
        if (
            isinstance(first_control, bool)
            or not isinstance(first_control, int)
            or not 0 <= first_control <= 6
        ):
            raise ValueError("Invalid first Bernstein control")
        basis = evaluate_polynomial_torque(self._bernstein, time_s / basis_duration_s)[
            first_control:
        ]
        result = np.einsum("ij,k->ijk", self._force_map, basis).reshape(
            len(self._names), -1
        )
        if not np.isfinite(result).all():
            raise ValueError("Native Bernstein derivative overflowed")
        result.setflags(write=False)
        return result
