"""Differential interchange for three serial intrinsic rotation coordinates.

SI units throughout. Rotations map child vectors to the joint parent frame;
angular velocity, acceleration and moment are expressed in that parent frame.
This is a coordinate chart, not permission to remove intermediate rigid bodies.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .se3 import matrix_to_quat, quat_exp, quat_to_matrix

Array = NDArray[np.float64]


class SingularChartError(ValueError):
    """The requested native-coordinate inverse is not numerically reliable."""


def _vector(value: ArrayLike, size: int = 3) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite vector of length {size}")
    return result


@dataclass(frozen=True)
class SerialRotationChart:
    """Three distinct co-located intrinsic axes (XYZ, ZYX, etc.).

    `max_condition` rejects unreliable inverse rate and effort maps. Forward
    maps remain defined at gimbal lock; no pseudoinverse silently changes them.
    Fixed pre/post joint transforms belong to the model adapter, not this chart.
    """

    axes: str
    max_condition: float = 1e8

    def __post_init__(self) -> None:
        if len(self.axes) != 3 or set(self.axes) != set("XYZ"):
            raise ValueError("Expected a permutation of uppercase XYZ")
        if not np.isfinite(self.max_condition) or self.max_condition <= 1:
            raise ValueError("max_condition must be finite and greater than one")

    def _kinematics(self, coordinates: ArrayLike) -> tuple[Array, Array]:
        q = _vector(coordinates)
        rotation = np.eye(3)
        columns = []
        for axis, angle in zip(self.axes, q, strict=True):
            basis = np.eye(3)["XYZ".index(axis)]
            columns.append(rotation @ basis)
            rotation = rotation @ quat_to_matrix(quat_exp(basis * angle))
        return rotation, np.column_stack(columns)

    def rotation(self, coordinates: ArrayLike) -> Array:
        """Child-to-parent SO(3) matrix, including at singular coordinates."""
        return self._kinematics(coordinates)[0]

    def quaternion(self, coordinates: ArrayLike) -> Array:
        """Unit Hamilton quaternion in explicit scalar-first wxyz order."""
        return matrix_to_quat(self.rotation(coordinates))

    def rate_map(self, coordinates: ArrayLike) -> Array:
        """E in omega_parent = E(q) qdot; columns are current screw axes."""
        return self._kinematics(coordinates)[1]

    def condition_number(self, coordinates: ArrayLike) -> float:
        """Spectral condition number of the native-coordinate rate map."""
        return float(np.linalg.cond(self.rate_map(coordinates)))

    def _invertible_map(self, coordinates: ArrayLike) -> Array:
        matrix = self.rate_map(coordinates)
        condition = float(np.linalg.cond(matrix))
        if not np.isfinite(condition) or condition > self.max_condition:
            raise SingularChartError(
                f"Native rate map condition {condition:g} exceeds {self.max_condition:g}"
            )
        return matrix

    def angular_velocity(self, coordinates: ArrayLike, rate: ArrayLike) -> Array:
        """Parent angular velocity in rad/s."""
        return self.rate_map(coordinates) @ _vector(rate)

    def coordinate_rate(self, coordinates: ArrayLike, omega: ArrayLike) -> Array:
        """Recover native rates; refuse singular or ill-conditioned inverses."""
        return np.asarray(
            np.linalg.solve(self._invertible_map(coordinates), _vector(omega)),
            dtype=np.float64,
        )

    def angular_acceleration(
        self, coordinates: ArrayLike, rate: ArrayLike, acceleration: ArrayLike
    ) -> Array:
        """E qddot + Edot qdot; includes the convective acceleration term."""
        matrix = self.rate_map(coordinates)
        velocity = _vector(rate)
        derivative = np.zeros((3, 3))
        prefix_velocity = np.zeros(3)
        for index in range(3):
            derivative[:, index] = np.cross(prefix_velocity, matrix[:, index])
            prefix_velocity += matrix[:, index] * velocity[index]
        return matrix @ _vector(acceleration) + derivative @ velocity

    def coordinate_acceleration(
        self, coordinates: ArrayLike, rate: ArrayLike, alpha: ArrayLike
    ) -> Array:
        """Recover qddot from parent alpha, subtracting convective acceleration."""
        bias = self.angular_acceleration(coordinates, rate, np.zeros(3))
        return self.coordinate_rate(coordinates, _vector(alpha) - bias)

    def coordinate_effort(self, coordinates: ArrayLike, moment: ArrayLike) -> Array:
        """tau = E.T moment preserves instantaneous virtual work."""
        return self.rate_map(coordinates).T @ _vector(moment)

    def parent_moment(self, coordinates: ArrayLike, effort: ArrayLike) -> Array:
        """Recover moment from scalar torques; refuses nonunique inverses."""
        return np.asarray(
            np.linalg.solve(self._invertible_map(coordinates).T, _vector(effort)),
            dtype=np.float64,
        )

    def coordinates(self, quaternion_wxyz: ArrayLike, reference: ArrayLike) -> Array:
        """Nearest nonsingular Euler branch to explicit prior coordinates.

        Both Tait-Bryan branches and 2*pi winding are considered. Continuity
        requires sufficiently sampled motion; a quaternion alone loses winding.
        """
        quaternion = _vector(quaternion_wxyz, 4)
        prior = _vector(reference)
        if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-10, rtol=0):
            raise ValueError("Expected a unit quaternion")
        matrix = quat_to_matrix(quaternion)
        # Detect lock before scipy chooses a nonunique angle and emits a warning.
        first_axis = "XYZ".index(self.axes[0])
        last_axis = "XYZ".index(self.axes[2])
        sine = abs(float(matrix[first_axis, last_axis]))
        if 1 - sine <= 4 * (1 / self.max_condition) ** 2:
            raise SingularChartError(
                "Quaternion lies at an ill-conditioned native chart inverse"
            )
        primary = Rotation.from_matrix(matrix).as_euler(self.axes)
        alternate = np.array(
            [primary[0] + np.pi, np.pi - primary[1], primary[2] + np.pi]
        )
        candidates = [
            item + 2 * np.pi * np.round((prior - item) / (2 * np.pi))
            for item in (primary, alternate)
        ]
        result = min(candidates, key=lambda item: float(np.linalg.norm(item - prior)))
        self._invertible_map(result)
        return result
