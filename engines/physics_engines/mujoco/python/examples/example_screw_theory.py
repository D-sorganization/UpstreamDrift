"""
Example: Screw Theory Operations

Demonstrates screw axes, exponential/logarithmic maps, and adjoint transforms.
"""

import numpy as np
from mujoco_humanoid_golf.screw_theory import (
    adjoint_transform,
    exponential_map,
    logarithmic_map,
    screw_axis,
    screw_to_transform,
)


def main() -> None:
    """Docstring for main."""

    # Example 1: Pure rotation
    axis = np.array([0, 0, 1])
    point = np.array([1, 0, 0])
    pitch_1: float = 0.0
    theta = np.pi / 2  # 90 degrees

    s_screw = screw_axis(axis, point, pitch_1)

    t_transform = exponential_map(s_screw, theta)

    # Example 2: Pure translation
    axis = np.array([1, 0, 0])
    point = np.array([0, 0, 0])
    pitch_2: float = float("inf")
    theta = 1.5

    s_screw = screw_axis(axis, point, pitch_2)

    t_transform = exponential_map(s_screw, theta)

    # Example 3: Screw motion
    axis = np.array([0, 0, 1])
    point = np.array([0, 0, 0])
    pitch_3: float = 0.1  # 0.1 m per radian
    theta = 2 * np.pi  # One full rotation

    s_screw = screw_axis(axis, point, pitch_3)

    t_transform = exponential_map(s_screw, theta)

    # Example 4: Logarithmic map
    s_orig = screw_axis(np.array([0, 0, 1]), np.array([1, 0, 0]), 0)
    theta_orig = np.pi / 3
    t_transform = exponential_map(s_orig, theta_orig)

    s_recovered, theta_recovered = logarithmic_map(t_transform)

    screw_orig = s_orig * theta_orig
    screw_recovered = s_recovered * theta_recovered
    np.linalg.norm(screw_orig - screw_recovered)

    # Example 5: Adjoint transformation
    t_transform = screw_to_transform(
        np.array([0, 0, 1]),
        np.array([1, 0, 0]),
        0,
        np.pi / 4,
    )
    ad_transform = adjoint_transform(t_transform)

    v_b = np.array([0, 0, 1, 0, 0, 0])  # Angular velocity about z
    ad_transform @ v_b

    # Example 6: Composition property
    t1 = exponential_map(np.array([0, 0, 1, 0, 0, 0]), np.pi / 4)
    t2 = exponential_map(np.array([0, 0, 0, 1, 0, 0]), 0.5)

    ad1 = adjoint_transform(t1)
    ad2 = adjoint_transform(t2)
    ad_comp = adjoint_transform(t1 @ t2)
    ad_product = ad1 @ ad2

    np.linalg.norm(ad_comp - ad_product, "fro")


if __name__ == "__main__":
    main()
