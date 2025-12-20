"""
Example: Featherstone Algorithms on 2-Link Robot

Demonstrates RNEA, CRBA, and ABA algorithms on a simple 2-link planar robot.
"""

import numpy as np
from mujoco_humanoid_golf.rigid_body_dynamics import aba, crba, rnea
from mujoco_humanoid_golf.spatial_algebra import mci, xlt

GRAVITY_M_S2 = 9.80665  # Standard gravity constant (m/s²)


def create_2link_model() -> dict:
    """Create a simple 2-link planar robot."""
    # Link parameters
    L1, L2 = 1.0, 0.8  # Lengths (m)  # noqa: N806 - Standard notation for link lengths
    m1, m2 = 1.0, 0.8  # Masses (kg)

    # Inertia of uniform density rod: I = (1/12)*m*L^2
    I1 = (1 / 12) * m1 * L1**2  # noqa: N806 - Standard notation for moment of inertia
    I2 = (1 / 12) * m2 * L2**2  # noqa: N806 - Standard notation for moment of inertia

    return {
        "NB": 2,
        "parent": np.array([-1, 0]),  # -1 means no parent
        "jtype": ["Rz", "Rz"],
        "gravity": np.array([0, 0, 0, 0, 0, -GRAVITY_M_S2]),
        "Xtree": [
            np.eye(6),  # Joint 1 at origin
            xlt(np.array([L1, 0, 0])),  # Joint 2 at end of link 1
        ],
        "I": [
            mci(m1, np.array([L1 / 2, 0, 0]), np.diag([0, 0, I1])),
            mci(m2, np.array([L2 / 2, 0, 0]), np.diag([0, 0, I2])),
        ],
    }


def main() -> None:
    """Docstring for main."""

    # Create model
    model = create_2link_model()

    # Test CRBA
    q = np.array([np.pi / 6, -np.pi / 4])

    H = crba(model, q)  # noqa: N806 - H is standard notation for mass matrix

    # Test RNEA
    q = np.array([np.pi / 4, -np.pi / 3])
    qd = np.array([0.5, -0.3])
    qdd = np.array([0.2, 0.1])

    rnea(model, q, qd, qdd)

    # Decompose torques
    tau_gravity = rnea(model, q, np.zeros(2), np.zeros(2))
    rnea(model, q, qd, np.zeros(2)) - tau_gravity
    H = crba(model, q)  # noqa: N806 - H is standard notation for mass matrix
    H @ qdd

    # Test ABA
    tau_applied = np.array([1.0, 0.5])

    qdd_computed = aba(model, q, qd, tau_applied)

    # Verify ABA and RNEA are inverses
    rnea(model, q, qd, qdd_computed)

    # Compare ABA with explicit inversion
    H = crba(model, q)  # noqa: N806 - H is standard notation for mass matrix
    tau_bias = rnea(model, q, qd, np.zeros(2))
    np.linalg.solve(H, tau_applied - tau_bias)


if __name__ == "__main__":
    main()
