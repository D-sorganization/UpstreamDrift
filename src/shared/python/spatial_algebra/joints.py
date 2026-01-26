"""
Joint kinematics and motion subspaces.

Implements joint transformation and motion subspace calculations
for various joint types.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

S_RX = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
S_RY = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
S_RZ = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
S_PX = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
S_PY = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
S_PZ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

for _vec in (S_RX, S_RY, S_RZ, S_PX, S_PY, S_PZ):
    _vec.flags.writeable = False

JOINT_AXIS_INDICES = {
    "Rx": 0,
    "Ry": 1,
    "Rz": 2,
    "Px": 3,
    "Py": 4,
    "Pz": 5,
}


def jcalc(
    jtype: str, q: float, out: npt.NDArray[np.float64] | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """
    Calculate joint transform and motion subspace.

    Returns:
        (xj_transform, s_subspace, dof_idx)
    """
    if out is None:
        xj_transform = np.zeros((6, 6), dtype=np.float64)
    else:
        xj_transform = out
        xj_transform.fill(0.0)

    dof_idx = -1

    if jtype == "Rx":
        c = np.cos(q)
        s = np.sin(q)
        xj_transform[0, 0] = 1.0
        xj_transform[1, 1] = c
        xj_transform[1, 2] = -s
        xj_transform[2, 1] = s
        xj_transform[2, 2] = c
        xj_transform[3, 3] = 1.0
        xj_transform[4, 4] = c
        xj_transform[4, 5] = -s
        xj_transform[5, 4] = s
        xj_transform[5, 5] = c
        s_subspace = S_RX
        dof_idx = 0
    elif jtype == "Ry":
        c = np.cos(q)
        s = np.sin(q)
        xj_transform[0, 0] = c
        xj_transform[0, 2] = s
        xj_transform[1, 1] = 1.0
        xj_transform[2, 0] = -s
        xj_transform[2, 2] = c
        xj_transform[3, 3] = c
        xj_transform[3, 5] = s
        xj_transform[4, 4] = 1.0
        xj_transform[5, 3] = -s
        xj_transform[5, 5] = c
        s_subspace = S_RY
        dof_idx = 1
    elif jtype == "Rz":
        c = np.cos(q)
        s = np.sin(q)
        xj_transform[0, 0] = c
        xj_transform[0, 1] = -s
        xj_transform[1, 0] = s
        xj_transform[1, 1] = c
        xj_transform[2, 2] = 1.0
        xj_transform[3, 3] = c
        xj_transform[3, 4] = -s
        xj_transform[4, 3] = s
        xj_transform[4, 4] = c
        xj_transform[5, 5] = 1.0
        s_subspace = S_RZ
        dof_idx = 2
    elif jtype == "Px":
        np.fill_diagonal(xj_transform, 1.0)
        xj_transform[4, 2] = q
        xj_transform[5, 1] = -q
        s_subspace = S_PX
        dof_idx = 3
    elif jtype == "Py":
        np.fill_diagonal(xj_transform, 1.0)
        xj_transform[3, 2] = -q
        xj_transform[5, 0] = q
        s_subspace = S_PY
        dof_idx = 4
    elif jtype == "Pz":
        np.fill_diagonal(xj_transform, 1.0)
        xj_transform[3, 1] = q
        xj_transform[4, 0] = -q
        s_subspace = S_PZ
        dof_idx = 5
    else:
        msg = (
            f"Unsupported joint type: {jtype}. Supported types: Rx, Ry, Rz, Px, Py, Pz"
        )
        raise ValueError(msg)

    return xj_transform, s_subspace, dof_idx
