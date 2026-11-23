"""
Joint kinematics and motion subspaces.

Implements joint transformation and motion subspace calculations
for various joint types.

Revolute and prismatic branches both use Featherstone coordinate transforms
(RBDA §2.8): passive rotation ``E = R^T`` via :func:`xrot`, and translation
via :func:`xlt`.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .transforms import xlt, xrot

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


def _featherstone_e_rot(jtype: str, q: float) -> npt.NDArray[np.float64]:
    """Passive 3x3 rotation for revolute joint ``jtype`` at angle ``q``."""
    c = math.cos(q)
    s = math.sin(q)
    if jtype == "Rx":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=np.float64)
    if jtype == "Ry":
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)
    if jtype == "Rz":
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    msg = f"unsupported revolute type: {jtype}"
    raise ValueError(msg)


def jcalc(
    jtype: str, q: float, out: npt.NDArray[np.float64] | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """
    Calculate joint transform and motion subspace.

    Returns:
        (xj_transform, s_subspace, dof_idx)
    """
    if jtype is None:
        raise ValueError("jtype must be provided")
    if not math.isfinite(q):
        msg = f"q must be finite, got {q!r}"
        raise ValueError(msg)
    xj_transform = (
        np.zeros((6, 6), dtype=np.float64) if out is None else out  # type: ignore[assignment]
    )

    dof_idx = -1

    if jtype == "Rx":
        xj_transform[:] = xrot(_featherstone_e_rot("Rx", q))
        s_subspace = S_RX
        dof_idx = 0
    elif jtype == "Ry":
        xj_transform[:] = xrot(_featherstone_e_rot("Ry", q))
        s_subspace = S_RY
        dof_idx = 1
    elif jtype == "Rz":
        xj_transform[:] = xrot(_featherstone_e_rot("Rz", q))
        s_subspace = S_RZ
        dof_idx = 2
    elif jtype == "Px":
        xj_transform[:] = xlt(np.array([q, 0.0, 0.0], dtype=np.float64))
        s_subspace = S_PX
        dof_idx = 3
    elif jtype == "Py":
        xj_transform[:] = xlt(np.array([0.0, q, 0.0], dtype=np.float64))
        s_subspace = S_PY
        dof_idx = 4
    elif jtype == "Pz":
        xj_transform[:] = xlt(np.array([0.0, 0.0, q], dtype=np.float64))
        s_subspace = S_PZ
        dof_idx = 5
    else:
        msg = (
            f"Unsupported joint type: {jtype}. Supported types: Rx, Ry, Rz, Px, Py, Pz"
        )
        raise ValueError(msg)

    return xj_transform, s_subspace, dof_idx
