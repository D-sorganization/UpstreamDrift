"""Torque-actuated neck for the MyoFullBody swing analysis (issue #11689).

MyoFullBody has no neck joints or neck muscles: the head is rigid on the torso.
The spec skeleton has three neck coordinates, so without an actuator the whole
neck effort is reserve by construction (100 % in every earlier receipt).  This
module adds a documented, *bounded* torque actuator per neck coordinate instead:
a signed pair of unit-moment-arm actuators with the maximum voluntary isometric
neck moment as capacity.  Saturation is therefore reported as reserve, not
hidden, and the actuator is *not* a muscle: no activation of it is a muscle
finding, and the head's motion is still not represented in MyoFullBody.

Capacities are healthy young male maxima (about 30 years) from the neck-strength
literature, rounded; where flexion and extension differ the smaller (flexion)
value is used for both signs because the sign convention of the spec coordinate
is not asserted here.  Verify against the papers before any publication use.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.myofullbody.redundancy import FrameBasis

SOURCE = (
    "Maximum isometric neck moments of young men: Vasavada, Li and Delp (2001) "
    "Spine 26(17):1904-1909; Vasavada, Danaraj and Siegmund (2008) J Biomech "
    "41:114-121.  Flexion 30, extension 52, lateral bending 36, axial rotation "
    "15 N m (means, rounded)."
)
CAPACITY_NM: dict[str, float] = {
    "NeckInputX": 30.0,  # flexion/extension: the smaller (flexion) maximum
    "NeckInputY": 36.0,  # lateral bending
    "NeckInputZ": 15.0,  # axial rotation
}
SIGNS = (("pos", 1.0), ("neg", -1.0))


def _present(order: tuple[str, ...], columns: list[int]) -> list[tuple[int, str]]:
    """``(position in columns, coordinate name)`` of the neck coordinates present."""
    return [(i, order[c]) for i, c in enumerate(columns) if order[c] in CAPACITY_NM]


def actuator_names(order: tuple[str, ...], columns: list[int]) -> list[str]:
    """Names of the neck torque actuators, a ``pos``/``neg`` pair per coordinate."""
    return [
        f"neck_torque_{name}_{sign}"
        for _, name in _present(order, columns)
        for sign, _ in SIGNS
    ]


def augment(
    basis: FrameBasis, order: tuple[str, ...], columns: list[int]
) -> FrameBasis:
    """Basis with the neck torque actuators appended after the muscles.

    Returns ``basis`` itself when ``columns`` hold no neck coordinate.  The
    appended actuators have zero passive force, ``active`` equal to the documented
    capacity, and a unit moment arm of the matching sign on their coordinate.
    """
    require(basis.moment.shape[1] == len(columns), "moment/columns size mismatch")
    present = _present(order, columns)
    if not present:
        return basis
    extra_active, extra_moment = [], []
    for i, name in present:
        for _, sign in SIGNS:
            row = np.zeros(len(columns))
            row[i] = sign
            extra_active.append(CAPACITY_NM[name])
            extra_moment.append(row)
    return FrameBasis(
        np.concatenate([basis.active, extra_active]),
        np.concatenate([basis.passive, np.zeros(len(extra_active))]),
        np.vstack([basis.moment, np.array(extra_moment)]),
        basis.phi,
    )


def split(activation: np.ndarray, n_muscles: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a solved activation vector into ``(muscles, neck torque actuators)``.

    Raises:
        ValueError: if ``n_muscles`` exceeds the vector length.
    """
    require(0 <= n_muscles <= activation.shape[-1], "n_muscles outside the vector")
    return activation[..., :n_muscles], activation[..., n_muscles:]


def torque(basis: FrameBasis, activation: np.ndarray, n_muscles: int) -> np.ndarray:
    """Spec generalised force of the neck torque actuators alone (zero if none)."""
    _, act = split(activation, n_muscles)
    rows = basis.moment[n_muscles:]
    return rows.T @ (act * basis.active[n_muscles:])
