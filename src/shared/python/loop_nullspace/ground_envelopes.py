"""Envelope quantities for the lower-body loop (NSX-60).

Factories for :func:`static_envelopes` over the ground-contact part of
``x = (tau, lambda, f_g)``: the net ground force, the load of a named group of
contacts (a foot), and the closure multiplier.  Each factory reads the frame's
decomposition, so frames with different active contact sets can share one
series.  Quantities are affine in the null coordinates ``z`` and are reported
as functionals of the declared family, never as measurements.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from src.shared.python.loop_nullspace.decomposition import Decomposition
from src.shared.python.loop_nullspace.envelopes import EnvelopeQuantity
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet

__all__ = ["closure_wrench_quantities", "ground_force_quantities"]

_AXES = "xyz"
_WELD_NAMES = ("fx", "fy", "fz", "tx", "ty", "tz")


def _decomposition(fs: FeasibleSet) -> Decomposition:
    dec = fs.decomposition
    if dec is None:
        raise ValueError("feasible set was built without a decomposition")
    return dec


def closure_wrench_quantities() -> Callable[[FeasibleSet], list[EnvelopeQuantity]]:
    """Factory of the six closure multipliers ``lambda`` (grip-frame wrench).

    ``lambda`` is the wrench the closing hand exerts on the club, row order
    ``[fx fy fz tx ty tz]`` (convention ``C-FRAME-GRIP``).  Raises
    ``ValueError`` from the factory unless the closure has six rows.
    """

    def build(fs: FeasibleSet) -> list[EnvelopeQuantity]:
        dec = _decomposition(fs)
        if dec.m != 6:
            raise ValueError(
                f"closure_wrench_quantities needs 6 multipliers, got {dec.m}"
            )
        lead = dec.n_actuated
        out = []
        for k, name in enumerate(_WELD_NAMES):
            force = k < 3
            out.append(
                EnvelopeQuantity(
                    f"lambda_{name}",
                    dec.N[lead + k],
                    float(dec.x_p[lead + k]),
                    "N" if force else "N m",
                    dec.scales.force if force else dec.scales.moment,
                )
            )
        return out

    return build


def ground_force_quantities(
    contacts_of: Callable[[FeasibleSet], Sequence[str]],
    groups: Mapping[str, Callable[[str], bool]] | None = None,
) -> Callable[[FeasibleSet], list[EnvelopeQuantity]]:
    """Factory of the net ground force and per-group vertical loads.

    ``contacts_of(fs)`` returns the contact names of the frame's ``f_g``
    blocks in order.  ``groups`` maps a label to a predicate on a contact name
    (for example a foot side); each adds the group's world-z load
    ``sum_k f_k . z`` (the contact normal is world z on a level floor).  The
    net force components ``ground_fx/fy/fz`` are always reported.  A group with
    no active member is the constant zero, never missing.
    """
    groups = dict(groups or {})

    def build(fs: FeasibleSet) -> list[EnvelopeQuantity]:
        dec = _decomposition(fs)
        names = list(contacts_of(fs))
        lead = dec.n_actuated + dec.m
        if dec.dim_x != lead + 3 * len(names):
            raise ValueError(
                f"{len(names)} contact names do not match dim_x={dec.dim_x}"
            )
        scale = dec.scales.force
        quantities = []
        for axis in range(3):
            rows = [lead + 3 * k + axis for k in range(len(names))]
            quantities.append(_sum_quantity(dec, f"ground_f{_AXES[axis]}", rows, scale))
        for label, member in groups.items():
            rows = [lead + 3 * k + 2 for k, n in enumerate(names) if member(n)]
            quantities.append(_sum_quantity(dec, label, rows, scale))
        return quantities

    return build


def _sum_quantity(
    dec: Decomposition, name: str, rows: Sequence[int], scale: float
) -> EnvelopeQuantity:
    coeff = dec.N[list(rows)].sum(axis=0) if rows else np.zeros(dec.nullity)
    const = float(dec.x_p[list(rows)].sum()) if rows else 0.0
    return EnvelopeQuantity(name, coeff, const, "N", scale)
