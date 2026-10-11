"""Lift kinetics from a same-input bundle in any engine (LIFT-7, #11747).

For chosen states of a lift bundle, each engine reports

* the ground-reaction breakdown of the shared contact law evaluated on its own
  kinematics (GCV-1: per-foot and net force, centre of pressure and free
  moment; :func:`~src.shared.python.biomechanics.ground_reaction_wrenches.\
ground_reaction_breakdown`), and
* the per-hand wrench on the grip from its closure KKT multiplier (GCV-7/8),
  or an unavailable analysis with a reason when the engine has none.

Moments about the centre of mass are not reported here: the same-input
adapters do not expose a system CoM, and a substituted point would read as a
real value.  They stay with the overlay provider until every engine can
supply its own CoM.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.biomechanics.grip_extraction import unavailable_analysis
from src.shared.python.biomechanics.grip_wrench import GripAnalysis, HandWrench
from src.shared.python.biomechanics.ground_reaction import FootReaction
from src.shared.python.biomechanics.ground_reaction_wrenches import (
    ground_reaction_breakdown,
)
from src.shared.python.force_overlay.bundle_provider import (
    contact_sample_wrenches,
    sphere_bodies,
)
from src.shared.python.motion_matching.same_input import (
    ALL_ENGINES,
    InputBundle,
    VectorPlant,
)

Array = NDArray[np.float64]

# Engines whose same-input adapter extracts the grip wrench from the KKT
# multiplier.  Others report the hand wrench as unavailable.
GRIP_ENGINES: tuple[str, ...] = ("mujoco", "drake")
FEET: tuple[str, ...] = ("left", "right")
_GRIP_FAILURES = (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError)
# Reference point for the breakdown's CoM-moment fields, which are discarded.
_UNUSED_COM = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class KineticsSample:
    """Ground reaction and grip wrench of one engine at one bundle state."""

    engine: str
    index: int
    time_s: float
    per_foot: Mapping[str, FootReaction]
    net: FootReaction
    grip: GripAnalysis


def _grip(
    engine: str, plant: VectorPlant, q: Array, v: Array, tau: Array
) -> GripAnalysis:
    if engine not in GRIP_ENGINES:
        return unavailable_analysis(
            f"{engine} same-input adapter has no closure-multiplier grip analysis"
        )
    try:
        return plant.grip_analysis(q, v, tau)
    except _GRIP_FAILURES as exc:
        return unavailable_analysis(f"{engine} grip solve failed: {exc}")


def _checked_indices(bundle: InputBundle, indices: Sequence[int]) -> list[int]:
    rows = [int(k) for k in indices]
    if not rows or any(not 0 <= k <= bundle.steps for k in rows):
        raise ValueError(
            f"indices must be a non-empty selection of states in [0, {bundle.steps}]"
        )
    return rows


def bundle_kinetics(
    engine: str, bundle: InputBundle, indices: Sequence[int]
) -> list[KineticsSample]:
    """Kinetics of ``engine`` at the bundle reference states ``indices``.

    State ``k`` uses ``reference_q[k]``, ``reference_v[k]`` and the effort
    applied from it (``efforts[min(k, steps - 1)]``).

    Postconditions: one sample per index, in order; a foot without load has no
    CoP or free moment (``None``), never zero.

    Raises:
        ValueError: For an unknown engine or indices outside the bundle.
    """
    if engine not in ALL_ENGINES:
        raise ValueError(f"engine must be one of {ALL_ENGINES}, got {engine!r}")
    rows = _checked_indices(bundle, indices)
    plant = VectorPlant(engine, bundle.spec_bytes)
    bodies = sphere_bodies(json.loads(bundle.spec_bytes))
    samples = []
    for k in rows:
        q, v = bundle.reference_q[k], bundle.reference_v[k]
        tau = bundle.efforts[min(k, bundle.steps - 1)]
        contacts = contact_sample_wrenches(
            plant.contact_samples(q, v), bodies, source=f"{engine}:shared_contact_law"
        )
        breakdown = ground_reaction_breakdown(contacts, _UNUSED_COM)
        samples.append(
            KineticsSample(
                engine=engine,
                index=k,
                time_s=k * float(bundle.dt_s),
                per_foot=dict(breakdown.per_foot),
                net=breakdown.net,
                grip=_grip(engine, plant, q, v, tau),
            )
        )
    return samples


def _gap(a: object, b: object) -> float | None:
    """Largest absolute component difference; ``None`` if either is missing."""
    if a is None or b is None:
        return None
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    return float(np.abs(left - right).max())


def _hand_force(hand: HandWrench | None) -> object:
    return None if hand is None else hand.force_on_club_n


def kinetics_differences(
    a: KineticsSample, b: KineticsSample
) -> dict[str, float | None]:
    """Per-quantity maximum absolute difference between two samples.

    Keys: ``grf_force_<foot>_n``, ``cop_<foot>_m``, ``free_moment_<foot>_nm``
    for ``left``, ``right`` and ``net``; ``hand_left_force_n``,
    ``hand_right_force_n``, ``grip_net_force_n`` and ``grip_couple_nm``.
    A quantity unavailable in either sample is ``None``.

    Raises:
        ValueError: When the samples are not of the same bundle state.
    """
    if a.index != b.index:
        raise ValueError(f"samples are of different states: {a.index} != {b.index}")
    out: dict[str, float | None] = {}
    pairs = [(foot, a.per_foot[foot], b.per_foot[foot]) for foot in FEET]
    pairs.append(("net", a.net, b.net))
    for key, ra, rb in pairs:
        out[f"grf_force_{key}_n"] = _gap(ra.force_n, rb.force_n)
        out[f"cop_{key}_m"] = _gap(ra.cop_m, rb.cop_m)
        out[f"free_moment_{key}_nm"] = _gap(ra.free_moment_nm, rb.free_moment_nm)
    out["hand_left_force_n"] = _gap(_hand_force(a.grip.left), _hand_force(b.grip.left))
    out["hand_right_force_n"] = _gap(
        _hand_force(a.grip.right), _hand_force(b.grip.right)
    )
    out["grip_net_force_n"] = _gap(a.grip.net_force_n, b.grip.net_force_n)
    out["grip_couple_nm"] = _gap(
        a.grip.couple_at_midpoint_nm, b.grip.couple_at_midpoint_nm
    )
    return out
