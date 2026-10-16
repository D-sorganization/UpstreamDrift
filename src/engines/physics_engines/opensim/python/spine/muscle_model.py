"""Thelen to Millard muscle swap for rigid-tendon solves (#12761).

The rigid-tendon static optimisation and joint-reaction stages set
``ignore_tendon_compliance = true`` on every muscle.  OpenSim 4.6 does not
implement that flag for ``Thelen2003Muscle`` (``initSystem`` raises), and the
published Bruno 2015 full-body model carries only Thelen muscles.  The swap
below rewrites each ``Thelen2003Muscle`` element as a
``Millard2012EquilibriumMuscle`` that keeps the muscle name, path, maximum
isometric force, optimal fibre length, tendon slack length, pennation angle,
maximum contraction velocity and the activation time constants.  Thelen-only
curve parameters are dropped and the Millard defaults apply, so the force
curves of the swapped muscle are OpenSim's Millard curves, not the Thelen
curves of the source file.  This is a declared modelling approximation;
receipts must state it (see :data:`SWAP_NOTE`).
"""

from __future__ import annotations

from pathlib import Path
import defusedxml.ElementTree as ET

__all__ = [
    "MUSCLE_MODEL_POLICIES",
    "SWAP_NOTE",
    "check_muscle_model_policy",
    "millard_equivalent_model",
    "thelen_muscle_count",
]

#: ``native`` keeps the model's muscles (a Thelen model then cannot run the
#: rigid-tendon solves); ``millard_equivalent`` applies the swap below.
MUSCLE_MODEL_POLICIES = ("native", "millard_equivalent")

SWAP_NOTE = (
    "Thelen2003Muscle swapped for Millard2012EquilibriumMuscle "
    "(same path, F0, lopt, lts, pennation, vmax; Millard default curves)"
)

_THELEN = "Thelen2003Muscle"
_MILLARD = "Millard2012EquilibriumMuscle"
_KEEP = frozenset(
    {
        "GeometryPath",
        "Appearance",
        "appliesForce",
        "isDisabled",
        "max_isometric_force",
        "optimal_fiber_length",
        "tendon_slack_length",
        "pennation_angle_at_optimal",
        "max_contraction_velocity",
        "activation_time_constant",
        "deactivation_time_constant",
        "min_control",
        "max_control",
    }
)


def millard_equivalent_model(
    model_path: str | Path, out_path: str | Path
) -> Path | None:
    """Write ``out_path`` with every Thelen muscle swapped; ``None`` if none.

    Preconditions: ``model_path`` is an existing OpenSim ``.osim`` file.
    Postconditions: when a path is returned the file holds no
    ``Thelen2003Muscle`` element and the muscle count and order are unchanged;
    when ``None`` is returned nothing was written.

    Raises:
        ValueError: ``model_path`` does not exist.
    """
    src = Path(model_path)
    if not src.is_file():
        raise ValueError(f"model file not found: {src}")
    tree = ET.parse(src)
    nodes = list(tree.iter(_THELEN))
    if not nodes:
        return None
    for node in nodes:
        node.tag = _MILLARD
        for child in list(node):
            if child.tag not in _KEEP and not child.tag.startswith("socket_"):
                node.remove(child)
    out = Path(out_path)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def thelen_muscle_count(model_path: str | Path) -> int:
    """Number of ``Thelen2003Muscle`` elements in the ``.osim`` file."""
    src = Path(model_path)
    if not src.is_file():
        raise ValueError(f"model file not found: {src}")
    return sum(1 for _ in ET.parse(src).iter(_THELEN))


def check_muscle_model_policy(policy: object) -> str:
    """Return ``policy`` if it is a known muscle-model policy.

    Raises:
        TypeError: ``policy`` is not a str.
        ValueError: ``policy`` is not in :data:`MUSCLE_MODEL_POLICIES`.
    """
    if not isinstance(policy, str):
        raise TypeError("muscle_model_policy must be a str")
    if policy not in MUSCLE_MODEL_POLICIES:
        raise ValueError(
            f"muscle_model_policy must be one of {MUSCLE_MODEL_POLICIES}, "
            f"got {policy!r}"
        )
    return policy
