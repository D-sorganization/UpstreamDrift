"""Canonical control-name registry for the 3D Simscape kinetic model.

This module is the SINGLE source of truth for the names of the polynomial
coefficients that drive ``GolfSwing3D_Kinetic.slx``. Both the MATLAB scaffold
(``motion_matching/shared/+control_names``) and the Python MachineLearning
workflow (``MachineLearning/export_torque_polynomials.py``) must read from
here to avoid drift.

Issue #4042 — extracted from ``export_torque_polynomials.py``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from src.shared.python.core.contracts import postcondition, precondition

TORQUE_TO_POLYNOMIAL_BASE: Final[dict[str, str]] = {
    "LScapLogs_ActuatorTorqueX": "LScapInputX",
    "LScapLogs_ActuatorTorqueY": "LScapInputY",
    "RScapLogs_ActuatorTorqueX": "RScapInputX",
    "RScapLogs_ActuatorTorqueY": "RScapInputY",
    "LSLogs_ActuatorTorqueX": "LSInputX",
    "LSLogs_ActuatorTorqueY": "LSInputY",
    "LSLogs_ActuatorTorqueZ": "LSInputZ",
    "RSLogs_ActuatorTorqueX": "RSInputX",
    "RSLogs_ActuatorTorqueY": "RSInputY",
    "RSLogs_ActuatorTorqueZ": "RSInputZ",
    "SpineLogs_ActuatorTorqueX": "SpineInputX",
    "SpineLogs_ActuatorTorqueY": "SpineInputY",
    "HipLogs_TranslationForceXInput": "TranslationInputX",
    "HipLogs_TranslationForceYInput": "TranslationInputY",
    "HipLogs_TranslationForceZInput": "TranslationInputZ",
    "HipLogs_HipTorqueXInput": "HipInputX",
    "HipLogs_HipTorqueYInput": "HipInputY",
    "HipLogs_HipTorqueZInput": "HipInputZ",
    "LScapTorqueXInput": "LScapInputX",
    "LScapTorqueYInput": "LScapInputY",
    "RScapTorqueXInput": "RScapInputX",
    "RScapTorqueYInput": "RScapInputY",
    "LSTorqueXInput": "LSInputX",
    "LSTorqueYInput": "LSInputY",
    "LSTorqueZInput": "LSInputZ",
    "RSTorqueXInput": "RSInputX",
    "RSTorqueYInput": "RSInputY",
    "RSTorqueZInput": "RSInputZ",
    "HipTorqueXInput": "HipInputX",
    "HipTorqueYInput": "HipInputY",
    "HipTorqueZInput": "HipInputZ",
}

COEFFICIENT_LETTERS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F", "G")
N_COEFFS_PER_JOINT: Final[int] = 7


def n_total_coefficients() -> int:
    """Return the total number of unique polynomial coefficients.

    Equals ``n_unique_bases * 7`` — the number of distinct scalars the
    Simulink model consumes per swing.
    """
    return len(set(TORQUE_TO_POLYNOMIAL_BASE.values())) * N_COEFFS_PER_JOINT


def joint_names() -> list[str]:
    """Return the ordered list of canonical torque-column names (registry keys).

    Postcondition: result is non-empty and contains no duplicates.
    """
    names = list(TORQUE_TO_POLYNOMIAL_BASE.keys())
    if len(names) != len(set(names)):
        raise ValueError("TORQUE_TO_POLYNOMIAL_BASE contains duplicate joint names")
    return names


@precondition(
    lambda joint, letter: joint in TORQUE_TO_POLYNOMIAL_BASE,
    message="joint must be a key in TORQUE_TO_POLYNOMIAL_BASE",
)
@precondition(
    lambda joint, letter: letter in COEFFICIENT_LETTERS,
    message="letter must be one of COEFFICIENT_LETTERS",
)
def coefficient_name(joint: str, letter: str) -> str:
    """Return the polynomial-coefficient name for ``(joint, letter)``.

    For example, ``coefficient_name("HipTorqueXInput", "A")`` returns
    ``"HipInputXA"``.
    """
    return f"{TORQUE_TO_POLYNOMIAL_BASE[joint]}{letter}"


def _unique_polynomial_bases() -> list[str]:
    """Return the ordered list of unique polynomial-input base names.

    The registry maps multiple torque-column aliases to the same polynomial
    base (e.g. both ``LSLogs_ActuatorTorqueX`` and ``LSTorqueXInput`` map to
    ``LSInputX``). The deduplicated, first-seen order is the canonical
    coefficient layout consumed by ``GolfSwing3D_Kinetic.slx``.
    """
    seen: set[str] = set()
    out: list[str] = []
    for base in TORQUE_TO_POLYNOMIAL_BASE.values():
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


@postcondition(
    lambda result: len(result) == len(set(result)),
    message="all_coefficient_names must be unique",
)
@postcondition(
    lambda result: len(result) == len(_unique_polynomial_bases()) * N_COEFFS_PER_JOINT,
    message="all_coefficient_names length must equal n_unique_bases * 7",
)
def all_coefficient_names() -> list[str]:
    """Return the full ordered list of polynomial-coefficient names.

    Ordering: outer loop over the deduplicated polynomial-base order
    (first-seen insertion order in ``TORQUE_TO_POLYNOMIAL_BASE``), inner
    loop over ``COEFFICIENT_LETTERS``. The result has
    ``n_unique_bases * 7`` entries with no duplicates.
    """
    out: list[str] = []
    for base in _unique_polynomial_bases():
        for letter in COEFFICIENT_LETTERS:
            out.append(f"{base}{letter}")
    return out


def manifest_sha256() -> str:
    """Return a sha256 over the canonical (joint, base, letters) manifest.

    The manifest is JSON-serialized with sorted keys disabled (ordering is
    significant) and no whitespace, then hashed. Any reordering, addition, or
    rename of a registry entry changes this digest — the locked-value test
    acts as a tripwire.
    """
    payload = {
        "torque_to_polynomial_base": list(TORQUE_TO_POLYNOMIAL_BASE.items()),
        "coefficient_letters": list(COEFFICIENT_LETTERS),
        "n_coeffs_per_joint": N_COEFFS_PER_JOINT,
    }
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


__all__ = [
    "COEFFICIENT_LETTERS",
    "N_COEFFS_PER_JOINT",
    "TORQUE_TO_POLYNOMIAL_BASE",
    "all_coefficient_names",
    "coefficient_name",
    "joint_names",
    "manifest_sha256",
    "n_total_coefficients",
]
