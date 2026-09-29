"""Tests for the canonical control-name registry (issue #4042).

Locks the polynomial-coefficient ordering shared by Python and MATLAB. A
sha256 tripwire detects silent drift; a JSON fixture asserts byte-for-byte
equivalence with the MATLAB scaffold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.shared.python.core.contracts import PreconditionError
from src.shared.python.motion_matching.control_names import (
    COEFFICIENT_LETTERS,
    N_COEFFS_PER_JOINT,
    TORQUE_TO_POLYNOMIAL_BASE,
    all_coefficient_names,
    coefficient_name,
    joint_names,
    manifest_sha256,
    n_total_coefficients,
)

# Locked manifest digest — regenerate ONLY when the registry intentionally
# changes, and update the MATLAB fixture in lockstep.
EXPECTED_MANIFEST_SHA256 = (
    "e502f965b0eaa5f770bb1e17fa85e41c2a881c844d071b38bacd3c3e753247a4"
)
FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "control_names_matlab.json"
)


@pytest.mark.unit
def test_torque_to_polynomial_base_nonempty() -> None:
    assert len(TORQUE_TO_POLYNOMIAL_BASE) >= 18
    assert all(
        isinstance(k, str) and isinstance(v, str)
        for k, v in TORQUE_TO_POLYNOMIAL_BASE.items()
    )


@pytest.mark.unit
def test_coefficient_letters_are_ABCDEFG() -> None:
    assert COEFFICIENT_LETTERS == ("A", "B", "C", "D", "E", "F", "G")
    assert N_COEFFS_PER_JOINT == 7
    assert len(COEFFICIENT_LETTERS) == N_COEFFS_PER_JOINT


@pytest.mark.unit
def test_all_coefficient_names_unique() -> None:
    names = all_coefficient_names()
    assert len(names) == len(set(names))


@pytest.mark.unit
def test_all_coefficient_names_length_equals_njoints_times_7() -> None:
    # "njoints" here is the number of unique polynomial bases (the underlying
    # set of joint controls) — multiple torque-column aliases collapse to the
    # same polynomial base.
    n_unique_bases = len(set(TORQUE_TO_POLYNOMIAL_BASE.values()))
    assert len(all_coefficient_names()) == n_unique_bases * 7
    assert n_total_coefficients() == n_unique_bases * 7


@pytest.mark.unit
def test_manifest_sha256_locked() -> None:
    """Tripwire: any registry edit must intentionally update this digest."""
    assert manifest_sha256() == EXPECTED_MANIFEST_SHA256


@pytest.mark.unit
def test_joint_names_and_coefficient_name() -> None:
    js = joint_names()
    assert len(js) == len(TORQUE_TO_POLYNOMIAL_BASE)
    assert coefficient_name("HipTorqueXInput", "A") == "HipInputXA"
    with pytest.raises(PreconditionError):
        coefficient_name("not_a_joint", "A")
    with pytest.raises(PreconditionError):
        coefficient_name("HipTorqueXInput", "Z")


@pytest.mark.unit
def test_python_and_matlab_orderings_match_via_fixture() -> None:
    """MATLAB equivalence is asserted by the fixture, regenerable via
    ``tools/regen_control_names_fixture.m`` when MATLAB is available."""
    assert FIXTURE_PATH.exists(), f"missing fixture: {FIXTURE_PATH}"
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fixture["coefficient_letters"] == list(COEFFICIENT_LETTERS)
    assert fixture["all_coefficient_names"] == all_coefficient_names()
    assert fixture["manifest_sha256"] == manifest_sha256()
    assert fixture["torque_to_polynomial_base"] == [
        list(item) for item in TORQUE_TO_POLYNOMIAL_BASE.items()
    ]
