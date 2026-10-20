"""Freshness comparison for generated Simscape grip-contact JSON (#13334)."""

from __future__ import annotations

import copy

import pytest

from src.shared.python.grip_contact.generated_freshness import (
    FRESHNESS_SCALE_RTOL,
    generated_json_matches,
)

_PAYLOAD = {
    "schema_version": "x/v1",
    "kernel_m_sha256": "ab12",
    "flag": True,
    "count": 3,
    "noise": [1.0e-16, -2.0e-16, 0.0],
    "coeffs": [[0.047, -12.5], [3.0, 1.0e-9]],
    "scalar": 0.25,
}


@pytest.mark.unit
def test_identical_payload_matches() -> None:
    assert generated_json_matches(_PAYLOAD, copy.deepcopy(_PAYLOAD))


@pytest.mark.unit
def test_last_ulp_noise_is_tolerated() -> None:
    """Cross-environment numpy/BLAS noise (measured below 1e-9 of scale) passes."""
    noisy = copy.deepcopy(_PAYLOAD)
    noisy["coeffs"][0][0] += 8.0e-8 * 1.0e-2  # well under 1e-8 of the 12.5 scale
    noisy["noise"][1] = 3.0e-16
    assert generated_json_matches(_PAYLOAD, noisy)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("coeffs", 0, 1), -12.5 * (1.0 + 1.0e-6)),
        (("scalar",), 0.25 * (1.0 + 1.0e-6)),
        (("count",), 4),
        (("flag",), False),
        (("kernel_m_sha256",), "ab13"),
    ],
)
def test_real_changes_are_detected(path: tuple, value: object) -> None:
    changed = copy.deepcopy(_PAYLOAD)
    node = changed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert not generated_json_matches(_PAYLOAD, changed)


@pytest.mark.unit
def test_structure_changes_are_detected() -> None:
    longer = copy.deepcopy(_PAYLOAD)
    longer["noise"].append(0.0)
    assert not generated_json_matches(_PAYLOAD, longer)
    extra = copy.deepcopy(_PAYLOAD)
    extra["new_key"] = 1
    assert not generated_json_matches(_PAYLOAD, extra)


@pytest.mark.unit
def test_tolerance_contract() -> None:
    assert pytest.approx(1.0e-8) == FRESHNESS_SCALE_RTOL
    with pytest.raises(ValueError, match="scale_rtol"):
        generated_json_matches(_PAYLOAD, _PAYLOAD, scale_rtol=0.0)
