"""Hashed payloads survive last-bit BLAS/libm drift (#14397, owner option A)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from src.shared.python.core.canonical_floats import (
    CANONICAL_ABSOLUTE_RESOLUTION,
    CANONICAL_SIGNIFICANT_DIGITS,
    canonical_floats,
)
from src.shared.python.model_appearance.schema import physics_spec_sha256

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = _REPO_ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def _ulp_drift(obj, steps: int):
    """``obj`` with every non-zero float moved ``steps`` ulps (sign kept)."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        out = obj
        for _ in range(abs(steps)):
            out = math.nextafter(out, math.inf if steps > 0 else -math.inf)
        return out
    if isinstance(obj, dict):
        return {k: _ulp_drift(v, steps) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ulp_drift(v, steps) for v in obj]
    return obj


def test_precision_sits_above_the_observed_drift() -> None:
    # Observed cross-kernel drift: <= 2.9e-14 absolute, ~1e-16 relative on
    # order-one values; cancellation residuals ~4e-18 absolute.
    assert CANONICAL_SIGNIFICANT_DIGITS == 12
    assert CANONICAL_ABSOLUTE_RESOLUTION == 1e-12


def test_canonical_floats_folds_ulp_noise_and_near_zero_residuals() -> None:
    value = 0.12345678901234
    assert canonical_floats(math.nextafter(value, 1.0)) == canonical_floats(value)
    assert canonical_floats(6.37e-17) == canonical_floats(5.55e-17) == 0.0
    # Cancellation residuals measured under AVX-512 vs generic OpenBLAS.
    assert canonical_floats(4.1269346367e-08) == canonical_floats(4.12693463674e-08)
    assert canonical_floats(-7.64085819758e-07) == canonical_floats(-7.64085819754e-07)
    assert math.copysign(1.0, canonical_floats(-0.0)) == 1.0
    assert math.copysign(1.0, canonical_floats(-1e-13)) == 1.0
    assert canonical_floats({"a": [1.0, 2.5e-13, True, "x", 3]}) == {
        "a": [1.0, 0.0, True, "x", 3]
    }
    assert math.isnan(canonical_floats(math.nan))
    assert canonical_floats(math.inf) == math.inf


def test_canonical_floats_keeps_real_changes() -> None:
    assert canonical_floats(0.1234567890) != canonical_floats(0.1234567891)
    assert canonical_floats(2e-12) != 0.0


@pytest.mark.parametrize("steps", [1, -1, 4])
def test_physics_spec_hash_ignores_ulp_drift(steps: int) -> None:
    spec = json.loads(_SPEC.read_bytes())
    drifted = _ulp_drift(copy.deepcopy(spec), steps)
    assert drifted != spec
    assert physics_spec_sha256(drifted) == physics_spec_sha256(spec)


def test_physics_spec_hash_still_sees_a_physical_change() -> None:
    spec = json.loads(_SPEC.read_bytes())
    physical = copy.deepcopy(spec)
    physical["gravity_m_s2"] = [0.0, 0.0, -9.80665 + 1e-9]
    assert physics_spec_sha256(physical) != physics_spec_sha256(spec)


def test_physics_spec_hash_ignores_full_precision_digests() -> None:
    spec = json.loads(_SPEC.read_bytes())
    revision = {"version": 1, "parent_physics_spec_sha256": "a" * 64}
    base = {**spec, "spec_revision": revision}
    restamped = {
        **base,
        "upper_body_sha256": "0" * 64,
        "spec_revision": {**revision, "parent_spec_canonical_sha256": "1" * 64},
    }
    assert physics_spec_sha256(restamped) == physics_spec_sha256(base)
    other_parent = {**base, "spec_revision": {**revision, "version": 2}}
    assert physics_spec_sha256(other_parent) != physics_spec_sha256(base)
