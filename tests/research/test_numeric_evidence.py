"""Contracts for portable publication-only numeric serialization."""

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.numeric_evidence import (
    canonicalize_published_numbers,
)


@pytest.mark.unit
def test_canonicalizer_rounds_nested_floats_without_changing_other_values() -> None:
    payload = {"values": (np.float64(1.23456789), -0.0), "decision": "full_rank"}

    assert canonicalize_published_numbers(payload) == {
        "values": [1.23457, 0.0],
        "decision": "full_rank",
    }


@pytest.mark.unit
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonicalizer_rejects_nonfinite_publication_values(value: float) -> None:
    with pytest.raises(ValueError, match="registered fixture"):
        canonicalize_published_numbers(value, context="registered fixture")


@pytest.mark.unit
def test_canonicalizer_requires_positive_precision() -> None:
    with pytest.raises(ValueError, match="significant_digits must be positive"):
        canonicalize_published_numbers(1.0, significant_digits=0)
