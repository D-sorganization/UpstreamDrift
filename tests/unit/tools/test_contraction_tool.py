from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from src.tools.contraction.verifier import (
    ContractionVerifier,
    compute_floquet_multipliers,
    linear_system_floquet_multipliers,
)

pytestmark = pytest.mark.unit


def test_contraction_verifier_estimates_stable_rate() -> None:
    verifier = ContractionVerifier(decay_rate=1.75, horizon=1.0, n_steps=80, seed=7)

    result = verifier.verify(n_trials=8, perturbation_scale=1e-3)

    assert result.n_trials == 8
    assert result.estimated_rate > 1.6
    assert result.estimated_rate < 1.9
    assert result.is_contracting


def test_estimate_contraction_rate_returns_float() -> None:
    rate = ContractionVerifier(decay_rate=0.5).estimate_contraction_rate(
        n_trials=4,
        perturbation_scale=1e-3,
    )

    assert rate > 0.0


def test_floquet_helpers_compute_multipliers() -> None:
    multipliers = linear_system_floquet_multipliers(
        system_matrix=np.diag([-1.0, -2.0]),
        period=0.5,
    )

    assert np.allclose(multipliers, np.array([np.exp(-0.5), np.exp(-1.0)]))
    assert np.allclose(compute_floquet_multipliers(np.diag([0.5, 0.25])), [0.5, 0.25])


def test_contraction_cli_outputs_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tools.contraction",
            "measure",
            "--decay-rate",
            "1.25",
            "--trials",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["is_contracting"] is True
    assert payload["estimated_rate"] > 1.0
