"""ZTCF additivity on the WSCG best-match MuJoCo model (Issue #14438).

Definitions follow the canonical nomenclature in
``docs/research/model_aware_matching/model_aware_matching.tex`` (section "ZTCF
and ZVCF as Exact Projections of the Regressor") and
``src/shared/python/simulation_backends/ztcf_zvcf.py``. All are pointwise, at
the same configuration ``q``:

- full: state ``(q, v)`` with the applied control ``u``;
- ZTCF: ``(q, v)`` with ``u = 0``, constraint reactions recomputed;
- ZVCF: ``(q, 0)`` with ``u = 0`` (gravity and constraint reactions only).

Frames and signs: every quantity is a generalized acceleration in MuJoCo's
joint coordinates for the pinned model (rad/s^2 for hinges, m/s^2 and rad/s^2
for the free joints), positive along each joint axis as declared in the MJCF,
with world gravity along -z. Generalized forces are MuJoCo's ``qfrc_*`` in the
same coordinates.

The issue's wording "ZTCF + ZVCF reconstructs the full response" conflicts with
those names: the literal sum counts the configuration (gravity) term twice. The
exact identities are asserted instead (supervisor-approved, pending Dieter):

    a_full = a_ZTCF + M^-1 (B u + J_c^T (lambda - lambda_0))
    a_ZTCF - a_ZVCF = M^-1 [(passive - bias)(q, v) - (passive - bias)(q, 0)
                            + J_c^T (lambda_0 - lambda_00)]

and the literal sum is characterized, not bounded. This is a model-scenario
result, not empirical validation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from scripts.research.proximal_distal_energy.wscg_model_selection import REPO_ROOT
from scripts.research.proximal_distal_energy.wscg_ztcf_additivity import (
    ABS_TOLERANCE,
    RELATIVE_TOLERANCE,
    SUMMARY_PATH,
    PinnedModelDriftError,
    load_pinned_model,
    run_additivity,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


@pytest.fixture(scope="module")
def series() -> dict[str, np.ndarray]:
    return run_additivity()


def _bound(series: dict[str, np.ndarray]) -> np.ndarray:
    return ABS_TOLERANCE + RELATIVE_TOLERANCE * series["full_norm"]


def test_full_equals_ztcf_plus_control_part(series: dict[str, np.ndarray]) -> None:
    assert np.all(series["control_part_norm"] > 0.0), "torques must be applied"
    assert np.all(series["split_residual"] <= _bound(series))


def test_ztcf_minus_zvcf_is_the_velocity_drift(series: dict[str, np.ndarray]) -> None:
    assert np.max(series["speed"]) > 1.0, "the run must leave rest"
    assert np.all(series["velocity_residual"] <= _bound(series))


def test_literal_ztcf_plus_zvcf_double_counts_the_zvcf_term(
    series: dict[str, np.ndarray],
) -> None:
    # a_ZTCF + a_ZVCF - a_full == a_ZVCF - control part, identically ...
    assert np.all(series["literal_identity_residual"] <= _bound(series))
    # ... so the literal residual is structurally nonzero, not a tolerance issue.
    assert np.median(series["literal_residual"] / series["full_norm"]) > 0.1


def test_checked_in_summary_matches_a_fresh_run(series: dict[str, np.ndarray]) -> None:
    summary = json.loads((REPO_ROOT / SUMMARY_PATH).read_text(encoding="utf-8"))
    assert summary["samples"] == len(series["time_s"])
    assert summary["max_split_residual"] <= summary["declared_tolerance"]["abs"] * 10
    np.testing.assert_allclose(
        summary["median_literal_residual_ratio"],
        np.median(series["literal_residual"] / series["full_norm"]),
        rtol=1e-6,
    )


def test_pinned_model_revision_is_enforced() -> None:
    with pytest.raises(PinnedModelDriftError, match="revision"):
        load_pinned_model(expected_revision="sha256:" + "0" * 64)
