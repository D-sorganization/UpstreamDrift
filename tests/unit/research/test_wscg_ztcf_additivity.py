"""ZTCF additivity on the WSCG best-match MuJoCo model (Issue #14438).

Definitions follow the canonical ZTCF/ZVCF of
``src/shared/python/simulation_backends/ztcf_zvcf.py`` (owner decision, epic
#14497 / #14494, matching the original MATLAB). All are pointwise, at the same
configuration ``q``:

- full: state ``(q, v)`` with the applied control ``u``;
- ZTCF: the drift, ``(q, v)`` with ``u = 0``, constraint reactions recomputed;
- ZVCF: the control component, torques on with velocity and gravity off. On
  this constrained model it carries the reaction change,
  ``M^-1 (B u + J_c^T (lambda - lambda_0))``; bare ``M^-1 B u`` is not used;
- zero-velocity, zero-torque: ``(q, 0)`` with ``u = 0`` (gravity and
  constraint reactions only). It is not ZVCF.

Frames and signs: every quantity is a generalized acceleration in MuJoCo's
joint coordinates for the pinned model (rad/s^2 for hinges, m/s^2 and rad/s^2
for the free joints), positive along each joint axis as declared in the MJCF,
with world gravity along -z. Generalized forces are MuJoCo's ``qfrc_*`` in the
same coordinates.

With these names "ZTCF + ZVCF reconstructs the full response" holds, with
gravity counted once:

    a_full = a_ZTCF + a_ZVCF,  a_ZVCF = M^-1 (B u + J_c^T (lambda - lambda_0))
    a_ZTCF - a_00 = M^-1 [(passive - bias)(q, v) - (passive - bias)(q, 0)
                          + J_c^T (lambda_0 - lambda_00)]

where ``a_00`` is the zero-velocity, zero-torque acceleration. The identity is
pointwise only: integrated counterfactual trajectories do not generally sum
(see ``simulation_backends.superposition``). This is a model-scenario result,
not empirical validation.
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


def test_ztcf_plus_zvcf_is_the_full_response(series: dict[str, np.ndarray]) -> None:
    assert np.all(series["zvcf_norm"] > 0.0), "torques must be applied"
    assert np.all(series["additivity_residual"] <= _bound(series))


def test_ztcf_minus_zero_velocity_zero_torque_is_the_velocity_drift(
    series: dict[str, np.ndarray],
) -> None:
    assert np.max(series["speed"]) > 1.0, "the run must leave rest"
    assert np.all(series["velocity_residual"] <= _bound(series))


def test_checked_in_summary_matches_a_fresh_run(series: dict[str, np.ndarray]) -> None:
    summary = json.loads((REPO_ROOT / SUMMARY_PATH).read_text(encoding="utf-8"))
    assert summary["samples"] == len(series["time_s"])
    assert summary["max_additivity_residual"] <= summary["declared_tolerance"]["abs"]
    np.testing.assert_allclose(
        summary["median_zvcf_share"],
        np.median(series["zvcf_norm"] / series["full_norm"]),
        rtol=1e-6,
    )


def test_pinned_model_revision_is_enforced() -> None:
    with pytest.raises(PinnedModelDriftError, match="revision"):
        load_pinned_model(expected_revision="sha256:" + "0" * 64)
