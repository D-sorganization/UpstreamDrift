"""Regression tests for ``CanonicalFitResult`` deprecated aliases.

Issues #4275 and #4276 reported that the migration to
``CanonicalFitResult`` dropped two backward-compatibility shims that
existing engine call sites still use:

  * ``theta``         — Pinocchio / OpenSim / Drake fit-swing tests
  * ``mujoco_version`` — MuJoCo provenance contract

Both must be readable from a ``CanonicalFitResult`` and emit a
``DeprecationWarning`` so callers are nudged toward the canonical names.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from src.shared.python.motion_matching import FitResult as PackageFitResult
from src.shared.python.motion_matching import SimFitResult
from src.shared.python.motion_matching.api_contracts import (
    FitResult as ApiFitResult,
)
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.leaderboard import LeaderboardRow
from src.shared.python.motion_matching.sim_out import FitResult as LegacySimFitResult

pytestmark = [pytest.mark.unit]


def _make_result(**overrides: object) -> CanonicalFitResult:
    base = {
        "theta_optimal": np.zeros(7, dtype=np.float64),
        "final_cost": 0.0,
        "final_rmse_m": 0.0,
        "solver_status": "success",
        "iterations": 1,
        "n_evaluations": 1,
        "wall_clock_s": 0.0,
        "message": "ok",
        "history": (),
        "method": "lm",
        "git_commit": "abcdef",
        "engine_version": "3.1.4",
        "target_hash": "deadbeef",
        "timestamp_utc": "1970-01-01T00:00:00Z",
    }
    base.update(overrides)
    return CanonicalFitResult(**base)  # type: ignore[arg-type]


def test_theta_alias_returns_theta_optimal_and_warns() -> None:
    """Reading ``result.theta`` returns ``theta_optimal`` and warns."""
    expected = np.arange(7, dtype=np.float64)
    result = _make_result(theta_optimal=expected)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed = result.theta

    np.testing.assert_array_equal(observed, expected)
    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "Expected DeprecationWarning when accessing the legacy `theta` alias"


def test_mujoco_version_alias_returns_engine_version_and_warns() -> None:
    """Reading ``result.mujoco_version`` returns ``engine_version`` and warns."""
    result = _make_result(engine_version="3.2.1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed = result.mujoco_version

    assert observed == "3.2.1"
    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "Expected DeprecationWarning when accessing the legacy `mujoco_version` alias"


def test_package_fit_result_preserves_legacy_sim_alias() -> None:
    """Package-level ``FitResult`` remains a compatibility alias for sim output."""
    assert PackageFitResult is SimFitResult
    assert SimFitResult is LegacySimFitResult


def test_role_specific_fit_result_names_are_explicit() -> None:
    """Non-canonical result payloads have names that describe their role."""
    assert LeaderboardRow.__name__ == "LeaderboardRow"
    assert ApiFitResult.__name__ == "ApiFitResult"


def test_api_fit_result_can_convert_to_canonical() -> None:
    """The API-contract fit payload has an explicit canonical conversion path."""
    api_result = ApiFitResult(
        coefficients=np.array([1.0, 2.0], dtype=np.float64),
        final_loss=3.5,
        metadata={"engine": "mujoco", "time_s": 0.25, "solver": "lm"},
    )

    canonical = CanonicalFitResult.from_api_contract(
        api_result,
        git_commit="abcdef0",
        engine_version="mujoco-3.6",
        target_hash="target",
        timestamp_utc="2026-06-11T00:00:00Z",
    )

    np.testing.assert_array_equal(canonical.theta_optimal, api_result.coefficients)
    assert canonical.final_cost == pytest.approx(3.5)
    assert canonical.solver_status == "success"
    assert canonical.method == "lm"
    assert canonical.meta["api_contract_metadata"] == api_result.metadata
