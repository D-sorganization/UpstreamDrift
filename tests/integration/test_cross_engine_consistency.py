"""Integration test for verifying consistency across physics engines.

Uses the shared pendulum fixtures (see ``tests/fixtures/fixtures_lib.py``,
re-exported through ``tests/integration/conftest.py``) to load the
gold-standard simple-pendulum URDF into every *available* physics engine and
assert their forward-dynamics drift accelerations agree.

Previously this module contained only a helper and **no test** (#7052). It now
runs a real value-asserting cross-engine consistency gate that skips cleanly
when fewer than two engines are installed.

Per Guideline M2/P3: Cross-engine validation with explicit tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.logging_pkg.logging_config import get_logger

from tests.fixtures.fixtures_lib import (
    _check_drake_available,
    _check_mujoco_available,
    _check_pinocchio_available,
    compute_accelerations,
    set_identical_state,
)

logger = get_logger(__name__)

# Tolerance multiplier for triangulation outlier detection
# A relaxed 10x threshold is used to identify engines with systematic deviations
TRIANGULATION_TOLERANCE_MULTIPLIER = 10.0

# Cross-engine forward-dynamics agreement tolerance (rad/s^2). The simple
# pendulum is a 1-DOF analytic system; independent engines should agree to
# well within 1e-3 rad/s^2 once realized to the same state.
ACCELERATION_AGREEMENT_TOL_RAD_S2 = 1e-3

pytestmark = [pytest.mark.integration]


def _get_available_engine_count() -> int:
    """Count available physics engines."""
    count = 0
    if _check_mujoco_available():
        count += 1
    if _check_drake_available():
        count += 1
    if _check_pinocchio_available():
        count += 1
    return count


_ENGINE_BACKEND_MODULE = {
    "mujoco": "mujoco",
    "drake": "pydrake",
    "pinocchio": "pinocchio",
}


def _drop_mock_backed_engines(engines: list) -> list:
    """Return only engines whose backend module is a real (non-mock) import.

    Another test in the same session can inject a ``MagicMock`` into
    ``sys.modules`` for an optional engine, which makes the availability probe
    report "installed" while the engine cannot actually simulate. Such engines
    are filtered out so this gate measures real cross-engine agreement.
    """
    import sys

    real: list = []
    for eng in engines:
        backend = getattr(eng, "engine", None)
        if backend is None:
            continue
        engine_type = str(getattr(backend, "engine_type", "")).lower()
        module_name = _ENGINE_BACKEND_MODULE.get(engine_type)
        module = sys.modules.get(module_name) if module_name else None
        if module is not None and type(module).__module__ == "unittest.mock":
            continue
        real.append(eng)
    return real


def test_available_engine_count_is_consistent() -> None:
    """The local availability count matches the per-engine probes."""
    expected = sum(
        (
            _check_mujoco_available(),
            _check_drake_available(),
            _check_pinocchio_available(),
        )
    )
    assert _get_available_engine_count() == expected


def test_cross_engine_drift_acceleration_agrees(
    all_available_pendulum_engines: list,
) -> None:
    """Every installed engine agrees on the pendulum drift acceleration.

    Sets an identical, off-equilibrium state (theta = 0.3 rad, v = 0) across
    all available engines, computes the zero-torque drift acceleration
    ``qacc = -M^-1 * bias`` via the shared helper, and asserts pairwise
    agreement within tolerance. Skips when < 2 engines are installed (handled
    by the ``all_available_pendulum_engines`` fixture).
    """
    engines = _drop_mock_backed_engines(all_available_pendulum_engines)
    if len(engines) < 2:
        pytest.skip(
            f"Need >= 2 real (non-mock) engines; have "
            f"{[getattr(e, 'name', '?') for e in engines]}"
        )

    q = np.array([0.3], dtype=np.float64)
    v = np.array([0.0], dtype=np.float64)
    set_identical_state(engines, q, v)

    accelerations = compute_accelerations(engines)
    assert len(accelerations) >= 2, (
        "Expected drift accelerations from at least two engines; "
        f"got {sorted(accelerations)}"
    )

    names = sorted(accelerations)
    reference = accelerations[names[0]]
    for name in names[1:]:
        qacc = accelerations[name]
        assert qacc.shape == reference.shape, (
            f"DOF mismatch: {name} {qacc.shape} vs {names[0]} {reference.shape}"
        )
        residual = float(np.max(np.abs(qacc - reference)))
        assert residual < ACCELERATION_AGREEMENT_TOL_RAD_S2, (
            f"Cross-engine drift acceleration disagreement between "
            f"{names[0]} and {name}: residual={residual:.3e} rad/s^2 "
            f"exceeds {ACCELERATION_AGREEMENT_TOL_RAD_S2:.1e} "
            f"({names[0]}={reference.tolist()}, {name}={qacc.tolist()})"
        )

    # The pendulum is released from rest off-vertical, so gravity must pull it
    # back toward equilibrium: drift acceleration is non-zero and finite.
    for name, qacc in accelerations.items():
        assert np.all(np.isfinite(qacc)), f"{name} produced non-finite qacc"
        assert float(np.abs(qacc[0])) > 1e-6, (
            f"{name} reported ~zero drift acceleration for an off-equilibrium "
            f"pendulum: {qacc.tolist()}"
        )
