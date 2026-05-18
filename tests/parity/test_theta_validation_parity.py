"""Cross-engine theta-coefficient validation parity (issue #4252).

Each engine's ``simulate_with_coefficients`` and ``fit_swing_<engine>``
entry point now delegates to
:func:`src.shared.python.motion_matching.validate_theta.validate_theta`,
so wrong-length / non-finite / out-of-bounds ``theta`` produces a
consistent :class:`ValueError` with a descriptive message regardless of
which engine the caller picks.

This file pins that contract:

1. The shared validator catches each failure mode with a useful message.
2. Each engine's ``simulate_with_coefficients`` (when its optional
   physics dependency is installed) raises :class:`ValueError` -- not
   ``RuntimeError``, not silent ``NaN`` -- on the same inputs.

The parity-of-error-behaviour assertion is the cross-engine guarantee
listed in CROSS_ENGINE_PARITY_SPEC.md §2.2.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from src.shared.python.motion_matching.validate_theta import (
    COEFFS_PER_JOINT,
    DEFAULT_THETA_BOUND_TABLE,
    validate_theta,
)

# --------------------------------------------------------------------------- #
# 1. Direct unit tests on the shared validator                                #
# --------------------------------------------------------------------------- #


class TestValidateThetaUnit:
    """Pin the shared validator's behaviour against §2.2."""

    def test_accepts_valid_flat_vector(self) -> None:
        theta = np.zeros(3 * COEFFS_PER_JOINT, dtype=np.float64)
        out = validate_theta(theta, n_joints=3)
        assert out.shape == (21,)
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]

    def test_accepts_2d_n_joints_by_seven(self) -> None:
        theta = np.zeros((4, COEFFS_PER_JOINT), dtype=np.float64)
        out = validate_theta(theta, n_joints=4)
        assert out.shape == (28,)

    def test_rejects_wrong_length_with_descriptive_message(self) -> None:
        theta = np.zeros(20, dtype=np.float64)  # not a multiple of 7
        with pytest.raises(ValueError, match=r"length 20 != n_joints\*7"):
            validate_theta(theta, n_joints=3)

    def test_rejects_correct_multiple_but_wrong_n_joints(self) -> None:
        theta = np.zeros(14, dtype=np.float64)  # 2 * 7
        with pytest.raises(ValueError, match=r"length 14 != n_joints\*7 = 3"):
            validate_theta(theta, n_joints=3)

    def test_rejects_nan(self) -> None:
        theta = np.zeros(7, dtype=np.float64)
        theta[3] = np.nan
        with pytest.raises(ValueError, match=r"non-finite.*NaN=1"):
            validate_theta(theta, n_joints=1)

    def test_rejects_inf(self) -> None:
        theta = np.zeros(7, dtype=np.float64)
        theta[0] = np.inf
        with pytest.raises(ValueError, match=r"non-finite.*Inf=1"):
            validate_theta(theta, n_joints=1)

    def test_bounds_pass_when_within(self) -> None:
        theta = np.zeros(7, dtype=np.float64)
        validate_theta(theta, n_joints=1, bounds=DEFAULT_THETA_BOUND_TABLE)

    def test_bounds_reject_when_violated(self) -> None:
        theta = np.zeros(7, dtype=np.float64)
        theta[0] = 1e9  # blow past the A bound (1000)
        with pytest.raises(ValueError, match=r"coefficient 'A'"):
            validate_theta(theta, n_joints=1, bounds=DEFAULT_THETA_BOUND_TABLE)

    def test_bounds_reject_negative_violation(self) -> None:
        theta = np.zeros(7, dtype=np.float64)
        theta[6] = -100.0  # G bound is +/-25
        with pytest.raises(ValueError, match=r"coefficient 'G'"):
            validate_theta(theta, n_joints=1, bounds=DEFAULT_THETA_BOUND_TABLE)

    def test_custom_name_in_error(self) -> None:
        theta = np.zeros(20, dtype=np.float64)
        with pytest.raises(ValueError, match=r"theta_optimal"):
            validate_theta(theta, n_joints=3, name="theta_optimal")

    def test_rejects_non_positive_n_joints(self) -> None:
        with pytest.raises(ValueError, match=r"n_joints"):
            validate_theta(np.zeros(7), n_joints=0)

    def test_rejects_malformed_bounds_pair(self) -> None:
        with pytest.raises(TypeError, match=r"bounds\['A'\]"):
            validate_theta(
                np.zeros(7),
                n_joints=1,
                bounds={"A": "not-a-tuple"},  # type: ignore[arg-type]
            )

    def test_rejects_lo_greater_than_hi(self) -> None:
        with pytest.raises(ValueError, match=r"lo > hi"):
            validate_theta(np.zeros(7), n_joints=1, bounds={"A": (5.0, -5.0)})

    def test_empirical_scaled_bounds_accepted(self) -> None:
        """PR #4278 ``coefficient_bound_strategy='empirical'`` toggle path."""
        scaled = {
            letter: (lo * 0.5, hi * 0.5)
            for letter, (lo, hi) in DEFAULT_THETA_BOUND_TABLE.items()
        }
        theta = np.zeros(7, dtype=np.float64)
        validate_theta(theta, n_joints=1, bounds=scaled)


# --------------------------------------------------------------------------- #
# 2. Per-engine integration: ValueError on bad theta, regardless of backend   #
# --------------------------------------------------------------------------- #


def _maybe_import(modname: str) -> Any | None:
    """Return the module or ``None`` if any optional dep is missing."""
    try:
        return importlib.import_module(modname)
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        return None


def _expect_value_error(sim_fn: Any, theta: Any, *, pattern: str) -> None:
    """Call ``sim_fn(theta)`` and assert it raises ``ValueError`` matching.

    If the optional engine binding is missing (e.g. ``pinocchio`` /
    ``mujoco`` / ``opensim`` / ``pydrake`` not installed in this Python),
    the call raises ``ImportError``/``ModuleNotFoundError`` BEFORE the
    validator runs; we ``pytest.skip`` in that case so the parity
    contract is exercised wherever the engine is actually available.
    """
    import re

    try:
        sim_fn(theta)
    except ValueError as exc:
        assert re.search(pattern, str(exc)), (
            f"engine raised ValueError but message {str(exc)!r} did not match {pattern!r}"
        )
        return
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
        pytest.skip(f"engine binding unavailable: {exc}")
    pytest.fail("engine did not raise ValueError on bad theta")


_ENGINE_MODULES = (
    "src.engines.physics_engines.mujoco.python.motion_matching.simulate",
    "src.engines.physics_engines.drake.python.motion_matching.simulate",
    "src.engines.physics_engines.pinocchio.python.motion_matching.simulate",
    "src.engines.physics_engines.opensim.python.motion_matching.simulate",
)
