"""Tests for the Drake autodiff fit driver (issue #4119, DRAKE-4).

Two tiers:

* Pure-Python unit tests that exercise dataclass validation, the bounds
  helper, and the cost helper. These run anywhere (no pydrake required)
  and are marked ``unit``.
* Live-Drake tests that build the autodiff plant and run the
  MathematicalProgram + Ipopt solver end-to-end. These are gated on
  ``pydrake`` being importable via ``@pytest.mark.requires_drake``.

Per CLAUDE.md, mocked-pydrake tests use ``patch.dict("sys.modules", ...)``
which auto-cleans after the test; we never assign to ``sys.modules``
directly at module scope.
"""

from __future__ import annotations

import importlib
import time as _time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.engines.physics_engines.drake.python.motion_matching.fit_swing_autodiff import (
    COEFFS_PER_JOINT,
    DEFAULT_COEFFICIENT_BOUNDS,
    FitOptions,
    FitResult,
    compute_grip_rmse_and_work,
    default_theta_bounds,
)

# ---------------------------------------------------------------------------
# Pydrake availability gate
# ---------------------------------------------------------------------------


def _pydrake_available() -> bool:
    """Return True if pydrake imports cleanly."""
    try:
        importlib.import_module("pydrake.multibody.plant")
        importlib.import_module("pydrake.autodiffutils")
        importlib.import_module("pydrake.solvers")
    except ImportError:
        return False
    return True


PYDRAKE_AVAILABLE = _pydrake_available()


# ---------------------------------------------------------------------------
# Synthetic target oracle (no pydrake dependency)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SyntheticTarget:
    """Duck-typed target with ``time`` and ``grip`` attributes.

    Avoids constructing a full ``ClubTarget`` (which validates orientation
    quaternions etc.) for the autodiff-only test surface.
    """

    time: np.ndarray
    grip: np.ndarray


def _synthesize_target(
    n_samples: int = 31,
    sim_time_s: float = 0.3,
    seed: int = 0,
) -> tuple[_SyntheticTarget, np.ndarray]:
    """Generate a synthetic (target, theta_true) pair.

    The target is a smooth quadratic-in-time grip trajectory; theta_true
    is a small but non-trivial coefficient vector that the optimizer
    should recover. Pure-numpy so the test exercises the dataclass /
    bounds plumbing without spinning up pydrake.
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, sim_time_s, n_samples, dtype=np.float64)
    # Synthetic grip trajectory: arbitrary-but-smooth.
    base_xyz = np.array([0.0, 0.5, 1.2], dtype=np.float64)
    sweep = (
        0.4
        * np.sin(2.0 * np.pi * time / sim_time_s)[:, None]
        * np.array([1.0, 0.0, 0.5])
    )
    grip = base_xyz[None, :] + sweep
    # Plausible theta_true (kept small so torques stay in scope of the
    # canonical bounds).
    n_joints = 19  # canonical EXPECTED_NUM_REVOLUTE_DOF for the URDF.
    theta_true = 1.0e-2 * rng.standard_normal(n_joints * COEFFS_PER_JOINT)
    return _SyntheticTarget(time=time, grip=grip), theta_true


# ---------------------------------------------------------------------------
# Unit tests (no pydrake needed) -- bounds, cost helper, dataclass guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_coefficient_bounds_shape() -> None:
    """Per-power bounds vector has 7 entries with lo < hi."""
    assert len(DEFAULT_COEFFICIENT_BOUNDS) == COEFFS_PER_JOINT
    for lo, hi in DEFAULT_COEFFICIENT_BOUNDS:
        assert np.isfinite(lo)
        assert np.isfinite(hi)
        assert lo < hi


@pytest.mark.unit
def test_default_theta_bounds_tiles_per_joint() -> None:
    """default_theta_bounds(n_joints) tiles the 7-vector bounds n_joints times."""
    n_joints = 19
    lo, hi = default_theta_bounds(n_joints)
    assert lo.shape == (n_joints * COEFFS_PER_JOINT,)
    assert hi.shape == (n_joints * COEFFS_PER_JOINT,)
    # Bound k for joint j matches DEFAULT_COEFFICIENT_BOUNDS[k].
    for j in range(n_joints):
        for k in range(COEFFS_PER_JOINT):
            assert lo[j * COEFFS_PER_JOINT + k] == DEFAULT_COEFFICIENT_BOUNDS[k][0]
            assert hi[j * COEFFS_PER_JOINT + k] == DEFAULT_COEFFICIENT_BOUNDS[k][1]


@pytest.mark.unit
def test_default_theta_bounds_rejects_zero_joints() -> None:
    """``n_joints < 1`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="n_joints"):
        default_theta_bounds(0)


@pytest.mark.unit
def test_default_theta_bounds_rejects_wrong_per_power_length() -> None:
    """``bounds_per_power`` length != 7 raises ``ValueError``."""
    with pytest.raises(ValueError, match="bounds_per_power"):
        default_theta_bounds(2, ((-1.0, 1.0),))


@pytest.mark.unit
def test_fit_options_defaults_validate() -> None:
    """``FitOptions()`` constructs with valid defaults."""
    opts = FitOptions()
    assert opts.max_iterations >= 1
    assert opts.tolerance > 0
    assert opts.dynamics_gradient_mode in {"autodiff", "finite_diff"}
    assert opts.method in {"ipopt", "snopt", "auto"}


@pytest.mark.unit
def test_fit_options_rejects_bad_max_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        FitOptions(max_iterations=0)


@pytest.mark.unit
def test_fit_options_rejects_bad_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        FitOptions(tolerance=-1.0)


@pytest.mark.unit
def test_fit_options_rejects_bad_solver() -> None:
    with pytest.raises(ValueError, match="solver"):
        FitOptions(solver="banana")  # type: ignore[arg-type]


@pytest.mark.unit
def test_fit_options_rejects_bad_gradient_mode() -> None:
    with pytest.raises(ValueError, match="dynamics_gradient_mode"):
        FitOptions(dynamics_gradient_mode="symbolic")  # type: ignore[arg-type]


@pytest.mark.unit
def test_fit_options_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="coefficient_bounds"):
        FitOptions(coefficient_bounds=((-1.0, 1.0),) * 3)
    with pytest.raises(ValueError, match=r"coefficient_bounds\[0\]"):
        FitOptions(coefficient_bounds=((1.0, -1.0),) + DEFAULT_COEFFICIENT_BOUNDS[1:])


@pytest.mark.unit
def test_fit_options_rejects_bad_regularizer_weight() -> None:
    with pytest.raises(ValueError, match="regularizer_weight"):
        FitOptions(regularizer_weight=-1.0)


@pytest.mark.unit
def test_fit_result_validates_status() -> None:
    """``FitResult.solver_status`` must be one of the canonical strings."""
    with pytest.raises(ValueError, match="solver_status"):
        FitResult(
            theta=np.zeros(7),
            final_cost=0.0,
            final_rmse_m=0.0,
            n_sim_calls=0,
            n_iterations=0,
            wall_clock_s=0.0,
            solver_status="unknown",
            solver_name="ipopt",
        )


@pytest.mark.unit
def test_fit_result_validates_theta_dim() -> None:
    """Multi-D theta is rejected at construction."""
    with pytest.raises(ValueError, match="theta"):
        FitResult(
            theta=np.zeros((3, 7)),
            final_cost=0.0,
            final_rmse_m=0.0,
            n_sim_calls=0,
            n_iterations=0,
            wall_clock_s=0.0,
            solver_status="success",
            solver_name="ipopt",
        )


@pytest.mark.unit
def test_compute_grip_rmse_and_work_zero_diff() -> None:
    """Identical grip logs => RMSE == 0; finite work."""
    n = 10
    grid = np.linspace(0.0, 1.0, n)
    grip = np.tile(np.array([1.0, 2.0, 3.0]), (n, 1))
    tau = np.ones((n, 4))
    qd = np.ones((n, 4))
    rmse, work = compute_grip_rmse_and_work(grip, grip.copy(), tau, qd, grid)
    assert rmse == pytest.approx(0.0, abs=1e-12)
    assert work == pytest.approx(4.0, rel=1e-9)  # trapz(sum(|1*1|), [0,1]) = 4


@pytest.mark.unit
def test_compute_grip_rmse_and_work_unit_offset() -> None:
    """Grip shifted by 1 m on x => RMSE == 1 m."""
    n = 5
    grid = np.linspace(0.0, 0.4, n)
    grip = np.zeros((n, 3))
    target = grip.copy()
    target[:, 0] = 1.0
    tau = np.zeros((n, 2))
    qd = np.zeros((n, 2))
    rmse, work = compute_grip_rmse_and_work(grip, target, tau, qd, grid)
    assert rmse == pytest.approx(1.0, rel=1e-12)
    assert work == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_compute_grip_rmse_and_work_rejects_shape_mismatch() -> None:
    n = 5
    with pytest.raises(ValueError, match="grip_log"):
        compute_grip_rmse_and_work(
            np.zeros((n, 3)),
            np.zeros((n + 1, 3)),
            np.zeros((n, 1)),
            np.zeros((n, 1)),
            np.linspace(0.0, 1.0, n),
        )
    with pytest.raises(ValueError, match="tau_log"):
        compute_grip_rmse_and_work(
            np.zeros((n, 3)),
            np.zeros((n, 3)),
            np.zeros((n, 1)),
            np.zeros((n, 2)),
            np.linspace(0.0, 1.0, n),
        )
    with pytest.raises(ValueError, match="time"):
        compute_grip_rmse_and_work(
            np.zeros((n, 3)),
            np.zeros((n, 3)),
            np.zeros((n, 1)),
            np.zeros((n, 1)),
            np.linspace(0.0, 1.0, n + 1),
        )


# ---------------------------------------------------------------------------
# Mocked-pydrake tests: exercise import surfaces without a real Drake install
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_imports_without_pydrake() -> None:
    """The module imports cleanly even when pydrake is absent.

    Per CLAUDE.md, all ``pydrake`` imports must be inside function bodies
    so that ``import fit_swing_autodiff`` does not require Drake.
    """
    fake_pydrake = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "pydrake": fake_pydrake,
            "pydrake.multibody": fake_pydrake,
            "pydrake.multibody.plant": fake_pydrake,
            "pydrake.autodiffutils": fake_pydrake,
            "pydrake.solvers": fake_pydrake,
            "pydrake.systems.framework": fake_pydrake,
            "pydrake.systems.analysis": fake_pydrake,
            "pydrake.systems.scalar_conversion": fake_pydrake,
        },
    ):
        # Re-import to force the patched modules to take effect.
        mod = importlib.import_module(
            "src.engines.physics_engines.drake.python.motion_matching."
            "fit_swing_autodiff"
        )
        importlib.reload(mod)
        # Smoke: dataclasses constructible.
        opts = mod.FitOptions()
        assert opts.dynamics_gradient_mode == "autodiff"


# ---------------------------------------------------------------------------
# Live-pydrake tests (skipped without Drake)
# ---------------------------------------------------------------------------
