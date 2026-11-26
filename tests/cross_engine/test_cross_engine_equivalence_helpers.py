"""Helper and metric unit tests for cross-engine equivalence work (issue #15245).

This module is **not** a CI gate. It holds shared helpers and smoke tests for:

- deterministic ``theta_truth`` generation (issue #4096 mirror),
- RMSE / quaternion metric utilities,
- safe ``simulate_with_coefficients`` lookup, and
- ``SimOut`` coercion for equivalence reporting.

The contact-free **gravity-only cross-engine equivalence gate** executed in CI is
``tests/motion_matching/test_cross_engine_equivalence.py`` (see
``.github/workflows/cross-engine-equivalence.yml`` job ``equivalence``). See
``docs/testing/integration-testing-strategy.md`` (Cross-Engine CI Claim Matrix).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

logger = logging.getLogger(__name__)

_N_JOINTS = 25
_COEFFS_PER_JOINT = 7
_THETA_LEN = _N_JOINTS * _COEFFS_PER_JOINT  # 175

_COEFF_AMPLITUDES = np.array(
    [1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0],
    dtype=float,
)


def _generate_theta_truth(seed: int = 42) -> np.ndarray:
    """Deterministic mirror of MATLAB ``generateRandomCoefficients(N)``."""
    if not np.isfinite(seed):
        raise ValueError("seed must be finite")
    rng = np.random.default_rng(seed)
    raw = rng.random(_THETA_LEN) - 0.5
    amplitudes = np.tile(2.0 * _COEFF_AMPLITUDES, _N_JOINTS)
    theta = raw * amplitudes
    for i in range(_THETA_LEN):
        amp = _COEFF_AMPLITUDES[i % _COEFFS_PER_JOINT]
        if not (-amp <= theta[i] <= amp):
            raise ValueError(f"theta[{i}] = {theta[i]} outside ±{amp} range")
    return theta


@dataclass(frozen=True)
class EngineSimResult:
    """Captured slice of ``SimOut`` for reporting helpers."""

    engine: str
    pose_name: str
    grip: np.ndarray
    grip_quat: np.ndarray
    clubhead: np.ndarray
    club_length: float


_ENGINE_MODULE_PATHS: dict[str, str] = {
    "simscape": "src.engines.simscape.python.motion_matching.simulate",
    "mujoco": "src.engines.physics_engines.mujoco.python.motion_matching.simulate",
    "drake": "src.engines.physics_engines.drake.python.motion_matching.simulate",
    "pinocchio": "src.engines.physics_engines.pinocchio.python.motion_matching.simulate",
    "opensim": "src.engines.physics_engines.opensim.python.motion_matching.simulate",
}


def _simulator_attr(
    module_path: str, attr: str = "simulate_with_coefficients"
) -> Callable[..., Any] | None:
    """Locate a registered engine's forward-sim entry point, or return None."""
    if module_path not in _ENGINE_MODULE_PATHS.values():
        raise ValueError("unregistered simulator module")
    try:
        module = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError, AttributeError, OSError) as exc:
        logger.debug("simulator module %s unavailable: %s", module_path, exc)
        return None
    return getattr(module, attr, None)


def _coerce_simout(
    raw: Any,
    engine: str,
    pose_name: str,
) -> EngineSimResult:
    """Reduce a ``SimOut``-like object to the slice equivalence reporting needs."""

    def _last(arr: Any, n: int) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            if a.shape != (n,):
                raise ValueError(
                    f"{engine}/{pose_name}: expected ({n},), got {a.shape}"
                )
            return a
        if a.ndim != 2 or a.shape[1] != n:
            raise ValueError(f"{engine}/{pose_name}: expected (N,{n}), got {a.shape}")
        return a[-1]

    def _get(name: str) -> Any:
        if hasattr(raw, name):
            return getattr(raw, name)
        if isinstance(raw, dict) and name in raw:
            return raw[name]
        raise AttributeError(f"{engine} SimOut missing field: {name}")

    grip = _last(_get("grip"), 3)
    grip_quat = _last(_get("grip_quat"), 4)
    clubhead = _last(_get("clubhead"), 3)
    club_length = float(np.linalg.norm(clubhead - grip))

    qn = float(np.linalg.norm(grip_quat))
    if not np.isfinite(qn) or qn <= 1e-6:
        raise ValueError(f"{engine}/{pose_name}: degenerate grip_quat with norm {qn}")
    grip_quat = grip_quat / qn

    return EngineSimResult(
        engine=engine,
        pose_name=pose_name,
        grip=grip,
        grip_quat=grip_quat,
        clubhead=clubhead,
        club_length=club_length,
    )


def _rmse_position(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two world positions in metres (per-sample Euclidean norm)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        return float(np.linalg.norm(a - b))
    per_sample = np.linalg.norm(a - b, axis=1)
    return float(np.sqrt(np.mean(per_sample**2)))


def _quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    """Angular distance between two unit quaternions, in degrees."""
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


@pytest.mark.unit
def test_theta_truth_is_deterministic() -> None:
    a = _generate_theta_truth(42)
    b = _generate_theta_truth(42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (_THETA_LEN,)


@pytest.mark.unit
def test_theta_truth_respects_amplitude_bounds() -> None:
    theta = _generate_theta_truth(42)
    for joint in range(_N_JOINTS):
        for k in range(_COEFFS_PER_JOINT):
            i = joint * _COEFFS_PER_JOINT + k
            amp = _COEFF_AMPLITUDES[k]
            assert -amp <= theta[i] <= amp, f"theta[{i}] = {theta[i]} outside ±{amp}"


@pytest.mark.unit
def test_theta_truth_rejects_non_finite_seed() -> None:
    with pytest.raises(ValueError, match="seed must be finite"):
        _generate_theta_truth(float("nan"))


@pytest.mark.unit
def test_nonzero_theta_polynomial_order_matches_across_engine_helpers() -> None:
    """Pure evaluators agree that column k is the coefficient on t^k."""
    from src.engines.physics_engines.drake.python.motion_matching.simulate import (
        evaluate_torque_polynomial as evaluate_drake_torque,
    )
    from src.engines.physics_engines.mujoco.python.motion_matching.torque_driver import (
        PolynomialTorqueDriver,
    )
    from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
        evaluate_polynomial_torque as evaluate_opensim_torque,
    )
    from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
        evaluate_polynomial_torque as evaluate_pinocchio_torque,
    )

    class FakeMuJoCoModel:
        def __init__(self, nu: int) -> None:
            self.nu = nu

    n_joints = 2
    theta = np.zeros((n_joints, _COEFFS_PER_JOINT), dtype=np.float64)
    theta[:, 0] = [1.0, -1.5]
    theta[:, 1] = [2.0, -3.0]
    theta[:, 6] = [0.5, -0.75]
    t = 0.25
    expected = theta[:, 0] + theta[:, 1] * t + theta[:, 6] * t**6

    mujoco_driver = PolynomialTorqueDriver(
        FakeMuJoCoModel(n_joints),
        theta,
        clip_to_ctrlrange=False,
    )

    np.testing.assert_allclose(mujoco_driver.evaluate(t), expected)
    np.testing.assert_allclose(
        evaluate_drake_torque(theta.reshape(-1), t, n_joints),
        expected,
    )
    np.testing.assert_allclose(evaluate_pinocchio_torque(theta, t), expected)
    np.testing.assert_allclose(evaluate_opensim_torque(theta, t), expected)


@pytest.mark.unit
def test_quat_angle_zero() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_angle_deg(q, q) < 1e-9


@pytest.mark.unit
def test_quat_angle_180() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_angle_deg(q, -q) < 1e-9


@pytest.mark.unit
def test_rmse_position_matches_norm_for_single_sample() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.003, 0.004, 0.0])
    assert abs(_rmse_position(a, b) - 0.005) < 1e-12


@pytest.mark.unit
def test_simulator_lookup_rejects_unregistered_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(_: str) -> Any:
        pytest.fail("Unregistered simulator must not be imported")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    with pytest.raises(ValueError, match="unregistered simulator module"):
        _simulator_attr("untrusted.module")


@dataclass
class _MockSimOut:
    grip: np.ndarray
    grip_quat: np.ndarray
    clubhead: np.ndarray


@pytest.mark.unit
def test_coerce_simout_from_dataclass_snapshot() -> None:
    raw = _MockSimOut(
        grip=np.array([0.1, 0.2, 0.3]),
        grip_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        clubhead=np.array([0.4, 0.2, 0.3]),
    )
    result = _coerce_simout(raw, "mujoco", "address")
    np.testing.assert_allclose(result.grip, raw.grip)
    np.testing.assert_allclose(result.clubhead, raw.clubhead)
    assert result.engine == "mujoco"
    assert result.pose_name == "address"
    assert result.club_length == pytest.approx(0.3, rel=1e-6)


@pytest.mark.unit
def test_coerce_simout_from_dict_time_series() -> None:
    raw = {
        "grip": np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]),
        "grip_quat": np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "clubhead": np.array([[0.3, 0.0, 0.0], [0.4, 0.2, 0.3]]),
    }
    result = _coerce_simout(raw, "drake", "impact")
    np.testing.assert_allclose(result.grip, [0.1, 0.2, 0.3])
    assert result.club_length == pytest.approx(0.3, rel=1e-6)
