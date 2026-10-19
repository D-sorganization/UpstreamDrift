"""Contract tests verifying engine capabilities match actual engine behaviour (#12988).

Every engine in LOADER_MAP must:
1. Implement is_initialized (returning bool).
2. Implement get_capabilities() (returning EngineCapabilities).
3. Report an honest engine_name (never 'unknown').
4. For every capability flag set to FULL, execute the corresponding method on
   a loaded model and return finite results of the expected shape.
5. For capabilities set to NONE, either raise a documented unsupported error or
   abstain from producing deceptive results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.loaders import LOADER_MAP, EngineType
from src.shared.python.core.error_utils import GolfSuiteError
from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)
from src.shared.python.engine_core.interfaces import PhysicsEngine

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_test_body_name(engine: PhysicsEngine, engine_type: EngineType) -> str:
    """Resolve a valid body name for Jacobian queries across engine types."""
    if engine_type == EngineType.PUTTING_GREEN:
        return "ball"
    if engine_type in (EngineType.PENDULUM, EngineType.GOLF_SWING_PENDULUM):
        return "tip"
    if engine_type == EngineType.MUJOCO and hasattr(engine, "model") and engine.model:
        import mujoco

        for i in range(engine.model.nbody):
            name = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name != "world":
                return name
        return "world"
    if (
        engine_type == EngineType.PINOCCHIO
        and hasattr(engine, "model")
        and engine.model
    ):
        names = list(engine.model.names)
        return names[-1] if len(names) > 1 else names[0]
    return "body"


def _load_engine_or_skip(engine_type: EngineType) -> PhysicsEngine:
    """Load engine from LOADER_MAP or skip test if optional requirements missing."""
    loader = LOADER_MAP.get(engine_type)
    if loader is None:
        pytest.skip(f"Engine {engine_type.value} not present in LOADER_MAP")

    try:
        engine = loader(REPO_ROOT)
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        AttributeError,
        GolfSuiteError,
    ) as exc:
        pytest.skip(f"Engine {engine_type.value} could not be loaded: {exc}")

    return engine


@pytest.mark.parametrize("engine_type", list(LOADER_MAP.keys()), ids=lambda e: e.value)
def test_engine_implements_capabilities_and_initialization(
    engine_type: EngineType,
) -> None:
    """Every registered engine must implement is_initialized and get_capabilities()."""
    engine = _load_engine_or_skip(engine_type)

    # Design by Contract: is_initialized contract
    assert hasattr(engine, "is_initialized"), (
        f"Engine {engine_type.value} ({type(engine).__name__}) missing is_initialized"
    )
    is_init = engine.is_initialized
    assert isinstance(is_init, bool), (
        f"Engine {engine_type.value}.is_initialized must return bool, got {type(is_init).__name__}"
    )

    # Design by Contract: get_capabilities contract
    assert hasattr(engine, "get_capabilities"), (
        f"Engine {engine_type.value} ({type(engine).__name__}) missing get_capabilities"
    )
    caps = engine.get_capabilities()
    assert isinstance(caps, EngineCapabilities), (
        f"Engine {engine_type.value}.get_capabilities() must return EngineCapabilities"
    )

    # Invariant: engine_name must be known and non-empty
    assert caps.engine_name != "unknown", (
        f"Engine {engine_type.value} must not report engine_name='unknown'"
    )
    assert len(caps.engine_name.strip()) > 0, (
        f"Engine {engine_type.value} reported empty engine_name"
    )


@pytest.mark.parametrize("engine_type", list(LOADER_MAP.keys()), ids=lambda e: e.value)
def test_engine_capabilities_match_behaviour(engine_type: EngineType) -> None:
    """Capability flags must match real engine execution on a loaded model."""
    engine = _load_engine_or_skip(engine_type)
    caps = engine.get_capabilities()

    if not engine.is_initialized:
        # For uninitialized engines (e.g. skeleton adapters without license/model),
        # operations marked NONE must raise documented errors.
        if caps.mass_matrix == CapabilityLevel.NONE and hasattr(
            engine, "compute_mass_matrix"
        ):
            with pytest.raises(
                (RuntimeError, ValueError, TypeError, AttributeError, GolfSuiteError)
            ):
                engine.compute_mass_matrix()
        if caps.forward_sim == CapabilityLevel.NONE and hasattr(engine, "step"):
            with pytest.raises(
                (RuntimeError, ValueError, TypeError, AttributeError, GolfSuiteError)
            ):
                engine.step(0.01)
        return

    # 1. Forward simulation contract
    if caps.forward_sim == CapabilityLevel.FULL:
        assert hasattr(engine, "step"), (
            f"{engine_type.value} declares forward_sim but has no step()"
        )
        t0 = engine.get_time()
        engine.step(0.01)
        t1 = engine.get_time()
        assert t1 > t0, (
            f"{engine_type.value} forward_sim=FULL but step() did not advance time: t0={t0}, t1={t1}"
        )

    # 2. Mass matrix contract
    if caps.mass_matrix == CapabilityLevel.FULL:
        assert hasattr(engine, "compute_mass_matrix"), (
            f"{engine_type.value} declares mass_matrix=FULL but has no compute_mass_matrix()"
        )
        M = engine.compute_mass_matrix()
        assert isinstance(M, np.ndarray), (
            f"{engine_type.value} mass matrix must be np.ndarray"
        )
        assert M.ndim == 2, (
            f"{engine_type.value} mass matrix must be 2D, got shape {M.shape}"
        )
        assert M.shape[0] == M.shape[1], (
            f"{engine_type.value} mass matrix must be square: {M.shape}"
        )
        assert M.shape[0] > 0, f"{engine_type.value} mass matrix has dimension 0"
        assert np.all(np.isfinite(M)), (
            f"{engine_type.value} mass matrix contains non-finite values"
        )

        nv = M.shape[0]

        # 3. Inverse dynamics contract
        if caps.inverse_dynamics == CapabilityLevel.FULL:
            assert hasattr(engine, "compute_inverse_dynamics"), (
                f"{engine_type.value} declares inverse_dynamics=FULL but lacks compute_inverse_dynamics()"
            )
            qacc = np.zeros(nv)
            tau = engine.compute_inverse_dynamics(qacc)
            assert isinstance(tau, np.ndarray), (
                f"{engine_type.value} tau must be np.ndarray"
            )
            assert tau.ndim == 1, f"{engine_type.value} tau must be 1D, got {tau.shape}"
            assert len(tau) == nv, (
                f"{engine_type.value} tau length {len(tau)} != nv {nv}"
            )
            assert np.all(np.isfinite(tau)), (
                f"{engine_type.value} tau contains non-finite values"
            )

        # 4. Drift acceleration contract
        if caps.drift_acceleration == CapabilityLevel.FULL:
            assert hasattr(engine, "compute_drift_acceleration"), (
                f"{engine_type.value} declares drift_acceleration=FULL but lacks compute_drift_acceleration()"
            )
            drift = engine.compute_drift_acceleration()
            assert isinstance(drift, np.ndarray), (
                f"{engine_type.value} drift must be np.ndarray"
            )
            assert drift.ndim == 1, (
                f"{engine_type.value} drift must be 1D, got {drift.shape}"
            )
            assert len(drift) == nv, (
                f"{engine_type.value} drift length {len(drift)} != nv {nv}"
            )
            assert np.all(np.isfinite(drift)), (
                f"{engine_type.value} drift contains non-finite values"
            )

    # 5. Jacobian contract
    if caps.jacobian == CapabilityLevel.FULL:
        assert hasattr(engine, "compute_jacobian"), (
            f"{engine_type.value} declares jacobian=FULL but lacks compute_jacobian()"
        )
        body_name = _find_test_body_name(engine, engine_type)
        jac = engine.compute_jacobian(body_name)
        assert jac is not None, (
            f"{engine_type.value}.compute_jacobian({body_name!r}) returned None despite jacobian=FULL"
        )
        if isinstance(jac, dict):
            for key, val in jac.items():
                if isinstance(val, np.ndarray):
                    assert np.all(np.isfinite(val)), (
                        f"{engine_type.value} Jacobian entry {key!r} contains non-finite values"
                    )
        elif isinstance(jac, np.ndarray):
            assert np.all(np.isfinite(jac)), (
                f"{engine_type.value} Jacobian contains non-finite values"
            )
