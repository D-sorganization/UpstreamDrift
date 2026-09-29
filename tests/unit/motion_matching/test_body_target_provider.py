"""Unit tests for body target support across physics engine providers (#10960 P0-1).

Verifies:
1. supports_body_target() returns True for engines declaring body-fitting routes
   (mujoco, drake, pinocchio) and False for other engines (opensim, myosuite, pendulum).
2. fit_swing with body target fails closed with NotImplementedError for unwired body-fit lanes.
3. execute_body_fit directly fails closed with NotImplementedError, writes no receipt file,
   and ignores caller-injected acceptance/metrics payloads.
4. Engines that do not support body targets fail closed with ValueError.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    execute_body_fit,
    has_body_target,
    resolve_body_target,
)
from src.engines.physics_engines.mujoco.python.motion_matching.provider import (
    MujocoFitSwingProvider,
)
from src.engines.physics_engines.drake.python.motion_matching.provider import (
    DrakeFitSwingProvider,
)
from src.engines.physics_engines.pinocchio.python.motion_matching.provider import (
    PinocchioFitSwingProvider,
)
from src.engines.physics_engines.opensim.python.motion_matching.provider import (
    OpenSimFitSwingProvider,
)
from src.engines.physics_engines.myosuite.python.motion_matching.provider import (
    MyoSuiteFitSwingProvider,
)
from src.engines.physics_engines.pendulum.python.motion_matching.provider import (
    PendulumFitSwingProvider,
)


@pytest.mark.unit
def test_supports_body_target_by_engine() -> None:
    """Supported engines return True; unsupported engines return False."""
    assert MujocoFitSwingProvider().supports_body_target() is True
    assert DrakeFitSwingProvider().supports_body_target() is True
    assert PinocchioFitSwingProvider().supports_body_target() is True

    assert OpenSimFitSwingProvider().supports_body_target() is False
    assert MyoSuiteFitSwingProvider().supports_body_target() is False
    assert PendulumFitSwingProvider().supports_body_target() is False


@pytest.mark.unit
def test_has_and_resolve_body_target_helpers() -> None:
    """has_body_target and resolve_body_target correctly identify and extract payloads."""
    body_payload = {"capture": "driver", "markers": ["RASI", "LASI"]}
    target_with_body = MultiSourceTarget(body=body_payload)

    assert has_body_target(target_with_body) is True
    assert resolve_body_target(target_with_body) is body_payload

    # Object without body
    assert has_body_target("not_a_target") is False
    with pytest.raises(TypeError, match="MultiSourceTarget"):
        resolve_body_target("not_a_target")


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider_cls,engine_name",
    [
        (MujocoFitSwingProvider, "mujoco"),
        (DrakeFitSwingProvider, "drake"),
        (PinocchioFitSwingProvider, "pinocchio"),
    ],
)
def test_body_target_fails_closed_in_providers(
    provider_cls: type, engine_name: str, tmp_path: Path
) -> None:
    """Mujoco, Drake, and Pinocchio propagate fail-closed error on unwired body lanes (#10960 P0-1)."""
    provider = provider_cls()
    body_payload = {
        "capture": "driver",
        "spec": "full_body_spec.json",
        "metrics": {"final_rmse_m": 0.014},
    }
    target = MultiSourceTarget(body=body_payload, metadata={"out_dir": str(tmp_path)})
    opts = FitOptions(maxiter=50)

    with pytest.raises(NotImplementedError, match="body-target fit lane is not wired"):
        provider.fit_swing(target, opts)

    # Must write no receipt files
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_execute_body_fit_direct_fails_closed(tmp_path: Path) -> None:
    """execute_body_fit raises NotImplementedError and writes no receipt file."""
    target = MultiSourceTarget(
        body={"capture": "driver"}, metadata={"out_dir": str(tmp_path)}
    )
    opts = FitOptions()

    with pytest.raises(NotImplementedError, match="body-target fit lane is not wired"):
        execute_body_fit("mujoco", target, opts)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_execute_body_fit_ignores_caller_acceptance_and_metrics(tmp_path: Path) -> None:
    """Caller-supplied acceptance verdicts and metrics must never yield a PASSED result (#10960 P0-1)."""
    fabricated_body = {
        "capture": "driver",
        "schema": "matched-swing-fit/injected-v1",
        "metrics": {
            "whole_marker_rmse_m": 0.001,
            "final_rmse_m": 0.001,
            "final_cost": 0.0001,
        },
        "acceptance": {
            "status": "PASSED",
            "is_physically_accepted": True,
        },
    }
    target = MultiSourceTarget(
        body=fabricated_body, metadata={"out_dir": str(tmp_path)}
    )
    opts = FitOptions()

    # Direct invocation fails closed without returning accepted verdict
    with pytest.raises(NotImplementedError, match="body-target fit lane is not wired"):
        execute_body_fit("mujoco", target, opts)

    # Provider invocation fails closed without returning accepted verdict
    provider = MujocoFitSwingProvider()
    with pytest.raises(NotImplementedError, match="body-target fit lane is not wired"):
        provider.fit_swing(target, opts)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_execute_body_fit_preconditions() -> None:
    """execute_body_fit enforces input validation DbC preconditions."""
    valid_target = MultiSourceTarget(body={"capture": "driver"})
    with pytest.raises(ValueError, match="engine_name"):
        execute_body_fit("", valid_target)

    with pytest.raises(TypeError, match="MultiSourceTarget"):
        execute_body_fit("mujoco", "invalid_target")


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider_cls",
    [
        OpenSimFitSwingProvider,
        MyoSuiteFitSwingProvider,
        PendulumFitSwingProvider,
    ],
)
def test_body_target_unsupported_engines_fail_closed(provider_cls: type) -> None:
    """Engines without full-body matching lanes fail closed on body targets."""
    provider = provider_cls()
    target = MultiSourceTarget(body={"capture": "driver"})
    opts = FitOptions()

    with pytest.raises(ValueError):
        provider.fit_swing(target, opts)
