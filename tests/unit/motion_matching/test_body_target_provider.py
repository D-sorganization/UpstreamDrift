"""Unit tests for body target support across physics engine providers (MS-12 / #10331).

Verifies:
1. supports_body_target() returns True for engines with verified body-fitting routes
   (mujoco, drake, pinocchio) and False for other engines (opensim, myosuite, pendulum).
2. fit_swing with body target delegates to the pipeline, sets receipt_path on
   CanonicalFitResult, and the receipt JSON exists on disk.
3. Engines that do not support body targets fail closed with ValueError.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from src.shared.python.motion_matching.fit_result import CanonicalFitResult
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
def test_body_target_supported_where_lane_exists(
    provider_cls: type, engine_name: str, tmp_path: Path
) -> None:
    """Mujoco, Drake, and Pinocchio delegate body targets and return valid receipts on disk."""
    provider = provider_cls()
    body_payload = {
        "capture": "driver",
        "spec": "full_body_spec.json",
        "metrics": {"final_rmse_m": 0.014},
    }
    target = MultiSourceTarget(body=body_payload, metadata={"out_dir": str(tmp_path)})
    opts = FitOptions(maxiter=50)

    result = provider.fit_swing(target, opts)

    assert isinstance(result, CanonicalFitResult)
    assert result.solver_status == "success"
    assert result.receipt_path is not None

    receipt_path = Path(result.receipt_path)
    assert receipt_path.is_file()

    receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_data["engine"] == engine_name
    assert receipt_data["status"] == "success"
    assert receipt_data["converged"] is True
    assert "acceptance" in receipt_data
    assert receipt_data["acceptance"]["is_physically_accepted"] is True


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
