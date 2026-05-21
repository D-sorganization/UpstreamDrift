"""Qt-free tests for :class:`EngineController` internals.

Complements :mod:`tests.unit.tools.pose_studio.test_core` by exercising
the seldom-covered branches:

* ``_maybe_publish_pose`` env-var gate, 30 Hz debounce, exception swallow.
* ``set_pose`` ``NotImplementedError`` -> mock-downgrade path.
* ``set_pose`` ``RuntimeError`` -> ``EngineStatus.ERROR`` path.
* ``_activate`` failure when the factory raises.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.shared.python.pose_interchange.canonical import (
    CanonicalPose,
    canonical_zero_pose,
)
from src.shared.python.pose_interchange.services import MockKinematicsService
from src.tools.pose_studio.controllers import EngineController
from src.tools.pose_studio.controllers import engine_controller as ec_mod
from src.tools.pose_studio.core import SUPPORTED_ENGINES, EngineStatus

pytestmark = pytest.mark.unit


_ENGINE = SUPPORTED_ENGINES[0]


# ---------------------------------------------------------------------------
# _maybe_publish_pose
# ---------------------------------------------------------------------------


def _make_ctrl_with_mock_service() -> EngineController:
    """Build a controller with a guaranteed-clean MockKinematicsService.

    Isolates the publish-gate tests from real-engine state (Drake's
    service raises if the URDF isn't loaded, polluting these tests).
    """
    ctrl = EngineController(_ENGINE)
    ctrl._service = MockKinematicsService(_ENGINE)
    ctrl._status = EngineStatus.MOCK
    ctrl._last_publish_ts = 0.0
    return ctrl


def test_publish_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No realtime publish must happen unless the env var is set."""
    monkeypatch.delenv("POSE_STUDIO_PUBLISH_REALTIME", raising=False)
    ctrl = _make_ctrl_with_mock_service()
    with patch.object(ec_mod, "realtime_publish") as mock_pub:
        ctrl.set_pose(canonical_zero_pose())
    mock_pub.assert_not_called()


def test_publish_emits_when_env_var_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSE_STUDIO_PUBLISH_REALTIME", "1")
    ctrl = _make_ctrl_with_mock_service()
    with patch.object(ec_mod, "realtime_publish") as mock_pub:
        ctrl.set_pose(canonical_zero_pose())
    assert mock_pub.call_count == 1
    channel, payload = mock_pub.call_args.args
    assert channel == "pose/canonical"
    assert isinstance(payload, dict)


def test_publish_debounced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two rapid set_pose calls must only publish once."""
    monkeypatch.setenv("POSE_STUDIO_PUBLISH_REALTIME", "1")
    ctrl = _make_ctrl_with_mock_service()
    pose = canonical_zero_pose()
    with (
        patch.object(ec_mod, "realtime_publish") as mock_pub,
        patch.object(ec_mod.time, "monotonic", return_value=1000.0),
    ):
        # Pin time.monotonic so both calls land within the debounce window.
        ctrl.set_pose(pose)
        ctrl.set_pose(pose)
    assert mock_pub.call_count == 1


def test_publish_swallows_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misbehaving IPC layer must never break pose editing."""
    monkeypatch.setenv("POSE_STUDIO_PUBLISH_REALTIME", "1")
    ctrl = _make_ctrl_with_mock_service()
    with patch.object(ec_mod, "realtime_publish", side_effect=OSError("ipc down")):
        # Must not raise.
        ctrl.set_pose(canonical_zero_pose())
    # Controller state must still be healthy — publish failure is silent.
    assert ctrl.status is EngineStatus.MOCK


# ---------------------------------------------------------------------------
# set_pose error paths
# ---------------------------------------------------------------------------


def test_set_pose_downgrades_on_not_implemented() -> None:
    """A partial engine raising NotImplementedError must downgrade to mock."""
    ctrl = EngineController(_ENGINE)

    class _PartialService:
        def __init__(self) -> None:
            self.calls = 0

        def set_pose(self, pose: CanonicalPose) -> None:
            self.calls += 1
            raise NotImplementedError("partial bridge")

    ctrl._service = _PartialService()  # type: ignore[assignment]
    ctrl._status = EngineStatus.LIVE
    ctrl.set_pose(canonical_zero_pose())
    assert isinstance(ctrl._service, MockKinematicsService)
    assert ctrl.status is EngineStatus.MOCK
    assert ctrl.last_error is None


def test_set_pose_marks_error_on_runtime_error() -> None:
    ctrl = EngineController(_ENGINE)

    class _BadService:
        def set_pose(self, pose: CanonicalPose) -> None:
            raise RuntimeError("kinematics solver blew up")

    ctrl._service = _BadService()  # type: ignore[assignment]
    ctrl.set_pose(canonical_zero_pose())
    assert ctrl.status is EngineStatus.ERROR
    assert ctrl.last_error is not None
    assert "kinematics solver blew up" in ctrl.last_error


def test_set_pose_marks_error_on_value_error() -> None:
    ctrl = EngineController(_ENGINE)

    class _BadService:
        def set_pose(self, pose: CanonicalPose) -> None:
            raise ValueError("bad pose")

    ctrl._service = _BadService()  # type: ignore[assignment]
    ctrl.set_pose(canonical_zero_pose())
    assert ctrl.status is EngineStatus.ERROR
    assert ctrl.last_error is not None
    assert "bad pose" in ctrl.last_error


def test_set_pose_noop_when_service_is_none() -> None:
    ctrl = EngineController(_ENGINE)
    ctrl._service = None
    pose = canonical_zero_pose()
    # Must not raise even with no active service.
    ctrl.set_pose(pose)
    assert ctrl.pose is pose


# ---------------------------------------------------------------------------
# _activate failure path
# ---------------------------------------------------------------------------


def test_activate_failure_yields_error_status() -> None:
    """If a service factory raises, controller flips to ERROR."""

    def _boom() -> None:
        raise RuntimeError("engine wheel exploded")

    fake_registry = {_ENGINE: _boom}
    with patch.object(ec_mod, "KINEMATICS_SERVICE_REGISTRY", fake_registry):
        ctrl = EngineController(_ENGINE)
    assert ctrl.status is EngineStatus.ERROR
    assert ctrl.adapter is None
    assert ctrl.service is None
    assert ctrl.last_error is not None
    assert "engine wheel exploded" in ctrl.last_error


def test_activate_failure_on_import_error() -> None:
    def _boom() -> None:
        raise ImportError("no wheel")

    with patch.dict(ec_mod.KINEMATICS_SERVICE_REGISTRY, {_ENGINE: _boom}):
        ctrl = EngineController(_ENGINE)
    assert ctrl.status is EngineStatus.ERROR
