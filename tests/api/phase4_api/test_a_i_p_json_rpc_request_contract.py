"""Tests for Phase 4 API: Force overlays, actuator controls, model explorer, AIP.

Validates Pydantic contract models and route logic for:
- Force/torque vector overlays (#1199)
- Per-actuator control sliders (#1198)
- Model explorer & URDF editor (#1200)
- AIP JSON-RPC server (#763)

See issue #1199, #1198, #1200, #763
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.api.models.requests import (
    ActuatorBatchCommandRequest,
    ActuatorCommandRequest,
    AIPJsonRpcRequest,
    ForceOverlayRequest,
    ModelCompareRequest,
    ModelExplorerRequest,
)
from src.api.models.responses import (
    ActuatorCommandResponse,
    ActuatorInfo,
    ActuatorPanelResponse,
    AIPCapability,
    AIPHandshakeResponse,
    AIPJsonRpcResponse,
    ForceOverlayResponse,
    ForceVector3D,
    ModelCompareResponse,
    ModelExplorerResponse,
    URDFTreeNode,
)

# ──────────────────────────────────────────────────────────────
#  Contract Tests: Force Overlay (#1199)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Actuator Controls (#1198)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Model Explorer (#1200)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: AIP JSON-RPC (#763)
# ──────────────────────────────────────────────────────────────


class TestAIPJsonRpcRequestContract:
    """Validate AIPJsonRpcRequest model."""

    def test_basic_request(self) -> None:
        """Basic JSON-RPC request."""
        req = AIPJsonRpcRequest(
            method="simulation.start",
            params={"engine_type": "mujoco"},
            id=1,
        )
        assert req.jsonrpc == "2.0"
        assert req.method == "simulation.start"
        assert req.id == 1

    def test_notification_no_id(self) -> None:
        """Notification (no id)."""
        req = AIPJsonRpcRequest(
            method="simulation.stop",
        )
        assert req.id is None

    def test_invalid_version_rejected(self) -> None:
        """Non-2.0 version should fail."""
        with pytest.raises(ValidationError, match="Only JSON-RPC 2.0"):
            AIPJsonRpcRequest(
                jsonrpc="1.0",
                method="test",
            )

    def test_positional_params(self) -> None:
        """Positional params as list."""
        req = AIPJsonRpcRequest(
            method="simulation.step",
            params=[5],
            id=2,
        )
        assert isinstance(req.params, list)
        assert req.params[0] == 5

    def test_string_id(self) -> None:
        """String request ID."""
        req = AIPJsonRpcRequest(
            method="system.ping",
            id="req-abc-123",
        )
        assert req.id == "req-abc-123"


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Dispatcher (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Methods (#763)
# ──────────────────────────────────────────────────────────────
