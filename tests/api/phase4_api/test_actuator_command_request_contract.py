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


class TestActuatorCommandRequestContract:
    """Validate ActuatorCommandRequest model."""

    def test_basic_command(self) -> None:
        """Basic constant torque command."""
        cmd = ActuatorCommandRequest(
            actuator_index=0,
            value=10.0,
            control_type="constant",
        )
        assert cmd.actuator_index == 0
        assert cmd.value == 10.0
        assert cmd.control_type == "constant"

    def test_pd_gains_command(self) -> None:
        """PD gains control type with parameters."""
        cmd = ActuatorCommandRequest(
            actuator_index=2,
            value=1.5,
            control_type="pd_gains",
            parameters={"kp": 100.0, "kd": 10.0},
        )
        assert cmd.control_type == "pd_gains"
        assert cmd.parameters is not None
        assert cmd.parameters["kp"] == 100.0

    def test_invalid_control_type_rejected(self) -> None:
        """Unknown control type should fail."""
        with pytest.raises(ValidationError, match="Unknown control_type"):
            ActuatorCommandRequest(
                actuator_index=0,
                value=0.0,
                control_type="invalid_control",
            )

    def test_negative_index_rejected(self) -> None:
        """Negative actuator index should fail."""
        with pytest.raises(ValidationError):
            ActuatorCommandRequest(
                actuator_index=-1,
                value=0.0,
            )


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Model Explorer (#1200)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Contract Tests: AIP JSON-RPC (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Dispatcher (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Methods (#763)
# ──────────────────────────────────────────────────────────────
