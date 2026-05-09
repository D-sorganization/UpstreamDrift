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


class TestAIPCapabilityContract:
    """Validate AIPCapability response model."""

    def test_capability(self) -> None:
        """Single capability descriptor."""
        cap = AIPCapability(
            name="simulation",
            version="1.0",
            methods=["simulation.start", "simulation.stop", "simulation.step"],
        )
        assert cap.name == "simulation"
        assert len(cap.methods) == 3


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Dispatcher (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Methods (#763)
# ──────────────────────────────────────────────────────────────
