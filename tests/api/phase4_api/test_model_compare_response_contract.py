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


class TestModelCompareResponseContract:
    """Validate ModelCompareResponse model."""

    def test_compare_response(self) -> None:
        """Comparison of two models."""
        model_a = ModelExplorerResponse(
            model_name="model_a",
            tree=[URDFTreeNode(id="link_base", name="base", node_type="root")],
            joint_count=2,
            link_count=3,
            file_path="a.urdf",
        )
        model_b = ModelExplorerResponse(
            model_name="model_b",
            tree=[URDFTreeNode(id="link_base", name="base", node_type="root")],
            joint_count=1,
            link_count=2,
            file_path="b.urdf",
        )
        resp = ModelCompareResponse(
            model_a=model_a,
            model_b=model_b,
            shared_joints=["hip"],
            unique_to_a=["shoulder"],
            unique_to_b=[],
        )
        assert len(resp.shared_joints) == 1
        assert resp.unique_to_a == ["shoulder"]
        assert resp.unique_to_b == []


# ──────────────────────────────────────────────────────────────
#  Contract Tests: AIP JSON-RPC (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Dispatcher (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Methods (#763)
# ──────────────────────────────────────────────────────────────
