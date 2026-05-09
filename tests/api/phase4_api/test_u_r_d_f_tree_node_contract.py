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


class TestURDFTreeNodeContract:
    """Validate URDFTreeNode response model."""

    def test_link_node(self) -> None:
        """Link tree node."""
        node = URDFTreeNode(
            id="link_torso",
            name="torso",
            node_type="link",
            children=["joint_shoulder"],
            properties={"type": "link", "mass": 5.0},
        )
        assert node.id == "link_torso"
        assert node.node_type == "link"
        assert "joint_shoulder" in node.children

    def test_joint_node(self) -> None:
        """Joint tree node with parent."""
        node = URDFTreeNode(
            id="joint_shoulder",
            name="shoulder",
            node_type="joint",
            parent_id="link_torso",
            children=["link_upper_arm"],
            properties={"joint_type": "revolute"},
        )
        assert node.parent_id == "link_torso"
        assert node.node_type == "joint"

    def test_root_node(self) -> None:
        """Root node (no parent)."""
        node = URDFTreeNode(
            id="link_base",
            name="base",
            node_type="root",
        )
        assert node.node_type == "root"
        assert node.parent_id is None


# ──────────────────────────────────────────────────────────────
#  Contract Tests: AIP JSON-RPC (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Dispatcher (#763)
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Methods (#763)
# ──────────────────────────────────────────────────────────────
