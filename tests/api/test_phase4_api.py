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


class TestForceOverlayRequestContract:
    """Validate ForceOverlayRequest model."""

    def test_default_values(self) -> None:
        """Defaults should be sensible."""
        req = ForceOverlayRequest()
        assert req.enabled is True
        assert req.force_types == ["applied"]
        assert req.scale_factor == 0.01
        assert req.color_by_magnitude is True
        assert req.body_filter is None
        assert req.show_labels is False

    def test_valid_force_types(self) -> None:
        """All valid force types accepted."""
        req = ForceOverlayRequest(
            force_types=["applied", "gravity", "contact", "bias", "all"]
        )
        assert len(req.force_types) == 5

    def test_invalid_force_type_rejected(self) -> None:
        """Unknown force type should fail validation."""
        with pytest.raises(ValidationError, match="Unknown force type"):
            ForceOverlayRequest(force_types=["invalid_type"])

    def test_scale_factor_range(self) -> None:
        """Scale factor must be positive and <= 1.0."""
        with pytest.raises(ValidationError):
            ForceOverlayRequest(scale_factor=0.0)
        with pytest.raises(ValidationError):
            ForceOverlayRequest(scale_factor=2.0)

    def test_body_filter(self) -> None:
        """Body filter should accept list of names."""
        req = ForceOverlayRequest(body_filter=["torso", "hand"])
        assert req.body_filter == ["torso", "hand"]


class TestForceVector3DContract:
    """Validate ForceVector3D response model."""

    def test_basic_vector(self) -> None:
        """Basic force vector creation."""
        vec = ForceVector3D(
            body_name="torso",
            force_type="applied",
            origin=[0.0, 1.0, 0.0],
            direction=[1.0, 0.0, 0.0],
            magnitude=50.0,
        )
        assert vec.body_name == "torso"
        assert vec.force_type == "applied"
        assert vec.magnitude == 50.0
        assert vec.color == [1.0, 0.0, 0.0, 1.0]  # Default red

    def test_custom_color_and_label(self) -> None:
        """Custom color and label."""
        vec = ForceVector3D(
            body_name="arm",
            force_type="gravity",
            origin=[0.0, 0.5, 0.0],
            direction=[0.0, -1.0, 0.0],
            magnitude=9.81,
            color=[0.0, 0.0, 1.0, 0.8],
            label="9.81 N",
        )
        assert vec.color == [0.0, 0.0, 1.0, 0.8]
        assert vec.label == "9.81 N"


class TestForceOverlayResponseContract:
    """Validate ForceOverlayResponse model."""

    def test_empty_response(self) -> None:
        """Response with no vectors."""
        resp = ForceOverlayResponse(
            sim_time=0.0,
            vectors=[],
            total_force_magnitude=0.0,
            total_torque_magnitude=0.0,
        )
        assert resp.sim_time == 0.0
        assert len(resp.vectors) == 0

    def test_response_with_vectors(self) -> None:
        """Response with multiple vectors."""
        vectors = [
            ForceVector3D(
                body_name="torso",
                force_type="applied",
                origin=[0.0, 1.0, 0.0],
                direction=[1.0, 0.0, 0.0],
                magnitude=50.0,
            ),
            ForceVector3D(
                body_name="arm",
                force_type="gravity",
                origin=[0.0, 0.5, 0.0],
                direction=[0.0, -1.0, 0.0],
                magnitude=9.81,
            ),
        ]
        resp = ForceOverlayResponse(
            sim_time=1.5,
            vectors=vectors,
            total_force_magnitude=59.81,
            total_torque_magnitude=50.0,
        )
        assert len(resp.vectors) == 2
        assert resp.total_force_magnitude == pytest.approx(59.81)


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


class TestActuatorBatchCommandContract:
    """Validate ActuatorBatchCommandRequest model."""

    def test_batch_commands(self) -> None:
        """Batch of valid commands."""
        batch = ActuatorBatchCommandRequest(
            commands=[
                ActuatorCommandRequest(actuator_index=0, value=5.0),
                ActuatorCommandRequest(actuator_index=1, value=-3.0),
            ]
        )
        assert len(batch.commands) == 2

    def test_empty_batch_rejected(self) -> None:
        """Empty batch should fail."""
        with pytest.raises(ValidationError):
            ActuatorBatchCommandRequest(commands=[])


class TestActuatorInfoContract:
    """Validate ActuatorInfo response model."""

    def test_basic_actuator(self) -> None:
        """Basic actuator descriptor."""
        info = ActuatorInfo(
            index=0,
            name="hip_rotation",
            control_type="constant",
            value=0.0,
            min_value=-3.14,
            max_value=3.14,
            units="N*m",
            joint_type="revolute",
        )
        assert info.name == "hip_rotation"
        assert info.min_value == -3.14
        assert info.max_value == 3.14


class TestActuatorPanelResponseContract:
    """Validate ActuatorPanelResponse model."""

    def test_panel_response(self) -> None:
        """Panel with multiple actuators."""
        actuators = [
            ActuatorInfo(
                index=i,
                name=f"joint_{i}",
                value=0.0,
                min_value=-100.0,
                max_value=100.0,
            )
            for i in range(3)
        ]
        resp = ActuatorPanelResponse(
            n_actuators=3,
            actuators=actuators,
            engine_name="mujoco",
        )
        assert resp.n_actuators == 3
        assert len(resp.actuators) == 3
        assert resp.engine_name == "mujoco"


class TestActuatorCommandResponseContract:
    """Validate ActuatorCommandResponse model."""

    def test_ok_response(self) -> None:
        """Successful command."""
        resp = ActuatorCommandResponse(
            actuator_index=0,
            applied_value=10.0,
            control_type="constant",
            status="ok",
            clamped=False,
        )
        assert resp.status == "ok"
        assert resp.clamped is False

    def test_clamped_response(self) -> None:
        """Command that was clamped to limits."""
        resp = ActuatorCommandResponse(
            actuator_index=1,
            applied_value=100.0,
            control_type="constant",
            status="ok",
            clamped=True,
        )
        assert resp.clamped is True


# ──────────────────────────────────────────────────────────────
#  Contract Tests: Model Explorer (#1200)
# ──────────────────────────────────────────────────────────────


class TestModelExplorerRequestContract:
    """Validate ModelExplorerRequest model."""

    def test_basic_request(self) -> None:
        """Basic model explorer request."""
        req = ModelExplorerRequest(model_path="src/shared/urdf/golfer.urdf")
        assert req.model_path == "src/shared/urdf/golfer.urdf"
        assert req.joint_values is None

    def test_with_joint_values(self) -> None:
        """Request with FK preview joints."""
        req = ModelExplorerRequest(
            model_path="test.urdf",
            joint_values={"shoulder": 0.5, "elbow": 1.0},
        )
        assert req.joint_values is not None
        assert req.joint_values["shoulder"] == 0.5


class TestModelCompareRequestContract:
    """Validate ModelCompareRequest model."""

    def test_compare_request(self) -> None:
        """Two-model comparison request."""
        req = ModelCompareRequest(
            model_a_path="model_a.urdf",
            model_b_path="model_b.urdf",
        )
        assert req.model_a_path == "model_a.urdf"
        assert req.model_b_path == "model_b.urdf"


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


class TestModelExplorerResponseContract:
    """Validate ModelExplorerResponse model."""

    def test_basic_response(self) -> None:
        """Model explorer response."""
        resp = ModelExplorerResponse(
            model_name="test_robot",
            tree=[
                URDFTreeNode(id="link_base", name="base", node_type="root"),
                URDFTreeNode(
                    id="joint_hip",
                    name="hip",
                    node_type="joint",
                    parent_id="link_base",
                ),
            ],
            joint_count=1,
            link_count=2,
            file_path="test.urdf",
        )
        assert resp.model_name == "test_robot"
        assert resp.joint_count == 1
        assert resp.link_count == 2
        assert len(resp.tree) == 2


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


class TestAIPHandshakeResponseContract:
    """Validate AIPHandshakeResponse model."""

    def test_handshake(self) -> None:
        """Full handshake response."""
        resp = AIPHandshakeResponse(
            server_name="UpstreamDrift AIP Server",
            protocol_version="2.0",
            capabilities=[
                AIPCapability(
                    name="simulation",
                    version="1.0",
                    methods=["simulation.start", "simulation.stop"],
                ),
                AIPCapability(
                    name="model",
                    version="1.0",
                    methods=["model.load", "model.query"],
                ),
            ],
            supported_methods=[
                "simulation.start",
                "simulation.stop",
                "model.load",
                "model.query",
            ],
        )
        assert resp.server_name == "UpstreamDrift AIP Server"
        assert len(resp.capabilities) == 2
        assert len(resp.supported_methods) == 4


class TestAIPJsonRpcResponseContract:
    """Validate AIPJsonRpcResponse model."""

    def test_success_response(self) -> None:
        """Successful RPC response."""
        resp = AIPJsonRpcResponse(
            result={"status": "ok", "data": [1, 2, 3]},
            id=1,
        )
        assert resp.jsonrpc == "2.0"
        assert resp.result is not None
        assert resp.error is None

    def test_error_response(self) -> None:
        """Error RPC response."""
        resp = AIPJsonRpcResponse(
            error={"code": -32601, "message": "Method not found"},
            id=2,
        )
        assert resp.result is None
        assert resp.error is not None
        assert resp.error["code"] == -32601


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Dispatcher (#763)
# ──────────────────────────────────────────────────────────────


class TestAIPDispatcher:
    """Test the JSON-RPC dispatcher logic."""

    def test_method_registry(self) -> None:
        """Registry stores and retrieves methods."""
        from src.api.aip.dispatcher import MethodRegistry

        registry = MethodRegistry()
        registry.register("test.hello", lambda: {"msg": "hello"}, "Say hello")

        assert "test.hello" in registry.list_methods()
        assert registry.get_method("test.hello") is not None
        assert registry.get_method("nonexistent") is None
        assert registry.get_description("test.hello") == "Say hello"

    def test_list_by_namespace(self) -> None:
        """Methods grouped by namespace."""
        from src.api.aip.dispatcher import MethodRegistry

        registry = MethodRegistry()
        registry.register("sim.start", lambda: None)
        registry.register("sim.stop", lambda: None)
        registry.register("model.load", lambda: None)

        namespaces = registry.list_by_namespace()
        assert "sim" in namespaces
        assert len(namespaces["sim"]) == 2
        assert "model" in namespaces
        assert len(namespaces["model"]) == 1

    @pytest.mark.asyncio
    async def test_dispatch_success(self) -> None:
        """Dispatch resolves method and returns result."""
        from src.api.aip.dispatcher import MethodRegistry, dispatch

        registry = MethodRegistry()
        registry.register("test.add", lambda a, b, **kw: a + b)

        result = await dispatch(
            registry,
            {
                "jsonrpc": "2.0",
                "method": "test.add",
                "params": [3, 4],
                "id": 1,
            },
        )

        assert result is not None
        assert result["result"] == 7
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_dispatch_method_not_found(self) -> None:
        """Unknown method returns -32601."""
        from src.api.aip.dispatcher import METHOD_NOT_FOUND, MethodRegistry, dispatch

        registry = MethodRegistry()

        result = await dispatch(
            registry,
            {
                "jsonrpc": "2.0",
                "method": "nonexistent",
                "id": 1,
            },
        )

        assert result is not None
        assert result["error"]["code"] == METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_dispatch_invalid_version(self) -> None:
        """Wrong JSON-RPC version returns error."""
        from src.api.aip.dispatcher import INVALID_REQUEST, MethodRegistry, dispatch

        registry = MethodRegistry()

        result = await dispatch(
            registry,
            {
                "jsonrpc": "1.0",
                "method": "test",
                "id": 1,
            },
        )

        assert result is not None
        assert result["error"]["code"] == INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_dispatch_notification(self) -> None:
        """Notification (no id) returns None."""
        from src.api.aip.dispatcher import MethodRegistry, dispatch

        registry = MethodRegistry()
        registry.register("test.noop", lambda **kw: None)

        result = await dispatch(
            registry,
            {
                "jsonrpc": "2.0",
                "method": "test.noop",
            },
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_with_kwargs(self) -> None:
        """Dispatch with named parameters."""
        from src.api.aip.dispatcher import MethodRegistry, dispatch

        registry = MethodRegistry()
        registry.register(
            "test.greet",
            lambda name="world", **kw: f"hello {name}",
        )

        result = await dispatch(
            registry,
            {
                "jsonrpc": "2.0",
                "method": "test.greet",
                "params": {"name": "alice"},
                "id": 42,
            },
        )

        assert result is not None
        assert result["result"] == "hello alice"
        assert result["id"] == 42


# ──────────────────────────────────────────────────────────────
#  Unit Tests: AIP Methods (#763)
# ──────────────────────────────────────────────────────────────


class TestAIPMethods:
    """Test the AIP method implementations."""

    def test_create_registry(self) -> None:
        """Registry should contain all expected methods."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        methods = registry.list_methods()

        # Verify expected methods exist
        assert "simulation.start" in methods
        assert "simulation.stop" in methods
        assert "simulation.step" in methods
        assert "simulation.status" in methods
        assert "simulation.set_control" in methods
        assert "model.load" in methods
        assert "model.query" in methods
        assert "model.list" in methods
        assert "analysis.metrics" in methods
        assert "analysis.export" in methods
        assert "analysis.time_series" in methods
        assert "system.capabilities" in methods
        assert "system.ping" in methods

    def test_system_ping(self) -> None:
        """Ping returns pong."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("system.ping")
        assert handler is not None
        result = handler()
        assert result["status"] == "pong"

    def test_system_capabilities(self) -> None:
        """Capabilities returns structured data."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("system.capabilities")
        assert handler is not None
        result = handler()
        assert "server_name" in result
        assert "capabilities" in result
        assert "supported_methods" in result
        assert len(result["supported_methods"]) > 0

    def test_simulation_start(self) -> None:
        """Start returns status."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("simulation.start")
        assert handler is not None
        result = handler(engine_type="pendulum", duration=1.0)
        assert result["status"] == "started"
        assert result["engine_type"] == "pendulum"

    def test_simulation_stop(self) -> None:
        """Stop returns stopped status."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("simulation.stop")
        assert handler is not None
        result = handler()
        assert result["status"] == "stopped"

    def test_simulation_status_no_engine(self) -> None:
        """Status with no engine returns not running."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("simulation.status")
        assert handler is not None
        result = handler()
        assert result["running"] is False

    def test_model_load_requires_path(self) -> None:
        """Model load with empty path returns error."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("model.load")
        assert handler is not None
        result = handler(path="")
        assert result["status"] == "error"

    def test_model_load_valid_path(self) -> None:
        """Model load with valid path returns loaded."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("model.load")
        assert handler is not None
        result = handler(path="test.urdf")
        assert result["status"] == "loaded"
        assert result["format"] == "urdf"

    def test_analysis_export(self) -> None:
        """Analysis export returns status."""
        from src.api.aip.methods import create_registry

        registry = create_registry()
        handler = registry.get_method("analysis.export")
        assert handler is not None
        result = handler(format="csv")
        assert result["status"] == "ok"
        assert result["format"] == "csv"
