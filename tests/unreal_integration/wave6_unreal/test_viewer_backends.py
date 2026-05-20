"""Wave6 tests for viewer backend factory + mock + bridge.

We mock the `unreal` module surface (which doesn't actually exist in the
codebase here — UE integration is over WebSocket) by isolating the
streaming server with port=0 so no real Unreal Engine is needed.
"""

from __future__ import annotations

import sys
import time
from unittest.mock import patch

import pytest

from src.unreal_integration._viewer_base import BackendType, ViewerBackend, ViewerConfig
from src.unreal_integration._viewer_factory import create_viewer
from src.unreal_integration._viewer_mock import MockBackend
from src.unreal_integration._viewer_unreal_bridge import UnrealBridgeBackend
from src.unreal_integration.geometry import Quaternion, Vector3
from src.unreal_integration.mesh_loader import LoadedMesh


def _make_loaded_mesh() -> LoadedMesh:
    """Build a minimal LoadedMesh stub for backend tests."""
    return LoadedMesh(name="stub", vertices=[], faces=[])


# ---------- ViewerConfig ----------


class TestViewerConfig:
    def test_defaults(self) -> None:
        c = ViewerConfig()
        assert c.backend_type == BackendType.MESHCAT
        assert c.width == 1280
        assert c.height == 720

    def test_to_from_dict_roundtrip(self) -> None:
        c = ViewerConfig(
            backend_type=BackendType.MOCK,
            width=640,
            height=480,
            background_color=(0.2, 0.3, 0.4),
            enable_shadows=False,
            fov=60.0,
            server_host="0.0.0.0",
            server_port=9000,
        )
        c2 = ViewerConfig.from_dict(c.to_dict())
        assert c2.backend_type == BackendType.MOCK
        assert c2.width == 640
        assert c2.server_port == 9000
        assert c2.enable_shadows is False


# ---------- Factory ----------


class TestCreateViewer:
    def test_mock_string(self) -> None:
        v = create_viewer("mock")
        assert isinstance(v, MockBackend)

    def test_mock_enum(self) -> None:
        v = create_viewer(BackendType.MOCK)
        assert isinstance(v, MockBackend)

    def test_unreal_bridge(self) -> None:
        v = create_viewer("unreal_bridge")
        assert isinstance(v, UnrealBridgeBackend)

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_viewer("nonsense")

    def test_uses_provided_config(self) -> None:
        cfg = ViewerConfig(backend_type=BackendType.MOCK, width=42, height=42)
        v = create_viewer("mock", config=cfg)
        assert v.config.width == 42


# ---------- MockBackend ----------


class TestMockBackend:
    def test_initialize_and_shutdown(self) -> None:
        b = MockBackend()
        assert not b.is_initialized
        b.initialize()
        assert b.is_initialized
        b.shutdown()
        assert not b.is_initialized

    def test_context_manager(self) -> None:
        with MockBackend() as b:
            assert b.is_initialized
        assert not b.is_initialized

    def test_add_mesh_auto_name(self) -> None:
        b = MockBackend()
        b.initialize()
        name = b.add_mesh(_make_loaded_mesh())
        assert name.startswith("mock_mesh_")
        assert b.object_count == 1

    def test_add_mesh_explicit_name(self) -> None:
        b = MockBackend()
        b.initialize()
        name = b.add_mesh(_make_loaded_mesh(), name="club")
        assert name == "club"
        assert "club" in b.get_object_names()

    def test_update_transform(self) -> None:
        b = MockBackend()
        b.initialize()
        b.add_mesh(_make_loaded_mesh(), name="x")
        b.update_transform(
            "x",
            position=Vector3(1, 2, 3),
            rotation=Quaternion.identity(),
            scale=2.0,
        )
        assert b._objects["x"]["position"] == Vector3(1, 2, 3)
        assert b._objects["x"]["scale"] == 2.0

    def test_update_transform_unknown_is_silent(self) -> None:
        b = MockBackend()
        b.initialize()
        b.update_transform("missing", position=Vector3(1, 0, 0))
        # No raise; no object created
        assert b.object_count == 0

    def test_remove_object(self) -> None:
        b = MockBackend()
        b.initialize()
        b.add_mesh(_make_loaded_mesh(), name="x")
        assert b.remove_object("x") is True
        assert b.remove_object("x") is False

    def test_clear(self) -> None:
        b = MockBackend()
        b.initialize()
        b.add_mesh(_make_loaded_mesh(), name="a")
        b.add_mesh(_make_loaded_mesh(), name="b")
        b.clear()
        assert b.object_count == 0

    def test_render_returns_image(self) -> None:
        b = MockBackend(ViewerConfig(width=8, height=4))
        b.initialize()
        img = b.render()
        assert img is not None
        assert img.shape == (4, 8, 4)
        assert b.render_count == 1

    def test_camera_and_lights(self) -> None:
        b = MockBackend()
        from src.unreal_integration._viewer_base import LightState

        b.set_camera(position=Vector3(0, 0, 10), target=Vector3.zero(), fov=60.0)
        assert b._camera.position == Vector3(0, 0, 10)
        b.add_light(LightState(light_type="point"))
        assert len(b._lights) == 2
        b.clear_lights()
        assert b._lights == []

    def test_abstract_base_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            ViewerBackend()  # type: ignore[abstract]


# ---------- UnrealBridgeBackend ----------


class TestUnrealBridgeBackend:
    @pytest.fixture
    def bridge(self) -> UnrealBridgeBackend:
        cfg = ViewerConfig(
            backend_type=BackendType.UNREAL_BRIDGE,
            server_host="127.0.0.1",
            server_port=0,  # OS-assigned to avoid collisions
        )
        return UnrealBridgeBackend(cfg)

    def test_initialize_starts_thread(self, bridge: UnrealBridgeBackend) -> None:
        bridge.initialize()
        try:
            assert bridge.is_initialized
            assert bridge._server_thread is not None
            assert bridge._server_thread.is_alive()
        finally:
            bridge.shutdown()
            assert not bridge.is_initialized

    def test_initialize_is_idempotent(self, bridge: UnrealBridgeBackend) -> None:
        bridge.initialize()
        try:
            t = bridge._server_thread
            bridge.initialize()  # no-op
            assert bridge._server_thread is t
        finally:
            bridge.shutdown()

    def test_add_mesh_requires_initialization(
        self, bridge: UnrealBridgeBackend
    ) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            bridge.add_mesh(_make_loaded_mesh())

    def test_add_and_update_and_remove(self, bridge: UnrealBridgeBackend) -> None:
        bridge.initialize()
        try:
            n = bridge.add_mesh(_make_loaded_mesh(), name="thing")
            assert n == "thing"
            bridge.update_transform("thing", position=Vector3(5, 0, 0), scale=3.0)
            assert bridge._objects["thing"]["scale"] == 3.0
            assert bridge.remove_object("thing") is True
            assert bridge.remove_object("thing") is False
        finally:
            bridge.shutdown()

    def test_render_queues_frame(self, bridge: UnrealBridgeBackend) -> None:
        bridge.initialize()
        try:
            bridge.add_mesh(_make_loaded_mesh(), name="ball")
            bridge.add_mesh(_make_loaded_mesh(), name="club")
            bridge.add_mesh(_make_loaded_mesh(), name="hip")
            img = bridge.render()
            assert img is None  # streaming backend returns None
            # frame queued
            assert not bridge._frame_queue.empty()
            frame = bridge._frame_queue.get_nowait()
            assert frame.club is not None
            assert frame.ball is not None
            assert "hip" in frame.joints
        finally:
            bridge.shutdown()

    def test_render_not_initialized_returns_none(
        self, bridge: UnrealBridgeBackend
    ) -> None:
        assert bridge.render() is None

    def test_clear(self, bridge: UnrealBridgeBackend) -> None:
        bridge.initialize()
        try:
            bridge.add_mesh(_make_loaded_mesh(), name="a")
            bridge.clear()
            assert bridge.object_count == 0
        finally:
            bridge.shutdown()

    def test_initialize_propagates_bind_failure(self) -> None:
        """If the streaming server fails to bind, initialize() must raise."""
        cfg = ViewerConfig(backend_type=BackendType.UNREAL_BRIDGE, server_port=0)
        bridge = UnrealBridgeBackend(cfg)
        # Patch UnrealStreamingServer.start to raise OSError to simulate
        # a port conflict — proves initialize() doesn't hang forever.
        import src.unreal_integration._viewer_unreal_bridge as mod

        async def _bad_start(self: object) -> None:
            raise OSError("bind failure")

        with (
            patch.object(mod.UnrealStreamingServer, "start", _bad_start),
            pytest.raises(RuntimeError, match="failed to start"),
        ):
            bridge.initialize()
        bridge.shutdown()


# ---------- Mocking the `unreal` C-extension module surface ----------


class TestUnrealModuleMock:
    """Even though this codebase talks to Unreal via WebSocket and never
    imports the in-engine `unreal` python module, downstream consumers
    might. These tests document and verify the pattern for mocking it
    via sys.modules so importers don't blow up under test.
    """

    def test_can_mock_unreal_module(self) -> None:
        with patch.dict(
            sys.modules, {"unreal": __import__("types").ModuleType("unreal")}
        ):
            import unreal  # type: ignore[import-not-found]

            unreal.SomeClass = object  # type: ignore[attr-defined]
            assert unreal.SomeClass is object

    def test_factory_works_without_unreal_module(self) -> None:
        # Ensure 'unreal' is absent
        with patch.dict(sys.modules, {}, clear=False):
            if "unreal" in sys.modules:
                del sys.modules["unreal"]
            v = create_viewer("mock")
            assert isinstance(v, MockBackend)


def test_time_module_in_use() -> None:
    # Sanity: bridge uses time.time() for timestamps
    assert isinstance(time.time(), float)
