"""Unit tests for the MeshCat viewer adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.dtack.viz import meshcat_viewer

pytestmark = pytest.mark.unit


class FakePinocchioVisualizer:
    instances: list[FakePinocchioVisualizer] = []

    def __init__(self, model, collision_model, visual_model):
        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model
        self.initViewer = Mock()
        self.loadViewerModel = Mock()
        self.display = Mock()
        self.clean = Mock()
        self.instances.append(self)


@pytest.fixture
def fake_meshcat(monkeypatch):
    FakePinocchioVisualizer.instances.clear()
    visualizer = MagicMock()
    visualizer_cls = Mock(return_value=visualizer)
    monkeypatch.setattr(meshcat_viewer, "MESHCAT_AVAILABLE", True)
    monkeypatch.setattr(
        meshcat_viewer, "viz", SimpleNamespace(Visualizer=visualizer_cls)
    )
    monkeypatch.setattr(meshcat_viewer, "MeshcatVisualizer", FakePinocchioVisualizer)
    monkeypatch.setattr(
        meshcat_viewer, "pin", SimpleNamespace(neutral=lambda model: np.zeros(model.nq))
    )
    return visualizer, visualizer_cls


def test_load_stores_native_visualizer_and_routes_separate_geometry(fake_meshcat):
    visualizer, visualizer_cls = fake_meshcat
    model = SimpleNamespace(nq=2)
    collision_model = object()
    visual_model = object()

    viewer = meshcat_viewer.MeshCatViewer(zmq_url=None, open_browser=False)
    viewer.load_model(model, visual_model, collision_model=collision_model)

    visualizer_cls.assert_called_once_with(zmq_url=None)
    visualizer.open.assert_not_called()
    native = FakePinocchioVisualizer.instances[-1]
    assert native.collision_model is collision_model
    assert native.visual_model is visual_model
    native.initViewer.assert_called_once_with(viewer=visualizer)
    native.loadViewerModel.assert_called_once_with(rootNodeName=viewer._root_node_name)


def test_display_dispatches_and_none_uses_neutral(fake_meshcat):
    visualizer, _ = fake_meshcat
    model = SimpleNamespace(nq=2)
    viewer = meshcat_viewer.MeshCatViewer(open_browser=False)
    viewer.load_model(model, object())
    native = FakePinocchioVisualizer.instances[-1]

    viewer.display([1.0, 2.0])
    viewer.display()

    np.testing.assert_array_equal(native.display.call_args_list[0].args[0], [1.0, 2.0])
    np.testing.assert_array_equal(native.display.call_args_list[1].args[0], [0.0, 0.0])


def test_two_positional_arguments_remain_backward_compatible(fake_meshcat):
    viewer = meshcat_viewer.MeshCatViewer(open_browser=False)
    viewer.load_model(SimpleNamespace(nq=1), "visual")

    native = FakePinocchioVisualizer.instances[-1]
    assert native.collision_model is None
    assert native.visual_model == "visual"


def test_reload_cleans_previous_adapter_root(fake_meshcat):
    visualizer, _ = fake_meshcat
    viewer = meshcat_viewer.MeshCatViewer(open_browser=False)
    viewer.load_model(SimpleNamespace(nq=1), object())
    first = FakePinocchioVisualizer.instances[-1]
    first.viewerRootNodeName = viewer._root_node_name
    first.viewer = visualizer
    first_scene = visualizer[viewer._root_node_name]

    viewer.load_model(SimpleNamespace(nq=1), object())

    first_scene.delete.assert_called_once_with()


@pytest.mark.parametrize("q", [[1.0], [np.nan, 0.0], [np.inf, 0.0]])
def test_display_rejects_invalid_configuration(fake_meshcat, q):
    viewer = meshcat_viewer.MeshCatViewer(open_browser=False)
    viewer.load_model(SimpleNamespace(nq=2), object())
    native = FakePinocchioVisualizer.instances[-1]

    with pytest.raises(ValueError):
        viewer.display(q)
    native.display.assert_not_called()


def test_display_requires_loaded_model(fake_meshcat):
    viewer = meshcat_viewer.MeshCatViewer(open_browser=False)

    with pytest.raises(RuntimeError, match="No model is loaded"):
        viewer.display([0.0])


def test_model_nq_must_be_an_integer(fake_meshcat):
    viewer = meshcat_viewer.MeshCatViewer(open_browser=False)
    viewer.load_model(SimpleNamespace(nq=1.5), object())

    with pytest.raises(ValueError, match="model.nq"):
        viewer.display([0.0])


def test_failed_load_clears_ready_state(fake_meshcat, monkeypatch):
    viewer = meshcat_viewer.MeshCatViewer(open_browser=False)
    failing = Mock()
    failing.initViewer.side_effect = RuntimeError("load failed")
    monkeypatch.setattr(meshcat_viewer, "MeshcatVisualizer", Mock(return_value=failing))

    with pytest.raises(RuntimeError, match="load failed"):
        viewer.load_model(SimpleNamespace(nq=1), object())
    with pytest.raises(RuntimeError, match="No model is loaded"):
        viewer.display([0.0])


def test_close_is_idempotent_and_does_not_close_external_server(fake_meshcat):
    visualizer, _ = fake_meshcat
    viewer = meshcat_viewer.MeshCatViewer(
        zmq_url="tcp://external:6000", open_browser=False
    )
    viewer.load_model(SimpleNamespace(nq=1), object())
    native = FakePinocchioVisualizer.instances[-1]
    native.viewerRootNodeName = viewer._root_node_name
    native.viewer = visualizer
    scene = visualizer[viewer._root_node_name]

    viewer.close()
    viewer.close()

    scene.delete.assert_called_once_with()
    native.clean.assert_not_called()
    visualizer.close.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        viewer.display([0.0])


def test_close_stops_owned_server_process(fake_meshcat):
    visualizer, _ = fake_meshcat
    process = Mock()
    process.poll.return_value = None
    visualizer.window = SimpleNamespace(server_proc=process)
    viewer = meshcat_viewer.MeshCatViewer(zmq_url=None, open_browser=False)

    viewer.close()

    process.kill.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=5.0)


def test_browser_open_failure_stops_owned_server_process(fake_meshcat):
    visualizer, _ = fake_meshcat
    process = Mock()
    process.poll.return_value = None
    visualizer.window = SimpleNamespace(server_proc=process)
    visualizer.open.side_effect = RuntimeError("browser unavailable")

    with pytest.raises(RuntimeError, match="browser unavailable"):
        meshcat_viewer.MeshCatViewer(zmq_url=None, open_browser=True)

    process.kill.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=5.0)
