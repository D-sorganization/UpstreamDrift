"""Unit tests for the Gepetto viewer adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.dtack.viz import geppetto_viewer

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
def fake_geppetto(monkeypatch):
    FakePinocchioVisualizer.instances.clear()
    client = Mock()
    client_cls = Mock(return_value=client)
    monkeypatch.setattr(geppetto_viewer, "GEPETTO_AVAILABLE", True)
    monkeypatch.setattr(
        geppetto_viewer, "_corbaserver", SimpleNamespace(Client=client_cls)
    )
    monkeypatch.setattr(geppetto_viewer, "GepettoVisualizer", FakePinocchioVisualizer)
    monkeypatch.setattr(
        geppetto_viewer,
        "pin",
        SimpleNamespace(neutral=lambda model: np.zeros(model.nq)),
    )
    return client, client_cls


def test_load_stores_native_visualizer_and_routes_geometry(fake_geppetto):
    client, _ = fake_geppetto
    model = SimpleNamespace(nq=2)
    collision_model = object()
    visual_model = object()
    viewer = geppetto_viewer.GeppettoViewer()

    viewer.load_model(model, visual_model, collision_model=collision_model)

    native = FakePinocchioVisualizer.instances[-1]
    assert native.collision_model is collision_model
    assert native.visual_model is visual_model
    native.initViewer.assert_called_once_with(
        viewer=client, windowName="python-pinocchio", sceneName="world"
    )
    native.loadViewerModel.assert_called_once_with(rootNodeName=viewer._root_node_name)


def test_display_dispatches_and_none_uses_neutral(fake_geppetto):
    viewer = geppetto_viewer.GeppettoViewer()
    model = SimpleNamespace(nq=2)
    viewer.load_model(model, object())
    native = FakePinocchioVisualizer.instances[-1]

    viewer.display([1.0, 2.0])
    viewer.display()

    np.testing.assert_array_equal(native.display.call_args_list[0].args[0], [1.0, 2.0])
    np.testing.assert_array_equal(native.display.call_args_list[1].args[0], [0.0, 0.0])


@pytest.mark.parametrize("q", [[1.0], [np.nan, 0.0], [np.inf, 0.0]])
def test_display_rejects_invalid_configuration(fake_geppetto, q):
    viewer = geppetto_viewer.GeppettoViewer()
    viewer.load_model(SimpleNamespace(nq=2), object())
    native = FakePinocchioVisualizer.instances[-1]

    with pytest.raises(ValueError):
        viewer.display(q)
    native.display.assert_not_called()


def test_connection_failure_is_not_reported_as_ready(monkeypatch):
    monkeypatch.setattr(geppetto_viewer, "GEPETTO_AVAILABLE", True)
    monkeypatch.setattr(geppetto_viewer, "GepettoVisualizer", object)
    monkeypatch.setattr(
        geppetto_viewer,
        "_corbaserver",
        SimpleNamespace(Client=Mock(side_effect=OSError("offline"))),
    )

    with pytest.raises(RuntimeError, match="connect to Geppetto"):
        geppetto_viewer.GeppettoViewer()


def test_failed_load_clears_ready_state(fake_geppetto, monkeypatch):
    viewer = geppetto_viewer.GeppettoViewer()
    failing = Mock()
    failing.loadViewerModel.side_effect = RuntimeError("load failed")
    monkeypatch.setattr(
        geppetto_viewer, "GepettoVisualizer", Mock(return_value=failing)
    )

    with pytest.raises(RuntimeError, match="load failed"):
        viewer.load_model(SimpleNamespace(nq=1), object())
    with pytest.raises(RuntimeError, match="No model is loaded"):
        viewer.display([0.0])


def test_missing_dependency_is_actionable(monkeypatch):
    monkeypatch.setattr(geppetto_viewer, "GEPETTO_AVAILABLE", False)

    with pytest.raises(ImportError, match="gepetto-viewer-corba"):
        geppetto_viewer.GeppettoViewer()


def test_display_requires_loaded_model(fake_geppetto):
    viewer = geppetto_viewer.GeppettoViewer()

    with pytest.raises(RuntimeError, match="No model is loaded"):
        viewer.display([0.0])


def test_reload_cleans_previous_adapter_root(fake_geppetto):
    viewer = geppetto_viewer.GeppettoViewer()
    viewer.load_model(SimpleNamespace(nq=1), object())
    first = FakePinocchioVisualizer.instances[-1]
    first.viewerRootNodeName = viewer._root_node_name
    first.viewer = SimpleNamespace(gui=Mock())

    viewer.load_model(SimpleNamespace(nq=1), object())

    first.viewer.gui.deleteNode.assert_called_once_with(viewer._root_node_name, True)


def test_close_is_idempotent_and_cleans_only_native_scene(fake_geppetto):
    client, _ = fake_geppetto
    viewer = geppetto_viewer.GeppettoViewer()
    viewer.load_model(SimpleNamespace(nq=1), object())
    native = FakePinocchioVisualizer.instances[-1]
    native.viewerRootNodeName = viewer._root_node_name
    native.viewer = SimpleNamespace(gui=Mock())

    viewer.close()
    viewer.close()

    native.viewer.gui.deleteNode.assert_called_once_with(viewer._root_node_name, True)
    native.clean.assert_not_called()
    client.shutdown.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        viewer.display([0.0])
