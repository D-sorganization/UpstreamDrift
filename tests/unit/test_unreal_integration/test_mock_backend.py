import numpy as np
import pytest

from src.unreal_integration.data_models import Quaternion, Vector3
from src.unreal_integration.mesh_loader import LoadedMesh
from src.unreal_integration.viewer_backends.config import ViewerConfig
from src.unreal_integration.viewer_backends.mock_backend import MockBackend


@pytest.fixture
def backend():
    config = ViewerConfig(width=640, height=480)
    b = MockBackend(config)
    return b


def test_mock_backend_initialization(backend):
    assert not backend.is_initialized
    backend.initialize()
    assert backend.is_initialized
    backend.shutdown()
    assert not backend.is_initialized


def test_mock_backend_add_mesh(backend):
    backend.initialize()
    mesh = LoadedMesh(
        name="mock",
        vertices=np.zeros((3, 3)),
        faces=np.zeros((3,), dtype=np.int32),
        materials=[],
        source_path="",
        format="",
    )

    # Add mesh
    name = backend.add_mesh(mesh, name="test_mesh")
    assert name == "test_mesh"
    assert "test_mesh" in backend._objects

    # Auto-generate name
    name2 = backend.add_mesh(mesh)
    assert name2.startswith("mock_mesh_")

    # Missing mesh
    with pytest.raises(ValueError, match="mesh must be provided"):
        backend.add_mesh(None)


def test_mock_backend_update_transform(backend):
    backend.initialize()
    mesh = LoadedMesh(
        name="mock",
        vertices=np.zeros((3, 3)),
        faces=np.zeros((3,), dtype=np.int32),
        materials=[],
        source_path="",
        format="",
    )
    name = backend.add_mesh(mesh, name="test_mesh")

    pos = Vector3(1.0, 2.0, 3.0)
    rot = Quaternion(0.0, 1.0, 0.0, 0.0)
    backend.update_transform(name, position=pos, rotation=rot, scale=2.5)

    obj = backend._objects[name]
    assert obj["position"] == pos
    assert obj["rotation"] == rot
    assert obj["scale"] == 2.5


def test_mock_backend_remove_clear(backend):
    backend.initialize()
    mesh = LoadedMesh(
        name="mock",
        vertices=np.zeros((3, 3)),
        faces=np.zeros((3,), dtype=np.int32),
        materials=[],
        source_path="",
        format="",
    )
    backend.add_mesh(mesh, name="mesh1")
    backend.add_mesh(mesh, name="mesh2")

    assert "mesh1" in backend._objects
    assert backend.remove_object("mesh1") is True
    assert "mesh1" not in backend._objects
    assert backend.remove_object("mesh1") is False

    with pytest.raises(ValueError, match="name must be provided"):
        backend.remove_object(None)

    backend.clear()
    assert len(backend._objects) == 0


def test_mock_backend_render(backend):
    backend.initialize()
    assert backend.render_count == 0
    img = backend.render()
    assert backend.render_count == 1
    assert img.shape == (480, 640, 4)
    assert img.dtype == np.uint8
    assert np.all(img == 0)
