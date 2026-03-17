"""Tests for model_registry."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from src.launchers.model_registry import (
    ModelRegistry,
    ModelSpec,
    _registry,
    get_model_registry,
)


@pytest.fixture
def mock_yaml_data():
    return {
        "models": [
            {
                "id": "model_1",
                "name": "First Model",
                "description": "Test description 1",
                "type": "engine_managed",
                "path": "path/to/model_1.urdf",
                "engine_type": "mujoco",
            },
            {
                "id": "model_2",
                "name": "Second Model",
                "description": "Test description 2",
                "type": "special_app",
                "path": "path/to/script.py",
            },
        ]
    }


def test_model_spec():
    """Test ModelSpec dataclass."""
    spec = ModelSpec(
        id="test",
        name="Test",
        description="A test spec.",
        type="engine_managed",
        path="foo/bar.urdf",
    )
    assert spec.id == "test"
    assert spec.engine_type is None


def test_model_registry_init():
    """Test initializing registry."""
    registry = ModelRegistry("custom/path.yaml")
    assert registry.config_path == Path("custom/path.yaml")
    assert not registry._loaded


def test_model_registry_load_success(mock_yaml_data):
    """Test parsing yaml config correctly."""
    registry = ModelRegistry()
    mock_file = mock_open(read_data=yaml.dump(mock_yaml_data))

    with (
        patch("builtins.open", mock_file),
        patch.object(Path, "exists", return_value=True),
    ):
        registry.load(Path("/fake/root"))

    assert registry._loaded
    assert len(registry.models) == 2

    model = registry.get_model_by_id("model_1")
    assert model is not None
    assert model.name == "First Model"
    assert model.engine_type == "mujoco"

    models = registry.get_all_models()
    assert len(models) == 2


def test_model_registry_load_missing_file():
    """Test behaviour when config file is missing."""
    registry = ModelRegistry()

    with patch.object(Path, "exists", return_value=False):
        registry.load(Path("/fake/root"))

    assert not registry._loaded
    assert len(registry.models) == 0


def test_model_registry_yaml_error():
    """Test yaml parsing error handling."""
    registry = ModelRegistry()
    mock_file = mock_open(read_data="invalid: yaml: content\n - - -")

    with (
        patch("builtins.open", mock_file),
        patch.object(Path, "exists", return_value=True),
        pytest.raises(yaml.YAMLError),
    ):
        registry.load(Path("/fake/root"))


def test_model_registry_type_error(mock_yaml_data):
    """Test type error when loading."""
    # Introduce bad data to cause TypeError in ModelSpec initialization
    bad_data = {"models": [{"id": "bad", "unknown_arg": "value"}]}
    registry = ModelRegistry()
    mock_file = mock_open(read_data=yaml.dump(bad_data))

    with (
        patch("builtins.open", mock_file),
        patch.object(Path, "exists", return_value=True),
        pytest.raises(TypeError),
    ):
        registry.load(Path("/fake/root"))


def test_model_registry_os_error():
    """Test OS error when loading file."""
    registry = ModelRegistry()

    with (
        patch("builtins.open", side_effect=OSError("Read failed")),
        patch.object(Path, "exists", return_value=True),
        pytest.raises(OSError),
    ):
        registry.load(Path("/fake/root"))


def test_get_model_by_id_not_found(mock_yaml_data):
    """Test getting an unknown model."""
    registry = ModelRegistry()
    mock_file = mock_open(read_data=yaml.dump(mock_yaml_data))
    with (
        patch("builtins.open", mock_file),
        patch.object(Path, "exists", return_value=True),
    ):
        registry.load(Path("/fake/root"))

    assert registry.get_model_by_id("nonexistent") is None


def test_get_global_registry():
    """Test the global singleton accessor."""
    assert get_model_registry() is _registry
