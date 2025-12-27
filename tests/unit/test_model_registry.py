"""Unit tests for ModelRegistry."""

import tempfile
from pathlib import Path

import yaml

from shared.python.model_registry import ModelRegistry


class TestModelRegistry:
    """Test cases for ModelRegistry."""

    def test_load_valid_registry(self):
        """Test loading a valid model registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            config_data = {
                "models": [
                    {
                        "id": "test_model",
                        "name": "Test Model",
                        "description": "A test model",
                        "type": "mjcf",
                        "path": "engines/test/model.xml",
                    }
                ]
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            registry = ModelRegistry(config_path)
            assert len(registry.models) == 1

            model = registry.get_model("test_model")
            assert model is not None
            assert model.id == "test_model"
            assert model.name == "Test Model"
            assert model.type == "mjcf"

    def test_load_empty_registry_file(self):
        """Test loading an empty registry file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            config_path.write_text("", encoding="utf-8")

            registry = ModelRegistry(config_path)
            assert len(registry.models) == 0

    def test_load_missing_registry(self):
        """Test loading when registry file doesn't exist."""
        # Using a path that definitely doesn't exist
        registry = ModelRegistry(Path("/nonexistent/path/models.yaml"))
        assert len(registry.models) == 0

    def test_load_malformed_yaml(self):
        """Test loading a malformed YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            # Write invalid YAML (tab character instead of spaces)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("models:\n\t- id: test")

            registry = ModelRegistry(config_path)
            # Should handle exception gracefully (log error) and have 0 models
            assert len(registry.models) == 0

    def test_load_invalid_model_format(self):
        """Test loading registry with invalid model structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            # Missing required fields like 'name', 'type'
            config_data = {
                "models": [
                    {
                        "id": "bad_model",
                        # Missing name, type, path
                    }
                ]
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            registry = ModelRegistry(config_path)
            # Should skip the bad model
            assert len(registry.models) == 0

    def test_get_all_models(self):
        """Test retrieving all models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            config_data = {
                "models": [
                    {
                        "id": "m1",
                        "name": "Model 1",
                        "description": "D1",
                        "type": "mjcf",
                        "path": "p1",
                    },
                    {
                        "id": "m2",
                        "name": "Model 2",
                        "description": "D2",
                        "type": "urdf",
                        "path": "p2",
                    },
                ]
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            registry = ModelRegistry(config_path)
            models = registry.get_all_models()
            assert len(models) == 2
            assert {m.id for m in models} == {"m1", "m2"}

    def test_get_models_by_type(self):
        """Test filtering models by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "models.yaml"
            config_data = {
                "models": [
                    {
                        "id": "m1",
                        "name": "M1",
                        "description": "D1",
                        "type": "mjcf",
                        "path": "p1",
                    },
                    {
                        "id": "m2",
                        "name": "M2",
                        "description": "D2",
                        "type": "drake",
                        "path": "p2",
                    },
                    {
                        "id": "m3",
                        "name": "M3",
                        "description": "D3",
                        "type": "mjcf",
                        "path": "p3",
                    },
                ]
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            registry = ModelRegistry(config_path)
            mjcf_models = registry.get_models_by_type("mjcf")
            assert len(mjcf_models) == 2
            assert {m.id for m in mjcf_models} == {"m1", "m3"}

            drake_models = registry.get_models_by_type("drake")
            assert len(drake_models) == 1
            assert drake_models[0].id == "m2"
