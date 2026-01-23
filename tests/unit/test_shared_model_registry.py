"""Unit tests for shared model registry."""

import unittest
from unittest.mock import mock_open, patch

import yaml

from src.shared.python.model_registry import ModelConfig, ModelRegistry


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_yaml = """
        models:
          - id: "test_model"
            name: "Test Model"
            description: "A test model"
            type: "urdf"
            path: "path/to/model.urdf"
          - id: "mj_model"
            name: "MuJoCo Model"
            description: "A MuJoCo model"
            type: "mjcf"
            path: "path/to/model.xml"
        """
        self.invalid_yaml = """
        invalid_key:
          - id: "test_model"
        """

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_registry_success(self, mock_file, mock_exists):
        """Test successful loading of model registry."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = self.valid_yaml
        # Need to ensure yaml.safe_load reads from the mock
        # But yaml.safe_load takes a stream, and mock_open returns a mock that acts as a stream

        # Since we are mocking open, we need to make sure yaml.safe_load works with it
        # Or we can mock yaml.safe_load directly
        with patch("yaml.safe_load", return_value=yaml.safe_load(self.valid_yaml)):
            registry = ModelRegistry("dummy_path.yaml")

        self.assertEqual(len(registry.models), 2)
        self.assertIn("test_model", registry.models)
        self.assertIn("mj_model", registry.models)

        model = registry.get_model("test_model")
        self.assertIsInstance(model, ModelConfig)
        if model is not None:
            self.assertEqual(model.name, "Test Model")
            self.assertEqual(model.type, "urdf")

    @patch("pathlib.Path.exists")
    def test_load_registry_not_found(self, mock_exists):
        """Test loading when registry file does not exist."""
        mock_exists.return_value = False
        registry = ModelRegistry("dummy_path.yaml")
        self.assertEqual(len(registry.models), 0)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_registry_invalid_format(self, mock_file, mock_exists):
        """Test loading registry with invalid format."""
        mock_exists.return_value = True
        with patch("yaml.safe_load", return_value=yaml.safe_load(self.invalid_yaml)):
            registry = ModelRegistry("dummy_path.yaml")

        self.assertEqual(len(registry.models), 0)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_registry_empty(self, mock_file, mock_exists):
        """Test loading empty registry."""
        mock_exists.return_value = True
        with patch("yaml.safe_load", return_value={}):
            registry = ModelRegistry("dummy_path.yaml")

        self.assertEqual(len(registry.models), 0)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_registry_yaml_error(self, mock_file, mock_exists):
        """Test loading registry with YAML syntax error."""
        mock_exists.return_value = True
        with patch("yaml.safe_load", side_effect=yaml.YAMLError("Error")):
            registry = ModelRegistry("dummy_path.yaml")

        self.assertEqual(len(registry.models), 0)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_registry_partial_failure(self, mock_file, mock_exists):
        """Test loading registry where one model is invalid."""
        partial_yaml = """
        models:
          - id: "valid_model"
            name: "Valid Model"
            description: "Valid"
            type: "urdf"
            path: "path.urdf"
          - id: "invalid_model"
            # Missing required fields
        """
        mock_exists.return_value = True
        with patch("yaml.safe_load", return_value=yaml.safe_load(partial_yaml)):
            registry = ModelRegistry("dummy_path.yaml")

        self.assertEqual(len(registry.models), 1)
        self.assertIn("valid_model", registry.models)

    def test_get_methods(self):
        """Test retrieval methods."""
        # Setup registry manually
        registry = ModelRegistry("dummy.yaml")
        # Bypass loading
        registry.models = {
            "m1": ModelConfig("m1", "M1", "D1", "urdf", "p1"),
            "m2": ModelConfig("m2", "M2", "D2", "mjcf", "p2"),
            "m3": ModelConfig("m3", "M3", "D3", "urdf", "p3"),
        }

        model_m1 = registry.get_model("m1")
        self.assertIsNotNone(model_m1)
        if model_m1 is not None:
            self.assertEqual(model_m1.id, "m1")
        self.assertIsNone(registry.get_model("nonexistent"))

        all_models = registry.get_all_models()
        self.assertEqual(len(all_models), 3)

        urdf_models = registry.get_models_by_type("urdf")
        self.assertEqual(len(urdf_models), 2)
        self.assertTrue(all(m.type == "urdf" for m in urdf_models))
