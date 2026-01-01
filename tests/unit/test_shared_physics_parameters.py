"""Unit tests for shared physics parameters."""

import json
import unittest
from unittest.mock import mock_open, patch

from shared.python.physics_parameters import (
    ParameterCategory,
    PhysicsParameter,
    PhysicsParameterRegistry,
    get_registry,
)


class TestPhysicsParameter(unittest.TestCase):
    """Test cases for PhysicsParameter."""

    def test_validation(self):
        """Test parameter validation."""
        param = PhysicsParameter(
            "TEST",
            10.0,
            "m",
            ParameterCategory.SIMULATION,
            "Test",
            "Test",
            min_value=0.0,
            max_value=20.0,
        )

        # Valid
        valid, msg = param.validate(5.0)
        self.assertTrue(valid)
        self.assertEqual(msg, "")

        # Out of bounds
        valid, msg = param.validate(-1.0)
        self.assertFalse(valid)
        self.assertIn("must be >=", msg)

        valid, msg = param.validate(25.0)
        self.assertFalse(valid)
        self.assertIn("must be <=", msg)

        # Type error
        valid, msg = param.validate("string")
        self.assertFalse(valid)
        self.assertIn("must be numeric", msg)

    def test_constant(self):
        """Test constant parameter."""
        param = PhysicsParameter(
            "CONST",
            10.0,
            "m",
            ParameterCategory.SIMULATION,
            "Test",
            "Test",
            is_constant=True,
        )

        valid, msg = param.validate(20.0)
        self.assertFalse(valid)
        self.assertIn("is a constant", msg)

    def test_to_dict(self):
        """Test dictionary conversion."""
        param = PhysicsParameter(
            "TEST", 10.0, "m", ParameterCategory.SIMULATION, "Test", "Test"
        )
        d = param.to_dict()
        self.assertEqual(d["value"], 10.0)
        self.assertEqual(d["category"], "simulation")


class TestPhysicsParameterRegistry(unittest.TestCase):
    """Test cases for PhysicsParameterRegistry."""

    def setUp(self):
        """Set up registry."""
        # Use a fresh registry for each test
        self.registry = PhysicsParameterRegistry()
        # Clear default parameters for cleaner testing, or just use them
        self.registry.parameters = {}

    def test_register_get(self):
        """Test registering and getting parameters."""
        param = PhysicsParameter(
            "TEST", 10.0, "m", ParameterCategory.SIMULATION, "Test", "Test"
        )
        self.registry.register(param)

        retrieved = self.registry.get("TEST")
        self.assertEqual(retrieved, param)
        self.assertIsNone(self.registry.get("NONEXISTENT"))

    def test_set_value(self):
        """Test setting parameter values."""
        param = PhysicsParameter(
            "TEST",
            10.0,
            "m",
            ParameterCategory.SIMULATION,
            "Test",
            "Test",
            min_value=0.0,
            max_value=20.0,
        )
        self.registry.register(param)

        success, msg = self.registry.set("TEST", 15.0)
        self.assertTrue(success)
        param = self.registry.get("TEST")
        self.assertIsNotNone(param)
        self.assertEqual(param.value, 15.0)

        success, msg = self.registry.set("TEST", 25.0)
        self.assertFalse(success)
        param = self.registry.get("TEST")
        self.assertIsNotNone(param)
        self.assertEqual(param.value, 15.0)  # Should not change

        success, msg = self.registry.set("NONEXISTENT", 1.0)
        self.assertFalse(success)

    def test_get_by_category(self):
        """Test filtering by category."""
        p1 = PhysicsParameter("P1", 1.0, "u", ParameterCategory.BALL, "D", "S")
        p2 = PhysicsParameter("P2", 2.0, "u", ParameterCategory.CLUB, "D", "S")

        self.registry.register(p1)
        self.registry.register(p2)

        ball_params = self.registry.get_by_category(ParameterCategory.BALL)
        self.assertEqual(len(ball_params), 1)
        self.assertEqual(ball_params[0], p1)

    def test_export_import_json(self):
        """Test JSON export and import."""
        p1 = PhysicsParameter("P1", 1.0, "u", ParameterCategory.BALL, "D", "S")
        self.registry.register(p1)

        # Mock open for export
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            self.registry.export_to_json("test.json")
            mock_file.assert_called_with("test.json", "w")
            # Verify write was called
            handle = mock_file()
            handle.write.assert_called()

        # Mock open for import
        import_data = {"P1": {"value": 2.0}}
        with patch(
            "builtins.open", new_callable=mock_open, read_data=json.dumps(import_data)
        ):
            count = self.registry.import_from_json("test.json")
            self.assertEqual(count, 1)
            param = self.registry.get("P1")
            self.assertIsNotNone(param)
            self.assertEqual(param.value, 2.0)

    def test_get_summary(self):
        """Test summary generation."""
        p1 = PhysicsParameter("P1", 1.0, "u", ParameterCategory.BALL, "D", "S")
        self.registry.register(p1)

        summary = self.registry.get_summary()
        self.assertIn("Physics Parameter Registry", summary)
        self.assertIn("P1", summary)
        self.assertIn("BALL", summary)

    def test_singleton(self):
        """Test singleton access."""
        reg1 = get_registry()
        reg2 = get_registry()
        self.assertIs(reg1, reg2)
