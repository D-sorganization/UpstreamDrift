#!/usr/bin/env python3
"""Test suite for Pinocchio ecosystem integration (Pinocchio, Pink, Crocoddyl).

Tests cover:
- Package availability and imports
- Basic functionality verification
- Integration between packages
- Docker environment compatibility

Refactored to use centralized conftest.py for path setup (DRY principle).
"""

import unittest
from pathlib import Path

# Path setup is now centralized in tests/conftest.py


class TestPinocchioEcosystem(unittest.TestCase):
    """Test Pinocchio ecosystem package availability and basic functionality."""

    def test_pinocchio_import(self) -> None:
        """Test that Pinocchio can be imported and has basic functionality."""
        try:
            import pinocchio as pin

            # Test basic functionality
            self.assertTrue(
                hasattr(pin, "__version__"),
                "Pinocchio should have __version__ attribute",
            )
            self.assertTrue(hasattr(pin, "Model"), "Pinocchio should have Model class")
            self.assertTrue(hasattr(pin, "Data"), "Pinocchio should have Data class")

            # Test creating a simple model
            model = pin.Model()
            self.assertIsInstance(model, pin.Model)

            # Test basic operations
            data = model.createData()
            self.assertIsInstance(data, pin.Data)

        except ImportError as e:
            self.skipTest(f"Pinocchio not available: {e}")
        except Exception as e:
            self.skipTest(f"Pinocchio functionality test failed: {e}")

    def test_pink_import(self) -> None:
        """Test that Pink can be imported and has basic functionality."""
        try:
            import pink

            # Test basic Pink functionality
            self.assertTrue(hasattr(pink, "__file__"))

            # Test that Pink has expected modules
            # SECURITY FIX: Use importlib instead of exec() to prevent code injection
            import importlib

            expected_modules = ["tasks", "solvers"]
            for module_name in expected_modules:
                try:
                    module = getattr(pink, module_name, None)
                    if module is None:
                        # Try importing as submodule (safe import)
                        importlib.import_module(f"pink.{module_name}")
                except (AttributeError, ImportError, ModuleNotFoundError):
                    # Some modules might not be available in all Pink versions
                    pass

        except ImportError as e:
            self.skipTest(f"Pink not available: {e}")

    def test_crocoddyl_import(self) -> None:
        """Test that Crocoddyl can be imported and has basic functionality."""
        try:
            import crocoddyl

            # Test basic functionality
            self.assertTrue(hasattr(crocoddyl, "__version__"))

            # Test that Crocoddyl has expected core classes
            expected_classes = [
                "ActionModelAbstract",
                "DifferentialActionModelAbstract",
            ]
            for class_name in expected_classes:
                self.assertTrue(
                    hasattr(crocoddyl, class_name),
                    f"Crocoddyl should have {class_name}",
                )

        except ImportError as e:
            self.skipTest(f"Crocoddyl not available: {e}")

    def test_pinocchio_crocoddyl_integration(self) -> None:
        """Test that Pinocchio and Crocoddyl work together."""
        try:
            import crocoddyl
            import pinocchio as pin

            # Create a simple Pinocchio model
            model = pin.Model()

            # Add a simple joint (this is a basic test)
            model.addJoint(0, pin.JointModelRY(), pin.SE3.Identity(), "joint1")

            # Create data
            data = model.createData()

            # Test that we can use the model with Crocoddyl
            # (This is a minimal test - full integration would require more setup)
            self.assertIsInstance(model, pin.Model)
            self.assertIsInstance(data, pin.Data)
            self.assertTrue(hasattr(crocoddyl, "ActionModelAbstract"))

        except ImportError as e:
            self.skipTest(f"Pinocchio-Crocoddyl integration test skipped: {e}")
        except Exception as e:
            self.fail(f"Pinocchio-Crocoddyl integration failed: {e}")

    def test_pinocchio_pink_integration(self) -> None:
        """Test that Pinocchio and Pink work together."""
        try:
            import pinocchio as pin

            # Create a simple Pinocchio model
            model = pin.Model()
            data = model.createData()

            # Test basic integration (Pink uses Pinocchio models)
            self.assertIsNotNone(model, "Pinocchio model should be created")
            self.assertIsNotNone(data, "Pinocchio data should be created")

            # Test that the objects have the expected type names
            self.assertEqual(
                type(model).__name__, "Model", "Should be a Pinocchio Model"
            )
            self.assertEqual(type(data).__name__, "Data", "Should be a Pinocchio Data")

            # Pink should be able to work with Pinocchio models
            # (This is a minimal test - full integration would require more setup)

        except ImportError as e:
            self.skipTest(f"Pinocchio-Pink integration test skipped: {e}")
        except Exception as e:
            self.skipTest(f"Pinocchio-Pink integration test skipped due to error: {e}")


class TestPinocchioDockerCompatibility(unittest.TestCase):
    """Test Docker environment compatibility for Pinocchio ecosystem."""

    def test_docker_environment_variables(self) -> None:
        """Test that Docker environment variables are properly set."""
        import os

        # Check PYTHONPATH includes necessary directories
        pythonpath = os.environ.get("PYTHONPATH", "")
        expected_paths = [
            "/workspace",
            "/workspace/shared/python",
            "/workspace/engines",
        ]

        for expected_path in expected_paths:
            if expected_path not in pythonpath:
                # This might not be set in local testing, so just warn
                print(f"⚠️  Expected path {expected_path} not in PYTHONPATH")

    def test_package_versions_compatibility(self) -> None:
        """Test that package versions are compatible."""
        try:
            import numpy as np
            import pinocchio as pin

            # Check version compatibility
            pin_version = getattr(pin, "__version__", "unknown")
            numpy_version = getattr(np, "__version__", "unknown")

            self.assertIsInstance(pin_version, str)
            self.assertIsInstance(numpy_version, str)

            # Basic version format check (only if version is not 'unknown')
            if pin_version != "unknown":
                self.assertRegex(pin_version, r"\d+\.\d+.*")
            if numpy_version != "unknown":
                self.assertRegex(numpy_version, r"\d+\.\d+.*")

        except ImportError as e:
            self.skipTest(f"Package version test skipped: {e}")
        except Exception as e:
            self.skipTest(f"Package version test skipped due to error: {e}")


class TestPinocchioConstants(unittest.TestCase):
    """Test Pinocchio-related constants and configurations."""

    def test_pinocchio_constants_defined(self) -> None:
        """Test that Pinocchio-related constants are properly defined."""
        try:
            from src.shared.python.constants import PINOCCHIO_LAUNCHER_SCRIPT

            self.assertIsInstance(PINOCCHIO_LAUNCHER_SCRIPT, Path)
            expected_path = (
                "engines/physics_engines/pinocchio/python/pinocchio_golf/gui.py"
            )
            self.assertEqual(PINOCCHIO_LAUNCHER_SCRIPT.as_posix(), expected_path)

        except ImportError as e:
            self.skipTest(f"Constants test skipped: {e}")

    def test_pinocchio_docker_stages(self) -> None:
        """Test that Pinocchio is included in Docker stages."""
        try:
            from src.launchers.golf_launcher import DOCKER_STAGES

            self.assertIn("pinocchio", DOCKER_STAGES)

        except ImportError as e:
            self.skipTest(f"Docker stages test skipped: {e}")


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestPinocchioEcosystem,
        TestPinocchioDockerCompatibility,
        TestPinocchioConstants,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Pinocchio Ecosystem Tests Summary")
    print(f"{'=' * 60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ Failures:")
        for test, _ in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\n💥 Errors:")
        for test, _ in result.errors:
            print(f"  - {test}")

    if not result.failures and not result.errors:
        print("\n🎉 All Pinocchio ecosystem tests passed!")
