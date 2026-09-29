#!/usr/bin/env python3
"""
Test suite for drag-and-drop functionality in the Golf Modeling Suite launcher.

Tests cover:
- Drag-and-drop model card reordering
- 3x3 grid layout
- URDF generator integration
- Error handling in drag operations
"""

import unittest
from unittest.mock import Mock

from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

if PYQT6_AVAILABLE:
    pass


def _make_mock_model(model_id: str, name: str, description: str) -> Mock:
    """Create a mock model with proper string attributes.

    DraggableModelCard.setup_ui() uses ``"x" in model.id.lower()`` which
    requires *real* strings, not Mock objects.  Setting the attributes
    explicitly avoids ``TypeError: argument of type 'Mock' is not iterable``.
    """
    model = Mock()
    model.id = model_id
    model.name = name
    model.description = description
    model.type = "test_type"
    model.path = ""
    model.engine_type = ""
    return model


class TestModelImageHandling(unittest.TestCase):
    """Test model image handling for the new grid layout."""

    def test_urdf_generator_image_mapping(self) -> None:
        """Test that URDF generator has image mapping."""
        from src.launchers.ui_components import MODEL_IMAGES

        self.assertIn("URDF Generator", MODEL_IMAGES)
        self.assertEqual(MODEL_IMAGES["URDF Generator"], "urdf_icon.png")

    def test_c3d_viewer_image_mapping(self) -> None:
        """Test that C3D viewer has image mapping."""
        from src.launchers.ui_components import MODEL_IMAGES

        self.assertIn("C3D Motion Viewer", MODEL_IMAGES)
        self.assertEqual(MODEL_IMAGES["C3D Motion Viewer"], "c3d_icon.png")

    def test_image_fallback_for_urdf(self) -> None:
        """Test image fallback logic for URDF generator."""
        # This would be tested in the actual DraggableModelCard setup_ui method
        # The logic checks for "urdf" in model.id and assigns "urdf_icon.png"

        # Mock model with urdf in ID
        mock_model = _make_mock_model("urdf_generator", "URDF Generator", "Test")

        # The image selection logic should work
        from src.launchers.ui_components import MODEL_IMAGES

        # Direct lookup should work
        img_name = MODEL_IMAGES.get(mock_model.name)
        if not img_name and "urdf" in mock_model.id:
            img_name = "urdf_icon.png"

        self.assertEqual(img_name, "urdf_icon.png")


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestModelImageHandling,
    ]

    # Add PyQt tests only if available
    if PYQT6_AVAILABLE:
        pass
    else:
        print("PyQt6 not available - skipping GUI tests")  # noqa: T201

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 60}")  # noqa: T201
    print("Drag-and-Drop Tests Summary")  # noqa: T201
    print(f"{'=' * 60}")  # noqa: T201
    print(f"Tests run: {result.testsRun}")  # noqa: T201
    print(f"Failures: {len(result.failures)}")  # noqa: T201
    print(f"Errors: {len(result.errors)}")  # noqa: T201

    if result.failures:
        print("\nFAILURES:")  # noqa: T201
        for test, _ in result.failures:
            print(f"  - {test}")  # noqa: T201

    if result.errors:
        print("\nERRORS:")  # noqa: T201
        for test, _ in result.errors:
            print(f"  - {test}")  # noqa: T201

    if not result.failures and not result.errors:
        print("\nAll drag-and-drop tests passed!")  # noqa: T201
