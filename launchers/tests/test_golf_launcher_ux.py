
import pytest
from PyQt6.QtWidgets import QLabel, QWidget, QFrame
from PyQt6.QtCore import Qt
from unittest.mock import MagicMock, patch

# Import the class to test
from launchers.golf_launcher import GolfLauncher

@pytest.fixture
def launcher(qtbot):
    """Fixture to create a GolfLauncher instance with mocked dependencies."""
    # Mocking external dependencies to avoid splash screen delays / errors
    with patch('launchers.golf_launcher.ModelRegistry'), \
         patch('launchers.golf_launcher.EngineManager'), \
         patch('launchers.golf_launcher.GolfSplashScreen'), \
         patch('launchers.golf_launcher.DockerCheckThread'):

        # Instantiate the launcher
        widget = GolfLauncher()
        qtbot.addWidget(widget)
        return widget

def test_search_empty_state(launcher, qtbot):
    """Test that a 'No models found' message appears when search yields no results."""

    # 1. Setup: Add some dummy models
    model1 = MagicMock()
    model1.id = "model1"
    model1.name = "Test Model 1"
    model1.description = "Description 1"

    # Mock the cards
    card1 = QFrame() # Use real QFrame so we can add it to layout
    card1.model = model1

    launcher.model_cards = {
        "model1": card1
    }
    launcher.model_order = ["model1"]

    # Ensure grid starts populated
    launcher._rebuild_grid()
    assert launcher.grid_layout.count() == 1

    # 2. Act: Search for something that doesn't exist
    launcher.update_search_filter("NonExistentXYZ")

    # 3. Assert: Check for empty state label
    # NOTE: This assertion is expected to fail before the fix is implemented
    # We expect count to be 0 currently.

    count = launcher.grid_layout.count()

    if count == 0:
        # Current behavior (Fail)
        pytest.fail("Grid is empty. Expected 'No models found' label.")

    # Expected behavior after fix
    assert count == 1
    item = launcher.grid_layout.itemAt(0)
    widget = item.widget()
    assert isinstance(widget, QLabel)
    assert "No models found matching 'nonexistentxyz'" in widget.text()
