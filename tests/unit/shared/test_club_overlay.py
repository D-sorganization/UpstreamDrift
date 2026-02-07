import unittest
from typing import Any

from src.shared.python.club_data.display import ClubTargetOverlay


class TestClubTargetOverlay(unittest.TestCase):
    def test_instantiation_fails(self):
        """Test that instantiating the abstract base class raises TypeError."""
        with self.assertRaises(TypeError):
            ClubTargetOverlay()

    def test_subclass_instantiation(self):
        """Test that a concrete subclass can be instantiated."""
        class ConcreteOverlay(ClubTargetOverlay):
            def render(self, renderer: Any) -> None:
                pass

        overlay = ConcreteOverlay()
        self.assertIsInstance(overlay, ClubTargetOverlay)

if __name__ == '__main__':
    unittest.main()
