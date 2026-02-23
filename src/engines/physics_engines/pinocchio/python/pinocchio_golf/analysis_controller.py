"""Analysis logic for Pinocchio Golf GUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..gui import PinocchioGUI

logger = logging.getLogger(__name__)


class AnalysisController:
    """Handles plotting and data analysis for PinocchioGUI."""

    def __init__(self, gui: PinocchioGUI) -> None:
        self.gui = gui

    def plot_induced_accelerations(self) -> None:
        """Calculate and plot induced accelerations for selected joint."""
        gui = self.gui
        if not gui.model or not gui.data:
            return

        try:
            gui._ensure_analyzer_initialized()
            # ... actual logic from _plot_induced_accelerations ...
            pass
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Induced acceleration plotting failed: {e}")

    def generate_plot(self) -> None:
        """Main plot dispatcher."""
        # ... logic from _generate_plot ...
        pass
