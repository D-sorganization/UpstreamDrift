"""FSP metrics Qt widget (Phase 3 of the FSP epic, issue #5504).

Displays slope, direction, and a textual summary of the clubhead
deviations produced by
:func:`src.shared.python.biomechanics.fsp_integration.compute_swing_fsp`.

The module is **headless-safe**: if PyQt6 is not importable, a no-op
stub class is exported so callers can ``from ... import FspMetricsWidget``
without crashing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


try:
    from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

    HAS_QT = True
except ImportError:  # pragma: no cover - environment-dependent
    HAS_QT = False


if HAS_QT:

    class FspMetricsWidget(QWidget):  # type: ignore[misc]
        """Vertical panel showing FSP slope, direction, and deviation summary."""

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            layout = QVBoxLayout(self)
            self._slope_label = QLabel("Slope: —", self)
            self._direction_label = QLabel("Direction: —", self)
            self._chart_placeholder = QLabel("(deviation chart)", self)
            layout.addWidget(self._slope_label)
            layout.addWidget(self._direction_label)
            layout.addWidget(self._chart_placeholder)

        # --------------------------------------------------------------
        # Public API
        # --------------------------------------------------------------

        def set_result(self, fsp_result: Any) -> None:
            """Update labels and deviation summary from an FSP result.

            DbC:
                Precondition: *fsp_result* must expose ``slope_deg`` and
                ``direction_deg`` attributes (raises ``AttributeError``
                otherwise).  ``clubhead_deviations`` is optional -- if
                absent, the summary line falls back to a placeholder.
            """
            slope = float(fsp_result.slope_deg)
            direction = float(fsp_result.direction_deg)
            self._slope_label.setText(f"Slope: {slope:.2f}°")
            self._direction_label.setText(f"Direction: {direction:.2f}°")
            self._chart_placeholder.setText(self._summarise(fsp_result))

        # --------------------------------------------------------------
        # Helpers
        # --------------------------------------------------------------

        @staticmethod
        def _summarise(fsp_result: Any) -> str:
            deviations = getattr(fsp_result, "clubhead_deviations", None)
            if deviations is None:
                return "(deviation chart)"
            try:
                arr = np.asarray(deviations, dtype=np.float64)
                if arr.size == 0:
                    return "Clubhead deviations: (no samples)"
                mean_val = float(np.mean(arr))
                max_val = float(np.max(np.abs(arr)))
                return (
                    f"Clubhead deviations: mean={mean_val:.3f} m, max={max_val:.3f} m"
                )
            except (TypeError, ValueError) as exc:
                logger.debug("Could not summarise deviations: %s", exc)
                return "(deviation chart)"

else:  # pragma: no cover - headless path

    class FspMetricsWidget:  # type: ignore[no-redef]
        """Stub when Qt is unavailable -- headless-safe."""

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def set_result(self, fsp_result: Any) -> None: ...


__all__ = ["FspMetricsWidget", "HAS_QT"]
