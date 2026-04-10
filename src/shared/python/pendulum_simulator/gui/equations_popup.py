"""
LaTeX-quality math popup for the Pendulum Simulator.

Displays rendered mathematical equations for:
- Mass matrix derivation and physical interpretation
- Full equations of motion for double, triple, and golfer model
- Coriolis, gravity, friction, and joint limit terms
- Energy conservation and Lagrangian derivation
- Golfer 8-DOF Baumgarte constrained system (KKT formulation)

Uses QTextBrowser with rich HTML + CSS for professional-quality rendering.

Design by Contract
------------------
- show_equations_popup(parent, topic) is the single entry point.
- topic must be one of the EquationTopic enum values.
- The popup is non-modal so the user can keep it open alongside the sim.

DRY
---
HTML template and styling are defined once in split modules:
- _equations_popup_css.py: shared CSS
- _equations_popup_dynamics_html.py: mass matrix, EOM, delta matrix HTML
- _equations_popup_jacobians_html.py: ZTCF, Jacobian, constraint Jacobian HTML
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from ._equations_popup_dynamics_html import (
    _DELTA_HTML,
    _EOM_HTML,
    _MASS_MATRIX_HTML,
)
from ._equations_popup_jacobians_html import (
    _CONSTRAINT_JACOBIAN_HTML,
    _JACOBIAN_HTML,
    _ZTCF_HTML,
)

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QDialog, QWidget

logger = logging.getLogger(__name__)


class EquationTopic(Enum):
    """Available equation topics."""

    MASS_MATRIX = "mass_matrix"
    EQUATIONS_OF_MOTION = "equations_of_motion"
    DELTA_MATRIX = "delta_matrix"
    ZTCF_MATRIX = "ztcf_matrix"
    JACOBIAN = "jacobian"
    CONSTRAINT_JACOBIAN = "constraint_jacobian"


# ---------------------------------------------------------------------------
# Topic registry
# ---------------------------------------------------------------------------

_TOPICS = {
    EquationTopic.MASS_MATRIX: ("Mass Matrix — Derivation", _MASS_MATRIX_HTML),
    EquationTopic.EQUATIONS_OF_MOTION: (
        "Equations of Motion — Full Reference",
        _EOM_HTML,
    ),
    EquationTopic.DELTA_MATRIX: ("Delta Matrix (M+) — Inverse Dynamics", _DELTA_HTML),
    EquationTopic.ZTCF_MATRIX: ("ZTCF Transfer Matrix — Endpoint Forces", _ZTCF_HTML),
    EquationTopic.JACOBIAN: ("Geometric Jacobian — Velocity Mapping", _JACOBIAN_HTML),
    EquationTopic.CONSTRAINT_JACOBIAN: (
        "Constraint Jacobian — Closed-Loop Kinematics",
        _CONSTRAINT_JACOBIAN_HTML,
    ),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def show_equations_popup(parent: QWidget | None, topic: EquationTopic) -> QDialog:
    """Show a non-modal equations popup.

    Pre: topic is a valid EquationTopic.
    Post: returns the QDialog instance (caller may discard).
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QDialog,
        QPushButton,
        QTextBrowser,
        QVBoxLayout,
    )

    if not (topic in _TOPICS):  # noqa: E713
        raise ValueError(f"Unknown topic: {topic}")
    title, html = _TOPICS[topic]

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumSize(720, 600)
    dlg.setStyleSheet("QDialog { background: #1a1a28; }")

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)

    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    browser.setHtml(html)
    browser.setStyleSheet("QTextBrowser { background: #1a1a28; border: none; }")
    layout.addWidget(browser)

    copy_btn = QPushButton("Copy to Clipboard")

    def _copy_text() -> None:
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(browser.toPlainText())

    copy_btn.clicked.connect(_copy_text)
    layout.addWidget(copy_btn)

    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.show()
    logger.info("Opened equations popup: %s", title)
    return dlg


__all__ = ["EquationTopic", "show_equations_popup"]
