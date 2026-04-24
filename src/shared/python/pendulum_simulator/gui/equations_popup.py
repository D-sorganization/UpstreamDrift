"""LaTeX-quality math popup for the Pendulum Simulator."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from .equations_popup_jacobian_content import (
    CONSTRAINT_JACOBIAN_HTML,
    JACOBIAN_HTML,
)
from .equations_popup_reference_content import (
    DELTA_HTML,
    EOM_HTML,
    MASS_MATRIX_HTML,
    ZTCF_HTML,
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


_TOPICS = {
    EquationTopic.MASS_MATRIX: ("Mass Matrix — Derivation", MASS_MATRIX_HTML),
    EquationTopic.EQUATIONS_OF_MOTION: (
        "Equations of Motion — Full Reference",
        EOM_HTML,
    ),
    EquationTopic.DELTA_MATRIX: ("Delta Matrix (M⁺) — Inverse Dynamics", DELTA_HTML),
    EquationTopic.ZTCF_MATRIX: ("ZTCF Transfer Matrix — Endpoint Forces", ZTCF_HTML),
    EquationTopic.JACOBIAN: ("Geometric Jacobian — Velocity Mapping", JACOBIAN_HTML),
    EquationTopic.CONSTRAINT_JACOBIAN: (
        "Constraint Jacobian — Closed-Loop Kinematics",
        CONSTRAINT_JACOBIAN_HTML,
    ),
}


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

    if topic not in _TOPICS:
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
