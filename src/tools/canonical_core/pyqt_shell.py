"""PyQt6 shell widgets for canonical-core tools."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.tools.canonical_core.registry import CanonicalCoreTool


class CanonicalCoreShellWidget(QWidget):
    """Thin PyQt6 app-shell entry for a canonical-core service surface."""

    def __init__(self, descriptor: CanonicalCoreTool, parent: QWidget | None = None):
        super().__init__(parent)
        self._descriptor = descriptor
        self.setObjectName(f"{descriptor.tool_id}_shell")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        heading = QLabel(self._descriptor.name, self)
        heading.setObjectName("canonicalCoreHeading")
        heading.setAlignment(Qt.AlignmentFlag.AlignLeft)

        description = QLabel(self._descriptor.description, self)
        description.setWordWrap(True)
        description.setObjectName("canonicalCoreDescription")

        body = QFrame(self)
        body.setObjectName("canonicalCoreServicePanel")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(10)

        route = QLabel(f"React route: {self._descriptor.web_route}", body)
        route.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        route.setObjectName("canonicalCoreRoute")

        surfaces = QLabel("Shells: PyQt6 desktop, React/Tauri web", body)
        surfaces.setObjectName("canonicalCoreSurfaces")

        service = QLabel("Service boundary: canonical-core services", body)
        service.setObjectName("canonicalCoreServiceBoundary")

        body_layout.addWidget(route)
        body_layout.addWidget(surfaces)
        body_layout.addWidget(service)
        body_layout.addStretch(1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        open_react = QPushButton("Open React surface", self)
        open_react.setEnabled(False)
        open_react.setToolTip(
            "The React shell consumes the same launcher manifest route."
        )
        actions.addWidget(open_react)

        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addWidget(body)
        layout.addLayout(actions)
        layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
