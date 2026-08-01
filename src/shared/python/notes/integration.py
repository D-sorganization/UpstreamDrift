"""Integration helper for embedding notes in host applications."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

from .notes_dock_widget import NotesDockWidget


def attach_notes_dock(
    main_window: QWidget,
    project_dir: Path | str,
    area: Qt.DockWidgetArea = Qt.DockWidgetArea.RightDockWidgetArea,
    title: str = "Notes",
) -> NotesDockWidget:
    """Attach a reusable notes dock to a Qt host window."""
    if not hasattr(main_window, "addDockWidget"):
        raise ValueError("main_window must support addDockWidget")

    dock = NotesDockWidget(project_dir=project_dir, title=title, parent=main_window)
    main_window.addDockWidget(area, dock)
    return dock
