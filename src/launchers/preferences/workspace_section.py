"""Preferences subpage: Workspace — default layout mode (Tools #2883).

The MATLAB-style workspace ships with two layout presets:

* ``SIDEBAR`` — vertical workspace alongside the editor (default)
* ``MATLAB_HOME`` — full MATLAB-home recreation

The setting is persisted by the launcher's preferences service. This
module only renders the widget and roundtrips the value through the
injected ``get_default`` / ``set_default`` callbacks (LoD: prefs
sections never touch Sidekick internals directly).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["LAYOUT_MODES", "WorkspaceSection"]


LAYOUT_MODES: tuple[tuple[str, str], ...] = (
    ("SIDEBAR", "Sidebar (compact, default)"),
    ("MATLAB_HOME", "MATLAB Home (full layout)"),
)


class WorkspaceSection:
    """Renders the workspace layout-mode picker."""

    SECTION_ID = "workspace"
    SECTION_LABEL = "Workspace"

    def __init__(
        self,
        *,
        get_default: Callable[[], str] | None = None,
        set_default: Callable[[str], None] | None = None,
    ) -> None:
        self._get_default = get_default or (lambda: LAYOUT_MODES[0][0])
        self._set_default = set_default or (lambda _v: None)
        self._combo: Any | None = None

    def build_widget(self) -> Any:
        from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

        from . import build_prefs_section

        widget = QWidget()
        row = QHBoxLayout(widget)
        row.addWidget(QLabel("Default layout mode:"))

        combo = QComboBox()
        for mode_id, label in LAYOUT_MODES:
            combo.addItem(label, mode_id)
        default = self._get_default()
        idx = combo.findData(default)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._on_changed)
        row.addWidget(combo)
        row.addStretch(1)
        self._combo = combo

        return build_prefs_section(self.SECTION_ID, self.SECTION_LABEL, [widget])

    def _on_changed(self, _idx: int) -> None:
        if self._combo is None:
            return
        data = self._combo.currentData()
        if isinstance(data, str):
            self._set_default(data)

    @property
    def selected_mode(self) -> str:
        if self._combo is None:
            return ""
        data = self._combo.currentData()
        return str(data) if isinstance(data, str) else ""
