"""Preferences subpage: Jupyter — notebook directory + kernel selection.

Tools PR #2889 ships the Phase 1 embedded Jupyter widget. This subpage
captures two user preferences:

* Default notebook directory — where new notebooks are written.
* Default kernel name — pre-selected when a new notebook opens.

When ``nbformat`` is not installed (the canonical signal that the
Jupyter feature is unavailable) the widget renders a single
information label explaining the install path, instead of disabled
inputs. This keeps the prefs dialog informative rather than confusing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.launchers.feature_menu import is_feature_available
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["JupyterSection"]


class JupyterSection:
    """Jupyter prefs subpage container."""

    SECTION_ID = "jupyter"
    SECTION_LABEL = "Jupyter"

    def __init__(
        self,
        *,
        get_notebook_dir: Callable[[], str] | None = None,
        set_notebook_dir: Callable[[str], None] | None = None,
        get_kernel: Callable[[], str] | None = None,
        set_kernel: Callable[[str], None] | None = None,
    ) -> None:
        self._get_dir = get_notebook_dir or (lambda: str(Path.home() / "notebooks"))
        self._set_dir = set_notebook_dir or (lambda _v: None)
        self._get_kernel = get_kernel or (lambda: "python3")
        self._set_kernel = set_kernel or (lambda _v: None)
        self._dir_edit: Any | None = None
        self._kernel_edit: Any | None = None
        self._available = is_feature_available("jupyter")

    @property
    def is_available(self) -> bool:
        return self._available

    def build_widget(self) -> Any:
        from PyQt6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QVBoxLayout,
            QWidget,
        )

        from . import build_prefs_section

        widget = QWidget()
        outer = QVBoxLayout(widget)

        if not self._available:
            outer.addWidget(
                QLabel(
                    "Jupyter integration is unavailable — install the optional "
                    "<code>nbformat</code> package to enable notebook tabs."
                )
            )
            return build_prefs_section(self.SECTION_ID, self.SECTION_LABEL, [widget])

        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Default notebook directory:"))
        dir_edit = QLineEdit(self._get_dir())
        dir_edit.editingFinished.connect(lambda: self._set_dir(dir_edit.text().strip()))
        row_dir.addWidget(dir_edit)
        self._dir_edit = dir_edit
        wrapper_dir = QWidget()
        wrapper_dir.setLayout(row_dir)
        outer.addWidget(wrapper_dir)

        row_kernel = QHBoxLayout()
        row_kernel.addWidget(QLabel("Default kernel name:"))
        kernel_edit = QLineEdit(self._get_kernel())
        kernel_edit.editingFinished.connect(
            lambda: self._set_kernel(kernel_edit.text().strip())
        )
        row_kernel.addWidget(kernel_edit)
        self._kernel_edit = kernel_edit
        wrapper_kernel = QWidget()
        wrapper_kernel.setLayout(row_kernel)
        outer.addWidget(wrapper_kernel)

        return build_prefs_section(self.SECTION_ID, self.SECTION_LABEL, [widget])
