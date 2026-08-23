"""Shared two-pane splitter assembly for the golf tool GUIs.

Extracted from the swing-flight-pipeline and ball-flight dashboards, whose
``_build_ui`` bodies were byte-identical (DRY gate): give a host widget an
``QHBoxLayout`` holding a horizontal controls|results splitter.
"""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QSplitter, QWidget


def install_two_pane_splitter(
    host: QWidget,
    left: QWidget,
    right: QWidget,
    sizes: Sequence[int] = (350, 650),
) -> QSplitter:
    """Lay ``host`` out as a horizontal ``left | right`` splitter.

    Returns the splitter so callers can keep a reference or tweak it.
    """
    layout = QHBoxLayout(host)
    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setSizes(list(sizes))
    layout.addWidget(splitter)
    return splitter
