# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""UI construction helpers for the main window."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .exercise_tab import ExerciseTab
from .widgets import ParameterSidebar, PlaybackControls


def build_central_widget(
    window: QWidget,
    exercise_configs: tuple[tuple[str, str], ...],
) -> tuple[
    QWidget,
    ParameterSidebar,
    QTabWidget,
    list[ExerciseTab],
    PlaybackControls,
    QLabel,
    QPushButton,
    QPushButton,
]:
    """Build the central widget and top-level layout components.

    Returns a tuple of:
        (central, sidebar, tabs, exercise_tabs, controls, status_label,
         left_sidebar_toggle_btn, right_sidebar_toggle_btn)

    The caller is responsible for connecting the sidebar toggle buttons.
    """
    central = QWidget()
    outer = QVBoxLayout(central)
    outer.setContentsMargins(8, 8, 8, 8)

    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(8)
    title = QLabel("Movement Optimizer")
    title.setProperty("class", "title")
    title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    title.setFixedHeight(max(28, title.sizeHint().height()))
    header_layout.addWidget(title, stretch=1)

    left_sidebar_toggle_btn = QPushButton("Hide left")
    left_sidebar_toggle_btn.setToolTip("Collapse/expand left parameter sidebar")
    left_sidebar_toggle_btn.setAccessibleName("Hide left sidebar")
    left_sidebar_toggle_btn.setAccessibleDescription("Hide the left parameter sidebar.")
    left_sidebar_toggle_btn.setMinimumWidth(84)
    left_sidebar_toggle_btn.setMinimumHeight(30)
    header_layout.addWidget(left_sidebar_toggle_btn)

    right_sidebar_toggle_btn = QPushButton("Right panel")
    right_sidebar_toggle_btn.setToolTip("Collapse/expand active right parameter panel")
    right_sidebar_toggle_btn.setAccessibleName("Right panel unavailable")
    right_sidebar_toggle_btn.setAccessibleDescription(
        "No active right parameter panel is available."
    )
    right_sidebar_toggle_btn.setMinimumWidth(92)
    right_sidebar_toggle_btn.setMinimumHeight(30)
    right_sidebar_toggle_btn.setEnabled(False)
    header_layout.addWidget(right_sidebar_toggle_btn)

    outer.addWidget(header)

    # Horizontal area: splitter (sidebar | right panel).
    content_row = QHBoxLayout()
    content_row.setContentsMargins(0, 0, 0, 0)
    content_row.setSpacing(0)
    outer.addLayout(content_row, stretch=1)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    splitter.setMinimumHeight(520)
    content_row.addWidget(splitter, stretch=1)

    sidebar = ParameterSidebar()
    splitter.addWidget(sidebar)

    right = QWidget()
    right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    right_lay = QVBoxLayout(right)
    right_lay.setContentsMargins(0, 0, 0, 0)

    tabs = QTabWidget()
    tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    exercise_tabs: list[ExerciseTab] = []
    for i, (display_name, _) in enumerate(exercise_configs):
        tab = ExerciseTab(display_name)
        tabs.addTab(tab, f"  {display_name}  ")
        tabs.setTabToolTip(i, f"View and optimize {display_name} trajectory")
        exercise_tabs.append(tab)
    right_lay.addWidget(tabs)

    controls = PlaybackControls()
    controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    right_lay.addWidget(controls)

    splitter.addWidget(right)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)

    status_label = QLabel("Ready")
    status_label.setProperty("class", "dim")
    status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    status_label.setFixedHeight(max(18, status_label.sizeHint().height()))
    outer.addWidget(status_label)

    return (
        central,
        sidebar,
        tabs,
        exercise_tabs,
        controls,
        status_label,
        left_sidebar_toggle_btn,
        right_sidebar_toggle_btn,
    )
