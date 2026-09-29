# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Offline help dialogs for the GUI Help menu."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HelpTopic:
    """Single offline help topic shown in the help center."""

    title: str
    body: str


HELP_TOPICS: dict[str, HelpTopic] = {
    "getting_started": HelpTopic(
        "Getting Started",
        """
        <h2>Getting Started</h2>
        <ol>
          <li>Select an exercise tab such as Bottoms Up Squat or Deadlift.</li>
          <li>Set body mass, height, bar load, and movement duration in the sidebar.</li>
          <li>Click <b>Run Optimization</b> or press <b>Ctrl+R</b>.</li>
          <li>Review the animation, joint-angle plots, torque plots, and balance summary.</li>
        </ol>
        <p>If a solve fails, reduce the bar load, increase duration, or reset defaults
        before trying again.</p>
        """,
    ),
    "parameters": HelpTopic(
        "Parameter Guide",
        "Use the parameter table for units, valid ranges, and modeling impact.",
    ),
    "results": HelpTopic(
        "Understanding Results",
        """
        <h2>Understanding Results</h2>
        <p><b>Angles</b> show joint positions over time. <b>Torques</b> show the
        rotational effort required at each joint in N*m. <b>Power</b> combines torque
        and joint velocity. <b>COM</b> plots show whether the center of mass stays
        inside the base of support.</p>
        <p>Lower cost is generally better for the configured objective, but it should
        be interpreted together with balance, peak torques, and movement realism.</p>
        """,
    ),
    "troubleshooting": HelpTopic(
        "Troubleshooting Optimization",
        """
        <h2>Troubleshooting Optimization</h2>
        <p>If the optimizer cannot find a balanced trajectory:</p>
        <ul>
          <li>Increase movement duration for a slower, easier movement.</li>
          <li>Reduce barbell mass.</li>
          <li>Reset segment multipliers to 1.0.</li>
          <li>Check that bar offsets are physically plausible.</li>
          <li>Try a less demanding exercise or range of motion.</li>
        </ul>
        """,
    ),
    "glossary": HelpTopic(
        "Glossary",
        "Common biomechanics and optimization terms used by Movement Optimizer.",
    ),
}


GLOSSARY: dict[str, str] = {
    "BOS": "Base of support: the stable area under the foot.",
    "COM": "Center of mass: the weighted average position of the model mass.",
    "Cost": "Objective value minimized by the optimizer; lower is generally better.",
    "N*m": "Newton-meter, the unit used for joint torque.",
    "ROM": "Range of motion: the angular distance covered by a joint or movement.",
    "Spline": "A smooth curve used to interpolate movement waypoints.",
    "Torque": "Rotational force required at a joint.",
}


class HelpCenterDialog(QDialog):
    """Shows offline help topics, parameter descriptions, and glossary terms.

    Precondition: parent must be a QWidget or None.
    """

    PARAMETERS: ClassVar[dict[str, tuple[str, str, str]]] = {
        "Body Mass": (
            "Total body mass of the lifter",
            "kg",
            "50-150",
        ),
        "Height": (
            "Standing height of the lifter",
            "m",
            "1.40-2.10",
        ),
        "Lower Leg": (
            "Multiplier on the base lower-leg segment length (shank). "
            "Values above 1.0 lengthen the segment; below 1.0 shorten it.",
            "x",
            "0.70-1.30",
        ),
        "Upper Leg": (
            "Multiplier on the base upper-leg segment length (thigh). "
            "Values above 1.0 lengthen the segment; below 1.0 shorten it.",
            "x",
            "0.70-1.30",
        ),
        "Torso": (
            "Multiplier on the base torso segment length. "
            "Values above 1.0 lengthen the segment; below 1.0 shorten it.",
            "x",
            "0.70-1.30",
        ),
        "Total Bar + Plates": (
            "Combined mass of the barbell and any loaded plates. An Olympic bar alone is 20 kg.",
            "kg",
            "0-300",
        ),
        "Bar Back Offset": (
            "Horizontal distance by which the bar is displaced behind the "
            "shoulder contact point. Positive values shift the bar rearward.",
            "m",
            "0.00-0.40",
        ),
        "Bar Drop Offset": (
            "Vertical distance by which the bar sits below the default "
            "shoulder height. Positive values lower the bar.",
            "m",
            "0.00-0.40",
        ),
        "Duration": (
            "Total time allowed for the movement from start to end position. "
            "Shorter durations produce faster, more dynamic trajectories.",
            "s",
            "0.5-5.0",
        ),
        "Smoothness": (
            "Penalty weight on joint-torque rate of change. Higher values "
            "encourage smoother, slower torque transitions at the cost of "
            "slightly higher peak torques.",
            "x",
            "0.1-5.0",
        ),
    }

    def __init__(self, parent: QWidget | None = None, initial_topic: str = "parameters") -> None:
        super().__init__(parent)
        self.setWindowTitle("Movement Optimizer Help")
        self.setMinimumWidth(680)
        self.setMinimumHeight(560)
        self._topic_indexes: dict[str, int] = {}
        self._build_ui()
        self.select_topic(initial_topic)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        header = QLabel("Offline help for setup, parameters, results, troubleshooting, and terms.")
        header.setWordWrap(True)
        outer.addWidget(header)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, stretch=1)

        for topic_id, topic in HELP_TOPICS.items():
            widget: QWidget
            if topic_id == "parameters":
                widget = self._build_parameter_tab()
            elif topic_id == "glossary":
                widget = self._build_glossary_tab()
            else:
                widget = self._build_text_tab(topic.body)
            self._topic_indexes[topic_id] = self.tabs.addTab(widget, topic.title)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        outer.addWidget(buttons)

    def _build_text_tab(self, html: str) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(html)
        return browser

    def _build_parameter_tab(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(6)
        grid.setColumnStretch(1, 1)

        # Column headers
        for col, heading in enumerate(("Parameter", "Description", "Unit", "Range")):
            lbl = QLabel(f"<b>{heading}</b>")
            grid.addWidget(lbl, 0, col)

        for row, (name, (desc, unit, rng)) in enumerate(self.PARAMETERS.items(), start=1):
            name_lbl = QLabel(name)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)

            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)

            unit_lbl = QLabel(unit)
            unit_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

            rng_lbl = QLabel(rng)
            rng_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

            grid.addWidget(name_lbl, row, 0)
            grid.addWidget(desc_lbl, row, 1)
            grid.addWidget(unit_lbl, row, 2)
            grid.addWidget(rng_lbl, row, 3)

        content.setLayout(grid)
        scroll.setWidget(content)
        return scroll

    def _build_glossary_tab(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1)

        for row, (term, definition) in enumerate(GLOSSARY.items()):
            term_label = QLabel(f"<b>{term}</b>")
            term_label.setAlignment(Qt.AlignmentFlag.AlignTop)

            definition_label = QLabel(definition)
            definition_label.setWordWrap(True)
            definition_label.setAlignment(Qt.AlignmentFlag.AlignTop)

            grid.addWidget(term_label, row, 0)
            grid.addWidget(definition_label, row, 1)

        content.setLayout(grid)
        scroll.setWidget(content)
        return scroll

    def select_topic(self, topic_id: str) -> None:
        """Select the requested topic tab, defaulting to the parameter guide."""
        self.tabs.setCurrentIndex(
            self._topic_indexes.get(topic_id, self._topic_indexes["parameters"])
        )


class ParameterHelpDialog(HelpCenterDialog):
    """Backward-compatible parameter-guide entrypoint."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, initial_topic="parameters")
