"""First-run onboarding dialog for the UpstreamDrift Launcher.

Provides a welcome overlay for new users with:
- Quick explanation of what UpstreamDrift is
- How to install engines
- How to select and launch a model
- Links to documentation
- Dismissible with "Don't show again" checkbox

All colors are sourced from QPalette roles — no hard-coded hex values.
Layout uses native PyQt6 widgets (QFrame, QLabel, QHBoxLayout) instead
of an inline-HTML content widget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QDesktopServices, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.theme.typography import Weights, get_display_font, get_qfont

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Config file path for storing onboarding dismissal state
ONBOARDING_CONFIG_PATH = Path.home() / ".upstreamdrift" / "onboarding_config.json"


def is_first_run() -> bool:
    """Check if this is the first run (onboarding not dismissed).

    Returns:
        True when the onboarding config does not exist or has not been dismissed.
    """
    if not ONBOARDING_CONFIG_PATH.exists():
        return True
    try:
        with open(ONBOARDING_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
        return not config.get("onboarding_dismissed", False)
    except (OSError, json.JSONDecodeError):
        return True


def dismiss_onboarding() -> None:
    """Mark onboarding as dismissed (don't show again).

    Postcondition: ONBOARDING_CONFIG_PATH exists and contains
    ``onboarding_dismissed: true``.
    """
    ONBOARDING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {"onboarding_dismissed": True}
    try:
        with open(ONBOARDING_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("Onboarding dismissed by user")
    except OSError as e:
        logger.warning(f"Failed to save onboarding config: {e}")


def _make_info_card(title: str, body: str, link_text: str, link_url: str) -> QFrame:
    """Build a single information card using native widgets.

    Args:
        title: Card heading text.
        body: Card body text.
        link_text: Visible text for the action link label.
        link_url: URL opened when the link label is clicked.

    Returns:
        A QFrame styled as a card using QPalette roles only.
    """
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)

    layout = QVBoxLayout(frame)
    layout.setSpacing(8)
    layout.setContentsMargins(12, 12, 12, 12)

    title_label = QLabel(title)
    title_label.setFont(get_qfont(size=13, weight=Weights.BOLD))
    title_label.setWordWrap(True)
    layout.addWidget(title_label)

    body_label = QLabel(body)
    body_label.setFont(get_qfont(size=11, weight=Weights.NORMAL))
    body_label.setWordWrap(True)
    body_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
    layout.addWidget(body_label)

    link_label = QLabel(f'<a href="{link_url}">{link_text}</a>')
    link_label.setFont(get_qfont(size=11, weight=Weights.NORMAL))
    link_label.setOpenExternalLinks(True)
    link_label.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(link_label)

    return frame


class OnboardingDialog(QDialog):
    """First-run onboarding dialog with welcome information.

    Uses only native PyQt6 widgets and QPalette color roles — no hard-coded
    hex colors, no inline HTML content widget, and no inline HTML for layout.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to UpstreamDrift")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the onboarding dialog UI using native widgets."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_label = QLabel("Welcome to UpstreamDrift")
        header_label.setFont(get_display_font(size=20, weight=Weights.BOLD))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        # Subtitle — uses placeholder-text palette role (theme-aware)
        subtitle_label = QLabel("Biomechanics and Robotics Platform")
        subtitle_label.setFont(get_qfont(size=12, weight=Weights.NORMAL))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        layout.addWidget(subtitle_label)

        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)

        # Card grid — two cards side-by-side
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        quick_start_card = _make_info_card(
            title="Quick Start",
            body="Launch your first physics model using the integrated grid layout.",
            link_text="Read the Guide",
            link_url=(
                "https://github.com/D-sorganization/UpstreamDrift"
                "/blob/main/docs/user_guide/getting_started.md"
            ),
        )
        cards_row.addWidget(quick_start_card)

        config_card = _make_info_card(
            title="Configurations",
            body=(
                "Adjust themes, install required dependencies,"
                " and set up your environment."
            ),
            link_text="Report an Issue",
            link_url="https://github.com/D-sorganization/UpstreamDrift/issues",
        )
        cards_row.addWidget(config_card)

        layout.addLayout(cards_row)

        layout.addStretch()

        # "Don't show again" checkbox
        self.chk_dont_show = QCheckBox("Don't show this welcome message again")
        self.chk_dont_show.setFont(get_qfont(size=10, weight=Weights.NORMAL))
        self.chk_dont_show.setChecked(False)
        layout.addWidget(self.chk_dont_show)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self._on_accepted)

        # Add help button
        help_btn = QPushButton("Open Documentation")
        help_btn.setAutoDefault(False)
        help_btn.clicked.connect(self._open_docs)
        button_box.addButton(help_btn, QDialogButtonBox.ButtonRole.HelpRole)

        layout.addWidget(button_box)

    def _on_accepted(self) -> None:
        """Handle dialog acceptance — optionally persist dismissal."""
        if self.chk_dont_show.isChecked():
            dismiss_onboarding()
        self.accept()

    def _open_docs(self) -> None:
        """Open the documentation in the system browser."""
        docs_url = (
            "https://github.com/D-sorganization/UpstreamDrift"
            "/blob/main/docs/user_guide/getting_started.md"
        )
        QDesktopServices.openUrl(QUrl(docs_url))


def show_onboarding_if_needed(parent: QWidget | None = None) -> bool:
    """Show onboarding dialog if it's the first run.

    Args:
        parent: Optional parent widget.

    Returns:
        True if onboarding was shown, False if skipped.
    """
    if is_first_run():
        dialog = OnboardingDialog(parent)
        dialog.exec()
        return True
    return False
