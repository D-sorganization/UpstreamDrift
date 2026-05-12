"""First-run onboarding dialog for the UpstreamDrift Launcher.

Provides a welcome overlay for new users with:
- Quick explanation of what UpstreamDrift is
- How to install engines
- How to select and launch a model
- Links to documentation
- Dismissible with "Don't show again" checkbox
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QPoint, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QTextBrowser,
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


def _get_theme_colors():
    """Get current theme colors, with fallback to dark theme defaults."""
    try:
        from src.shared.python.theme import DARK_THEME, get_theme_manager

        manager = get_theme_manager()
        return manager.get_current_colors() if manager else DARK_THEME
    except ImportError:
        from src.shared.python.theme import DARK_THEME

        return DARK_THEME


def is_first_run() -> bool:
    """Check if this is the first run (onboarding not dismissed)."""
    if not ONBOARDING_CONFIG_PATH.exists():
        return True
    try:
        with open(ONBOARDING_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
        return not config.get("onboarding_dismissed", False)
    except (OSError, json.JSONDecodeError):
        return True


def dismiss_onboarding() -> None:
    """Mark onboarding as dismissed (don't show again)."""
    ONBOARDING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {"onboarding_dismissed": True}
    try:
        with open(ONBOARDING_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("Onboarding dismissed by user")
    except OSError as e:
        logger.warning(f"Failed to save onboarding config: {e}")


class OnboardingDialog(QDialog):
    """First-run onboarding dialog with welcome information."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to UpstreamDrift!")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the onboarding dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_label = QLabel("Welcome to UpstreamDrift")
        header_label.setFont(get_display_font(size=20, weight=Weights.BOLD))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        # Subtitle
        subtitle_label = QLabel("Biomechanics & Robotics Platform")
        subtitle_label.setFont(get_qfont(size=12, weight=Weights.NORMAL))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        layout.addWidget(subtitle_label)

        # Content browser with welcome information
        content = QTextBrowser()
        content.setReadOnly(True)
        content.setOpenExternalLinks(True)
        content.setHtml(self._get_welcome_html())
        content.setStyleSheet("background-color: transparent; border: none;")
        content.setMinimumHeight(350)
        layout.addWidget(content)

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

    def _get_welcome_html(self) -> str:
        """Generate the welcome HTML content."""
        return """
        <style>
            body { 
                font-family: 'Inter', 'Segoe UI', sans-serif; 
                line-height: 1.5;
                color: #e0e0e0;
                margin: 0;
            }
            .hero {
                text-align: center;
                padding: 10px 0 20px 0;
                border-bottom: 1px solid #333;
                margin-bottom: 20px;
            }
            .hero h2 {
                background: linear-gradient(90deg, #FF8800, #FF5500);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0 0 10px 0;
                font-size: 28px;
            }
            .grid {
                display: flex;
                gap: 15px;
                margin-bottom: 20px;
            }
            .card {
                background: #252526;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                padding: 15px;
                flex: 1;
            }
            .card h3 {
                color: #FF8800;
                margin: 0 0 10px 0;
                font-size: 16px;
            }
            .card p {
                margin: 0 0 10px 0;
                font-size: 13px;
                color: #aaa;
            }
            a.btn {
                display: inline-block;
                background: #FF8800;
                color: #1e1e1e;
                text-decoration: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            a.btn:hover { background: #FF9933; }
            a.btn-outline {
                background: transparent;
                color: #FF8800;
                border: 1px solid #FF8800;
            }
            a.btn-outline:hover { background: rgba(255,136,0,0.1); }
        </style>
        
        <div class="hero">
            <h2>UpstreamDrift</h2>
            <p style="color: #aaa; margin:0; font-size: 14px;">Next-Generation Biomechanics & Robotics Platform</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>🚀 Quick Start</h3>
                <p>Launch your first physics model using the integrated grid layout.</p>
                <a class="btn" href="https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/user_guide/getting_started.md">Read the Guide</a>
            </div>
            <div class="card">
                <h3>⚙️ Configurations</h3>
                <p>Adjust themes, install required dependencies, and set up your environment.</p>
                <a class="btn btn-outline" href="https://github.com/D-sorganization/UpstreamDrift/issues">Report an Issue</a>
            </div>
        </div>
        """

    def _on_accepted(self) -> None:
        """Handle dialog acceptance."""
        if self.chk_dont_show.isChecked():
            dismiss_onboarding()
        self.accept()

    def _open_docs(self) -> None:
        """Open the documentation in the system browser."""
        docs_url = "https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/user_guide/getting_started.md"
        QDesktopServices.openUrl(QUrl(docs_url))


def show_onboarding_if_needed(parent: QWidget | None = None) -> bool:
    """Show onboarding dialog if it's the first run.

    Args:
        parent: Optional parent widget

    Returns:
        True if onboarding was shown, False if skipped
    """
    if is_first_run():
        dialog = OnboardingDialog(parent)
        dialog.exec()
        return True
    return False
