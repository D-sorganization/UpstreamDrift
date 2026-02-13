"""Centralized style constants for UpstreamDrift Qt widgets.

This module provides reusable stylesheet constants and helper functions
to eliminate duplicated inline setStyleSheet() calls across the codebase.
All constants produce the same CSS as the original inline styles they replace.

Usage:
    from shared.python.theme.style_constants import Styles

    # Simple constant usage
    widget.setStyleSheet(Styles.STATUS_SUCCESS)
    button.setStyleSheet(Styles.BTN_PRIMARY)

    # Helper methods for parameterized styles
    widget.setStyleSheet(Styles.color_swatch(r, g, b))
    widget.setStyleSheet(Styles.status_chip(bg_color, text_color))
"""

from __future__ import annotations


class Styles:
    """Centralized stylesheet constants grouped by widget type.

    All constants are class-level strings suitable for setStyleSheet() calls.
    Grouped into categories: status labels, buttons, consoles, text, and layout.
    """

    # ══════════════════════════════════════════════════════════════════
    # Status Label Colors
    # ══════════════════════════════════════════════════════════════════

    STATUS_SUCCESS = "color: #30D158;"
    """Green text for success/running/ready states."""

    STATUS_ERROR = "color: #FF375F;"
    """Red text for error/failure states."""

    STATUS_WARNING = "color: #FFD60A;"
    """Yellow text for warning/checking/pending states."""

    STATUS_INACTIVE = "color: #aaaaaa;"
    """Gray text for inactive/ready/idle states."""

    STATUS_INFO = "color: #64b5f6;"
    """Blue text for informational/in-progress states."""

    STATUS_SUCCESS_BOLD = "color: #30D158; font-weight: bold;"
    """Bold green text for prominent success indicators."""

    STATUS_ERROR_BOLD = "color: #FF375F; font-weight: bold;"
    """Bold red text for prominent error indicators."""

    STATUS_INACTIVE_BOLD = "color: #aaaaaa; font-weight: bold;"
    """Bold gray text for status indicators in idle state."""

    # ══════════════════════════════════════════════════════════════════
    # Simple Status Colors (for settings dialogs, key validation, etc.)
    # ══════════════════════════════════════════════════════════════════

    COLOR_GREEN = "color: green;"
    """Simple green text for positive status."""

    COLOR_RED = "color: red;"
    """Simple red text for negative status."""

    COLOR_ORANGE = "color: orange;"
    """Simple orange text for partial/warning status."""

    COLOR_GRAY = "color: gray;"
    """Simple gray text for neutral status."""

    COLOR_RESET = ""
    """Empty stylesheet to reset styling."""

    # ══════════════════════════════════════════════════════════════════
    # Button Styles
    # ══════════════════════════════════════════════════════════════════

    BTN_RUN = "QPushButton { background-color: #4CAF50; color: white; }"
    """Green run/start button (matches simulation_gui_base STYLE_BUTTON_RUN)."""

    BTN_STOP = "QPushButton { background-color: #f44336; color: white; }"
    """Red stop button (matches simulation_gui_base STYLE_BUTTON_STOP)."""

    BTN_PRIMARY = (
        "QPushButton {"
        "  background-color: #0A84FF; color: white; border: none;"
        "  padding: 5px 10px; border-radius: 4px; font-weight: bold;"
        "}"
        "QPushButton:hover { background-color: #0077E6; }"
    )
    """Primary blue action button (Help, main actions)."""

    BTN_SECONDARY = (
        "QPushButton {"
        "  background-color: #484f58; color: white; border: none;"
        "  padding: 5px 10px; border-radius: 4px;"
        "}"
        "QPushButton:hover { background-color: #6e7681; }"
    )
    """Secondary gray button (Settings, auxiliary actions)."""

    BTN_AI_CHAT = (
        "QPushButton {"
        "  background-color: #1976d2; color: white;"
        "  padding: 8px 16px; font-weight: bold; border-radius: 4px;"
        "}"
        "QPushButton:hover { background-color: #1565c0; }"
        "QPushButton:checked { background-color: #0d47a1; }"
    )
    """AI assistant toggle button."""

    BTN_LAUNCH_READY = (
        "QPushButton {"
        "  background-color: #2da44e; color: white;"
        "  border-radius: 6px; padding: 0 40px;"
        "}"
        "QPushButton:disabled {"
        "  background-color: #444444; color: #888888;"
        "}"
        "QPushButton:hover:!disabled {"
        "  background-color: #2c974b;"
        "}"
    )
    """Launch button in ready/select state."""

    BTN_SAVE = "background-color: #107c10; color: white; padding: 8px;"
    """Green save button for config/settings panels."""

    BTN_DOCKER_RUN = (
        "background-color: #107c10; color: white;padding: 10px; font-weight: bold;"
    )
    """Docker simulation run button."""

    BTN_DOCKER_STOP = (
        "background-color: #d13438; color: white;padding: 10px; font-weight: bold;"
    )
    """Docker simulation stop button."""

    BTN_DOCKER_REBUILD = (
        "background-color: #8b5cf6; color: white;padding: 10px; font-weight: bold;"
    )
    """Docker environment rebuild button."""

    BTN_GENERATE_PLOT = (
        "QPushButton {"
        "  background-color: #2ca02c; color: white;"
        "  font-weight: bold; padding: 8px;"
        "}"
        "QPushButton:hover { background-color: #238c23; }"
    )
    """Green generate/plot button."""

    BTN_ADVANCED_ANALYSIS = (
        "QPushButton {"
        "  background-color: #9467bd; color: white;"
        "  font-weight: bold; padding: 8px;"
        "}"
        "QPushButton:hover { background-color: #8c564b; }"
    )
    """Purple advanced analysis/dialog button."""

    BTN_DANGER = (
        "QPushButton {"
        "  background-color: #d62728;"
        "}"
        "QPushButton:hover {"
        "  background-color: #a81f20;"
        "}"
    )
    """Red danger/destructive action button (clear, delete)."""

    BTN_LAYOUT_EDIT_ON = (
        "QPushButton {"
        "  background-color: #007acc; color: white;"
        "  border: 1px solid #0099ff;"
        "}"
    )
    """Layout edit mode active button."""

    BTN_LAYOUT_LOCKED = "QPushButton {  background-color: #444444; color: #cccccc;}"
    """Layout locked/inactive button."""

    BTN_LAYOUT_TOGGLE = (
        "QPushButton { background: #444; color: #ccc; padding: 8px 16px; }"
        "QPushButton:checked { background: #007acc; color: white; }"
    )
    """Layout lock toggle button (checkable)."""

    BTN_RECORD_CHECKED = (
        "QPushButton:checked { background-color: #d62728; color: white; "
        "font-weight: bold; }"
    )
    """Record button checked state (recording active)."""

    BTN_RECORD_CHECKED_LIGHT = "QPushButton:checked { background-color: #ffcccc; }"
    """Record button checked state (lighter variant)."""

    BTN_ACTION_BLUE = "background-color: #0078d4; font-weight: bold;"
    """Blue action button (generate, apply)."""

    BTN_ACTION_GREEN = "background-color: #107c10; font-weight: bold;"
    """Green action button (apply, confirm)."""

    BTN_SEND = (
        "QPushButton {"
        "  background-color: #0A84FF; color: white; border: none;"
        "  padding: 6px 16px; border-radius: 4px; font-weight: bold;"
        "}"
        "QPushButton:hover { background-color: #409CFF; }"
        "QPushButton:disabled {"
        "  background-color: #333; color: #666;"
        "}"
    )
    """Send message button for chat interfaces."""

    BTN_CLOSE_ROUND = (
        "QPushButton {"
        "  background-color: transparent; border: 1px solid #555;"
        "  border-radius: 12px; color: #aaa; font-size: 16px;"
        "  padding: 0px; min-width: 24px; min-height: 24px;"
        "}"
        "QPushButton:hover { background-color: #ff4444; color: white; }"
    )
    """Round close button for panels and overlays."""

    # ══════════════════════════════════════════════════════════════════
    # Console / Monospace Text Styles
    # ══════════════════════════════════════════════════════════════════

    CONSOLE_DARK = (
        "QPlainTextEdit {"
        "  background-color: #1e1e1e; color: #d4d4d4;"
        "  font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;"
        "  font-size: 11px; border: none;"
        "}"
    )
    """Dark console/terminal output widget."""

    CONSOLE_BUILD = (
        "background-color: #1e1e1e; color: #00ff00;"
        "font-family: 'Cascadia Code', Consolas, monospace; font-size: 11px;"
    )
    """Build console with green text on dark background."""

    CONSOLE_DIAGNOSTICS = (
        "QTextBrowser {"
        "  background-color: #1e1e1e; color: #d4d4d4;"
        "  font-family: 'Segoe UI', sans-serif; font-size: 13px;"
        "  padding: 12px;"
        "}"
    )
    """Diagnostics browser with dark theme."""

    CONSOLE_LOG_GREEN = (
        "QTextEdit {"
        "  background-color: #0d0d0d; color: #00ff00;"
        "  font-family: 'Cascadia Code', Consolas, monospace;"
        "  font-size: 11px;"
        "}"
    )
    """Process log viewer with green text."""

    CONSOLE_LOG_LIGHT = (
        "QTextEdit {"
        "  background-color: #0d0d0d; color: #d4d4d4;"
        "  font-family: 'Cascadia Code', Consolas, monospace;"
        "  font-size: 11px;"
        "}"
    )
    """Application log viewer with light text."""

    CONSOLE_MONOSPACE = "font-family: Consolas, monospace; font-size: 10pt;"
    """Simple monospace font for log/code text."""

    CONSOLE_MONOSPACE_SMALL = "font-family: monospace;"
    """Minimal monospace font style."""

    # ══════════════════════════════════════════════════════════════════
    # Text / Label Styles
    # ══════════════════════════════════════════════════════════════════

    TEXT_BOLD = "font-weight: bold;"
    """Bold text."""

    TEXT_BOLD_PADDING = "font-weight: bold; padding: 5px;"
    """Bold text with padding (recording labels, status headers)."""

    TEXT_MUTED = "color: #888888; font-size: 11px;"
    """Muted gray small text (timestamps, info labels)."""

    TEXT_MUTED_SMALL = "color: #888; font-size: 10px;"
    """Slightly smaller muted text."""

    TEXT_SECONDARY = "color: #666; font-size: 11px;"
    """Secondary text for status/descriptions."""

    TEXT_TERTIARY = "color: #7f8c8d;"
    """Tertiary muted text."""

    TEXT_SUBTITLE = "color: #666; font-size: 9pt; margin: 5px;"
    """Subtitle text with margin spacing."""

    TEXT_ITALIC_NOTE = "font-style: italic; font-size: 9pt;"
    """Italic note text (instructions, hints)."""

    TEXT_ITALIC_SMALL = "font-style: italic; font-size: 10px;"
    """Small italic text for interpolation/info notes."""

    TEXT_LABEL_BOLD_WHITE = "font-weight: bold; color: #e0e0e0;"
    """Bold white/light label (role labels in chat)."""

    TEXT_CONTENT_TRANSPARENT = "background-color: transparent; color: #e0e0e0;"
    """Transparent background with light text (chat content areas)."""

    TEXT_SUBTITLE_LAUNCHER = "color: #666; font-size: 13px;"
    """Subtitle label for launcher headers."""

    TEXT_SUCCESS_PADDED = "color: #2ecc71; padding: 0 10px;"
    """Green success text with horizontal padding."""

    TEXT_STATUS_LABEL_MUTED = "color: #666666;"
    """Muted label text for status fields."""

    # ══════════════════════════════════════════════════════════════════
    # Recording Status Indicators
    # ══════════════════════════════════════════════════════════════════

    RECORDING_ACTIVE = (
        "background-color: #d62728; color: white; font-weight: bold; padding: 5px;"
    )
    """Recording actively in progress."""

    RECORDING_STOPPED = (
        "background-color: #ff7f0e; color: white; font-weight: bold; padding: 5px;"
    )
    """Recording stopped with data available."""

    RECORDING_IDLE = "font-weight: bold; padding: 5px;"
    """No recording data (idle state)."""

    # ══════════════════════════════════════════════════════════════════
    # Layout / Container Styles
    # ══════════════════════════════════════════════════════════════════

    TRANSPARENT_BG = "background: transparent;"
    """Transparent background for containers."""

    SCROLL_AREA_TRANSPARENT = "QScrollArea { background: transparent; }"
    """Transparent scroll area background."""

    SPLITTER_HANDLE = "QSplitter::handle { background-color: #484f58; }"
    """Dark splitter handle."""

    LABEL_TRANSPARENT = "QLabel { border: none; background: transparent; }"
    """Transparent label with no border (image containers)."""

    LABEL_TITLE_TRANSPARENT = "border: none; background: transparent;"
    """Transparent label style for card titles."""

    NOTICE_WARNING = "background-color: #fff3cd; padding: 6px;"
    """Yellow warning notice background."""

    INSTRUCTIONS_BOX = "padding: 10px; background-color: #e8f4f8; border-radius: 5px;"
    """Light blue instructions/help box."""

    CONTAINER_DARK = "background-color: #1e1e1e;"
    """Dark container background (message areas, panels)."""

    # ══════════════════════════════════════════════════════════════════
    # Chat / AI Panel Styles
    # ══════════════════════════════════════════════════════════════════

    CHAT_MESSAGE_USER = (
        "MessageWidget {"
        "  background-color: #2d2d2d; border: 1px solid #3c3c3c;"
        "  border-radius: 8px; margin-left: 40px;"
        "}"
        "QLabel { color: #FF8800; }"
    )
    """User message bubble in chat."""

    CHAT_MESSAGE_ASSISTANT = (
        "MessageWidget {"
        "  background-color: #252526; border: 1px solid #333333;"
        "  border-radius: 8px; margin-right: 40px;"
        "}"
        "QLabel { color: #e0e0e0; }"
    )
    """Assistant message bubble in chat."""

    CHAT_MESSAGE_SYSTEM = (
        "MessageWidget {"
        "  background-color: #332b00; border: 1px solid #665500;"
        "  border-radius: 8px;"
        "}"
        "QLabel { color: #FF8800; }"
    )
    """System message bubble in chat."""

    CHAT_SCROLL_AREA = "QScrollArea {  background-color: #1e1e1e; border: none;}"
    """Chat scroll area with dark background."""

    CHAT_INPUT = (
        "QTextEdit {"
        "  background-color: #2d2d2d; color: #e0e0e0;"
        "  border: 1px solid #404040; border-radius: 8px;"
        "  padding: 8px; font-size: 13px;"
        "}"
        "QTextEdit:focus { border-color: #0A84FF; }"
    )
    """Chat input text field."""

    CHAT_HEADER = (
        "QWidget {  background-color: #252526;  border-bottom: 1px solid #333333;}"
    )
    """Chat panel header bar."""

    CHAT_INPUT_CONTAINER = (
        "QWidget {  background-color: #252526;  border-top: 1px solid #333333;}"
    )
    """Chat input container bar."""

    CHAT_COMBO = (
        "QComboBox {"
        "  background-color: #333; color: #ccc;"
        "  border: 1px solid #555; border-radius: 4px;"
        "  padding: 2px 8px; font-size: 11px;"
        "}"
    )
    """Styled combo box for chat mode selection."""

    CHAT_SPLITTER = "QSplitter::handle {  background-color: #333333;  height: 2px;}"
    """Chat panel splitter handle."""

    # ══════════════════════════════════════════════════════════════════
    # Execution Mode Styles
    # ══════════════════════════════════════════════════════════════════

    EXEC_MODE_WARNING = "color: #FFD60A; font-weight: bold; margin-left: 10px;"
    """Warning-colored execution mode label (local/unknown)."""

    EXEC_MODE_DOCKER = "color: #30D158; font-weight: bold; margin-left: 10px;"
    """Green execution mode label (Docker)."""

    EXEC_MODE_WSL = "color: #0A84FF; font-weight: bold; margin-left: 10px;"
    """Blue execution mode label (WSL)."""

    # ══════════════════════════════════════════════════════════════════
    # Overlay Styles
    # ══════════════════════════════════════════════════════════════════

    OVERLAY_STATUS = (
        "color: white; font-weight: bold; font-size: 14pt; "
        "background-color: rgba(0, 0, 0, 150); "
        "padding: 5px 10px; border-radius: 5px;"
    )
    """Overlay status label with semi-transparent background."""

    OVERLAY_CLOSE_BTN = (
        "QPushButton { "
        "  color: white; background-color: rgba(255, 0, 0, 180); "
        "  border-radius: 15px; font-weight: bold; font-size: 16px;"
        "}"
        "QPushButton:hover { background-color: rgba(255, 50, 50, 255); }"
    )
    """Round red close button for overlays."""

    OVERLAY_REC_BTN = (
        "QPushButton { "
        "  color: white; background-color: rgba(200, 0, 0, 150); "
        "  border: 1px solid white; border-radius: 4px; padding: 5px 15px;"
        "}"
        "QPushButton:checked { background-color: red; }"
    )
    """Overlay record button with checked state."""

    OVERLAY_PAUSE_BTN = (
        "QPushButton { "
        "  color: white; background-color: rgba(0, 0, 0, 150); "
        "  border: 1px solid white; border-radius: 4px; padding: 5px 15px;"
        "}"
    )
    """Overlay pause button."""

    # ══════════════════════════════════════════════════════════════════
    # Pose Estimation Title Styles
    # ══════════════════════════════════════════════════════════════════

    TITLE_BANNER = "background-color: #0078d4; font-weight: bold;"
    """Blue title banner for pose estimation panels."""

    # ══════════════════════════════════════════════════════════════════
    # Chat Dock Widget Styles
    # ══════════════════════════════════════════════════════════════════

    CHAT_DOCK_PANEL = (
        "QWidget {"
        "  background-color: #1e1e1e; color: #e0e0e0;"
        "  border: 1px solid #333333;"
        "}"
    )
    """Chat dock widget panel background."""

    CHAT_DOCK_SCROLL = "QScrollArea { background-color: #1e1e1e; border: none; }"
    """Chat dock scroll area."""

    CHAT_DOCK_ROLE_USER = "font-weight: bold; color: #FF8800; font-size: 12px;"
    """User role label in chat dock."""

    CHAT_DOCK_ROLE_ASSISTANT = "font-weight: bold; color: #58a6ff; font-size: 12px;"
    """Assistant role label in chat dock."""

    CHAT_DOCK_CONTENT = "color: #e0e0e0; font-size: 12px;"
    """Message content text in chat dock."""

    CHAT_DOCK_STATUS = "color: #888; font-size: 10px;"
    """Status label in chat dock footer."""

    CHAT_DOCK_STATUS_CONNECTED = "color: #3fb950; font-size: 10px;"
    """Connected status in chat dock."""

    CHAT_DOCK_STATUS_ERROR = "color: #f85149; font-size: 10px;"
    """Error status in chat dock."""

    CHAT_DOCK_INPUT = (
        "QTextEdit {"
        "  background-color: #2d2d2d; color: #e0e0e0;"
        "  border: 1px solid #404040; border-radius: 6px; padding: 6px;"
        "}"
    )
    """Chat dock input field."""

    CHAT_DOCK_SEND_BTN = (
        "QPushButton {"
        "  background-color: #0A84FF; color: white; border: none;"
        "  border-radius: 4px; font-weight: bold;"
        "}"
        "QPushButton:hover { background-color: #409CFF; }"
    )
    """Chat dock send button."""

    CHAT_DOCK_MESSAGE_USER = (
        "QFrame {"
        "  background-color: #2d2d2d; border: 1px solid #3c3c3c;"
        "  border-radius: 8px; margin: 4px 4px 4px 40px;"
        "}"
    )
    """Chat dock user message frame."""

    CHAT_DOCK_MESSAGE_ASSISTANT = (
        "QFrame {"
        "  background-color: #252526; border: 1px solid #333;"
        "  border-radius: 8px; margin: 4px 40px 4px 4px;"
        "}"
    )
    """Chat dock assistant message frame."""

    # ══════════════════════════════════════════════════════════════════
    # Status Bar Styles (MuJoCo Main Window)
    # ══════════════════════════════════════════════════════════════════

    STATUSBAR_DARK = (
        "QStatusBar {"
        "  background-color: #2c3e50; color: white; font-weight: bold;"
        "}"
        "QStatusBar::item { border: none; }"
    )
    """Dark status bar with no item borders."""

    STATUSBAR_MODEL = "color: #3498db; padding: 0 10px;"
    """Blue model info label in status bar."""

    STATUSBAR_TIME = TEXT_SUCCESS_PADDED
    """Green time label in status bar (same as TEXT_SUCCESS_PADDED)."""

    STATUSBAR_CAMERA = "color: #9b59b6; padding: 0 10px;"
    """Purple camera info label in status bar."""

    STATUSBAR_SEPARATOR = "color: #7f8c8d;"
    """Gray separator in status bar."""

    STATUSBAR_STATE_RUNNING = TEXT_SUCCESS_PADDED
    """Green state label when simulation is running (same as TEXT_SUCCESS_PADDED)."""

    STATUSBAR_STATE_PAUSED = "color: #f39c12; padding: 0 10px;"
    """Orange state label when simulation is paused."""

    STATUSBAR_RECORDING = "color: #e74c3c; padding: 0 10px;"
    """Red recording status label."""

    STATUSBAR_RECORDING_ACTIVE = "color: #e74c3c; font-weight: bold; padding: 0 10px;"
    """Bold red recording status label (active recording)."""

    STATUSBAR_RECORDING_DONE = "color: #f39c12; padding: 0 10px;"
    """Orange recording status label (recording completed)."""

    # ══════════════════════════════════════════════════════════════════
    # Pose Estimation GUI Styles
    # ══════════════════════════════════════════════════════════════════

    POSE_EST_TITLE = "font-size: 18px; font-weight: bold; margin-bottom: 10px;"
    """Title banner for pose estimation GUIs."""

    BTN_RUN_OPENPOSE = (
        "background-color: #28a745; color: white; padding: 10px; font-weight: bold;"
    )
    """Green run button for OpenPose analysis."""

    BTN_RUN_MEDIAPIPE = (
        "background-color: #ff9900; color: white; padding: 10px; font-weight: bold;"
    )
    """Orange run button for MediaPipe analysis."""

    # ══════════════════════════════════════════════════════════════════
    # Shortcuts Overlay Fallback Styles (no-theme mode)
    # ══════════════════════════════════════════════════════════════════

    SHORTCUT_KEY_BADGE = (
        "background-color: #242424; color: #FFFFFF; "
        "border: 1px solid #404040; border-radius: 4px; padding: 2px 6px;"
    )
    """Key badge in shortcuts overlay (fallback)."""

    SHORTCUT_PLUS = "color: #A0A0A0;"
    """Plus separator in shortcuts overlay (fallback)."""

    SHORTCUT_CONTENT = (
        "QFrame#shortcutsContent {"
        "  background-color: #1A1A1A;"
        "  border: 1px solid #404040;"
        "  border-radius: 12px;"
        "}"
    )
    """Shortcuts overlay content panel (fallback)."""

    SHORTCUT_TITLE = "color: #FFFFFF;"
    """Shortcuts overlay title (fallback)."""

    SHORTCUT_CLOSE_BTN = (
        "QPushButton {"
        "  background-color: transparent; color: #A0A0A0;"
        "  border: none; border-radius: 14px; font-size: 14px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2D2D2D; color: #FFFFFF;"
        "}"
    )
    """Shortcuts overlay close button (fallback)."""

    SHORTCUT_HINT = "color: #666666;"
    """Shortcuts overlay footer hint (fallback)."""

    SHORTCUT_CATEGORY_HEADER = "color: #0A84FF;"
    """Shortcuts overlay category header (fallback)."""

    SHORTCUT_DESC = "color: #E0E0E0;"
    """Shortcuts overlay description text (fallback)."""

    # ══════════════════════════════════════════════════════════════════
    # Launcher Header / GroupBox Styles
    # ══════════════════════════════════════════════════════════════════

    HEADER_TITLE_LARGE = "font-size: 24px; font-weight: bold; color: #ffffff;"
    """Large white header title."""

    HEADER_SUBTITLE = "font-size: 14px; color: #cccccc;"
    """Gray subtitle below header title."""

    GROUPBOX_DARK = (
        "QGroupBox { font-weight: bold; border: 1px solid #555; "
        "margin-top: 10px; }\n"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
        "padding: 0 5px; }"
    )
    """Dark-themed group box with subtle border."""

    BTN_POLY_GENERATOR = (
        "background-color: #0078d4; color: white; padding: 8px; font-weight: bold;"
    )
    """Blue polynomial generator tool button."""

    BTN_SIGNAL_TOOLKIT = (
        "background-color: #6b5b95; color: white; padding: 8px; font-weight: bold;"
    )
    """Purple signal toolkit tool button."""

    TEXT_HELP_HINT = "color: #aaa; font-style: italic; font-size: 11px;"
    """Muted italic hint text for control mode help."""

    CONSOLE_LOG_DARK = (
        "background-color: #1e1e1e; color: #ddd;font-family: Consolas; font-size: 10pt;"
    )
    """Dark console log with Consolas font."""

    # ══════════════════════════════════════════════════════════════════
    # Color Swatch Defaults (Visualization Tab)
    # ══════════════════════════════════════════════════════════════════

    SWATCH_SKY_DEFAULT = "background-color: rgb(51, 77, 102);"
    """Default sky color swatch button."""

    SWATCH_GROUND_DEFAULT = "background-color: rgb(51, 51, 51);"
    """Default ground color swatch button."""

    # ══════════════════════════════════════════════════════════════════
    # Pose Editor Styles
    # ══════════════════════════════════════════════════════════════════

    POSE_STATUS_SUCCESS = "font-weight: bold; color: #2e7d32;"
    """Pose editor success status."""

    POSE_STATUS_ERROR = "font-weight: bold; color: #c62828;"
    """Pose editor error status."""

    POSE_INFO_MUTED = "color: #666; font-size: 10px;"
    """Pose editor info text."""

    POSE_HEADER_MARGIN = "margin-top: 8px;"
    """Pose editor section header margin."""

    # ══════════════════════════════════════════════════════════════════
    # Helper Methods for Parameterized Styles
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def color_swatch(r: int, g: int, b: int) -> str:
        """Generate a color swatch button style.

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)

        Returns:
            Stylesheet string for a color swatch button.
        """
        return f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"

    @staticmethod
    def status_chip(bg_color: str, text_color: str) -> str:
        """Generate a status chip label style.

        Args:
            bg_color: Background color (hex string)
            text_color: Text color (hex string)

        Returns:
            Stylesheet string for a status chip label.
        """
        return (
            f"background-color: {bg_color}; color: {text_color}; "
            f"padding: 2px 6px; border-radius: 4px;"
        )

    @staticmethod
    def colored_bold(color: str) -> str:
        """Generate bold colored text style.

        Args:
            color: Text color (hex string)

        Returns:
            Stylesheet string for bold colored text.
        """
        return f"color: {color}; font-weight: bold;"

    @staticmethod
    def no_image_label(color: str) -> str:
        """Generate a no-image fallback label style.

        Args:
            color: Text color for the 'No Image' label.

        Returns:
            Stylesheet for a transparent italic label.
        """
        return (
            f"QLabel {{ color: {color}; font-style: italic; "
            f"border: none; background: transparent; }}"
        )


__all__ = [
    "Styles",
]
