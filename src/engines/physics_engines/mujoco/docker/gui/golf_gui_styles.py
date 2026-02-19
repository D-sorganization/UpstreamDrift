"""
Style configuration mixin for the Golf Simulation GUI.

Extracted from GolfSimulationGUI to respect SRP:
visual styling is independent of simulation logic and tab construction.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from tkinter import ttk
except ImportError:
    pass


class StyleMixin:
    """Style configuration for the Golf Simulation GUI.

    Requires host class: no dependencies beyond tkinter ttk.
    """

    def setup_styles(self) -> None:
        """Configure modern styling for the application."""
        style = ttk.Style()
        style.theme_use("clam")

        colors = self._get_color_scheme()
        self._configure_notebook_styles(style, colors)
        self._configure_frame_styles(style, colors)
        self._configure_label_styles(style, colors)
        self._configure_button_styles(style)
        self._configure_widget_styles(style, colors)

    @staticmethod
    def _get_color_scheme() -> dict[str, str]:
        return {
            "bg": "#2b2b2b",
            "fg": "#ffffff",
            "select_bg": "#404040",
            "select_fg": "#ffffff",
            "accent": "#0078d4",
            "success": "#107c10",
            "warning": "#ff8c00",
            "error": "#d13438",
            "purple": "#8b5cf6",
        }

    @staticmethod
    def _configure_notebook_styles(style: ttk.Style, colors: dict[str, str]) -> None:
        style.configure("Modern.TNotebook", background=colors["bg"], borderwidth=0)
        style.configure(
            "Modern.TNotebook.Tab",
            background=colors["select_bg"],
            foreground=colors["fg"],
            padding=[20, 12],
            font=("Segoe UI", 10, "bold"),
            focuscolor="none",
        )
        style.map(
            "Modern.TNotebook.Tab",
            background=[
                ("selected", colors["accent"]),
                ("active", colors["select_bg"]),
                ("!active", colors["select_bg"]),
            ],
            foreground=[
                ("selected", "#ffffff"),
                ("active", colors["fg"]),
                ("!active", colors["fg"]),
            ],
            padding=[
                ("selected", [20, 12]),
                ("active", [20, 12]),
                ("!active", [20, 12]),
            ],
        )

    @staticmethod
    def _configure_frame_styles(style: ttk.Style, colors: dict[str, str]) -> None:
        style.configure("Modern.TFrame", background=colors["bg"])
        style.configure(
            "Card.TFrame", background=colors["select_bg"], relief="flat", borderwidth=1
        )

    @staticmethod
    def _configure_label_styles(style: ttk.Style, colors: dict[str, str]) -> None:
        style.configure(
            "Modern.TLabel",
            background=colors["bg"],
            foreground=colors["fg"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Title.TLabel",
            background=colors["bg"],
            foreground=colors["fg"],
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Heading.TLabel",
            background=colors["bg"],
            foreground=colors["accent"],
            font=("Segoe UI", 12, "bold"),
        )

    @staticmethod
    def _configure_button_styles(style: ttk.Style) -> None:
        style.configure("Modern.TButton", font=("Segoe UI", 10), padding=[15, 8])
        style.configure(
            "Primary.TButton", font=("Segoe UI", 11, "bold"), padding=[20, 10]
        )
        style.configure(
            "Success.TButton", font=("Segoe UI", 11, "bold"), padding=[20, 10]
        )
        style.configure("Warning.TButton", font=("Segoe UI", 10), padding=[15, 8])
        style.configure(
            "Danger.TButton", font=("Segoe UI", 11, "bold"), padding=[15, 8]
        )

    @staticmethod
    def _configure_widget_styles(style: ttk.Style, colors: dict[str, str]) -> None:
        style.configure(
            "Modern.TCombobox",
            fieldbackground=colors["select_bg"],
            background=colors["select_bg"],
            foreground=colors["fg"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Modern.TCheckbutton",
            background=colors["bg"],
            foreground=colors["fg"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Modern.TLabelframe",
            background=colors["bg"],
            foreground=colors["accent"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Modern.TLabelframe.Label",
            background=colors["bg"],
            foreground=colors["accent"],
            font=("Segoe UI", 11, "bold"),
        )
