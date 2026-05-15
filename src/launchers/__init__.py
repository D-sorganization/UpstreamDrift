"""Launchers namespace package.

Contains entry-point launcher modules for the various physics engines and GUI
front-ends.
"""

from .about_dialog import build_about_html, gather_version_info, show_about_dialog
from .base import BaseLauncher, LaunchItem, run_launcher
from .shot_tracer import main as shot_tracer_main

__all__: list[str] = [
    "BaseLauncher",
    "LaunchItem",
    "build_about_html",
    "gather_version_info",
    "run_launcher",
    "show_about_dialog",
    "shot_tracer_main",
]
