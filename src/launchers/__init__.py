"""Launchers namespace package.

Contains entry-point launcher modules for the various physics engines and GUI
front-ends.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = [
    "BaseLauncher",
    "LaunchItem",
    "build_about_html",
    "gather_version_info",
    "run_launcher",
    "show_about_dialog",
    "shot_tracer_main",
]


def __getattr__(name: str) -> Any:
    if name in {"build_about_html", "gather_version_info", "show_about_dialog"}:
        from . import about_dialog

        return getattr(about_dialog, name)
    if name in {"BaseLauncher", "LaunchItem", "run_launcher"}:
        from . import base

        return getattr(base, name)
    if name == "shot_tracer_main":
        from .shot_tracer import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
