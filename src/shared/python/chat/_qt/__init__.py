# ruff: noqa: E501
"""Private submodule package for ``ChatDockWidget`` internals.

Splits the historical monolithic ``_chat_dock_widget_qt`` module into
smaller focused units. Every public name continues to be re-exported
from ``chat._chat_dock_widget_qt`` so downstream consumers see no API
change (Law of Demeter — external code never imports from ``_qt``).
"""

from __future__ import annotations
