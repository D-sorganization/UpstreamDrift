"""Embeddable-tool adapter for the MyoSuite dashboard.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the MyoSuite dashboard as a tab or
dock widget instead of always opening a standalone top-level window.
Constructed at import time by
:mod:`src.engines.physics_engines.myosuite.python.__init__`, guarded by
:func:`contextlib.suppress` so headless contexts (no PyQt6, no
``myosuite`` wheel) can still import the package.

Embeddability notes
-------------------
The MyoSuite engine has historically lacked a Qt GUI — the standalone
``main()`` simply pointed users at the web docs. The dashboard added in
this refactor is intentionally lightweight: it surfaces engine status
and usage hints rather than embedding a live MuJoCo viewport. The
optional :class:`MyoSuitePhysicsEngine` import is deferred to a user
action so widget construction stays headless-safe.

Part of Subtask 5 / #4998 of EPIC #4993.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import (
    EmbedCapabilities,
    get_embeddable_tool,
    register_embeddable_tool,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_MyoSuiteDashboardEmbedAdapter"]


class _MyoSuiteDashboardEmbedAdapter:
    """Adapter exposing :class:`MainWidget` through the embed contract."""

    tool_id: str = "myosim_suite"

    def __init__(self) -> None:
        # Track every widget we hand out so :meth:`cleanup` can dispose
        # of resources even if the host forgets to delete the widget.
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1000, 700),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: the GUI module pulls in PyQt6. Keeping this
        # import inside ``create_main_widget`` lets the adapter (and the
        # import-time registry side-effect) load cleanly in headless
        # contexts.
        from .gui import MainWidget

        widget = MainWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        """Best-effort: forward cleanup to every widget we handed out.

        Idempotent: hosts may call cleanup more than once during
        shutdown. Never raises — the host's shutdown path must not
        depend on us.
        """
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("myosim_suite widget cleanup raised")

    def is_dirty(self) -> bool:
        # The MyoSuite dashboard is a passive status surface — there is
        # no in-memory state worth prompting the user about on close.
        return False


# Register on first import. Guard against re-registration so that
# importing the adapter module from both the package ``__init__`` and
# from a test does not raise ``ValueError: already registered``.
_ADAPTER = _MyoSuiteDashboardEmbedAdapter()
if get_embeddable_tool(_ADAPTER.tool_id) is None:
    register_embeddable_tool(_ADAPTER)
