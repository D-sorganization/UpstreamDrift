"""Embeddable-tool adapter for the Motion-Match Preview / Starting-Pose Matcher.

Implements the :class:`~src.shared.python.launcher_embed.EmbeddableTool`
protocol so the launcher can host the Motion-Match Preview as a tab or
dock widget instead of always opening it as a standalone window. Part
of Subtask 5 / #4998 of EPIC #4993.

The package is named ``starting_pose_matcher`` for historical reasons,
but the launcher tile id is ``motion_target_preview`` (see
``src/config/models.yaml``); the adapter advertises the latter.
"""

# background: yes (defaults); cleanup idempotent (swap-then-clear). The
# Motion-Match Preview holds expensive loaded-mocap / pose-edit state, so
# backgrounding (keep-running on close) preserves it cheaply; no scarce GPU
# context held at the adapter level, so structural defaults apply (#6013).

from __future__ import annotations

from typing import Any

from src.shared.python.launcher_embed import EmbedCapabilities
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["_MotionMatchPreviewEmbedAdapter"]


class _MotionMatchPreviewEmbedAdapter:
    """Adapter exposing :class:`MainWidget` through the embed contract."""

    tool_id: str = "motion_target_preview"

    def __init__(self) -> None:
        # Track every widget we hand out so :meth:`cleanup` can dispose
        # of resources even if the host forgets to delete the widget.
        self._widgets: list[Any] = []

    def embed_capabilities(self) -> EmbedCapabilities:
        # The Motion-Match Preview is the largest GUI in the codebase
        # (~3.1k lines of widget logic); a dock would crush the 3D
        # viewport, so prefer a full tab and reserve a generous
        # minimum size.
        return EmbedCapabilities(
            supports_embedded=True,
            prefers_dock=False,
            min_size=(1024, 720),
            requires_separate_qapplication=False,
        )

    def create_main_widget(self, parent: Any) -> Any:
        # Lazy import: ``gui_main_widget`` pulls in PyQt6 and matplotlib,
        # and we want the registry / adapter to import cleanly in
        # headless contexts (CI, docs builds) where those may be
        # unavailable until a fixture installs ``QT_QPA_PLATFORM=
        # offscreen``.
        from .gui_main_widget import MainWidget

        widget = MainWidget(parent)
        self._widgets.append(widget)
        return widget

    def cleanup(self) -> None:
        # Idempotent: hosts may call cleanup more than once during
        # shutdown. Forward to every widget we handed out, but never
        # raise — the host's shutdown path must not depend on us.
        widgets, self._widgets = self._widgets, []
        for widget in widgets:
            try:
                widget.cleanup()
            except Exception:  # pragma: no cover - defensive
                logger.exception("motion_target_preview widget cleanup raised")

    def is_dirty(self) -> bool:
        # Session edits are user-driven and only persisted via the
        # explicit "Save session" button; there is no in-memory dirty
        # flag we can poll cheaply. Returning False keeps the host's
        # tab-close path silent — the host treats False as "skip the
        # prompt", which matches the historical behaviour where closing
        # the standalone window also skipped any confirmation.
        return False
