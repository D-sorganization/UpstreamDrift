"""Tour Matching Viewer package (Visuals Handoff Step 3)."""

from __future__ import annotations

import contextlib

from src.tools.tour_matching_viewer.core import (
    ClubOnlyCompareView,
    ReplayData,
    ViewerFrame,
    body_poses_from_state,
    club_only_compare_from_ui_result,
    load_replay,
    viewer_frame,
)

with contextlib.suppress(ImportError):
    from src.shared.python.launcher_embed import (
        get_embeddable_tool,
        register_embeddable_tool,
    )
    from src.tools.tour_matching_viewer._embed_adapter import (
        _TourMatchingViewerEmbedAdapter,
    )

    if get_embeddable_tool(_TourMatchingViewerEmbedAdapter.tool_id) is None:
        register_embeddable_tool(_TourMatchingViewerEmbedAdapter())

__all__ = [
    "ClubOnlyCompareView",
    "ReplayData",
    "ViewerFrame",
    "body_poses_from_state",
    "club_only_compare_from_ui_result",
    "load_replay",
    "viewer_frame",
]
