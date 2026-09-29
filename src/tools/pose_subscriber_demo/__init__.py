"""Pose Subscriber demo — live mirror of Pose Studio's canonical pose.

This is the consumer half of the cross-tool live-pose demo (Subtask 6 of
EPIC #4993). It subscribes to the ``pose/canonical`` realtime channel
and renders the most recent canonical pose with a coarse forward-
kinematics skeleton. Designed to be embedded in the launcher as a tab or
dock, it exists primarily to prove out the realtime IPC layer end-to-
end with a real GUI consumer.

Public entry point::

    python -m src.tools.pose_subscriber_demo

The :class:`_PoseSubscriberDemoEmbedAdapter` is registered with the
embeddable-tool registry on import so the launcher can host the demo
without spawning a separate process.
"""

from __future__ import annotations

from src.shared.python.launcher_embed import (
    get_embeddable_tool,
    register_embeddable_tool,
)

from ._embed_adapter import _PoseSubscriberDemoEmbedAdapter

# Module-level singleton: registries key on ``tool_id`` so a single
# instance is sufficient. Constructing the adapter is cheap (it does not
# build a Qt widget until ``create_main_widget`` is called).
_ADAPTER = _PoseSubscriberDemoEmbedAdapter()
# Guard against double-import (e.g. test reloads). The registry rejects
# duplicate ids by design — we want a quiet no-op here instead.
if get_embeddable_tool(_ADAPTER.tool_id) is None:
    register_embeddable_tool(_ADAPTER)


__all__ = ["_ADAPTER"]
