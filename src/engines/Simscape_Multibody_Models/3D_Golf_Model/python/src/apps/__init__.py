"""Apps package for the 3D Golf Model engine.

The C3D viewer's embed-adapter is registered here at import time so the
launcher can host the viewer in a tab/dock without spawning a separate
process. The registration is best-effort: if the
:mod:`launcher_embed` contract is unavailable (e.g., shared package not
on ``sys.path`` in some embedded contexts), the import is silently
skipped so importing :mod:`apps` keeps working in those contexts.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    from src.shared.python.launcher_embed import (
        get_embeddable_tool,
        register_embeddable_tool,
    )

    from ._embed_adapter import _C3DViewerEmbedAdapter

    _ADAPTER = _C3DViewerEmbedAdapter()
    # Guard against double-import (e.g. test reloads). The registry
    # rejects duplicate ids by design — we want a quiet no-op here
    # instead of a hard error.
    if get_embeddable_tool(_ADAPTER.tool_id) is None:
        register_embeddable_tool(_ADAPTER)
