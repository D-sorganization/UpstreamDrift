"""Trusted local JSON-line projection worker; no Qt application imports."""

from __future__ import annotations

import json
import sys
from typing import Any

from .necromatcher import NecromatcherLibrary
from .necromatcher_projection import project_fit_frame


def main() -> None:
    """Serve verified library requests until the owning process closes stdin."""
    for line in sys.stdin:
        response: dict[str, Any]
        try:
            root, fit_id, frame_index = json.loads(line)
            result = project_fit_frame(NecromatcherLibrary(root), fit_id, frame_index)
            response = {"result": result}
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            OSError,
            ImportError,
            RuntimeError,
        ) as exc:
            response = {"type": type(exc).__name__, "error": str(exc)}
        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
