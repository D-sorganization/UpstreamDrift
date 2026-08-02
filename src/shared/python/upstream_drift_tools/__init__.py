"""Deprecated compatibility alias for ``shared.python.sidekick``."""

from __future__ import annotations

import importlib
import sys
import warnings

from src.shared.python.import_aliases import install_aliases

warnings.warn(
    "upstream_drift_tools is deprecated and will be removed in a future release. "
    "Import from shared.python.sidekick instead.",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = importlib.import_module("shared.python.sidekick")
install_aliases(
    {
        "sidekick": "shared.python.sidekick",
        "upstream_drift_tools": "shared.python.sidekick",
        "shared.python.upstream_drift_tools": "shared.python.sidekick",
        "src.shared.python.sidekick": "shared.python.sidekick",
        "src.shared.python.upstream_drift_tools": "shared.python.sidekick",
    }
)

sys.modules[__name__] = _canonical
