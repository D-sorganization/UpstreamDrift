"""Deprecation notice for the legacy ``simulation_config.json`` colours (HRG-B2, #13167).

Stdlib only: the Docker image runs this package without the repository's
``src`` tree. Body colours now live in an ``appearance-v1`` document or
``<spec>.appearance.json`` sidecar and the one default-golfer contract
(``src/config/appearance_defaults.json``); the ``colors`` block of this config
is read for backward compatibility only. Use
``model_appearance.appearance_selection.overrides_from_legacy_config`` to
convert a config once.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

MESSAGE = (
    "simulation_config.json 'colors' is deprecated: body colours belong in an "
    "appearance-v1 document or <spec>.appearance.json sidecar (convert once with "
    "model_appearance.appearance_selection.overrides_from_legacy_config)"
)


def warn_if_legacy_colors(
    config: Mapping[str, Any] | None, stacklevel: int = 2
) -> bool:
    """Emit one ``DeprecationWarning`` when ``config`` carries ``colors``."""
    if config and config.get("colors"):
        warnings.warn(MESSAGE, DeprecationWarning, stacklevel=stacklevel + 1)
        return True
    return False
