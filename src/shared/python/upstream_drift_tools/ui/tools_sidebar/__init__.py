"""Tools sidebar package for UpstreamDrift UI.

Re-exports the Sidekick sidebar builder so host code can import from either
``sidekick.ui.tools_sidebar`` or ``upstream_drift_tools.ui.tools_sidebar``
without caring about the internal package layout.
"""

from sidekick.ui.tools_sidebar import (  # noqa: F401
    create_tools_sidebar,
)

__all__ = ["create_tools_sidebar"]
