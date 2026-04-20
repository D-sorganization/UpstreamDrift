"""Launch monitor ingestion API.

Adapters that parse TrackMan and FlightScope CSV / JSON exports and
normalise the data into :class:`LaunchMonitorShot` — the single source
of truth for all post-import analysis.

Usage::

    from src.shared.python.launch_monitor import TrackManAdapter, FlightScopeAdapter

    shots = TrackManAdapter.from_csv("session.csv")
    lc = shots[0].to_launch_conditions()
"""

from .adapters import FlightScopeAdapter, TrackManAdapter
from .types import LaunchMonitorShot

__all__ = [
    "FlightScopeAdapter",
    "LaunchMonitorShot",
    "TrackManAdapter",
]
