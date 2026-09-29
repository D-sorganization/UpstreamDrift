"""Fallback engines for the cross-engine dashboard.

The implementation moved to the shared cross-engine analysis service
(issue #7455) so the desktop dashboard and the web API share one compute
path. This module remains as a thin re-export for backwards compatibility.
"""

from __future__ import annotations

from src.shared.python.analysis.cross_engine import StubEngine as StubEngine

__all__ = ["StubEngine"]
