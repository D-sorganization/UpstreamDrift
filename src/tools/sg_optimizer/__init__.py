"""Strokes Gained Optimizer — UpstreamDrift launcher tile.

Phase 1 ships only the headless library and CLI; the PyQt6 UI is added in
Phase 3 (#6272). Importing this package does not require Qt.

Epic: #6269.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
