"""Video Analyzer tool for golf swing analysis.

Provides pose-based golf swing assessment using MediaPipe (optional
runtime dependency).  Core math utilities are available in headless
environments without any external dependencies.

Audit status (issue #6089): SCAFFOLD — this package ships the
headless analytics layer (:class:`SwingAnalyzer`, type definitions)
but has no PyQt6 GUI module yet.  A GUI and EmbeddableTool adapter
are planned as a follow-up to issue #6089.

Public API:
    - :class:`~src.tools.video_analyzer.analyzer.SwingAnalyzer`
    - :class:`~src.tools.video_analyzer.types.Landmark`
    - :class:`~src.tools.video_analyzer.types.PoseFrame`
    - :class:`~src.tools.video_analyzer.types.PostureMetrics`
"""

from src.tools.video_analyzer.analyzer import SwingAnalyzer
from src.tools.video_analyzer.types import Landmark, PoseFrame, PostureMetrics

__all__ = [
    "Landmark",
    "PoseFrame",
    "PostureMetrics",
    "SwingAnalyzer",
]
