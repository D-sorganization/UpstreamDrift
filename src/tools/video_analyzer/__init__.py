"""Video Analyzer tool for golf swing analysis.

Provides pose-based golf swing assessment using MediaPipe (optional
runtime dependency).  Core math utilities are available in headless
environments without any external dependencies.

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
