"""
Video Analyzer for Golf Swing Analysis

This package provides video-based golf swing analysis capabilities
integrated with the UpstreamDrift Golf Modeling Suite.

Features:
- Pose estimation using MediaPipe
- Swing phase detection
- Angle measurements (hip, shoulder, spine)
- Tempo and timing analysis
- Professional reports and recommendations

Usage:
    from src.tools.video_analyzer import SwingAnalyzer, VideoProcessor

    # Analyze a video file
    analyzer = SwingAnalyzer()
    results = analyzer.analyze_video("swing.mp4")

    # Get swing metrics
    print(f"Tempo: {results.tempo_ratio:.2f}")
    print(f"X-Factor: {results.x_factor:.1f}°")
"""

from .analyzer import SwingAnalyzer
from .video_processor import VideoProcessor
from .pose_estimator import PoseEstimator
from .types import (
    SwingAnalysis,
    SwingPhase,
    BodyAngles,
    TempoMetrics,
    BalanceMetrics,
    SwingScores,
)

__version__ = "1.0.0"
__all__ = [
    "SwingAnalyzer",
    "VideoProcessor",
    "PoseEstimator",
    "SwingAnalysis",
    "SwingPhase",
    "BodyAngles",
    "TempoMetrics",
    "BalanceMetrics",
    "SwingScores",
]
