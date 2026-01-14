"""
Injury Prevention and Risk Assessment Module

This module provides biomechanical analysis tools for assessing injury risk
during the golf swing, with a focus on the lumbar spine (the most common
site of golf-related injuries).

Key Components:
- SpinalLoadAnalyzer: Calculate compression, shear, and torsion forces on the lumbar spine
- JointStressAnalyzer: Assess stress on all major joints (hip, shoulder, wrist, knee, elbow)
- InjuryRiskScorer: Aggregate risk scoring with evidence-based thresholds
- SwingModificationRecommender: Suggest safer swing alternatives

Scientific Foundation:
- Low back pain affects 25-58% of golfers (highest injury incidence)
- Compressive forces can reach 8x body weight on L4-L5 during the swing
- The "crunch factor" from lateral bending creates asymmetric vertebral loading
- Modern swing X-factor stretch increases thoracolumbar loading

References:
- Lindsay & Horton (2002) The lumbar spine and low back pain in golf
- Cole & Grimshaw (2008) Biomechanics of the Modern Golf Swing
- Gluck et al. (2008) Golf-Related Low Back Pain Prevention Strategies
- Cejka et al. (2024) Biomechanical parameters associated with lower back pain
"""

from .spinal_load_analysis import SpinalLoadAnalyzer, SpinalLoadResult, SpinalRiskLevel
from .joint_stress import JointStressAnalyzer, JointStressResult
from .injury_risk import InjuryRiskScorer, InjuryRiskReport, RiskLevel
from .swing_modifications import SwingModificationRecommender, SwingModification

__all__ = [
    "SpinalLoadAnalyzer",
    "SpinalLoadResult",
    "SpinalRiskLevel",
    "JointStressAnalyzer",
    "JointStressResult",
    "InjuryRiskScorer",
    "InjuryRiskReport",
    "RiskLevel",
    "SwingModificationRecommender",
    "SwingModification",
]
