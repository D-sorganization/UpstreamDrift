
"""
Tutorial 03: Injury Risk Analysis
=================================

This tutorial demonstrates how to use the Injury Risk Analysis module
to evaluate a golf swing for potential injury risks.

We will:
1. Initialize the InjuryRiskScorer
2. Create synthetic swing data (mocking a biomechanical analysis)
3. Calculate spinal and joint risks
4. Generate a comprehensive report
"""

import sys
from pathlib import Path

# Ensure we can import from the suite
# Adjust this path if running from a different directory
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from shared.python.injury.injury_risk import (
    InjuryRiskReport,
    InjuryRiskScorer,
    RiskFactor,
)
from shared.python.injury.spinal_load_analysis import create_example_analysis


def run_tutorial():
    print("🏌️  Starting Injury Risk Analysis Tutorial\n")

    # 1. Spinal Load Analysis
    # -----------------------
    print("1. Performing Spinal Load Analysis...")
    spinal_analyzer, spinal_result = create_example_analysis()
    
    print(f"   Peak Compression: {spinal_result.peak_compression_bw:.1f}x BW")
    print(f"   Overall Spinal Risk: {spinal_result.overall_risk.value.upper()}")
    
    # 2. Joint Stress Analysis
    # ------------------------
    print("\n2. Performing Joint Stress Analysis (Mock Data)...")
    # In a real scenario, this would come from JointStressAnalyzer
    # Here we mock the results structure expected by the Scorer
    from unittest.mock import Mock
    
    joint_results = {}
    joints = ["hip_lead", "hip_trail", "shoulder_lead", "shoulder_trail", 
              "elbow_lead", "elbow_trail", "wrist_lead", "wrist_trail"]
              
    for joint in joints:
        mock_res = Mock()
        mock_res.risk_score = 45.0 if "lead" in joint else 20.0 # Higher risk on lead side
        mock_res.impingement_risk = True if joint == "hip_lead" else False
        joint_results[joint] = mock_res
        
    print("   Joint analysis complete.")

    # 3. Comprehensive Risk Scoring
    # -----------------------------
    print("\n3. Generating Comprehensive Risk Report...")
    scorer = InjuryRiskScorer()
    
    # Mocking other inputs
    swing_metrics = {
        "sequence_timing_error": 0.05,
        "tempo_ratio": 3.0,
        "early_extension": 5.0
    }
    training_load = {
        "acwr": 1.1, # Acute:Chronic Workload Ratio
        "weekly_swings": 400
    }
    
    report = scorer.score(
        spinal_result=spinal_result,
        joint_results=joint_results,
        swing_metrics=swing_metrics,
        training_load=training_load
    )
    
    # 4. Results
    # ----------
    print("\n" + "="*40)
    print("INJURY RISK REPORT")
    print("="*40)
    print(f"Overall Risk Score: {report.overall_risk_score:.1f}/100")
    print(f"Risk Level:         {report.overall_risk_level.value.upper()}")
    
    print("\nTop Risk Factors:")
    for risk in report.top_risks:
        print(f" - {risk.replace('_', ' ').title()}")
        
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f" - {rec}")
        
    print("\n" + "="*40)
    print("\nTutorial Complete! ✅")

if __name__ == "__main__":
    run_tutorial()
