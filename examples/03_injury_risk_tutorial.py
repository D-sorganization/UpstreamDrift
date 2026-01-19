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

import logging
import sys
from pathlib import Path

# Ensure we can import from the suite
# Adjust this path if running from a different directory
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))


from shared.python.injury.injury_risk import (  # noqa: E402
    InjuryRiskScorer,
)
from shared.python.injury.spinal_load_analysis import (  # noqa: E402
    create_example_analysis,
)

# Setup logger for tutorial
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tutorial")


def run_tutorial() -> None:
    logger.info("🏌️  Starting Injury Risk Analysis Tutorial\n")

    # 1. Spinal Load Analysis
    # -----------------------
    logger.info("1. Performing Spinal Load Analysis...")
    spinal_analyzer, spinal_result = create_example_analysis()

    logger.info(f"   Peak Compression: {spinal_result.peak_compression_bw:.1f}x BW")
    logger.info(f"   Overall Spinal Risk: {spinal_result.overall_risk.value.upper()}")

    # 2. Joint Stress Analysis
    # ------------------------
    logger.info("\n2. Performing Joint Stress Analysis (Mock Data)...")
    # In a real scenario, this would come from JointStressAnalyzer
    # Here we mock the results structure expected by the Scorer
    from unittest.mock import Mock

    joint_results = {}
    joints = [
        "hip_lead",
        "hip_trail",
        "shoulder_lead",
        "shoulder_trail",
        "elbow_lead",
        "elbow_trail",
        "wrist_lead",
        "wrist_trail",
    ]

    for joint in joints:
        mock_res = Mock()
        mock_res.risk_score = (
            45.0 if "lead" in joint else 20.0
        )  # Higher risk on lead side
        mock_res.impingement_risk = True if joint == "hip_lead" else False
        joint_results[joint] = mock_res

    logger.info("   Joint analysis complete.")

    # 3. Comprehensive Risk Scoring
    # -----------------------------
    logger.info("\n3. Generating Comprehensive Risk Report...")
    scorer = InjuryRiskScorer()

    # Mocking other inputs
    swing_metrics = {
        "sequence_timing_error": 0.05,
        "tempo_ratio": 3.0,
        "early_extension": 5.0,
    }
    training_load = {"acwr": 1.1, "weekly_swings": 400}  # Acute:Chronic Workload Ratio

    report = scorer.score(
        spinal_result=spinal_result,
        joint_results=joint_results,
        swing_metrics=swing_metrics,
        training_load=training_load,
    )

    # 4. Results
    # ----------
    logger.info("\n" + "=" * 40)
    logger.info("INJURY RISK REPORT")
    logger.info("=" * 40)
    logger.info(f"Overall Risk Score: {report.overall_risk_score:.1f}/100")
    logger.info(f"Risk Level:         {report.overall_risk_level.value.upper()}")

    logger.info("\nTop Risk Factors:")
    for risk in report.top_risks:
        logger.info(f" - {risk.replace('_', ' ').title()}")

    logger.info("\nRecommendations:")
    for rec in report.recommendations:
        logger.info(f" - {rec}")

    logger.info("\n" + "=" * 40)
    logger.info("\nTutorial Complete! ✅")


if __name__ == "__main__":
    run_tutorial()
