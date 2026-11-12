"""Honesty and safety labeling for injury heuristics (issue #15011)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.dataclasses import CITATION_SPINAL_LOAD
from src.shared.python.injury.heuristic_disclaimers import (
    INJURY_RISK_SCORER_LIMITATIONS,
    SPINAL_LOAD_HEURISTIC_LIMITATIONS,
)
from src.shared.python.injury.injury_risk import InjuryRiskScorer
from src.shared.python.injury.spinal_load_analysis import SpinalLoadAnalyzer

pytestmark = [pytest.mark.unit]


class TestSpinalLoadHonesty:
    def test_analyzer_docstring_labels_heuristic_not_inverse_dynamics(self) -> None:
        doc = SpinalLoadAnalyzer.__doc__ or ""
        lowered = doc.lower()
        assert "not inverse dynamics" in lowered or "not inverse-dynamics" in lowered
        assert "heuristic" in lowered

    def test_analyze_sets_heuristic_limitations_on_result(self) -> None:
        analyzer = SpinalLoadAnalyzer(body_weight=80.0)
        n = 20
        t = np.linspace(0, 1.0, n)
        zero = np.zeros(n)
        result = analyzer.analyze(
            joint_angles={
                "lumbar_flexion": zero,
                "lumbar_lateral": zero,
                "lumbar_rotation": zero,
            },
            joint_velocities={
                "lumbar_flexion": zero,
                "lumbar_lateral": zero,
                "lumbar_rotation": zero,
            },
            joint_torques={
                "lumbar_flexion": zero,
                "lumbar_lateral": zero,
                "lumbar_rotation": zero,
            },
            time=t,
        )
        assert result.method_limitations == SPINAL_LOAD_HEURISTIC_LIMITATIONS
        assert "not inverse dynamics" in result.method_limitations
        assert result.user_facing_method_label().startswith("Heuristic")

    def test_citation_does_not_claim_validated_thresholds(self) -> None:
        notes = (CITATION_SPINAL_LOAD.notes or "").lower()
        assert "validated by" not in notes
        assert "threshold" not in notes or "not" in notes
        assert "8" in notes or "hosea" in notes

    def test_recommendations_avoid_clinical_clearance_wording(self) -> None:
        from src.shared.python.injury.spinal_load_analysis import (
            create_example_analysis,
        )

        analyzer, result = create_example_analysis()
        recs = analyzer.get_recommendations(result)
        joined = " ".join(recs).lower()
        assert "clinical" not in joined
        assert "diagnos" not in joined
        assert "within safe limits" not in joined


class TestInjuryRiskScorerHonesty:
    def test_report_includes_limitations(self) -> None:
        report = InjuryRiskScorer().score()
        assert report.limitations == INJURY_RISK_SCORER_LIMITATIONS
        assert "unvalidated" in report.limitations.lower()

    def test_scorer_docstring_does_not_claim_evidence_based_thresholds(self) -> None:
        doc = InjuryRiskScorer.__doc__ or ""
        lowered = doc.lower()
        assert "evidence-based thresholds" not in lowered
