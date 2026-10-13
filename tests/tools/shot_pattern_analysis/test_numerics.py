"""Numerical acceptance uses fixed cases, steps, and the unchanged 5 cm gate."""

import pytest

from src.tools.shot_pattern_analysis.core import AnalysisConfig
from src.tools.shot_pattern_analysis.physics import ShotOutcome
from src.tools.shot_pattern_analysis.numerics import build_refinement

pytestmark = pytest.mark.unit


class FakePhysics:
    def __init__(self, error_scale=1.0):
        self.calls = []
        self.error_scale = error_scale

    def simulate(self, *, face_deg, path_deg, nominal_face_deg, config):
        self.calls.append((face_deg, path_deg, nominal_face_deg, config.dt_s))
        x = 100 + self.error_scale * config.dt_s
        return ShotOutcome(x, 0.0, x, 0.0, 3000.0, 0.0)


def test_refinement_covers_all_prescribed_cases_and_fixed_gate():
    engine = FakePhysics()
    result = build_refinement({"test": AnalysisConfig()}, engine=engine)
    assert len(result["cases"]) == 30
    assert len(engine.calls) == 90
    assert {row[-1] for row in engine.calls} == {0.02, 0.01, 0.005}
    assert {row[0] - row[2] for row in engine.calls} == {-3.0, 0.0, 3.0}
    assert result["criterion_m"] == 0.05
    assert result["status"] == "passed"
    assert result["max_dt_020_vs_010_m"] == pytest.approx(0.01)


def test_refinement_does_not_relax_gate_on_failure():
    result = build_refinement({"test": AnalysisConfig()}, engine=FakePhysics(10))
    assert result["status"] == "failed"
    assert result["criterion_m"] == 0.05


def test_refinement_requires_configs():
    with pytest.raises(ValueError):
        build_refinement({}, engine=FakePhysics())
