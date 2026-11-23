"""A supplied-pose FK comparison must identify the operation it qualifies."""

import json

import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.parity_report import build_parity_report
from src.shared.python.motion_matching.parity_schema import UnifiedParityReport

pytestmark = pytest.mark.unit


class KinematicPlant:
    """Matching FK outputs with deliberately unusable dynamics."""

    plant_sha = "synthetic-model"
    coordinate_order = ("q0",)

    def marker_positions(self, q: np.ndarray, attachments: dict) -> np.ndarray:
        return np.repeat(q[:, :, None], 3, axis=2)

    def step(self, *args: object) -> None:
        raise AssertionError("An FK report must not imply a replay was performed")


@pytest.fixture
def report(monkeypatch: pytest.MonkeyPatch) -> UnifiedParityReport:
    from src.shared.python.engine_core import engine_availability

    monkeypatch.setattr(engine_availability, "is_engine_available", lambda _: False)
    candidate = MatchedSwingCandidate(
        metadata=CandidateMetadata(
            coordinate_names=("q0",),
            actuator_names=("tau0",),
            profile=CandidateProfile.DYNAMIC,
        ),
        time_s=np.array([0.0, 0.1, 0.2]),
        q=np.zeros((3, 1)),
        v=np.ones((3, 1)),
        tau=np.full((3, 1), 100.0),
    )
    return build_parity_report(
        candidate,
        plants={"pinocchio": KinematicPlant(), "drake": KinematicPlant()},
        engines=["pinocchio", "drake"],
        attachments={"marker": ("body", (0.0, 0.0, 0.0))},
        spec={},
    )


def test_matching_fk_identifies_qualified_operation_even_with_dynamic_candidate(
    report: UnifiedParityReport,
) -> None:
    # Existing numerical verdicts remain scoped to the computation that ran.
    assert report.status == "PASSED"
    assert report.is_parity_accepted
    assert report.evaluation_operation == "forward_kinematics"
    assert "dynamic replay" in report.note
    assert "physical acceptance" in report.note


def test_operation_and_limits_survive_json_and_markdown(
    report: UnifiedParityReport,
) -> None:
    payload = json.loads(report.to_json())
    assert payload["evaluation_operation"] == "forward_kinematics"
    restored = UnifiedParityReport.from_dict(payload)
    assert restored.evaluation_operation == report.evaluation_operation
    markdown = restored.render_markdown()
    assert "**Evaluation Operation:** `forward_kinematics`" in markdown
    assert report.note in markdown


def test_legacy_report_does_not_infer_an_operation_from_passing_status() -> None:
    report = UnifiedParityReport.from_dict(
        {"status": "PASSED", "is_parity_accepted": True}
    )
    assert report.evaluation_operation == "unspecified"
    assert "**Evaluation Operation:** `unspecified`" in report.render_markdown()


@pytest.mark.parametrize("operation", [None, "", "   ", 42])
def test_invalid_operation_is_refused_when_loading_report(operation: object) -> None:
    with pytest.raises(ValueError, match="evaluation_operation"):
        UnifiedParityReport.from_dict({"evaluation_operation": operation})
