"""Unit tests for Tour Matching Viewer force inspection and counterfactual UI (MV-06, #10482)."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.shared.python.motion_matching.candidate import (
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_session import CandidateSession
from src.tools.tour_matching_viewer.force_inspection import ForceInspectionWidget
from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

pytestmark = [
    skip_if_unavailable("pyqt6"),
    pytest.mark.unit,
]


@pytest.fixture(scope="module")
def qapp() -> Any:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def _create_mock_session(
    has_forces: bool = True,
    is_dynamic: bool = True,
    fz: float = 650.0,
) -> CandidateSession:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    coord_names = tuple(spec["coordinate_order"])
    n_coords = len(coord_names)

    time_s = np.array([0.0, 0.05, 0.10], dtype=np.float64)
    q = np.zeros((3, n_coords), dtype=np.float64)
    v = np.zeros((3, n_coords), dtype=np.float64) if is_dynamic else None
    tau = np.ones((3, n_coords), dtype=np.float64) * 10.0 if is_dynamic else None
    if tau is not None and n_coords >= 2:
        tau[:, 0] = 15.0
        tau[:, 1] = -5.0

    ext = (
        np.array(
            [
                [5.0, 2.0, fz, 0.5, 0.2, 0.1],
                [6.0, 3.0, fz + 10.0, 0.6, 0.3, 0.1],
                [7.0, 4.0, fz + 20.0, 0.7, 0.4, 0.1],
            ],
            dtype=np.float64,
        )
        if has_forces
        else None
    )
    meta = CandidateMetadata(
        profile=CandidateProfile.DYNAMIC if is_dynamic else CandidateProfile.KINEMATIC,
        coordinate_names=coord_names,
    )
    markers = CandidateMarkers(
        model_markers_m=np.zeros((3, 4, 3), dtype=np.float64),
        target_markers_m=np.zeros((3, 4, 3), dtype=np.float64),
    )
    cand = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        markers=markers,
        auxiliary=CandidateAuxiliary(external_forces=ext),
    )
    return CandidateSession(
        candidate=cand,
        specification=spec,
        candidate_sha256="viewer_test_cand_sha",
        model_sha256="viewer_test_model_sha",
        coordinate_names=coord_names,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        external_forces=ext,
        is_accepted=True,
    )


def test_force_inspection_widget_kinematic_session(qapp: Any) -> None:
    """Kinematic session disables counterfactual button and clears telemetry."""
    widget = ForceInspectionWidget()
    session = _create_mock_session(has_forces=False, is_dynamic=False)
    widget.set_candidate_session(session)

    assert not widget._fork_btn.isEnabled()
    assert "kinematic" in widget._fork_status_label.text().lower()


def test_force_inspection_widget_dynamic_session_telemetry(qapp: Any) -> None:
    """Dynamic session updates Fz, CoP, and torque telemetry on frame change."""
    widget = ForceInspectionWidget()
    session = _create_mock_session(has_forces=True, is_dynamic=True, fz=700.0)
    widget.set_candidate_session(session)

    assert widget._fork_btn.isEnabled()

    widget.update_frame(1)
    assert "710.0 N" in widget._fz_label.text()
    assert "CoP" in widget._cop_label.text()
    assert "Peak |tau|: 15.0 N*m" in widget._peak_torque_label.text()


def test_force_inspection_widget_counterfactual_signal(qapp: Any) -> None:
    """Clicking fork button generates rollout and emits counterfactualForked signal."""
    widget = ForceInspectionWidget()
    session = _create_mock_session(has_forces=True, is_dynamic=True)
    widget.set_candidate_session(session)

    received_forks = []
    widget.counterfactualForked.connect(received_forks.append)

    widget.update_frame(0)
    widget._fork_btn.click()

    assert len(received_forks) == 1
    fork = received_forks[0]
    assert fork.fork_frame_idx == 0
    assert "Forked" in widget._fork_status_label.text()


def test_tour_matching_viewer_embeds_force_widget(qapp: Any) -> None:
    """TourMatchingViewerWidget hosts force_widget and propagates session/frames."""
    viewer = TourMatchingViewerWidget()
    assert hasattr(viewer, "force_widget")
    assert isinstance(viewer.force_widget, ForceInspectionWidget)

    session = _create_mock_session(has_forces=True, is_dynamic=True, fz=800.0)
    viewer.load_candidate_session(session)

    assert viewer.force_widget._session is session
    viewer.render_frame(1)
    assert viewer.force_widget._current_frame_idx == 1
