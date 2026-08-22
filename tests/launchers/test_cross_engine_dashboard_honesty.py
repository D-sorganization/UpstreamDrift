"""Chart-honesty tests for the cross-engine dashboard (#8816, #8817).

- A stub-backed series is never labeled as the real engine.
- The robustness chart never replicates one aggregate score per engine as
  if it were per-engine data.
"""

from __future__ import annotations

import pytest

from src.launchers import cross_engine_dashboard as ced
from src.shared.python.analysis.cross_engine import BACKEND_REAL, BACKEND_STUB

pytestmark = pytest.mark.unit

_CV_SUMMARY = {
    "cv_total_energy_final": 0.1,
    "cv_end_effector_speed_final": 0.05,
    "cv_peak_end_effector_speed": 0.02,
}


class TestStubLabeling:
    """#8817 — stub-backed series must be labeled as such."""

    def test_stub_backed_series_never_labeled_as_real_engine(self) -> None:
        real_label = ced._format_engine_result_label("mujoco", backend=BACKEND_REAL)
        stub_label = ced._format_engine_result_label("mujoco", backend=BACKEND_STUB)
        assert stub_label != real_label
        assert "stub" in stub_label.lower()
        assert "unavailable" in stub_label.lower()
        # The stub label must not claim mujoco's native conventions.
        assert "qvel" not in stub_label

    def test_real_backend_keeps_convention_label(self) -> None:
        label = ced._format_engine_result_label("mujoco", backend=BACKEND_REAL)
        assert "qvel" in label

    def test_pendulum_stub_is_not_flagged_as_substituted(self) -> None:
        # pendulum_stub is requested as a stub — no substitution occurred.
        label = ced._format_engine_result_label("pendulum_stub", backend=BACKEND_STUB)
        assert "unavailable" not in label.lower()

    def test_log_label_declares_stub(self) -> None:
        log_label = ced._format_engine_result_log_label("drake", backend=BACKEND_STUB)
        assert "stub" in log_label.lower()


class TestRobustnessChartSeries:
    """#8816 — no fake per-engine breakdown of one aggregate score."""

    def test_per_engine_scores_are_used_when_available(self) -> None:
        labels, values = ced._robustness_chart_series(
            ["mujoco", "drake"],
            _CV_SUMMARY,
            backends={"mujoco": BACKEND_REAL, "drake": BACKEND_STUB},
            robustness_per_engine={"mujoco": 0.9, "drake": 0.4},
        )
        assert values == [0.9, 0.4]
        assert len(labels) == 2
        assert "stub" in labels[1].lower()

    def test_no_replicated_aggregate_per_engine(self) -> None:
        """Without per-engine data the chart collapses to one aggregate bar."""
        labels, values = ced._robustness_chart_series(
            ["mujoco", "drake", "pinocchio"], _CV_SUMMARY
        )
        assert len(values) == 1
        assert len(labels) == 1
        assert "aggregate" in labels[0].lower()
        # It must NOT present one bar per engine.
        assert "mujoco" not in labels[0]
