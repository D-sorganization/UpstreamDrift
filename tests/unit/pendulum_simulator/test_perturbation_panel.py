"""Offscreen tests for the perturbation panel result display (issue #12829).

The background worker emits ``[PerturbationSummary]`` from
``PendulumPerturbationAnalyzer.run_batch``; finishing a batch must render that
summary instead of raising on a dict/dataclass mismatch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from src.shared.python.perturbation.config import (  # noqa: E402
    PerturbationConfig,
    PerturbationSummary,
)
from src.shared.python.perturbation.statistics import (  # noqa: E402
    MetricStatistics,
    compute_metric_statistics,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def panel():
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    from src.shared.python.pendulum_simulator.gui.perturbation_panel import (
        PerturbationPanel,
    )

    app = QApplication.instance() or QApplication([])
    widget = PerturbationPanel()
    yield widget
    widget.deleteLater()
    app.processEvents()


def _summary(speed_stats: MetricStatistics, n_trials: int = 4) -> PerturbationSummary:
    return PerturbationSummary(
        engine_name="pendulum",
        config=PerturbationConfig(n_trials=n_trials, perturb_mode="multiplicative"),
        robustness_score=0.875,
        metrics={"end_effector_speed_final": speed_stats},
        success_rate=1.0,
        execution_time_sec=0.1,
    )


def _labels(panel) -> dict[str, str]:
    return {k: lbl.text() for k, lbl in panel._result_labels.items()}


def test_on_finished_renders_worker_summary(panel) -> None:
    stats = compute_metric_statistics(np.array([40.0, 42.0, 44.0, 46.0]))
    panel._on_finished([_summary(stats)])

    labels = _labels(panel)
    assert labels["Mode"] == "multiplicative"
    assert labels["Mean"] == "43.000 m/s"
    assert labels["Min"] == "40.000 m/s"
    assert labels["Max"] == "46.000 m/s"
    assert labels["CV"] == f"{float(stats.cv) * 100:.2f} %"
    assert labels["Score"] == "0.875"
    assert labels["Trials"] == "4"
    assert "4 trials" in panel._status_label.text()


def test_nan_cv_is_shown_as_not_applicable(panel) -> None:
    # NSX-82 (#12711): cv is NaN when the mean is near zero.
    stats = MetricStatistics(
        mean=0.0,
        std=0.5,
        cv=float("nan"),
        min_val=-1.0,
        max_val=1.0,
        median=0.0,
        iqr=0.5,
        p5=-0.9,
        p95=0.9,
    )
    panel._on_finished([_summary(stats)])

    assert _labels(panel)["CV"] == "n/a (mean ~ 0)"


def test_all_trials_failed_summary_does_not_raise(panel) -> None:
    empty = PerturbationSummary(
        engine_name="pendulum",
        config=PerturbationConfig(n_trials=3),
        robustness_score=0.0,
        metrics={},
        success_rate=0.0,
        execution_time_sec=0.0,
    )
    panel._on_finished([empty])

    assert "No successful trials" in panel._status_label.text()
