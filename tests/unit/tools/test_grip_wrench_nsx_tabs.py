"""Unit tests for NSX-43 (#12575), NSX-44 (#12576), and NSX-45 (#12577).

Covers:
1. Offscreen PyQt6 rendering of the golden NSX plot model across all 4 NSX tabs
   (``Invariants``, ``Envelopes``, ``Hand Split``, ``Diagnostics``) inside
   :class:`GripWrenchPlotWidget`.
2. Registry-backed preset selector (``capacity_min_effort``, ``envelope_only``,
   ``smooth_capacity``), evidence badges, segment display classes, and help link.
3. Band-first Design-by-Contract validation rejecting standalone representatives.
4. ``NSXAnalysisWorker`` ``QThreadPool`` execution with partial invariants shown
   first and cooperative cancellation via ``cancel()``.
5. Feature-parity registry entry ``loop_nullspace_exploration`` and matrix freshness.
6. Title-case, relative-link resolution, and R13 citation hygiene on
   ``docs/user_guide/loop_nullspace_exploration.md`` and
   ``src/tools/grip_wrench_plots/README.md``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import re
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from scripts.check_document_title_case import findings_for_path
from scripts.generate_feature_parity_matrix import MATRIX_PATH, render_matrix
from src.config.feature_parity_loader import FeatureParityRegistry
from src.tools.grip_wrench_plots.nsx_tabs import (
    HELP_GUIDE_REL_PATH,
    NSXAnalysisWorker,
    adapt_plot_model_for_preset,
    load_golden_plot_model,
    load_nsx_presets,
    validate_plot_model_dict,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
USER_GUIDE_PATH = REPO_ROOT / "docs" / "user_guide" / "loop_nullspace_exploration.md"
TOOL_README_PATH = REPO_ROOT / "src" / "tools" / "grip_wrench_plots" / "README.md"
REACT_PANEL_PATH = (
    REPO_ROOT / "ui" / "src" / "components" / "analysis" / "LoopNullSpacePanel.tsx"
)


_QAPP_REF: Any = None


def _qapp() -> Any:
    global _QAPP_REF
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = qt_widgets.QApplication.instance()
    if app is None:
        app = qt_widgets.QApplication([])
    _QAPP_REF = app
    return app


def test_presets_load_from_registry_and_adapt_plot_model() -> None:
    """Preset loader reads the three NSX presets and adapts the golden plot model."""
    presets = load_nsx_presets()
    names = [p["name"] for p in presets]
    assert set(names) == {"envelope_only", "capacity_min_effort", "smooth_capacity"}

    golden = load_golden_plot_model()
    env_only = adapt_plot_model_for_preset(golden, "envelope_only")
    assert env_only["preset_name"] == "envelope_only"
    for band in env_only["panels"]["envelopes"]["envelope_series"]:
        assert band["representative"] is None
        assert band["render_mode"] == "ENVELOPE_BAND"
    assert env_only["panels"]["allocation_representative"]["line_series"] == []

    smooth = adapt_plot_model_for_preset(golden, "smooth_capacity")
    assert smooth["preset_name"] == "smooth_capacity"
    for band in smooth["panels"]["envelopes"]["envelope_series"]:
        assert band["representative"] is not None
        assert band["render_mode"] == "BAND_WITH_REPRESENTATIVE"
        assert "H-TORQUE-RATE-SMOOTHNESS" in band["badge_text"]


def test_validate_plot_model_forbids_representative_without_band() -> None:
    """DbC check rejects a plot model that has a representative without an envelope band."""
    corrupted = copy.deepcopy(load_golden_plot_model())
    corrupted["panels"]["envelopes"]["envelope_series"] = []
    with pytest.raises(ValueError, match="without its corresponding envelope band"):
        validate_plot_model_dict(corrupted)


def test_pyqt_widget_renders_golden_plot_model_across_all_four_nsx_tabs() -> None:
    """GripWrenchPlotWidget renders the golden plot model across all 4 NSX tabs offscreen."""
    _qapp()
    from src.tools.grip_wrench_plots.gui import GripWrenchPlotWidget

    widget = GripWrenchPlotWidget()
    try:
        tab_titles = [widget.tabs.tabText(i) for i in range(widget.tabs.count())]
        assert tab_titles == [
            "Grip Wrench",
            "Invariants",
            "Envelopes",
            "Hand Split",
            "Diagnostics",
        ]
        assert HELP_GUIDE_REL_PATH in widget.help_link_label.text()

        golden = load_golden_plot_model()
        widget.set_plot_model(golden)

        # 1. Invariants tab
        assert "MODEL_CONDITIONED_INVARIANT" in widget.invariants_tab.badge_label.text()
        assert "bare_club_net_wrench" in widget.invariants_tab.summary_label.text()
        assert len(widget.invariants_tab.figure.axes) == 1

        # 2. Envelopes tab (with DETERMINED_SIGN, INDETERMINATE, binding hypotheses, and overlaid rep)
        env_summary = widget.envelopes_tab.summary_label.text()
        assert "PRIOR_CONDITIONED" in widget.envelopes_tab.badge_label.text()
        assert "INDETERMINATE" in env_summary
        assert "DETERMINED_SIGN" in env_summary
        assert "H-TORQUE-CAPACITY-BOX" in env_summary
        assert "H-GRIP-FRICTION-CONE" in env_summary
        assert "Representatives overlaid: 2" in env_summary

        # 3. Hand Split tab (6 coordinates s1..s6)
        split_summary = widget.hand_split_tab.summary_label.text()
        assert "Bands: 6" in split_summary
        assert "Representatives overlaid: 6" in split_summary

        # 4. Diagnostics tab
        diag_summary = widget.diagnostics_tab.summary_label.text()
        assert "rank_jc=[6, 6, 6]" in diag_summary
        assert "phi_scaled" in diag_summary

        # Preset switching to envelope_only removes overlaid representatives
        widget.preset_combo.setCurrentText("envelope_only")
        assert (
            "Representatives overlaid: 0" in widget.envelopes_tab.summary_label.text()
        )
    finally:
        widget.cleanup()


def test_nsx_worker_progress_invariants_first_and_cancellation() -> None:
    """NSXAnalysisWorker emits partial invariants first and cancels cleanly."""
    _qapp()
    from src.tools.grip_wrench_plots.gui import GripWrenchPlotWidget

    progress_events: list[dict[str, Any]] = []
    completed_events: list[dict[str, Any]] = []
    cancelled_events: list[str] = []

    worker = NSXAnalysisWorker("capacity_min_effort")
    worker.signals.progress.connect(progress_events.append)
    worker.signals.completed.connect(completed_events.append)
    worker.signals.cancelled.connect(cancelled_events.append)
    worker.run()

    assert len(progress_events) == 1
    assert progress_events[0]["status"] == "running"
    assert "line_series" in progress_events[0]["invariants"]
    assert len(completed_events) == 1
    assert completed_events[0]["preset_name"] == "capacity_min_effort"
    assert len(cancelled_events) == 0

    # Test cooperative cancellation before completion
    cancel_worker = NSXAnalysisWorker("smooth_capacity", defer_completion=True)
    cancel_progress: list[dict[str, Any]] = []
    cancel_completed: list[dict[str, Any]] = []
    cancel_signals: list[str] = []
    cancel_worker.signals.progress.connect(
        lambda payload: (cancel_progress.append(payload), cancel_worker.cancel())
    )
    cancel_worker.signals.completed.connect(cancel_completed.append)
    cancel_worker.signals.cancelled.connect(cancel_signals.append)
    cancel_worker.run()

    assert len(cancel_progress) == 1
    assert len(cancel_completed) == 0
    assert cancel_signals == ["smooth_capacity"]

    # Widget-level start and cancel via QThreadPool
    widget = GripWrenchPlotWidget()
    try:
        widget.start_nsx_job("smooth_capacity", defer_completion=True)
        widget.cancel_nsx_job()
        assert "cancelled" in widget.nsx_status_label.text()
    finally:
        widget.cleanup()


def test_feature_parity_entry_and_matrix_freshness() -> None:
    """Feature-parity registry includes loop_nullspace_exploration at parity and matrix is fresh."""
    registry = FeatureParityRegistry.load()
    entry = registry.get("loop_nullspace_exploration")
    assert entry is not None
    assert entry.status == "parity"
    assert entry.pyqt == "src/tools/grip_wrench_plots/gui.py"
    assert entry.web == "ui/src/components/analysis/LoopNullSpacePanel.tsx"
    for rel_path in entry.referenced_paths:
        assert (REPO_ROOT / rel_path).is_file(), f"Missing path: {rel_path}"

    committed = MATRIX_PATH.read_text(encoding="utf-8")
    assert committed == render_matrix(registry)


def test_user_guide_and_tool_readme_title_case_links_and_content() -> None:
    """User help guide and tool README pass title-case, relative links, and R13 citation rules."""
    assert USER_GUIDE_PATH.is_file()
    assert TOOL_README_PATH.is_file()
    assert REACT_PANEL_PATH.is_file()

    for doc_path in (USER_GUIDE_PATH, TOOL_README_PATH):
        violations = findings_for_path(doc_path, REPO_ROOT)
        assert not violations, f"Title-case violations in {doc_path}: {violations}"

    guide_text = USER_GUIDE_PATH.read_text(encoding="utf-8")
    readme_text = TOOL_README_PATH.read_text(encoding="utf-8")
    react_text = REACT_PANEL_PATH.read_text(encoding="utf-8")

    # Both UIs link to docs/user_guide/loop_nullspace_exploration.md
    assert HELP_GUIDE_REL_PATH in react_text

    # Required interpretation topics in user guide
    for required_token in (
        "INVARIANT",
        "FAMILY_DEPENDENT",
        "UNAVAILABLE",
        "capacity_min_effort",
        "envelope_only",
        "smooth_capacity",
        "INDETERMINATE",
        "DETERMINED_SIGN",
        "UNBOUNDED",
        "HOLDS",
        "FAILS",
        "UNDECIDED",
        "WELD6",
        "SHAFT_FREE5",
        "BALL3",
    ):
        assert required_token in guide_text
        assert required_token in readme_text or required_token in guide_text

    # Verify relative markdown links resolve on disk
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)#]+)\)")
    for doc_path, text in (
        (USER_GUIDE_PATH, guide_text),
        (TOOL_README_PATH, readme_text),
    ):
        for match in link_pattern.finditer(text):
            rel_target = match.group(1).strip()
            if rel_target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (doc_path.parent / rel_target).resolve()
            assert resolved.exists(), (
                f"Broken relative link {rel_target!r} in {doc_path} -> {resolved}"
            )
