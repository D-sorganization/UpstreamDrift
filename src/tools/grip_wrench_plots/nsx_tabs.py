"""PyQt6 Loop Null-Space Exploration (NSX) tabs and QThreadPool worker (NSX-43, #12575).

Renders the framework-free NSX plot model across four tabs:
- ``Invariants``
- ``Envelopes`` (band-first with ``DETERMINED_SIGN``, ``INDETERMINATE``,
  ``UNBOUNDED`` segments, ``binding_hypotheses``, and overlaid representative)
- ``Hand Split`` (grip-frame coordinates ``s1..s6``)
- ``Diagnostics`` (closure rank, condition number, Chebyshev radius, admission residuals)

All mathematical computation lives in the backend/plot model; this module performs
pure presentation, preset selection from ``loop_nullspace_registry.json``, and
cooperative background job execution via ``QThreadPool`` / ``QRunnable``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import json
import os
from pathlib import Path
import threading
from typing import Any

os.environ.setdefault("QT_API", "pyqt6")

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

__all__ = [
    "HELP_GUIDE_REL_PATH",
    "NSXAnalysisWorker",
    "NSXPlotTab",
    "NSXWorkerSignals",
    "adapt_plot_model_for_preset",
    "load_golden_plot_model",
    "load_nsx_presets",
    "validate_plot_model_dict",
]

HELP_GUIDE_REL_PATH = "docs/user_guide/loop_nullspace_exploration.md"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[3]


def load_nsx_presets(registry_path: Path | None = None) -> list[dict[str, Any]]:
    """Load verified NSX presets from ``src/config/loop_nullspace_registry.json``."""
    target = registry_path or (
        _repo_root() / "src" / "config" / "loop_nullspace_registry.json"
    )
    raw = json.loads(target.read_text(encoding="utf-8"))
    presets_raw = raw.get("presets", [])
    if not isinstance(presets_raw, list) or not presets_raw:
        raise ValueError("loop_nullspace_registry.json must contain non-empty presets")
    return [dict(item) for item in presets_raw]


def load_golden_plot_model() -> dict[str, Any]:
    """Load the cross-UI golden plot model fixture dictionary."""
    root = _repo_root()
    shared_path = (
        root
        / "src"
        / "shared"
        / "python"
        / "loop_nullspace"
        / "golden_plot_model_v1.json"
    )
    local_path = Path(__file__).resolve().parent / "golden_plot_model_v1.json"
    source = shared_path if shared_path.is_file() else local_path
    data = json.loads(source.read_text(encoding="utf-8"))
    return validate_plot_model_dict(data)


def validate_plot_model_dict(plot_model: Mapping[str, Any]) -> dict[str, Any]:
    """Validate NSX plot model structure and enforce band-first DbC invariant.

    Raises:
        ValueError: If required keys are missing or if an allocation
            representative appears without its corresponding envelope band.
    """
    if not isinstance(plot_model, Mapping):
        raise ValueError("plot_model must be a mapping")
    panels = plot_model.get("panels")
    time_s = plot_model.get("time_s")
    if not isinstance(panels, Mapping) or not isinstance(time_s, Sequence):
        raise ValueError("plot_model requires 'panels' mapping and 'time_s' sequence")

    env_ids: set[str] = set()
    for panel_key in ("envelopes", "hand_split"):
        panel_obj = panels.get(panel_key, {})
        if isinstance(panel_obj, Mapping):
            for band in panel_obj.get("envelope_series", ()):
                qid = band.get("quantity_id")
                if qid:
                    env_ids.add(str(qid))

    alloc_panel = panels.get("allocation_representative", {})
    if isinstance(alloc_panel, Mapping):
        for rep_line in alloc_panel.get("line_series", ()):
            ref_id = rep_line.get("envelope_ref_id") or rep_line.get("quantity_id")
            if not ref_id or str(ref_id) not in env_ids:
                raise ValueError(
                    f"Allocation representative {ref_id!r} cannot be rendered "
                    "without its corresponding envelope band"
                )
    return dict(plot_model)


def _update_band_for_preset(
    band: dict[str, Any], preset_name: str, preset_hyps: list[str]
) -> None:
    if preset_name == "envelope_only":
        band["representative"] = None
        band["render_mode"] = "ENVELOPE_BAND"
    else:
        if band.get("representative") is None:
            band["representative"] = list(band.get("chebyshev_center", []))
        band["render_mode"] = "BAND_WITH_REPRESENTATIVE"
    hyps_str = ", ".join(preset_hyps)
    band["badge_text"] = f"PRIOR_CONDITIONED [{preset_name}: {hyps_str}]"


def adapt_plot_model_for_preset(
    base_model: Mapping[str, Any], preset_name: str
) -> dict[str, Any]:
    """Return a validated plot model configured for ``preset_name``."""
    presets = {p["name"]: p for p in load_nsx_presets()}
    if preset_name not in presets:
        raise ValueError(f"Unknown NSX preset: {preset_name!r}")
    preset_hyps = [str(h) for h in presets[preset_name].get("hypotheses", [])]
    adapted: dict[str, Any] = copy.deepcopy(dict(base_model))
    adapted["preset_name"] = preset_name
    panels = adapted.get("panels", {})

    for panel_key in ("envelopes", "hand_split"):
        panel = panels.get(panel_key, {})
        for band in panel.get("envelope_series", []):
            _update_band_for_preset(band, preset_name, preset_hyps)

    alloc_panel = panels.get("allocation_representative", {})
    if preset_name == "envelope_only":
        alloc_panel["line_series"] = []
        alloc_panel["envelope_series"] = []
    else:
        hyps_str = ", ".join(preset_hyps)
        for line in alloc_panel.get("line_series", []):
            line["badge_text"] = f"PRIOR_CONDITIONED [{preset_name}: {hyps_str}]"
            line["render_mode"] = "BAND_WITH_REPRESENTATIVE"
        for band in alloc_panel.get("envelope_series", []):
            _update_band_for_preset(band, preset_name, preset_hyps)
    return validate_plot_model_dict(adapted)


class NSXWorkerSignals(QObject):
    """Qt signals emitted by :class:`NSXAnalysisWorker`."""

    progress = pyqtSignal(dict)
    completed = pyqtSignal(dict)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)


class NSXAnalysisWorker(QRunnable):
    """Cancellable ``QThreadPool`` job worker for NSX plot-model computation."""

    def __init__(
        self,
        preset_name: str = "capacity_min_effort",
        *,
        analyzer: Callable[[str], dict[str, Any]] | None = None,
        defer_completion: bool = False,
    ) -> None:
        super().__init__()
        if not preset_name or not str(preset_name).strip():
            raise ValueError("preset_name must be a non-empty string")
        self.preset_name = str(preset_name)
        self._analyzer = analyzer
        self._defer_completion = bool(defer_completion)
        self._cancel_event = threading.Event()
        self.signals = NSXWorkerSignals()

    def cancel(self) -> None:
        """Request cooperative cancellation of the background NSX job."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested."""
        return self._cancel_event.is_set()

    def _compute_model(self) -> dict[str, Any]:
        if self._analyzer is not None:
            return validate_plot_model_dict(self._analyzer(self.preset_name))
        return adapt_plot_model_for_preset(load_golden_plot_model(), self.preset_name)

    def run(self) -> None:
        """Execute partial-invariants emission followed by full plot-model solve."""
        if self.is_cancelled():
            self.signals.cancelled.emit(self.preset_name)
            return
        try:
            model = self._compute_model()
            panels = model.get("panels", {})
            inv_panel = panels.get("invariants", {})
            partial_payload = {
                "status": "running",
                "preset_name": self.preset_name,
                "time_s": list(model.get("time_s", [])),
                "invariants": dict(inv_panel),
            }
            self.signals.progress.emit(partial_payload)
            if self._defer_completion:
                self._cancel_event.wait(0.25)
            if self.is_cancelled():
                self.signals.cancelled.emit(self.preset_name)
                return
            self.signals.completed.emit(model)
        except (ValueError, RuntimeError, TypeError, KeyError, OSError) as exc:
            self.signals.failed.emit(str(exc))


def _clean_floats(values: Sequence[Any]) -> list[float]:
    return [float("nan") if v is None else float(v) for v in values]


def _draw_invariants_panel(
    figure: Figure, panel: Mapping[str, Any], time_s: Sequence[float]
) -> tuple[str, str]:
    figure.clear()
    lines = list(panel.get("line_series", ()))
    if not lines or not time_s:
        return ("UNAVAILABLE", "No invariant series loaded")
    ax = figure.add_subplot(111)
    t_vals = [float(t) for t in time_s]
    labels: list[str] = []
    badge = str(lines[0].get("badge_text", "MODEL_CONDITIONED_INVARIANT"))
    for series in lines:
        y_vals = _clean_floats(series.get("values", ()))
        lbl = f"{series.get('label', series.get('quantity_id'))} ({series.get('units', '')})"
        ax.plot(t_vals, y_vals, label=lbl, linewidth=1.6)
        labels.append(str(series.get("quantity_id", "")))
    ax.set_xlabel("Time (s)")
    ax.set_title(str(panel.get("title", "Allocation-Invariant Quantities")))
    ax.legend(loc="best", fontsize=7)
    summary = f"Rendered {len(lines)} invariant series: {', '.join(labels)}"
    return (badge, summary)


def _shade_envelope_segments(
    ax: Any,
    t_vals: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    segments: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    classes_seen: set[str] = set()
    hyps_seen: set[str] = set()
    for seg in segments:
        disp_cls = str(seg.get("display_class", "DETERMINED_SIGN"))
        classes_seen.add(disp_cls)
        for hyp in seg.get("binding_hypotheses", ()):
            hyps_seen.add(str(hyp))
        s_idx = max(0, int(seg.get("start_index", 0)))
        e_idx = min(len(t_vals) - 1, int(seg.get("end_index", 0)))
        if s_idx <= e_idx and t_vals:
            t_seg = t_vals[s_idx : e_idx + 1]
            l_seg = lower[s_idx : e_idx + 1]
            u_seg = upper[s_idx : e_idx + 1]
            color = "#f59e0b" if disp_cls == "INDETERMINATE" else "#38bdf8"
            hatch = "//" if disp_cls == "UNBOUNDED" else None
            ax.fill_between(t_seg, l_seg, u_seg, color=color, alpha=0.25, hatch=hatch)
    return (classes_seen, hyps_seen)


def _draw_envelope_panel(
    figure: Figure, panel: Mapping[str, Any], time_s: Sequence[float]
) -> tuple[str, str]:
    figure.clear()
    bands = list(panel.get("envelope_series", ()))
    if not bands or not time_s:
        return ("UNAVAILABLE", "No envelope series loaded")
    ax = figure.add_subplot(111)
    t_vals = [float(t) for t in time_s]
    badge = str(bands[0].get("badge_text", "PRIOR_CONDITIONED"))
    all_classes: set[str] = set()
    all_hyps: set[str] = set()
    rep_count = 0
    for band in bands:
        low = _clean_floats(band.get("lower", ()))
        high = _clean_floats(band.get("upper", ()))
        lbl = str(band.get("label", band.get("quantity_id", "band")))
        ax.fill_between(t_vals, low, high, alpha=0.2, label=f"{lbl} [band]")
        cls_set, hyp_set = _shade_envelope_segments(
            ax, t_vals, low, high, band.get("segments", ())
        )
        all_classes.update(cls_set)
        all_hyps.update(hyp_set)
        rep = band.get("representative")
        if isinstance(rep, Sequence):
            rep_vals = _clean_floats(rep)
            ax.plot(
                t_vals, rep_vals, linestyle="--", linewidth=1.5, label=f"{lbl} [rep]"
            )
            rep_count += 1
    ax.set_xlabel("Time (s)")
    ax.set_title(str(panel.get("title", "Feasible Envelopes")))
    cls_txt = ", ".join(sorted(all_classes)) or "NONE"
    hyp_txt = ", ".join(sorted(all_hyps)) or "NONE"
    summary = (
        f"Bands: {len(bands)} | Representatives overlaid: {rep_count} | "
        f"Segments: {cls_txt} | Binding: {hyp_txt}"
    )
    return (badge, summary)


def _draw_diagnostics_panel(
    figure: Figure, panel: Mapping[str, Any], time_s: Sequence[float]
) -> tuple[str, str]:
    figure.clear()
    metrics = panel.get("metrics", {})
    if not isinstance(metrics, Mapping) or not metrics or not time_s:
        return ("UNAVAILABLE", "No diagnostics loaded")
    ax = figure.add_subplot(111)
    t_vals = [float(t) for t in time_s]
    for key in (
        "rank_jc",
        "condition_number_jc",
        "equality_nullity",
        "chebyshev_radius",
        "phi_scaled",
        "phidot_scaled",
        "phiddot_scaled",
    ):
        series = metrics.get(key)
        if isinstance(series, Sequence) and len(series) == len(t_vals):
            ax.plot(t_vals, _clean_floats(series), label=key)
    ax.set_xlabel("Time (s)")
    ax.set_title(str(panel.get("title", "Diagnostics")))
    ax.legend(loc="best", fontsize=7)
    rank_vals = metrics.get("rank_jc", [])
    cond_vals = metrics.get("condition_number_jc", [])
    badge = "MODEL_CONDITIONED_INVARIANT [diagnostics | rank & admission gates]"
    summary = (
        f"rank_jc={list(rank_vals)} | condition_number_jc={list(cond_vals)} | "
        f"keys={sorted(metrics.keys())}"
    )
    return (badge, summary)


class NSXPlotTab(QWidget):
    """Single NSX tab displaying an evidence badge, summary, and matplotlib canvas."""

    def __init__(self, panel_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.panel_key = panel_key
        layout = QVBoxLayout(self)
        self.badge_label = QLabel("Evidence badge: awaiting NSX plot model")
        self.badge_label.setWordWrap(True)
        self.summary_label = QLabel("No NSX data rendered")
        self.summary_label.setWordWrap(True)
        self.figure = Figure(figsize=(5.5, 4.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.badge_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.canvas, 1)

    def render_panel(self, panel: Mapping[str, Any], time_s: Sequence[float]) -> None:
        """Render ``panel`` onto this tab's figure and update evidence badge."""
        if self.panel_key == "invariants":
            badge, summary = _draw_invariants_panel(self.figure, panel, time_s)
        elif self.panel_key in {"envelopes", "hand_split"}:
            badge, summary = _draw_envelope_panel(self.figure, panel, time_s)
        else:
            badge, summary = _draw_diagnostics_panel(self.figure, panel, time_s)
        self.badge_label.setText(badge)
        self.summary_label.setText(summary)
        self.canvas.draw_idle()

    def clear_panel(self) -> None:
        """Reset tab figure and labels."""
        self.figure.clear()
        self.badge_label.setText("Evidence badge: awaiting NSX plot model")
        self.summary_label.setText("No NSX data rendered")
        self.canvas.draw_idle()
