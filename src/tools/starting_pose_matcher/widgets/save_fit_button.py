"""Save-fit button + JSON serialiser for the starting-pose matcher.

Slice 3/3 of issue #4707. Encapsulates a "Save fit" button that takes
the :class:`CanonicalFitResult` produced by
:class:`~src.tools.starting_pose_matcher.widgets.run_fit_button.RunFitButton`
and writes a JSON document containing theta, residuals, engine name,
engine version, and the source-file sha256 hash.

The serialiser :func:`serialize_fit_result` is a pure function so it
can be tested without instantiating any Qt widget. The widget itself
is a thin :class:`QWidget` that owns a button + status label and
calls into :class:`QFileDialog` when clicked.

Design notes:
* DRY — the JSON shape lives in one helper, used by both the widget
  and the headless test suite.
* DbC — :func:`serialize_fit_result` validates its inputs and raises
  :class:`ValueError` / :class:`TypeError` with descriptive messages.
* LoD — the widget never reaches into ``RunFitButton`` internals; it
  receives a result + engine name explicitly via :meth:`set_result`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..session_schema import SESSION_SCHEMA_VERSION

__all__ = [
    "FIT_RESULT_SCHEMA_VERSION",
    "SaveFitButton",
    "compute_source_file_sha256",
    "serialize_fit_result",
]

# Schema version for the save-fit JSON document. Independent of the
# session-schema version so the two can evolve separately.
FIT_RESULT_SCHEMA_VERSION: int = 1


def _to_float_list(values: Any, *, field: str) -> list[float]:
    """Coerce a 1D array-like to a plain ``list[float]`` for JSON.

    Accepts ``None`` (returns ``[]``), numpy arrays, tuples, lists, and
    any iterable of numbers. Raises :class:`TypeError` for unsupported
    inputs so callers see a clear error rather than a silent JSON error.
    """
    if values is None:
        return []
    # numpy arrays and similar — use .tolist() if available
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        try:
            flat = tolist()
        except Exception as exc:  # noqa: BLE001 — surface conversion failures
            raise TypeError(f"{field}: tolist() failed: {exc}") from exc
    elif isinstance(values, Iterable):
        flat = list(values)
    else:
        raise TypeError(
            f"{field} must be array-like or None, got {type(values).__name__}"
        )
    out: list[float] = []
    for x in flat:
        try:
            out.append(float(x))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field}: cannot coerce {x!r} to float") from exc
    return out


def compute_source_file_sha256(path: str | Path | None) -> str | None:
    """Return the sha256 hex digest of ``path`` or ``None`` if unset.

    A missing or unreadable path raises :class:`FileNotFoundError` so
    the caller can surface a precise error to the user instead of
    silently writing a bad provenance hash.
    """
    if path is None or path == "":
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"source file not found: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or mapping key, falling back to ``default``."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def serialize_fit_result(
    result: Any,
    *,
    engine_name: str,
    source_file: str | Path | None = None,
    residuals: Any = None,
) -> dict[str, Any]:
    """Convert a :class:`CanonicalFitResult` to a JSON-ready dict.

    Parameters
    ----------
    result:
        A :class:`CanonicalFitResult` (or any compatible object/mapping
        exposing ``theta_optimal``, ``engine_version``, ``method``,
        ``solver_status``, ``final_cost``, ``final_rmse_m``, ``message``,
        ``timestamp_utc``).
    engine_name:
        The registry key the GUI used to look up the provider. This
        complements ``engine_version``, which the engine itself reports.
    source_file:
        Optional path to the C3D / mocap file that produced the target.
        When set, its sha256 hex digest is embedded under ``source.sha256``.
    residuals:
        Optional per-frame residuals array. Falls back to
        ``result.meta.get('residuals')`` or an empty list.

    Returns
    -------
    dict
        JSON-serialisable mapping with the schema documented in
        :data:`FIT_RESULT_SCHEMA_VERSION`.

    Raises
    ------
    ValueError
        If ``result`` is ``None`` or ``engine_name`` is empty.
    TypeError
        If ``theta_optimal`` / ``residuals`` cannot be coerced to a
        list of floats.
    """
    if result is None:
        raise ValueError("result must not be None")
    if not isinstance(engine_name, str) or not engine_name:
        raise ValueError("engine_name must be a non-empty string")

    theta = _get(result, "theta_optimal")
    if theta is None:
        # Legacy fallback: some tests use plain dicts with "theta".
        theta = _get(result, "theta")
    theta_list = _to_float_list(theta, field="theta_optimal")

    if residuals is None:
        meta = _get(result, "meta") or {}
        if isinstance(meta, dict):
            residuals = meta.get("residuals")
    residuals_list = _to_float_list(residuals, field="residuals")

    sha256 = compute_source_file_sha256(source_file)

    doc: dict[str, Any] = {
        "schema_version": FIT_RESULT_SCHEMA_VERSION,
        "session_schema_version": SESSION_SCHEMA_VERSION,
        "engine": {
            "name": engine_name,
            "version": str(_get(result, "engine_version", "") or ""),
            "method": str(_get(result, "method", "") or ""),
        },
        "fit": {
            "theta_optimal": theta_list,
            "residuals": residuals_list,
            "final_cost": _coerce_float_or_none(_get(result, "final_cost")),
            "final_rmse_m": _coerce_float_or_none(_get(result, "final_rmse_m")),
            "solver_status": str(_get(result, "solver_status", "") or ""),
            "iterations": _coerce_int_or_none(_get(result, "iterations")),
            "n_evaluations": _coerce_int_or_none(_get(result, "n_evaluations")),
            "wall_clock_s": _coerce_float_or_none(_get(result, "wall_clock_s")),
            "message": str(_get(result, "message", "") or ""),
        },
        "provenance": {
            "git_commit": str(_get(result, "git_commit", "") or ""),
            "target_hash": str(_get(result, "target_hash", "") or ""),
            "timestamp_utc": str(_get(result, "timestamp_utc", "") or ""),
        },
        "source": {
            "path": str(source_file) if source_file else None,
            "sha256": sha256,
        },
    }
    return doc


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_fit_result_json(
    path: str | Path,
    result: Any,
    *,
    engine_name: str,
    source_file: str | Path | None = None,
    residuals: Any = None,
) -> Path:
    """Serialise ``result`` and write the JSON document to ``path``.

    Returns the resolved :class:`Path` written. Raises the same
    exceptions as :func:`serialize_fit_result`, plus :class:`OSError`
    if the file cannot be written.
    """
    doc = serialize_fit_result(
        result,
        engine_name=engine_name,
        source_file=source_file,
        residuals=residuals,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return out


class SaveFitButton(QWidget):
    """Self-contained Save-fit button + status widget.

    Wire it up by calling :meth:`set_result` whenever a fit completes
    (typically from ``RunFitButton.finished``) and :meth:`set_source_file`
    when the user loads a new C3D / xlsx target file. The button is
    disabled until both a result and an engine name are available.
    """

    saved = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: Any = None
        self._engine: str = ""
        self._source_file: str | None = None
        self._default_dir: Path = Path.cwd()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = QHBoxLayout()
        self.btn_save = QPushButton("Save fit")
        self.btn_save.setObjectName("primary")
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip(
            "Save the most recent CanonicalFitResult to a JSON file "
            "(theta, residuals, engine, engine_version, source sha256)."
        )
        self.btn_save.clicked.connect(self.save_to_file_dialog)
        row.addWidget(self.btn_save)
        layout.addLayout(row)
        self.lbl_status = QLabel("No fit to save yet.")
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    # ------------------------------------------------------------------ #
    # Public input setters                                               #
    # ------------------------------------------------------------------ #

    def set_result(self, result: Any, *, engine_name: str = "") -> None:
        """Cache a new fit result + engine name and refresh the button."""
        self._result = result
        if engine_name:
            self._engine = engine_name
        self._refresh_enabled()
        if result is not None:
            self.lbl_status.setText("Fit ready to save.")

    def set_engine_name(self, engine_name: str) -> None:
        """Cache the engine-registry key used for this fit."""
        self._engine = engine_name or ""
        self._refresh_enabled()

    def set_source_file(self, path: str | Path | None) -> None:
        """Cache the C3D / xlsx source path for sha256 provenance."""
        self._source_file = str(path) if path else None

    def set_default_dir(self, path: str | Path) -> None:
        """Where the file dialog opens by default."""
        self._default_dir = Path(path)

    def _refresh_enabled(self) -> None:
        ready = self._result is not None and bool(self._engine)
        self.btn_save.setEnabled(ready)

    # ------------------------------------------------------------------ #
    # Save action                                                        #
    # ------------------------------------------------------------------ #

    def save_to_path(self, path: str | Path) -> Path:
        """Headless-friendly save: write JSON directly to ``path``."""
        if self._result is None:
            raise ValueError("no fit result to save; call set_result(...) first")
        if not self._engine:
            raise ValueError("no engine name; call set_engine_name(...) first")
        out = write_fit_result_json(
            path,
            self._result,
            engine_name=self._engine,
            source_file=self._source_file,
        )
        self.lbl_status.setText(f"Saved fit to {out.name}.")
        self.saved.emit(str(out))
        return out

    def save_to_file_dialog(self) -> Path | None:
        """Prompt the user for a path and write the JSON document."""
        if self._result is None or not self._engine:
            return None
        default_name = f"fit_{self._engine}.json"
        default_path = str(self._default_dir / default_name)
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            "Save fit result",
            default_path,
            "JSON files (*.json);;All files (*)",
        )
        if not chosen:
            self.lbl_status.setText("Save cancelled.")
            return None
        try:
            return self.save_to_path(chosen)
        except (OSError, ValueError, TypeError, FileNotFoundError) as exc:
            msg = f"Save failed: {type(exc).__name__}: {exc}"
            self.lbl_status.setText(msg)
            self.failed.emit(msg)
            return None
