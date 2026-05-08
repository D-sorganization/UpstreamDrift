"""Data-sources panel for the starting-pose matcher (issue #4480).

A self-contained Qt widget that lets the user pick any combination of:

* a club source (xlsx / .mat / c3d) — toggleable between *Club only* and
  *Club + ball*
* a body-markers source (c3d) — with a marker-set combo

Plus a shared ``AlignOptions`` row (sample rate, duration, time-alignment).

Kept in its own module so ``gui.py`` stays close to its pre-existing
1200-line budget instead of growing further; the main window simply
embeds this panel as one of its left-column sections.

The panel emits a single ``targets_changed`` signal that carries either
``None`` (no slots loaded yet) or a fully-validated ``MultiSourceTarget``.
The owning window decides what to do with it (rebuild plots, recompute
costs, etc.).
"""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.motion_matching.load_club_target import load_club_target
from src.shared.python.motion_matching.multi_source_target import MultiSourceTarget
from src.shared.python.motion_matching.target import AlignOptions, ClubTarget
from src.tools.starting_pose_matcher.session_schema import (
    DEFAULT_BODY_MARKER_SET,
    DEFAULT_BODY_MARKER_SETS,
    AlignOptionsBlock,
    BodySourceBlock,
    ClubSourceBlock,
    DataSourcesBlock,
    default_data_sources,
)

logger = logging.getLogger(__name__)


# Generic file-dialog filters (no vendor / lab / person names per #4480).
CLUB_FILE_FILTER = "Mocap club data (*.xlsx *.xlsm *.mat *.c3d)"
BODY_FILE_FILTER = "Mocap body data (*.c3d)"


def _safe_load_body_target(
    path: str,
    *,
    opts: AlignOptions,
    impact_source: ClubTarget | None,
) -> Any:
    """Best-effort import of ``load_body_target``.

    The body-target loader (issue #4477 / #4478) may not have landed
    yet on ``main``. If absent, raise a clear ``RuntimeError`` so the
    UI can surface a friendly message instead of crashing.
    """
    try:
        from src.shared.python.motion_matching.load_body_target import (  # type: ignore[import-not-found]
            load_body_target,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Body-marker loader not available in this build. "
            "It will be enabled when issues #4477 / #4478 land."
        ) from exc
    return load_body_target(Path(path), opts=opts, impact_source=impact_source)


def _safe_extract_ball_impact(club: ClubTarget) -> Any:
    """Best-effort import of ``extract_ball_impact_from_clubtarget``."""
    try:
        from src.shared.python.motion_matching.club_ball_target import (  # type: ignore[import-not-found]
            extract_ball_impact_from_clubtarget,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Ball-impact extractor not available in this build. "
            "It will be enabled when issue #4479 lands."
        ) from exc
    return extract_ball_impact_from_clubtarget(club)


class DataSourcesPanel(QGroupBox):
    """The ``Data sources`` group-box.

    Public surface:
        targets_changed   -- pyqtSignal[object]:  emits MultiSourceTarget | None
        current_targets() -- returns the latest MultiSourceTarget | None
        snapshot()        -- returns DataSourcesBlock for session save
        restore()         -- restores from a DataSourcesBlock (no auto-load)
        align_options()   -- builds the live AlignOptions from the spinboxes
    """

    targets_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Data sources", parent)

        # Loaded targets (cached for toggling Club ↔ Club+Ball without re-load)
        self._club_target: ClubTarget | None = None  # plain ClubTarget cache
        self._club_path: str | None = None
        self._body_target: Any = None
        self._body_path: str | None = None
        self._latest_targets: MultiSourceTarget | None = None

        self._build_ui()

    # ----------------------------------------------------------------- UI ---

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 14, 8, 8)
        outer.setSpacing(6)

        # ---- Club row ---------------------------------------------------
        club_grid = QGridLayout()
        club_grid.setVerticalSpacing(2)
        self.cb_club = QCheckBox("Club")
        self.cb_club.setObjectName("cb_club")
        self.cb_club.setToolTip(
            "Load a club-trajectory source (xlsx / .mat / c3d). "
            "Enables the club cost terms downstream."
        )
        self.btn_club_browse = QPushButton("Browse…")
        self.btn_club_browse.setObjectName("btn_club_browse")
        self.lbl_club_path = QLabel("(no file)")
        self.lbl_club_path.setObjectName("lbl_club_path")
        self.lbl_club_path.setStyleSheet("color:#94a3b8;")
        club_grid.addWidget(self.cb_club, 0, 0)
        club_grid.addWidget(self.btn_club_browse, 0, 1)
        club_grid.addWidget(self.lbl_club_path, 0, 2)
        club_grid.setColumnStretch(2, 1)

        # Sub-radios: Club only vs Club + ball
        self.rb_club_only = QRadioButton("Club only")
        self.rb_club_only.setObjectName("rb_club_only")
        self.rb_club_only.setChecked(True)
        self.rb_club_ball = QRadioButton("Club + ball")
        self.rb_club_ball.setObjectName("rb_club_ball")
        self._club_mode_group = QButtonGroup(self)
        self._club_mode_group.addButton(self.rb_club_only)
        self._club_mode_group.addButton(self.rb_club_ball)
        sub = QHBoxLayout()
        sub.setContentsMargins(20, 0, 0, 0)
        sub.addWidget(self.rb_club_only)
        sub.addWidget(self.rb_club_ball)
        sub.addStretch(1)
        club_grid.addLayout(sub, 1, 0, 1, 3)
        outer.addLayout(club_grid)

        # ---- Body row ---------------------------------------------------
        body_grid = QGridLayout()
        body_grid.setVerticalSpacing(2)
        self.cb_body = QCheckBox("Body markers")
        self.cb_body.setObjectName("cb_body")
        self.cb_body.setToolTip(
            "Load a full-body anatomical-marker source (c3d). "
            "Enables the body cost terms downstream."
        )
        self.btn_body_browse = QPushButton("Browse…")
        self.btn_body_browse.setObjectName("btn_body_browse")
        self.lbl_body_path = QLabel("(no file)")
        self.lbl_body_path.setObjectName("lbl_body_path")
        self.lbl_body_path.setStyleSheet("color:#94a3b8;")
        body_grid.addWidget(self.cb_body, 0, 0)
        body_grid.addWidget(self.btn_body_browse, 0, 1)
        body_grid.addWidget(self.lbl_body_path, 0, 2)
        body_grid.setColumnStretch(2, 1)

        # Marker-set combo
        self.combo_marker_set = QComboBox()
        self.combo_marker_set.setObjectName("combo_marker_set")
        self.combo_marker_set.addItems(list(DEFAULT_BODY_MARKER_SETS))
        self.combo_marker_set.setCurrentText(DEFAULT_BODY_MARKER_SET)
        ms_row = QHBoxLayout()
        ms_row.setContentsMargins(20, 0, 0, 0)
        ms_row.addWidget(QLabel("Marker set:"))
        ms_row.addWidget(self.combo_marker_set)
        ms_row.addStretch(1)
        body_grid.addLayout(ms_row, 1, 0, 1, 3)
        outer.addLayout(body_grid)

        # ---- Time alignment + sample rate / duration --------------------
        align_grid = QGridLayout()
        align_grid.setVerticalSpacing(4)
        align_grid.addWidget(QLabel("Time alignment:"), 0, 0)
        self.rb_align_impact = QRadioButton("Impact-aligned")
        self.rb_align_impact.setObjectName("rb_align_impact")
        self.rb_align_impact.setChecked(True)
        self.rb_align_address = QRadioButton("Address-aligned")
        self.rb_align_address.setObjectName("rb_align_address")
        self._align_group = QButtonGroup(self)
        self._align_group.addButton(self.rb_align_impact)
        self._align_group.addButton(self.rb_align_address)
        ta_row = QHBoxLayout()
        ta_row.addWidget(self.rb_align_impact)
        ta_row.addWidget(self.rb_align_address)
        ta_row.addStretch(1)
        align_grid.addLayout(ta_row, 0, 1)

        align_grid.addWidget(QLabel("Sample rate (Hz):"), 1, 0)
        self.spin_sample_rate = QSpinBox()
        self.spin_sample_rate.setObjectName("spin_sample_rate")
        self.spin_sample_rate.setRange(50, 10000)
        self.spin_sample_rate.setValue(1000)
        self.spin_sample_rate.setSingleStep(100)
        align_grid.addWidget(self.spin_sample_rate, 1, 1)

        align_grid.addWidget(QLabel("Duration (s):"), 2, 0)
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setObjectName("spin_duration")
        self.spin_duration.setRange(0.05, 5.0)
        self.spin_duration.setValue(0.300)
        self.spin_duration.setDecimals(3)
        self.spin_duration.setSingleStep(0.05)
        align_grid.addWidget(self.spin_duration, 2, 1)
        outer.addLayout(align_grid)

        # ---- wire signals ----------------------------------------------
        self.btn_club_browse.clicked.connect(self._on_browse_club)
        self.btn_body_browse.clicked.connect(self._on_browse_body)
        self.cb_club.toggled.connect(self._on_club_enabled)
        self.cb_body.toggled.connect(self._on_body_enabled)
        self.rb_club_only.toggled.connect(self._on_club_mode_changed)
        self.rb_club_ball.toggled.connect(self._on_club_mode_changed)
        # Re-emit when align spinboxes change so callers can refresh
        self.spin_sample_rate.valueChanged.connect(self._emit_targets)
        self.spin_duration.valueChanged.connect(self._emit_targets)
        self.rb_align_impact.toggled.connect(self._emit_targets)

    # -------------------------------------------------------- public API ---

    def current_targets(self) -> MultiSourceTarget | None:
        """Return the most recent successfully-built ``MultiSourceTarget``."""
        return self._latest_targets

    def align_options(self) -> AlignOptions:
        """Build a live ``AlignOptions`` from the spinboxes / radio."""
        alignment = "impact" if self.rb_align_impact.isChecked() else "address"
        return AlignOptions(
            sample_rate_hz=float(self.spin_sample_rate.value()),
            simulation_time_s=float(self.spin_duration.value()),
            time_alignment=alignment,  # type: ignore[arg-type]
        )

    def snapshot(self) -> DataSourcesBlock:
        """Snapshot the panel state for session-JSON save."""
        return DataSourcesBlock(
            club=ClubSourceBlock(
                enabled=self.cb_club.isChecked(),
                file_path=self._club_path,
                include_ball=self.rb_club_ball.isChecked(),
            ),
            body=BodySourceBlock(
                enabled=self.cb_body.isChecked(),
                file_path=self._body_path,
                marker_set=self.combo_marker_set.currentText(),
            ),
            align=AlignOptionsBlock(
                sample_rate_hz=float(self.spin_sample_rate.value()),
                simulation_time_s=float(self.spin_duration.value()),
                time_alignment=(
                    "impact" if self.rb_align_impact.isChecked() else "address"
                ),
            ),
        )

    def restore(self, block: DataSourcesBlock | None) -> None:
        """Restore widget state from a session block.

        Does NOT auto-load files — the user can hit Browse again or
        the calling window can drive the load.  This avoids surprising
        side-effects when a session is loaded from a different machine
        whose paths no longer resolve.
        """
        # Clear cached targets to prevent stale data when restoring a
        # session with different (or no) source files.
        self._club_target = None
        self._body_target = None

        b = block or default_data_sources()
        self.cb_club.setChecked(b.club.enabled)
        self._club_path = b.club.file_path
        self.lbl_club_path.setText(
            Path(b.club.file_path).name if b.club.file_path else "(no file)"
        )
        if b.club.include_ball:
            self.rb_club_ball.setChecked(True)
        else:
            self.rb_club_only.setChecked(True)

        self.cb_body.setChecked(b.body.enabled)
        self._body_path = b.body.file_path
        self.lbl_body_path.setText(
            Path(b.body.file_path).name if b.body.file_path else "(no file)"
        )
        idx = self.combo_marker_set.findText(b.body.marker_set)
        if idx >= 0:
            self.combo_marker_set.setCurrentIndex(idx)

        self.spin_sample_rate.setValue(int(round(b.align.sample_rate_hz)))
        self.spin_duration.setValue(float(b.align.simulation_time_s))
        if b.align.time_alignment == "address":
            self.rb_align_address.setChecked(True)
        else:
            self.rb_align_impact.setChecked(True)

    # --------------------------------------------------------- handlers ---

    def _on_club_enabled(self, _checked: bool) -> None:
        self._emit_targets()

    def _on_body_enabled(self, _checked: bool) -> None:
        self._emit_targets()

    def _on_club_mode_changed(self, _checked: bool) -> None:
        # Only react on the "becomes-checked" transition to avoid double-fires.
        if not (self.rb_club_only.isChecked() or self.rb_club_ball.isChecked()):
            return
        self._emit_targets()

    def _on_browse_club(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select club-trajectory file",
            self._club_path or "",
            CLUB_FILE_FILTER,
        )
        if not path:
            return
        self._load_club(path)

    def _on_browse_body(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select body-marker C3D file",
            self._body_path or "",
            BODY_FILE_FILTER,
        )
        if not path:
            return
        self._load_body(path)

    # ----------------------------------------------------------- loaders ---

    def _load_club(self, path: str) -> None:
        try:
            ct = load_club_target(Path(path), opts=self.align_options())
        except Exception as exc:  # noqa: BLE001 — surfaced as a Qt warning.
            QMessageBox.warning(
                self,
                "Failed to load club source",
                f"Could not parse {Path(path).name}:\n{exc}",
            )
            logger.warning("Club load failed: %s", exc)
            return
        self._club_target = ct
        self._club_path = path
        self.lbl_club_path.setText(Path(path).name)
        if not self.cb_club.isChecked():
            self.cb_club.setChecked(True)  # auto-enable the row
        self._emit_targets()

    def _load_body(self, path: str) -> None:
        try:
            body = _safe_load_body_target(
                path,
                opts=self.align_options(),
                impact_source=self._club_target,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Failed to load body source",
                f"Could not load body markers from {Path(path).name}:\n{exc}",
            )
            logger.warning("Body load failed: %s", exc)
            return
        self._body_target = body
        self._body_path = path
        self.lbl_body_path.setText(Path(path).name)
        if not self.cb_body.isChecked():
            self.cb_body.setChecked(True)
        self._emit_targets()

    # ----------------------------------------------- target assembly ---

    def _resolve_club_slot(self) -> Any:
        """Return the active club slot (``ClubTarget`` / ``ClubBallTarget`` / None).

        Toggling Club ↔ Club+Ball without re-loading is implemented by
        building a ``ClubBallTarget`` from the cached ``ClubTarget`` via
        ``extract_ball_impact_from_clubtarget``.
        """
        if not self.cb_club.isChecked() or self._club_target is None:
            return None
        if self.rb_club_only.isChecked():
            return self._club_target
        # Club + ball
        try:
            return _safe_extract_ball_impact(self._club_target)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Ball-impact extraction failed",
                f"Could not extract ball impact from club trajectory:\n{exc}",
            )
            logger.warning("Ball-impact extraction failed: %s", exc)
            # Fall back to plain club so the user keeps a working slot.
            return self._club_target

    def _resolve_body_slot(self) -> Any:
        if not self.cb_body.isChecked():
            return None
        return self._body_target

    def _emit_targets(self) -> None:
        club = self._resolve_club_slot()
        body = self._resolve_body_slot()
        if club is None and body is None:
            self._latest_targets = None
            self.targets_changed.emit(None)
            return
        try:
            mst = MultiSourceTarget(club=club, body=body)
        except ValueError as exc:
            # Validation rules: at-least-one slot, matching timegrids.
            QMessageBox.warning(
                self,
                "Mismatched target timegrids",
                str(exc),
            )
            logger.warning("MultiSourceTarget build failed: %s", exc)
            self._latest_targets = None
            self.targets_changed.emit(None)
            return
        except TypeError as exc:
            QMessageBox.warning(self, "Invalid target object", str(exc))
            logger.warning("MultiSourceTarget type error: %s", exc)
            self._latest_targets = None
            self.targets_changed.emit(None)
            return
        self._latest_targets = mst
        self.targets_changed.emit(mst)

    # ----------------------------------------------- test helpers ---

    def _force_set_club_target(self, club: ClubTarget | None, path: str | None) -> None:
        """Test-only helper: install a pre-built ClubTarget without going
        through the file dialog.  Avoids needing real files in unit tests.
        """
        self._club_target = club
        self._club_path = path
        self.lbl_club_path.setText(Path(path).name if path else "(no file)")
        with suppress(Exception):
            self.cb_club.setChecked(club is not None)
        self._emit_targets()

    def _force_set_body_target(self, body: Any, path: str | None) -> None:
        """Test-only helper: install a pre-built body target."""
        self._body_target = body
        self._body_path = path
        self.lbl_body_path.setText(Path(path).name if path else "(no file)")
        with suppress(Exception):
            self.cb_body.setChecked(body is not None)
        self._emit_targets()


def _try_clamp_signed_int(value: int, lo: int, hi: int) -> int:
    """Bound an integer.  Used by tests that drive spinboxes."""
    return max(lo, min(hi, int(value)))


# Convenience: a tiny utility used by tests to confirm a numpy
# array_equal contract holds between two targets that share a timegrid.
def shared_timegrid_ok(a: Any, b: Any) -> bool:
    """Return ``True`` iff ``a.time`` and ``b.time`` are exactly equal."""
    if a is None or b is None:
        return False
    ta, tb = getattr(a, "time", None), getattr(b, "time", None)
    if ta is None or tb is None:
        return False
    return ta.shape == tb.shape and np.array_equal(ta, tb)
