"""Match panel: which cameras, which observation set, which variant (#9797).

The user names a match, ticks the views to use, picks the observation set
and the source: *triangulate* (two or more views, the reconstruction) or
*image space* (any number of views, the model fitted to the 2-D keypoints
with cameras borrowed from another variant). :meth:`MatchPanel.selection`
is a plain dataclass the command builders consume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QWidget,
)

from src.motion_capture.variants import is_valid_name
from src.shared.python.core.contracts import require

from . import commands
from .session import SessionMedia

TRIANGULATE = "triangulate"
IMAGE_SPACE = "image_space"


@dataclass(frozen=True)
class MatchSelection:
    """What the Match panel asks for."""

    name: str
    views: tuple[str, ...]
    observation_set: str
    source: str  # TRIANGULATE | IMAGE_SPACE
    cameras_from: str = ""

    def __post_init__(self) -> None:
        require(is_valid_name(self.name), "invalid variant name", self.name)
        require(
            self.source in (TRIANGULATE, IMAGE_SPACE), "unknown source", self.source
        )
        if self.source == TRIANGULATE:
            require(
                not self.views or len(self.views) >= 2,
                "triangulation needs at least two views",
            )
        else:
            require(len(self.views) >= 1, "an image-space match needs a view")

    @property
    def image_space(self) -> bool:
        return self.source == IMAGE_SPACE


def reconstruct_args(
    session: Path,
    selection: MatchSelection,
    *,
    measurements: Sequence[str],
    cameras: Path | None,
    intrinsics: Path | None,
    exclude_joints: Sequence[str] = (),
) -> list[str]:
    """``rig reconstruct`` for the selection. Precondition: triangulate source."""
    require(not selection.image_space, "image-space matches use fit-model")
    return commands.reconstruct_command(
        session,
        measurements=measurements,
        cameras=cameras,
        intrinsics=intrinsics,
        exclude_joints=exclude_joints,
        variant=selection.name,
        views=selection.views,
        observations=selection.observation_set,
    )


def fit_model_args(
    session: Path, selection: MatchSelection, *, model: str, fit_lengths: bool
) -> list[str]:
    """``rig fit-model`` for the selection: image-space when the source says so."""
    return commands.fit_model_command(
        session,
        model=model,
        fit_lengths=fit_lengths,
        variant=selection.name,
        from_views=selection.views if selection.image_space else (),
        cameras_from=selection.cameras_from,
        observations=selection.observation_set,
    )


class MatchPanel(QGroupBox):
    """Variant name, view checkboxes, observation set and source."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Match", parent)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("variant name (blank = default match)")
        self.views_box = QWidget()
        self.views_layout = QHBoxLayout(self.views_box)
        self.views_layout.setContentsMargins(0, 0, 0, 0)
        self.view_checks: dict[str, QCheckBox] = {}
        self.set_combo = QComboBox()
        self.triangulate_radio = QRadioButton("triangulate (2+ views)")
        self.image_radio = QRadioButton("image space (1+ views, cameras from)")
        self.triangulate_radio.setChecked(True)
        self.cameras_combo = QComboBox()
        self.cameras_combo.setEnabled(False)
        self.image_radio.toggled.connect(self.cameras_combo.setEnabled)
        self.summary = QLabel("no session loaded")
        source = QHBoxLayout()
        source.addWidget(self.triangulate_radio)
        source.addWidget(self.image_radio)
        source.addWidget(self.cameras_combo, 1)
        form = QFormLayout(self)
        form.addRow("Variant", self.name_edit)
        form.addRow("Views", self.views_box)
        form.addRow("Observation set", self.set_combo)
        form.addRow("Source", source)
        form.addRow("Registered", self.summary)

    def load(self, media: SessionMedia | None) -> None:
        """Offer the session's views, sets and existing variants."""
        for check in self.view_checks.values():
            self.views_layout.removeWidget(check)
            check.deleteLater()
        self.view_checks = {}
        self.set_combo.clear()
        self.cameras_combo.clear()
        if media is None:
            self.summary.setText("no session loaded")
            return
        for view in media.views:
            check = QCheckBox(view.view)
            check.setChecked(True)
            self.views_layout.addWidget(check)
            self.view_checks[view.view] = check
        for name in media.observation_sets or ("observations",):
            self.set_combo.addItem(name)
        for variant in media.variants:
            if variant.has_reconstruction:
                self.cameras_combo.addItem(variant.label, variant.name)
        lines = [
            f"{v.label}: {','.join(v.views)} · {v.source.get('kind')}"
            + (" · model" if v.has_model_fit else "")
            for v in media.variants
        ]
        self.summary.setText("\n".join(lines) or "none yet")

    def set_selection(self, selection: MatchSelection) -> None:
        self.name_edit.setText(selection.name)
        for name, check in self.view_checks.items():
            check.setChecked(name in selection.views or not selection.views)
        index = self.set_combo.findText(selection.observation_set)
        if index >= 0:
            self.set_combo.setCurrentIndex(index)
        (
            self.image_radio if selection.image_space else self.triangulate_radio
        ).setChecked(True)
        index = self.cameras_combo.findData(selection.cameras_from)
        if index >= 0:
            self.cameras_combo.setCurrentIndex(index)

    def selection(self) -> MatchSelection:
        """Precondition: the ticked views suit the source (raises otherwise)."""
        views = tuple(n for n, c in self.view_checks.items() if c.isChecked())
        all_views = len(views) == len(self.view_checks)
        source = IMAGE_SPACE if self.image_radio.isChecked() else TRIANGULATE
        return MatchSelection(
            name=self.name_edit.text().strip(),
            views=views if (source == IMAGE_SPACE or not all_views) else (),
            observation_set=self.set_combo.currentText() or "observations",
            source=source,
            cameras_from=str(self.cameras_combo.currentData() or ""),
        )
