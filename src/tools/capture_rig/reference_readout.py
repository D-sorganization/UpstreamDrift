"""One metric-reference readout for model and captured-video analysis."""

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from src.motion_capture.coaching import ReferenceGeometry
from src.motion_capture.coaching.measurements import reference_distances
from src.motion_capture.reference.model import ReferenceMotion
from src.motion_capture.reference.registration import ReferenceRegistration


class ReferenceReadout(QWidget):
    """Select a motion landmark and scene reference; show current world metres."""

    def __init__(self) -> None:
        super().__init__()
        self.landmark = QComboBox()
        self.reference = QComboBox()
        self.value = QLabel("Choose a motion and add a 3D reference.")
        self.value.setWordWrap(True)
        self._evaluate: Callable[[str, str], float] | None = None
        self.landmark.setAccessibleName("Measured Motion Landmark")
        self.reference.setAccessibleName("Measurement Reference")
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow("Motion Landmark", self.landmark)
        form.addRow("World Reference", self.reference)
        form.addRow(self.value)
        note = QLabel(
            "World metres (Y up). Planes use signed distance to the infinite plane. Values depend on the model and its registration."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.landmark.currentIndexChanged.connect(self._refresh)
        self.reference.currentIndexChanged.connect(self._refresh)

    @staticmethod
    def _choices(combo: QComboBox, choices: tuple[tuple[str, str], ...]) -> None:
        existing = tuple(
            (combo.itemText(i), combo.itemData(i)) for i in range(combo.count())
        )
        if existing == choices:
            return
        selected = combo.currentData()
        with QSignalBlocker(combo):
            combo.clear()
            for title, identity in choices:
                combo.addItem(title, identity)
            index = combo.findData(selected)
            combo.setCurrentIndex(max(0, index))

    def set_context(
        self,
        motion: ReferenceMotion | None,
        registration: ReferenceRegistration,
        geometry: ReferenceGeometry,
        scene_time: float,
        scene_id: str,
    ) -> None:
        """Replace the immutable scene snapshot and refresh without losing selection."""
        self._evaluate = None
        names = motion.joint_names if motion is not None else ()
        self._choices(self.landmark, tuple((name, name) for name in names))
        self._choices(
            self.reference,
            tuple(
                (item.title, item.id) for item in (*geometry.planes, *geometry.points)
            ),
        )
        if motion is not None:

            def evaluate(joint: str, reference: str) -> float:
                return float(
                    reference_distances(
                        motion,
                        registration,
                        geometry,
                        [scene_time],
                        joint,
                        reference,
                        scene_id=scene_id,
                    )[0]
                )

            self._evaluate = evaluate
        self._refresh()

    def _refresh(self) -> None:
        joint, reference = self.landmark.currentData(), self.reference.currentData()
        if self._evaluate is None or joint is None or reference is None:
            self.value.setText("Choose a motion and add a 3D reference.")
            return
        try:
            distance = self._evaluate(joint, reference)
        except ValueError as exc:
            self.value.setText(f"Unavailable: {exc}")
            return
        self.value.setText(
            f"{distance:.4f} m"
            if np.isfinite(distance)
            else "Unavailable: missing motion sample"
        )
