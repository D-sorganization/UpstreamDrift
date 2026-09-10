"""Model-only playback inside the established video coaching/drawing interface."""

from pathlib import Path
from typing import cast

from PyQt6.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

from src.motion_capture.coaching import ReferenceGeometry
from src.motion_capture.reference.evidence import asset_identity
from src.motion_capture.reference.model import ReferenceMotion
from src.motion_capture.reference.registration import ReferenceRegistration

from .coaching_dialog import CoachingDialog
from .geometry_controls import GeometryControls
from .model_coaching_source import ModelCoachingSource
from .model_frame_source import ModelFrameSource, ModelViewRecipe
from .reference_appearance import MotionAppearanceControls
from .reference_controls import SpatialControls


class ModelAnalysisDialog(CoachingDialog):
    """Reuse capture drawing, selection, undo, playback and worker export controls."""

    def __init__(
        self, asset: ReferenceMotion, root: Path, parent: QWidget | None = None
    ) -> None:
        initial = ModelFrameSource.from_motion(asset)
        self.model_source = ModelCoachingSource(initial.recipe, root)
        initial.close()
        self.geometry_controls: GeometryControls | None = None
        self.spatial: SpatialControls | None = None
        super().__init__(root, asset.id, parent, media=self.model_source)
        self.setWindowTitle(f"Model Analysis · {asset.title}[*]")
        self.resize(1000, 900)
        self._inspectors()
        self.status.setText("Virtual Camera · Source Geometry and Timing Preserved")

    def _inspectors(self) -> None:
        recipe = self.model_source.reader.recipe
        tabs = QTabWidget()
        tabs.setMaximumHeight(240)
        self.appearance = MotionAppearanceControls(
            recipe.appearance, has_club=bool(recipe.asset.club_edges)
        )
        self.appearance.changed.connect(self._appearance_changed)
        self.spatial = SpatialControls(
            recipe.registration, "motion", recipe.camera.image_size_px
        )
        self.spatial.changed.connect(self._placement_changed)
        self.spatial.pending_changed.connect(self._sync)
        geometry = recipe.geometry or ReferenceGeometry(
            scene_id=asset_identity(recipe.asset)
        )
        self.geometry_controls = GeometryControls(geometry)
        self.geometry_controls.changed.connect(self._geometry_changed)
        self.geometry_controls.pending_changed.connect(self._sync)
        for label, widget in (
            ("Appearance", self.appearance),
            ("Placement and Handedness", self.spatial),
            ("3D References", self.geometry_controls),
        ):
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setWidget(widget)
            tabs.addTab(area, label)
        layout = cast(QVBoxLayout, self.layout())
        layout.insertWidget(0, tabs)

    def _replace(self, **changes: object) -> None:
        current = self.model_source.reader.recipe
        recipe = ModelViewRecipe.model_validate(current.model_dump() | changes)
        self.model_source.replace_recipe(recipe)
        self.reader = self.model_source.reader
        self._show_frame(self.slider.value())

    def _appearance_changed(self) -> None:
        current = self.model_source.reader.recipe
        self._replace(appearance=self.appearance.updated(current.appearance))

    def _placement_changed(self, registration: ReferenceRegistration) -> None:
        self._replace(registration=registration)

    def _geometry_changed(self, geometry: ReferenceGeometry) -> None:
        self._replace(geometry=geometry)

    def _has_unsaved(self) -> bool:
        return (
            super()._has_unsaved()
            or bool(self.geometry_controls and self.geometry_controls.pending)
            or bool(self.spatial and self.spatial.pending)
        )

    def save(self) -> bool:
        """Apply valid pending display settings before saving the shared document."""
        if self.geometry_controls and self.geometry_controls.pending:
            if not self.geometry_controls.apply_selected():
                self.status.setText("Resolve invalid 3D reference fields before saving")
                return False
        if self.spatial and self.spatial.pending and not self.spatial.apply_placement():
            self.status.setText("Resolve invalid placement fields before saving")
            return False
        result = super().save()
        if result:
            self.status.setText("Model Analysis Saved")
        return result
