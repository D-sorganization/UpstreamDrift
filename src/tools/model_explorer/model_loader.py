"""Model load/download actions and info-formatting helpers for ModelLoaderDialog.

Provides ModelLoaderMixin, which supplies:
- _load_selected_model   — resolve selection and emit model_selected signal
- _download_human_model  — download from human-gazebo repository
- _on_import_button      — import a URDF/MJCF file via file dialog
- _on_imported_context_menu — rename / delete an imported model
- _format_*_info helpers — produce human-readable info text for each category

Intended to be mixed into ModelLoaderDialog.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QMessageBox

from src.shared.python.logging_pkg.logger_utils import get_logger

logger = get_logger(__name__)


class ModelLoaderMixin:
    """Mixin providing model loading actions and info formatters for the loader dialog."""

    # ------------------------------------------------------------------
    # Load / import actions
    # ------------------------------------------------------------------

    def _load_selected_model(self, category: str) -> None:
        """Load the selected model for *category* and close the dialog.

        Args:
            category: Model category key (e.g. 'human', 'golf_clubs', 'discovered').

        """
        if category is None:
            raise ValueError("category must be provided")
        if category == "human":
            model_key = self.human_combo.currentData()
        elif category == "golf_clubs":
            model_key = self.club_combo.currentData()
        elif hasattr(self, f"{category}_combo"):
            combo = getattr(self, f"{category}_combo")
            model_key = combo.currentData()
        else:
            # Fallback: nothing to load without a registered combo
            return

        if model_key:
            self.selected_category = category
            self.selected_model = model_key

            if (
                category == "human"
                and hasattr(self, "default_human_chk")
                and self.default_human_chk.isChecked()
            ):
                from PyQt6.QtCore import QSettings

                settings = QSettings("GolfModelingSuite", "URDFGenerator")
                settings.setValue("default_human_model", model_key)
                logger.info(f"Set default human model to: {model_key}")

            self.model_selected.emit(category, model_key)
            self.accept()

    def _download_human_model(self) -> None:
        """Download the currently selected human model from human-gazebo."""
        model_key = self.human_combo.currentData()
        if not model_key:
            return

        reply = QMessageBox.question(
            self,
            "Download Model",
            f"Download human model '{self.human_combo.currentText()}' "
            "from human-gazebo repository?\n\n"
            "This will download the URDF file and associated mesh files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                urdf_path = self.library.download_human_model(model_key)
                if urdf_path:
                    QMessageBox.information(
                        self,
                        "Download Complete",
                        f"Model downloaded successfully to:\n{urdf_path}",
                    )
                else:
                    QMessageBox.warning(
                        self, "Download Failed", "Failed to download model files."
                    )
            except (RuntimeError, ValueError, OSError) as e:
                logger.error(f"Download error: {e}")
                QMessageBox.critical(
                    self, "Error", f"Download failed with error:\n{str(e)}"
                )

    def _on_import_button(self) -> None:
        """Open a file dialog to import a URDF/MJCF model."""
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Model File",
            "",
            "Model Files (*.urdf *.xml *.mjcf);;All Files (*)",
        )
        if path:
            if self.library.import_model(path):
                self._reload_imported_models()
                QMessageBox.information(self, "Success", "Model imported successfully.")
            else:
                QMessageBox.warning(self, "Error", "Failed to import model.")

    def _on_imported_context_menu(self, pos: Any) -> None:
        """Show a context menu for renaming or deleting an imported model."""
        from PyQt6.QtWidgets import QInputDialog, QMenu

        item = self.imported_tree.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")

        viewport = self.imported_tree.viewport()
        if viewport is None:
            return

        action = menu.exec(viewport.mapToGlobal(pos))

        from PyQt6.QtCore import Qt

        model_path = item.data(0, Qt.ItemDataRole.UserRole + 1)

        if action == rename_action:
            old_name = item.text(0)
            new_name, ok = QInputDialog.getText(
                self, "Rename Model", "New name:", text=old_name
            )
            if ok and new_name and new_name != old_name:
                if self.library.rename_imported_model(model_path, new_name):
                    self._reload_imported_models()
                else:
                    QMessageBox.warning(self, "Error", "Failed to rename model.")

        elif action == delete_action:
            reply = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Are you sure you want to delete '{item.text(0)}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.library.delete_imported_model(model_path):
                    self._reload_imported_models()
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete model.")

    # ------------------------------------------------------------------
    # Info formatters
    # ------------------------------------------------------------------

    def _format_human_info(self, model_info: dict[str, Any]) -> str:
        """Return a formatted info string for a human biomechanical model."""
        return (
            f"Name: {model_info['name']}\n"
            f"Description: {model_info['description']}\n"
            f"License: {model_info['license']}\n\n"
            f"Repository: https://github.com/gbionics/human-gazebo\n"
        )

    def _format_golf_clubs_info(self, model_info: dict[str, Any]) -> str:
        """Return a formatted info string for a golf club model."""
        return (
            f"Club: {model_info['name']}\n"
            f"Loft: {model_info['loft']}°\n"
            f"Length: {model_info['length'] * 100:.1f} cm "
            f"({model_info['length'] / 0.0254:.1f} inches)\n"
            f"Total Mass: {model_info['mass'] * 1000:.1f} g\n"
            f"  - Head: {model_info['head_mass'] * 1000:.1f} g\n"
            f"  - Shaft: {model_info['shaft_mass'] * 1000:.1f} g\n"
            f"  - Grip: {model_info['grip_mass'] * 1000:.1f} g\n\n"
            f"The URDF will be automatically generated with realistic "
            f"geometry and inertial properties.\n"
        )

    def _format_generic_model_info(self, model_info: dict[str, Any]) -> str:
        """Return a formatted info string for a generic (pendulum/robotic/component) model."""
        return (
            f"Name: {model_info['name']}\n"
            f"Type: {model_info.get('type', 'Unknown').upper()}\n"
            f"Description: {model_info['description']}\n"
            f"Path: {model_info.get('path', 'N/A')}\n\n"
            f"Click 'Load' to view this model.\n"
        )

    def _format_discovered_info(self, model_info: dict[str, Any]) -> str:
        """Return a formatted info string for a discovered repository model."""
        return (
            f"Name: {model_info['name']}\n"
            f"Type: {model_info['type'].upper()}\n"
            f"Path: {model_info['description']}\n\n"
            f"Click 'Load Selected Repository Model' to view.\n"
        )

    def _format_embedded_info(self, model_info: dict[str, Any]) -> str:
        """Return a formatted info string for an embedded MJCF model."""
        content_preview = (
            model_info["content"][:200] + "..."
            if len(model_info["content"]) > 200
            else model_info["content"]
        )
        return (
            f"Name: {model_info['name']}\n"
            f"Type: Embedded MJCF\n"
            f"Description: {model_info['description']}\n\n"
            f"Content Preview:\n{content_preview}\n"
        )

    def _format_robot_descriptions_info(self, model_info: dict[str, Any]) -> str:
        """Return a formatted info string for a robot_descriptions community model."""
        return (
            f"Name: {model_info['name']}\n"
            f"Type: {model_info['type'].upper()}\n"
            f"Package: {model_info.get('package', 'robot_descriptions')}\n"
            f"Path: {model_info['path']}\n\n"
            f"Description: {model_info['description']}\n"
        )

    def _format_imported_info(self, model_info: dict[str, Any]) -> str:
        """Return a formatted info string for a user-imported model."""
        return (
            f"Name: {model_info['name']}\n"
            f"Type: {model_info['type'].upper()}\n"
            f"Path: {model_info['path']}\n\n"
            f"User imported model. Right-click in the list to Rename or Delete.\n"
        )
