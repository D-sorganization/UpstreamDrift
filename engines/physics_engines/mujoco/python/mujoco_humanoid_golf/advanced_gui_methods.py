import json
import logging
from pathlib import Path

import mujoco

logger = logging.getLogger(__name__)


class AdvancedGuiMethodsMixin:
    """Mixin class providing configuration loading methods."""

    def _load_launch_config(self) -> None:
        """Load configuration passed from launcher if available."""
        current_dir = Path.cwd()
        potential_paths = [
            current_dir / "simulation_config.json",
            current_dir.parent / "simulation_config.json",
            current_dir.parent / "docker" / "simulation_config.json",
            # If running from package
            Path(__file__).parent.parent.parent / "simulation_config.json",
        ]

        config_data = {}
        for path in potential_paths:
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        config_data = json.load(f)
                    logger.info("Loaded configuration from %s", path)
                    break
                except Exception as e:
                    logger.warning("Failed to parse config file: %s (%s)", path, e)

        if not config_data:
            return

        # Determine model based on config
        target_model = "full_body"

        # Find index
        model_index = 0
        found_model = False
        # self.model_configs must be defined in the main class
        if hasattr(self, "model_configs"):
            for i, cfg in enumerate(self.model_configs):
                if cfg["name"] == target_model:
                    model_index = i
                    found_model = True
                    break

        if found_model and hasattr(self, "model_combo"):
            # Set combo box (this triggers load_current_model via signal)
            self.model_combo.setCurrentIndex(model_index)

        # Apply colors if present
        if "colors" in config_data:
            self._apply_config_colors(config_data["colors"])

    def _apply_config_colors(self, colors: dict) -> None:
        """Apply colors from config to the model."""
        if not hasattr(self, "sim_widget") or self.sim_widget.model is None:
            return

        # Helper to set color for geoms containing string
        def set_color_contain(name_part: str, rgba: list) -> None:
            for i in range(self.sim_widget.model.ngeom):
                name = mujoco.mj_id2name(
                    self.sim_widget.model, mujoco.mjtObj.mjOBJ_GEOM, i
                )
                if name and name_part in name:
                    self.sim_widget.model.geom_rgba[i] = rgba

        if "shirt" in colors:
            set_color_contain("torso", colors["shirt"])
            set_color_contain("upper_arm", colors["shirt"])

        if "pants" in colors:
            set_color_contain("thigh", colors["pants"])
            set_color_contain("shin", colors["pants"])

        if "shoes" in colors:
            set_color_contain("foot", colors["shoes"])

        if "skin" in colors:
            set_color_contain("head", colors["skin"])
            set_color_contain("hand", colors["skin"])
            set_color_contain("forearm", colors["skin"])

        if "club" in colors:
            set_color_contain("club", colors["club"])

        self.sim_widget._render_once()

    def on_ellipsoid_visualization_changed(self, state: int) -> None:
        """Handle ellipsoid visualization toggle."""
        if hasattr(self, "sim_widget"):
            # Check if mobility ellipsoid checkbox is checked
            show_mobility = False
            if hasattr(self, "show_mobility_ellipsoid_cb"):
                show_mobility = self.show_mobility_ellipsoid_cb.isChecked()

            # Check if force ellipsoid checkbox is checked
            show_force = False
            if hasattr(self, "show_force_ellipsoid_cb"):
                show_force = self.show_force_ellipsoid_cb.isChecked()

            # Update visualization
            self.sim_widget.set_ellipsoid_visualization(show_mobility, show_force)
