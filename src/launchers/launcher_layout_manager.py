"""Layout management for the Golf Launcher.

This module provides centralized layout persistence and grid management
for the Golf Modeling Suite launcher application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.launchers.launcher_constants import (
    TILE_SCALE_DEFAULT,
    ViewMode,
    validate_tile_scale,
    view_mode_settings,
)
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QGridLayout

logger = get_logger(__name__)


class LayoutConfig:
    """Configuration constants for layout management."""

    GRID_COLUMNS = 4
    DEFAULT_WINDOW_WIDTH = 1280
    DEFAULT_WINDOW_HEIGHT = 800
    MIN_WINDOW_Y = 50  # Ensure window title bar is visible


def _view_mode_from_string(name: str | None) -> ViewMode:
    """Parse a stored string into a :class:`ViewMode`, defaulting to COMPACT."""
    if not name:
        return ViewMode.COMPACT
    try:
        return ViewMode[str(name).strip().upper()]
    except KeyError:
        logger.warning("Unknown view_mode %r, falling back to COMPACT", name)
        return ViewMode.COMPACT


class LayoutManager:
    """Manages layout persistence and grid organization for the launcher.

    This class handles:
    - Model order tracking and persistence
    - Layout save/load operations
    - Grid rebuilding logic
    - Drag-and-drop model swapping
    """

    def __init__(
        self,
        config_file: Path,
        available_models: dict[str, Any],
        get_model_func: Any,
        create_card_func: Any,
        create_header_func: Any = None,
    ) -> None:
        """Initialize the layout manager.

        Args:
            config_file: Path to the layout configuration JSON file.
            available_models: Dictionary of available model configurations.
            get_model_func: Callback to retrieve a model by ID.
            create_card_func: Callback to create a model card widget.
        """
        if config_file is None:
            raise ValueError("config_file must be provided")
        self.config_file = config_file
        self.config_dir = config_file.parent
        self.available_models = available_models
        self._get_model = get_model_func
        self._create_card = create_card_func
        self._create_header = create_header_func

        # State
        self.model_order: list[str] = []
        self.model_cards: dict[str, Any] = {}
        self.edit_mode = False
        self.current_filter_text = ""
        self.current_view_mode: ViewMode = ViewMode.COMPACT
        self.tile_scale: float = TILE_SCALE_DEFAULT
        self.current_category_filter = "All"

    def initialize_model_order(self, default_ids: list[str] | None = None) -> None:
        """Set a sensible default grid ordering.

        Args:
            default_ids: Optional list of default model IDs to use.
        """
        if default_ids is None:
            default_ids = [
                "mujoco_unified",
                "drake_golf",
                "pinocchio_golf",
                "opensim_golf",
                "myosim_suite",
                "putting_green",
                "matlab_suite",
                "c3d_viewer",
                "openpose_analysis",
                "mediapipe_analysis",
                "model_explorer",
                "video_analyzer",
                "data_explorer",
                "project_map",
            ]

        # Filter to available models
        available_ids = [
            model_id for model_id in default_ids if model_id in self.available_models
        ]
        missing_ids = [
            model_id
            for model_id in default_ids
            if model_id not in self.available_models
        ]

        self.model_order = available_ids

        logger.info(
            f"Model order initialized with {len(self.model_order)} of {len(default_ids)} tiles"
        )
        if missing_ids:
            logger.warning(f"Missing models from defaults: {missing_ids}")
            logger.debug(f"Available model IDs: {list(self.available_models.keys())}")

    def save_layout(self, window_state: dict[str, Any]) -> None:
        """Save the current model layout to configuration file.

        Args:
            window_state: Dictionary containing window geometry and UI options.
        """
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)

            layout_data = {
                "model_order": self.model_order,
                "selected_model": window_state.get("selected_model"),
                "window_geometry": window_state.get("geometry", {}),
                "options": window_state.get("options", {}),
                "view_mode": self.current_view_mode.name.lower(),
                "tile_scale": float(self.tile_scale),
            }

            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, indent=2)

            logger.info(f"Layout saved to {self.config_file}")

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Failed to save layout: {e}")

    def load_layout(self) -> dict[str, Any] | None:
        """Load the saved model layout from configuration file.

        Returns:
            Loaded layout data dict, or None if no saved layout exists.
        """
        try:
            if not self.config_file.exists():
                logger.info("No saved layout found, using defaults")
                return None

            with open(self.config_file, encoding="utf-8") as f:
                layout_data = json.load(f)

            # Restore model order if valid
            saved_order = [
                model_id
                for model_id in layout_data.get("model_order", [])
                if model_id in self.available_models
            ]
            if saved_order:
                self.model_order = saved_order
                logger.info("Model layout restored from saved configuration")

            # View-mode + tile-scale are additive keys; missing ones use defaults.
            self.current_view_mode = _view_mode_from_string(
                layout_data.get("view_mode")
            )
            raw_scale = layout_data.get("tile_scale")
            if raw_scale is not None:
                try:
                    self.tile_scale = validate_tile_scale(float(raw_scale))
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Invalid tile_scale %r in saved layout: %s", raw_scale, exc
                    )

            return layout_data

        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.error(f"Failed to load layout from {self.config_file}: {e}")
            return None

    def sync_model_cards(self) -> None:
        """Ensure widgets match the current model order."""
        # Remove cards that are no longer selected
        for model_id in list(self.model_cards.keys()):
            if model_id not in self.model_order:
                widget = self.model_cards.pop(model_id)
                widget.setParent(None)
                widget.deleteLater()

        # Create cards for any newly added models
        _scale, _cols, show_desc, is_list = view_mode_settings(self.current_view_mode)
        for model_id in self.model_order:
            if model_id not in self.model_cards:
                model = self._get_model(model_id)
                if model:
                    self.model_cards[model_id] = self._build_card(
                        model,
                        tile_scale=self.tile_scale,
                        show_description=show_desc,
                        list_mode=is_list,
                    )

    def apply_model_selection(self, selected_ids: list[str]) -> list[str]:
        """Apply a new set of selected models from the layout dialog.

        Args:
            selected_ids: List of model IDs selected by the user.

        Returns:
            The new ordered list of model IDs.
        """
        # Keep existing order for models that are still selected
        if selected_ids is None:
            raise ValueError("selected_ids must be provided")
        ordered_selection = [
            model_id for model_id in self.model_order if model_id in selected_ids
        ]

        # Append newly selected models
        for model_id in selected_ids:
            if model_id not in ordered_selection and model_id in self.available_models:
                ordered_selection.append(model_id)

        self.model_order = ordered_selection
        return self.model_order

    def swap_models(self, source_id: str, target_id: str) -> bool:
        """Swap two models in the grid layout.

        Args:
            source_id: ID of the source model being dragged.
            target_id: ID of the target model being dropped on.

        Returns:
            True if swap was successful, False otherwise.
        """
        if source_id is None:
            raise ValueError("source_id must be provided")
        if not self.edit_mode:
            return False

        try:
            idx1 = self.model_order.index(source_id)
            idx2 = self.model_order.index(target_id)

            # Swap in list
            self.model_order[idx1], self.model_order[idx2] = (
                self.model_order[idx2],
                self.model_order[idx1],
            )
            return True

        except ValueError:
            return False  # ID not found

    def get_filtered_order(self) -> list[str]:
        """Get model order filtered by current search text and category.

        Returns:
            List of model IDs matching the current filters.
        """
        filtered = []
        for model_id in self.model_order:
            model = self._get_model(model_id)
            if not model:
                continue

            # Category filter
            if self.current_category_filter != "All":
                cat = self._get_model_category(model)
                if cat != self.current_category_filter:
                    continue

            # Search text filter
            if self.current_filter_text:
                search_content = f"{model.name} {model.id} {model.description}".lower()
                if self.current_filter_text not in search_content:
                    continue

            filtered.append(model_id)

        return filtered

    def _get_model_category(self, model: Any) -> str:
        """Determine the category of a model for layout grouping."""
        launcher = getattr(model, "launcher", None)
        if isinstance(launcher, dict):
            cat = launcher.get("category")
        else:
            cat = getattr(launcher, "category", None) if launcher else None

        if cat:
            if cat == "physics_engine":
                return "Core Physics Engines"
            if cat == "tool":
                return "Analysis Tools"
            if cat == "external":
                return "Utilities"

        t = getattr(model, "type", "").lower()
        if t in [
            "custom_humanoid",
            "drake",
            "pinocchio",
            "opensim",
            "myosim",
            "putting_green",
        ]:
            return "Core Physics Engines"
        if t == "matlab_suite":
            return "Matlab Simscape Models"
        if t == "special_app":
            return "Analysis Tools"
        return "Utilities"

    def _build_card(self, model: Any, **kwargs: Any) -> Any:
        """Invoke ``_create_card`` with optional keyword arguments.

        Falls back to a positional-only call so legacy callbacks that accept
        ``(model,)`` continue to work.
        """
        try:
            return self._create_card(model, **kwargs)
        except TypeError:
            return self._create_card(model)

    def set_view_mode(self, mode: ViewMode) -> None:
        """Apply a new :class:`ViewMode` and propagate scaling to existing cards.

        The actual grid is not rebuilt here — call :meth:`rebuild_grid` after.
        """
        if not isinstance(mode, ViewMode):
            raise TypeError(f"mode must be a ViewMode, got {type(mode).__name__}")
        scale, _cols, show_desc, is_list = view_mode_settings(mode)
        self.current_view_mode = mode
        self.tile_scale = scale
        # When switching into/out of LIST mode the grid topology changes, so
        # we drop existing cards and let rebuild_grid recreate them with the
        # right list_mode flag. For grid->grid switches an in-place
        # set_tile_scale is sufficient.
        was_list = any(
            getattr(c, "_list_mode", False) for c in self.model_cards.values()
        )
        if is_list != was_list:
            for c in list(self.model_cards.values()):
                c.setParent(None)
                c.deleteLater()
            self.model_cards.clear()
        else:
            for card in self.model_cards.values():
                if hasattr(card, "set_tile_scale"):
                    card.set_tile_scale(
                        scale,
                        show_description=show_desc,
                        list_mode=is_list,
                    )

    def set_tile_scale(self, scale: float) -> None:
        """Update tile_scale and resize all live cards in place."""
        self.tile_scale = validate_tile_scale(scale)
        _scale, _cols, show_desc, is_list = view_mode_settings(self.current_view_mode)
        for card in self.model_cards.values():
            if hasattr(card, "set_tile_scale"):
                card.set_tile_scale(
                    self.tile_scale,
                    show_description=show_desc,
                    list_mode=is_list,
                )

    def rebuild_grid(self, grid_layout: QGridLayout) -> None:  # noqa: C901
        """Rebuild the grid layout based on current model order and view mode.

        Args:
            grid_layout: The Qt grid layout to populate.
        """
        # Clean current layout
        if grid_layout is None:
            raise ValueError("grid_layout must be provided")
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)

        scale, columns, show_desc, is_list = view_mode_settings(self.current_view_mode)
        # Honour any explicit tile_scale set by the zoom slider, but fall
        # back to the view-mode default if it matches the previous mode.
        active_scale = self.tile_scale if self.tile_scale > 0 else scale

        # Get filtered model order
        filtered_order = self.get_filtered_order()

        # Group widgets by category maintaining order
        categories: dict[str, list[Any]] = {
            "Core Physics Engines": [],
            "Analysis Tools": [],
            "Matlab Simscape Models": [],
            "Utilities": [],
            "Other": [],
        }

        for model_id in filtered_order:
            if model_id not in self.model_cards:
                model = self._get_model(model_id)
                if model:
                    self.model_cards[model_id] = self._build_card(
                        model,
                        tile_scale=active_scale,
                        show_description=show_desc,
                        list_mode=is_list,
                    )
            else:
                # Existing card — make sure it matches current scale/mode.
                card = self.model_cards[model_id]
                if hasattr(card, "set_tile_scale"):
                    card.set_tile_scale(
                        active_scale,
                        show_description=show_desc,
                        list_mode=is_list,
                    )

            if model_id in self.model_cards:
                model = self._get_model(model_id)
                cat = self._get_model_category(model) if model else "Other"
                if cat not in categories:
                    cat = "Other"
                categories[cat].append(self.model_cards[model_id])

        # Add to grid
        row = 0
        for cat_name, widgets in categories.items():
            if not widgets:
                continue

            # Add section header spanning the active number of columns.
            if self._create_header:
                header = self._create_header(cat_name)
                grid_layout.addWidget(header, row, 0, 1, columns)
            row += 1

            col = 0
            for widget in widgets:
                if is_list:
                    # Each card occupies a full row, one card per row.
                    grid_layout.addWidget(widget, row, 0, 1, 1)
                    row += 1
                else:
                    grid_layout.addWidget(widget, row, col)
                    col += 1
                    if col >= columns:
                        col = 0
                        row += 1

            if not is_list and col > 0:
                row += 1

    def set_edit_mode(self, enabled: bool) -> None:
        """Set layout edit mode.

        Args:
            enabled: Whether editing is enabled.
        """
        if enabled is None:
            raise ValueError("enabled must be provided")
        self.edit_mode = enabled

        # Update all cards to accept/reject drops
        for card in self.model_cards.values():
            card.setAcceptDrops(enabled)

    def update_search_filter(self, text: str) -> None:
        """Update the search filter text.

        Args:
            text: Search text to filter by.
        """
        self.current_filter_text = text.lower()


def compute_centered_geometry(
    screen_width: int,
    screen_height: int,
    window_width: int = LayoutConfig.DEFAULT_WINDOW_WIDTH,
    window_height: int = LayoutConfig.DEFAULT_WINDOW_HEIGHT,
    screen_x: int = 0,
    screen_y: int = 0,
) -> tuple[int, int, int, int]:
    """Compute centered window geometry.

    Args:
        screen_width: Available screen width.
        screen_height: Available screen height.
        window_width: Desired window width.
        window_height: Desired window height.
        screen_x: Screen X offset.
        screen_y: Screen Y offset.

    Returns:
        Tuple of (x, y, width, height) for centered window.
    """
    if screen_width is None:
        raise ValueError("screen_width must be provided")
    x = screen_x + (screen_width - window_width) // 2
    y = screen_y + (screen_height - window_height) // 2

    # Ensure window title bar is visible
    y = max(y, LayoutConfig.MIN_WINDOW_Y)

    return x, y, window_width, window_height
