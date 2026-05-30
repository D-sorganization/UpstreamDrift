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
    """Parse a stored string into a :class:`ViewMode`, defaulting to LIST_LARGE.

    Accepts both the current enum key names (e.g. ``"LARGE"``) and legacy
    aliases that may have been persisted by older launchers before PR #5688
    renamed the enum members (e.g. ``"comfortable"``).

    Backward-compat mapping (pre-#5688 → current)
    -----------------------------------------------
    ``"comfortable"``  → ``ViewMode.LARGE``       (tile grid, NOT list)
    ``"compact"``      → ``ViewMode.MEDIUM``
    ``"dense"``        → ``ViewMode.SMALL``
    ``"list"``         → ``ViewMode.LIST_LARGE``
    ``"panel"``        → ``ViewMode.LIST_LARGE``  (old sidebar panel layout)
    ``"floating"``     → ``ViewMode.LIST_LARGE``  (old floating window layout)

    DbC postcondition: always returns a :class:`ViewMode` member.
    """
    if not name:
        return ViewMode.LIST_LARGE
    # Backward compat: map pre-#5688 token names → new canonical enum key names.
    # "comfortable" must map → LARGE (tile grid), NOT list mode (fixes #5690).
    _compat: dict[str, str] = {
        "comfortable": "LARGE",
        "compact": "MEDIUM",
        "dense": "SMALL",
        "list": "LIST_LARGE",
        # Pre-#5688 layout tokens that were never valid enum names
        "panel": "LIST_LARGE",
        "floating": "LIST_LARGE",
    }
    raw = str(name).strip().lower()
    key = _compat.get(raw, str(name).strip().upper())
    try:
        mode = ViewMode[key]
    except KeyError:
        logger.warning("Unknown view_mode %r, falling back to LIST_LARGE", name)
        mode = ViewMode.LIST_LARGE
    assert isinstance(mode, ViewMode), (  # DbC postcondition
        f"_view_mode_from_string postcondition violated: got {mode!r}"
    )
    return mode


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
        self.current_view_mode: ViewMode = ViewMode.LIST_LARGE
        self.tile_scale: float = TILE_SCALE_DEFAULT
        self.current_category_filter = "All"
        self.favorites: list[str] = []
        self.launch_stats: dict[str, dict[str, Any]] = {}

    def record_launch(self, model_id: str) -> None:
        """Increment launch count and record the last launched time for history tracking."""
        if model_id is None:
            raise ValueError("model_id must be provided")
        if model_id not in self.launch_stats:
            self.launch_stats[model_id] = {"count": 0, "last_launched": ""}
        self.launch_stats[model_id]["count"] += 1
        from datetime import datetime

        self.launch_stats[model_id]["last_launched"] = datetime.now().isoformat()

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
                "library_tool",
            ]
            for model_id, model in self.available_models.items():
                is_hidden = getattr(model, "hidden", False)
                if not is_hidden and model_id not in default_ids:
                    default_ids.append(model_id)

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
                "favorites": self.favorites,
                "launch_stats": self.launch_stats,
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
                if model_id in self.available_models or model_id == "library_tool"
            ]
            if saved_order:
                for model_id, model in self.available_models.items():
                    is_hidden = getattr(model, "hidden", False)
                    if not is_hidden and model_id not in saved_order:
                        saved_order.append(model_id)
                self.model_order = saved_order
                logger.info("Model layout restored from saved configuration")

            self.favorites = layout_data.get("favorites", [])
            self.favorites = [
                fid
                for fid in self.favorites
                if fid in self.available_models or fid == "library_tool"
            ]
            self.launch_stats = layout_data.get("launch_stats", {})

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
        _compact = self.current_view_mode == ViewMode.LIST_SMALL
        for model_id in self.model_order:
            if model_id not in self.model_cards:
                model = self._get_model(model_id)
                if model:
                    self.model_cards[model_id] = self._build_card(
                        model,
                        tile_scale=self.tile_scale,
                        show_description=show_desc,
                        list_mode=is_list,
                        list_compact=_compact,
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

    def get_model_categories(self, model: Any) -> list[str]:
        """Determine the list of categories this model belongs to."""
        launcher = getattr(model, "launcher", None)
        cats = []
        if isinstance(launcher, dict):
            raw_cats = launcher.get("category") or launcher.get("categories")
        else:
            raw_cats = (
                getattr(launcher, "category", None)
                or getattr(launcher, "categories", None)
                if launcher
                else None
            )

        if raw_cats:
            if isinstance(raw_cats, str):
                cats.append(raw_cats)
            elif isinstance(raw_cats, list):
                cats.extend(raw_cats)

        mapped_cats = set()
        mapping = {
            "physics_engine": "Engines",
            "biomechanics": "Biomechanics",
            "simulation": "Simulation",
            "motion_matching": "Tools",
            "motion_capture": "Tools",
            "tool": "Tools",
            "external": "Tools",
            "documentation": "Documentation",
        }
        for cat in cats:
            cat_norm = str(cat).strip().lower()
            if cat_norm in mapping:
                mapped_cats.add(mapping[cat_norm])

        if not mapped_cats:
            t = str(getattr(model, "type", "") or "").lower()
            model_id = str(getattr(model, "id", "") or "").lower()

            if t in (
                "gait",
                "sit_stand",
                "movement_optimizer",
                "opensim_biomech",
                "biomechanics",
                "physics_informed",
            ) or any(
                term in model_id or term in t
                for term in ["gait", "sit_to_stand", "biomech", "pinn"]
            ):
                mapped_cats.add("Biomechanics")
            elif t in (
                "custom_humanoid",
                "drake",
                "pinocchio",
                "opensim",
                "myosim",
                "matlab_suite",
            ) or model_id in [
                "mujoco_unified",
                "drake_golf",
                "pinocchio_golf",
                "opensim_golf",
                "myosim_suite",
                "matlab_suite",
            ]:
                mapped_cats.add("Engines")
            elif t in ("putting_green", "golf_simulation") or model_id in [
                "putting_green",
                "golf_simulation_suite",
                "bunkershot3d",
                "pendulum_simulator",
                "shot_tracer",
                "cross_engine_dashboard",
                "terrain_engine",
                "golf_environment",
                "bunker_shot",
                "swing_flight_pipeline",
                "ball_flight_simulator",
                "putting_green_gui",
            ]:
                mapped_cats.add("Simulation")
            elif t == "document" or model_id in ["project_map", "library_tool"]:
                mapped_cats.add("Documentation")
            else:
                mapped_cats.add("Tools")

        model_id = getattr(model, "id", "").lower()
        if model_id == "movement_optimizer":
            mapped_cats.add("Tools")
            mapped_cats.add("Biomechanics")
        if model_id == "cross_engine_dashboard":
            mapped_cats.add("Tools")
            mapped_cats.add("Simulation")
        if model_id == "putting_green":
            mapped_cats.add("Tools")
            mapped_cats.add("Simulation")
        if model_id == "library_tool":
            mapped_cats.add("Documentation")
            mapped_cats.add("Tools")
        if model_id == "project_map":
            mapped_cats.add("Documentation")

        return list(mapped_cats)

    def _get_model_category(self, model: Any) -> str:
        """Compatibility wrapper for legacy category queries."""
        cats = self.get_model_categories(model)
        if not cats:
            return "Tools & Data"
        first = cats[0]
        compat = {
            "Engines": "Physics Engines",
            "Tools": "Tools & Data",
        }
        return compat.get(first, first)

    def get_filtered_order(self) -> list[str]:
        """Get model order filtered by current search text and category.

        Returns:
            List of model IDs matching the current filters.
        """
        source_list = self.model_order

        if self.current_category_filter == "Favorites":
            source_list = [mid for mid in self.model_order if mid in self.favorites]
        elif self.current_category_filter == "History":
            launched = []
            unlaunched = []
            for mid in self.model_order:
                stats = self.launch_stats.get(mid, {})
                count = stats.get("count", 0)
                if count > 0:
                    launched.append((mid, count, stats.get("last_launched", "")))
                else:
                    unlaunched.append(mid)
            launched.sort(key=lambda x: (x[1], x[2]), reverse=True)
            source_list = [x[0] for x in launched] + unlaunched

        filtered = []
        for model_id in source_list:
            model = self._get_model(model_id)
            if not model:
                continue

            if (
                self.current_category_filter not in ("All", "Favorites", "History")
                and not self.current_filter_text
            ):
                categories = self.get_model_categories(model)
                if self.current_category_filter not in categories:
                    continue

            if self.current_filter_text:
                search_content = f"{model.name} {model.id} {model.description}".lower()
                if self.current_filter_text not in search_content:
                    continue

            filtered.append(model_id)

        return filtered

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
            try:
                mode = ViewMode(int(mode))
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    f"mode must be a ViewMode, got {type(mode).__name__}"
                ) from exc
        scale, _cols, show_desc, is_list = view_mode_settings(mode)
        _compact = mode == ViewMode.LIST_SMALL
        self.current_view_mode = mode
        self.tile_scale = scale
        # When switching list topology the cards need full rebuild.
        was_list = any(
            getattr(c, "_list_mode", False) for c in self.model_cards.values()
        )
        was_compact = any(
            getattr(c, "_list_compact", False) for c in self.model_cards.values()
        )
        if is_list != was_list or _compact != was_compact:
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
                        list_compact=_compact,
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
        # Clean current layout. Reusable model cards must be hidden before
        # detaching: on Qt, a visible parentless widget becomes a top-level
        # window, which made search/filter rebuilds flash every tile onscreen.
        if grid_layout is None:
            raise ValueError("grid_layout must be provided")

        # Clear stretch factors from previous layout
        for c in range(grid_layout.columnCount()):
            grid_layout.setColumnStretch(c, 0)

        reusable_card_ids = {id(card) for card in self.model_cards.values()}
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    if id(widget) in reusable_card_ids:
                        widget.hide()
                    else:
                        widget.deleteLater()
                    widget.setParent(None)

        scale, base_cols, show_desc, is_list = view_mode_settings(
            self.current_view_mode
        )
        # Honour any explicit tile_scale set by the zoom slider, but fall
        # back to the view-mode default if it matches the previous mode.
        active_scale = self.tile_scale if self.tile_scale > 0 else scale

        # Dynamically determine columns based on available scroll area viewport width if not in list mode
        if is_list:
            columns = 1
        else:
            available_width = 800  # fallback
            container = grid_layout.parentWidget()
            is_real_widget = (
                container is not None
                and not hasattr(container, "mock_add_spec")
                and type(container).__name__
                not in ("Mock", "MagicMock", "NonCallableMagicMock")
            )
            if is_real_widget:
                viewport = container.parentWidget()
                if viewport is not None and type(viewport).__name__ not in (
                    "Mock",
                    "MagicMock",
                ):
                    w = viewport.width()
                    if isinstance(w, int | float) and not hasattr(w, "mock_add_spec"):
                        available_width = w
                    scroll_area = viewport.parentWidget()
                    if (
                        scroll_area is not None
                        and type(scroll_area).__name__ not in ("Mock", "MagicMock")
                        and hasattr(scroll_area, "viewport")
                    ):
                        v = scroll_area.viewport()
                        if v is not None and type(v).__name__ not in (
                            "Mock",
                            "MagicMock",
                        ):
                            vw = v.width()
                            if isinstance(vw, int | float) and not hasattr(
                                vw, "mock_add_spec"
                            ):
                                available_width = vw

            card_width = max(100, int(240 * active_scale))
            spacing = 20
            try:
                s = grid_layout.spacing()
                if (
                    isinstance(s, (int, float))
                    and not hasattr(s, "mock_add_spec")
                    and s >= 0
                ):
                    spacing = int(s)
            except (AttributeError, TypeError, ValueError):
                pass

            # Allow some margin on the sides (e.g. 20px total padding)
            usable_width = available_width - 20
            columns = max(1, (usable_width + spacing) // (card_width + spacing))

        if not is_list:
            for c in range(columns):
                grid_layout.setColumnStretch(c, 1)

        # Get filtered model order
        filtered_order = self.get_filtered_order()

        widgets_to_add = []
        for model_id in filtered_order:
            if model_id not in self.model_cards:
                model = self._get_model(model_id)
                if model:
                    self.model_cards[model_id] = self._build_card(
                        model,
                        tile_scale=active_scale,
                        show_description=show_desc,
                        list_mode=is_list,
                        list_compact=(self.current_view_mode == ViewMode.LIST_SMALL),
                    )
            else:
                # Existing card — make sure it matches current scale/mode.
                card = self.model_cards[model_id]
                if hasattr(card, "set_tile_scale"):
                    card.set_tile_scale(
                        active_scale,
                        show_description=show_desc,
                        list_mode=is_list,
                        list_compact=(self.current_view_mode == ViewMode.LIST_SMALL),
                    )

            if model_id in self.model_cards:
                widgets_to_add.append(self.model_cards[model_id])

        # Add to grid as a flat, continuously wrapping list (no headers!)
        row = 0
        col = 0
        for widget in widgets_to_add:
            if is_list:
                # Each card occupies a full row, one card per row.
                grid_layout.addWidget(widget, row, 0, 1, 1)
                widget.show()
                row += 1
            else:
                grid_layout.addWidget(widget, row, col)
                widget.show()
                col += 1
                if col >= columns:
                    col = 0
                    row += 1

        # Final cleanup for grid layout rows
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
