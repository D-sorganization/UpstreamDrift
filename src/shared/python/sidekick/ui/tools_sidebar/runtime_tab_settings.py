"""Settings descriptors + panels for the Terminal, Python REPL, Workspace tabs.

Each of these tabs now declares a :class:`SidebarTabSettingsDescriptor`, so the
sidebar's ⚙ gear opens a real configuration surface instead of being disabled:

* **Terminal / Workspace** — adjustable colours and border via
  :class:`AppearanceSettingsPanel`.
* **Python REPL** — the same appearance controls plus a package editor
  (:class:`PythonReplSettingsPanel`) for the scientific packages preloaded
  into the window.

DRY: appearance persists as a JSON-safe :class:`PanelAppearance` payload
through the existing ``SidebarTabSettingsStore``; package preferences reuse the
validated :class:`CalculatorStartupConfig`. Live application goes through each
widget's public ``apply_appearance`` / ``apply_startup_config`` (LOD) found via
the host's narrow ``tab_widget`` accessor.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from .appearance import (
    DEFAULT_DARK_PANEL_APPEARANCE,
    DEFAULT_LIGHT_PANEL_APPEARANCE,
    MAX_BORDER_RADIUS,
    MAX_BORDER_WIDTH,
    PanelAppearance,
    coerce_appearance,
    is_panel_appearance,
)
from .appearance_settings_controls import _ColorButton
from .calculator_startup import (
    CalculatorStartupConfig,
    CalculatorStartupImport,
    default_repl_startup_config,
)
from .qt_compat import QtCore, QtWidgets
from .settings import SidebarTabSettingsDescriptor, SidebarTabSettingsSchema

__all__ = [
    "PYTHON_REPL_TAB_ID",
    "PYTHON_REPL_TAB_SETTINGS",
    "TERMINAL_TAB_ID",
    "TERMINAL_TAB_SETTINGS",
    "WORKSPACE_TAB_ID",
    "WORKSPACE_TAB_SETTINGS",
    "AppearanceSettingsPanel",
    "PythonReplSettingsPanel",
    "apply_appearance_to_tab",
    "build_python_repl_settings_panel",
    "build_terminal_settings_panel",
    "build_workspace_settings_panel",
    "parse_startup_rows",
]

_logger = logging.getLogger(__name__)

TERMINAL_TAB_ID = "terminal"
PYTHON_REPL_TAB_ID = "python_repl"
WORKSPACE_TAB_ID = "workspace"

_APPEARANCE_KEYS = frozenset(
    {"foreground", "background", "border_color", "border_width", "border_radius"}
)
_REPL_KEYS = _APPEARANCE_KEYS | {"startup_imports"}


# ─── Pure helpers (headless-testable) ────────────────────────────


def apply_appearance_to_tab(
    sidebar: Any, tab_id: str, appearance: PanelAppearance
) -> bool:
    """Apply ``appearance`` to the live tab widget if it supports it.

    LOD: locates the widget through the host's ``tab_widget`` accessor and
    calls only its public ``apply_appearance``. Returns ``True`` on success.
    """
    accessor = getattr(sidebar, "tab_widget", None)
    if not callable(accessor):
        return False
    try:
        widget = accessor(tab_id)
    except Exception:  # noqa: BLE001 - never let lookup break Save
        _logger.debug("tab_widget(%r) lookup failed", tab_id, exc_info=True)
        return False
    apply = getattr(widget, "apply_appearance", None)
    if not callable(apply):
        return False
    apply(appearance)
    return True


def parse_startup_rows(
    rows: Sequence[tuple[str, str, bool]],
) -> CalculatorStartupConfig:
    """Build a validated :class:`CalculatorStartupConfig` from editor rows.

    Blank rows (no module and no alias) are skipped. A missing alias defaults
    to the module's final path segment.

    Args:
        rows: ``(module, alias, enabled)`` triples from the package editor.

    Returns:
        A validated config (raises on invalid module/alias or duplicates).

    Raises:
        ValueError: If a module/alias is invalid or aliases collide.
    """
    imports: list[CalculatorStartupImport] = []
    for module, alias, enabled in rows:
        module = str(module).strip()
        alias = str(alias).strip()
        if not module and not alias:
            continue
        if not alias and module:
            alias = module.split(".")[-1]
        imports.append(
            CalculatorStartupImport(module=module, alias=alias, enabled=bool(enabled))
        )
    return CalculatorStartupConfig(tuple(imports))


# ─── Settings descriptors ────────────────────────────────────────


def _appearance_defaults(base: PanelAppearance) -> dict[str, Any]:
    return dict(base.to_dict())


def build_terminal_settings_panel(sidebar: Any, tab_id: str) -> QtWidgets.QWidget:
    """Widget factory for the Terminal tab settings dialog."""
    return AppearanceSettingsPanel(
        sidebar, tab_id, base_appearance=DEFAULT_DARK_PANEL_APPEARANCE
    )


def build_workspace_settings_panel(sidebar: Any, tab_id: str) -> QtWidgets.QWidget:
    """Widget factory for the Workspace tab settings dialog."""
    return AppearanceSettingsPanel(
        sidebar, tab_id, base_appearance=DEFAULT_LIGHT_PANEL_APPEARANCE
    )


def build_python_repl_settings_panel(sidebar: Any, tab_id: str) -> QtWidgets.QWidget:
    """Widget factory for the Python REPL tab settings dialog."""
    return PythonReplSettingsPanel(
        sidebar, tab_id, base_appearance=DEFAULT_DARK_PANEL_APPEARANCE
    )


TERMINAL_TAB_SETTINGS = SidebarTabSettingsDescriptor(
    schema=SidebarTabSettingsSchema(
        version=1,
        defaults=_appearance_defaults(DEFAULT_DARK_PANEL_APPEARANCE),
        allowed_keys=_APPEARANCE_KEYS,
    ),
    widget_factory=build_terminal_settings_panel,
)

WORKSPACE_TAB_SETTINGS = SidebarTabSettingsDescriptor(
    schema=SidebarTabSettingsSchema(
        version=1,
        defaults=_appearance_defaults(DEFAULT_LIGHT_PANEL_APPEARANCE),
        allowed_keys=_APPEARANCE_KEYS,
    ),
    widget_factory=build_workspace_settings_panel,
)

PYTHON_REPL_TAB_SETTINGS = SidebarTabSettingsDescriptor(
    schema=SidebarTabSettingsSchema(
        version=1,
        defaults={
            **_appearance_defaults(DEFAULT_DARK_PANEL_APPEARANCE),
            "startup_imports": default_repl_startup_config().to_list(),
        },
        allowed_keys=_REPL_KEYS,
    ),
    widget_factory=build_python_repl_settings_panel,
)


# ─── Color button ────────────────────────────────────────────────


# ─── Appearance settings panel ───────────────────────────────────


class AppearanceSettingsPanel(QtWidgets.QWidget):
    """Colour + border editor shared by the Terminal and Workspace tabs.

    Args:
        sidebar: Host exposing ``tab_settings`` / ``update_tab_settings`` and
            (optionally) ``tab_widget``.
        tab_id: The tab these settings belong to.
        base_appearance: Defaults used for Reset and missing fields.
        parent: Optional Qt parent.

    Raises:
        TypeError: If ``sidebar`` is ``None`` or ``base_appearance`` is wrong.
        ValueError: If ``tab_id`` is empty.
    """

    def __init__(
        self,
        sidebar: Any,
        tab_id: str,
        *,
        base_appearance: PanelAppearance = DEFAULT_DARK_PANEL_APPEARANCE,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if sidebar is None:
            raise TypeError("sidebar must be provided")
        if not isinstance(tab_id, str) or not tab_id.strip():
            raise ValueError("tab_id must be a non-empty string")
        if not is_panel_appearance(base_appearance):
            raise TypeError("base_appearance must be a PanelAppearance")
        super().__init__(parent)
        self.setObjectName("SidekickAppearanceSettingsPanel")
        self._sidebar = sidebar
        self._tab_id = tab_id
        self._base = base_appearance
        self._build_ui()
        self._load_current()

    # -- construction ------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_appearance_group())
        self._build_extra(layout)

        self._status = QtWidgets.QLabel("", self)
        self._status.setObjectName("SidekickAppearanceStatus")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        row = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("Reset to Defaults", self)
        reset.setObjectName("SidekickAppearanceReset")
        reset.clicked.connect(self._on_reset)
        row.addWidget(reset)
        row.addStretch(1)
        save = QtWidgets.QPushButton("Save", self)
        save.setObjectName("SidekickAppearanceSave")
        save.clicked.connect(self._on_save)
        row.addWidget(save)
        layout.addLayout(row)

    def _build_appearance_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Colours & Border", self)
        form = QtWidgets.QFormLayout(group)

        self._fg_button = _ColorButton(self._base.foreground, group)
        self._fg_button.setObjectName("SidekickAppearanceForeground")
        form.addRow("Text colour", self._fg_button)

        self._bg_button = _ColorButton(self._base.background, group)
        self._bg_button.setObjectName("SidekickAppearanceBackground")
        form.addRow("Background", self._bg_button)

        self._border_button = _ColorButton(self._base.border_color, group)
        self._border_button.setObjectName("SidekickAppearanceBorderColor")
        form.addRow("Border colour", self._border_button)

        self._width_spin = QtWidgets.QSpinBox(group)
        self._width_spin.setObjectName("SidekickAppearanceBorderWidth")
        self._width_spin.setRange(0, MAX_BORDER_WIDTH)
        self._width_spin.setSuffix(" px")
        form.addRow("Border width", self._width_spin)

        self._radius_spin = QtWidgets.QSpinBox(group)
        self._radius_spin.setObjectName("SidekickAppearanceBorderRadius")
        self._radius_spin.setRange(0, MAX_BORDER_RADIUS)
        self._radius_spin.setSuffix(" px")
        form.addRow("Corner radius", self._radius_spin)

        return group

    def _build_extra(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Hook for subclasses to add extra controls (default: none)."""

    # -- state <-> widgets ------------------------------------------

    def _load_current(self) -> None:
        appearance = coerce_appearance(self._current_values(), self._base)
        self._apply_to_widgets(appearance)

    def _apply_to_widgets(self, appearance: PanelAppearance) -> None:
        self._fg_button.set_color(appearance.foreground)
        self._bg_button.set_color(appearance.background)
        self._border_button.set_color(appearance.border_color)
        self._width_spin.setValue(appearance.border_width)
        self._radius_spin.setValue(appearance.border_radius)

    def _current_values(self) -> Mapping[str, Any]:
        getter = getattr(self._sidebar, "tab_settings", None)
        if not callable(getter):
            return {}
        try:
            payload = getter(self._tab_id)
        except Exception:  # noqa: BLE001 - degrade to defaults on store error
            _logger.debug("Reading %s tab settings failed", self._tab_id, exc_info=True)
            return {}
        if isinstance(payload, Mapping):
            raw = payload.get("values", {})
            if isinstance(raw, Mapping):
                return raw
        return {}

    def collect(self) -> PanelAppearance:
        """Return the appearance currently described by the widgets."""
        return coerce_appearance(
            {
                "foreground": self._fg_button.color(),
                "background": self._bg_button.color(),
                "border_color": self._border_button.color(),
                "border_width": self._width_spin.value(),
                "border_radius": self._radius_spin.value(),
            },
            self._base,
        )

    # -- subclass hooks ---------------------------------------------

    def _extra_values(self) -> dict[str, Any]:
        """Return extra settings keys to persist (default: none)."""
        return {}

    def _apply_extra_live(self) -> str:
        """Apply extra settings to the live widget; return a status suffix."""
        return ""

    # -- handlers ----------------------------------------------------

    def _on_save(self) -> None:
        appearance = self.collect()
        try:
            extra = self._extra_values()
        except ValueError as exc:
            self._status.setText(str(exc))
            return
        values: dict[str, Any] = {**appearance.to_dict(), **extra}
        if not self._persist(values):
            self._status.setText("Could not persist settings.")
            return
        applied = apply_appearance_to_tab(self._sidebar, self._tab_id, appearance)
        suffix = self._apply_extra_live()
        note = " Applied to the live tab." if applied else ""
        self._status.setText("Settings saved." + note + suffix)

    def _persist(self, values: Mapping[str, Any]) -> bool:
        setter = getattr(self._sidebar, "update_tab_settings", None)
        if not callable(setter):
            return False
        try:
            setter(self._tab_id, dict(values))
        except Exception:  # noqa: BLE001 - report failure to the user
            _logger.debug("Persisting %s settings failed", self._tab_id, exc_info=True)
            return False
        return True

    def _on_reset(self) -> None:
        self._apply_to_widgets(self._base)
        self._reset_extra()
        self._status.setText("Reset to defaults — click Save to apply.")

    def _reset_extra(self) -> None:
        """Hook for subclasses to reset extra controls (default: none)."""


# ─── Python REPL settings panel ──────────────────────────────────

_STARTUP_COLUMNS = ("Module", "Alias", "Enabled")


class PythonReplSettingsPanel(AppearanceSettingsPanel):
    """Appearance editor plus the preloaded-package editor for the REPL."""

    def _build_extra(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Preloaded packages", self)
        group_layout = QtWidgets.QVBoxLayout(group)

        hint = QtWidgets.QLabel(
            "Packages imported into every new Python REPL. Missing optional "
            "packages are skipped with a warning, not an error.",
            group,
        )
        hint.setWordWrap(True)
        group_layout.addWidget(hint)

        self._packages_table = QtWidgets.QTableWidget(0, len(_STARTUP_COLUMNS), group)
        self._packages_table.setObjectName("SidekickReplPackagesTable")
        self._packages_table.setHorizontalHeaderLabels(list(_STARTUP_COLUMNS))
        self._packages_table.horizontalHeader().setStretchLastSection(True)
        group_layout.addWidget(self._packages_table)

        button_row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add Package", group)
        add.setObjectName("SidekickReplPackagesAdd")
        add.clicked.connect(lambda: self._add_package_row("", "", enabled=True))
        button_row.addWidget(add)
        remove = QtWidgets.QPushButton("Remove Selected", group)
        remove.setObjectName("SidekickReplPackagesRemove")
        remove.clicked.connect(self._remove_selected_row)
        button_row.addWidget(remove)
        button_row.addStretch(1)
        group_layout.addLayout(button_row)

        layout.addWidget(group)
        self._load_packages(self._current_startup_config())

    # -- package table helpers --------------------------------------

    def _current_startup_config(self) -> CalculatorStartupConfig:
        raw = self._current_values().get("startup_imports")
        if raw in (None, ""):
            return default_repl_startup_config()
        try:
            return CalculatorStartupConfig.from_list(raw)
        except (TypeError, ValueError):
            return default_repl_startup_config()

    def _load_packages(self, config: CalculatorStartupConfig) -> None:
        self._packages_table.setRowCount(0)
        for item in config.imports:
            self._add_package_row(item.module, item.alias, enabled=item.enabled)

    def _add_package_row(self, module: str, alias: str, *, enabled: bool) -> None:
        row = self._packages_table.rowCount()
        self._packages_table.insertRow(row)
        self._packages_table.setItem(row, 0, QtWidgets.QTableWidgetItem(module))
        self._packages_table.setItem(row, 1, QtWidgets.QTableWidgetItem(alias))
        check = QtWidgets.QTableWidgetItem()
        check.setFlags(
            QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        check.setCheckState(
            QtCore.Qt.CheckState.Checked if enabled else QtCore.Qt.CheckState.Unchecked
        )
        self._packages_table.setItem(row, 2, check)

    def _remove_selected_row(self) -> None:
        row = self._packages_table.currentRow()
        if row >= 0:
            self._packages_table.removeRow(row)

    def package_rows(self) -> list[tuple[str, str, bool]]:
        """Return the editor rows as ``(module, alias, enabled)`` triples."""
        rows: list[tuple[str, str, bool]] = []
        for row in range(self._packages_table.rowCount()):
            module_item = self._packages_table.item(row, 0)
            alias_item = self._packages_table.item(row, 1)
            check_item = self._packages_table.item(row, 2)
            module = module_item.text() if module_item else ""
            alias = alias_item.text() if alias_item else ""
            enabled = bool(
                check_item and check_item.checkState() == QtCore.Qt.CheckState.Checked
            )
            rows.append((module, alias, enabled))
        return rows

    def collect_startup_config(self) -> CalculatorStartupConfig:
        """Return a validated config from the editor (raises on bad input)."""
        return parse_startup_rows(self.package_rows())

    # -- hooks -------------------------------------------------------

    def _extra_values(self) -> dict[str, Any]:
        try:
            config = self.collect_startup_config()
        except ValueError as exc:
            raise ValueError(f"Invalid package list: {exc}") from exc
        self._pending_config = config
        return {"startup_imports": config.to_list()}

    def _apply_extra_live(self) -> str:
        config = getattr(self, "_pending_config", None)
        if config is None:
            return ""
        accessor = getattr(self._sidebar, "tab_widget", None)
        widget = accessor(self._tab_id) if callable(accessor) else None
        applier = getattr(widget, "apply_startup_config", None)
        if not callable(applier):
            return " Package changes apply to new REPL sessions."
        result = applier(config)
        if getattr(result, "warnings", ()):  # optional deps not installed
            missing = ", ".join(w.module for w in result.warnings)
            return f" Packages applied (unavailable: {missing})."
        return " Packages applied to the active REPL."

    def _reset_extra(self) -> None:
        self._load_packages(default_repl_startup_config())
