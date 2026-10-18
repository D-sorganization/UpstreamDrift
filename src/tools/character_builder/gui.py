"""PyQt6 shell for the Character Builder (CMB-3, #11654).

All behaviour lives in :class:`~.core.CharacterBuilderModel`; this module only
binds widgets to it.
"""

from __future__ import annotations

from typing import Any

from PyQt6 import QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.club_models import CLUBS

from src.shared.python.model_appearance import library as appearance_library_mod
from src.shared.python.model_appearance.appearance_selection import rgb_to_hex
from src.shared.python.model_appearance.schema import document_from_dict
from src.shared.python.ui.hex_color_field import ColorPicker, HexColorField

from .core import PARAMETER_RANGES, CharacterBuilderModel

logger = get_logger(__name__)

__all__ = ["CharacterBuilderWidget"]

# noqa: gui-thread/ok - compiling a spec takes about 50 ms and exports write one
# file, so the handlers run inline rather than through AsyncActionBar.

_LABELS = {
    "stature_m": "Stature (m)",
    "mass_kg": "Mass (kg)",
    "trunk_scale": "Trunk Length Scale",
    "arm_scale": "Arm Length Scale",
    "shoulder_scale": "Shoulder Breadth Scale",
    "grip_roll_deg": "Grip Roll (deg)",
}
_EXPORTS = (
    ("spec", "Export Spec JSON"),
    ("urdf", "Export URDF"),
    ("mjcf", "Export MJCF"),
    ("osim", "Export OpenSim"),
    ("gltf", "Export glTF"),
    ("appearance", "Export Appearance"),
)
# (appearance choice name, appearance_library() key, combo label)
_APPEARANCE_FIELDS = (
    ("skin_tone", "skin_tones", "Skin Tone"),
    ("clothing", "clothing", "Clothing"),
    ("club_finish", "club_finishes", "Club Finish"),
    ("headwear", "headwear", "Headwear"),
    ("ground_material", "ground_materials", "Ground"),
    ("glove", "glove", "Glove"),
)
# (appearance colour choice, label) written as document ``materials`` entries
_COLOUR_FIELDS = (
    ("skin_color", "Custom Skin Color"),
    ("hair_color", "Hair Color"),
    ("top_color", "Shirt Color"),
    ("bottom_color", "Trousers/Shorts Color"),
)


def _library_options(library: dict[str, Any], key: str) -> list[str]:
    """Pickable names for one appearance field (``list()`` keys a dict too)."""
    return list(library[key])


class CharacterBuilderWidget(QtWidgets.QWidget):
    """Preset picker, parameter form, live summary and export buttons."""

    def __init__(
        self, parent: Any = None, *, color_picker: ColorPicker | None = None
    ) -> None:
        super().__init__(parent)
        self._color_picker = color_picker
        self.model = CharacterBuilderModel()
        self._syncing = False
        self._build_ui()
        self._load_preset_into_form(self.preset_combo.currentData())

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Custom", None)
        for preset_id in self.model.preset_ids():
            self.preset_combo.addItem(preset_id.replace("_", " ").title(), preset_id)
        form.addRow("Preset", self.preset_combo)
        self.spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for name, (low, high) in PARAMETER_RANGES.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(3 if name != "mass_kg" else 1)
            spin.setSingleStep(0.01 if name != "mass_kg" else 1.0)
            spin.setObjectName(f"spin_{name}")
            spin.valueChanged.connect(lambda v, n=name: self._on_spin(n, v))
            self.spins[name] = spin
            form.addRow(_LABELS[name], spin)
        self.club_combo = QtWidgets.QComboBox()
        self.club_combo.addItems(sorted(CLUBS))
        self.club_combo.currentTextChanged.connect(
            lambda text: self._on_edit("club", text)
        )
        form.addRow("Club", self.club_combo)
        layout.addLayout(form)
        layout.addWidget(self._build_appearance_group())
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary, 1)
        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        self.export_buttons: dict[str, QtWidgets.QPushButton] = {}
        for fmt, text in _EXPORTS:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _checked=False, f=fmt: self.export(f))
            self.export_buttons[fmt] = button
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.preset_combo.currentIndexChanged.connect(
            lambda _i: self._load_preset_into_form(self.preset_combo.currentData())
        )

    def _build_appearance_group(self) -> QtWidgets.QGroupBox:
        """Skin/clothing/club/headwear/ground combos, defaults from the library."""
        group = QtWidgets.QGroupBox("Appearance")
        form = QtWidgets.QFormLayout(group)
        library = self.model.appearance_library()
        defaults = self.model.appearance_document()
        current = {
            "skin_tone": defaults["skin_tone"],
            "clothing": defaults["clothing"],
            "club_finish": defaults["club_finish"],
            "headwear": defaults["head"]["headwear"],
            "ground_material": defaults["environment"]["ground_material"],
            "glove": defaults.get("glove", "none"),
        }
        self.appearance_combos: dict[str, QtWidgets.QComboBox] = {}
        for name, library_key, label in _APPEARANCE_FIELDS:
            combo = QtWidgets.QComboBox()
            combo.addItems(_library_options(library, library_key))
            combo.setCurrentText(current[name])
            combo.currentTextChanged.connect(
                lambda text, n=name: self._on_appearance_change(n, text)
            )
            self.appearance_combos[name] = combo
            form.addRow(label, combo)
        self.colour_fields: dict[str, HexColorField] = {}
        for name, label in _COLOUR_FIELDS:
            form.addRow(label, self._build_colour_row(name))
        return group

    def _current_colours(self) -> dict[str, str]:
        """Hex of the document's current skin, hair, shirt and trousers."""
        doc = document_from_dict(self.model.appearance_document())
        mats = appearance_library_mod.library_materials(doc)
        preset = appearance_library_mod.CLOTHING[doc.clothing]
        hair = appearance_library_mod.headwear_material_name(doc)
        names = {
            "skin_color": doc.skin_tone,
            "hair_color": hair
            or appearance_library_mod.HEADWEAR_DEFAULT_MATERIAL["hair"],
            "top_color": preset.get("torso", doc.skin_tone),
            "bottom_color": preset.get(
                appearance_library_mod.PANTS_PART, doc.skin_tone
            ),
        }
        return {k: rgb_to_hex(mats[v].rgba) for k, v in names.items()}

    def _build_colour_row(self, name: str) -> QtWidgets.QWidget:
        """Colour field plus a Reset button that clears the custom choice."""
        field = HexColorField(
            self._current_colours()[name],
            object_name=f"colour_{name}",
            color_picker=self._color_picker,
        )
        field.colour_changed.connect(
            lambda hex_, n=name: self._on_colour_change(n, hex_)
        )
        field.line_edit.editingFinished.connect(
            lambda n=name, f=field: self._on_colour_change(n, f.text())
        )
        self.colour_fields[name] = field
        reset = QtWidgets.QPushButton("Reset")
        reset.setObjectName(f"reset_{name}")
        reset.clicked.connect(lambda _c=False, n=name: self._on_colour_reset(n))
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        layout.addWidget(reset)
        return row

    def _on_colour_change(self, name: str, value: str) -> None:
        if value == self._current_colours()[name] and name not in (
            self.model.appearance_choices()
        ):
            return  # focus loss without an edit is not a custom choice
        self._on_appearance_change(name, value)

    def _on_colour_reset(self, name: str) -> None:
        try:
            self.model.set_appearance(name, None)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        field = self.colour_fields[name]
        field.line_edit.setText(self._current_colours()[name])
        self.status.setText("")

    def _on_appearance_change(self, name: str, value: str) -> None:
        try:
            self.model.set_appearance(name, value)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("")

    def _load_preset_into_form(self, preset_id: str | None) -> None:
        if preset_id is not None:
            self.model.apply_preset(preset_id)
        self._syncing = True
        try:
            values = self.model.parameters_dict()
            for name, spin in self.spins.items():
                spin.setValue(values[name])
            self.club_combo.setCurrentText(values["club"])
        finally:
            self._syncing = False
        self._refresh()

    def _on_spin(self, name: str, value: float) -> None:
        self._on_edit(name, value)

    def _on_edit(self, name: str, value: float | str) -> None:
        if self._syncing:
            return
        try:
            self.model.set_parameter(name, value)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._syncing = True
        self.preset_combo.setCurrentIndex(0)
        self._syncing = False
        self._refresh()

    def _refresh(self) -> None:
        try:
            self.summary.setPlainText(self.model.summary_text())
            self.status.setText("")
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Character compile failed: %s", exc)
            self.summary.setPlainText("")
            self.status.setText(f"Cannot compile: {exc}")

    def export(self, fmt: str, directory: str | None = None) -> str | None:
        """Export in ``fmt``; asks for a folder unless ``directory`` is given."""
        if directory is None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose Export Folder"
            )
        if not directory:
            return None
        try:
            path = self.model.export(fmt, directory)
        except (ValueError, FileNotFoundError, OSError) as exc:
            logger.warning("Character export failed: %s", exc)
            self.status.setText(f"Export failed: {exc}")
            return None
        self.status.setText(f"Wrote {path}")
        return str(path)

    def cleanup(self) -> None:
        """Nothing to release; kept idempotent for the embed contract."""
