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

from src.shared.python.motion_matching.execution.grip_selection import (
    GRIP_PRESET_CHOICES,
    GRIP_STYLE_CHOICES,
)

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


# (grip pick, label, min, max, step, decimals, (hand, field) of the resolved grip)
_GRIP_SPINS = (
    ("lead_shaft_rotation", "Lead Hand Rotation (deg, + Stronger)", -60, 60, 1, 1),
    ("trail_shaft_rotation", "Trail Hand Rotation (deg, + Stronger)", -60, 60, 1, 1),
    ("lead_palm_diagonal", "Lead Palm Diagonal (deg)", 5, 85, 1, 1),
    ("trail_palm_diagonal", "Trail Palm Diagonal (deg)", 5, 85, 1, 1),
    ("hand_spacing", "Hand Spacing (m)", 0.01, 0.2, 0.0025, 4),
)
_GRIP_VALUE = {
    "lead_shaft_rotation": lambda g: g.lead.alpha_deg,
    "trail_shaft_rotation": lambda g: g.trail.alpha_deg,
    "lead_palm_diagonal": lambda g: g.lead.psi_deg,
    "trail_palm_diagonal": lambda g: g.trail.psi_deg,
    "hand_spacing": lambda g: g.hand_spacing_m,
}
_HAND_PICKS = tuple(_GRIP_VALUE)
_VISUAL_GRIP_SPINS = (
    (
        "grip_strength_deg",
        "Visual Grip Strength (deg, Picture Only)",
        -30.0,
        30.0,
        1.0,
        1,
    ),
    (
        "lead_roll_offset_deg",
        "Lead Roll Offset (deg, Picture Only)",
        -30.0,
        30.0,
        1.0,
        1,
    ),
    (
        "trail_roll_offset_deg",
        "Trail Roll Offset (deg, Picture Only)",
        -30.0,
        30.0,
        1.0,
        1,
    ),
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
        scroll = QtWidgets.QScrollArea()  # keeps the builder's minimum height
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(240)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        scroll.setWidget(self._build_grip_group())
        layout.addWidget(scroll, 1)
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

    def _build_grip_group(self) -> QtWidgets.QGroupBox:
        """Grip presets and sliders; the summary above is the live preview."""
        group = QtWidgets.QGroupBox("Grip")
        form = QtWidgets.QFormLayout(group)
        self.grip_preset_combo = QtWidgets.QComboBox()
        self.grip_preset_combo.addItem("From the model", None)
        for name in GRIP_PRESET_CHOICES:
            label = name.replace("_", " ").title() if "_" in name else name.title()
            self.grip_preset_combo.addItem(label, name)
        self.grip_preset_combo.currentIndexChanged.connect(self._on_grip_preset)
        form.addRow("Grip Strength", self.grip_preset_combo)
        self.grip_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for name, label, low, high, step, decimals in _GRIP_SPINS:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setObjectName(f"grip_{name}")
            spin.valueChanged.connect(lambda v, n=name: self._on_grip_pick(n, v))
            self.grip_spins[name] = spin
            form.addRow(label, spin)
        self.grip_style_combo = QtWidgets.QComboBox()
        self.grip_style_combo.addItem("From the model", None)
        for name in GRIP_STYLE_CHOICES:
            self.grip_style_combo.addItem(name, name)
        self.grip_style_combo.currentIndexChanged.connect(self._on_grip_style)
        form.addRow("Grip Style", self.grip_style_combo)
        self.grip_thumb_spin = QtWidgets.QDoubleSpinBox()
        self.grip_thumb_spin.setRange(0.0, 45.0)
        self.grip_thumb_spin.setObjectName("grip_grip_thumb_offset_deg")
        self.grip_thumb_spin.setValue(10.0)
        self.grip_thumb_spin.valueChanged.connect(
            lambda v: self._on_grip_pick("grip_thumb_offset_deg", v)
        )
        form.addRow("Lead Thumb Offset (deg, Picture Only)", self.grip_thumb_spin)
        self.visual_grip_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for name, label, low, high, step, decimals in _VISUAL_GRIP_SPINS:
            vspin = QtWidgets.QDoubleSpinBox()
            vspin.setRange(low, high)
            vspin.setSingleStep(step)
            vspin.setDecimals(decimals)
            vspin.setObjectName(f"grip_{name}")
            vspin.setValue(0.0)
            vspin.valueChanged.connect(lambda v, n=name: self._on_grip_pick(n, v))
            self.visual_grip_spins[name] = vspin
            form.addRow(label, vspin)
        reset = QtWidgets.QPushButton("Reset Grip")
        reset.setObjectName("reset_grip")
        reset.clicked.connect(lambda _c=False: self._on_grip_reset())
        form.addRow(reset)
        form.addRow(
            QtWidgets.QLabel(
                "Strength, palm diagonal, spacing and style change the physics: "
                "they make a new model revision that needs a refit."
            )
        )
        return group

    def _sync_grip_spins(self) -> None:
        """Show the resolved grip in the spin boxes without making picks."""
        try:
            grip = self.model.resolved_grip()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if grip is None:
            return
        was, self._syncing = self._syncing, True
        try:
            for name, spin in self.grip_spins.items():
                spin.setValue(_GRIP_VALUE[name](grip))
            if "grip_style" not in self.model.grip_picks():
                idx = self.grip_style_combo.findData(grip.style.value)
                if idx >= 0:
                    self.grip_style_combo.setCurrentIndex(idx)
        finally:
            self._syncing = was

    def _set_grip(self, name: str, value: float | str | None) -> bool:
        try:
            self.model.set_grip(name, value)
        except ValueError as exc:
            self.status.setText(str(exc))
            return False
        self._refresh()
        return True

    def _on_grip_preset(self, _index: int) -> None:
        if self._syncing:
            return
        for name in _HAND_PICKS:  # a preset resets the hands it names
            self.model.set_grip(name, None)
        if self._set_grip("preset", self.grip_preset_combo.currentData()):
            self._sync_grip_spins()

    def _on_grip_pick(self, name: str, value: float) -> None:
        if not self._syncing:
            self._set_grip(name, value)

    def _on_grip_style(self, _index: int) -> None:
        if not self._syncing:
            self._set_grip("grip_style", self.grip_style_combo.currentData())

    def _on_grip_reset(self) -> None:
        for name in self.model.grip_picks():
            self.model.set_grip(name, None)
        self._syncing = True
        self.grip_preset_combo.setCurrentIndex(0)
        self.grip_style_combo.setCurrentIndex(0)
        self.grip_thumb_spin.setValue(10.0)
        for spin in self.visual_grip_spins.values():
            spin.setValue(0.0)
        self._syncing = False
        self._sync_grip_spins()
        self._refresh()

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
        self._sync_grip_spins()

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
