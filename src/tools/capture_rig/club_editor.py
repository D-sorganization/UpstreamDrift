"""Edit player equipment while retaining sources and unknown specifications."""

from __future__ import annotations

from typing import cast

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)
from sidekick.ui.widgets.unit_aware_input import UnitAwareInput

from src.shared.python.club_data.catalog import (
    ClaimStatus,
    ClubIdentity,
    ClubRecord,
    Component,
    Property,
    PropertyClaim,
    SpecificationSource,
)
from src.shared.python.club_data.player_clubs import PlayerClub

from .dialog_controls import save_cancel_buttons

from . import styling


class EquipmentQuantity(QWidget):
    """Evidence selection wraps the fleet's canonical-SI unit control."""

    def __init__(self, club: PlayerClub, prop: Property, component: Component) -> None:
        super().__init__()
        self.key = (prop, component)
        self._original = next(
            (c for c in club.overrides if (c.property, c.component) == self.key), None
        )
        claims = [
            c
            for c in club.effective_record().claims
            if (c.property, c.component) == self.key
        ]
        claim = claims[0] if len(claims) == 1 else None
        self.mode = QComboBox()
        self.mode.addItems(["Use Original", "Measured", "Estimated", "Unknown"])
        self.input = UnitAwareInput(
            "length" if prop == "length" else "mass",
            decimals=6,
            default_unit="in" if prop == "length" else "g",
            min_value=0,
        )
        self.input.setAccessibleName(f"{component} {prop}")
        self.mode.setAccessibleName(f"{component} {prop} evidence")
        self.input.set_value(
            claim.si_value() if claim and claim.value else 0, is_si=True
        )
        self._initial_si = self.input.value_si()
        if self._original and self._original.status in {
            "measured",
            "estimated",
            "unknown",
        }:
            status = self._original.status
            self.mode.setCurrentText(status.title())
        self._initial_mode = self.mode.currentText()
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        if self._original and self._initial_mode != "Use Original":
            claims = [
                c for c in club.base.claims if (c.property, c.component) == self.key
            ]
            claim = claims[0] if len(claims) == 1 else None
        self._original_text = (
            f"{claim.value:g} {claim.unit} ({claim.status})"
            if claim is not None and claim.value is not None
            else "Conflicting sources — review before use"
            if len(claims) > 1
            else "Unknown"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.mode)
        layout.addWidget(self.input)
        layout.addWidget(self.summary, 1)
        self.mode.currentTextChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        editable = self.mode.currentText() in {"Measured", "Estimated"}
        self.input.setVisible(editable)
        self.summary.setVisible(not editable)
        self.summary.setText(
            "Unknown" if self.mode.currentText() == "Unknown" else self._original_text
        )

    def claim(self) -> PropertyClaim | None:
        mode, value = self.mode.currentText(), self.input.value_si()
        if mode == self._initial_mode and value == self._initial_si:
            return self._original
        if mode == "Use Original":
            return None
        prop, component = self.key
        return PropertyClaim(
            property=prop,
            component=component,
            value=None if mode == "Unknown" else value,
            unit="m" if prop == "length" else "kg",
            status=cast(ClaimStatus, mode.lower()),
            source=SpecificationSource(
                kind="player",
                title=f"Player {mode.lower()} specification",
                license="user-provided",
                method="Entered in My Clubs",
            ),
        )


class ClubEditorDialog(QDialog):
    """Save is explicit; cancelling discards the draft without changing the bag."""

    def __init__(
        self, club: PlayerClub | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._club = club or PlayerClub(
            label="My Club",
            base=ClubRecord(identity=ClubIdentity(model="Custom", club_type="other")),
        )
        self.setWindowTitle("Edit Club" if club else "Add Custom Club")
        self.resize(640, 500)
        self.label = QLineEdit(self._club.label)
        self.label.setMaxLength(200)
        identity = self._club.identity
        self.number = QLineEdit(identity.number or "")
        self.number.setMaxLength(40)
        self.club_type = QComboBox()
        self.club_type.addItems(
            ["driver", "wood", "hybrid", "iron", "wedge", "putter", "other"]
        )
        self.club_type.setCurrentText(identity.club_type)
        self._custom = not identity.manufacturer
        self.number.setReadOnly(not self._custom)
        self.club_type.setEnabled(self._custom)
        self.length = EquipmentQuantity(self._club, "length", "assembled")
        self.head_mass = EquipmentQuantity(self._club, "mass", "head")
        self.notes = QPlainTextEdit(self._club.notes)
        self.notes.setAccessibleName("Club notes")
        self.notes.setPlaceholderText(
            "Shaft, grip, measurement method, fitting or lesson notes…"
        )
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        for text, field in (
            ("Name", self.label),
            ("Club Type", self.club_type),
            ("Number / Loft Label", self.number),
            ("Playing Length", self.length),
            ("Head Mass", self.head_mass),
        ):
            form.addRow(text, field)
        layout.addLayout(form)
        hint = QLabel(
            "Use Original retains the source specification. Unknown clears its use for this club. "
            "Choose Measured or Estimated only for values you know. Club dimensions alone cannot calibrate cameras."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.notes, 1)
        layout.addWidget(self.status)
        buttons = save_cancel_buttons(self)
        layout.addWidget(buttons)
        styling.apply_theme(self)

    def record(self) -> PlayerClub:
        base = self._club.base
        if self._custom:
            data = base.identity.model_dump()
            data.update(
                number=self.number.text().strip() or None,
                club_type=self.club_type.currentText(),
            )
            base = ClubRecord(**{**base.model_dump(), "identity": ClubIdentity(**data)})
        fields = (self.length, self.head_mass)
        keys = {field.key for field in fields}
        replacement = {field.key: field.claim() for field in fields}
        overrides = []
        for claim in self._club.overrides:
            key = (claim.property, claim.component)
            updated = replacement.pop(key, claim) if key in keys else claim
            if updated is not None:
                overrides.append(updated)
        overrides.extend(claim for claim in replacement.values() if claim is not None)
        return PlayerClub(
            club_id=self._club.club_id,
            label=self.label.text(),
            base=base,
            overrides=tuple(overrides),
            notes=self.notes.toPlainText(),
            archived=self._club.archived,
        )

    def accept(self) -> None:
        try:
            self.record()
        except ValueError as exc:
            self.status.setText(f"Club was not saved: {exc}")
            return
        super().accept()
