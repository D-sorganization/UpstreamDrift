"""Self-documenting numeric/enum input widget (epic #5968, Phase 2).

:class:`HelpfulField` is a thin PyQt6 wrapper that consumes the
*existing* :class:`~src.shared.python.ux.field_metadata.FieldMetadata`
registry and configures a single input control so that help copy,
units, valid range and the default-source attribution all flow from one
source (DRY — no metadata logic is duplicated here).

The wrapper:

* sets ``toolTip`` from ``short_help`` and ``whatsThis`` from
  ``long_help`` + ``default_source`` (the ``[?]`` affordance copy);
* derives a validator / spinbox range from ``valid_range``;
* emits :attr:`field_violated` (``field_id``, ``value``) whenever a
  checked value breaches the declared numeric range.

It intentionally does *not* draw the popover or own the workflow spine
— that look/affordance work is a human design decision (deferred).

DbC: the constructor validates its inputs at the boundary and raises a
descriptive ``KeyError``/``TypeError`` for unknown ids or bad types.
LoD: callers read help text and values directly off the wrapper rather
than reaching through ``field._registry.get(id).short_help``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QWidget

from src.shared.python.contracts import require
from src.shared.python.ux.field_metadata import FieldMetadata, FieldRegistry

if TYPE_CHECKING:
    from src.shared.python.ux.field_metadata import ValidRange


def _is_numeric_range(rng: ValidRange) -> bool:
    return (
        rng is not None
        and len(rng) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in rng)
    )


def _build_whats_this(fm: FieldMetadata) -> str:
    """Compose the long-help body shown by the ``[?]`` affordance."""
    parts = [fm.long_help.rstrip()]
    if fm.units:
        parts.append(f"Units: {fm.units}")
    if fm.example:
        parts.append(f"Example: {fm.example}")
    parts.append(f"Default source: {fm.default_source}")
    return "\n\n".join(parts)


class HelpfulField(QWidget):
    """A metadata-driven input wrapper.

    Parameters
    ----------
    field_id
        Dotted id of a field present in ``registry``.
    registry
        A validated :class:`FieldRegistry`.  The field's metadata is the
        single source for tooltip, whats-this, range and default.
    parent
        Optional Qt parent.

    Signals
    -------
    field_violated(str, float)
        Emitted by :meth:`check_value` when ``value`` falls outside the
        field's numeric ``valid_range``.  Enum and free-form fields never
        emit (they have no numeric bounds).
    """

    field_violated = pyqtSignal(str, float)

    def __init__(
        self,
        field_id: str,
        *,
        registry: FieldRegistry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        require(
            isinstance(registry, FieldRegistry),
            "HelpfulField requires a FieldRegistry",
            registry,
        )
        # ``registry.get`` raises KeyError for unknown ids (DbC boundary).
        self._metadata: FieldMetadata = registry.get(field_id)
        self._field_id = field_id
        self._editor = self._build_editor(self._metadata)
        self._apply_help(self._metadata)

    # ---- construction helpers --------------------------------------

    def _build_editor(self, fm: FieldMetadata) -> QWidget:
        if _is_numeric_range(fm.valid_range):
            lo, hi = fm.valid_range  # type: ignore[misc]
            spin = QDoubleSpinBox(self)
            spin.setRange(float(lo), float(hi))
            spin.setDecimals(6)
            if isinstance(fm.default, (int, float)) and not isinstance(
                fm.default, bool
            ):
                spin.setValue(float(fm.default))
            if fm.units:
                spin.setSuffix(f" {fm.units}")
            return spin
        if fm.valid_range is not None:
            combo = QComboBox(self)
            combo.addItems([str(v) for v in fm.valid_range])
            idx = combo.findText(str(fm.default))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            return combo
        line = QLineEdit(self)
        if fm.default is not None:
            line.setText(str(fm.default))
        return line

    def _apply_help(self, fm: FieldMetadata) -> None:
        self.setToolTip(fm.short_help)
        self.setWhatsThis(_build_whats_this(fm))
        self._editor.setToolTip(fm.short_help)
        self._editor.setWhatsThis(_build_whats_this(fm))
        self.setAccessibleName(fm.label)
        self._editor.setAccessibleName(fm.label)

    # ---- public surface (LoD) --------------------------------------

    @property
    def field_id(self) -> str:
        """The dotted id this field is bound to."""
        return self._field_id

    @property
    def metadata(self) -> FieldMetadata:
        """The bound :class:`FieldMetadata` (read-only, frozen)."""
        return self._metadata

    def editor(self) -> QWidget:
        """Return the underlying input control."""
        return self._editor

    def value(self) -> float | str:
        """Return the current editor value (float for numeric fields)."""
        if isinstance(self._editor, QDoubleSpinBox):
            return self._editor.value()
        if isinstance(self._editor, QLineEdit):
            return self._editor.text()
        return self._editor.currentText()

    def check_value(self, value: float) -> bool:
        """Validate ``value`` against the numeric range.

        Returns ``True`` if in range.  Emits :attr:`field_violated`
        and returns ``False`` on breach.  Non-numeric fields are always
        considered in range (they have no bounds).
        """
        rng = self._metadata.valid_range
        if not _is_numeric_range(rng):
            return True
        lo, hi = rng  # type: ignore[misc]
        if float(lo) <= float(value) <= float(hi):
            return True
        self.field_violated.emit(self._field_id, float(value))
        return False


__all__ = ["HelpfulField"]
