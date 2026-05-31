"""Provenance-aware value label (epic #5968, Phase 2).

:class:`ProvenanceValueLabel` is a thin PyQt6 wrapper around the
*existing* :class:`~src.shared.python.ux.provenance.ProvenanceValue`
dataclass (DRY).  It renders the value (with display units) as a label
and exposes the value's :meth:`ProvenanceValue.describe` text via
``toolTip`` and ``whatsThis`` so a hover or right-click answers "why
does this say 500?" without leaving the screen.

When the underlying record has inputs, the label is flagged "linked"
(:meth:`is_linked`) so the host can render a small link affordance; the
exact glyph/badge styling is a human design decision (deferred).

This module owns no provenance logic — it formats and surfaces what the
dataclass already computes (DRY, LoD).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QWidget

from src.shared.python.ux.provenance import ProvenanceValue


class ProvenanceValueLabel(QLabel):
    """A read-only label that carries its value's provenance.

    Parameters
    ----------
    provenance_value
        The :class:`ProvenanceValue` to display.
    parent
        Optional Qt parent.

    Raises
    ------
    TypeError
        If ``provenance_value`` is not a :class:`ProvenanceValue`
        (DbC boundary check).
    """

    def __init__(
        self,
        provenance_value: ProvenanceValue,
        parent: QWidget | None = None,
    ) -> None:
        if not isinstance(provenance_value, ProvenanceValue):
            raise TypeError(
                "ProvenanceValueLabel requires a ProvenanceValue, got "
                f"{type(provenance_value).__name__}"
            )
        super().__init__(parent)
        self._pv = provenance_value
        self.setText(self._format_text(provenance_value))
        self.setToolTip(provenance_value.describe())
        self.setWhatsThis(provenance_value.describe())
        if provenance_value.label:
            self.setAccessibleName(provenance_value.label)

    @staticmethod
    def _format_text(pv: ProvenanceValue) -> str:
        if pv.display_units:
            return f"{pv.value} {pv.display_units}"
        return f"{pv.value}"

    @property
    def provenance_value(self) -> ProvenanceValue:
        """The bound :class:`ProvenanceValue` (read-only)."""
        return self._pv

    def is_linked(self) -> bool:
        """Return ``True`` when the value derives from named inputs.

        A linked value can be traced to other fields; the host UI should
        render a link/badge affordance.  Constants (no inputs) are not
        linked.
        """
        return bool(self._pv.record.inputs)


__all__ = ["ProvenanceValueLabel"]
