"""Qt widget displaying a single :class:`SegmentProperties` instance.

The panel is a self-contained ``QGroupBox`` that surfaces every field
of a :class:`SegmentProperties` for the currently-selected segment in
either the C3D Viewer or the Motion-Match Preview matcher. It is a
**pure view** — it does not own the data and never mutates it.

Number formatting policy
------------------------
* ``length_m``      — 3 decimal places, fixed-point, suffix ``"m"``.
* ``mass_kg``       — 3 decimal places, fixed-point, suffix ``"kg"``.
* ``com_xyz_m``     — 3 decimal places per component, fixed-point.
* ``inertia_tensor`` — 4 significant figures, scientific notation.
* Principal moments — 4 significant figures, scientific notation,
  sorted ascending (computed via ``numpy.linalg.eigvalsh``).

Lazy / optional wiring
----------------------
Both host applications (C3D Viewer's ``viewer_3d_tab`` and the
matcher's ``live_view_controller``) call :meth:`set_segment` when a
selection changes. Passing ``None`` collapses the panel to a clear
"no selection" state — every value field shows the placeholder
``"—"`` (em dash) and the header reads ``"Segment Properties"``.

Design by Contract
------------------
:meth:`set_segment` raises :class:`TypeError` if its argument is
neither ``None`` nor a :class:`SegmentProperties` instance. The
constructor takes no required arguments — a freshly-instantiated
panel is in the "no selection" state.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..segment_properties import SegmentProperties

__all__ = ["SegmentPropertiesPanel"]

_PLACEHOLDER = "—"  # em-dash
_HEADER_DEFAULT = "Segment Properties"
_LENGTH_FMT = "{value:.3f} m"
_MASS_FMT = "{value:.3f} kg"
_COM_FMT = "({x:+.3f}, {y:+.3f}, {z:+.3f}) m"
_TENSOR_CELL_FMT = "{value:+.4e}"
_PRINCIPAL_FMT = "{value:.4e}"


def _monospace_font() -> QFont:
    """Return a monospaced font for inertia-tensor + principal-moment cells."""
    font = QFont("Courier New")
    font.setStyleHint(QFont.StyleHint.TypeWriter)
    font.setFixedPitch(True)
    return font


class SegmentPropertiesPanel(QGroupBox):
    """Read-only panel rendering one :class:`SegmentProperties` value.

    The panel never edits the object it displays; callers wishing to
    update the view simply call :meth:`set_segment` with a new
    instance (or ``None`` to clear).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_HEADER_DEFAULT, parent)
        self._mono_font = _monospace_font()
        self._build_ui()
        self._segment: SegmentProperties | None = None
        # Start in cleared state so labels are non-empty before first
        # ``set_segment`` call.
        self.set_segment(None)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @property
    def current_segment(self) -> SegmentProperties | None:
        """The :class:`SegmentProperties` currently displayed (or ``None``)."""
        return self._segment

    def set_segment(self, props: SegmentProperties | None) -> None:
        """Update the panel to reflect *props*.

        Args:
            props: The :class:`SegmentProperties` to display, or
                ``None`` to clear the panel to its "no selection"
                state.

        Raises:
            TypeError: If *props* is neither ``None`` nor a
                :class:`SegmentProperties` instance (DbC).
        """
        if props is not None and not isinstance(props, SegmentProperties):
            raise TypeError(
                f"props must be a SegmentProperties or None, got {type(props).__name__}"
            )
        self._segment = props
        if props is None:
            self._render_empty()
        else:
            self._render_segment(props)

    # ------------------------------------------------------------------ #
    # UI construction                                                    #
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._name_label = QLabel(_PLACEHOLDER)
        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSizeF(name_font.pointSizeF() + 1.0)
        self._name_label.setFont(name_font)
        self._name_label.setObjectName("segment_name_label")
        outer.addWidget(self._name_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        outer.addLayout(form)

        self._length_label = self._make_value_label("length_value")
        self._mass_label = self._make_value_label("mass_value")
        self._com_label = self._make_value_label("com_value")
        self._source_method_label = self._make_value_label("source_method_value")
        self._source_subject_label = self._make_value_label("source_subject_value")

        form.addRow("Length:", self._length_label)
        form.addRow("Mass:", self._mass_label)
        form.addRow("CoM offset:", self._com_label)
        form.addRow("Source method:", self._source_method_label)
        form.addRow("Subject params:", self._source_subject_label)

        # Inertia tensor — 3x3 grid of monospaced labels.
        tensor_box = QGroupBox("Inertia tensor (kg·m²)")
        tensor_box.setObjectName("inertia_tensor_box")
        tensor_layout = QGridLayout(tensor_box)
        tensor_layout.setHorizontalSpacing(10)
        tensor_layout.setVerticalSpacing(2)
        tensor_layout.setContentsMargins(8, 8, 8, 8)
        self._tensor_cells: list[list[QLabel]] = []
        for r in range(3):
            row_cells: list[QLabel] = []
            for c in range(3):
                cell = QLabel(_PLACEHOLDER)
                cell.setFont(self._mono_font)
                cell.setAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                cell.setObjectName(f"inertia_cell_{r}_{c}")
                cell.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
                )
                tensor_layout.addWidget(cell, r, c)
                row_cells.append(cell)
            self._tensor_cells.append(row_cells)
        outer.addWidget(tensor_box)

        # Principal moments.
        principal_row = QFormLayout()
        principal_row.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._principal_label = QLabel(_PLACEHOLDER)
        self._principal_label.setObjectName("principal_moments_value")
        self._principal_label.setFont(self._mono_font)
        principal_row.addRow("Principal moments:", self._principal_label)
        outer.addLayout(principal_row)

        outer.addStretch(1)

    def _make_value_label(self, object_name: str) -> QLabel:
        """Return a read-only value ``QLabel`` configured for the panel."""
        label = QLabel(_PLACEHOLDER)
        label.setObjectName(object_name)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    # ------------------------------------------------------------------ #
    # Rendering                                                          #
    # ------------------------------------------------------------------ #
    def _render_empty(self) -> None:
        """Reset every label to the empty/placeholder state."""
        self.setTitle(_HEADER_DEFAULT)
        self._name_label.setText(_PLACEHOLDER)
        for label in (
            self._length_label,
            self._mass_label,
            self._com_label,
            self._source_method_label,
            self._source_subject_label,
            self._principal_label,
        ):
            label.setText(_PLACEHOLDER)
        for row in self._tensor_cells:
            for cell in row:
                cell.setText(_PLACEHOLDER)

    def _render_segment(self, props: SegmentProperties) -> None:
        """Populate every label from *props*."""
        self.setTitle(f"Segment Properties — {props.name}")
        self._name_label.setText(props.name)
        self._length_label.setText(_LENGTH_FMT.format(value=float(props.length_m)))
        self._mass_label.setText(_MASS_FMT.format(value=float(props.mass_kg)))

        com = np.asarray(props.com_xyz_m, dtype=float).reshape(3)
        self._com_label.setText(
            _COM_FMT.format(x=float(com[0]), y=float(com[1]), z=float(com[2]))
        )

        self._source_method_label.setText(props.source_method)
        self._source_subject_label.setText(
            f"height={props.source_subject_height_m:.3f} m, "
            f"mass={props.source_subject_mass_kg:.3f} kg"
        )

        tensor = np.asarray(props.inertia_tensor, dtype=float).reshape(3, 3)
        for r in range(3):
            for c in range(3):
                self._tensor_cells[r][c].setText(
                    _TENSOR_CELL_FMT.format(value=float(tensor[r, c]))
                )

        # Principal moments — sorted ascending eigenvalues.
        eigenvalues = np.linalg.eigvalsh(tensor)
        eigenvalues = np.sort(eigenvalues)
        principal_text = "  ".join(
            _PRINCIPAL_FMT.format(value=float(v)) for v in eigenvalues
        )
        self._principal_label.setText(principal_text)
