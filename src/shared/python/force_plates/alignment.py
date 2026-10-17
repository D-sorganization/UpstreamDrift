"""Clock alignment between a plate trial and the model time axis (OF-05).

A plate records on its own clock.  :class:`PlateTimeAlignment` states how that
clock maps onto model time and, crucially, where the mapping came from: an
alignment obtained by cross-correlating the plates against a model must never
be used to validate that same model, because the agreement is then built in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from src.shared.python.force_plates.contracts import (
    finite_non_negative,
    finite_real,
    non_empty_text,
)

__all__ = ["AlignmentKind", "PlateTimeAlignment"]

AlignmentKind = Literal["absolute_utc", "shot_impact", "manual", "cross_correlation"]


@dataclass(frozen=True)
class PlateTimeAlignment:
    """Mapping ``model_time = trial_time + offset_s`` with its provenance.

    ``sigma_offset_s`` is the 1-sigma uncertainty of ``offset_s``.  It is a
    common-mode clock error shared by every frame, so it is profiled (see
    ``ReadingOptions.offset_shift_s``) rather than added to per-frame sigmas.
    ``kind`` names how the offset was obtained; ``note`` explains it.
    ``derived_from_model`` names the model identity a ``cross_correlation``
    alignment was fitted against (required for that kind, forbidden otherwise).

    Raises ``TypeError`` for non-numeric numbers and ``ValueError`` for
    non-finite offsets, a negative or non-finite sigma, an unknown ``kind``, an
    empty ``note``, or an inconsistent ``derived_from_model``.
    """

    offset_s: float
    sigma_offset_s: float
    kind: AlignmentKind
    note: str
    derived_from_model: str | None = None

    def __post_init__(self) -> None:
        finite_real(self.offset_s, "offset_s")
        finite_non_negative(self.sigma_offset_s, "sigma_offset_s")
        if self.kind not in get_args(AlignmentKind):
            raise ValueError(
                f"kind must be one of {get_args(AlignmentKind)}, got {self.kind!r}"
            )
        non_empty_text(self.note, "note")
        self._check_derivation()

    def _check_derivation(self) -> None:
        if self.kind == "cross_correlation":
            non_empty_text(self.derived_from_model, "derived_from_model")
        elif self.derived_from_model is not None:
            raise ValueError(
                "derived_from_model must be None unless kind is 'cross_correlation'"
            )

    def trial_time(self, model_time_s: float, shift_s: float = 0.0) -> float:
        """Trial-clock time [s] of ``model_time_s``: ``model - offset - shift``.

        ``shift_s`` profiles the common-mode offset: it is added to
        ``offset_s`` for this call only.  Both inputs must be finite.
        """
        model = finite_real(model_time_s, "model_time_s")
        shift = finite_real(shift_s, "shift_s")
        return model - self.offset_s - shift

    def validates(self, model_identity: str) -> bool:
        """False iff this alignment was derived from ``model_identity``.

        An alignment cross-correlated against a model cannot independently
        validate that model; every other alignment can.
        """
        non_empty_text(model_identity, "model_identity")
        return self.derived_from_model != model_identity
