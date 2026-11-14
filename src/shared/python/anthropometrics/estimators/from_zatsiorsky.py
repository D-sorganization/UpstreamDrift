"""Zatsiorsky-Seluyanov (1985) :class:`~anthropometrics.Estimator`.

Loads the published Zatsiorsky-Seluyanov ratios from
:file:`ratios/zatsiorsky_seluyanov_1985.json` at construction
time and emits a :class:`SubjectAnthropometrics` driven by the
shared ratio-table algebra in :mod:`_base`.

These are the *raw* 1985 values, before de Leva's 1996
adjustments. If you want the de Leva-adjusted version, use
:class:`from_de_leva.DeLevaEstimator` instead.
"""

from __future__ import annotations

from pathlib import Path

from ._json_loader import load_ratio_table
from ._ratio_table_estimate import JsonRatioTableEstimator

_DEFAULT_RATIO_FILE = (
    Path(__file__).resolve().parent / "ratios" / "zatsiorsky_seluyanov_1985.json"
)


class ZatsiorskyEstimator(JsonRatioTableEstimator):
    """Estimate :class:`SubjectAnthropometrics` from Zatsiorsky-Seluyanov 1985."""

    def __init__(self, ratio_file: Path | None = None) -> None:
        """Load the ratio table (defaults to the bundled JSON)."""
        path = ratio_file if ratio_file is not None else _DEFAULT_RATIO_FILE
        self._table = load_ratio_table(path)
