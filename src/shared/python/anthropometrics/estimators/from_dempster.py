"""Dempster (1955) :class:`~anthropometrics.Estimator` implementation.

Loads the published Dempster ratios from
:file:`ratios/dempster_1955.json` at construction time and emits
a :class:`SubjectAnthropometrics` driven by the shared
ratio-table algebra in :mod:`_base`.
"""

from __future__ import annotations

from pathlib import Path

from ._json_loader import load_ratio_table
from ._ratio_table_estimate import JsonRatioTableEstimator

_DEFAULT_RATIO_FILE = Path(__file__).resolve().parent / "ratios" / "dempster_1955.json"


class DempsterEstimator(JsonRatioTableEstimator):
    """Estimate :class:`SubjectAnthropometrics` from Dempster (1955).

    The published Dempster ratios live entirely in
    :file:`ratios/dempster_1955.json` (loaded once at
    construction). The estimator is sex-agnostic — Dempster's
    cadaveric study did not publish a sex breakdown, so the same
    ratios are used for ``"M"``, ``"F"``, and ``"unspecified"``.
    """

    def __init__(self, ratio_file: Path | None = None) -> None:
        """Load the ratio table (defaults to the bundled JSON)."""
        path = ratio_file if ratio_file is not None else _DEFAULT_RATIO_FILE
        self._table = load_ratio_table(path)
