"""Regression tests for #3676.

The mmHg->Pa conversion factor had four representations, two of which
differed in value: the canonical ``MMHG_TO_PASCAL`` (133.322387415) and
truncated 133.322 copies in ``MMHG_TO_PA_CONV`` (constants.py),
``MMHG_TO_PASCAL_FACTOR`` (steam_engine.py), and bare literals in the
calc_backend syngas-water router. They are now all aliases of the single
full-precision constant.
"""

from __future__ import annotations

from sidekick.calculators.thermo.steam_engine import MMHG_TO_PASCAL_FACTOR
from sidekick.process_calculators.constants import MMHG_TO_PA, MMHG_TO_PA_CONV
from sidekick.utils.unit_constants import MMHG_TO_PASCAL

CANONICAL = 133.322387415


def test_canonical_constant_is_full_precision() -> None:
    assert MMHG_TO_PASCAL == CANONICAL


def test_mmhg_to_pa_conv_is_canonical_alias() -> None:
    """constants.MMHG_TO_PA_CONV must not be the truncated 133.322 literal."""
    assert MMHG_TO_PA_CONV == CANONICAL
    assert MMHG_TO_PA_CONV != 133.322


def test_mmhg_to_pa_alias_matches() -> None:
    assert MMHG_TO_PA == CANONICAL


def test_steam_engine_factor_is_canonical() -> None:
    """steam_engine.MMHG_TO_PASCAL_FACTOR references the shared constant."""
    assert MMHG_TO_PASCAL_FACTOR == CANONICAL
    assert MMHG_TO_PASCAL_FACTOR != 133.322
