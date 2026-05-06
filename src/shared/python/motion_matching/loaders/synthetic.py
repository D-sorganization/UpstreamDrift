"""Synthetic-target stub.

The full implementation depends on issues #014 (Simscape callback wiring) and
#018 (``simulate_with_coefficients``). This stub keeps the public signature
stable so callers can already target it; it raises ``NotImplementedError`` so
nobody accidentally relies on a no-op result.
"""

from __future__ import annotations

import logging

import numpy as np

from ..club_target import AlignOptions, ClubTarget

logger = logging.getLogger(__name__)


def synthesize_target_from_coefficients(
    theta: np.ndarray, opts: AlignOptions
) -> ClubTarget:
    """Build a ``ClubTarget`` from a known coefficient vector ``theta``.

    Pending issues #014 and #018. The function signature matches
    ``CLUB_IK_SPEC.md`` so optimizer code can already type-check against it.

    Raises:
        NotImplementedError: always, until #014/#018 land.
    """
    _ = (np.asarray(theta), opts)
    logger.debug(
        "synthesize_target_from_coefficients called with theta of shape %s",
        np.asarray(theta).shape,
    )
    raise NotImplementedError(
        "synthesize_target_from_coefficients depends on #014 (Simscape "
        "callback) and #018 (simulate_with_coefficients). Stub only."
    )
