"""NSX-92 anthro receipt acceptance (blocked on spec decisions, #12721)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.xfail(
    strict=True,
    reason=(
        "NSX-92 anthro NoiseAwareModel and receipt require owner decisions 1–4 "
        "on #12721; allocation core is frozen per issue scope."
    ),
)
def test_anthro_noise_aware_receipt_is_unvalidated_and_replay_verified() -> None:
    """Placeholder: fails until anthro receipt + replay land on main."""
    assert False, "spec pending on #12721"
