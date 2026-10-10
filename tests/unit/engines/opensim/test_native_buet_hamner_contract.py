"""Portable request-contract checks for the reviewed regional source recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.native_buet_hamner_assembly import (
    BuetHamnerAssemblyRequest,
)

pytestmark = pytest.mark.unit

_BUET_SHA = "b66a31e08087327f8756d587f471b0494a3d9d6f7f6d510de66beba4a9e79bf6"
_HAMNER_SHA = "349ce7a44f1541794f2283cab73bb954270228f2f50808de549ec5097a1687cf"


def test_reviewed_source_declaration_is_immutable_and_paths_are_distinct() -> None:
    request = BuetHamnerAssemblyRequest(
        Path("buet.osim"),
        _BUET_SHA,
        Path("hamner.osim"),
        _HAMNER_SHA,
        Path("fresh.osim"),
    )
    assert request.buet_sha256 == _BUET_SHA
    assert request.hamner_sha256 == _HAMNER_SHA
    with pytest.raises(ValueError, match="distinct"):
        BuetHamnerAssemblyRequest(
            Path("buet.osim"),
            _BUET_SHA,
            Path("hamner.osim"),
            _HAMNER_SHA,
            Path("buet.osim"),
        )


@pytest.mark.parametrize("digest", ["0" * 64, "z" * 64, "A" * 64])
def test_unreviewed_or_noncanonical_source_digest_is_rejected(digest: str) -> None:
    with pytest.raises(ValueError, match="source SHA-256|reviewed source"):
        BuetHamnerAssemblyRequest(
            Path("buet.osim"),
            digest,
            Path("hamner.osim"),
            _HAMNER_SHA,
            Path("fresh.osim"),
        )
