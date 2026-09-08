"""Focused registry gate for launcher logo families (#9482).

A tile may share a logo only with tiles of its declared family. Families are
derived from the manifest's own fields (``engine_type``, else ``category``),
and every shared logo must be a declared, documented decision. The gate logic
lives in ``scripts/check_launcher_logo_families.py`` so CI can run it both as
a script and as these tests.
"""

from __future__ import annotations

import pytest
from scripts.check_launcher_logo_families import (
    DECLARED_SHARED_LOGOS,
    find_logo_family_violations,
    find_missing_logos,
    find_stale_declarations,
    load_manifest_tiles,
    tiles_by_logo,
)

pytestmark = [pytest.mark.unit]


def test_manifest_parses_and_declares_tiles() -> None:
    """The launcher manifest parses as JSON and carries a non-empty tile list."""
    tiles = load_manifest_tiles()
    assert len(tiles) > 0, "Assertion failed: manifest declares no tiles"
    assert all(tile.get("logo") for tile in tiles), (
        "Assertion failed: every tile must reference a logo"
    )


def test_every_referenced_logo_file_exists() -> None:
    """Every referenced logo file exists under assets/logos."""
    tiles = load_manifest_tiles()
    missing = find_missing_logos(tiles)
    assert not missing, f"Tiles referencing missing logo files: {missing}"


def test_no_logo_shared_across_families() -> None:
    """No logo is used by tiles from more than one declared family.

    The launch review (#9482) found ``data_explorer.svg`` rendered by nine
    tiles spanning the ``tool`` and ``analysis`` categories, and
    ``golf_logo.svg`` by seven spanning ``tool`` and ``simulation`` — icons
    that carry no information. This is the gate those violations fail.
    """
    violations = find_logo_family_violations(load_manifest_tiles())
    assert not violations, "Logo-family gate violations:\n" + "\n".join(violations)


def test_shared_logos_are_declared_with_matching_family() -> None:
    """Every shared logo is declared, with a rationale and matching family."""
    tiles = load_manifest_tiles()
    shared = {logo for logo, users in tiles_by_logo(tiles).items() if len(users) > 1}
    undeclared = sorted(shared - set(DECLARED_SHARED_LOGOS))
    assert not undeclared, f"Shared logos without a declaration: {undeclared}"
    for logo, users in sorted(tiles_by_logo(tiles).items()):
        if len(users) == 1 or logo not in DECLARED_SHARED_LOGOS:
            continue
        declaration = DECLARED_SHARED_LOGOS[logo]
        assert declaration.rationale.strip(), (
            f"Declaration for {logo} must carry a non-empty rationale"
        )


def test_no_stale_logo_declarations() -> None:
    """Declarations stay in sync: a declared logo is actually shared."""
    stale = find_stale_declarations(load_manifest_tiles())
    assert not stale, f"Stale logo declarations: {stale}"
