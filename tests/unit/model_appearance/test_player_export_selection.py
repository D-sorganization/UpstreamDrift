"""Catalog selection must preserve the legacy pair and support shared previews."""

import pytest

from scripts.export_player_models import _selected_players

pytestmark = pytest.mark.unit


def test_expanded_selection_is_ordered_unique_and_preserves_legacy_pair():
    """New IDs share one rendering invocation without changing ``both``."""
    catalog = ("tiger_woods", "ben_hogan", "sam_snead", "jack_nicklaus")
    assert _selected_players(("both",), catalog) == catalog[:2]
    assert _selected_players(("all",), catalog) == catalog
    assert _selected_players(("sam_snead", "both", "sam_snead"), catalog) == (
        "sam_snead",
        "tiger_woods",
        "ben_hogan",
    )
    with pytest.raises(ValueError, match="Unknown player"):
        _selected_players(("missing",), catalog)
    with pytest.raises(ValueError, match="at least one"):
        _selected_players((), catalog)
