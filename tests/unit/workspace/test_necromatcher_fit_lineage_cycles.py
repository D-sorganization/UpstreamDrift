"""Recursive recall rejects cycles and releases its guard after failure."""

from pathlib import Path

import pytest

from src.shared.python.workspace import necromatcher as owner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("parents", [{"a": "a"}, {"a": "b", "b": "a"}])
def test_recursive_lineage_rejects_without_recursion_error(
    fit_case: tuple, monkeypatch: pytest.MonkeyPatch, parents: dict[str, str]
) -> None:
    library, source, _ = fit_case
    for identifier in parents:
        library.add_fit(identifier, "practice", source)
    original = owner.read_kinematic_fit

    def recursive(path: Path, recalled: owner.NecromatcherLibrary, swing: str) -> dict:
        return recalled.load_fit(parents[path.stem])

    monkeypatch.setattr(owner, "read_kinematic_fit", recursive)
    with pytest.raises(ValueError, match="lineage.*cycle"):
        library.load_fit("a")
    monkeypatch.setattr(owner, "read_kinematic_fit", original)
    assert library.load_fit("a")["capture_id"]
