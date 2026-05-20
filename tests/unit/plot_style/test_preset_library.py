"""Unit tests for :class:`PresetLibrary` and bundled JSON theme presets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.plot_style import (
    BUILTIN_PRESET_NAMES,
    PlotStyleSet,
    PresetLibrary,
)
from src.shared.python.plot_style.preset_library import _load_builtin_preset

# ---------- Round-trip ------------------------------------------------------


@pytest.mark.parametrize("preset_name", BUILTIN_PRESET_NAMES)
def test_preset_round_trip_through_json(preset_name: str, tmp_path: Path) -> None:
    """Every shape / color / colormap variant in each preset round-trips."""
    library = PresetLibrary.default()
    style_set = library[preset_name]

    # Serialise once to capture the canonical document, then save -> reload.
    original_payload = style_set.to_json()

    out_path = tmp_path / f"{preset_name}.json"
    style_set.save(out_path)
    reloaded = PlotStyleSet.load(out_path)

    assert reloaded.to_json() == original_payload


@pytest.mark.parametrize("preset_name", BUILTIN_PRESET_NAMES)
def test_preset_json_file_matches_loaded_set(preset_name: str) -> None:
    """The on-disk JSON document matches the in-memory PlotStyleSet output."""
    library = PresetLibrary.default()
    loaded_payload = library[preset_name].to_json()

    # Reload directly from disk via the helper to guarantee we hit the file.
    direct = _load_builtin_preset(preset_name)
    assert direct.to_json() == loaded_payload


# ---------- Enumeration -----------------------------------------------------


def test_default_library_lists_all_four_presets() -> None:
    library = PresetLibrary.default()
    names = library.names()
    assert names == list(BUILTIN_PRESET_NAMES)
    assert len(names) == 4
    assert len(library) == 4


def test_default_library_iter_and_contains() -> None:
    library = PresetLibrary.default()
    assert "default" in library
    assert "nonexistent" not in library
    assert 42 not in library  # type: ignore[operator]
    assert list(iter(library)) == list(BUILTIN_PRESET_NAMES)


def test_each_preset_has_curated_entries() -> None:
    """Every built-in preset ships at least the five canonical targets."""
    expected_names = {
        "club_head",
        "ball",
        "left_hand",
        "right_hand",
        "skeleton",
    }
    library = PresetLibrary.default()
    for preset_name in BUILTIN_PRESET_NAMES:
        entries = library[preset_name].entries
        names = {entry.name for entry in entries}
        assert (
            expected_names <= names
        ), f"preset {preset_name!r} missing entries: {expected_names - names}"


# ---------- Lookup errors ---------------------------------------------------


def test_unknown_preset_raises_helpful_keyerror() -> None:
    library = PresetLibrary.default()
    with pytest.raises(KeyError, match="unknown preset 'nope'"):
        _ = library["nope"]


def test_unknown_preset_lists_available_in_message() -> None:
    library = PresetLibrary.default()
    with pytest.raises(KeyError) as info:
        _ = library["nope"]
    msg = str(info.value)
    for name in BUILTIN_PRESET_NAMES:
        assert name in msg


def test_non_string_preset_name_raises_typeerror() -> None:
    library = PresetLibrary.default()
    with pytest.raises(TypeError, match="must be a string"):
        _ = library[42]  # type: ignore[index]


# ---------- Construction validation ----------------------------------------


def test_rejects_non_dict_presets_argument() -> None:
    with pytest.raises(TypeError, match="presets must be a dict"):
        PresetLibrary(presets=[])  # type: ignore[arg-type]


def test_rejects_empty_string_preset_name() -> None:
    style_set = PlotStyleSet()
    with pytest.raises(ValueError, match="non-empty"):
        PresetLibrary(presets={"": style_set})


def test_rejects_non_plotstyleset_value() -> None:
    with pytest.raises(TypeError, match="PlotStyleSet"):
        PresetLibrary(presets={"x": "not a set"})  # type: ignore[dict-item]


def test_empty_library_unknown_lookup_message_lists_none() -> None:
    library = PresetLibrary()
    assert library.names() == []
    with pytest.raises(KeyError, match="\\(none\\)"):
        _ = library["anything"]


# ---------- JSON files are valid PlotStyleSet docs --------------------------


@pytest.mark.parametrize("preset_name", BUILTIN_PRESET_NAMES)
def test_preset_json_file_has_v1_schema(preset_name: str) -> None:
    """Every shipped JSON file declares schema_version == 1."""
    from importlib import resources

    resource = resources.files("src.shared.python.plot_style.presets").joinpath(
        f"{preset_name}.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema_version"] == 1
    assert isinstance(payload["entries"], list)
    assert len(payload["entries"]) >= 1
