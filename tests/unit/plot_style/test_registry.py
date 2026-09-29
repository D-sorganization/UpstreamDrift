"""Unit tests for :mod:`src.shared.python.plot_style.registry`."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.colors as mcolors
import pytest

from src.shared.python.plot_style import (
    ColormapId,
    CustomColormap,
    get_colormap,
    list_colormaps,
    register_custom_colormap,
    unregister_custom_colormap,
)
from src.shared.python.plot_style.colormaps import (
    SEMANTIC_COLORMAP_ALIASES,
    resolve_colormap_alias,
)
from src.shared.python.plot_style.registry import _BUILTIN_MPL_NAME


@pytest.fixture
def custom_cmap() -> CustomColormap:
    return CustomColormap(
        name="test_cmap_alpha",
        stops=((0.0, "#000000"), (0.5, "#888888"), (1.0, "#ffffff")),
    )


@pytest.fixture(autouse=True)
def _clean_custom_registry() -> Iterator[None]:
    """Snapshot + restore the process-global custom-colormap registry."""
    from src.shared.python.plot_style import registry as _reg

    snapshot = dict(_reg._CUSTOM_COLORMAPS)
    try:
        yield
    finally:
        _reg._CUSTOM_COLORMAPS.clear()
        _reg._CUSTOM_COLORMAPS.update(snapshot)


# ---------- get_colormap (built-ins + aliases) -------------------------


def test_every_colormap_id_resolves_to_matplotlib_colormap() -> None:
    for cmap_id in ColormapId:
        result = get_colormap(cmap_id)
        assert isinstance(result, mcolors.Colormap)


def test_semantic_aliases_resolve_to_their_target() -> None:
    for alias, target in SEMANTIC_COLORMAP_ALIASES.items():
        assert get_colormap(alias).name == get_colormap(target).name


def test_resolve_alias_helper_used_for_every_alias() -> None:
    for alias in SEMANTIC_COLORMAP_ALIASES:
        # sanity: the helper is what the registry delegates to
        target = resolve_colormap_alias(alias)
        assert get_colormap(alias).name == _BUILTIN_MPL_NAME[target]


def test_string_value_of_colormap_id_resolves_too() -> None:
    assert get_colormap("viridis").name == get_colormap(ColormapId.VIRIDIS).name


def test_string_value_of_alias_resolves_too() -> None:
    assert get_colormap("velocity").name == get_colormap(ColormapId.VELOCITY).name


def test_get_colormap_rejects_unknown_string() -> None:
    with pytest.raises(KeyError, match="not registered"):
        get_colormap("nonexistent_colormap")


def test_get_colormap_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="ColormapId or str"):
        get_colormap(42)  # type: ignore[arg-type]


def test_spectral_resolves_to_capitalised_matplotlib_name() -> None:
    cmap = get_colormap(ColormapId.SPECTRAL)
    assert cmap.name == "Spectral"


# ---------- register_custom_colormap -----------------------------------


def test_register_then_get_round_trips(custom_cmap: CustomColormap) -> None:
    register_custom_colormap(custom_cmap)
    result = get_colormap(custom_cmap.name)
    assert isinstance(result, mcolors.LinearSegmentedColormap)
    assert result.name == custom_cmap.name


def test_register_is_idempotent_for_same_stops(
    custom_cmap: CustomColormap,
) -> None:
    register_custom_colormap(custom_cmap)
    first = get_colormap(custom_cmap.name)
    # Equivalent CustomColormap (same name + stops) should be a no-op
    # — i.e. the stored colormap object is unchanged.
    twin = CustomColormap(name=custom_cmap.name, stops=custom_cmap.stops)
    register_custom_colormap(twin)
    assert get_colormap(custom_cmap.name) is first  # custom registry caches


def test_register_rejects_conflicting_stops(
    custom_cmap: CustomColormap,
) -> None:
    register_custom_colormap(custom_cmap)
    conflict = CustomColormap(
        name=custom_cmap.name,
        stops=((0.0, "#ff0000"), (1.0, "#0000ff")),
    )
    with pytest.raises(ValueError, match="already registered with different stops"):
        register_custom_colormap(conflict)


def test_register_rejects_non_dataclass() -> None:
    with pytest.raises(TypeError, match="CustomColormap"):
        register_custom_colormap("viridis")  # type: ignore[arg-type]


def test_register_rejects_name_clashing_with_builtin() -> None:
    clash = CustomColormap(
        name="viridis",
        stops=((0.0, "#000"), (1.0, "#fff")),
    )
    with pytest.raises(ValueError, match="clashes"):
        register_custom_colormap(clash)


def test_register_rejects_name_clashing_with_semantic_alias() -> None:
    clash = CustomColormap(
        name="velocity",
        stops=((0.0, "#000"), (1.0, "#fff")),
    )
    with pytest.raises(ValueError, match="clashes"):
        register_custom_colormap(clash)


# ---------- unregister_custom_colormap ---------------------------------


def test_unregister_removes_the_entry(custom_cmap: CustomColormap) -> None:
    register_custom_colormap(custom_cmap)
    unregister_custom_colormap(custom_cmap.name)
    with pytest.raises(KeyError):
        get_colormap(custom_cmap.name)


def test_unregister_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="nonexistent"):
        unregister_custom_colormap("nonexistent")


def test_unregister_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="string"):
        unregister_custom_colormap(123)  # type: ignore[arg-type]


# ---------- list_colormaps ---------------------------------------------


def test_list_includes_every_colormap_id_value() -> None:
    names = list_colormaps()
    for cmap_id in ColormapId:
        assert cmap_id.value in names


def test_list_includes_registered_custom_colormaps(
    custom_cmap: CustomColormap,
) -> None:
    register_custom_colormap(custom_cmap)
    names = list_colormaps()
    assert custom_cmap.name in names
    # Built-ins still come first.
    assert names.index(custom_cmap.name) >= len(ColormapId)


def test_list_returns_tuple() -> None:
    assert isinstance(list_colormaps(), tuple)
