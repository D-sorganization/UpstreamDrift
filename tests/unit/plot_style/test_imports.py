"""Smoke tests: every public name imports cleanly."""

from __future__ import annotations


def test_top_level_imports() -> None:
    from src.shared.python.plot_style import (
        SCHEMA_VERSION,
        SEMANTIC_COLORMAP_ALIASES,
        ColormapId,
        ColorResolver,
        ColorScale,
        CustomColormap,
        CustomMeshSpec,
        DataChannel,
        DataDrivenColor,
        MarkerRenderer,
        MarkerShape,
        MarkerStyle,
        PaletteColor,
        PlotStyleSet,
        PlotStyleSpec,
        RGBATuple,
        StaticColor,
        resolve_colormap_alias,
    )

    # Anchor a reference to each name so unused-import lint is happy and
    # the smoke test does some runtime work.
    assert SCHEMA_VERSION >= 1
    assert isinstance(SEMANTIC_COLORMAP_ALIASES, dict)
    assert ColorResolver is not None
    assert ColorScale is not None
    assert CustomColormap is not None
    assert CustomMeshSpec is not None
    assert DataChannel is not None
    assert DataDrivenColor is not None
    assert MarkerRenderer is not None
    assert MarkerShape is not None
    assert MarkerStyle is not None
    assert PaletteColor is not None
    assert PlotStyleSet is not None
    assert PlotStyleSpec is not None
    assert RGBATuple is not None
    assert StaticColor is not None
    assert ColormapId is not None
    assert callable(resolve_colormap_alias)


def test_subpackages_import() -> None:
    import importlib

    for name in (
        "src.shared.python.plot_style.widgets",
        "src.shared.python.plot_style.renderers",
        "src.shared.python.plot_style.resolvers",
        "src.shared.python.plot_style.shapes",
    ):
        module = importlib.import_module(name)
        assert hasattr(module, "__all__")
