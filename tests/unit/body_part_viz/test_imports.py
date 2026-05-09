"""Public-API smoke test for the body_part_viz package."""

from __future__ import annotations


def test_public_api_imports() -> None:
    from src.shared.python.body_part_viz import (
        BindingKind,
        BodyPartShape,
        FittedShape,
        MarkerBinding,
        ShapeFitter,
        ShapeRenderer,
        ShapeTheme,
    )

    expected = {
        BindingKind,
        BodyPartShape,
        FittedShape,
        MarkerBinding,
        ShapeFitter,
        ShapeRenderer,
        ShapeTheme,
    }
    assert all(obj is not None for obj in expected)


def test_subpackages_importable() -> None:
    import src.shared.python.body_part_viz.fitters as fitters
    import src.shared.python.body_part_viz.renderers as renderers
    import src.shared.python.body_part_viz.shapes as shapes

    for pkg in (shapes, fitters, renderers):
        assert hasattr(pkg, "__all__")
        assert pkg.__all__ == []
