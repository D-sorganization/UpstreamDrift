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
    # ``shapes`` is populated as concrete shapes land (issue #4759).
    assert "LineShape" in shapes.__all__
    assert "CylinderShape" in shapes.__all__
    assert "EllipsoidShape" in shapes.__all__
    assert "CapsuleShape" in shapes.__all__
    assert "CompositeShape" in shapes.__all__
    # ``fitters`` and ``renderers`` are still empty in this wave.
    assert fitters.__all__ == []
    assert renderers.__all__ == []
