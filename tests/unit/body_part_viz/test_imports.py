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
    # Shapes populated by Wave 2 (#4759) — primitives.
    for shape_name in (
        "LineShape",
        "CylinderShape",
        "EllipsoidShape",
        "CapsuleShape",
        "CompositeShape",
    ):
        assert shape_name in shapes.__all__
    # Fitters wave (#4756) ships three concrete strategies.
    assert set(fitters.__all__) == {
        "BetweenTwoMarkersFitter",
        "ClusterKabschFitter",
        "ProcrustesAnisotropicFitter",
    }
    # Renderers wave (#4760 + #4762) ships matplotlib + pyqtgl backends.
    assert "MatplotlibRenderer" in renderers.__all__
