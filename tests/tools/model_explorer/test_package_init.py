"""Tests for src/tools/model_explorer/__init__.py lazy imports."""

from __future__ import annotations

import pytest

import src.tools.model_explorer as me


def test_eager_imports_work() -> None:
    assert me.URDFBuilder is not None
    assert me.SegmentManager is not None
    assert me.Handedness is not None


def test_module_metadata() -> None:
    assert me.__version__
    assert "URDFGeneratorWindow" in me.__all__


def test_unknown_attr_raises() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = me.NotAThing  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "name",
    [
        "KinematicTree",
        "EndEffectorLibrary",
    ],
)
def test_lazy_non_gui_imports(name: str) -> None:
    """Some lazy targets do not require Qt and should resolve."""
    obj = getattr(me, name)
    assert obj is not None
