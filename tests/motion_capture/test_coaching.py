"""Coaching layers preserve image geometry and never become pose observations."""

from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.coaching import Drawing, DrawingLayer, History, render_layer

pytestmark = pytest.mark.unit


def line(**overrides: object) -> Drawing:
    return Drawing.model_validate(
        dict(kind="line", start=(10, 20), end=(70, 20), **overrides)
    )


def layer(*shapes: Drawing) -> DrawingLayer:
    return DrawingLayer(view="front", width=100, height=80, frames=30, shapes=shapes)


def test_roundtrip_visibility_and_source_immutability(tmp_path: Path) -> None:
    drawing = line(first=5, last=10, colour="#ff0000", stroke=3)
    original = np.zeros((80, 100, 3), dtype=np.uint8)
    document = layer(drawing)
    path = tmp_path / "drawings.json"
    document.save(path)
    restored = DrawingLayer.load(path)
    assert restored == document
    assert not render_layer(original, restored, 4).any()
    assert render_layer(original, restored, 5)[20, 40, 2] > 200
    assert not render_layer(original, restored, 11).any()
    assert not original.any()


@pytest.mark.parametrize("kind", ["line", "arrow", "circle", "ellipse", "rectangle"])
def test_each_shape_renders_and_hits_its_reference(kind: str) -> None:
    shape = Drawing(kind=kind, start=(10, 10), end=(50, 50))
    document = layer(shape)
    assert render_layer(np.zeros((80, 100, 3), dtype=np.uint8), document, 0).any()
    point = (30, 30) if kind in ("line", "arrow") else (30, 10)
    assert shape.hit(point, tolerance=2)
    assert not shape.hit((95, 75), tolerance=2)


def test_history_delete_clear_and_new_branch_drop_redo() -> None:
    initial = layer(line())
    history = History(initial)
    moved = initial.with_shape(initial.shapes[0].translated(5, 6))
    history.apply(moved)
    history.apply(moved.without(initial.shapes[0].id))
    assert not history.current.shapes
    history.undo()
    assert history.current == moved
    history.undo()
    assert history.current == initial
    history.redo()
    history.apply(layer())
    assert not history.can_redo
    history.undo()
    assert history.current == moved


def test_invalid_documents_and_mismatched_media_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        line(first=10, last=2)
    with pytest.raises(ValueError):
        line(colour="not a colour")
    with pytest.raises(ValueError):
        layer(line().translated(100, 0))
    same = line()
    with pytest.raises(ValueError):
        layer(same, same)
    with pytest.raises(ValueError):
        render_layer(np.zeros((40, 50, 3), dtype=np.uint8), layer(same), 0)
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version":"future"}', encoding="utf-8")
    with pytest.raises(ValueError):
        DrawingLayer.load(path)


def test_failed_atomic_save_preserves_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "drawings.json"
    layer(line()).save(path)
    previous = path.read_bytes()

    def fail_replace(self: Path, target: Path) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError):
        layer().save(path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
