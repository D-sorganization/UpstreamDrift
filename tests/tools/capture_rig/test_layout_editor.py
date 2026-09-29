"""Interactive multiview layout editor widget (#9812)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("cv2")

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

from src.tools.capture_rig import layout_editor, layout_editor_canvas
from src.tools.capture_rig.layout_editor import (
    HISTORY_LIMIT,
    EditorDialogs,
    LayoutEditor,
    hint,
)
from src.tools.capture_rig.layout_model import (
    PRESET_NAMES,
    Cell,
    Crop,
    LayoutSpec,
    SourceRef,
    Tile,
    cell_rect,
    preset,
)
from src.tools.capture_rig.layout_presets import SESSION, USER, LayoutStore

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None  # keep the application alive for the module


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


SOURCES = (
    SourceRef(kind="live", view="face_on"),
    SourceRef(kind="live", view="dtl"),
    SourceRef(kind="recorded", view="face_on"),
    SourceRef(kind="overlay", view="dtl", variants=("v1",)),
)
RED = (0, 0, 255)
GREEN = (0, 255, 0)


def _frame(colour: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty((48, 64, 3), dtype=np.uint8)
    frame[:] = colour
    return frame


def _provider(ref: SourceRef) -> np.ndarray | None:
    if ref.view == "face_on":
        return _frame(RED)
    if ref.view == "dtl":
        return _frame(GREEN)
    return None


def _store(tmp_path: Path) -> LayoutStore:
    session = tmp_path / "session"
    session.mkdir()
    return LayoutStore(user_root=tmp_path / "user", session=session)


class _Dialogs:
    """Scripted stand-ins for the modal dialogs."""

    def __init__(self, name: str | None = "mine", yes: bool = True) -> None:
        self.name = name
        self.yes = yes
        self.notices: list[str] = []

    def as_dialogs(self) -> EditorDialogs:
        return EditorDialogs(
            ask_name=lambda _default: self.name,
            confirm=lambda _text: self.yes,
            notify=self.notices.append,
        )


def _editor(
    tmp_path: Path | None = None, *, dialogs: _Dialogs | None = None
) -> tuple[LayoutEditor, list[LayoutSpec]]:
    _app()
    store = _store(tmp_path) if tmp_path is not None else None
    editor = LayoutEditor(
        preset("two_by_two", SOURCES),
        sources=SOURCES,
        frame_provider=_provider,
        store=store,
        dialogs=(dialogs or _Dialogs()).as_dialogs(),
    )
    received: list[LayoutSpec] = []
    editor.layout_changed.connect(received.append)
    return editor, received


# ---------------------------------------------------------------- hints


def test_hint_states_what_and_why_disabled() -> None:
    assert hint("Rotate the tile.", True, "no tile") == "Rotate the tile."
    text = hint("Rotate the tile.", False, "select a tile first")
    assert text.startswith("Rotate the tile.")
    assert "Disabled: select a tile first" in text


def test_every_control_has_a_tooltip() -> None:
    editor, _ = _editor()
    missing = [
        w.objectName() or type(w).__name__
        for w in editor.controls()
        if not w.toolTip().strip()
    ]
    assert missing == []


def test_tile_controls_explain_why_disabled_without_selection() -> None:
    editor, _ = _editor()
    editor.select(None)
    assert not editor.rotate_cw_button.isEnabled()
    assert "Disabled:" in editor.rotate_cw_button.toolTip()
    editor.select(Cell(0, 0))
    assert editor.rotate_cw_button.isEnabled()
    assert "Disabled:" not in editor.rotate_cw_button.toolTip()


# ------------------------------------------------------- programmatic edits


def test_spec_round_trip_and_set_spec_resets_history() -> None:
    editor, received = _editor()
    spec = preset("side_by_side", SOURCES[:2]).renamed("pair")
    editor.set_spec(spec)
    assert editor.spec() == spec
    assert received == []  # loading is not an edit
    assert not editor.can_undo()
    assert editor.rows_spin.value() == 1 and editor.cols_spin.value() == 2


def test_move_tile_swaps_and_emits_once() -> None:
    editor, received = _editor()
    assert editor.move_tile(Cell(0, 0), Cell(1, 1))
    spec = editor.spec()
    assert spec.tile_at(1, 1).source == SOURCES[0]  # type: ignore[union-attr]
    assert spec.tile_at(0, 0).source == SOURCES[3]  # type: ignore[union-attr]
    assert len(received) == 1 and received[0] == spec


def test_move_tile_to_illegal_target_is_reported_not_raised() -> None:
    editor, received = _editor()
    editor.set_spec(preset("primary_plus_strip", SOURCES))
    # the 3x4 tile cannot be re-anchored at (3, 0): it would leave the grid
    assert not editor.move_tile(Cell(0, 0), Cell(3, 0))
    assert received == []
    assert "leaves" in editor.status_text() or "overlap" in editor.status_text()


def test_rotate_flip_fit_label_round_trip() -> None:
    editor, received = _editor()
    editor.rotate(0, 1)
    editor.rotate(0, 1)
    editor.rotate(0, -1)
    editor.flip(0, "h")
    editor.flip(0, "v")
    editor.flip(0, "v")
    editor.set_fit(0, "fill")
    editor.set_label_visible(0, False)
    editor.set_label(0, "Face on")
    tile = editor.spec().tiles[0]
    assert tile.rotation == 90
    assert tile.flip_h is True and tile.flip_v is False
    assert tile.fit == "fill"
    assert tile.show_label is False and tile.label == "Face on"
    assert len(received) == 9
    assert editor.spec().tiles[1] == preset("two_by_two", SOURCES).tiles[1]


def test_crop_round_trip_and_reset() -> None:
    editor, received = _editor()
    crop = Crop(0.25, 0.1, 0.5, 0.8)
    editor.set_crop(1, crop)
    assert editor.spec().tiles[1].crop == crop
    editor.reset_crop(1)
    assert editor.spec().tiles[1].crop.is_full
    assert len(received) == 2


def test_set_source_updates_tile() -> None:
    editor, _ = _editor()
    editor.set_source(3, SOURCES[0])
    assert editor.spec().tiles[3].source == SOURCES[0]
    editor.set_source(3, SourceRef.empty())
    assert editor.spec().tiles[3].source.is_empty


def test_bad_index_is_a_precondition_failure() -> None:
    editor, _ = _editor()
    with pytest.raises(Exception, match="tile index"):
        editor.rotate(7, 1)


# --------------------------------------------------------- grid and spans


def test_grid_resize_keeps_fitting_tiles_and_reports_dropped() -> None:
    editor, received = _editor()
    dropped = editor.set_grid(1, 2)
    spec = editor.spec()
    assert (spec.rows, spec.cols) == (1, 2)
    assert {t.source for t in dropped} == {SOURCES[2], SOURCES[3]}
    assert spec.tile_at(0, 0).source == SOURCES[0]  # type: ignore[union-attr]
    assert spec.tile_at(0, 1).source == SOURCES[1]  # type: ignore[union-attr]
    assert "2 tile" in editor.status_text()
    assert len(received) == 1


def test_grid_grow_fills_new_cells_with_empty_tiles() -> None:
    editor, _ = _editor()
    editor.set_grid(3, 3)
    spec = editor.spec()
    assert len(spec.tiles) == 9
    assert all(spec.tile_at(r, c) is not None for r in range(3) for c in range(3))
    assert spec.tile_at(2, 2).source.is_empty  # type: ignore[union-attr]


def test_grid_spinners_drive_set_grid() -> None:
    editor, received = _editor()
    assert editor.rows_spin.maximum() == 4 and editor.cols_spin.maximum() == 4
    editor.rows_spin.setValue(4)
    assert editor.spec().rows == 4
    assert len(received) == 1


def test_span_grow_swallows_neighbour_and_shrink_refills() -> None:
    editor, received = _editor()
    assert editor.change_span(0, dcol=1)
    spec = editor.spec()
    big = spec.tile_at(0, 1)
    assert big is not None and big.source == SOURCES[0]
    assert big.cell.colspan == 2 and len(spec.tiles) == 3
    assert editor.change_span(editor.tile_index(Cell(0, 0)), dcol=-1)
    spec = editor.spec()
    assert spec.tile_at(0, 1).source.is_empty  # type: ignore[union-attr]
    assert len(spec.tiles) == 4
    assert not editor.change_span(0, dcol=-1)  # already a single cell
    assert len(received) == 2


def test_span_grow_past_grid_is_refused() -> None:
    editor, received = _editor()
    assert not editor.change_span(1, dcol=1)  # (0, 1) is the last column
    assert received == []


# ---------------------------------------------------------------- presets


def test_builtin_presets_load_with_the_editor_sources() -> None:
    editor, received = _editor()
    for name in PRESET_NAMES:
        editor.apply_preset(name)
        assert editor.spec().name == name
    assert editor.spec().tiles[0].source == SOURCES[0]
    assert len(received) == len(PRESET_NAMES)
    assert editor.preset_combo.count() >= len(PRESET_NAMES)


def test_preset_keeps_the_canvas_size() -> None:
    editor, _ = _editor()
    editor.set_spec(preset("single", SOURCES).renamed("x"))
    editor.set_spec(LayoutSpec(name="wide", rows=1, cols=1, canvas=(1920, 1080)))
    editor.apply_preset("two_by_two")
    assert editor.spec().canvas == (1920, 1080)


def test_sources_can_be_reassigned() -> None:
    editor, _ = _editor()
    new = (SourceRef(kind="recorded", view="z"),)
    editor.set_sources(new)
    assert editor.sources == new
    assert editor.source_combo.count() == 2  # (empty) + one source
    editor.apply_preset("side_by_side")
    assert editor.spec().tiles[0].source == new[0]
    assert editor.spec().tiles[1].source.is_empty


# ------------------------------------------------------------------ store


def test_save_as_load_delete_through_store(tmp_path: Path) -> None:
    dialogs = _Dialogs(name="mine")
    editor, received = _editor(tmp_path, dialogs=dialogs)
    editor.rotate(0, 1)
    assert editor.save_as() is True
    store = editor.store
    assert store is not None and store.exists("mine", USER)
    assert store.load("mine", USER).tiles[0].rotation == 90
    assert any(
        editor.preset_combo.itemData(i) == ("mine", USER)
        for i in range(editor.preset_combo.count())
    )
    editor.set_spec(preset("single"))
    assert editor.load_preset("mine", USER)
    assert editor.spec().name == "mine"
    assert editor.spec().tiles[0].rotation == 90
    assert editor.delete_preset("mine", USER)
    assert not store.exists("mine", USER)
    assert not any(
        editor.preset_combo.itemData(i) == ("mine", USER)
        for i in range(editor.preset_combo.count())
    )


def test_save_as_cancelled_or_bad_name_is_reported(tmp_path: Path) -> None:
    dialogs = _Dialogs(name=None)
    editor, _ = _editor(tmp_path, dialogs=dialogs)
    assert editor.save_as() is False
    dialogs.name = "a/b"
    assert editor.save_as() is False
    assert dialogs.notices and "name" in dialogs.notices[-1]


def test_save_to_session_scope(tmp_path: Path) -> None:
    editor, _ = _editor(tmp_path, dialogs=_Dialogs(name="take"))
    editor.scope_combo.setCurrentIndex(editor.scope_combo.findData(SESSION))
    assert editor.save_as()
    assert (tmp_path / "session" / "layouts" / "take.json").is_file()


def test_delete_declined_keeps_layout(tmp_path: Path) -> None:
    dialogs = _Dialogs(name="keep", yes=False)
    editor, _ = _editor(tmp_path, dialogs=dialogs)
    assert editor.save_as()
    assert editor.delete_preset("keep", USER) is False
    assert editor.store is not None and editor.store.exists("keep", USER)


def test_store_controls_disabled_without_store() -> None:
    editor, _ = _editor()
    assert not editor.save_button.isEnabled()
    assert "Disabled:" in editor.save_button.toolTip()
    assert editor.save_as() is False


def test_builtin_cannot_be_deleted(tmp_path: Path) -> None:
    editor, _ = _editor(tmp_path)
    editor.preset_combo.setCurrentIndex(0)  # a builtin
    assert not editor.delete_button.isEnabled()
    assert "Disabled:" in editor.delete_button.toolTip()


# -------------------------------------------------------------- undo/redo


def test_undo_redo_round_trip() -> None:
    editor, received = _editor()
    start = editor.spec()
    editor.rotate(0, 1)
    editor.flip(0, "h")
    assert editor.can_undo() and not editor.can_redo()
    editor.undo()
    assert editor.spec().tiles[0].flip_h is False
    assert editor.spec().tiles[0].rotation == 90
    editor.undo()
    assert editor.spec() == start
    assert not editor.can_undo() and editor.can_redo()
    editor.redo()
    editor.redo()
    assert editor.spec().tiles[0].flip_h is True
    assert not editor.can_redo()
    assert len(received) == 6  # two edits, two undos, two redos


def test_undo_history_is_capped_at_twenty() -> None:
    editor, _ = _editor()
    assert HISTORY_LIMIT == 20
    for _ in range(30):
        editor.rotate(0, 1)
    steps = 0
    while editor.can_undo():
        editor.undo()
        steps += 1
    assert steps == HISTORY_LIMIT


def test_new_edit_clears_redo() -> None:
    editor, _ = _editor()
    editor.rotate(0, 1)
    editor.undo()
    assert editor.can_redo()
    editor.flip(0, "v")
    assert not editor.can_redo()
    assert not editor.redo_button.isEnabled()
    assert "Disabled:" in editor.redo_button.toolTip()


def test_undo_without_history_is_a_noop() -> None:
    editor, received = _editor()
    editor.undo()
    editor.redo()
    assert received == []


# ------------------------------------------------------------- thumbnails


def test_thumbnails_are_composed_from_the_frame_provider() -> None:
    editor, _ = _editor()
    canvas = editor.canvas
    composite = canvas.composite()
    assert composite is not None and composite.dtype == np.uint8
    size = (composite.shape[1], composite.shape[0])
    spec = editor.spec()
    for index, colour in ((0, RED), (1, GREEN)):
        x0, y0, x1, y1 = cell_rect(spec, spec.tiles[index].cell, size)
        centre = composite[(y0 + y1) // 2, (x0 + x1) // 2]
        assert tuple(int(c) for c in centre) == colour
    assert not canvas.pixmap_item.pixmap().isNull()


def test_missing_frames_show_placeholder_without_error() -> None:
    _app()
    editor = LayoutEditor(preset("two_by_two", SOURCES), sources=SOURCES)
    composite = editor.canvas.composite()
    assert composite is not None and composite.shape[2] == 3
    editor.set_frame_provider(_provider)
    editor.refresh_thumbnails()
    x0, y0, x1, y1 = cell_rect(
        editor.spec(),
        editor.spec().tiles[0].cell,
        (composite.shape[1], composite.shape[0]),
    )
    centre = editor.canvas.composite()[(y0 + y1) // 2, (x0 + x1) // 2]
    assert tuple(int(c) for c in centre) == RED


# ------------------------------------------------------------- canvas hits


def test_canvas_maps_points_to_cells_and_selection_follows() -> None:
    editor, _ = _editor()
    canvas = editor.canvas
    w, h = canvas.scene_size()
    assert canvas.cell_at(QPointF(w * 0.75, h * 0.75)) == Cell(1, 1)
    assert canvas.cell_at(QPointF(-1.0, 5.0)) is None
    canvas.tile_pressed.emit(1, 1)
    assert editor.selected_cell() == Cell(1, 1)
    assert editor.source_combo.currentData() == SOURCES[3]


def test_canvas_drop_moves_tile_and_crop_drag_sets_crop() -> None:
    editor, received = _editor()
    canvas = editor.canvas
    canvas.tile_dropped.emit(Cell(0, 0), Cell(0, 1))
    assert editor.spec().tile_at(0, 1).source == SOURCES[0]  # type: ignore[union-attr]
    editor.select(Cell(0, 1))
    editor.crop_button.setChecked(True)
    assert canvas.crop_mode
    canvas.crop_dragged.emit(Cell(0, 1), (0.25, 0.25, 0.5, 0.5))
    tile = editor.spec().tile_at(0, 1)
    assert tile is not None and tile.crop == Crop(0.25, 0.25, 0.5, 0.5)
    canvas.crop_dragged.emit(Cell(0, 1), (0.5, 0.0, 0.5, 1.0))  # crop within crop
    tile = editor.spec().tile_at(0, 1)
    assert tile is not None and tile.crop == Crop(0.5, 0.25, 0.25, 0.5)
    assert len(received) == 3


def test_canvas_normalised_rect_from_drag() -> None:
    editor, _ = _editor()
    canvas = editor.canvas
    spec = editor.spec()
    x0, y0, x1, y1 = cell_rect(spec, spec.tiles[0].cell, canvas.scene_size())
    rect = layout_editor_canvas.normalised_rect(
        (x0, y0, x1, y1), QPointF(x0, y0), QPointF((x0 + x1) / 2, (y0 + y1) / 2)
    )
    assert rect == pytest.approx((0.0, 0.0, 0.5, 0.5))
    assert (
        layout_editor_canvas.normalised_rect(
            (0, 0, 10, 10), QPointF(1, 1), QPointF(1, 1)
        )
        is None
    )


def test_canvas_highlights_selection_with_theme_colours() -> None:
    editor, _ = _editor()
    editor.select(Cell(0, 0))
    item = editor.canvas.overlay_items()[0]
    assert item.pen().color() == editor.canvas.colours.selected
    assert (
        editor.canvas.overlay_items()[1].pen().color() == editor.canvas.colours.border
    )
    assert editor.canvas.backgroundBrush().color() == editor.canvas.colours.background


def test_canvas_mouse_drag_between_cells_moves_the_tile() -> None:
    editor, received = _editor()
    canvas = editor.canvas
    canvas.resize(400, 300)
    spec = editor.spec()
    size = canvas.scene_size()
    ax0, ay0, ax1, ay1 = cell_rect(spec, spec.tiles[0].cell, size)
    bx0, by0, bx1, by1 = cell_rect(spec, spec.tiles[3].cell, size)
    start = QPointF((ax0 + ax1) / 2, (ay0 + ay1) / 2)
    end = QPointF((bx0 + bx1) / 2, (by0 + by1) / 2)
    canvas.begin_press(start)
    canvas.drag_to(end)
    canvas.release_at(end)
    assert editor.spec().tile_at(1, 1).source == SOURCES[0]  # type: ignore[union-attr]
    assert len(received) == 1
    # a click without movement selects rather than moves
    canvas.begin_press(start)
    canvas.release_at(start)
    assert editor.selected_cell() == Cell(0, 0)
    assert len(received) == 1


# ------------------------------------------------------------------ theme


@pytest.mark.parametrize("module", [layout_editor, layout_editor_canvas])
def test_no_literal_colours(module: object) -> None:
    text = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    hex_colour = r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b"
    assert re.search(hex_colour, text) is None
    assert re.search(r"QColor\(\s*['\"]", text) is None
    assert re.search(r"\brgb\(", text) is None
