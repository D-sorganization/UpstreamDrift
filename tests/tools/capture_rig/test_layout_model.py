"""Multiview layout model and frame compositor (#9810)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.tools.capture_rig.layout_model import (
    FITS,
    PRESET_NAMES,
    ROTATIONS,
    SCHEMA_VERSION,
    Cell,
    Crop,
    LayoutSpec,
    Palette,
    SourceRef,
    Tile,
    cell_rect,
    compose,
    preset,
)

pytestmark = [pytest.mark.unit]

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)


def _solid(colour: tuple[int, int, int], w: int = 64, h: int = 48) -> np.ndarray:
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:] = colour
    return frame


def _quadrants(w: int = 64, h: int = 48) -> np.ndarray:
    """Top-left red, top-right green, bottom-left blue, bottom-right white."""
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[: h // 2, : w // 2] = RED
    frame[: h // 2, w // 2 :] = GREEN
    frame[h // 2 :, : w // 2] = BLUE
    frame[h // 2 :, w // 2 :] = WHITE
    return frame


def _mean(canvas: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return canvas[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def _live(view: str) -> SourceRef:
    return SourceRef(kind="live", view=view)


def _spec(*tiles: Tile, rows: int = 2, cols: int = 2) -> LayoutSpec:
    return LayoutSpec(name="t", rows=rows, cols=cols, tiles=tiles, canvas=(320, 240))


def _tile(view: str, row: int, col: int, **kwargs: Any) -> Tile:
    return Tile(source=_live(view), cell=Cell(row, col), show_label=False, **kwargs)


# ---------------------------------------------------------------- model


class TestSourceRef:
    def test_keys_are_distinct_per_kind_and_variants(self) -> None:
        assert _live("a").key == "live:a"
        assert SourceRef(kind="recorded", view="a").key == "recorded:a"
        overlay = SourceRef(kind="overlay", view="a", variants=("v1", "v2"))
        assert overlay.key == "overlay:a:v1+v2"
        assert SourceRef.empty().key == "empty"
        assert SourceRef.empty().is_empty

    def test_round_trip(self) -> None:
        ref = SourceRef(kind="overlay", view="face_on", variants=("base",))
        assert SourceRef.from_dict(ref.to_dict()) == ref

    @pytest.mark.parametrize(
        ("payload", "field"),
        [
            ({"kind": "hologram", "view": "a"}, "kind"),
            ({"kind": "live", "view": ""}, "view"),
            ({"kind": "live", "view": "a", "variants": "v1"}, "variants"),
            ({"kind": "empty", "view": "a"}, "view"),
        ],
    )
    def test_validation_names_field(self, payload: dict[str, Any], field: str) -> None:
        with pytest.raises(ValueError, match=field):
            SourceRef.from_dict(payload)


class TestTileAndSpec:
    def test_round_trip_full_spec(self) -> None:
        spec = LayoutSpec(
            name="pair",
            rows=1,
            cols=2,
            tiles=(
                Tile(
                    source=_live("a"),
                    cell=Cell(0, 0),
                    rotation=90,
                    flip_h=True,
                    crop=Crop(0.1, 0.2, 0.5, 0.5),
                    fit="fill",
                    label="Face on",
                ),
                Tile(source=SourceRef.empty(), cell=Cell(0, 1), show_label=False),
            ),
            canvas=(1280, 720),
            background=(1, 2, 3),
        )
        payload = spec.to_dict()
        assert payload["schema_version"] == SCHEMA_VERSION
        assert LayoutSpec.from_dict(payload) == spec
        # provenance and unknown keys are tolerated on the way in
        assert LayoutSpec.from_dict({**payload, "provenance": {}}) == spec

    @pytest.mark.parametrize(
        ("mutation", "field"),
        [
            ({"schema_version": "rig-layout/9.0.0"}, "schema_version"),
            ({"rows": 0}, "rows"),
            ({"rows": 5}, "rows"),
            ({"cols": "2"}, "cols"),
            ({"name": ""}, "name"),
            ({"canvas": [0, 720]}, "canvas"),
            ({"canvas": [1280]}, "canvas"),
            ({"background": [1, 2]}, "background"),
            ({"background": [1, 2, 300]}, "background"),
            ({"tiles": "none"}, "tiles"),
        ],
    )
    def test_spec_validation_names_field(
        self, mutation: dict[str, Any], field: str
    ) -> None:
        payload = {**preset("side_by_side", [_live("a"), _live("b")]).to_dict()}
        payload.update(mutation)
        with pytest.raises(ValueError, match=field):
            LayoutSpec.from_dict(payload)

    @pytest.mark.parametrize(
        ("mutation", "field"),
        [
            ({"rotation": 45}, "rotation"),
            ({"fit": "zoom"}, "fit"),
            ({"flip_h": "yes"}, "flip_h"),
            ({"flip_v": 1}, "flip_v"),
            ({"show_label": None}, "show_label"),
            ({"label": 3}, "label"),
            ({"crop": {"x": -0.1, "y": 0, "w": 1, "h": 1}}, "crop.x"),
            ({"crop": {"x": 0.5, "y": 0, "w": 0.6, "h": 1}}, "crop"),
            ({"crop": {"x": 0, "y": 0, "w": 0, "h": 1}}, "crop.w"),
            ({"cell": {"row": 2, "col": 0}}, "cell"),
            ({"cell": {"row": -1, "col": 0}}, "cell.row"),
            ({"cell": {"row": 0, "col": -1}}, "cell.col"),
            ({"cell": {"row": 0, "col": 0, "rowspan": 0}}, "cell.rowspan"),
            ({"cell": {"row": 0, "col": 1, "colspan": 2}}, "cell"),
            ({"source": {"kind": "live", "view": ""}}, "view"),
        ],
    )
    def test_tile_validation_names_field(
        self, mutation: dict[str, Any], field: str
    ) -> None:
        payload = preset("side_by_side", [_live("a"), _live("b")]).to_dict()
        payload["tiles"][0].update(mutation)
        with pytest.raises(ValueError, match=field):
            LayoutSpec.from_dict(payload)

    def test_overlapping_tiles_rejected(self) -> None:
        with pytest.raises(ValueError, match="tiles.*overlap"):
            _spec(
                _tile(
                    "a",
                    0,
                    0,
                ),
                _tile("b", 0, 0),
            )
        with pytest.raises(ValueError, match="tiles.*overlap"):
            _spec(
                Tile(source=_live("a"), cell=Cell(0, 0, rowspan=2, colspan=2)),
                _tile("b", 1, 1),
            )

    def test_with_without_move(self) -> None:
        spec = _spec(_tile("a", 0, 0), _tile("b", 0, 1))
        added = spec.with_tile(_tile("c", 1, 1))
        assert spec.tile_at(1, 1) is None
        assert added.tile_at(1, 1) is not None
        assert len(spec.tiles) == 2  # immutability
        replaced = spec.with_tile(_tile("z", 0, 0))
        tile = replaced.tile_at(0, 0)
        assert tile is not None and tile.source.view == "z"
        removed = added.without_tile(0, 1)
        assert removed.tile_at(0, 1) is None and len(removed.tiles) == 2
        moved = spec.move_tile(Cell(0, 0), Cell(1, 1))
        assert moved.tile_at(0, 0) is None
        target = moved.tile_at(1, 1)
        assert target is not None and target.source.view == "a"
        swapped = spec.move_tile(Cell(0, 0), Cell(0, 1))
        first, second = swapped.tile_at(0, 0), swapped.tile_at(0, 1)
        assert first is not None and first.source.view == "b"
        assert second is not None and second.source.view == "a"

    def test_move_missing_or_outside_rejected(self) -> None:
        spec = _spec(_tile("a", 0, 0))
        with pytest.raises(ValueError):
            spec.move_tile(Cell(1, 1), Cell(0, 0))
        with pytest.raises(ValueError):
            spec.move_tile(Cell(0, 0), Cell(3, 3))

    def test_sources_and_source_keys(self) -> None:
        spec = _spec(_tile("a", 0, 0), Tile(source=SourceRef.empty(), cell=Cell(0, 1)))
        assert spec.source_keys() == ("live:a",)


class TestPresets:
    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_presets_cover_their_grid_without_overlap(self, name: str) -> None:
        sources = [_live(f"cam{i}") for i in range(16)]
        spec = preset(name, sources)
        covered: list[tuple[int, int]] = []
        for tile in spec.tiles:
            for r in range(tile.cell.row, tile.cell.row + tile.cell.rowspan):
                for c in range(tile.cell.col, tile.cell.col + tile.cell.colspan):
                    covered.append((r, c))
        assert sorted(covered) == [
            (r, c) for r in range(spec.rows) for c in range(spec.cols)
        ]
        assert len(covered) == len(set(covered))
        assert spec.name == name
        assert LayoutSpec.from_dict(spec.to_dict()) == spec

    def test_preset_shapes(self) -> None:
        assert (preset("single").rows, preset("single").cols) == (1, 1)
        assert (preset("side_by_side").rows, preset("side_by_side").cols) == (1, 2)
        assert (preset("three_across").rows, preset("three_across").cols) == (1, 3)
        assert (preset("two_by_two").rows, preset("two_by_two").cols) == (2, 2)
        assert (preset("three_by_three").rows, preset("three_by_three").cols) == (
            3,
            3,
        )
        four = preset("four_by_four")
        assert (four.rows, four.cols, len(four.tiles)) == (4, 4, 16)
        strip = preset("primary_plus_strip", [_live("main"), _live("s1")])
        big = strip.tile_at(0, 0)
        assert big is not None and big.cell.rowspan >= 2 and big.cell.colspan >= 2
        assert big.source.view == "main"
        assert sum(1 for t in strip.tiles if t.source.is_empty) == len(strip.tiles) - 2

    def test_fewer_sources_leave_empty_tiles(self) -> None:
        spec = preset("two_by_two", [_live("a")])
        assert spec.source_keys() == ("live:a",)
        assert sum(1 for t in spec.tiles if t.source.is_empty) == 3

    def test_unknown_preset(self) -> None:
        with pytest.raises(ValueError, match="preset"):
            preset("hexagon")


# ---------------------------------------------------------- compositor


class TestCompose:
    def test_canvas_size_and_background(self) -> None:
        spec = _spec(_tile("a", 0, 0))
        out = compose({}, spec, palette=Palette(background=(9, 8, 7)))
        assert out.shape == (240, 320, 3) and out.dtype == np.uint8
        # the three empty cells show the background
        assert tuple(_mean(out, cell_rect(spec, Cell(1, 1), (320, 240)))) == (9, 8, 7)
        assert compose({}, spec, size=(64, 32)).shape == (32, 64, 3)

    def test_solid_frame_lands_in_its_cell(self) -> None:
        spec = _spec(_tile("a", 1, 0, fit="stretch"))
        out = compose({"live:a": _solid(RED)}, spec)
        rect = cell_rect(spec, Cell(1, 0), (320, 240))
        assert tuple(_mean(out, rect)) == RED
        assert tuple(_mean(out, cell_rect(spec, Cell(0, 0), (320, 240)))) != RED

    def test_missing_source_is_placeholder(self) -> None:
        spec = _spec(
            _tile("ghost", 0, 0), Tile(source=SourceRef.empty(), cell=Cell(0, 1))
        )
        pal = Palette(background=(0, 0, 0), placeholder=(50, 60, 70))
        out = compose({}, spec, palette=pal)
        assert tuple(_mean(out, cell_rect(spec, Cell(0, 0), (320, 240)))) == (
            50,
            60,
            70,
        )
        # the empty tile shows a label by default so is only mostly placeholder
        mean = _mean(out, cell_rect(spec, Cell(0, 1), (320, 240)))
        assert np.all(np.abs(mean - np.array([50, 60, 70])) < 40)

    @pytest.mark.parametrize("fit", FITS)
    def test_fit_modes(self, fit: str) -> None:
        # wide cell 160x120 receives a tall 30x60 frame
        spec = _spec(_tile("a", 0, 0, fit=fit))
        frame = _quadrants(w=30, h=60)
        out = compose({"live:a": frame}, spec, palette=Palette(background=(0, 0, 0)))
        x0, y0, x1, y1 = cell_rect(spec, Cell(0, 0), (320, 240))
        cell = out[y0:y1, x0:x1]
        black = (cell.sum(axis=2) == 0).mean()
        if fit == "fit":
            # letterboxed: 60 px wide content in a 160 px cell -> ~62 % black
            assert 0.5 < black < 0.7
            assert tuple(cell[5, 80 - 20]) == RED and tuple(cell[5, 80 + 20]) == GREEN
        elif fit == "fill":
            assert black == 0.0
            # centre-cropped vertically: the middle row still splits red/green
            assert tuple(cell[2, 10]) == RED and tuple(cell[2, 150]) == GREEN
            assert tuple(cell[117, 10]) == BLUE and tuple(cell[117, 150]) == WHITE
        else:
            assert black == 0.0
            assert tuple(cell[2, 2]) == RED and tuple(cell[117, 157]) == WHITE

    @pytest.mark.parametrize("rotation", ROTATIONS)
    @pytest.mark.parametrize("flip_h", [False, True])
    @pytest.mark.parametrize("flip_v", [False, True])
    def test_rotation_and_flip_move_the_red_quadrant(
        self, rotation: int, flip_h: bool, flip_v: bool
    ) -> None:
        spec = _spec(
            _tile(
                "a",
                0,
                0,
                rotation=rotation,
                flip_h=flip_h,
                flip_v=flip_v,
                fit="stretch",
            ),
            rows=1,
            cols=1,
        )
        out = compose({"live:a": _quadrants(48, 48)}, spec, size=(48, 48))
        # reproduce the expected transform with numpy: rotate clockwise, then flip
        expected = np.rot90(_quadrants(48, 48), k=-(rotation // 90))
        if flip_h:
            expected = expected[:, ::-1]
        if flip_v:
            expected = expected[::-1]
        for (ry, rx), colour in {
            (0, 0): expected[0, 0],
            (0, 1): expected[0, -1],
            (1, 0): expected[-1, 0],
            (1, 1): expected[-1, -1],
        }.items():
            region = out[ry * 24 : ry * 24 + 24, rx * 24 : rx * 24 + 24]
            assert tuple(region.reshape(-1, 3).mean(axis=0).round()) == tuple(colour)

    def test_crop_selects_region(self) -> None:
        spec = _spec(_tile("a", 0, 0, crop=Crop(0.5, 0.0, 0.5, 0.5), fit="stretch"))
        out = compose({"live:a": _quadrants()}, spec)
        assert tuple(_mean(out, cell_rect(spec, Cell(0, 0), (320, 240)))) == GREEN
        spec = _spec(_tile("a", 0, 0, crop=Crop(0.0, 0.5, 0.5, 0.5), fit="stretch"))
        out = compose({"live:a": _quadrants()}, spec)
        assert tuple(_mean(out, cell_rect(spec, Cell(0, 0), (320, 240)))) == BLUE

    def test_spanning_tile_fills_its_span(self) -> None:
        spec = _spec(
            Tile(
                source=_live("a"),
                cell=Cell(0, 0, rowspan=2, colspan=1),
                fit="stretch",
                show_label=False,
            )
        )
        out = compose({"live:a": _solid(RED)}, spec)
        assert tuple(_mean(out, (0, 0, 160, 240))) == RED

    def test_four_by_four_with_sixteen_sources(self) -> None:
        sources = [_live(f"cam{i}") for i in range(16)]
        spec = preset("four_by_four", sources).to_dict()
        for tile in spec["tiles"]:
            tile["show_label"] = False
            tile["fit"] = "stretch"
        layout = LayoutSpec.from_dict(spec)
        frames = {
            f"live:cam{i}": _solid((i * 16, 255 - i * 16, 128)) for i in range(16)
        }
        out = compose(frames, layout, size=(640, 400))
        for i, tile in enumerate(layout.tiles):
            rect = cell_rect(layout, tile.cell, (640, 400))
            assert tuple(_mean(out, rect)) == (i * 16, 255 - i * 16, 128)

    def test_labels_draw_something(self) -> None:
        plain = Tile(
            source=_live("a"), cell=Cell(0, 0), fit="stretch", show_label=False
        )
        labelled = Tile(
            source=_live("a"), cell=Cell(0, 0), fit="stretch", label="Face on"
        )
        frames = {"live:a": _solid(RED)}
        without = compose(frames, _spec(plain))
        with_label = compose(frames, _spec(labelled))
        assert not np.array_equal(without, with_label)

    def test_grayscale_frame_is_accepted(self) -> None:
        spec = _spec(_tile("a", 0, 0, fit="stretch"))
        gray = np.full((48, 64), 200, dtype=np.uint8)
        out = compose({"live:a": gray}, spec)
        assert tuple(_mean(out, cell_rect(spec, Cell(0, 0), (320, 240)))) == (
            200,
            200,
            200,
        )

    def test_preconditions(self) -> None:
        spec = _spec(_tile("a", 0, 0))
        with pytest.raises(ValueError):
            compose({}, spec, size=(0, 10))
        with pytest.raises(ValueError):
            compose({"live:a": np.zeros((4, 4, 3), dtype=np.float32)}, spec)

    def test_compose_is_pure(self) -> None:
        spec = _spec(_tile("a", 0, 0))
        frame = _solid(RED)
        before = frame.copy()
        first = compose({"live:a": frame}, spec)
        second = compose({"live:a": frame}, spec)
        assert np.array_equal(frame, before)
        assert np.array_equal(first, second)

    def test_performance_sixteen_tiles(self) -> None:
        """A full 4x4 composes at interactive rates.

        Wall-clock budgets flake on a loaded machine, so this takes the best
        of several rounds (the fastest round is the one least disturbed by
        other work) and leaves generous headroom over the ~12 fps the live
        preview asks for. It catches an algorithmic regression, not a busy
        CI runner.
        """
        sources = [_live(f"cam{i}") for i in range(16)]
        layout = preset("four_by_four", sources)
        frames = {f"live:cam{i}": _solid((i, i, i), 640, 400) for i in range(16)}
        compose(frames, layout, size=(1280, 720))  # warm up
        rounds = []
        for _ in range(5):
            start = time.perf_counter()
            compose(frames, layout, size=(1280, 720))
            rounds.append((time.perf_counter() - start) * 1000)
        assert min(rounds) < 200, rounds


class TestPalette:
    def test_from_hex(self) -> None:
        pal = Palette.from_hex(background="#102030", label_text="#ffffff")
        assert pal.background == (0x30, 0x20, 0x10)
        assert pal.label_text == (255, 255, 255)
        assert pal.placeholder == Palette().placeholder

    def test_from_hex_rejects_bad_value(self) -> None:
        with pytest.raises(ValueError, match="background"):
            Palette.from_hex(background="blue")


def test_no_qt_import() -> None:
    """The compositor stays pure numpy/cv2 so every consumer can share it."""
    import src.tools.capture_rig.layout_model as module

    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "PyQt" not in source
