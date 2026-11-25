"""The model card drawn on request in the clip (#15150, E2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export import model_card as mc
from src.tools.native_viewer_export import model_card_panel as mp
from src.tools.native_viewer_export.core import ExportSettings, SwingInput
from src.tools.native_viewer_export.core import export_swing

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COORDS = ("A", "B")


def _swing(provenance: dict | None = None) -> SwingInput:
    shaft = {"model": "chain", "flex": "stiff", "flex_cpm": 258.0, "n_links": 6}
    spec = {
        "coordinate_order": list(COORDS),
        "club": {"name": "Driver", "shaft": shaft},
    }
    q = np.zeros((101, 2))
    bundle = InputBundle(
        spec_bytes=json.dumps(spec).encode(),
        coordinate_order=COORDS,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((100, 2)),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
        provenance=provenance or {},
    )
    return SwingInput(bundle, bundle.reference_q, "driver", "Driver", "mujoco")


def _card(provenance: dict | None = None) -> mc.ModelCard:
    return mc.model_card_of(_swing(provenance), ExportSettings(), None)


def test_lines_follow_the_settings_field_list_with_their_tags() -> None:
    cfg = mp.load_panel_settings()
    card = _card({"max_pose_drift_m": 0.0123})
    lines = mp.panel_lines(card, cfg)
    assert [ln.field for ln in lines] == [e["field"] for e in cfg.fields]
    for line, entry in zip(lines, cfg.fields, strict=True):
        assert line.tag == getattr(card, entry["field"]).tag
        assert line.text.startswith(entry["label"])
    by_field = {ln.field: ln for ln in lines}
    assert by_field["replay_kind"].tag == "measured"
    assert "kinematic replay" in by_field["replay_kind"].text.lower()
    assert by_field["fit_error"].tag == "fitted"
    assert "0.0123" in by_field["fit_error"].text
    assert by_field["club"].tag == "assumed"
    assert "chain" in by_field["club"].text


def test_line_format_and_field_list_come_from_settings() -> None:
    cfg = mp.load_panel_settings()
    fields = ({"field": "club", "label": "Shaft", "keys": ["flex"]},)
    custom = mp.CardPanelSettings(**{**cfg.as_dict(), "fields": fields})
    custom = mp.CardPanelSettings(
        **{**custom.as_dict(), "line_format": "{tag} / {label} = {value}"}
    )
    (line,) = mp.panel_lines(_card(), custom)
    assert line.text == "assumed / Shaft = stiff"


def test_unknown_fields_show_as_unknown() -> None:
    cfg = mp.load_panel_settings()
    lines = {ln.field: ln for ln in mp.panel_lines(_card(), cfg)}
    fit = lines["fit_error"]
    assert fit.tag == "unknown"
    assert cfg.unknown_text in fit.text
    impact = lines["impact_model"]
    assert impact.tag == "unknown"
    assert cfg.unknown_text in impact.text


def test_settings_validate_colours_positions_and_fields() -> None:
    cfg = mp.load_panel_settings()
    assert set(cfg.tag_colours) == set(mc.TAGS)
    with pytest.raises(ValueError, match="position"):
        mp.CardPanelSettings(**{**cfg.as_dict(), "position": "middle"})
    with pytest.raises(ValueError, match="tag_colours"):
        mp.CardPanelSettings(**{**cfg.as_dict(), "tag_colours": {"measured": [1, 2]}})
    with pytest.raises(ValueError, match="field"):
        bad = ({"field": "colour", "label": "X"},)
        mp.CardPanelSettings(**{**cfg.as_dict(), "fields": bad})
    with pytest.raises(TypeError, match="ModelCard"):
        mp.panel_lines({"schema": "x"}, cfg)  # type: ignore[arg-type]


def test_the_option_is_off_by_default() -> None:
    assert mp.load_panel_settings().enabled_by_default is False
    assert ExportSettings().model_card_panel is False
    assert ExportSettings.preview().model_card_panel is False
    with pytest.raises(TypeError, match="model_card_panel"):
        ExportSettings(model_card_panel="yes")  # type: ignore[arg-type]


class _Writer:
    def __init__(self, sink: list[np.ndarray]) -> None:
        self.sink = sink

    def append_data(self, frame: np.ndarray) -> None:
        self.sink.append(frame.copy())

    def close(self) -> None:
        pass


class _FakeBackend:
    engine = "fake"

    def unavailable_reason(self) -> str | None:
        return None

    def render(self, swing, settings, indices, overlay, ball=None):
        for _k in indices:
            yield {
                v: np.full((settings.height, settings.width, 3), 40, np.uint8)
                for v in settings.views
            }


def _first_frame(tmp_path: Path, panel: bool) -> np.ndarray:
    frames: list[np.ndarray] = []
    settings = ExportSettings(
        views=("overhead",),
        multiview=False,
        width=640,
        height=480,
        speeds=(1.0,),
        impact_card=False,
        model_card_panel=panel,
    )
    export_swing(
        _FakeBackend(),
        _swing(),
        settings,
        tmp_path / str(panel),
        writer_factory=lambda p, fps: _Writer(frames),
    )
    return frames[0]


def test_export_with_the_option_on_draws_the_panel(tmp_path: Path) -> None:
    off = _first_frame(tmp_path, panel=False)
    on = _first_frame(tmp_path, panel=True)
    changed = np.any(on != off, axis=2)
    assert changed.any(), "the model card panel was not drawn"
    cfg = mp.load_panel_settings()
    ys, xs = np.nonzero(changed)
    if cfg.position.endswith("right"):
        assert xs.min() >= off.shape[1] // 2
    if cfg.position.startswith("bottom"):
        assert ys.min() >= off.shape[0] // 4
    unknown_rgb = np.array(cfg.tag_colours["unknown"], dtype=np.uint8)
    assert np.any(np.all(on == unknown_rgb, axis=2)), "tag colour not drawn"


def test_cli_flag_falls_back_to_the_settings_default() -> None:
    from argparse import Namespace

    from src.tools.native_viewer_export.cli import _card_panel_flag

    assert _card_panel_flag(Namespace(model_card_panel=None)) is False
    assert _card_panel_flag(Namespace(model_card_panel=True)) is True
    assert _card_panel_flag(Namespace()) is False
