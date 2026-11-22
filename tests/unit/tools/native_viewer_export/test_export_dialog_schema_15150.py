"""Export Clips dialog controls generated from the request schema (#15150 G6)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from src.tools.native_viewer_export import dialog_core as dc
from src.tools.native_viewer_export import request_schema

pytestmark = pytest.mark.unit


def _options() -> dict[str, dict[str, Any]]:
    return {o["name"]: o for o in request_schema.describe({})["options"]}


def test_widget_type_follows_option_type() -> None:
    options = _options()
    assert dc.widget_for(options["event_captions"]) == "checkbox"
    assert dc.widget_for(options["no_hud"]) == "checkbox"
    assert dc.widget_for(options["preset"]) == "combo"
    assert dc.widget_for(options["fps"]) == "int_spin"
    assert dc.widget_for(options["shading_opacity"]) == "float_spin"
    assert dc.widget_for(options["appearance"]) == "path"
    assert dc.widget_for(options["interface_ellipsoid"]) == "line"
    with pytest.raises(ValueError, match="kind"):
        dc.widget_for({"kind": "blob", "choices": None})


def test_every_settable_option_has_exactly_one_control() -> None:
    generated = [s.name for _, specs in dc.grouped_specs() for s in specs]
    assert len(generated) == len(set(generated))
    assert set(generated) | dc.HAND_OPTIONS == dc.settable_option_names()
    assert not set(generated) & dc.HAND_OPTIONS
    assert len(dc.settable_option_names()) > 80


def test_collect_options_sends_only_changes() -> None:
    specs = [s for _, g in dc.grouped_specs() for s in g]
    raw = {s.name: s.default for s in specs}
    assert dc.collect_options(specs, raw) == {}
    raw.update(fps=30, preset="preview", no_hud=True, interface_ellipsoid="a, b")
    raw.update(axial_impact_blackout=False, camera_height=None, club="  ")
    assert dc.collect_options(specs, raw) == {
        "fps": 30,
        "preset": "preview",
        "no_hud": True,
        "interface_ellipsoid": ["a", "b"],
        "axial_impact_blackout": False,
    }


def test_prefill_takes_bundle_and_swing_name() -> None:
    base = dc.default_choices()
    filled = dc.prefill_choices(base, "/x/run_a.npz")
    assert (filled.bundle, filled.swing) == ("/x/run_a.npz", "run_a")
    assert dc.prefill_choices(base, None) is base


def test_extra_options_reach_the_validated_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "b.npz").write_bytes(b"")
    monkeypatch.setenv("UD_NATIVE_EXPORT_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("UD_NATIVE_EXPORT_OUTPUT_ROOT", str(tmp_path / "out"))
    choices = dc.prefill_choices(dc.default_choices(), tmp_path / "b.npz")
    choices = dc.ClipChoices(**{**choices.__dict__, "extra_options": {"fps": 24}})
    argv = dc.validate_choices(choices)["argv"]
    assert argv[argv.index("--fps") + 1] == "24"
    bad = dc.ClipChoices(**{**choices.__dict__, "extra_options": {"fps": "x"}})
    with pytest.raises(ValueError):
        dc.validate_choices(bad)


def _qt() -> Any:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication

        from src.tools.native_viewer_export import export_dialog
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    return QApplication.instance() or QApplication([]), export_dialog


def test_dialog_has_a_widget_for_every_generated_option() -> None:
    from PyQt6.QtWidgets import QGroupBox, QScrollArea

    _, module = _qt()
    dialog = module.ExportClipsDialog(queue=object())
    names = dc.settable_option_names() - dc.HAND_OPTIONS
    assert set(dialog.option_readers) == names
    for name in names:
        assert dialog.findChild(object, f"option_{name}") is not None, name
    groups = [g for g in dialog.findChildren(QGroupBox) if g.isCheckable()]
    assert len(groups) == len(dc.grouped_specs())
    assert not any(g.isChecked() for g in groups)
    assert dialog.findChild(QScrollArea) is not None
    assert dialog.choices().extra_options == {}


def test_dialog_prefill_and_changed_option_flow_into_the_request() -> None:
    _, module = _qt()
    dialog = module.ExportClipsDialog(queue=object())
    dialog.prefill("/x/run_a.npz")
    assert dialog.swing_edit.text() == "run_a"
    dialog.findChild(object, "option_fps").setValue(30)
    dialog.findChild(object, "option_no_hud").setChecked(True)
    assert dialog.choices().extra_options == {"fps": 30, "no_hud": True}
