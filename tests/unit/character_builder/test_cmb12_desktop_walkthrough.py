"""CMB-12 (#11663): desktop walkthrough screenshots stay reproducible."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/capture_character_builder_screenshots.py"
TUTORIAL = REPO_ROOT / "docs/user_guide/character_builder_tutorial.md"


def _script():
    spec = importlib.util.spec_from_file_location("capture_cb_shots", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="Path"):
        _script().capture("not/a/path")


def test_tutorial_embeds_every_committed_screenshot() -> None:
    script = _script()
    text = TUTORIAL.read_text(encoding="utf-8")
    linked = re.findall(r"!\[[^\]]+\]\((images/character_builder/[^)]+)\)", text)
    assert linked == [f"images/character_builder/{n}" for n in script.STEP_FILES]
    for rel in linked:
        assert (TUTORIAL.parent / rel).is_file(), rel


@pytest.mark.parametrize("fmt", ["spec", "urdf", "mjcf", "osim"])
def test_desktop_custom_edit_matches_api_preset_override(
    tmp_path: Path, fmt: str
) -> None:
    """The walkthrough's claim: same hash and same bytes, only the name differs."""
    from src.shared.python.humanoid_character_builder import spec_export
    from src.tools.character_builder.core import CharacterBuilderModel

    script = _script()
    model = CharacterBuilderModel(script.PRESET_ID)
    model.set_parameter("mass_kg", script.MASS_KG)
    api = spec_export.compile_character(script.PRESET_ID, {"mass_kg": script.MASS_KG})
    assert model.compile().sha256 == api.sha256
    path = model.export(fmt, tmp_path)
    api_text, _media, api_name = spec_export.export_character(api, fmt)
    assert path.name.startswith("custom_") and api_name.startswith(script.PRESET_ID)
    assert path.read_text(encoding="utf-8") == api_text


def test_capture_renders_each_step(tmp_path: Path) -> None:
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6 import QtGui

    script = _script()
    paths = script.capture(tmp_path)
    assert [p.name for p in paths] == list(script.STEP_FILES)
    for path in paths:
        image = QtGui.QImage(str(path))
        assert not image.isNull(), path
        # grab() renders at the device pixel ratio, so allow a HiDPI multiple.
        assert image.width() % script.WIDGET_SIZE[0] == 0
        assert image.height() % script.WIDGET_SIZE[1] == 0
    # Each step changes what the builder shows, so no two frames are identical.
    assert len({p.read_bytes() for p in paths}) == len(paths)
