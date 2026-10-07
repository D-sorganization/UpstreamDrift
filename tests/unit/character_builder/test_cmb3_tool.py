"""CMB-3 (#11654): Character Builder desktop tool, tile and adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _model():
    from src.tools.character_builder.core import CharacterBuilderModel

    return CharacterBuilderModel


def test_core_preset_and_edit_flow() -> None:
    model = _model()("junior")
    assert model.preset is not None and model.preset.id == "junior"
    model.set_parameter("mass_kg", 50.0)
    assert model.preset is None
    assert model.parameters.mass_kg == 50.0


def test_core_rejects_bad_edits_without_changing_state() -> None:
    model = _model()("junior")
    before = model.parameters
    with pytest.raises(ValueError):
        model.set_parameter("stature_m", 9.0)
    with pytest.raises(ValueError, match="unknown"):
        model.set_parameter("wingspan", 2.0)
    assert model.parameters == before


def test_core_summary_mentions_hash_and_limits() -> None:
    text = _model()("tour_average_female").summary_text()
    assert "Spec SHA-256" in text and "Limitations" in text


@pytest.mark.parametrize("fmt", ["spec", "urdf", "mjcf", "osim"])
def test_core_export_writes_file(tmp_path: Path, fmt: str) -> None:
    path = _model()("junior").export(fmt, tmp_path)
    assert path.is_file() and path.stat().st_size > 1000


def test_core_export_matches_api_bytes(tmp_path: Path) -> None:
    from src.shared.python.humanoid_character_builder import spec_export

    path = _model()("senior").export("spec", tmp_path)
    api_text = spec_export.export_character(
        spec_export.compile_character("senior", {}), "spec"
    )[0]
    assert path.read_text(encoding="utf-8") == api_text


def test_adapter_contract_without_qt() -> None:
    from src.shared.python.launcher_embed import get_embeddable_tool

    adapter = get_embeddable_tool("character_builder")
    assert adapter is not None
    assert adapter.embed_capabilities().supports_embedded
    assert adapter.is_dirty() is False
    adapter.cleanup()
    adapter.cleanup()  # idempotent


def test_tile_registered_in_models_yaml() -> None:
    import yaml

    data = yaml.safe_load((REPO_ROOT / "src/config/models.yaml").read_text("utf-8"))
    tiles = {m["id"]: m for m in data["models"]}
    tile = tiles["character_builder"]
    assert tile["path"] == "src/tools/character_builder/__main__.py"
    assert tile["launcher"]["category"] == "tool"
    assert (REPO_ROOT / tile["path"]).is_file()


def test_entry_point_declared() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'character_builder = "src.tools.character_builder._embed_adapter"' in text


def test_manifest_points_at_the_new_tool() -> None:
    manifest = json.loads(
        (REPO_ROOT / "src/config/launcher_manifest.json").read_text("utf-8")
    )
    tile = next(t for t in manifest["tiles"] if t["id"] == "character_builder")
    assert tile["path"] == "src/tools/character_builder/__main__.py"


def test_feature_parity_no_longer_gap() -> None:
    registry = json.loads(
        (REPO_ROOT / "src/config/feature_parity.json").read_text("utf-8")
    )
    entry = registry["features"]["tools.character_builder"]
    assert entry["status"] == "parity"
    assert (REPO_ROOT / entry["pyqt"]).exists()


class TestWidget:
    @pytest.fixture
    def widget(self):
        QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from src.tools.character_builder.gui import CharacterBuilderWidget

        w = CharacterBuilderWidget()
        yield w
        w.cleanup()
        del app

    def test_initial_summary_shows_hash(self, widget) -> None:
        assert "Spec SHA-256" in widget.summary.toPlainText()

    def test_picking_preset_fills_form(self, widget) -> None:
        index = widget.preset_combo.findData("junior")
        widget.preset_combo.setCurrentIndex(index)
        assert widget.spins["stature_m"].value() == pytest.approx(1.52)
        assert widget.club_combo.currentText() == "iron7"

    def test_editing_switches_to_custom_and_recompiles(self, widget) -> None:
        widget.preset_combo.setCurrentIndex(widget.preset_combo.findData("junior"))
        before = widget.summary.toPlainText()
        widget.spins["mass_kg"].setValue(55.0)
        assert widget.preset_combo.currentIndex() == 0
        assert widget.summary.toPlainText() != before
        assert widget.model.parameters.mass_kg == 55.0

    def test_export_button_writes_file(self, widget, tmp_path: Path) -> None:
        written = widget.export("spec", str(tmp_path))
        assert written is not None and Path(written).is_file()
        assert "Wrote" in widget.status.text()
