"""Write-safety contracts for visual-only player skin refreshes."""

import hashlib
import json

import pytest

from scripts import refresh_player_skins as refresh
from src.shared.python.model_appearance.schema import physics_spec_sha256

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "case, error",
    [
        ("valid", None),
        ("missing", FileNotFoundError),
        ("json", json.JSONDecodeError),
        ("type", ValueError),
        ("pose", ValueError),
    ],
)
def test_refresh_preserves_physics_and_rejects_bad_inputs(
    tmp_path, monkeypatch, case, error
):
    """Refresh visual artifacts only; reject invalid inputs before any writes."""
    folder = tmp_path / "seve_ballesteros"
    folder.mkdir()
    raw, xml = (
        b'{"coordinate_order":["root_x"],"physics":{"mass":80}}',
        b"<mujoco>old</mujoco>",
    )
    spec = json.loads(raw)
    pose = {
        "target_physics_spec_sha256": physics_spec_sha256(spec),
        "target_model_xml_sha256": hashlib.sha256(xml).hexdigest(),
        "coordinate_order": ["root_x"],
        "q": [0.25],
    }
    for name, value in (
        ("model.json", raw),
        ("model.xml", xml),
        ("review_pose.json", json.dumps(pose).encode()),
        ("receipt.json", b'{"old":true}'),
    ):
        (folder / name).write_bytes(value)
    monkeypatch.setattr(
        refresh,
        "export_full_body_mjcf",
        lambda *_a, **_k: ("<mujoco>new</mujoco>", {"exported": True}),
    )
    receipt = folder / "receipt.json"
    if case == "missing":
        receipt.unlink()
    elif case in {"json", "type"}:
        receipt.write_text("{" if case == "json" else "[]")
    elif case == "pose":
        pose["target_model_xml_sha256"] = "0" * 64
        (folder / "review_pose.json").write_text(json.dumps(pose))
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    if case == "valid":
        refresh.refresh_player(folder)
        assert (folder / "model.json").read_bytes() == raw
        visual, appearance = (
            json.loads((folder / name).read_text())
            for name in ("model.visual.json", "model.appearance.json")
        )
        assert visual["appearance"] == appearance
        receipt_data = json.loads(receipt.read_text())
        assert (receipt_data["profile"], receipt_data["export"]) == (
            refresh.player_profile(folder.name),
            {"exported": True},
        )
        updated = json.loads((folder / "review_pose.json").read_text())
        assert (updated["q"], updated["coordinate_order"]) == (
            pose["q"],
            pose["coordinate_order"],
        )
        assert (
            updated["target_model_xml_sha256"]
            == hashlib.sha256((folder / "model.xml").read_bytes()).hexdigest()
        )
    else:
        with pytest.raises(error):
            refresh.refresh_player(folder)
        assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
