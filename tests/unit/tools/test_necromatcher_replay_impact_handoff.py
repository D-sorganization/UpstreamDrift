"""Native tile routing and early authenticated bundle identity guards."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import pytest

pytest.importorskip("PyQt6")

from src.tools.necromatcher import replay_impact_dialog as owner

pytestmark = pytest.mark.unit


def test_local_research_action_uses_public_authenticated_loader(monkeypatch):
    dispatched = []
    loaded = []
    monkeypatch.setattr(
        owner,
        "load_research_impact_shot",
        lambda *args: loaded.append(args) or "admitted",
    )
    dialog = SimpleNamespace(
        open_golf=SimpleNamespace(isEnabled=lambda: True),
        run={"run_id": "a" * 32},
        replay_id="replay",
        session=SimpleNamespace(library="canonical library"),
        _work=lambda operation, target: dispatched.append((operation, target)),
    )
    owner.ReplayImpactDialog._open_golf(dialog)
    assert dispatched[0][0] == "golf"
    assert dispatched[0][1]() == "admitted"
    library, replay, run, metadata = loaded[0]
    assert (library, replay, run) == ("canonical library", "replay", "a" * 32)
    assert metadata.source_kind.value == "model_contact"
    assert (
        metadata.aim_context.provenance
        == "operator_declared_identity_for_local_research"
    )
    assert metadata.aim_context.source_to_target_rotation == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert metadata.shot_id != metadata.session_id
    assert metadata.created_at_utc


def test_local_research_action_disabled_does_not_load():
    dialog = SimpleNamespace(
        open_golf=SimpleNamespace(isEnabled=lambda: False), run=None
    )
    owner.ReplayImpactDialog._open_golf(dialog)


def test_foreign_run_zip_rejected_before_receipt_or_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "wrong-run.zip"
    with ZipFile(archive, "w") as output:
        for name in ("trajectory.json", "impact-receipt.json", "result.json"):
            output.writestr(name, "{}")
        output.writestr(
            "request.json", json.dumps({"replay_id": "replay", "run_id": "b" * 32})
        )

    def unexpected(*args: Any) -> Any:
        pytest.fail("Foreign run must not reach receipt/viewer import")

    monkeypatch.setattr(owner, "load_replay_impact_receipt", unexpected)
    session = SimpleNamespace(download=lambda *args: archive)
    with pytest.raises(ValueError, match="another replay/run"):
        owner.load_verified_impact_curve(session, "replay", "a" * 32)


@pytest.mark.parametrize("kind", ["authored_replay", "kinematic_fit"])
def test_native_tile_routes_only_registered_replays(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tools.necromatcher import gui
    from src.shared.python import workspace

    dialogs: list[Any] = []

    def make_dialog(*args: Any, **kwargs: Any) -> Any:
        dialogs.append((args, kwargs))
        return SimpleNamespace(show=lambda: None)

    monkeypatch.setattr(gui, "ReplayImpactDialog", make_dialog)
    monkeypatch.setattr(
        workspace, "NativeImpactSession", lambda library: "owned session"
    )
    status = SimpleNamespace(setText=lambda message: None)
    library = SimpleNamespace(
        root=tmp_path,
        assets=lambda swing: [SimpleNamespace(dataset_id="asset", kind=kind)],
    )
    widget = SimpleNamespace(
        library=library,
        asset_list="asset",
        swing_list="swing",
        _id=lambda widget: widget,
        status=status,
        _impact_dialogs=[],
    )
    gui.NecromatcherWidget._replay_impact(widget)
    assert len(dialogs) == (1 if kind == "authored_replay" else 0)
    if dialogs:
        assert dialogs[0][0][:2] == ("asset", "owned session")
        assert dialogs[0][1] == {"library_root": tmp_path}
