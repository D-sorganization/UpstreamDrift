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
