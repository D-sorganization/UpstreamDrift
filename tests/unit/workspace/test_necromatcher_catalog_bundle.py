"""Atomic and hash-bound persistence for catalog model bundles."""

from hashlib import sha256
import io
import json
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import pytest

from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.workspace.necromatcher import NecromatcherLibrary

pytestmark = pytest.mark.unit


def _bundle_args():
    visual = b'{"coordinate_order":["root_x"]}'
    receipt = b'{"source":"catalog"}'
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
        for name, payload in (("model.visual.json", visual), ("receipt.json", receipt)):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED
            archive.writestr(info, payload)
    return {
        "player_id": "sample_player",
        "display_name": "Sample Player",
        "session_id": "sample_session",
        "model_id": "sample_model",
        "bundle_asset_id": "sample_bundle",
        "model_xml": b"<mujoco/>",
        "bundle": buffer.getvalue(),
        "visual_sha256": sha256(visual).hexdigest(),
        "source_manifest_sha256": sha256(receipt).hexdigest(),
        "dofs": ("root_x",),
        "coordinate_units": ("m",),
        "source_profile_sha256": "a" * 64,
        "physics_spec_sha256": "b" * 64,
        "model_version": "catalog-v1",
    }


def test_bundle_persistence_is_hash_checked_and_idempotent(tmp_path):
    library = NecromatcherLibrary.create(tmp_path / "library")
    args = _bundle_args()

    library.add_catalog_model_bundle(**args)
    project_before = (library.root / "project.json").read_bytes()
    reopened = NecromatcherLibrary(library.root)
    model = reopened.load_asset(args["model_id"])
    assert (
        reopened.load_model_definition(args["model_id"])
        == b'{"coordinate_order":["root_x"]}'
    )
    assert (
        reopened.load_model_source_manifest(args["model_id"]) == b'{"source":"catalog"}'
    )
    assert model.metadata["bundle_asset_id"] == args["bundle_asset_id"]

    reopened.add_catalog_model_bundle(**args)
    assert (library.root / "project.json").read_bytes() == project_before
    assert len(reopened.assets(args["session_id"])) == 2


def test_collision_does_not_write_assets_or_replace_player(tmp_path):
    library = NecromatcherLibrary.create(tmp_path / "library")
    library.add_player("sample_player", "Existing Player")
    project_before = (library.root / "project.json").read_bytes()

    with pytest.raises(StateError, match="identity collision"):
        library.add_catalog_model_bundle(**_bundle_args())

    assert (library.root / "project.json").read_bytes() == project_before
    assert list((library.root / "assets").glob("*")) == []
    assert [player.display_name for player in library.players()] == ["Existing Player"]


def test_project_save_failure_rolls_back_files_and_registration(tmp_path, monkeypatch):
    library = NecromatcherLibrary.create(tmp_path / "library")
    project_before = (library.root / "project.json").read_bytes()

    def fail_save(_project):
        raise OSError("simulated durable-write failure")

    monkeypatch.setattr(library._store, "_save", fail_save)
    with pytest.raises(OSError, match="durable-write"):
        library.add_catalog_model_bundle(**_bundle_args())

    assert (library.root / "project.json").read_bytes() == project_before
    assert list((library.root / "assets").glob("*")) == []
    assert library.players() == []
    assert library.swings() == []
