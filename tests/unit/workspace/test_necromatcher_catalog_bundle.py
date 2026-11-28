"""Atomic and hash-bound persistence for catalog model bundles."""

from hashlib import sha256
import io
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import pytest

from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.workspace.necromatcher import (
    CatalogModelBundle,
    NecromatcherLibrary,
)

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


def _register_bundle(library, args):
    return library.add_catalog_model_bundle(CatalogModelBundle(**args))


def test_bundle_persistence_is_hash_checked_and_idempotent(tmp_path):
    library = NecromatcherLibrary.create(tmp_path / "library")
    args = _bundle_args()

    _register_bundle(library, args)
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

    _register_bundle(reopened, args)
    assert (library.root / "project.json").read_bytes() == project_before
    assert len(reopened.assets(args["session_id"])) == 2
    bundle = reopened.load_asset(args["bundle_asset_id"])
    bundle_path = Path(bundle.path)
    bundle_path.write_bytes(bundle_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        reopened.load_model_definition(args["model_id"])


def test_collision_does_not_write_assets_or_replace_player(tmp_path):
    library = NecromatcherLibrary.create(tmp_path / "library")
    library.add_player("sample_player", "Existing Player")
    project_before = (library.root / "project.json").read_bytes()

    with pytest.raises(StateError, match="identity collision"):
        _register_bundle(library, _bundle_args())

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
        _register_bundle(library, _bundle_args())

    assert (library.root / "project.json").read_bytes() == project_before
    assert list((library.root / "assets").glob("*")) == []
    assert library.players() == []
    assert library.swings() == []


def test_duplicate_bundle_members_are_rejected_before_writing(tmp_path):
    library = NecromatcherLibrary.create(tmp_path / "library")
    args = _bundle_args()
    duplicate = io.BytesIO()
    with ZipFile(duplicate, "w", compression=ZIP_STORED) as archive:
        archive.writestr("model.visual.json", b'{"coordinate_order":["root_x"]}')
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("model.visual.json", b'{"coordinate_order":["root_x"]}')
        archive.writestr("receipt.json", b'{"source":"catalog"}')
    args["bundle"] = duplicate.getvalue()
    project_before = (library.root / "project.json").read_bytes()

    with pytest.raises(ValueError, match="members do not match"):
        _register_bundle(library, args)

    assert (library.root / "project.json").read_bytes() == project_before
    assert not (library.root / "assets").exists()


def test_reference_baseline_bytes_are_retained_and_hash_checked(tmp_path):
    """Persist an exact trusted baseline only for appearance-only candidates."""
    library = NecromatcherLibrary.create(tmp_path / "library")
    args = _bundle_args()
    baseline = b'{"best_fit":"reference"}'
    visual = b'{"coordinate_order":["root_x"]}'
    receipt = b'{"source":"catalog"}'
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
        for name, payload in (
            ("model.visual.json", visual),
            ("receipt.json", receipt),
            ("reference-baseline.json", baseline),
        ):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED
            archive.writestr(info, payload)
    args.update(
        bundle=buffer.getvalue(),
        candidate_mode="appearance_only",
        reference_baseline_id="reference archive member",
        reference_baseline_sha256=sha256(baseline).hexdigest(),
    )
    _register_bundle(library, args)
    assert library.load_reference_baseline(args["model_id"]) == baseline
    model = library.load_asset(args["model_id"])
    assert model.metadata["candidate_mode"] == "appearance_only"
    assert model.metadata["reference_baseline_id"] == "reference archive member"

    sized_library = NecromatcherLibrary.create(tmp_path / "sized-library")
    invalid = _bundle_args()
    invalid.update(reference_baseline_id="unbound baseline")
    with pytest.raises(ValueError, match="must not claim"):
        _register_bundle(sized_library, invalid)
