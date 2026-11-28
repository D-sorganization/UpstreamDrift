"""Safe, hash-bound import of sized catalog golfer models."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.shared.python.model_appearance.player_models import export_player
from src.shared.python.workspace.necromatcher import NecromatcherLibrary
from src.shared.python.workspace.necromatcher_catalog_import import import_catalog_model
from src.shared.python.workspace.necromatcher_native import load_native_model_binding

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240)]


@pytest.fixture(scope="module")
def candidate_dir(tmp_path_factory):
    """Build one real source-attributed catalog export for persistence tests."""
    output = tmp_path_factory.mktemp("catalog-candidate") / "tiger_woods"
    export_player("tiger_woods", output)
    return output


@pytest.fixture(scope="module")
def trusted_reference_bytes():
    """Read the exact fitted Tiger baseline shipped in the model archive."""
    archive = Path(__file__).resolve().parents[3] / "assets/player_models/player-models.zip"
    with ZipFile(archive) as source:
        return source.read("reference-skins/tiger_woods/model.json")


@pytest.fixture(scope="module")
def reference_skin_dir(tmp_path_factory, trusted_reference_bytes):
    """Export a new player's appearance against a shared fitted baseline."""
    source = tmp_path_factory.mktemp("reference-baseline") / "model.json"
    source.write_bytes(trusted_reference_bytes)
    output = tmp_path_factory.mktemp("reference-skin") / "rory_mcilroy"
    export_player("rory_mcilroy", output, base_spec=source)
    return output


def test_persists_bundle_hashes_and_native_binding(candidate_dir, tmp_path):
    """Persist XML, visual definition and source manifest for later binding."""
    root = tmp_path / "chosen-library"
    library = NecromatcherLibrary.create(root)
    result = import_catalog_model(library, "tiger_woods", candidate_dir)

    reopened = NecromatcherLibrary(root)
    assert [p.subject_id for p in reopened.players()] == ["tiger_woods"]
    model = reopened.load_asset(result.model_id)
    assert model.kind == "native_model"
    assert model.metadata["coordinate_units"]
    bundle = reopened.load_asset(result.bundle_asset_id)
    assert model.metadata["bundle_asset_id"] == result.bundle_asset_id
    assert model.metadata["visual_definition_sha256"] == result.visual_sha256
    assert model.metadata["source_manifest_sha256"] == result.source_manifest_sha256
    assert (root / bundle.path).is_file()
    assert (
        reopened.load_model_definition(result.model_id)
        == (candidate_dir / "model.visual.json").read_bytes()
    )
    assert (
        reopened.load_model_source_manifest(result.model_id)
        == (candidate_dir / "receipt.json").read_bytes()
    )

    binding = load_native_model_binding(
        reopened,
        result.model_id,
        reopened.load_model_definition(result.model_id),
        tuple(model.metadata["coordinate_units"]),
    )
    assert binding.model_hash == model.metadata["hash"]
    assert binding.coordinate_units == tuple(model.metadata["coordinate_units"])

    repeated = import_catalog_model(reopened, "tiger_woods", candidate_dir)
    assert repeated == result


def test_invalid_export_leaves_library_byte_identical(candidate_dir, tmp_path):
    """Reject a stale source manifest before writing a subject or any assets."""
    root = tmp_path / "invalid-library"
    library = NecromatcherLibrary.create(root)
    bad = tmp_path / "bad-candidate"
    bad.mkdir()
    for name in (
        "model.xml",
        "model.json",
        "model.visual.json",
        "model.appearance.json",
        "receipt.json",
    ):
        (bad / name).write_bytes((candidate_dir / name).read_bytes())
    receipt_path = bad / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["profile"]["mass_kg"] += 1
    receipt_path.write_text(json.dumps(receipt))
    before = (root / "project.json").read_bytes()

    with pytest.raises(ValueError, match="profile metadata"):
        import_catalog_model(library, "tiger_woods", bad)

    assert (root / "project.json").read_bytes() == before
    assert not (root / "assets").exists()
    assert library.players() == []


def test_reference_skin_retains_trusted_physics_and_baseline_bytes(
    candidate_dir, reference_skin_dir, trusted_reference_bytes, tmp_path
):
    """Store a distinct skin candidate bound to the exact shipped baseline."""
    root = tmp_path / "chosen-library"
    library = NecromatcherLibrary.create(root)
    sized = import_catalog_model(library, "tiger_woods", candidate_dir)
    skin = import_catalog_model(
        library,
        "rory_mcilroy",
        reference_skin_dir,
        reference_baseline=trusted_reference_bytes,
        reference_baseline_id="player-models.zip:reference-skins/tiger_woods/model.json",
    )

    assert sized.model_id != skin.model_id
    assert sized.mode == "sized_candidate"
    assert skin.mode == "appearance_only"
    model = library.load_asset(skin.model_id)
    assert model.metadata["candidate_mode"] == "appearance_only"
    assert model.metadata["reference_baseline_id"] == (
        "player-models.zip:reference-skins/tiger_woods/model.json"
    )
    assert model.metadata["reference_baseline_sha256"] == (
        "sha256:" + hashlib.sha256(trusted_reference_bytes).hexdigest()
    )
    assert library.load_reference_baseline(skin.model_id) == trusted_reference_bytes

    spec = json.loads((reference_skin_dir / "model.json").read_bytes())
    baseline = json.loads(trusted_reference_bytes)
    visual = json.loads(library.load_model_definition(skin.model_id))
    visual.pop("appearance")
    assert spec == baseline == visual
    assert skin.physics_spec_sha256 == model.metadata["physics_spec_sha256"]
    assert skin.model_version == model.metadata["model_version"]
    binding = load_native_model_binding(
        library,
        skin.model_id,
        library.load_model_definition(skin.model_id),
        tuple(model.metadata["coordinate_units"]),
    )
    assert binding.model_hash == model.metadata["hash"]


def test_reference_skin_rejects_mismatched_explicit_baseline(
    reference_skin_dir, trusted_reference_bytes, tmp_path
):
    """Reject a caller baseline whose physical specification differs."""
    baseline = json.loads(trusted_reference_bytes)
    baseline["coordinate_order"] = ["not_a_catalog_coordinate"]
    library = NecromatcherLibrary.create(tmp_path / "chosen-library")
    before = (library.root / "project.json").read_bytes()

    with pytest.raises(ValueError, match="differs from the trusted reference baseline"):
        import_catalog_model(
            library,
            "rory_mcilroy",
            reference_skin_dir,
            reference_baseline=json.dumps(baseline).encode(),
            reference_baseline_id="caller:deliberately-mismatched-baseline",
        )

    assert (library.root / "project.json").read_bytes() == before
    assert library.players() == []


def test_subject_collision_does_not_partially_register(candidate_dir, tmp_path):
    """A manually existing player with the catalog id is never overwritten."""
    root = tmp_path / "collision-library"
    library = NecromatcherLibrary.create(root)
    library.add_player("tiger_woods", "A different player")
    before = (root / "project.json").read_bytes()

    with pytest.raises(Exception, match="collision|already exists"):
        import_catalog_model(library, "tiger_woods", candidate_dir)

    assert (root / "project.json").read_bytes() == before
    assert not (root / "assets").exists()
    assert library.players()[0].display_name == "A different player"
