"""Safe, hash-bound import of sized catalog golfer models."""

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
    output = tmp_path_factory.mktemp("catalog-candidate") / "tiger_woods"
    export_player("tiger_woods", output)
    return output


@pytest.fixture(scope="module")
def trusted_reference_bytes():
    archive = (
        Path(__file__).resolve().parents[3] / "assets/player_models/player-models.zip"
    )
    with ZipFile(archive) as source:
        return source.read("reference-skins/tiger_woods/model.json")


@pytest.fixture(scope="module")
def reference_skin_dir(tmp_path_factory, trusted_reference_bytes):
    source = tmp_path_factory.mktemp("reference-baseline") / "model.json"
    source.write_bytes(trusted_reference_bytes)
    output = tmp_path_factory.mktemp("reference-skin") / "moe_norman"
    export_player("moe_norman", output, base_spec=source)
    return output


def test_persists_bundle_hashes_and_native_binding(candidate_dir, tmp_path):
    root = tmp_path / "library"
    model_record, bundle_record = import_catalog_model(
        NecromatcherLibrary.create(root), "tiger_woods", candidate_dir
    )
    library = NecromatcherLibrary(root)
    model = library.load_asset(model_record.dataset_id)
    assert model_record.dataset_id.startswith("tiger_woods-sized-mujoco-")
    definition = library.load_model_definition(model_record.dataset_id)
    assert definition == (candidate_dir / "model.visual.json").read_bytes()
    assert (
        library.load_model_source_manifest(model_record.dataset_id)
        == (candidate_dir / "receipt.json").read_bytes()
    )

    binding = load_native_model_binding(
        library,
        model_record.dataset_id,
        definition,
        tuple(model.metadata["coordinate_units"]),
    )
    assert binding.model_hash == model.metadata["hash"]
    assert import_catalog_model(library, "tiger_woods", candidate_dir) == (
        model_record,
        bundle_record,
    )


def test_reference_skin_retains_trusted_physics_and_baseline_bytes(
    reference_skin_dir, trusted_reference_bytes, tmp_path
):
    root = tmp_path / "chosen-library"
    library = NecromatcherLibrary.create(root)
    skin_model, _ = import_catalog_model(
        library,
        "moe_norman",
        reference_skin_dir,
        reference_baseline=trusted_reference_bytes,
        reference_baseline_id="player-models.zip:reference-skins/tiger_woods/model.json",
    )

    assert skin_model.metadata["candidate_mode"] == "appearance_only"
    assert skin_model.dataset_id.startswith("moe_norman-reference-skin-mujoco-")
    model = library.load_asset(skin_model.dataset_id)
    assert model.metadata["reference_baseline_id"].startswith("player-models.zip:")
    assert (
        library.load_reference_baseline(skin_model.dataset_id)
        == trusted_reference_bytes
    )

    definition = library.load_model_definition(skin_model.dataset_id)
    visual = json.loads(definition)
    visual.pop("appearance")
    assert visual == json.loads(trusted_reference_bytes)
    assert model.metadata["model_version"]
    binding = load_native_model_binding(
        library,
        skin_model.dataset_id,
        definition,
        tuple(model.metadata["coordinate_units"]),
    )
    assert binding.model_hash == model.metadata["hash"]


def test_reference_skin_rejects_mismatched_explicit_baseline(
    reference_skin_dir, trusted_reference_bytes, tmp_path
):
    baseline = json.loads(trusted_reference_bytes)
    baseline["coordinate_order"] = ["not_a_catalog_coordinate"]
    library = NecromatcherLibrary.create(tmp_path / "chosen-library")
    before = (library.root / "project.json").read_bytes()

    with pytest.raises(ValueError, match="differs from the trusted reference baseline"):
        import_catalog_model(
            library,
            "moe_norman",
            reference_skin_dir,
            reference_baseline=json.dumps(baseline).encode(),
            reference_baseline_id="caller:deliberately-mismatched-baseline",
        )

    assert (library.root / "project.json").read_bytes() == before
    assert library.players() == []


def test_subject_collision_does_not_partially_register(candidate_dir, tmp_path):
    root = tmp_path / "collision-library"
    library = NecromatcherLibrary.create(root)
    library.add_player("tiger_woods", "A different player")
    before = (root / "project.json").read_bytes()

    with pytest.raises(Exception, match="collision|already exists"):
        import_catalog_model(library, "tiger_woods", candidate_dir)

    assert (root / "project.json").read_bytes() == before
    assert not (root / "assets").exists()
    assert library.players()[0].display_name == "A different player"
