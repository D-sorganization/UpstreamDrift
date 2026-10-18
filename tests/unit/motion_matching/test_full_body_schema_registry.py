"""Full-body schema registry: one dispatch point for v1 and v2 (HRG-D1, #13170).

v1 stays the default and its committed documents keep their exact bytes and
identities, because receipts are keyed by ``spec_sha256`` and never rewritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching import full_body_schema as registry
from src.shared.python.motion_matching.full_body_spec import (
    FULL_BODY_SCHEMA_VERSION,
    canonical_sha256,
    save_full_body_spec,
    upper_body_slice,
    validate_full_body_spec,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPECS = ROOT / "docs/development/full_body_models"

#: (file sha256 of the committed bytes, canonical document sha256), pinned
#: on main before HRG-D1 so a schema change can never rewrite a v1 identity.
V1_IDENTITIES = {
    "full_body_spec_v1.json": (
        "09cff1393a48b7f44e2ceaac9132473c64a33f9f853823c5e7d7fef80802e6b3",
        "06272a18eadda9fb89b2bca2c1948f7c57963ebb8e635193bf10947943c67a5f",
    ),
    "full_body_spec_v2.json": (
        "2927f5f81c9952165df1e7dc038a3d96f0eaca3a39953555233eb76124342abb",
        "4f7ba8250a152c1ce5db86abf33e66b77510bd4c7c58c02f7890a86a97dd355a",
    ),
    "full_body_spec_anthro_driver.json": (
        "a45d8c7c7168e8c68fa204f758e5a936b1816dfa256901cf3818d4d992ced930",
        "ab724a31539dcfc956b4dd3f7b75ebe380aa9319a2c442323b650c736d07eb99",
    ),
    "full_body_spec_anthro_iron7.json": (
        "5715ad935765cd37f7c5ee2abd2a4a1b2169ce5bc920e4d4eddf6af5a7833643",
        "11bf35afb936d56971dd81ddf7da40194583ff5c3f1ac6ff2a5630712372bd40",
    ),
}


def test_v1_is_the_default_schema() -> None:
    assert registry.DEFAULT_FULL_BODY_SCHEMA == FULL_BODY_SCHEMA_VERSION
    assert FULL_BODY_SCHEMA_VERSION == "full-body-v1"
    assert registry.supported_full_body_schemas() == ("full-body-v1", "full-body-v2")


@pytest.mark.parametrize("schema_id", ["full-body-v1", "full-body-v2"])
def test_registry_resolves_each_schema_by_id_and_by_document(schema_id: str) -> None:
    by_id = registry.full_body_schema(schema_id)
    assert by_id.schema_id == schema_id
    assert registry.full_body_schema({"schema_version": schema_id}) is by_id


@pytest.mark.parametrize("bad", ["full-body-v0", "", None, 1])
def test_unknown_schema_fails_closed(bad: object) -> None:
    with pytest.raises(ValueError, match="Unsupported schema version"):
        registry.full_body_schema({"schema_version": bad})


def test_require_names_the_consumer_and_accepted_set() -> None:
    v2 = {"schema_version": "full-body-v2"}
    assert registry.require_full_body_schema(v2, consumer="x").schema_id == (
        "full-body-v2"
    )
    with pytest.raises(ValueError, match=r"pelvis relocation.*full-body-v1"):
        registry.require_full_body_schema(
            v2, consumer="pelvis relocation", accepted=("full-body-v1",)
        )
    with pytest.raises(ValueError, match="accepted"):
        registry.require_full_body_schema(v2, consumer="x", accepted=("nope",))
    with pytest.raises(TypeError, match="Mapping"):
        registry.require_full_body_schema("full-body-v2", consumer="x")  # type: ignore[arg-type]


def test_trunk_topologies_are_declared_once_per_schema() -> None:
    v1 = registry.full_body_schema("full-body-v1").trunk
    v2 = registry.full_body_schema("full-body-v2").trunk
    assert v1.coordinates == ("SpineInputX", "SpineInputY", "TorsoInput")
    assert v2.coordinates == (
        "lumbar_flexion",
        "lumbar_lateral_bending",
        "lumbar_axial_rotation",
        "thoracic_flexion",
        "thoracic_lateral_bending",
        "thoracic_axial_rotation",
    )
    assert [j.level for j in v2.joints] == ["L5_S1", "T12_L1"]
    # Every trunk coordinate has exactly one ISB (isb_spine) axis and sign.
    for trunk in (v1, v2):
        axes = {c: trunk.isb_axis(c) for c in trunk.coordinates}
        assert set(a for a, _ in axes.values()) == {
            "flexion_extension",
            "lateral_bending",
            "axial_rotation",
        }


@pytest.mark.parametrize("name", sorted(V1_IDENTITIES))
def test_committed_v1_documents_keep_their_bytes_and_identity(name: str) -> None:
    path = SPECS / name
    raw = path.read_bytes()
    file_sha, canonical = V1_IDENTITIES[name]
    assert hashlib.sha256(raw).hexdigest() == file_sha
    document = json.loads(raw)
    assert document["schema_version"] == "full-body-v1"
    assert canonical_sha256(document) == canonical
    validate_full_body_spec(document, upper_body_slice(document))
    assert registry.full_body_schema(document).schema_id == "full-body-v1"


@pytest.mark.parametrize("name", sorted(V1_IDENTITIES))
def test_v1_save_round_trip_is_byte_identical(name: str, tmp_path: Path) -> None:
    document = json.loads((SPECS / name).read_text(encoding="utf-8"))
    out = save_full_body_spec(document, tmp_path / name)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert canonical_sha256(reloaded) == V1_IDENTITIES[name][1]
