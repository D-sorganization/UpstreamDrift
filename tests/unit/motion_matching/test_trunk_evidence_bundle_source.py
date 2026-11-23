"""Trunk evidence must migrate the exact embedded model, not mutable defaults."""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import full_body_migration as migration
from src.shared.python.motion_matching import full_body_v2_geometry as geometry
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.motion_matching.same_input import reference

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "docs/development/full_body_models/evidence/trunk_visuals/"
    "render_trunk_visuals_evidence.py"
)


@pytest.mark.parametrize("default_exists", [True, False])
def test_migration_uses_embedded_spec_and_records_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, default_exists: bool
) -> None:
    module_spec = importlib.util.spec_from_file_location("trunk_evidence", SCRIPT)
    assert module_spec is not None and module_spec.loader is not None
    evidence = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(evidence)
    embedded = {"coordinate_order": ["q"], "grip_offset_m": 0.1}
    spec_bytes = json.dumps(embedded, indent=2).encode()
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=("q",),
        dt_s=0.1,
        q0=np.zeros(1),
        v0=np.zeros(1),
        efforts=np.zeros((1, 1)),
        reference_q=np.zeros((2, 1)),
        reference_v=np.zeros((2, 1)),
        reference_engine="synthetic",
    )
    source = tmp_path / "source.npz"
    bundle.save(source)
    mutable = tmp_path / "mutable.json"
    if default_exists:
        # Same coordinates are insufficient: geometry differs.
        mutable.write_text(json.dumps({**embedded, "grip_offset_m": 0.9}))
    monkeypatch.setattr(evidence, "V1_SPEC", mutable, raising=False)
    monkeypatch.setattr(evidence, "V2_SPEC", tmp_path / "v2.json")
    monkeypatch.setattr(evidence, "V2_BUNDLE", tmp_path / "v2.npz")
    derived = {**embedded, "topology": "synthetic-v2"}

    def derive(doc: dict, **kwargs: object) -> dict:
        assert doc == embedded
        return derived

    class Migrator:
        def __init__(self, original: dict, target: dict) -> None:
            assert original == embedded
            assert target == derived

        def migrate_coordinates(self, q: np.ndarray, order: tuple) -> np.ndarray:
            assert order == bundle.coordinate_order
            np.testing.assert_array_equal(q, bundle.reference_q)
            return q

    def generate(
        spec: bytes, times: np.ndarray, q: np.ndarray, **kwargs: dict
    ) -> InputBundle:
        np.testing.assert_array_equal(times, [0.0, 0.1])
        return replace(bundle, spec_bytes=spec, provenance=kwargs["provenance"])

    monkeypatch.setattr(geometry, "derive_landmark_v2_spec", derive)
    monkeypatch.setattr(migration, "TrunkMigrator", Migrator)
    monkeypatch.setattr(reference, "generate_reference_bundle", generate)
    result_bytes, result_path = evidence._build_v2_bundle(source)
    result = InputBundle.load(result_path)
    assert json.loads(result_bytes) == derived
    assert (
        result.provenance["source_v1_spec_sha256"]
        == hashlib.sha256(spec_bytes).hexdigest()
    )
    assert result.provenance["derived_v2_spec_sha256"] == result.spec_sha256
    assert (
        result.provenance["source_v1_bundle_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert (
        result.provenance["migration_source_sha256"]
        == hashlib.sha256(Path(migration.__file__).read_bytes()).hexdigest()
    )
