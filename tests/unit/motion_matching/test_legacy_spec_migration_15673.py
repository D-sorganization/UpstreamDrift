"""Unit tests for migrating legacy specs to SubjectProfile and canonical pins (#15673)."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.model_appearance import canonical_golfer_gate as gate
from src.shared.python.model_appearance import golfer_model_versions as gmv
from src.shared.python.motion_matching import legacy_spec_migration as lsm
from src.shared.python.motion_matching import model_library_contracts as mlc

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_DRV = _ROOT / "tests/fixtures/club_face/spec_driver.json"
_I7 = _ROOT / "tests/fixtures/club_face/spec_iron7.json"
_V1 = _ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def _sha(label: str) -> str:
    """Return a deterministic 64-char SHA-256 hex digest for test fixtures."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _spec(*, hand: str | None = None, deprecated: bool = False) -> dict[str, Any]:
    """Load a deep-copied legacy driver spec fixture with optional overrides."""
    s = json.loads(_DRV.read_text(encoding="utf-8"))
    if deprecated:
        s["subject"]["hand_visual_backend"] = "capsule"
        s["subject"]["body_visual_backend"] = "loft"
    if hand is not None:
        s["subject"]["handedness"] = hand
    return s


def test_extract_subject_profile_from_legacy_spec() -> None:
    """Extract stature, mass, handedness, and anthropometry SHA-256 into SubjectProfile."""
    s = _spec()
    prof = lsm.extract_subject_profile_from_legacy_spec(s)
    assert isinstance(prof, mlc.SubjectProfile)
    assert prof.height_m == pytest.approx(1.71) and prof.mass_kg == pytest.approx(78.0)
    assert prof.handedness == "right"
    assert prof.source_ref == s["subject"]["de_leva_table_sha256"]
    assert len(prof.content_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        prof.mass_kg = 80.0  # type: ignore[misc]


def test_extract_subject_profile_hashes_anthropometry_and_rejects_invalid() -> None:
    """Hash anthropometry when digest absent and reject missing/negative biometrics."""
    s = {
        "subject": {"stature_m": 1.82, "mass_kg": 84.5, "handedness": "left"},
        "anthropometry": {"table": "custom", "segments": {"thigh": {"length_m": 0.44}}},
    }
    p1 = lsm.extract_subject_profile_from_legacy_spec(s, subject_id="sub_custom")
    assert p1 == lsm.extract_subject_profile_from_legacy_spec(
        s, subject_id="sub_custom"
    )
    assert p1.handedness == "left" and p1.source_ref and len(p1.source_ref) == 64
    for bad in ({}, {"subject": {"stature_m": -1.0, "mass_kg": 75.0}}):
        with pytest.raises(ValueError):
            lsm.extract_subject_profile_from_legacy_spec(bad)


def test_migrate_right_handed_legacy_spec_to_canonical() -> None:
    """Upgrade right-handed capsule/loft spec to canonical-v2 with ModelDefinition and lock."""
    s = _spec(deprecated=True)
    before, ctrl, solv = copy.deepcopy(s), _sha("ctrl"), _sha("solv")
    res = lsm.migrate_legacy_spec_to_canonical(
        s, controller_hash=ctrl, solver_hash=solv
    )
    assert s == before and res.status == "migrated"
    assert (
        res.model_definition
        and res.model_lock
        and res.subject_profile
        and res.migrated_spec
    )
    cur = gate.require_canonical_golfer(res.migrated_spec)
    assert gmv.deprecated_parts(res.migrated_spec) == ()
    assert (
        res.migrated_spec["subject"]["hand_visual_backend"] == cur.hand_visual_backend
    )
    assert res.migrated_spec["subject"]["body_visual_backend"] == "skinned_cc0"
    assert res.model_definition.registry_version == cur.version
    assert res.model_lock.plant_hash == res.model_definition.plant_hash
    assert res.model_lock.controller_hash == ctrl and res.model_lock.solver_hash == solv


@pytest.mark.parametrize("bad", ["latest", "main", "not-a-sha256"])
def test_migrate_rejects_mutable_controller_or_solver_hash(bad: str) -> None:
    """Refuse mutable or non-SHA256 controller/solver pins fail-closed."""
    s = _spec()
    with pytest.raises(mlc.MutableReferenceError):
        lsm.migrate_legacy_spec_to_canonical(
            s, controller_hash=bad, solver_hash=_sha("s")
        )
    with pytest.raises(mlc.MutableReferenceError):
        lsm.migrate_legacy_spec_to_canonical(
            s, controller_hash=_sha("c"), solver_hash=bad
        )


def test_left_handed_and_non_migratable_specs_return_deprecated_fallback() -> None:
    """Return deprecated_fallback honestly with reason for left-handed or incomplete specs."""
    lh = _spec(hand="left", deprecated=True)
    before = copy.deepcopy(lh)
    r_lh = lsm.migrate_legacy_spec_to_canonical(
        lh, controller_hash=_sha("c"), solver_hash=_sha("s")
    )
    assert lh == before and r_lh.status == "deprecated_fallback"
    assert (
        r_lh.model_definition is None
        and r_lh.model_lock is None
        and r_lh.migrated_spec is None
    )
    assert (
        r_lh.subject_profile is not None and r_lh.subject_profile.handedness == "left"
    )
    assert "left" in r_lh.reason.lower() and "wrist" in r_lh.reason.lower()

    v1 = json.loads(_V1.read_text(encoding="utf-8"))
    r_v1 = lsm.migrate_legacy_spec_to_canonical(
        v1, controller_hash=_sha("c"), solver_hash=_sha("s")
    )
    assert r_v1.status == "deprecated_fallback" and r_v1.reason.strip()
    assert r_v1.model_definition is None and r_v1.subject_profile is None


def test_batch_migrate_legacy_specs_deterministic_and_persists(tmp_path: Path) -> None:
    """Process spec batches deterministically and produce reproducible SHA-256 receipts."""
    lh_path = tmp_path / "left_driver.json"
    lh_path.write_text(json.dumps(_spec(hand="left")), encoding="utf-8")
    out = tmp_path / "library"
    ra = lsm.batch_migrate_legacy_specs([_DRV, _I7, lh_path, _V1], output_dir=out)
    rb = lsm.batch_migrate_legacy_specs([lh_path, _V1, _I7, _DRV])

    assert (
        ra.total_count == 4
        and ra.migrated_count == 2
        and ra.deprecated_fallback_count == 2
    )
    assert ra.receipt_sha256 == rb.receipt_sha256 and len(ra.receipt_sha256) == 64
    assert ra.source_paths == rb.source_paths

    res = mlc.ModelLibraryResolver.from_directory(out)
    for item in ra.results:
        if item.status == "migrated":
            assert item.model_definition and item.subject_profile
            assert (
                res.resolve(item.model_definition.content_sha256)
                == item.model_definition
            )
            assert (
                res.resolve(item.subject_profile.content_sha256) == item.subject_profile
            )
