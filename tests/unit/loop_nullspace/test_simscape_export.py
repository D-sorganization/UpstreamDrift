"""NSX-55: Simscape closed-loop export format, loader and capability gate (#12583).

The checked-in fixture is a ``synthetic_test_fixture``: it is serialised from the
analytic planar golfer, never from MATLAB, and the loader treats it as test-only
with capability NONE.  Real acceptance needs a ``matlab_export`` file produced
by R2025b.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.loop_nullspace.contracts import InputProvenance
from src.shared.python.loop_nullspace.fixtures import planar_golfer_snapshot
from src.shared.python.loop_nullspace.simscape_export import (
    SCHEMA_ID,
    export_from_snapshot,
    export_to_dict,
    load_simscape_export,
    parse_simscape_export,
    snapshot_from_export,
    verify_model_file,
)

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures/loop_nullspace/simscape/synthetic_planar_closed_loop_v1.json"
)
SHA = "a" * 64
INPUT = InputProvenance("test", "unit")


# The analytic snapshot is computed with numpy/LAPACK, whose results differ in
# the last bit across BLAS builds (CI measured 4.4e-16 abs / 2.1e-16 rel, one
# ulp); floats are compared to a few ulps, everything else exactly.
_RTOL = 1e-12
_ATOL = 1e-14


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _assert_tree_close(actual: Any, expected: Any, path: str = "$") -> None:
    """Exact structure and non-float leaves; floats within a few ulps."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            _assert_tree_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_tree_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)), path
        assert np.isclose(actual, expected, rtol=_RTOL, atol=_ATOL), path
    else:
        assert actual == expected, path


def _matlab_doc(**prov: Any) -> dict[str, Any]:
    """A fixture body relabelled as an R2025b export (structure only)."""
    doc = _fixture()
    doc["provenance"].update(
        provenance_kind="matlab_export",
        model_scope="golf_model",
        matlab_release="R2025b",
        matlab_version="25.2.0.0 (R2025b)",
        model_file="rig.slx",
        model_sha256=SHA,
        exported_at="2026-10-14T00:00:00Z",
    )
    doc["provenance"].update(prov)
    return doc


def test_fixture_is_honestly_labelled_synthetic() -> None:
    prov = _fixture()["provenance"]
    assert prov["provenance_kind"] == "synthetic_test_fixture"
    assert prov["matlab_version"] is None
    assert prov["matlab_release"] is None
    assert prov["model_sha256"] is None


def test_fixture_loads_with_capability_none() -> None:
    export = load_simscape_export(FIXTURE)
    assert export.schema == SCHEMA_ID
    assert export.capability() is CapabilityLevel.NONE
    assert "synthetic_test_fixture" in export.capability_reason()
    assert len(export.states) >= 1


def test_fixture_matches_analytic_planar_snapshot() -> None:
    export = load_simscape_export(FIXTURE)
    ref = planar_golfer_snapshot().snapshot
    snap = snapshot_from_export(export, 0, input_provenance=INPUT, test_only=True)
    for name in ("q", "v", "a", "M", "h", "J_c", "Jdot_v", "S"):
        np.testing.assert_allclose(
            getattr(snap, name), getattr(ref, name), rtol=_RTOL, atol=_ATOL
        )
    assert snap.topology == ref.topology


def test_fixture_is_regenerated_exactly_from_the_analytic_model() -> None:
    expected = export_to_dict(export_from_snapshot(planar_golfer_snapshot().snapshot))
    _assert_tree_close(_fixture(), expected)


def test_tree_comparison_rejects_a_real_difference() -> None:
    doc = _fixture()
    bumped = copy.deepcopy(doc)
    bumped["states"][0]["q"][0] += 1e-9
    with pytest.raises(AssertionError):
        _assert_tree_close(bumped, doc)
    renamed = copy.deepcopy(doc)
    renamed["provenance"]["provenance_kind"] = "matlab_export"
    with pytest.raises(AssertionError):
        _assert_tree_close(renamed, doc)


def test_snapshot_refused_without_test_only_for_synthetic() -> None:
    export = load_simscape_export(FIXTURE)
    with pytest.raises(CapabilityUnavailable, match="synthetic_test_fixture"):
        snapshot_from_export(export, 0, input_provenance=INPUT)


def test_test_only_snapshot_is_tagged_in_model_provenance() -> None:
    export = load_simscape_export(FIXTURE)
    snap = snapshot_from_export(export, 0, input_provenance=INPUT, test_only=True)
    assert "synthetic_test_fixture" in snap.model_provenance.version


def test_r2025b_matlab_export_gets_full_capability() -> None:
    export = parse_simscape_export(_matlab_doc())
    assert export.capability() is CapabilityLevel.FULL
    assert export.declared_capability(Capability.CLOSED_LOOP_SNAPSHOT) is (
        CapabilityLevel.FULL
    )
    assert export.declared_capability(Capability.MASS_MATRIX) is (CapabilityLevel.NONE)
    snap = snapshot_from_export(export, 0, input_provenance=INPUT)
    assert snap.model_provenance.content_hash == SHA
    assert snap.model_provenance.name == "simscape"


@pytest.mark.parametrize("release", ["R2026a", "R2025a", "R2024b"])
def test_other_matlab_releases_are_not_acceptance_evidence(release: str) -> None:
    doc = _matlab_doc(matlab_release=release, matlab_version=f"25.9.0.0 ({release})")
    export = parse_simscape_export(doc)
    assert export.capability() is CapabilityLevel.NONE
    assert release in export.capability_reason()
    with pytest.raises(CapabilityUnavailable, match=release):
        snapshot_from_export(export, 0, input_provenance=INPUT)


def test_matlab_export_requires_full_provenance() -> None:
    for key in ("matlab_version", "model_file", "model_sha256", "exported_at"):
        doc = _matlab_doc()
        doc["provenance"][key] = None
        with pytest.raises(ValueError, match=key):
            parse_simscape_export(doc)


def test_matlab_version_must_name_the_claimed_release() -> None:
    doc = _matlab_doc(matlab_version="25.9.0.0 (R2026a)")
    with pytest.raises(ValueError, match="matlab_version"):
        parse_simscape_export(doc)


@pytest.mark.parametrize("sha", ["abc", "G" * 64, "A" * 64, 5])
def test_model_sha256_must_be_lowercase_hex64(sha: Any) -> None:
    with pytest.raises(ValueError, match="model_sha256"):
        parse_simscape_export(_matlab_doc(model_sha256=sha))


def test_synthetic_fixture_must_not_carry_matlab_identity() -> None:
    for key, value in (
        ("matlab_version", "25.2.0.0 (R2025b)"),
        ("matlab_release", "R2025b"),
        ("model_sha256", SHA),
    ):
        doc = _fixture()
        doc["provenance"][key] = value
        with pytest.raises(ValueError, match=key):
            parse_simscape_export(doc)


def test_unknown_provenance_kind_is_rejected() -> None:
    doc = _fixture()
    doc["provenance"]["provenance_kind"] = "hand_typed"
    with pytest.raises(ValueError, match="provenance_kind"):
        parse_simscape_export(doc)


def test_schema_version_must_match() -> None:
    doc = _fixture()
    doc["schema"] = "simscape-closed-loop-snapshot/v2"
    with pytest.raises(ValueError, match="schema"):
        parse_simscape_export(doc)


def _mutate(path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    doc = copy.deepcopy(_fixture())
    node: Any = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return doc


def test_non_finite_values_are_rejected() -> None:
    doc = _mutate(("states", 0, "h", 0), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        parse_simscape_export(doc)


def test_wrong_shape_is_rejected() -> None:
    doc = _mutate(("states", 0, "J_c"), [[1.0, 2.0]])
    with pytest.raises(ValueError, match="J_c"):
        parse_simscape_export(doc)


def test_asymmetric_mass_matrix_is_rejected() -> None:
    doc = copy.deepcopy(_fixture())
    doc["states"][0]["M"][0][1] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        parse_simscape_export(doc)


def test_indefinite_mass_matrix_is_rejected() -> None:
    doc = copy.deepcopy(_fixture())
    n = len(doc["states"][0]["q"])
    doc["states"][0]["M"] = (-np.eye(n)).tolist()
    with pytest.raises(ValueError, match="positive definite"):
        parse_simscape_export(doc)


def test_malformed_json_text_is_a_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_simscape_export(bad)


def test_empty_states_rejected() -> None:
    doc = _fixture()
    doc["states"] = []
    with pytest.raises(ValueError, match="states"):
        parse_simscape_export(doc)


def test_state_index_out_of_range() -> None:
    export = load_simscape_export(FIXTURE)
    with pytest.raises(ValueError, match="index"):
        snapshot_from_export(export, 99, input_provenance=INPUT, test_only=True)


def test_verify_model_file_detects_mismatch(tmp_path: Path) -> None:
    import hashlib

    model = tmp_path / "rig.slx"
    model.write_bytes(b"model bytes")
    good = hashlib.sha256(b"model bytes").hexdigest()
    export = parse_simscape_export(_matlab_doc(model_sha256=good))
    verify_model_file(export, model)
    model.write_bytes(b"edited")
    with pytest.raises(ValueError, match="model_sha256"):
        verify_model_file(export, model)


def test_verify_model_file_refuses_synthetic() -> None:
    export = load_simscape_export(FIXTURE)
    with pytest.raises(ValueError, match="no model"):
        verify_model_file(export, FIXTURE)


# --------------------------------------------------- real R2025b rig export --

REAL = FIXTURE.with_name("r2025b_planar_rig_export_v1.json")
# Measured agreement of the checked-in export with the analytic model is below
# 1e-13 of each quantity's largest entry; the bound leaves three orders margin.
RIG_RTOL = 1e-10


def test_checked_in_r2025b_export_is_a_genuine_matlab_export() -> None:
    export = load_simscape_export(REAL)
    prov = export.provenance
    assert prov.provenance_kind == "matlab_export"
    assert prov.matlab_release == "R2025b"
    assert "(R2025b)" in str(prov.matlab_version)
    assert prov.model_file == "NSX_PlanarRig.slx"
    assert prov.model_scope == "verification_rig"
    assert export.capability() is CapabilityLevel.NONE
    assert "verification rig" in export.capability_reason()
    assert "Not the golf model" in prov.note


def test_r2025b_rig_export_agrees_with_the_analytic_planar_model() -> None:
    export = load_simscape_export(REAL)
    with pytest.raises(CapabilityUnavailable, match="verification rig"):
        snapshot_from_export(export, 0, input_provenance=INPUT)
    snap = snapshot_from_export(export, 0, input_provenance=INPUT, test_only=True)
    ref = planar_golfer_snapshot().snapshot
    np.testing.assert_array_equal(snap.q, ref.q)
    for name in ("M", "h", "J_c", "Jdot_v"):
        got, want = getattr(snap, name), getattr(ref, name)
        scale = float(np.max(np.abs(want)))
        assert np.max(np.abs(got - want)) <= RIG_RTOL * scale, name
    np.testing.assert_allclose(
        snap.grip_frame.origin, ref.grip_frame.origin, rtol=0, atol=1e-12
    )


def test_r2025b_rig_export_satisfies_the_closure_identities() -> None:
    export = load_simscape_export(REAL)
    snap = snapshot_from_export(export, 0, input_provenance=INPUT, test_only=True)
    assert np.max(np.abs(snap.J_c @ snap.v)) < 1e-12
    assert np.max(np.abs(snap.J_c @ snap.a + snap.Jdot_v)) < 1e-12


def test_rig_spec_carries_inputs_only_and_matches_the_fixture_layout() -> None:
    from src.shared.python.loop_nullspace.simscape_rig_spec import planar_rig_spec

    spec = planar_rig_spec()
    assert set(spec["rig"]) >= {"upper_arm_length", "club_mass", "gravity"}
    assert spec["closure_rows"] == [1, 2]
    assert spec["topology"] == _fixture()["topology"]
    assert spec["inertias"] == _fixture()["inertias"]
    state = spec["states"][0]
    assert set(state) == {"q", "v", "a"}  # no M, h, J_c: those come from Simscape
    assert "Not the golf model" in spec["note"]


def test_verification_rig_scope_never_grants_capability_even_from_r2025b() -> None:
    export = parse_simscape_export(_matlab_doc(model_scope="verification_rig"))
    assert export.provenance.matlab_release == "R2025b"
    assert export.capability() is CapabilityLevel.NONE
    with pytest.raises(CapabilityUnavailable, match="verification rig"):
        snapshot_from_export(export, 0, input_provenance=INPUT)
    snapshot_from_export(export, 0, input_provenance=INPUT, test_only=True)


def test_golf_model_r2026a_is_none() -> None:
    doc = _matlab_doc(matlab_release="R2026a", matlab_version="26.1 (R2026a)")
    assert parse_simscape_export(doc).capability() is CapabilityLevel.NONE


@pytest.mark.parametrize("scope", ["", "golf", None])
def test_model_scope_must_be_a_known_value(scope: Any) -> None:
    with pytest.raises(ValueError, match="model_scope"):
        parse_simscape_export(_matlab_doc(model_scope=scope))


def test_model_scope_is_required() -> None:
    doc = _fixture()
    del doc["provenance"]["model_scope"]
    with pytest.raises(ValueError, match="model_scope"):
        parse_simscape_export(doc)


def test_synthetic_fixture_cannot_claim_golf_scope() -> None:
    doc = _fixture()
    doc["provenance"]["model_scope"] = "golf_model"
    with pytest.raises(ValueError, match="model_scope"):
        parse_simscape_export(doc)
