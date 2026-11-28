# fmt: off
# ruff: noqa: E301, E302, E305, E306
"""TDD contract tests for SimulationOutput, shared wrench/CoP invariants, and lineage (#15731)."""

from __future__ import annotations
import dataclasses, hashlib, json  # noqa: E401
import pytest
from src.shared.python.biomechanics.metric_definitions.schema import DimensionKind, FilterDifferentiationDescriptor, GolfMetricDefinitionEntry, NormalisationKind, SourceKind
from src.shared.python.reporting.swing_report.simulation_output import (
    DEFAULT_COP_FZ_THRESHOLD_N, DEFAULT_SUPERPOSITION_RTOL, SCHEMA_VERSION, CapabilityFlags, FieldAvailability,
    LineageValidationError, QuantityField, RegistryAlignment, SimulationLineage, SimulationOutput,
    SimulationOutputContractError, SuperpositionGateError, UnavailableFieldError, compute_ground_cop,
    compute_midhands_wrench, evaluate_ztcf_zvcf_superposition, require_ztcf_zvcf_superposition, validate_registry_entry_alignment,
)
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_H = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)
def _sha(text: str) -> str:
    """Return a deterministic 64-hex SHA-256 digest for fixtures."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def _lineage(**over: object) -> SimulationLineage:
    """Construct a valid immutable SimulationLineage for tests."""
    kw: dict[str, object] = {
        "run_id": _H[0], "match_id": f"sha256:{_H[1]}", "model_definition_sha256": _H[2], "subject_profile_sha256": _H[3],
        "match_artifact_sha256": _H[4], "model_assets_sha256": {"golfer.xml": _sha("golfer.xml")},
        "registry_alignment": RegistryAlignment(metric_ids=("x_factor_jcs_axial",), metric_definitions_sha256=_sha("reg-v1")),
    }
    kw.update(over)
    return SimulationLineage(**kw)  # type: ignore[arg-type]
def _make_output(**over: object) -> SimulationOutput:
    """Construct a minimal valid SimulationOutput instance for tests."""
    f_l, f_r = QuantityField.available((10.0, 0.0, 5.0), units="N", frame="world_Zup"), QuantityField.available((-2.0, 4.0, 3.0), units="N", frame="world_Zup")
    m_l, m_r = QuantityField.available((0.5, -0.2, 0.1), units="N*m", frame="world_Zup"), QuantityField.available((0.1, 0.4, -0.3), units="N*m", frame="world_Zup")
    p_l, p_r = (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)
    mid = compute_midhands_wrench(f_l, f_r, p_l, p_r, left_moment_nm=m_l, right_moment_nm=m_r)
    ground = compute_ground_cop(QuantityField.available((12.0, -4.0, 750.0), units="N", frame="world_Zup"), QuantityField.available((75.0, -150.0, 10.0), units="N*m", frame="world_Zup"))
    defaults: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "lineage": _lineage(), "engine": "mujoco", "coordinate_frame": "world_Zup", "time_s": (0.0, 0.01, 0.02),
        "topology": {"dofs": ("pelvis_tx", "pelvis_ty")}, "force_application_points": {"left_hand_m": p_l, "right_hand_m": p_r, "midhands_m": mid.midhands_point_m},
        "capabilities": CapabilityFlags(FieldAvailability.AVAILABLE, FieldAvailability.AVAILABLE, FieldAvailability.AVAILABLE, ground.cop_m.availability, FieldAvailability.UNAVAILABLE, True, False),
        "left_hand_force_n": f_l, "right_hand_force_n": f_r, "left_hand_couple_nm": m_l, "right_hand_couple_nm": m_r,
        "midhands_force_n": mid.force_n, "midhands_moment_nm": mid.moment_nm, "ground_force_n": ground.ground_force_n, "ground_moment_nm": ground.ground_moment_nm, "cop_m": ground.cop_m,
    }
    defaults.update(over)
    return SimulationOutput(**defaults)  # type: ignore[arg-type]
def test_unavailable_field_never_zero_fills_and_fails_closed() -> None:
    """UNAVAILABLE fields forbid numeric payloads and fail closed on raw access."""
    miss = QuantityField.unavailable(units="N*m", frame="world_Zup", reason="no intrinsic couple")
    assert miss.availability is FieldAvailability.UNAVAILABLE and miss.values is None
    with pytest.raises(UnavailableFieldError, match="no intrinsic couple"):
        miss.require_values()
    with pytest.raises(SimulationOutputContractError, match="zero-fill|values"):
        QuantityField(FieldAvailability.UNAVAILABLE, "N*m", "world_Zup", (0.0, 0.0, 0.0), "x")
    with pytest.raises(SimulationOutputContractError, match="reason"):
        QuantityField(FieldAvailability.UNAVAILABLE, "N*m", "world_Zup", None, "")
    with pytest.raises(SimulationOutputContractError, match="finite"):
        QuantityField.available((1.0, float("nan"), 0.0), units="N", frame="world_Zup")
    out = _make_output(left_hand_couple_nm=miss)
    assert "left_hand_couple_nm" in out.unavailable_fields()
    with pytest.raises(UnavailableFieldError):
        out.require_field_values("left_hand_couple_nm")
def test_midhands_equivalent_couple_with_and_without_intrinsic_couples() -> None:
    """M_mid = M_L + M_R + (p_L - p_mid) x F_L + (p_R - p_mid) x F_R."""
    f_l, f_r = QuantityField.available((0.0, 10.0, 0.0), units="N", frame="world_Zup"), QuantityField.available((0.0, -4.0, 0.0), units="N", frame="world_Zup")
    m_l, m_r = QuantityField.available((1.0, 2.0, 3.0), units="N*m", frame="world_Zup"), QuantityField.available((-0.5, 1.0, 0.5), units="N*m", frame="world_Zup")
    p_l, p_r = (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)
    full = compute_midhands_wrench(f_l, f_r, p_l, p_r, left_moment_nm=m_l, right_moment_nm=m_r)
    assert full.midhands_point_m == pytest.approx((0.0, 0.0, 0.1)) and full.force_n.require_values() == pytest.approx((0.0, 6.0, 0.0))
    assert full.moment_nm.require_values() == pytest.approx((1.9, 3.0, 3.5)) and full.intrinsic_couples_included and full.intrinsic_couples_mode == "full"
    absent = QuantityField.unavailable(units="N*m", frame="world_Zup", reason="no couple")
    xfer = compute_midhands_wrench(f_l, f_r, p_l, p_r, left_moment_nm=absent, right_moment_nm=absent)
    assert xfer.moment_nm.require_values() == pytest.approx((1.4, 0.0, 0.0)) and not xfer.intrinsic_couples_included and xfer.intrinsic_couples_mode == "transfer_only"
    blocked = compute_midhands_wrench(QuantityField.unavailable(units="N", frame="world_Zup", reason="missing F"), f_r, p_l, p_r)
    assert blocked.force_n.values is None and blocked.moment_nm.values is None
def test_ztcf_zvcf_superposition_gate_requires_verified_state_and_tolerance() -> None:
    """Superposition passes only when verified fixed (q, qdot, contact) and <= 1e-4."""
    assert pytest.approx(1e-4) == DEFAULT_SUPERPOSITION_RTOL
    tot, ztcf = QuantityField.available((100.0, 0.0, 0.0), units="N", frame="world_Zup"), QuantityField.available((40.0, 0.0, 0.0), units="N", frame="world_Zup")
    ok, exact, bad = QuantityField.available((60.005, 0.0, 0.0), units="N", frame="world_Zup"), QuantityField.available((60.0, 0.0, 0.0), units="N", frame="world_Zup"), QuantityField.available((60.05, 0.0, 0.0), units="N", frame="world_Zup")
    assert evaluate_ztcf_zvcf_superposition(tot, ztcf, ok, fixed_state_contact_verified=True).passed
    assert not evaluate_ztcf_zvcf_superposition(tot, ztcf, exact, fixed_state_contact_verified=False).passed
    with pytest.raises(SuperpositionGateError, match="verified"):
        require_ztcf_zvcf_superposition(tot, ztcf, exact, fixed_state_contact_verified=False)
    with pytest.raises(SuperpositionGateError, match="residual"):
        require_ztcf_zvcf_superposition(tot, ztcf, bad, fixed_state_contact_verified=True)
def test_ground_cop_threshold_preserves_6d_wrench_and_marks_low_fz_unavailable() -> None:
    """CoP is UNAVAILABLE when F_z < 5.0 N while full 6D ground wrench stays intact."""
    assert pytest.approx(5.0) == DEFAULT_COP_FZ_THRESHOLD_N
    low_f, low_m = QuantityField.available((1.5, -0.5, 4.9), units="N", frame="world_Zup"), QuantityField.available((0.8, -0.4, 0.2), units="N*m", frame="world_Zup")
    low = compute_ground_cop(low_f, low_m)
    assert low.ground_force_n.require_values() == pytest.approx((1.5, -0.5, 4.9)) and low.ground_moment_nm.require_values() == pytest.approx((0.8, -0.4, 0.2))
    assert low.cop_m.values is None and low.cop_m.availability is FieldAvailability.UNAVAILABLE
    assert compute_ground_cop(low_f, low_m, fz_threshold_n=4.0).cop_m.require_values() == pytest.approx((0.4 / 4.9, 0.8 / 4.9, 0.0))
    scop = compute_ground_cop(QuantityField.available(((0.0, 0.0, 2.0), (0.0, 0.0, 100.0)), units="N", frame="world_Zup"), QuantityField.available(((0.0, 0.0, 0.0), (20.0, -30.0, 5.0)), units="N*m", frame="world_Zup")).cop_m.require_values()
    assert scop[0] is None and scop[1] == pytest.approx((0.3, 0.2, 0.0))
@pytest.mark.parametrize("bad", ["latest", "main", "feat/x", "../model", "", "abc123"])
def test_lineage_rejects_mutable_refs_and_invalid_hashes(bad: str) -> None:
    """SimulationLineage rejects mutable refs, paths, and non-SHA256 digests."""
    for key in ("run_id", "match_id", "model_definition_sha256", "subject_profile_sha256", "match_artifact_sha256"):
        with pytest.raises(LineageValidationError):
            _lineage(**{key: bad})
    with pytest.raises(LineageValidationError):
        _lineage(model_assets_sha256={"golfer.xml": bad})
def test_lineage_asset_paths_and_14421_registry_alignment() -> None:
    """Asset paths must be relative and #14421 registry alignment must stay valid."""
    with pytest.raises(LineageValidationError):
        _lineage(model_assets_sha256={"../escape.xml": _H[0]})
    for kw in ({"registry_epic": "#99999"}, {"registry_owner": "other"}, {"definition_version": "latest"}):
        with pytest.raises(LineageValidationError):
            RegistryAlignment(**kw)  # type: ignore[arg-type]
    entry = GolfMetricDefinitionEntry(metric_id="x_factor_jcs_axial", segments=("pelvis", "thorax"), landmark_proxy="joint_centres", dimension=DimensionKind.three_d, reference_frame="world_Zup", cardan_order="XYZ", sign_convention="right_handed", units="deg", event_anchors=("top", "impact"), normalisation=NormalisationKind.none, source=SourceKind.modelled, definition_version="1.0.0", cites=("DOI:10.1080/02640414.2013.775474",), filter_descriptor=FilterDifferentiationDescriptor(lowpass_cutoff_hz=12.0, method="butterworth_4th", min_sampling_hz=100.0))
    align = RegistryAlignment(definition_version="1.0.0", metric_ids=("x_factor_jcs_axial",))
    validate_registry_entry_alignment(entry, align)
    with pytest.raises(LineageValidationError):
        validate_registry_entry_alignment(entry.model_copy(update={"metric_id": "unknown"}), align)
def test_simulation_output_is_frozen_and_round_trips_json() -> None:
    """SimulationOutput enforces schema 1.0.0, frozen immutability, and JSON round-trip."""
    out = _make_output()
    assert out.schema_version == "simulation-output/1.0.0"
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.__setattr__("schema_version", "2.0.0")
    payload = out.to_json()
    assert SimulationOutput.from_json(payload) == out
    bad = json.loads(payload)
    bad["schema_version"] = "simulation-output/999.0.0"
    with pytest.raises(SimulationOutputContractError, match="schema_version"):
        SimulationOutput.from_dict(bad)
# fmt: on
