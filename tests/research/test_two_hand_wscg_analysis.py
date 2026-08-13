"""Scientific acceptance tests for the archived WSCG two-hand audit."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import rcParams

from scripts.research.proximal_distal_energy.make_two_hand_wscg_figures import _style
from scripts.research.proximal_distal_energy.run_two_hand_wscg_analysis import (
    build_outputs,
    write_outputs,
)


def test_publication_style_uses_portable_minus_glyph() -> None:
    """Core PDF fonts must not render negative ticks as missing-glyph question marks."""
    _style()

    assert rcParams["pdf.use14corefonts"] is True
    assert rcParams["axes.unicode_minus"] is False


@pytest.fixture(scope="module")
def evidence() -> tuple[dict, dict[str, np.ndarray]]:
    return build_outputs()


def test_archived_force_and_couple_reconstruction_closes(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    for case in record["cases"].values():
        assert case["maximum_resultant_reconstruction_residual_n"] < 2e-9
        assert case["maximum_couple_reconstruction_residual_nm"] < 0.1
        assert case["maximum_internal_force_power_identity_residual_w"] < 1e-8


def test_pointwise_ztcf_has_no_command_but_retains_negative_couple(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    ztcf = record["cases"]["ztcf"]

    assert ztcf["maximum_abs_individual_command_torque_nm"] == pytest.approx(0.0)
    assert ztcf["maximum_abs_net_command_torque_nm"] == pytest.approx(0.0)
    assert ztcf["minimum_equivalent_couple_nm"] < -19.0
    assert abs(ztcf["free_torque_at_minimum_nm"]) < 1e-9
    assert ztcf["force_moment_at_minimum_nm"] < -19.0


def test_base_ztcf_delta_decomposition_is_exact(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    residual = record["decomposition"][
        "maximum_abs_base_minus_ztcf_minus_delta_couple_residual_nm"
    ]
    assert residual < 1e-9


def test_reported_late_reversal_is_recovered_with_small_resampling_shift(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    base = record["cases"]["base"]
    ztcf = record["cases"]["ztcf"]

    assert base["crossings"]["late"]["time_s"] == pytest.approx(0.2708239208821296)
    assert ztcf["crossings"]["late"]["time_s"] == pytest.approx(0.27001592205391867)
    assert base["late_crossing_resampling"]["maximum_absolute_shift_s"] < 1e-5
    assert ztcf["late_crossing_resampling"]["maximum_absolute_shift_s"] < 1e-5


def test_spacing_and_rigid_rotation_geometry_contracts(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, arrays = evidence
    scale = arrays["sweep__spacing_scale"]
    moment = arrays["sweep__spacing_force_moment_nm"]
    unit_moment = moment[np.flatnonzero(np.isclose(scale, 1.0))[0]]

    np.testing.assert_allclose(moment, scale * unit_moment, atol=1e-11)
    assert record["geometry_sweep"]["maximum_co_rotation_residual_nm"] < 1e-10


def test_evidence_registers_complete_provenance_and_cache_precision(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence

    assert record["schema_version"] == "wscg-two-hand-wrench-audit-v2"
    assert "git_sha" not in record["provenance"]
    assert len(record["source_sha256"]) == 9
    assert record["state_match"]["maximum_contact_position_difference_m"] < 2e-7
    assert record["state_match"]["maximum_contact_velocity_difference_m_s"] < 1e-5
    assert record["state_match"]["maximum_clubhead_speed_difference_mph"] < 4e-5
    for case in record["cases"].values():
        assert case["maximum_local_force_projection_residual_n"] < 1.1
        assert case["maximum_out_of_plane_to_in_plane_force_ratio"] < 0.003


def test_power_claim_exposes_archive_discrepancy_and_uses_crossing_interval(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    ztcf = record["cases"]["ztcf"]

    assert ztcf["maximum_archived_linear_power_discrepancy_w"] > 70.0
    assert ztcf["maximum_internal_force_power_identity_residual_w"] < 1e-8
    assert ztcf["negative_interval_duration_s"] == pytest.approx(
        ztcf["crossings"]["late"]["time_s"] - ztcf["crossings"]["all"][0]["time_s"]
    )
    assert ztcf["negative_interval_reconstructed_force_work_j"] == pytest.approx(
        -149.41705223245467
    )
    assert ztcf["negative_interval_archived_linear_work_j"] == pytest.approx(
        -149.4489293798335
    )
    assert ztcf["negative_interval_force_power_sign_disagreement_fraction"] == 0.0
    assert ztcf["maximum_clubhead_speed_time_s"] == pytest.approx(0.1933)


def test_evidence_outputs_are_byte_deterministic(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_json, first_npz = write_outputs(first)
    second_json, second_npz = write_outputs(second)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_npz.read_bytes() == second_npz.read_bytes()
