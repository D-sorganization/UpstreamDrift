"""Scientific acceptance tests for the archived WSCG two-hand audit."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_two_hand_wscg_analysis import (
    build_outputs,
)


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
        assert case["maximum_force_power_identity_residual_w"] < 1e-8


def test_pointwise_ztcf_has_no_command_but_retains_negative_couple(
    evidence: tuple[dict, dict[str, np.ndarray]],
) -> None:
    record, _ = evidence
    ztcf = record["cases"]["ztcf"]

    assert ztcf["maximum_abs_command_torque_nm"] == pytest.approx(0.0)
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
