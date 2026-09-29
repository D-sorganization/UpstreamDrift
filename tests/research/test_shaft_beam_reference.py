"""Scientific contracts for the reduced-modal and distributed-shaft comparison."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.shaft_beam_reference import (
    BeamIdentificationConfig,
    BeamReferenceConfig,
    identify_beam_parameters,
    model_matrices,
    modal_basis,
    run_beam_reference_study,
)
from scripts.research.proximal_distal_energy.run_shaft_beam_reference import (
    write_beam_reference_evidence,
)
from scripts.research.proximal_distal_energy.make_shaft_beam_reference_figures import (
    render_beam_reference_figures,
)


pytestmark = pytest.mark.scientific


def test_beam_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="element_count"):
        replace(BeamReferenceConfig.publication_default(), element_count=1)
    with pytest.raises(ValueError, match="mode_count"):
        replace(BeamReferenceConfig.publication_default(), mode_count=0)
    with pytest.raises(ValueError, match="frequency_sigma_hz"):
        replace(
            BeamIdentificationConfig.publication_default(),
            frequency_sigma_hz=0.0,
        )


def test_shared_fe_matrices_and_modes_are_well_posed() -> None:
    config = BeamReferenceConfig.publication_default()
    mass, stiffness = model_matrices(config)
    frequencies, modes = modal_basis(mass, stiffness, config.mode_count)

    np.testing.assert_allclose(mass, mass.T, atol=1e-12)
    np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(mass)) > 0.0
    assert np.min(np.linalg.eigvalsh(stiffness)) > 0.0
    assert np.all(np.diff(frequencies) > 0.0)
    np.testing.assert_allclose(
        modes.T @ mass @ modes,
        np.eye(config.mode_count),
        atol=1e-12,
    )


def test_synthetic_parameter_identification_recovers_declared_truth() -> None:
    result = identify_beam_parameters(BeamIdentificationConfig.publication_default())

    assert result.converged
    assert result.maximum_frequency_residual_hz < 1e-6
    assert result.youngs_modulus_pa == pytest.approx(
        result.declared_truth_youngs_modulus_pa, rel=2e-4
    )
    assert result.head_mass_kg == pytest.approx(
        result.declared_truth_head_mass_kg, rel=2e-4
    )
    assert result.youngs_modulus_interval_pa[0] < result.youngs_modulus_pa
    assert result.youngs_modulus_interval_pa[1] > result.youngs_modulus_pa
    assert result.head_mass_interval_kg[0] < result.head_mass_kg
    assert result.head_mass_interval_kg[1] > result.head_mass_kg


def test_higher_order_reference_converges_and_closes_energy() -> None:
    result = run_beam_reference_study()

    assert result.element_convergence_relative < 5e-3
    assert result.reference_energy_closure_j < 2e-5
    assert result.reduced_energy_closure_j < 2e-5
    assert result.reference_peak_tip_deflection_m > 0.0
    assert result.reduced_peak_tip_deflection_m > 0.0
    assert result.high_frequency_tip_rms_discrepancy_m > 1e-5
    assert result.low_frequency_tip_rms_discrepancy_m < (
        result.high_frequency_tip_rms_discrepancy_m
    )
    assert result.claim_status == "synthetic_structural_comparison_only"


def test_writer_emits_deterministic_bounded_evidence(tmp_path) -> None:
    json_path, npz_path = write_beam_reference_evidence(tmp_path)
    first = json_path.read_bytes()
    write_beam_reference_evidence(tmp_path)

    assert json_path.read_bytes() == first
    record = json.loads(first)
    assert record["schema_version"] == "proximal-distal-shaft-beam-v1"
    assert record["claim_status"] == "synthetic_structural_comparison_only"
    assert record["calibration_status"] == "not_equipment_calibration"
    assert record["shared_fe_authority"].endswith("physics.flexible_shaft")
    assert record["open_gate"] == "couple_distributed_beam_into_forward_two_hand_solve"
    with np.load(npz_path) as arrays:
        assert arrays["high_reference_tip_deflection_m"].ndim == 1
        assert arrays["low_reduced_tip_deflection_m"].shape == arrays["time_s"].shape


def test_figure_builder_emits_paired_publication_formats(tmp_path) -> None:
    data_dir = tmp_path / "data"
    figure_dir = tmp_path / "figures"
    write_beam_reference_evidence(data_dir)
    outputs = render_beam_reference_figures(data_dir, figure_dir)

    assert len(outputs) == 6
    assert {path.suffix for path in outputs} == {".pdf", ".svg"}
    assert all(path.stat().st_size > 1_000 for path in outputs)
