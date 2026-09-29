from __future__ import annotations

import numpy as np

from src.shared.python.estimation.identifiability import (
    finite_difference_jacobian,
    probe_identifiability,
    sweep_parameter,
)
from src.shared.python.estimation.synthetic_fixtures import (
    length_mass_parameter_spec,
    two_link_observation_model,
)


def test_finite_difference_jacobian_recovers_linear_columns() -> None:
    def model(params: np.ndarray) -> np.ndarray:
        return np.array([2.0 * params[0], params[0] + 3.0 * params[1]])

    jacobian = finite_difference_jacobian(model, np.array([1.0, 2.0]))

    np.testing.assert_allclose(jacobian, [[2.0, 0.0], [1.0, 3.0]], atol=1.0e-8)


def test_probe_identifiability_finds_unobservable_mass_scale() -> None:
    parameters = np.array([0.4, 0.3, 75.0], dtype=np.float64)
    report = probe_identifiability(
        two_link_observation_model,
        parameters,
        length_mass_parameter_spec(),
        tolerance=1.0e-10,
    )

    assert report.rank == 2
    assert report.condition_number == float("inf")
    null_vectors = list(report.nullspace_directions.values())
    assert len(null_vectors) == 1
    mass_axis_alignment = abs(float(null_vectors[0][2]))
    assert mass_axis_alignment > 0.999


def test_sweep_parameter_changes_lengths_but_not_output_shape() -> None:
    parameters = np.array([0.4, 0.3, 75.0], dtype=np.float64)

    outputs = sweep_parameter(
        two_link_observation_model,
        parameters,
        parameter_index=0,
        values=np.array([0.35, 0.4, 0.45]),
    )

    assert outputs.shape == (3, two_link_observation_model(parameters).size)
    assert not np.allclose(outputs[0], outputs[-1])


def test_report_to_dict_is_json_ready() -> None:
    parameters = np.array([0.4, 0.3, 75.0], dtype=np.float64)
    report = probe_identifiability(
        two_link_observation_model,
        parameters,
        length_mass_parameter_spec(),
        tolerance=1.0e-10,
    )

    payload = report.to_dict()

    assert payload["parameter_names"] == [
        "upper_length_m",
        "lower_length_m",
        "mass_scale",
    ]
    assert payload["rank"] == 2
    assert "sv_2" in payload["nullspace_directions"]
