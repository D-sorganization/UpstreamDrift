"""Fixed synthetic outputs and validation order protect statistics refactoring."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/statistics_refactor_reference.json").read_text()
)


@pytest.mark.parametrize("kind", tuple(_REFERENCE["cases"]))
def test_statistics_match_pre_refactor_outputs_and_errors_exactly(kind):
    from src.tools.shot_pattern_analysis.density import shared_histogram_density
    from src.tools.shot_pattern_analysis.dispersion_stats import landing_dispersion

    function = (
        shared_histogram_density if kind.startswith("density") else landing_dispersion
    )
    for case in _REFERENCE["cases"][kind]:
        inputs = dict(case["inputs"])
        if isinstance(inputs.get("bins"), list):
            inputs["bins"] = tuple(inputs["bins"])
        if "error" in case:
            with pytest.raises(ValueError) as error:
                function(**inputs)
            observed = {
                "type": type(error.value).__name__,
                "message": str(error.value),
                "cause_type": type(error.value.__cause__).__name__
                if error.value.__cause__
                else None,
            }
            assert observed == case["error"]
        else:
            result = function(**inputs)
            if kind == "density":
                result = asdict(result)
            observed = json.loads(
                json.dumps(result, default=lambda value: value.tolist())
            )
            assert observed == case["output"]


def test_validated_density_inputs_preserve_pattern_and_mass_order():
    from src.tools.shot_pattern_analysis.density import _validated_density_inputs

    inputs = _validated_density_inputs(
        {"Fade": [[2, 1]], "Draw": [[1, -1]]},
        (3, 0),
        0.05,
        0.5,
        (4, 5),
        (0.95, 0.5),
    )
    assert list(inputs.points) == ["Fade", "Draw"]
    assert inputs.shape == (4, 5)
    assert inputs.masses == (0.95, 0.5)
    np.testing.assert_array_equal(inputs.target, [3.0, 0.0])


def test_dispersion_validation_consumes_iterable_once_after_target_validation():
    from src.tools.shot_pattern_analysis.dispersion_stats import (
        _validated_dispersion_inputs,
    )

    seen = []

    def points():
        for point in [(2.0, 1.0), (4.0, -1.0)]:
            seen.append(point)
            yield point

    with pytest.raises(ValueError, match="target_x_m must be finite"):
        _validated_dispersion_inputs(points(), float("inf"), None)
    assert seen == []
    coordinates, nominal = _validated_dispersion_inputs(points(), 3.0, (3.0, 0.0))
    assert seen == [(2.0, 1.0), (4.0, -1.0)]
    np.testing.assert_array_equal(coordinates, seen)
    np.testing.assert_array_equal(nominal, [3.0, 0.0])
