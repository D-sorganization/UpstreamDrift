"""NSX-100: the pure request builder of the Variability & Consistency controls."""

from __future__ import annotations

import pytest

from src.shared.python.loop_nullspace.sensitivity_options import option_catalogue
from src.tools.grip_wrench_plots.variability_request import (
    DisturbanceChoice,
    VariabilityChoices,
    parse_channels,
    sensitivity_options,
)

pytestmark = pytest.mark.unit
CAT = option_catalogue()


def _choice(**kw) -> DisturbanceChoice:
    base = dict(
        name="torque_add",
        stage="torque",
        spectrum="white",
        persistent=False,
        channels=(("shoulder", 0.004), ("elbow", 0.009)),
        units="N m",
        bandwidth_hz=None,
    )
    base.update(kw)
    return DisturbanceChoice(**base)


def test_parse_channels() -> None:
    assert parse_channels("shoulder=0.004, elbow = 0.009") == (
        ("shoulder", 0.004),
        ("elbow", 0.009),
    )
    for bad in ("", "shoulder", "a=x", "a=1,a=2", "a=-1", "a=nan"):
        with pytest.raises(ValueError):
            parse_channels(bad)


def test_options_match_the_service_model_and_diagonal_covariance() -> None:
    from src.api.services.loop_nullspace_service import SensitivityOptions

    choices = VariabilityChoices(
        disturbances=(_choice(),),
        mechanisms=(),
        strategies=None,
        outcome_priority=("face_angle",),
    )
    out = sensitivity_options(choices, CAT)
    model = SensitivityOptions.model_validate(out)
    assert model.disturbance_spec[0].covariance == [[0.004, 0.0], [0.0, 0.009]]
    assert out["outcome_priority"] == ["face_angle"]
    assert out["strategy_set"] is None
    model.to_kwargs()


def test_persistent_disturbance_has_no_spectrum() -> None:
    out = sensitivity_options(
        VariabilityChoices((_choice(persistent=True),), (), None, ("face_angle",)),
        CAT,
    )
    assert out["disturbance_spec"][0]["spectrum"] is None
    assert out["disturbance_spec"][0]["persistent"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        {"stage": "not_a_stage"},
        {"spectrum": "brown"},
    ],
)
def test_values_outside_the_catalogue_are_refused(mutate) -> None:
    with pytest.raises(ValueError, match="catalogue"):
        sensitivity_options(
            VariabilityChoices((_choice(**mutate),), (), None, ("face_angle",)), CAT
        )


def test_mechanism_strategy_and_outcome_must_be_catalogued() -> None:
    for kw in (
        {"mechanisms": ("H-INVENTED",)},
        {"strategies": ("S99",)},
        {"outcome_priority": ("elbow_angle",)},
        {"outcome_priority": ()},
    ):
        args = dict(
            disturbances=(_choice(),),
            mechanisms=(),
            strategies=None,
            outcome_priority=("face_angle",),
        )
        args.update(kw)
        with pytest.raises(ValueError):
            sensitivity_options(VariabilityChoices(**args), CAT)


def test_a_disturbance_is_required() -> None:
    with pytest.raises(ValueError, match="disturbance"):
        VariabilityChoices((), (), None, ("face_angle",))
