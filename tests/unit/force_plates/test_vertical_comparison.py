"""OF-08 (#13086): model-versus-measured vertical comparison, hand-worked oracles.

Plate trials are the OF-05 hand trials (identity placement, so stance xy is
world xy; sensing depth 15 mm; load sigma ``hypot(2 N, 1 % F)``; CoP sigma
10 mm; depth sigma 2 mm).  Model series use the engine-style foot keys ``"L"``
and ``"R"`` mapped to the plate's anatomical feet.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.biomechanics.ground_reaction import GroundReactionSeries
from src.shared.python.force_plates import (
    PlateTimeAlignment,
    ReadingOptions,
    VerticalPlateTrial,
)
from src.shared.python.force_plates.comparison import (
    ComparisonOptions,
    compare_vertical,
)
from src.shared.python.force_plates.readings import covers
from tests.unit.force_plates.test_readings import NONUNIFORM, _hand_trial, _placed

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FOOT_MAP = {"left": "L", "right": "R"}
DEPTH = 0.015
SIGMA_COP = 0.010
SIGMA_DEPTH = 0.002
LEFT_XY = (-0.2, 0.0)
RIGHT_XY = (0.2, 0.0)
FRAMES = np.arange(0.001, 0.015, 0.001)


def _align(sigma_offset_s: float = 1e-4, **kw: Any) -> PlateTimeAlignment:
    base = {"offset_s": 0.0, "sigma_offset_s": sigma_offset_s, "kind": "manual"}
    return PlateTimeAlignment(**{**base, "note": "test clock", **kw})


def _options(**kw: Any) -> ComparisonOptions:
    base = {
        "reading": ReadingOptions(coverage=2.0),
        "foot_map": FOOT_MAP,
        "frame_id": "test:identity",
        "model_identity": "model-under-test",
    }
    return ComparisonOptions(**{**base, **kw})


def _constant_trial(left: float = 100.0, right: float = 700.0) -> VerticalPlateTrial:
    return _hand_trial(
        NONUNIFORM, [left] * 4, [right] * 4, [list(LEFT_XY)] * 4, [list(RIGHT_XY)] * 4
    )


def _model(
    times: np.ndarray,
    fz: dict[str, np.ndarray],
    cop_xy: dict[str, tuple[float, float]],
    shear_x: dict[str, float] | None = None,
    ground_m: float = 0.0,
) -> GroundReactionSeries:
    shear = shear_x or {}
    n = times.size
    force, cop, contact, nan3 = {}, {}, {}, {}
    for key, load in fz.items():
        fx = shear.get(key, 0.0) * load
        force[key] = np.column_stack([fx, np.zeros(n), load])
        cop[key] = np.tile([*cop_xy[key], ground_m], (n, 1))
        contact[key] = load > 0.0
        nan3[key] = np.full((n, 3), np.nan)
    return GroundReactionSeries(
        times_s=times,
        force_n=force,
        cop_m=cop,
        free_moment_nm=nan3,
        moment_about_com_nm=nan3,
        force_moment_about_com_nm=nan3,
        in_contact=contact,
        com_m=np.full((n, 3), np.nan),
        ground_height_m=np.full(n, ground_m),
    )


def _constant_model(scale_left: float = 1.0, **kw: Any) -> GroundReactionSeries:
    n = FRAMES.size
    fz = {"L": np.full(n, 100.0 * scale_left), "R": np.full(n, 700.0)}
    return _model(FRAMES, fz, {"L": LEFT_XY, "R": RIGHT_XY}, **kw)


# ----------------------------------------------------------- coverage --
def test_covers_is_true_inside_the_trial_and_false_outside() -> None:
    placed = _placed(_constant_trial())
    opts = ReadingOptions(coverage=2.0)
    assert covers(placed, 0.0, _align(), opts)
    assert covers(placed, 0.015, _align(), opts)
    assert not covers(placed, 0.0151, _align(), opts)
    assert not covers(placed, 0.0, _align(), ReadingOptions(2.0, offset_shift_s=1e-3))


# ---------------------------------------------------------- residuals --
def test_identical_model_and_measurement_give_zero_residuals() -> None:
    result = compare_vertical(
        _constant_model(), _placed(_constant_trial()), _align(), _options()
    )
    for foot in ("left", "right"):
        assert result.load[foot].n_frames == FRAMES.size
        assert result.load[foot].max_abs_z == pytest.approx(0.0, abs=1e-12)
        assert result.cop[foot].n_frames == FRAMES.size
        assert result.cop[foot].max_abs_z == pytest.approx(0.0, abs=1e-9)
    assert result.n_skipped == 0


def test_five_percent_load_error_gives_the_hand_computed_z() -> None:
    result = compare_vertical(
        _constant_model(1.05), _placed(_constant_trial()), _align(), _options()
    )
    sigma = math.hypot(2.0, 0.01 * 100.0)  # constant trial: no skew, no bias
    left = result.load["left"]
    np.testing.assert_allclose(left.residual, 5.0)
    assert left.rms_z == pytest.approx(5.0 / sigma)
    assert left.max_abs_z == pytest.approx(5.0 / sigma)
    assert result.load["right"].max_abs_z == pytest.approx(0.0, abs=1e-12)


def test_model_cop_is_projected_onto_the_sensing_plane() -> None:
    # Left shear Fx = 0.2 Fz: the plate reports c' = c - h Fx/Fz.  Put the model
    # CoP h * 0.2 = 3 mm ahead of the measured one, so only the projection makes
    # them agree.
    ratio = 0.2
    shifted = (LEFT_XY[0] + DEPTH * ratio, LEFT_XY[1])
    n = FRAMES.size
    model = _model(
        FRAMES,
        {"L": np.full(n, 100.0), "R": np.full(n, 700.0)},
        {"L": shifted, "R": RIGHT_XY},
        shear_x={"L": ratio},
    )
    result = compare_vertical(model, _placed(_constant_trial()), _align(), _options())
    left = result.cop["left"]
    np.testing.assert_allclose(left.residual, 0.0, atol=1e-12)
    # The x sigma carries the depth term |Fx/Fz| sigma_h; y does not.
    np.testing.assert_allclose(left.sigma[:, 0], math.hypot(SIGMA_COP, ratio * 0.002))
    np.testing.assert_allclose(left.sigma[:, 1], SIGMA_COP)


def test_frames_outside_the_trial_are_skipped_and_counted() -> None:
    times = np.array([0.005, 0.010, 0.020, 0.030])
    model = _model(
        times,
        {"L": np.full(4, 100.0), "R": np.full(4, 700.0)},
        {"L": LEFT_XY, "R": RIGHT_XY},
    )
    result = compare_vertical(model, _placed(_constant_trial()), _align(), _options())
    assert result.n_compared == 2
    assert result.n_skipped == 2
    np.testing.assert_allclose(result.load["left"].times_s, [0.005, 0.010])


def test_unloaded_cop_frames_are_skipped_not_zero() -> None:
    trial = _hand_trial(
        NONUNIFORM,
        [100.0, 10.0, 10.0, 100.0],
        [700.0] * 4,
        [list(LEFT_XY), [np.nan, np.nan], [np.nan, np.nan], list(LEFT_XY)],
        [list(RIGHT_XY)] * 4,
    )
    result = compare_vertical(_constant_model(), _placed(trial), _align(), _options())
    # Every frame lies in an interval touching an unloaded sample.
    assert result.cop["left"].n_frames == 0
    assert result.cop["left"].time_of_max_s is None
    assert result.cop["right"].n_frames == FRAMES.size


# ------------------------------------------------------ lead crossing --
SLOPE_N_PER_S = 200.0 / 0.015
SHIFT_S = 0.002


def _ramp_trial() -> VerticalPlateTrial:
    # RH: lead = left rises 300 -> 500 N linearly, trail = 800 - lead.
    left = [300.0 + SLOPE_N_PER_S * t for t in NONUNIFORM]
    return _hand_trial(
        NONUNIFORM,
        left,
        [800.0 - x for x in left],
        [list(LEFT_XY)] * 4,
        [list(RIGHT_XY)] * 4,
    )


def _ramp_model() -> GroundReactionSeries:
    lead = 300.0 + SLOPE_N_PER_S * (FRAMES + SHIFT_S)  # crosses 0.5 2 ms early
    return _model(FRAMES, {"L": lead, "R": 800.0 - lead}, {"L": LEFT_XY, "R": RIGHT_XY})


def _crossing_sigma(sigma_offset_s: float) -> float:
    # At the crossing L = R = 400 N: sigma_F = hypot(2, 4) per foot, and
    # sigma_f = hypot(R sigma_L, L sigma_R) / (L + R)^2; slope df/dt = S / 800.
    sigma_load = math.hypot(2.0, 4.0)
    sigma_f = math.hypot(400.0 * sigma_load, 400.0 * sigma_load) / 800.0**2
    sigma_t = sigma_f / (SLOPE_N_PER_S / 800.0)
    return math.hypot(sigma_t, sigma_offset_s)


def test_lead_crossing_delta_and_sigma_are_hand_computed() -> None:
    result = compare_vertical(
        _ramp_model(), _placed(_ramp_trial()), _align(1e-4), _options()
    )
    crossing = result.lead_crossing
    assert crossing.measured_s == pytest.approx(0.0075)
    assert crossing.model_s == pytest.approx(0.0075 - SHIFT_S)
    assert crossing.sigma_s == pytest.approx(_crossing_sigma(1e-4), rel=1e-3)
    assert crossing.resolved
    assert crossing.delta_s == pytest.approx(-SHIFT_S)


def test_unresolved_crossing_delta_is_not_reported_as_a_number() -> None:
    # sigma_offset 1 ms: k * sigma = 2 * 1.03 ms > the 2 ms shift.
    result = compare_vertical(
        _ramp_model(), _placed(_ramp_trial()), _align(1e-3), _options()
    )
    crossing = result.lead_crossing
    assert crossing.sigma_s == pytest.approx(_crossing_sigma(1e-3), rel=1e-3)
    assert not crossing.resolved
    assert crossing.delta_s is None


def test_no_crossing_is_reported_as_absent() -> None:
    result = compare_vertical(
        _constant_model(), _placed(_constant_trial()), _align(), _options()
    )
    assert result.lead_crossing.model_s is None
    assert result.lead_crossing.measured_s is None
    assert not result.lead_crossing.resolved


# -------------------------------------------------- labels and refusals --
def test_unqualified_trial_is_labelled() -> None:
    result = compare_vertical(
        _constant_model(), _placed(_constant_trial()), _align(), _options()
    )
    assert result.qualified is False
    assert result.label == "unqualified measurement"
    assert result.frame_id == "test:identity"
    assert result.alignment_kind == "manual"
    assert result.to_dict()["label"] == "unqualified measurement"


def test_validation_against_a_self_derived_alignment_is_refused() -> None:
    derived = _align(kind="cross_correlation", derived_from_model="model-under-test")
    placed = _placed(_constant_trial())
    with pytest.raises(ValueError, match="derived from"):
        compare_vertical(
            _constant_model(), placed, derived, _options(purpose="validation")
        )
    diagnostic = compare_vertical(_constant_model(), placed, derived, _options())
    assert diagnostic.purpose == "diagnostic"


def test_model_ground_must_match_the_plate_surface() -> None:
    with pytest.raises(ValueError, match="ground"):
        compare_vertical(
            _constant_model(ground_m=0.05),
            _placed(_constant_trial()),
            _align(),
            _options(),
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"foot_map": {"left": "L"}}, "foot_map"),
        ({"foot_map": {"left": "L", "right": "X"}}, "model"),
        ({"foot_map": {"left": "L", "right": "L"}}, "distinct"),
        ({"purpose": "acceptance"}, "purpose"),
        ({"frame_id": ""}, "frame_id"),
    ],
)
def test_option_and_mapping_refusals(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compare_vertical(
            _constant_model(), _placed(_constant_trial()), _align(), _options(**changes)
        )


def test_type_refusals() -> None:
    placed = _placed(_constant_trial())
    with pytest.raises(TypeError, match="GroundReactionSeries"):
        compare_vertical(object(), placed, _align(), _options())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ComparisonOptions"):
        compare_vertical(_constant_model(), placed, _align(), object())  # type: ignore[arg-type]
