"""OF-01 (#13061): the vertical plate trial contract and its declared uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.force_plates import (
    FOOT_LABELS,
    NET_LABEL,
    PlateProvenance,
    PlateUncertainty,
    loaded_net_cop,
)
from tests.unit.force_plates import _trial_factory as tf

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


# ------------------------------------------------------------- uncertainty --
def test_force_sigma_combines_floor_and_fraction_in_quadrature() -> None:
    unc = tf.uncertainty()
    # hypot(2, 0.01 * 300) = hypot(2, 3)
    np.testing.assert_allclose(unc.force_sigma_n(np.array([300.0])), [np.hypot(2, 3)])
    np.testing.assert_allclose(unc.force_sigma_n(np.array([0.0])), [2.0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("force_floor_n", 0.0),
        ("force_floor_n", -1.0),
        ("force_floor_n", float("nan")),
        ("force_fraction", -0.01),
        ("force_fraction", float("inf")),
        ("cop_m", 0.0),
        ("sensing_depth_sigma_m", -0.001),
        ("sensing_depth_sigma_m", float("nan")),
    ],
)
def test_uncertainty_must_be_declared_positive_and_finite(field, value) -> None:
    kwargs = {
        "force_floor_n": 2.0,
        "force_fraction": 0.01,
        "cop_m": 0.01,
        "sensing_depth_sigma_m": 0.002,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        PlateUncertainty(**kwargs, source="test")


def test_uncertainty_must_name_its_source() -> None:
    with pytest.raises(ValueError, match="source"):
        PlateUncertainty(
            force_floor_n=2.0,
            force_fraction=0.0,
            cop_m=0.01,
            sensing_depth_sigma_m=0.0,
            source="",
        )


def test_uncertainty_has_no_defaults() -> None:
    with pytest.raises(TypeError):
        PlateUncertainty()  # type: ignore[call-arg]


# -------------------------------------------------------------- provenance --
def test_provenance_is_unqualified_without_a_qualification_reference() -> None:
    assert tf.provenance().qualified is False
    qualified = PlateProvenance(
        device="d",
        source_format="f",
        source_revision="r",
        file_sha256={},
        qualification="docs/development/planning/DV-13061.md#device-1",
    )
    assert qualified.qualified is True


def test_provenance_rejects_malformed_digests() -> None:
    with pytest.raises(ValueError, match="sha256"):
        PlateProvenance(
            device="d", source_format="f", source_revision="r", file_sha256={"a": "xyz"}
        )


def test_provenance_digests_are_read_only() -> None:
    prov = PlateProvenance(
        device="d", source_format="f", source_revision="r", file_sha256={"a": "0" * 64}
    )
    with pytest.raises(TypeError):
        prov.file_sha256["b"] = "1" * 64  # type: ignore[index]


@pytest.mark.parametrize("field", ["device", "source_format", "source_revision"])
def test_provenance_requires_identity_fields(field) -> None:
    kwargs = {"device": "d", "source_format": "f", "source_revision": "r"}
    kwargs[field] = ""
    with pytest.raises(ValueError, match=field):
        PlateProvenance(**kwargs, file_sha256={})


# ------------------------------------------------------------------- trial --
def test_valid_trial_exposes_feet_lead_and_trail() -> None:
    trial = tf.make_trial()
    assert FOOT_LABELS == ("left", "right")
    assert NET_LABEL == "net"
    assert trial.n_samples == 3
    assert (trial.lead_foot, trial.trail_foot) == ("left", "right")
    lh = tf.make_trial(handedness="LH")
    assert (lh.lead_foot, lh.trail_foot) == ("right", "left")


def test_net_load_and_lead_fraction_are_hand_computed() -> None:
    trial = tf.make_trial()
    np.testing.assert_allclose(trial.net_load_n(), [800.0, 800.0, 800.0])
    np.testing.assert_allclose(trial.lead_fraction(), [0.5, 0.375, 5.0 / 800.0])


def test_lead_fraction_is_nan_when_the_plates_are_unloaded() -> None:
    trial = tf.make_trial(
        load_n={
            "left": np.array([400.0, 300.0, 5.0]),
            "right": np.array([400.0, 500.0, 5.0]),
        },
        cop_m={
            "left": np.array(tf.LEFT_COP_M),
            "right": np.array([tf.RIGHT_COP_M[0], tf.RIGHT_COP_M[1], (np.nan, np.nan)]),
            "net": np.vstack([tf._net_cop()[:2], [np.nan, np.nan]]),
        },
    )
    assert np.isnan(trial.lead_fraction()[2])


def test_load_sigma_follows_the_declared_uncertainty() -> None:
    trial = tf.make_trial()
    np.testing.assert_allclose(trial.load_sigma_n("left")[1], np.hypot(2.0, 3.0))


def test_trial_arrays_are_read_only_copies() -> None:
    times = np.array(tf.TIMES_S)
    trial = tf.make_trial(times_s=times)
    times[0] = 99.0
    assert trial.times_s[0] == 0.0
    with pytest.raises(ValueError):
        trial.times_s[0] = 1.0
    with pytest.raises(ValueError):
        trial.load_n["left"][0] = 1.0


@pytest.mark.parametrize(
    "times",
    [[0.0, 0.0, 0.01], [0.0, 0.01, 0.005], [0.0, np.nan, 0.01], [0.0, 0.005]],
)
def test_times_must_be_finite_strictly_increasing_and_match(times) -> None:
    with pytest.raises(ValueError, match="times_s"):
        tf.make_trial(times_s=np.array(times))


def test_feet_must_be_exactly_left_and_right() -> None:
    with pytest.raises(ValueError, match="load_n"):
        tf.make_trial(
            load_n={"lead": np.array(tf.LEFT_N), "right": np.array(tf.RIGHT_N)}
        )
    with pytest.raises(ValueError, match="cop_m"):
        tf.make_trial(
            cop_m={"left": np.array(tf.LEFT_COP_M), "right": np.array(tf.RIGHT_COP_M)}
        )


def test_small_negative_noise_is_kept_unclamped_within_three_floor_sigmas() -> None:
    # An unloaded plate reads slightly negative from noise; clamping would bias it.
    trial = tf.make_trial(
        load_n={"left": np.array([400.0, 300.0, -5.9]), "right": np.array(tf.RIGHT_N)}
    )
    assert trial.load_n["left"][2] == -5.9


@pytest.mark.parametrize("bad", [-6.1, -1000.0, np.nan, np.inf])
def test_load_must_be_finite_and_not_below_the_noise_floor(bad) -> None:
    with pytest.raises(ValueError, match="load_n"):
        tf.make_trial(
            load_n={"left": np.array([400.0, bad, 5.0]), "right": np.array(tf.RIGHT_N)}
        )


def test_load_must_respect_declared_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        tf.make_trial(capacity_n=700.0)


def test_cop_must_be_nan_exactly_below_the_source_threshold() -> None:
    bad_left = np.array(tf.LEFT_COP_M)
    bad_left[2] = (0.0, 0.0)  # 5 N foot cannot carry a CoP
    with pytest.raises(ValueError, match="threshold"):
        tf.make_trial(
            cop_m={
                "left": bad_left,
                "right": np.array(tf.RIGHT_COP_M),
                "net": tf._net_cop(),
            }
        )
    missing = np.array(tf.RIGHT_COP_M)
    missing[0] = (np.nan, np.nan)  # 400 N foot must carry one
    with pytest.raises(ValueError, match="threshold"):
        tf.make_trial(
            cop_m={
                "left": np.array(tf.LEFT_COP_M),
                "right": missing,
                "net": tf._net_cop(),
            }
        )


def test_cop_components_are_nan_together() -> None:
    half = np.array(tf.RIGHT_COP_M)
    half[1, 1] = np.nan
    with pytest.raises(ValueError, match="cop_m"):
        tf.make_trial(
            cop_m={"left": np.array(tf.LEFT_COP_M), "right": half, "net": tf._net_cop()}
        )


def test_net_cop_must_be_the_load_weighted_cop_of_the_loaded_feet() -> None:
    net = tf._net_cop()
    net[1, 0] += 0.005
    with pytest.raises(ValueError, match="net"):
        tf.make_trial(
            cop_m={
                "left": np.array(tf.LEFT_COP_M),
                "right": np.array(tf.RIGHT_COP_M),
                "net": net,
            }
        )


@pytest.mark.parametrize("bw", [0.0, -800.0, np.nan])
def test_body_weight_must_be_positive_and_finite(bw) -> None:
    with pytest.raises(ValueError, match="body_weight_n"):
        tf.make_trial(body_weight_n=bw)


@pytest.mark.parametrize("depth", [-0.001, np.nan, np.inf])
def test_sensing_depth_must_be_finite_and_non_negative(depth) -> None:
    with pytest.raises(ValueError, match="sensing_depth_m"):
        tf.make_trial(sensing_depth_m=depth)


def test_sensing_depth_may_be_zero_and_is_exposed() -> None:
    assert tf.make_trial(sensing_depth_m=0.0).sensing_depth_m == 0.0
    assert tf.make_trial().sensing_depth_m == pytest.approx(0.015)


def test_handedness_must_be_rh_or_lh() -> None:
    with pytest.raises(ValueError, match="handedness"):
        tf.make_trial(handedness="right")


@pytest.mark.parametrize("events", [{"impact": 0.5}, {"impact": np.nan}, {"": 0.0}])
def test_events_must_be_named_and_inside_the_trial(events) -> None:
    with pytest.raises(ValueError, match="events_s"):
        tf.make_trial(events_s=events)


def test_corner_forces_must_be_eight_finite_channels() -> None:
    good = np.zeros((3, 8))
    assert tf.make_trial(corner_forces_n=good).corner_forces_n.shape == (3, 8)
    with pytest.raises(ValueError, match="corner_forces_n"):
        tf.make_trial(corner_forces_n=np.zeros((3, 4)))
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="corner_forces_n"):
        tf.make_trial(corner_forces_n=bad)


def test_net_cop_of_the_fixture_is_hand_computed() -> None:
    trial = tf.make_trial()
    # (300 * -0.18 + 500 * 0.19) / 800 and (300 * 0 + 500 * 0.01) / 800
    np.testing.assert_allclose(trial.cop_m["net"][1], [0.05125, 0.00625])
    np.testing.assert_allclose(trial.cop_m["net"][0], [0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cop_min_load_n", 0.0),
        ("capacity_n", -1.0),
        ("body_weight_n", True),
        ("body_weight_n", "800"),
    ],
)
def test_scalar_declarations_are_validated(field, value) -> None:
    with pytest.raises((ValueError, TypeError), match=field):
        tf.make_trial(**{field: value})


def test_numpy_scalars_are_accepted() -> None:
    assert tf.make_trial(body_weight_n=np.float32(800.0)).body_weight_n == 800.0


def test_event_values_must_be_numbers() -> None:
    with pytest.raises(ValueError, match="events_s"):
        tf.make_trial(events_s={"impact": "0.005"})
    with pytest.raises(ValueError, match="events_s"):
        tf.make_trial(events_s={"impact": True})


def test_infinite_cop_and_two_dimensional_times_are_refused() -> None:
    cop = np.array(tf.RIGHT_COP_M)
    cop[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        tf.make_trial(
            cop_m={"left": np.array(tf.LEFT_COP_M), "right": cop, "net": tf._net_cop()}
        )
    with pytest.raises(ValueError, match="times_s"):
        tf.make_trial(times_s=np.array([tf.TIMES_S]))


def test_load_sigma_is_only_defined_per_foot() -> None:
    with pytest.raises(ValueError, match="foot"):
        tf.make_trial().load_sigma_n("net")


def test_provenance_rejects_an_empty_calibration_id() -> None:
    with pytest.raises(ValueError, match="calibration_id"):
        PlateProvenance(
            device="d",
            source_format="f",
            source_revision="r",
            file_sha256={},
            calibration_id="",
        )


def test_replace_re_runs_the_invariants() -> None:
    import dataclasses

    with pytest.raises(ValueError, match="handedness"):
        dataclasses.replace(tf.make_trial(), handedness="X")


def test_types_are_checked() -> None:
    with pytest.raises(TypeError, match="provenance"):
        tf.make_trial(provenance="open-fp")
    with pytest.raises(TypeError, match="uncertainty"):
        tf.make_trial(uncertainty=0.01)


# ------------------------------------------------------------ net CoP rule --
@pytest.mark.parametrize(
    ("loads", "cops", "match"),
    [
        ({"left": [1.0]}, {"left": [[0, 0]], "right": [[0, 0]]}, "load_n"),
        ({"left": [1.0], "right": [1.0]}, {"left": [[0, 0]]}, "cop_m"),
        (
            {"left": [1.0, 2.0], "right": [1.0]},
            {"left": [[0, 0]], "right": [[0, 0]]},
            "shape",
        ),
        ({"left": [1.0], "right": [1.0]}, {"left": [0, 0], "right": [[0, 0]]}, "shape"),
    ],
)
def test_loaded_net_cop_validates_its_inputs(loads, cops, match) -> None:
    with pytest.raises(ValueError, match=match):
        loaded_net_cop(loads, cops)


def test_loaded_net_cop_ignores_feet_without_a_cop() -> None:
    net = loaded_net_cop(
        {"left": [400.0, 5.0, 5.0], "right": [400.0, 795.0, 5.0]},
        {
            "left": [[-0.2, 0.0], [np.nan, np.nan], [np.nan, np.nan]],
            "right": [[0.2, 0.1], [0.2, 0.1], [np.nan, np.nan]],
        },
    )
    np.testing.assert_allclose(net[0], [0.0, 0.05])
    np.testing.assert_allclose(net[1], [0.2, 0.1])
    assert np.isnan(net[2]).all()
