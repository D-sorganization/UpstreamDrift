"""Gates for the ball_flight_trajectory/1 export (ADR-0047 H1, #9350).

The record is defined in Tools; this repo builds it from the documented
contract without importing the vendored package at runtime. So the
schema assertions here are written out longhand rather than delegated
to a validator — they *are* this side's copy of the contract, and if
Tools changes the wire they must be the thing that fails.

The cross-family gate additionally parses a record produced here with
the vendored reader when the pin carries it, which is the only check
that proves the two halves agree rather than merely resembling each
other. It skips with a reason on a pin that predates the Tools half and
arms itself on the next vendor bump.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.physics import flight_models as fm
from src.shared.python.physics.flight_trajectory_export import (
    BALL_FLIGHT_TRAJECTORY_FORMAT,
    FLIGHT_FRAME_ID,
    PROVENANCE_FIELDS,
    TRAJECTORY_RECORD_FIELDS,
    UD_FLIGHT_FAMILY,
    VELOCITY_CHANNEL,
    flight_result_to_trajectory_record,
    trajectory_parameter_digest,
    trajectory_record_to_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

BALL_SPEED_MPS = 74.0
LAUNCH_ANGLE_RAD = math.radians(10.9)
SPIN_RATE_RPM = 2686.0
MAX_TIME_S = 12.0
TIME_STEP_S = 0.01

_VENDOR_INTERCHANGE = "shared.python.swing_sim.flight_interchange"


def _launch() -> fm.UnifiedLaunchConditions:
    return fm.UnifiedLaunchConditions(
        ball_speed=BALL_SPEED_MPS,
        launch_angle=LAUNCH_ANGLE_RAD,
        spin_rate=SPIN_RATE_RPM,
    )


def _simulate(model_type: fm.FlightModelType) -> fm.FlightResult:
    fm.FlightModelRegistry.reset()
    model = fm.FlightModelRegistry.get_model(model_type)
    try:
        return model.simulate(_launch(), max_time=MAX_TIME_S, dt=TIME_STEP_S)
    finally:
        fm.FlightModelRegistry.reset()


def _assert_schema_valid(record: dict[str, Any]) -> None:
    """Assert the record satisfies the documented wire, longhand."""
    assert sorted(record) == list(TRAJECTORY_RECORD_FIELDS)
    assert record["format"] == BALL_FLIGHT_TRAJECTORY_FORMAT
    assert record["frame_id"] == FLIGHT_FRAME_ID
    assert isinstance(record["source_id"], str)
    assert record["source_id"].strip() == record["source_id"] != ""

    channels = record["channels"]
    assert channels == sorted(set(channels))
    assert set(channels) <= {"spin_rad_s", VELOCITY_CHANNEL}

    provenance = record["provenance"]
    assert sorted(provenance) == list(PROVENANCE_FIELDS)
    for field in PROVENANCE_FIELDS:
        assert isinstance(provenance[field], str) and provenance[field]
    digest = provenance["parameter_digest"]
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)

    samples = record["samples"]
    assert len(samples) >= 2
    expected_keys = sorted({"position_m", "time_s", *channels})
    previous: float | None = None
    for sample in samples:
        assert sorted(sample) == expected_keys
        time_s = sample["time_s"]
        assert isinstance(time_s, float) and math.isfinite(time_s) and time_s >= 0.0
        assert previous is None or time_s > previous
        previous = time_s
        for channel in ("position_m", *channels):
            vector = sample[channel]
            assert isinstance(vector, list) and len(vector) == 3
            assert all(math.isfinite(component) for component in vector)


class TestParameterDigest:
    def test_matches_the_documented_algorithm(self) -> None:
        """The recipe is contract: Tools computes the identical digest."""
        parameters = {"cl": 0.24, "cd": 0.22}
        expected = hashlib.sha256(
            json.dumps(
                parameters, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        assert trajectory_parameter_digest(parameters) == expected

    def test_is_key_order_independent(self) -> None:
        assert trajectory_parameter_digest({"a": 1.0, "b": 2.0}) == (
            trajectory_parameter_digest({"b": 2.0, "a": 1.0})
        )

    def test_empty_parameters_are_refused(self) -> None:
        with pytest.raises(ValueError, match="nonempty"):
            trajectory_parameter_digest({})

    def test_non_finite_parameter_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            trajectory_parameter_digest({"cd": math.inf})

    def test_non_numeric_parameter_is_refused(self) -> None:
        with pytest.raises(TypeError, match="finite number or a string"):
            trajectory_parameter_digest({"cd": [0.22]})  # type: ignore[dict-item]


class TestNamedModelExports:
    @pytest.mark.parametrize(
        ("model_type", "expected_name"),
        [
            (fm.FlightModelType.WATERLOO_PENNER, "Waterloo/Penner"),
            (fm.FlightModelType.MACDONALD_HANZELY, "MacDonald-Hanzely"),
        ],
    )
    def test_flight_exports_to_a_schema_valid_record(
        self, model_type: fm.FlightModelType, expected_name: str
    ) -> None:
        result = _simulate(model_type)
        record = flight_result_to_trajectory_record(result, model_type=model_type)
        _assert_schema_valid(record)
        assert record["provenance"]["model_family"] == UD_FLIGHT_FAMILY
        assert record["provenance"]["model_name"] == expected_name
        assert record["channels"] == [VELOCITY_CHANNEL]
        assert len(record["samples"]) == len(result.trajectory)

    @pytest.mark.parametrize(
        "model_type",
        [fm.FlightModelType.WATERLOO_PENNER, fm.FlightModelType.MACDONALD_HANZELY],
    )
    def test_provenance_digests_the_declared_coefficients(
        self, model_type: fm.FlightModelType
    ) -> None:
        """Issue #8978's attributable coefficient set is what travels."""
        result = _simulate(model_type)
        record = flight_result_to_trajectory_record(result)
        assert result.coefficients
        assert record["provenance"]["parameter_digest"] == (
            trajectory_parameter_digest(result.coefficients)
        )

    def test_the_two_models_are_distinguishable_on_the_wire(self) -> None:
        """ADR-0047: models stay named, never reconciled."""
        penner = flight_result_to_trajectory_record(
            _simulate(fm.FlightModelType.WATERLOO_PENNER)
        )
        hanzely = flight_result_to_trajectory_record(
            _simulate(fm.FlightModelType.MACDONALD_HANZELY)
        )
        assert penner["provenance"]["model_name"] != hanzely["provenance"]["model_name"]
        assert (
            penner["provenance"]["parameter_digest"]
            != hanzely["provenance"]["parameter_digest"]
        )

    def test_samples_are_the_retained_integrator_points(self) -> None:
        """Never resampled: P8 playback replays exactly these."""
        result = _simulate(fm.FlightModelType.WATERLOO_PENNER)
        record = flight_result_to_trajectory_record(result)
        for point, sample in zip(result.trajectory, record["samples"], strict=True):
            assert sample["time_s"] == float(point.time)
            assert sample["position_m"] == [float(value) for value in point.position]
            assert sample[VELOCITY_CHANNEL] == [
                float(value) for value in point.velocity
            ]

    def test_default_source_id_names_family_and_model(self) -> None:
        result = _simulate(fm.FlightModelType.WATERLOO_PENNER)
        assert flight_result_to_trajectory_record(result)["source_id"] == (
            f"{UD_FLIGHT_FAMILY}:Waterloo/Penner"
        )
        assert (
            flight_result_to_trajectory_record(result, source_id="run-17")["source_id"]
            == "run-17"
        )


class TestDeterministicBytes:
    def test_equal_records_serialize_to_identical_bytes(self) -> None:
        result = _simulate(fm.FlightModelType.WATERLOO_PENNER)
        first = trajectory_record_to_json(flight_result_to_trajectory_record(result))
        second = trajectory_record_to_json(flight_result_to_trajectory_record(result))
        assert first == second

    def test_serialized_keys_are_sorted_and_compact(self) -> None:
        record = flight_result_to_trajectory_record(
            _simulate(fm.FlightModelType.WATERLOO_PENNER)
        )
        text = trajectory_record_to_json(record)
        assert ", " not in text and '": ' not in text
        assert text.startswith('{"channels":["velocity_mps"],"format":')
        assert json.loads(text) == record

    def test_repeated_simulations_of_one_model_agree_byte_for_byte(self) -> None:
        """A deterministic integrator plus a deterministic wire."""
        first = trajectory_record_to_json(
            flight_result_to_trajectory_record(
                _simulate(fm.FlightModelType.WATERLOO_PENNER)
            )
        )
        second = trajectory_record_to_json(
            flight_result_to_trajectory_record(
                _simulate(fm.FlightModelType.WATERLOO_PENNER)
            )
        )
        assert first == second


class TestRefusalGates:
    def test_missing_coefficients_are_refused(self) -> None:
        """An unattributable trajectory must not reach the wire."""
        result = _simulate(fm.FlightModelType.WATERLOO_PENNER)
        stripped = fm.FlightResult(
            trajectory=result.trajectory,
            model_name=result.model_name,
        )
        with pytest.raises(ValueError, match="nonempty"):
            flight_result_to_trajectory_record(stripped)

    def test_single_sample_flight_is_refused(self) -> None:
        result = _simulate(fm.FlightModelType.WATERLOO_PENNER)
        with pytest.raises(ValueError, match="fewer than two"):
            flight_result_to_trajectory_record(
                fm.FlightResult(
                    trajectory=list(result.trajectory[:1]),
                    model_name=result.model_name,
                    coefficients=dict(result.coefficients),
                )
            )

    def test_non_monotone_times_are_refused(self) -> None:
        result = _simulate(fm.FlightModelType.WATERLOO_PENNER)
        points = list(result.trajectory[:4])
        points[2] = fm.TrajectoryPoint(
            time=points[1].time,
            position=points[2].position,
            velocity=points[2].velocity,
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            flight_result_to_trajectory_record(
                fm.FlightResult(
                    trajectory=points,
                    model_name=result.model_name,
                    coefficients=dict(result.coefficients),
                )
            )

    def test_non_finite_position_is_refused(self) -> None:
        result = _simulate(fm.FlightModelType.WATERLOO_PENNER)
        points = list(result.trajectory[:4])
        points[1] = fm.TrajectoryPoint(
            time=points[1].time,
            position=np.array([0.0, math.nan, 0.0]),
            velocity=points[1].velocity,
        )
        with pytest.raises(ValueError, match="position_m must be finite"):
            flight_result_to_trajectory_record(
                fm.FlightResult(
                    trajectory=points,
                    model_name=result.model_name,
                    coefficients=dict(result.coefficients),
                )
            )

    def test_mislabelled_model_type_is_refused(self) -> None:
        """One model's samples must never carry another's identity."""
        result = _simulate(fm.FlightModelType.WATERLOO_PENNER)
        with pytest.raises(ValueError, match="disagrees with the result"):
            flight_result_to_trajectory_record(
                result, model_type=fm.FlightModelType.MACDONALD_HANZELY
            )

    def test_non_result_argument_is_refused(self) -> None:
        with pytest.raises(TypeError, match="must be a FlightResult"):
            flight_result_to_trajectory_record(object())  # type: ignore[arg-type]


def _vendor_interchange() -> Any:
    """Import the vendored record module, or skip with a reason.

    Skips on a pin that predates the Tools half of ADR-0047 H1
    (D-sorganization/Tools#4888) and on a checkout without the
    submodule materialised; arms itself on the next vendor bump with
    no edit here.
    """
    try:
        found = importlib.util.find_spec(_VENDOR_INTERCHANGE)
    except (ImportError, ValueError):
        found = None
    if found is None:
        pytest.skip(
            f"{_VENDOR_INTERCHANGE} is absent from the pinned vendor/ud-tools "
            "tree (pin predates Tools#4888, or the submodule is not "
            "materialised); the cross-family gate arms on the next pin bump"
        )
    return importlib.import_module(_VENDOR_INTERCHANGE)


class TestCrossFamilySanity:
    """ADR-0047's cross-family gates, armed by the vendor pin."""

    def test_ud_record_parses_with_the_vendored_reader(self) -> None:
        """The two halves agree, rather than merely resembling each other."""
        interchange = _vendor_interchange()
        record = flight_result_to_trajectory_record(
            _simulate(fm.FlightModelType.WATERLOO_PENNER)
        )
        text = trajectory_record_to_json(record)
        parsed = interchange.ball_flight_trajectory_from_json(text)
        assert parsed.provenance.model_family == UD_FLIGHT_FAMILY
        assert parsed.provenance.model_name == "Waterloo/Penner"
        assert parsed.frame_id == interchange.FLIGHT_FRAME_ID
        assert parsed.channels == (VELOCITY_CHANNEL,)
        assert interchange.ball_flight_trajectory_to_json(parsed) == text

    def test_digest_algorithm_agrees_across_repositories(self) -> None:
        """This repo reimplements the digest; it must produce Tools' value."""
        interchange = _vendor_interchange()
        parameters = {"cd0": 0.21, "cd1": 0.05, "lift_scale": 0.7}
        assert trajectory_parameter_digest(parameters) == (
            interchange.parameter_digest(parameters)
        )

    def test_identical_launch_conditions_give_same_order_carry(self) -> None:
        """The ADR's analytic cross-family gate.

        The families are independent implementations with their own
        coefficient calibrations, so their carries are deliberately
        *not* reconciled — only their order of magnitude is gated.
        A sign error, a frame flip, or a unit slip on either side
        breaks this band; a legitimate calibration difference does not.

        Measured at authoring time against Tools#4888: 244.65 m from
        both families, ratio 1.0000 — Tools' family began as a port of
        this one and their Waterloo/Penner defaults still coincide. The
        band stays deliberately loose so that an honest recalibration on
        either side does not fail a gate that exists to catch defects.
        """
        _vendor_interchange()
        from shared.python.swing_sim.flight import (
            FlightModelRegistry as ToolsRegistry,
        )
        from shared.python.swing_sim.flight import (
            FlightModelType as ToolsModelType,
        )
        from shared.python.swing_sim.flight import (
            LaunchConditions as ToolsLaunch,
        )

        ud_carry = _simulate(fm.FlightModelType.WATERLOO_PENNER).carry_distance
        tools_model = ToolsRegistry.get_model(ToolsModelType.WATERLOO_PENNER)
        tools_carry = tools_model.simulate(
            ToolsLaunch(
                ball_speed=BALL_SPEED_MPS,
                launch_angle=LAUNCH_ANGLE_RAD,
                spin_rate=SPIN_RATE_RPM,
            ),
            max_time=MAX_TIME_S,
            dt=TIME_STEP_S,
        ).carry_distance

        assert ud_carry > 0.0 and tools_carry > 0.0
        ratio = ud_carry / tools_carry
        assert 0.5 < ratio < 2.0, (
            "Waterloo/Penner carries from the two families are no longer the "
            f"same order: UD {ud_carry:.1f} m vs Tools {tools_carry:.1f} m "
            f"(ratio {ratio:.2f}). Magnitudes are not reconciled by design, "
            "but a factor beyond 2x points at a frame, sign, or unit defect "
            "rather than a calibration difference."
        )

    def test_families_are_labelled_apart_on_the_wire(self) -> None:
        """Same model name, different family — the ADR's whole point."""
        interchange = _vendor_interchange()
        ud_record = flight_result_to_trajectory_record(
            _simulate(fm.FlightModelType.WATERLOO_PENNER)
        )
        assert ud_record["provenance"]["model_family"] == UD_FLIGHT_FAMILY
        assert ud_record["provenance"]["model_family"] != (
            interchange.TOOLS_FLIGHT_FAMILY
        )
