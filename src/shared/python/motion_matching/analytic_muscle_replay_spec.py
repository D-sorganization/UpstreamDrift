"""Strict immutable inputs for a bounded synthetic muscle replay fixture.

This module only decodes physical inputs. It does not run activation or force
calculations and does not admit an execution implementation or runtime.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, NoReturn

SCHEMA = "synthetic-analytic-muscle-replay-input-v2"
PROFILE_ID = "rotational-1d-constant-moment-arm-no-contact-v1"
MAX_JSON_BYTES = 65_536
MAX_MUSCLES = 8
MAX_STEPS = 128
MAX_IDENTIFIER_BYTES = 128
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "profile_id",
        "coordinate",
        "clock",
        "initial_state",
        "activation_dynamics",
        "plant",
        "muscles",
        "raw_excitation_commands",
        "raw_excitation_rate_limits_per_s",
        "producer_provenance",
    }
)


@dataclass(frozen=True)
class CoordinateSpec:
    coordinate_id: str
    position_unit: str
    velocity_unit: str
    effort_unit: str
    inertia_unit: str
    moment_arm_unit: str


@dataclass(frozen=True)
class ClockSpec:
    start_time_s: float
    dt_s: float
    step_count: int
    command_hold: str
    command_times_s: tuple[float, ...]


@dataclass(frozen=True)
class InitialState:
    q_rad: float
    v_rad_s: float
    activations: tuple[float, ...]
    previous_raw_excitations: tuple[float, ...]


@dataclass(frozen=True)
class ActivationParameters:
    tau_act_s: float
    tau_deact_s: float
    min_activation: float
    update_rule: str


@dataclass(frozen=True)
class PlantSpec:
    inertia_kg_m2: float
    external_torque_nm: float
    contact: str
    constraints: str
    integration_rule: str


@dataclass(frozen=True)
class MuscleGeometry:
    reference_q_rad: float
    reference_musculotendon_length_m: float
    moment_arm_m_per_rad: float


@dataclass(frozen=True)
class MuscleParameters:
    maximum_isometric_force_n: float
    optimal_fiber_length_m: float
    tendon_slack_length_m: float
    max_contraction_velocity_lopt_per_s: float
    pennation_angle_rad: float
    parallel_damping_n_s_per_m: float
    force_length_width: float


@dataclass(frozen=True)
class MuscleSpec:
    muscle_id: str
    geometry: MuscleGeometry
    parameters: MuscleParameters


@dataclass(frozen=True)
class ProducerProvenance:
    producer_kind: str
    source_revision: str | None


@dataclass(frozen=True)
class AnalyticReplaySpec:
    schema: str
    profile_id: str
    coordinate: CoordinateSpec
    clock: ClockSpec
    initial_state: InitialState
    activation_dynamics: ActivationParameters
    plant: PlantSpec
    muscles: tuple[MuscleSpec, ...]
    raw_excitation_commands: tuple[tuple[float, ...], ...]
    raw_excitation_rate_limits_per_s: tuple[float, ...]
    producer_provenance: ProducerProvenance | None


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"nonfinite JSON constant is forbidden: {value}")


def _object(value: Any, keys: frozenset[str], where: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{where} must be a JSON object")
    if value.keys() != keys:
        missing = sorted(keys - value.keys())
        extra = sorted(value.keys() - keys)
        _fail(f"{where} must have exact keys (missing={missing}, extra={extra})")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{where} must be nonempty text")
    if value != value.strip():
        _fail(f"{where} must not have surrounding whitespace")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{where} must be valid UTF-8 text") from exc
    if len(encoded) > MAX_IDENTIFIER_BYTES:
        _fail(f"{where} exceeds the {MAX_IDENTIFIER_BYTES}-byte identifier limit")
    if any(unicodedata.category(char) == "Cc" for char in value):
        _fail(f"{where} must not contain control characters")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{where} must be a finite number") from exc
    if not math.isfinite(result):
        _fail(f"{where} must be a finite number")
    return result


def _positive(value: Any, where: str) -> float:
    result = _number(value, where)
    if result <= 0.0:
        _fail(f"{where} must be positive")
    return result


def _vector(value: Any, size: int, where: str) -> tuple[float, ...]:
    if type(value) is not list or len(value) != size:
        _fail(f"{where} has an invalid shape; expected {size} values")
    return tuple(_number(item, f"{where}[{index}]") for index, item in enumerate(value))


def _enum(value: Any, expected: str, where: str) -> str:
    if value != expected or not isinstance(value, str):
        _fail(f"{where} must be {expected!r}")
    return expected


def _parse_coordinate(value: Any) -> CoordinateSpec:
    keys = frozenset(
        {
            "coordinate_id",
            "position_unit",
            "velocity_unit",
            "effort_unit",
            "inertia_unit",
            "moment_arm_unit",
        }
    )
    row = _object(value, keys, "coordinate")
    units = {
        "position_unit": "rad",
        "velocity_unit": "rad/s",
        "effort_unit": "N*m",
        "inertia_unit": "kg*m^2",
        "moment_arm_unit": "m/rad",
    }
    for field, expected in units.items():
        _enum(row[field], expected, f"coordinate.{field}")
    return CoordinateSpec(
        coordinate_id=_text(row["coordinate_id"], "coordinate.coordinate_id"),
        position_unit="rad",
        velocity_unit="rad/s",
        effort_unit="N*m",
        inertia_unit="kg*m^2",
        moment_arm_unit="m/rad",
    )


def _parse_clock(value: Any) -> ClockSpec:
    row = _object(
        value,
        frozenset(
            {"start_time_s", "dt_s", "step_count", "command_hold", "command_times_s"}
        ),
        "clock",
    )
    start = _number(row["start_time_s"], "clock.start_time_s")
    dt = _positive(row["dt_s"], "clock.dt_s")
    count = row["step_count"]
    if type(count) is not int or not 1 <= count <= MAX_STEPS:
        _fail(f"clock.step_count must be an integer in [1, {MAX_STEPS}]")
    _enum(
        row["command_hold"],
        "zero_order_hold_left_endpoint",
        "clock.command_hold",
    )
    raw_times = row["command_times_s"]
    if type(raw_times) is not list or len(raw_times) != count:
        _fail("clock.command_times_s has an invalid shape")
    times = tuple(
        _number(item, f"clock.command_times_s[{index}]")
        for index, item in enumerate(raw_times)
    )
    previous_time: float | None = None
    last_expected_time: float | None = None
    tolerance = min(1e-12, dt * 1e-6)
    for index, actual in enumerate(times):
        if previous_time is not None and actual <= previous_time:
            _fail("clock.command_times_s must be strictly increasing")
        try:
            offset = index * dt
            expected = start + offset
        except OverflowError as exc:
            raise ValueError("clock grid is not representable") from exc
        if not math.isfinite(offset) or not math.isfinite(expected):
            _fail("clock grid contains a nonfinite expected time")
        if index and expected <= start + (index - 1) * dt:
            _fail("clock grid does not advance representably")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            _fail(f"clock.command_times_s[{index}] is off the expected grid")
        previous_time = actual
        last_expected_time = expected

    terminal_offset = count * dt
    terminal = start + terminal_offset
    if (
        not math.isfinite(terminal_offset)
        or not math.isfinite(terminal)
        or last_expected_time is None
        or terminal <= last_expected_time
        or terminal <= times[-1]
    ):
        _fail("clock terminal time is nonfinite or does not advance representably")
    return ClockSpec(start, dt, count, "zero_order_hold_left_endpoint", times)


def _parse_activation(value: Any) -> ActivationParameters:
    row = _object(
        value,
        frozenset({"tau_act_s", "tau_deact_s", "min_activation", "update_rule"}),
        "activation_dynamics",
    )
    minimum = _number(row["min_activation"], "activation_dynamics.min_activation")
    if not 0.0 < minimum < 1.0:
        _fail("activation_dynamics.min_activation must be in (0, 1)")
    return ActivationParameters(
        tau_act_s=_positive(row["tau_act_s"], "activation_dynamics.tau_act_s"),
        tau_deact_s=_positive(row["tau_deact_s"], "activation_dynamics.tau_deact_s"),
        min_activation=minimum,
        update_rule=_enum(
            row["update_rule"],
            "explicit_euler_pinned_activation_dynamics",
            "activation_dynamics.update_rule",
        ),
    )


def _parse_plant(value: Any) -> PlantSpec:
    row = _object(
        value,
        frozenset(
            {
                "inertia_kg_m2",
                "external_torque_nm",
                "contact",
                "constraints",
                "integration_rule",
            }
        ),
        "plant",
    )
    return PlantSpec(
        inertia_kg_m2=_positive(row["inertia_kg_m2"], "plant.inertia_kg_m2"),
        external_torque_nm=_number(
            row["external_torque_nm"], "plant.external_torque_nm"
        ),
        contact=_enum(row["contact"], "none", "plant.contact"),
        constraints=_enum(row["constraints"], "none", "plant.constraints"),
        integration_rule=_enum(
            row["integration_rule"], "semi_implicit_euler_v1", "plant.integration_rule"
        ),
    )


def _parse_muscle(value: Any, index: int) -> MuscleSpec:
    where = f"muscles[{index}]"
    row = _object(value, frozenset({"muscle_id", "geometry", "parameters"}), where)
    muscle_id = _text(row["muscle_id"], f"{where}.muscle_id")
    geometry_row = _object(
        row["geometry"],
        frozenset(
            {
                "reference_q_rad",
                "reference_musculotendon_length_m",
                "moment_arm_m_per_rad",
            }
        ),
        f"{where}.geometry",
    )
    geometry = MuscleGeometry(
        reference_q_rad=_number(
            geometry_row["reference_q_rad"], f"{where}.geometry.reference_q_rad"
        ),
        reference_musculotendon_length_m=_positive(
            geometry_row["reference_musculotendon_length_m"],
            f"{where}.geometry.reference_musculotendon_length_m",
        ),
        moment_arm_m_per_rad=_number(
            geometry_row["moment_arm_m_per_rad"],
            f"{where}.geometry.moment_arm_m_per_rad",
        ),
    )
    parameter_row = _object(
        row["parameters"],
        frozenset(
            {
                "maximum_isometric_force_n",
                "optimal_fiber_length_m",
                "tendon_slack_length_m",
                "max_contraction_velocity_lopt_per_s",
                "pennation_angle_rad",
                "parallel_damping_n_s_per_m",
                "force_length_width",
            }
        ),
        f"{where}.parameters",
    )
    pennation = _number(
        parameter_row["pennation_angle_rad"], f"{where}.parameters.pennation_angle_rad"
    )
    damping = _number(
        parameter_row["parallel_damping_n_s_per_m"],
        f"{where}.parameters.parallel_damping_n_s_per_m",
    )
    if not 0.0 <= pennation < math.pi / 2.0:
        _fail(f"{where}.parameters.pennation_angle_rad must be in [0, pi/2)")
    if damping < 0.0:
        _fail(f"{where}.parameters.parallel_damping_n_s_per_m must be nonnegative")
    parameters = MuscleParameters(
        maximum_isometric_force_n=_positive(
            parameter_row["maximum_isometric_force_n"],
            f"{where}.parameters.maximum_isometric_force_n",
        ),
        optimal_fiber_length_m=_positive(
            parameter_row["optimal_fiber_length_m"],
            f"{where}.parameters.optimal_fiber_length_m",
        ),
        tendon_slack_length_m=_positive(
            parameter_row["tendon_slack_length_m"],
            f"{where}.parameters.tendon_slack_length_m",
        ),
        max_contraction_velocity_lopt_per_s=_positive(
            parameter_row["max_contraction_velocity_lopt_per_s"],
            f"{where}.parameters.max_contraction_velocity_lopt_per_s",
        ),
        pennation_angle_rad=pennation,
        parallel_damping_n_s_per_m=damping,
        force_length_width=_positive(
            parameter_row["force_length_width"],
            f"{where}.parameters.force_length_width",
        ),
    )
    return MuscleSpec(muscle_id, geometry, parameters)


def _parse_provenance(value: Any) -> ProducerProvenance | None:
    if value is None:
        return None
    row = _object(
        value,
        frozenset({"producer_kind", "source_revision"}),
        "producer_provenance",
    )
    revision = row["source_revision"]
    if revision is not None and (
        not isinstance(revision, str) or _REVISION_PATTERN.fullmatch(revision) is None
    ):
        _fail(
            "producer_provenance.source_revision must be null or 40 lowercase hex chars"
        )
    return ProducerProvenance(
        producer_kind=_text(row["producer_kind"], "producer_provenance.producer_kind"),
        source_revision=revision,
    )


def _parse_initial_state(
    value: Any,
    muscles: tuple[MuscleSpec, ...],
    activation: ActivationParameters,
) -> InitialState:
    initial_row = _object(
        value,
        frozenset({"q_rad", "v_rad_s", "activations", "previous_raw_excitations"}),
        "initial_state",
    )
    initial_activations = _vector(
        initial_row["activations"], len(muscles), "initial_state.activations"
    )
    previous_raw = _vector(
        initial_row["previous_raw_excitations"],
        len(muscles),
        "initial_state.previous_raw_excitations",
    )
    for index, value in enumerate(initial_activations):
        if not activation.min_activation <= value <= 1.0:
            _fail(f"initial_state.activations[{index}] is outside [min_activation, 1]")
    for index, value in enumerate(previous_raw):
        if not 0.0 <= value <= 1.0:
            _fail(f"initial_state.previous_raw_excitations[{index}] is outside [0, 1]")
    initial = InitialState(
        q_rad=_number(initial_row["q_rad"], "initial_state.q_rad"),
        v_rad_s=_number(initial_row["v_rad_s"], "initial_state.v_rad_s"),
        activations=initial_activations,
        previous_raw_excitations=previous_raw,
    )
    for muscle in muscles:
        geometry = muscle.geometry
        length = (
            geometry.reference_musculotendon_length_m
            - geometry.moment_arm_m_per_rad * (initial.q_rad - geometry.reference_q_rad)
        )
        if not math.isfinite(length) or length <= 0.0:
            _fail(
                f"initial musculotendon length for {muscle.muscle_id!r} must be finite and positive"
            )
    return initial


def _parse_raw_excitation_schedule(
    commands_value: Any,
    rate_limits_value: Any,
    step_count: int,
    muscles: tuple[MuscleSpec, ...],
    previous_raw: tuple[float, ...],
    dt_s: float,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    rate_limits = _vector(
        rate_limits_value,
        len(muscles),
        "raw_excitation_rate_limits_per_s",
    )
    if any(rate < 0.0 for rate in rate_limits):
        _fail("raw_excitation_rate_limits_per_s values must be nonnegative")
    if type(commands_value) is not list or len(commands_value) != step_count:
        _fail("raw_excitation_commands has an invalid step shape")
    commands_list: list[tuple[float, ...]] = []
    previous = previous_raw
    for row_index, command_row in enumerate(commands_value):
        command = _vector(
            command_row,
            len(muscles),
            f"raw_excitation_commands[{row_index}]",
        )
        for muscle_index, value in enumerate(command):
            if not 0.0 <= value <= 1.0:
                _fail(f"raw_excitation_commands[{row_index}] must lie in [0, 1]")
            delta = rate_limits[muscle_index] * dt_s
            if not math.isfinite(delta):
                _fail(f"slew bounds overflow for muscle index {muscle_index}")
            lower = max(0.0, previous[muscle_index] - delta)
            upper = min(1.0, previous[muscle_index] + delta)
            if not math.isfinite(lower) or not math.isfinite(upper):
                _fail(f"slew bounds are nonfinite for muscle index {muscle_index}")
            if value < lower or value > upper:
                _fail(
                    f"raw_excitation_commands[{row_index}][{muscle_index}] violates slew bounds"
                )
        commands_list.append(command)
        previous = command
    return tuple(commands_list), rate_limits


def decode_analytic_replay_spec(raw_json: bytes) -> AnalyticReplaySpec:
    """Decode bounded JSON into immutable inputs for the synthetic profile only.

    Producer revision text is syntax-checked but never resolved. This function
    performs no model, activation, muscle-force, or replay calculations.
    """
    if not isinstance(raw_json, bytes):
        _fail("raw_json must be bytes")
    if len(raw_json) > MAX_JSON_BYTES:
        _fail(f"JSON payload exceeds byte limit of {MAX_JSON_BYTES}")
    if raw_json.startswith(b"\xef\xbb\xbf"):
        _fail("UTF-8 BOM is not permitted")
    try:
        text = raw_json.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("JSON payload must be valid UTF-8") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid JSON payload") from exc
    row = _object(payload, _TOP_LEVEL_KEYS, "root")
    if row["schema"] != SCHEMA:
        _fail(f"schema must be {SCHEMA!r}")
    if row["profile_id"] != PROFILE_ID:
        _fail(f"profile_id must be {PROFILE_ID!r}")

    clock = _parse_clock(row["clock"])
    muscles_value = row["muscles"]
    if type(muscles_value) is not list or not 1 <= len(muscles_value) <= MAX_MUSCLES:
        _fail(f"muscle limit requires between 1 and {MAX_MUSCLES} muscles")
    muscles = tuple(
        _parse_muscle(item, index) for index, item in enumerate(muscles_value)
    )
    muscle_ids = tuple(item.muscle_id for item in muscles)
    if len(set(muscle_ids)) != len(muscle_ids):
        _fail("muscle IDs must be unique")

    activation = _parse_activation(row["activation_dynamics"])
    initial = _parse_initial_state(row["initial_state"], muscles, activation)
    commands, rate_limits = _parse_raw_excitation_schedule(
        row["raw_excitation_commands"],
        row["raw_excitation_rate_limits_per_s"],
        clock.step_count,
        muscles,
        initial.previous_raw_excitations,
        clock.dt_s,
    )

    return AnalyticReplaySpec(
        schema=SCHEMA,
        profile_id=PROFILE_ID,
        coordinate=_parse_coordinate(row["coordinate"]),
        clock=clock,
        initial_state=initial,
        activation_dynamics=activation,
        plant=_parse_plant(row["plant"]),
        muscles=muscles,
        raw_excitation_commands=commands,
        raw_excitation_rate_limits_per_s=rate_limits,
        producer_provenance=_parse_provenance(row["producer_provenance"]),
    )
