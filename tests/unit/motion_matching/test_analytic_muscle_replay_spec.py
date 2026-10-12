from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

from src.shared.python.motion_matching.analytic_muscle_replay_spec import (
    MAX_JSON_BYTES,
    MAX_MUSCLES,
    MAX_STEPS,
    decode_analytic_replay_spec,
)


def _muscle(muscle_id: str) -> dict[str, Any]:
    return {
        "muscle_id": muscle_id,
        "geometry": {
            "reference_q_rad": 0.0,
            "reference_musculotendon_length_m": 0.32,
            "moment_arm_m_per_rad": 0.04,
        },
        "parameters": {
            "maximum_isometric_force_n": 1000.0,
            "optimal_fiber_length_m": 0.1,
            "tendon_slack_length_m": 0.2,
            "max_contraction_velocity_lopt_per_s": 10.0,
            "pennation_angle_rad": 0.2,
            "parallel_damping_n_s_per_m": 0.05,
            "force_length_width": 0.56,
        },
    }


def _record(*, muscles: int = 2, steps: int = 2) -> dict[str, Any]:
    muscle_rows = [_muscle(f"muscle_{index}") for index in range(muscles)]
    return {
        "schema": "synthetic-analytic-muscle-replay-input-v2",
        "profile_id": "rotational-1d-constant-moment-arm-no-contact-v1",
        "coordinate": {
            "coordinate_id": "elbow_flexion",
            "position_unit": "rad",
            "velocity_unit": "rad/s",
            "effort_unit": "N*m",
            "inertia_unit": "kg*m^2",
            "moment_arm_unit": "m/rad",
        },
        "clock": {
            "start_time_s": 0.0,
            "dt_s": 0.001,
            "step_count": steps,
            "command_hold": "zero_order_hold_left_endpoint",
            "command_times_s": [index * 0.001 for index in range(steps)],
        },
        "initial_state": {
            "q_rad": 0.0,
            "v_rad_s": 0.0,
            "activations": [0.05] * muscles,
            "previous_raw_excitations": [0.05] * muscles,
        },
        "activation_dynamics": {
            "tau_act_s": 0.01,
            "tau_deact_s": 0.04,
            "min_activation": 0.001,
            "update_rule": "explicit_euler_pinned_activation_dynamics",
        },
        "plant": {
            "inertia_kg_m2": 0.1,
            "external_torque_nm": 0.0,
            "contact": "none",
            "constraints": "none",
            "integration_rule": "semi_implicit_euler_v1",
        },
        "muscles": muscle_rows,
        "raw_excitation_commands": [[0.05] * muscles for _ in range(steps)],
        "raw_excitation_rate_limits_per_s": [20.0] * muscles,
        "producer_provenance": None,
    }


def _encode(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=True).encode("utf-8")


def _decode_record(value: dict[str, Any]) -> Any:
    return decode_analytic_replay_spec(_encode(value))


def test_valid_fixture_returns_frozen_nested_records_and_tuples() -> None:
    value = _record()
    value["plant"]["external_torque_nm"] = 0.25
    decoded = _decode_record(value)
    assert decoded.schema == "synthetic-analytic-muscle-replay-input-v2"
    assert decoded.clock.command_times_s == (0.0, 0.001)
    assert decoded.muscles[0].muscle_id == "muscle_0"
    assert decoded.raw_excitation_commands == ((0.05, 0.05), (0.05, 0.05))
    assert decoded.initial_state.previous_raw_excitations == (0.05, 0.05)
    assert decoded.activation_dynamics.min_activation == 0.001
    assert decoded.plant.external_torque_nm == 0.25
    assert is_dataclass(decoded) and is_dataclass(decoded.muscles[0])
    with pytest.raises(FrozenInstanceError):
        decoded.clock.dt_s = 1.0
    with pytest.raises(TypeError):
        decoded.raw_excitation_commands[0][0] = 1.0


def test_raw_zero_is_retained_separately_from_activation_floor() -> None:
    value = _record(steps=1)
    value["activation_dynamics"]["min_activation"] = 0.004
    value["initial_state"]["activations"] = [0.004, 0.004]
    value["initial_state"]["previous_raw_excitations"] = [0.0, 0.0]
    value["raw_excitation_commands"] = [[0.0, 0.0]]
    decoded = _decode_record(value)
    assert decoded.raw_excitation_commands == ((0.0, 0.0),)
    assert decoded.activation_dynamics.min_activation == 0.004
    assert all(
        a >= decoded.activation_dynamics.min_activation
        for a in decoded.initial_state.activations
    )
    assert decoded.initial_state.previous_raw_excitations == (0.0, 0.0)
    assert not hasattr(decoded, "effective_excitations")


def test_provenance_is_syntactic_only_and_ids_are_bounded_clean_text() -> None:
    value = _record(steps=1)
    value["producer_provenance"] = {
        "producer_kind": "synthetic-fixture",
        "source_revision": "a" * 40,
    }
    decoded = _decode_record(value)
    assert decoded.producer_provenance.source_revision == "a" * 40

    for bad_producer in ("", " padded", "producer\nkind", "x" * 129):
        value = _record(steps=1)
        value["producer_provenance"] = {
            "producer_kind": bad_producer,
            "source_revision": None,
        }
        with pytest.raises(ValueError, match="producer_kind"):
            _decode_record(value)

    for bad in ("", " padded", "padded ", "bad\nname", "x" * 129, "é" * 65):
        value = _record(steps=1)
        value["coordinate"]["coordinate_id"] = bad
        with pytest.raises(ValueError, match="coordinate_id"):
            _decode_record(value)

    boundary = _record(steps=1)
    boundary["coordinate"]["coordinate_id"] = "é" * 64
    assert _decode_record(boundary).coordinate.coordinate_id == "é" * 64
    over_boundary = _record(steps=1)
    over_boundary["coordinate"]["coordinate_id"] = "é" * 65
    with pytest.raises(ValueError, match="coordinate_id"):
        _decode_record(over_boundary)

    for bad_revision in ("A" * 40, "a" * 39, "../" + "a" * 37):
        value = _record(steps=1)
        value["producer_provenance"] = {
            "producer_kind": "fixture",
            "source_revision": bad_revision,
        }
        with pytest.raises(ValueError, match="source_revision"):
            _decode_record(value)


def test_unknown_or_missing_keys_fail_at_nested_objects() -> None:
    value = _record(steps=1)
    value["surprise"] = 1
    with pytest.raises(ValueError, match="exact keys"):
        _decode_record(value)

    value = _record(steps=1)
    del value["plant"]["contact"]
    with pytest.raises(ValueError, match="exact keys"):
        _decode_record(value)


def test_duplicate_keys_bom_invalid_utf8_and_nonfinite_constants_reject() -> None:
    duplicate = b'{"schema":"one","schema":"two"}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        decode_analytic_replay_spec(duplicate)
    with pytest.raises(ValueError, match="BOM"):
        decode_analytic_replay_spec(b"\xef\xbb\xbf{}")
    with pytest.raises(ValueError, match="UTF-8"):
        decode_analytic_replay_spec(b"\xff")
    with pytest.raises(ValueError, match="nonfinite"):
        decode_analytic_replay_spec(b'{"x":NaN}')


def test_byte_cap_is_checked_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.shared.python.motion_matching.analytic_muscle_replay_spec as module

    def forbidden_parse(*_args: object, **_kwargs: object) -> object:
        pytest.fail("oversize payload reached JSON parser")

    monkeypatch.setattr(module.json, "loads", forbidden_parse)
    with pytest.raises(ValueError, match="byte limit"):
        decode_analytic_replay_spec(b" " * (MAX_JSON_BYTES + 1))


def test_json_byte_cap_accepts_exact_limit() -> None:
    payload = _encode(_record(steps=1))
    padded = payload + b" " * (MAX_JSON_BYTES - len(payload))
    assert len(padded) == MAX_JSON_BYTES
    assert decode_analytic_replay_spec(padded).clock.step_count == 1


def test_muscle_and_step_caps_accept_limit_and_reject_one_over() -> None:
    max_muscles = _record(muscles=MAX_MUSCLES, steps=1)
    assert len(_decode_record(max_muscles).muscles) == MAX_MUSCLES
    with pytest.raises(ValueError, match="muscle limit"):
        _decode_record(_record(muscles=MAX_MUSCLES + 1, steps=1))

    max_steps = _record(muscles=1, steps=MAX_STEPS)
    assert len(_decode_record(max_steps).raw_excitation_commands) == MAX_STEPS
    with pytest.raises(ValueError, match="step_count"):
        _decode_record(_record(muscles=1, steps=MAX_STEPS + 1))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("initial_state", "q_rad"), True),
        (("initial_state", "q_rad"), float("inf")),
        (("activation_dynamics", "tau_act_s"), 0.0),
        (("activation_dynamics", "min_activation"), 1.0),
        (("plant", "inertia_kg_m2"), 0.0),
        (("plant", "external_torque_nm"), float("nan")),
        (("muscles", 0, "parameters", "maximum_isometric_force_n"), -1.0),
        (("muscles", 0, "parameters", "pennation_angle_rad"), math.pi / 2),
        (("muscles", 0, "parameters", "parallel_damping_n_s_per_m"), -0.1),
    ],
)
def test_invalid_types_and_physical_parameter_domains_reject(
    path: tuple[str | int, ...], replacement: object
) -> None:
    value = _record(steps=1)
    target: Any = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError):
        _decode_record(value)


def test_profile_enums_shapes_and_duplicate_muscle_ids_reject() -> None:
    value = _record(steps=1)
    value["plant"]["contact"] = "enabled"
    with pytest.raises(ValueError, match="contact"):
        _decode_record(value)

    value = _record(steps=1)
    value["muscles"][1]["muscle_id"] = value["muscles"][0]["muscle_id"]
    with pytest.raises(ValueError, match="unique"):
        _decode_record(value)

    value = _record(steps=1)
    value["raw_excitation_commands"][0].pop()
    with pytest.raises(ValueError, match="shape"):
        _decode_record(value)

    value = _record(steps=1)
    value["muscles"][0]["geometry"]["unexpected"] = 2.0
    with pytest.raises(ValueError, match="exact keys"):
        _decode_record(value)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("clock", "step_count"), True),
        (("clock", "step_count"), 1.0),
        (("raw_excitation_rate_limits_per_s", 0), -0.1),
        (("initial_state", "activations", 0), -0.01),
        (("initial_state", "activations", 0), 1.01),
        (("initial_state", "previous_raw_excitations", 0), -0.01),
        (("initial_state", "previous_raw_excitations", 0), 1.01),
        (("raw_excitation_commands", 0, 0), -0.01),
        (("raw_excitation_commands", 0, 0), 1.01),
    ],
)
def test_command_state_rate_and_step_count_bounds(
    path: tuple[str | int, ...], replacement: object
) -> None:
    value = _record(steps=1)
    target: Any = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError):
        _decode_record(value)


def test_clock_requires_finite_strictly_advancing_grid_and_endpoint() -> None:
    value = _record(steps=1)
    value["clock"]["dt_s"] = 0.0
    with pytest.raises(ValueError, match="dt_s"):
        _decode_record(value)

    value = _record(steps=1)
    value["clock"]["start_time_s"] = 1e16
    value["clock"]["dt_s"] = 0.1
    value["clock"]["command_times_s"] = [1e16]
    with pytest.raises(ValueError, match="terminal|advance"):
        _decode_record(value)

    value = _record(steps=2)
    value["clock"]["start_time_s"] = 1e12
    value["clock"]["dt_s"] = 0.01
    value["clock"]["command_times_s"] = [1e12, 1e12 + 0.0101]
    with pytest.raises(ValueError, match="grid|clock"):
        _decode_record(value)

    value = _record(steps=2)
    value["clock"]["command_times_s"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="increasing"):
        _decode_record(value)


def test_slew_uses_exact_inclusive_clipped_bounds_without_slack() -> None:
    value = _record(steps=1)
    value["initial_state"]["previous_raw_excitations"] = [0.05, 0.05]
    value["clock"]["dt_s"] = 0.1
    value["clock"]["command_times_s"] = [0.0]
    value["raw_excitation_rate_limits_per_s"] = [0.1, 0.1]
    upper = min(1.0, 0.05 + 0.1 * 0.1)
    value["raw_excitation_commands"] = [[upper, 0.05]]
    assert _decode_record(value).raw_excitation_commands[0][0] == upper

    value["raw_excitation_commands"][0][0] = math.nextafter(upper, math.inf)
    with pytest.raises(ValueError, match="slew"):
        _decode_record(value)

    value["raw_excitation_rate_limits_per_s"] = [0.0, 0.0]
    value["raw_excitation_commands"] = [[math.nextafter(0.05, math.inf), 0.05]]
    with pytest.raises(ValueError, match="slew"):
        _decode_record(value)

    later = _record(steps=2)
    later["clock"]["dt_s"] = 0.1
    later["clock"]["command_times_s"] = [0.0, 0.1]
    later["raw_excitation_rate_limits_per_s"] = [0.1, 0.1]
    later_upper = min(1.0, 0.05 + 0.1 * 0.1)
    later["raw_excitation_commands"] = [
        [0.05, 0.05],
        [math.nextafter(later_upper, math.inf), 0.05],
    ]
    with pytest.raises(ValueError, match="slew"):
        _decode_record(later)


def test_derived_initial_musculotendon_length_must_be_finite_and_positive() -> None:
    value = _record(steps=1)
    value["initial_state"]["q_rad"] = 1e308
    value["muscles"][0]["geometry"]["reference_q_rad"] = -1e308
    value["muscles"][0]["geometry"]["moment_arm_m_per_rad"] = 1.0
    with pytest.raises(ValueError, match="musculotendon length"):
        _decode_record(value)

    value = _record(steps=1)
    value["muscles"][0]["geometry"]["reference_musculotendon_length_m"] = 0.01
    value["initial_state"]["q_rad"] = 1.0
    with pytest.raises(ValueError, match="musculotendon length"):
        _decode_record(value)


def test_recursive_json_huge_integer_and_lone_surrogate_reject() -> None:
    recursive = b"[" * 20000 + b"0" + b"]" * 20000
    with pytest.raises(ValueError):
        decode_analytic_replay_spec(recursive)

    huge_integer = "1" + "0" * 400
    raw = _encode(_record(steps=1)).replace(
        b'"q_rad":0.0', f'"q_rad":{huge_integer}'.encode()
    )
    with pytest.raises(ValueError, match="q_rad"):
        decode_analytic_replay_spec(raw)

    lone_surrogate = _encode(_record(steps=1)).replace(
        b'"coordinate_id":"elbow_flexion"', b'"coordinate_id":"\\ud800"'
    )
    with pytest.raises(ValueError, match="UTF-8"):
        decode_analytic_replay_spec(lone_surrogate)


def test_fresh_import_does_not_load_numeric_gui_or_biomechanics_modules() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    payload = _encode(_record(steps=1))
    script = (
        "import sys; "
        "forbidden=('numpy', 'PyQt6', 'biomechanics', 'same_input', 'muscle_allocation', "
        "'controller', 'controllers', 'distributed_feedback', 'opensim', 'OpenSim', 'casadi'); "
        "find_forbidden=lambda: [name for name in sys.modules "
        "if any(part in forbidden for part in name.split('.'))]; "
        "assert not find_forbidden(), find_forbidden(); "
        "from src.shared.python.motion_matching.analytic_muscle_replay_spec import "
        "decode_analytic_replay_spec; "
        f"decode_analytic_replay_spec({payload!r}); "
        "assert not find_forbidden(), find_forbidden()"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
