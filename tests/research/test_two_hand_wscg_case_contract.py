"""Contract tests for the extracted WSCG per-case helpers (#13852)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_two_hand_wscg_analysis import (
    CASE_NAMES,
    OUT_OF_PLANE_AXIS,
    _analyze_case,
    _load_case,
    _vector,
)
from scripts.research.proximal_distal_energy.run_two_hand_wscg_case import (
    _case_grip_wrench_signals,
    _case_metrics_record,
    _case_power_channels,
    _require_case_analysis_inputs,
)
from src.shared.python.contracts import PreconditionError
from tests.research.wscg_analysis_test_support import JSON_PATH

pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def base() -> np.ndarray:
    return _load_case("base")


@pytest.fixture(scope="module")
def geometry(base: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            _vector(base, "lead_position", "m"),
            _vector(base, "trail_position", "m"),
            _vector(base, "midpoint_position", "m"),
        )
    )


def test_inputs_contract_accepts_committed_case(base: np.ndarray) -> None:
    _require_case_analysis_inputs("base", base)


def test_inputs_contract_rejects_unknown_case(base: np.ndarray) -> None:
    with pytest.raises(ValueError, match="unknown case"):
        _require_case_analysis_inputs("mystery", base)


def test_inputs_contract_rejects_nonfinite_time(base: np.ndarray) -> None:
    broken = base.copy()
    broken["time_s"][3] = np.nan
    with pytest.raises(PreconditionError, match="time_s"):
        _require_case_analysis_inputs("base", broken)


def test_inputs_contract_rejects_nonfinite_power_column(base: np.ndarray) -> None:
    broken = base.copy()
    broken["lead_linear_power_w"][0] = np.inf
    with pytest.raises(PreconditionError, match="lead_linear_power_w"):
        _require_case_analysis_inputs("base", broken)


def test_grip_signals_postconditions(base: np.ndarray, geometry: np.ndarray) -> None:
    signals = _case_grip_wrench_signals(base, geometry)
    count = base["time_s"].size

    assert signals["grip_separation"].shape == (count,)
    assert np.all(signals["grip_separation"] > 0.0)
    assert signals["resultant"].shape == (count, 3)
    np.testing.assert_allclose(
        signals["resultant"], signals["lead_force"] + signals["trail_force"]
    )
    assert signals["maximum_in_plane_force"] > 0.0
    lead_out = signals["lead_force"] @ OUT_OF_PLANE_AXIS
    trail_out = signals["trail_force"] @ OUT_OF_PLANE_AXIS
    assert signals["max_out_of_plane_force_n"] == pytest.approx(
        max(np.max(np.abs(lead_out)), np.max(np.abs(trail_out)))
    )


def test_grip_signals_match_analyze_case_outputs(
    base: np.ndarray, geometry: np.ndarray
) -> None:
    signals = _case_grip_wrench_signals(base, geometry)
    metrics, arrays = _analyze_case("base", base, geometry)

    for key in ("lead_axial", "trail_normal", "force_moment", "equivalent_couple"):
        np.testing.assert_array_equal(arrays[key], signals[key])
    assert metrics["maximum_in_plane_hand_force_n"] == signals["maximum_in_plane_force"]
    assert (
        metrics["maximum_out_of_plane_hand_force_n"]
        == signals["max_out_of_plane_force_n"]
    )


def test_power_channels_postconditions(base: np.ndarray, geometry: np.ndarray) -> None:
    grip = _case_grip_wrench_signals(base, geometry)
    power = _case_power_channels(base, grip)
    count = base["time_s"].size

    assert all(values.shape == (count,) for values in power.values())
    np.testing.assert_allclose(
        power["decomposed_force_power"],
        power["rigid_force_power"] + power["deformation_power"],
    )
    np.testing.assert_allclose(
        power["contact_power"],
        power["source_linear_power"] + power["source_angular_power"],
    )
    np.testing.assert_allclose(
        power["contact_force_power"], power["decomposed_force_power"], atol=1e-6
    )


def test_power_channels_reject_incomplete_grip_signals(
    base: np.ndarray, geometry: np.ndarray
) -> None:
    grip = dict(_case_grip_wrench_signals(base, geometry))
    del grip["grip_vector"]
    with pytest.raises(PreconditionError, match="grip signals"):
        _case_power_channels(base, grip)


def test_metrics_record_rejects_missing_power_channels(
    base: np.ndarray, geometry: np.ndarray
) -> None:
    grip = _case_grip_wrench_signals(base, geometry)
    with pytest.raises(PreconditionError, match="power channels"):
        _case_metrics_record(base, grip, {})


@pytest.mark.parametrize("name", CASE_NAMES)
def test_metrics_record_equals_committed_case_output(
    name: str, geometry: np.ndarray
) -> None:
    """The extracted record reproduces the previously published metrics exactly."""
    data = _load_case(name)
    grip = _case_grip_wrench_signals(data, geometry)
    metrics = _case_metrics_record(data, grip, _case_power_channels(data, grip))
    committed = json.loads(JSON_PATH.read_text(encoding="utf-8"))["cases"][name]

    assert json.loads(json.dumps(metrics)) == committed
    assert list(metrics) == list(committed)
    assert _analyze_case(name, data, geometry)[0] == metrics
