"""Tests for the frozen MOSAIC benchmark protocol and tolerance change policy."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.estimation.mosaic.benchmark_protocol import (
    BenchmarkProtocol,
    PROTOCOL_CONFIG_PATH,
    load_benchmark_protocol,
    verify_protocol_integrity,
)

pytestmark = pytest.mark.unit


def test_frozen_protocol_loads_and_has_expected_keys() -> None:
    protocol = load_benchmark_protocol()
    assert isinstance(protocol, BenchmarkProtocol)
    assert protocol.fixture.n_trials == 2
    assert protocol.fixture.n_nodes == 150
    assert protocol.fixture.sample_rate_hz == 200.0
    assert protocol.fixture.marker_noise_m == 5e-4
    assert protocol.fixture.data_seed == 8
    assert protocol.fixture.prior_seed == 2
    assert protocol.fixture.geometry_scale == 1.05

    # Frozen tolerances
    assert protocol.tolerances.geometry_error_max_m == 1e-4  # < 0.1 mm
    assert protocol.tolerances.torque_rms_ratio_max <= 0.25  # 20 % nominal, 25 % max
    assert protocol.tolerances.structural_rank == 9
    assert protocol.tolerances.unobservable_parameters == ("m_0",)
    assert protocol.tolerances.open_loop_replay_rms_max_rad <= 0.035
    assert protocol.tolerances.open_loop_horizon_s == 0.75


def test_protocol_integrity_fails_if_modified_without_trailer(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = load_benchmark_protocol()
    # Verification succeeds on untouched protocol
    assert verify_protocol_integrity(protocol) is True

    # Modified protocol without trailer fails
    tampered_dict = protocol.to_dict()
    tampered_dict["tolerances"]["geometry_error_max_m"] = 0.005  # widened!
    tampered = BenchmarkProtocol.from_dict(tampered_dict)
    with pytest.raises(AssertionError, match="Tolerance-Change-Evidence"):
        verify_protocol_integrity(tampered)
