"""Canonical-v2 remap tests for :class:`DrakeAdapter` (CC-28)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.shared.python.engine_core.capabilities import CapabilityLevel
from src.shared.python.pose_interchange.adapters.drake import (
    HYDROELASTIC_CONTACT_DIVERGENCE,
    DrakeAdapter,
    DrakeNamedState,
    DrakeNativeState,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]

pytestmark = pytest.mark.unit


def _canonical_state() -> dict[str, np.ndarray | float]:
    half = np.sqrt(0.5)
    return {
        "q": np.array([1.0, 2.0, 3.0, half, 0.0, 0.0, half, 0.25, -0.5]),
        "v": np.array([10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.5, -0.75]),
        "a": np.array([0.1, 0.2, 0.3, 0.0, 2.0, 0.0, 0.05, -0.07]),
        "t": 1.25,
    }


def test_drake_canonical_v2_remaps_quaternion_floating_state() -> None:
    adapter = DrakeAdapter()

    native = adapter.from_canonical_state(_canonical_state())

    assert isinstance(native, DrakeNativeState)
    half = np.sqrt(0.5)
    np.testing.assert_allclose(
        native.q,
        [half, 0.0, 0.0, half, 1.0, 2.0, 3.0, 0.25, -0.5],
        atol=1e-12,
    )
    np.testing.assert_allclose(native.v[:3], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(native.v[3:6], [10.0, 20.0, 30.0], atol=1e-12)
    np.testing.assert_allclose(native.a[:3], [-2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(native.a[3:6], [0.1, 0.2, 0.3], atol=1e-12)


def test_drake_canonical_v2_round_trip_identity() -> None:
    adapter = DrakeAdapter()
    state = _canonical_state()

    native = adapter.from_canonical_state(state)
    recovered = adapter.to_canonical_state(native)

    np.testing.assert_allclose(recovered["q"], state["q"], atol=1e-12)
    np.testing.assert_allclose(recovered["v"], state["v"], atol=1e-12)
    np.testing.assert_allclose(recovered["a"], state["a"], atol=1e-12)
    assert recovered["t"] == pytest.approx(state["t"])


def test_drake_cc7_named_state_mapping_is_lossless() -> None:
    adapter = DrakeAdapter()
    state = {
        "pelvis_xyz": np.array([0.1, 0.2, 0.3]),
        "pelvis_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "lead_wrist": np.array([0.25]),
    }

    native = adapter.from_canonical(state)
    recovered = adapter.to_canonical(native)

    assert isinstance(native, DrakeNamedState)
    assert set(recovered) == set(state)
    for key, value in state.items():
        np.testing.assert_allclose(recovered[key], value, atol=0.0)


def test_drake_adapter_declares_cc28_capabilities() -> None:
    adapter = DrakeAdapter()
    report = adapter.get_capabilities()

    assert adapter.capabilities.supports("forward_sim")
    assert adapter.capabilities.supports("inverse_dynamics")
    assert adapter.capabilities.supports("contact")
    assert adapter.capabilities.supports("state_control_gradients")
    assert report.engine_name == "Drake"
    assert report.forward_sim == CapabilityLevel.FULL
    assert report.inverse_dynamics == CapabilityLevel.FULL
    assert report.contact_step == CapabilityLevel.FULL
    assert report.state_control_gradients == CapabilityLevel.FULL
    assert report.parameter_gradients == CapabilityLevel.PARTIAL
    assert report.trajectory_opt == CapabilityLevel.FULL
    assert report.extra["gradient_scalar"] == "AutoDiffXd"
    assert report.extra["model_exports"] == ("urdf", "sdf")


def test_drake_registers_hydroelastic_contact_divergence() -> None:
    divergence = DrakeAdapter().registered_divergences()[0]

    assert divergence is HYDROELASTIC_CONTACT_DIVERGENCE
    assert divergence.engines == ("drake", "pinocchio")
    assert "hydroelastic" in divergence.rationale
    assert divergence.tolerance > 0.0


def test_drake_divergence_registry_document_matches_code() -> None:
    registry_path = (
        _REPO_ROOT / "docs" / "conformance" / ("canonical_core_divergences.yaml")
    )

    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entry = payload["divergences"][0]

    assert entry["id"] == HYDROELASTIC_CONTACT_DIVERGENCE.id
    assert tuple(entry["engines"]) == HYDROELASTIC_CONTACT_DIVERGENCE.engines
    assert entry["check_name"] == HYDROELASTIC_CONTACT_DIVERGENCE.check_name
    assert entry["metric_name"] == HYDROELASTIC_CONTACT_DIVERGENCE.metric_name
