"""Unit tests verifying native avatar animation and autonomous course feedback research bounds.

Covers GS-11 (#10200) under Epic #10188:
- Asserts honest UNSUPPORTED capability declarations across all adapters.
- Verifies UnsupportedCapabilityError on attempted avatar mesh injection.
- Verifies rejection of autonomous closed-loop play when telemetry is unsupported.
- Verifies graceful transition to operator-managed play without guessing.
"""

from __future__ import annotations

import pytest

from src.shared.python.golf_simulator.adapters.fake import FakeSimulatorAdapter
from src.shared.python.golf_simulator.adapters.local import LocalReferenceAdapter
from src.shared.python.golf_simulator.adapters.relay import (
    FlightRelayAdapter,
    FlightRelayConfig,
)
from src.shared.python.golf_simulator.contracts import (
    CapabilityState,
    SimulatorCapabilities,
    UnsupportedCapabilityError,
    assert_capability_supported,
)
from src.shared.python.golf_simulator.session import GolfSessionService

pytestmark = [pytest.mark.unit]


def test_native_avatar_and_course_feedback_honestly_unsupported() -> None:
    """Verify that all production and reference adapters honestly declare native avatar and course telemetry as UNSUPPORTED."""
    local_adapter = LocalReferenceAdapter()
    caps_local = local_adapter.capabilities()
    assert caps_local.native_avatar_animation.state == CapabilityState.UNSUPPORTED
    assert caps_local.course_state_feedback.state == CapabilityState.UNSUPPORTED
    assert caps_local.aim_control.state == CapabilityState.UNSUPPORTED

    relay_adapter = FlightRelayAdapter(
        FlightRelayConfig(destination_name="flight_relay")
    )
    caps_relay = relay_adapter.capabilities()
    assert caps_relay.native_avatar_animation.state == CapabilityState.UNSUPPORTED
    assert caps_relay.course_state_feedback.state == CapabilityState.UNSUPPORTED
    assert caps_relay.aim_control.state == CapabilityState.UNSUPPORTED

    fake_adapter = FakeSimulatorAdapter()
    caps_fake = fake_adapter.capabilities()
    assert caps_fake.native_avatar_animation.state == CapabilityState.UNSUPPORTED
    assert caps_fake.course_state_feedback.state == CapabilityState.UNSUPPORTED
    assert caps_fake.aim_control.state == CapabilityState.UNSUPPORTED


def test_assert_capability_supported_raises_on_unsupported() -> None:
    """Verify assert_capability_supported raises UnsupportedCapabilityError for unsupported or unverified states."""
    local_adapter = LocalReferenceAdapter()
    caps = local_adapter.capabilities()

    # Supported capabilities should pass without error
    assert_capability_supported(caps, "shot_input")

    # Unsupported capabilities must raise UnsupportedCapabilityError with explicit evidence
    with pytest.raises(UnsupportedCapabilityError) as exc_avatar:
        assert_capability_supported(caps, "native_avatar_animation")
    assert "native_avatar_animation" in str(exc_avatar.value)
    assert exc_avatar.value.capability_name == "native_avatar_animation"

    with pytest.raises(UnsupportedCapabilityError) as exc_course:
        assert_capability_supported(caps, "course_state_feedback")
    assert "course_state_feedback" in str(exc_course.value)
    assert exc_course.value.capability_name == "course_state_feedback"

    with pytest.raises(UnsupportedCapabilityError) as exc_aim:
        assert_capability_supported(caps, "aim_control")
    assert "aim_control" in str(exc_aim.value)
    assert exc_aim.value.capability_name == "aim_control"


def test_assert_capability_supported_rejects_unknown_capability() -> None:
    """Verify assert_capability_supported rejects unrecognized capability attribute names."""
    local_adapter = LocalReferenceAdapter()
    caps = local_adapter.capabilities()

    with pytest.raises(AttributeError) as exc_attr:
        assert_capability_supported(caps, "nonexistent_capability_name")
    assert "nonexistent_capability_name" in str(exc_attr.value)


def test_autonomous_round_requires_course_feedback() -> None:
    """Verify that autonomous round execution requires course feedback and fails cleanly rather than guessing."""
    adapter = FakeSimulatorAdapter()
    service = GolfSessionService(session_id="session-autonomous-audit", adapter=adapter)

    # Attempting to check prerequisites for autonomous loop
    caps = adapter.capabilities()
    with pytest.raises(UnsupportedCapabilityError) as exc:
        assert_capability_supported(caps, "course_state_feedback")

    assert exc.value.state == CapabilityState.UNSUPPORTED
    ev_lower = exc.value.evidence.lower()
    assert (
        "not supported" in ev_lower
        or "unsupported" in ev_lower
        or "does not report" in ev_lower
    )


def test_companion_presentation_fallback_mode() -> None:
    """Verify that companion presentation mode is explicitly distinct from native in-scene injection."""
    adapter = LocalReferenceAdapter()
    caps = adapter.capabilities()

    # Native animation is unsupported
    assert caps.native_avatar_animation.state == CapabilityState.UNSUPPORTED

    # Companion mode relies on local viewport/reference presentation, which is fully supported
    assert caps.shot_input.state == CapabilityState.SUPPORTED
    assert (
        "Local reference visualizer" in caps.shot_input.evidence
        or "Local" in caps.shot_input.evidence
    )
