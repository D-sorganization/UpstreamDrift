"""Typed attachment port compatibility rules (CMB-8, #11659)."""

from __future__ import annotations

import pytest

from src.shared.python.model_generation.editor.attachment_ports import (
    PortPolarity,
    PortType,
    TypedPort,
    check_port_compatibility,
    parse_port_polarity,
    parse_port_type,
    side_from_tags,
)

pytestmark = pytest.mark.unit

S, P = PortPolarity.SOCKET, PortPolarity.PLUG


def test_matching_socket_and_plug_are_compatible() -> None:
    verdict = check_port_compatibility(
        TypedPort(PortType.GRIP, S), TypedPort(PortType.GRIP, P)
    )
    assert verdict.ok and verdict.reason == ""


@pytest.mark.parametrize(
    ("host", "part", "fragment"),
    [
        (TypedPort(PortType.GRIP, P), TypedPort(PortType.GRIP, P), "not a socket"),
        (TypedPort(PortType.GRIP, S), TypedPort(PortType.GRIP, S), "not a plug"),
        (TypedPort(PortType.HIP, S), TypedPort(PortType.SHOULDER, P), "type mismatch"),
        (
            TypedPort(PortType.ANKLE, S, side="left"),
            TypedPort(PortType.ANKLE, P, side="right"),
            "side mismatch",
        ),
    ],
)
def test_incompatible_pairs_give_a_reason(
    host: TypedPort, part: TypedPort, fragment: str
) -> None:
    verdict = check_port_compatibility(host, part)
    assert not verdict.ok
    assert fragment in verdict.reason


def test_side_none_matches_either_side() -> None:
    host = TypedPort(PortType.ANKLE, S, side="left")
    assert check_port_compatibility(host, TypedPort(PortType.ANKLE, P)).ok


def test_payload_limit_is_enforced() -> None:
    host = TypedPort(PortType.GRIP, S, max_payload_kg=1.0)
    part = TypedPort(PortType.GRIP, P)
    assert check_port_compatibility(host, part, part_mass_kg=0.9).ok
    over = check_port_compatibility(host, part, part_mass_kg=1.5)
    assert not over.ok and "payload" in over.reason


def test_preconditions_raise() -> None:
    with pytest.raises(ValueError):
        TypedPort(PortType.GRIP, S, side="middle")
    with pytest.raises(ValueError):
        TypedPort(PortType.GRIP, S, max_payload_kg=0.0)
    with pytest.raises(ValueError):
        check_port_compatibility(None, TypedPort(PortType.GRIP, P))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_port_type("flange")
    with pytest.raises(TypeError):
        parse_port_polarity(3)  # type: ignore[arg-type]


def test_parsers_and_side_tags() -> None:
    assert parse_port_type(" Grip ") is PortType.GRIP
    assert parse_port_polarity("PLUG") is PortPolarity.PLUG
    assert side_from_tags(("hand", "Left")) == "left"
    assert side_from_tags(("left", "right")) is None
    assert side_from_tags(()) is None
