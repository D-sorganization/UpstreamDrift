"""Connection graph contracts for native Simscape inventories."""

from collections.abc import Sequence
from typing import Any

import pytest

from src.shared.python.motion_matching.native_inventory import (
    physical_connection_components,
)


def block(
    path: str,
    parent: str,
    kind: str = "SimscapeMultibodyBlock",
    ports: Sequence[dict[str, Any]] = (),
    commented: str = "off",
    parameters: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "path": path,
        "parent": parent,
        "block_type": kind,
        "commented": commented,
        "parameters": list(parameters),
        "connectivity": list(ports),
    }


def port(kind: str, *destinations: str) -> dict[str, Any]:
    return {"Type": kind, "SrcEndpoint": [], "DstEndpoint": list(destinations)}


def test_bridges_subsystem_ports_without_merging_joint_sides() -> None:
    blocks = [
        block("m/sub", "m", "SubSystem", [port("LConn1", "m/world:RConn1")]),
        block("m/world", "m", ports=[port("RConn1", "m/sub:LConn1")]),
        block(
            "m/sub/base",
            "m/sub",
            "PMIOPort",
            [port("RConn1", "m/sub/joint:LConn1")],
            parameters=[
                {"name": "Side", "expression": "Left"},
                {"name": "Port", "resolved_numeric": True, "numeric_value": 2},
            ],
        ),
        block(
            "m/sub/joint",
            "m/sub",
            ports=[port("LConn1", "m/sub/base:RConn1"), port("RConn1")],
        ),
    ]
    groups = physical_connection_components(
        {"schema_version": 2, "model": "m", "blocks": blocks}
    )
    assert ("m/sub/joint:RConn1",) in groups
    assert (
        tuple(
            sorted(
                [
                    "m/world:RConn1",
                    "m/sub:LConn1",
                    "m/sub/base:RConn1",
                    "m/sub/joint:LConn1",
                ]
            )
        )
        in groups
    )


def test_excludes_commented_ancestors_but_rejects_through() -> None:
    blocks = [
        block("m/off", "m", "SubSystem", commented="on"),
        block("m/off/beam", "m/off", ports=[port("LConn1")]),
    ]
    data = {"schema_version": 2, "model": "m", "blocks": blocks}
    assert physical_connection_components(data) == ()
    blocks[0]["commented"] = "through"
    with pytest.raises(ValueError, match="through"):
        physical_connection_components(data)


def test_rejects_dangling_endpoint_and_duplicate_paths() -> None:
    data = {
        "schema_version": 2,
        "model": "m",
        "blocks": [block("m/a", "m", ports=[port("LConn1", "m/missing:RConn1")])],
    }
    with pytest.raises(ValueError, match="Unknown physical endpoint"):
        physical_connection_components(data)
    data["blocks"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        physical_connection_components(data)
