"""Read native Simscape physical wire nets without approximating mechanisms.

These nets represent wires and subsystem boundaries only. Solid-frame offsets,
rigid transforms, joints, constraints and active variants require separate
qualification before this graph can become a dynamics model.
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def _records(value: Any) -> list[dict[str, Any]]:
    """MATLAB JSON encodes a singleton struct differently from an array."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError("Expected native record or array of records")


def _active_blocks(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("schema_version") != 2:
        raise ValueError("Stable physical endpoints require inventory schema 2")
    records = _records(document["blocks"])
    blocks = {item["path"]: item for item in records}
    if len(blocks) != len(records):
        raise ValueError("Duplicate native block path")
    active = {}
    for path, block in blocks.items():
        current = path
        visited = set()
        enabled = True
        while current != document["model"]:
            if current in visited or current not in blocks:
                raise ValueError(f"Invalid native ancestry at {current}")
            visited.add(current)
            ancestor = blocks[current]
            state = ancestor["commented"]
            if state == "through":
                raise ValueError("Commented-through routing requires native resolution")
            if state not in ("on", "off"):
                raise ValueError(f"Unknown comment state {state}")
            enabled = enabled and state == "off"
            current = ancestor["parent"]
        if enabled:
            active[path] = block
    return active


def _connect(graph: dict[str, set[str]], left: str, right: str) -> None:
    if left not in graph or right not in graph:
        raise ValueError(f"Unknown physical endpoint: {left} or {right}")
    graph[left].add(right)
    graph[right].add(left)


def _bridge_subsystems(
    graph: dict[str, set[str]], blocks: dict[str, dict[str, Any]]
) -> None:
    boundaries: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for path, block in blocks.items():
        if block["block_type"] != "PMIOPort":
            continue
        parameters = {item["name"]: item for item in _records(block["parameters"])}
        side = parameters["Side"]["expression"]
        port = parameters["Port"]
        if side not in ("Left", "Right") or not port.get("resolved_numeric"):
            raise ValueError(f"Unresolved physical boundary at {path}")
        number = port["numeric_value"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or int(number) != number
            or number < 1
        ):
            raise ValueError(f"Invalid physical port number at {path}")
        boundaries[block["parent"], side].append((int(number), path))
    for (parent, side), ports in boundaries.items():
        if len({number for number, _ in ports}) != len(ports):
            raise ValueError(f"Duplicate physical port number at {parent}")
        kind = "LConn" if side == "Left" else "RConn"
        for index, (_, path) in enumerate(sorted(ports), 1):
            internal = [
                endpoint for endpoint in graph if endpoint.rsplit(":", 1)[0] == path
            ]
            if len(internal) != 1:
                raise ValueError(f"Expected one internal boundary endpoint at {path}")
            _connect(graph, f"{parent}:{kind}{index}", internal[0])


def physical_connection_components(
    document: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic physical wire nets, excluding commented ancestry.

    Fail on dangling endpoints instead of fabricating connections. Physical
    signal nets are included; consumers must identify frame ports from native
    component metadata. No ports are merged merely because they share a block.
    """
    blocks = _active_blocks(document)
    all_blocks = {item["path"] for item in _records(document["blocks"])}
    graph: dict[str, set[str]] = {}
    connections = []
    for path, block in blocks.items():
        for port in _records(block["connectivity"]):
            if not port["Type"].startswith(("LConn", "RConn")):
                continue
            endpoint = f"{path}:{port['Type']}"
            graph[endpoint] = set()
            connections.append((endpoint, port))
    for endpoint, port in connections:
        for key in ("SrcEndpoint", "DstEndpoint"):
            targets = port[key]
            if not isinstance(targets, list):
                raise ValueError("Native endpoints must be arrays")
            for target in targets:
                target_path = target.rsplit(":", 1)[0]
                if target_path in all_blocks and target_path not in blocks:
                    continue
                _connect(graph, endpoint, target)
    _bridge_subsystems(graph, blocks)
    groups = []
    unseen = set(graph)
    while unseen:
        pending = [min(unseen)]
        component = set()
        while pending:
            endpoint = pending.pop()
            if endpoint in component:
                continue
            component.add(endpoint)
            pending.extend(graph[endpoint] - component)
        unseen.difference_update(component)
        groups.append(tuple(sorted(component)))
    return tuple(sorted(groups))
