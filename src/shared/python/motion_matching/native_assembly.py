"""Assemble native fixed frames while retaining every articulated connection."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.native_inventory import (
    physical_connection_components,
    uncommented_blocks,
)
from src.shared.python.motion_matching.native_solids import (
    NativeParameters,
    SolidProperties,
    solid_properties,
)
from src.shared.python.motion_matching.native_transforms import rigid_transform


class RigidFrameGraph:
    """Edges map coordinates of the destination frame into the source frame."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, NDArray[np.float64]]] = {}

    def connect(
        self, source: str, destination: str, transform: NDArray[np.float64]
    ) -> None:
        matrix = np.asarray(transform, dtype=float)
        if (
            matrix.shape != (4, 4)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], [0, 0, 0, 1])
        ):
            raise ValueError("Invalid homogeneous transform")
        rotation = matrix[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0
        ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0):
            raise ValueError("Invalid rigid rotation")
        existing = self._edges.get(source, {}).get(destination)
        if existing is not None and not np.allclose(
            existing, matrix, atol=1e-9, rtol=0
        ):
            raise ValueError("Inconsistent duplicate rigid edge")
        self._edges.setdefault(source, {})[destination] = matrix.copy()
        self._edges.setdefault(destination, {})[source] = np.linalg.inv(matrix)

    def component(self, root: str) -> dict[str, NDArray[np.float64]]:
        if root not in self._edges:
            raise ValueError(f"Unknown rigid frame {root}")
        poses = {root: np.eye(4)}
        pending = [root]
        while pending:
            source = pending.pop()
            for destination, transform in self._edges[source].items():
                pose = poses[source] @ transform
                if destination in poses:
                    if not np.allclose(poses[destination], pose, atol=1e-8, rtol=0):
                        raise ValueError(f"Inconsistent rigid cycle at {destination}")
                else:
                    poses[destination] = pose
                    pending.append(destination)
        return poses


@dataclass(frozen=True)
class NativeAssembly:
    """Uncompiled geometry graph, solids and explicitly retained joints."""

    graph: RigidFrameGraph
    solids: Mapping[str, SolidProperties]
    joints: Mapping[str, Mapping[str, Any]]
    cut_weld: str


def solid_reference(path: str) -> str:
    return "solid_reference:" + path


def assemble_native_frames(
    document: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, list[str]]],
    cut_weld: str,
) -> NativeAssembly:
    """Compose wire nets, solid frames and fixed transforms; open one weld.

    Joint primitives remain separate. No dynamic state or actuator conversion
    is inferred by this geometric assembly.
    """
    blocks = uncommented_blocks(document)
    if cut_weld not in blocks or not blocks[cut_weld]["library_reference"].endswith(
        "/Weld Joint"
    ):
        raise ValueError("Loop cut must identify a native weld")
    graph = RigidFrameGraph()
    for net in physical_connection_components(document):
        for endpoint in net:
            graph.connect(net[0], endpoint, np.eye(4))
    solids = {}
    joints = {}
    for path, block in blocks.items():
        library = " ".join(block["library_reference"].split())
        if "Solid" in library:
            properties = solid_properties(block)
            solids[path] = properties
            if path not in bindings:
                raise ValueError(f"Missing native solid port bindings: {path}")
            connections = block["connectivity"]
            if isinstance(connections, dict):
                connections = [connections]
            if set(bindings[path]) != {port["Type"] for port in connections}:
                raise ValueError(f"Incomplete native solid port coverage: {path}")
            for port, aliases in bindings[path].items():
                if not aliases:
                    raise ValueError("Empty native frame alias list")
                transform = properties.frames[aliases[0]]
                if any(
                    not np.allclose(
                        transform, properties.frames[name], atol=1e-9, rtol=0
                    )
                    for name in aliases
                ):
                    raise ValueError(
                        "Measured frame aliases disagree with solid geometry"
                    )
                graph.connect(solid_reference(path), f"{path}:{port}", transform)
        elif library == "sm_lib/Frames and Transforms/Rigid Transform":
            graph.connect(path + ":LConn1", path + ":RConn1", rigid_transform(block))
        elif library == "sm_lib/Frames and Transforms/World Frame":
            graph.connect("world", path + ":RConn1", np.eye(4))
        elif library.startswith("sm_lib/Joints/"):
            if NativeParameters(block).text("JointMode") != "Normal":
                raise ValueError(f"Unsupported native joint mode: {path}")
            if library.endswith("/Weld Joint") and path != cut_weld:
                graph.connect(path + ":LConn1", path + ":RConn1", np.eye(4))
            else:
                joints[path] = block
    if set(bindings) != set(solids):
        raise ValueError("Port binding inventory differs from native solids")
    return NativeAssembly(graph, solids, joints, cut_weld)
