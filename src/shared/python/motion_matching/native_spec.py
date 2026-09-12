"""Portable native-derived rigid-body tree plus an explicit weld closure."""

from collections.abc import Mapping, Sequence
from typing import Any

from src.shared.python.motion_matching.native_assembly import (
    assemble_native_frames,
    solid_reference,
)
from src.shared.python.motion_matching.native_inventory import uncommented_blocks
from src.shared.python.motion_matching.native_solids import NativeParameters

_PRIMITIVES = {
    "Bushing Joint": ("Px", "Py", "Pz", "Rx", "Ry", "Rz"),
    "Universal Joint": ("Rx", "Ry"),
    "Gimbal Joint": ("Rx", "Ry", "Rz"),
    "Revolute Joint": ("Rz",),
}


def order_native_tree(edges: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Require one connected directed tree rooted at world."""
    children = [edge["child"] for edge in edges]
    if len(set(children)) != len(children) or "world" in children:
        raise ValueError("Native tree has multiple parents or a cycle through world")
    reached = {"world"}
    pending = list(edges)
    ordered = []
    while pending:
        eligible = [edge for edge in pending if edge["parent"] in reached]
        if not eligible:
            raise ValueError("Native joint tree is cyclic or disconnected")
        for edge in eligible:
            ordered.append(edge)
            reached.add(edge["child"])
            pending.remove(edge)
    return ordered


def build_native_spec(
    document: Mapping[str, Any],
    bindings: Mapping[str, Any],
    schema: Mapping[str, Any],
    cut_weld: str,
) -> dict[str, Any]:
    """Preserve all solids, joint primitives and requested native marker frames.

    This geometry export does not yet qualify actuation, damping, limits, or
    solver behavior. Consumers must not use it as a dynamics parity certificate.
    """
    assembly = assemble_native_frames(document, bindings, cut_weld)
    groups = {}
    owner = {}
    for root in ["world"] + [solid_reference(path) for path in assembly.solids]:
        if root not in owner:
            poses = assembly.graph.component(root)
            groups[root] = poses
            owner.update(dict.fromkeys(poses, root))
    bodies: dict[str, dict[str, Any]] = {
        root: {"name": root, "solids": []} for root in groups
    }
    for path, solid in assembly.solids.items():
        reference = solid_reference(path)
        root = owner[reference]
        solids_list: list[dict[str, Any]] = bodies[root]["solids"]
        solids_list.append(
            {
                "name": path,
                "mass_kg": solid.mass_kg,
                "com_m": solid.com_m.tolist(),
                "inertia_com_kg_m2": solid.inertia_com_kg_m2.tolist(),
                "placement": groups[root][reference].tolist(),
            }
        )
    coordinates = {
        (entry["block_path"], entry["primitive"]): entry["name"]
        for entry in schema["coordinates"]
    }
    if len(coordinates) != len(schema["coordinates"]):
        raise ValueError("Duplicate native coordinate identity")
    edges = []
    used_coordinates = set()
    for path, block in assembly.joints.items():
        if path == cut_weld:
            continue
        kind = " ".join(block["library_reference"].split()).rsplit("/", 1)[1]
        if kind not in _PRIMITIVES:
            raise ValueError(f"Unsupported native joint {kind}")
        base, follower = path + ":LConn1", path + ":RConn1"
        parent, child = owner[base], owner[follower]
        primitives = []
        for primitive in _PRIMITIVES[kind]:
            key = (path, primitive)
            if key not in coordinates:
                raise ValueError(f"Missing native coordinate {key}")
            used_coordinates.add(key)
            primitives.append({"primitive": primitive, "coordinate": coordinates[key]})
        edges.append(
            {
                "name": path,
                "parent": parent,
                "child": child,
                "parent_to_base": groups[parent][base].tolist(),
                "child_to_follower": groups[child][follower].tolist(),
                "primitives": primitives,
            }
        )
    if used_coordinates != set(coordinates):
        raise ValueError("Native coordinate inventory was not completely preserved")
    ordered = order_native_tree(edges)
    if set(groups) != {"world"} | {edge["child"] for edge in ordered}:
        raise ValueError("Rigid bodies missing from native joint tree")
    frames = []
    for frame in schema["frames"]:
        path, label = frame["port"].rsplit("/", 1)
        if label not in ("B", "F", "W"):
            raise ValueError(f"Unsupported native reference frame port {label}")
        endpoint = path + (":LConn1" if label == "B" else ":RConn1")
        root = owner[endpoint]
        frames.append(
            {
                "name": frame["name"],
                "body": root,
                "placement": groups[root][endpoint].tolist(),
            }
        )
    base, follower = cut_weld + ":LConn1", cut_weld + ":RConn1"
    closure = {
        "name": cut_weld,
        "body_a": owner[base],
        "body_b": owner[follower],
        "placement_a": groups[owner[base]][base].tolist(),
        "placement_b": groups[owner[follower]][follower].tolist(),
    }
    configurations = [
        block
        for block in uncommented_blocks(document).values()
        if "Mechanism" in block["library_reference"]
    ]
    if len(configurations) != 1:
        raise ValueError("Native model requires one explicit mechanism configuration")
    configuration = NativeParameters(configurations[0])
    if configuration.text("UniformGravity") != "Constant":
        raise ValueError("Only explicit constant native gravity is supported")
    return {
        "schema_version": 1,
        "qualification": "native-derived geometry; dynamics unqualified",
        "gravity_m_s2": configuration.vector(
            "GravityVector", "acceleration", 3
        ).tolist(),
        "coordinate_order": [entry["name"] for entry in schema["coordinates"]],
        "bodies": list(bodies.values()),
        "joints": ordered,
        "frames": frames,
        "closure": closure,
    }
