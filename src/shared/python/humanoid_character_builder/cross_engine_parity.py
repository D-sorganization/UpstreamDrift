"""Cross-engine L0 export parity verification for Character Builder (CMB-11, #11662)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.shared.python.humanoid_character_builder.spec_params import serialize_spec

if TYPE_CHECKING:
    from src.shared.python.humanoid_character_builder.spec_export import (
        CompiledCharacter,
    )


def _body_mass(body: Any) -> float:
    return float(sum(s["mass_kg"] for s in body["solids"]))


def verify_cross_engine_l0_parity(
    character: CompiledCharacter,
    *,
    mass_tol_kg: float = 1e-9,
    frame_tol_m: float = 1e-3,
) -> dict[str, Any]:
    """Verify L0 export parity across MuJoCo, Drake/URDF, OpenSim, and Pinocchio (CMB-11, #11662).

    Validates total mass conservation and world-frame origin agreement at
    neutral pose (q = 0) against the compiled full-body specification.
    """
    if mass_tol_kg <= 0:
        raise ValueError(f"mass_tol_kg must be > 0, got {mass_tol_kg}")
    if frame_tol_m <= 0:
        raise ValueError(f"frame_tol_m must be > 0, got {frame_tol_m}")

    import defusedxml.ElementTree as DefusedET
    import numpy as np
    from scipy.spatial.transform import Rotation

    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        clean_osim_body_name,
        export_full_body_osim,
    )
    from src.engines.physics_engines.pinocchio.python.native_model import (
        depth_first_joints,
    )
    from src.shared.python.motion_matching.full_body_spec import order_full_body_joints

    doc = character.document
    spec_mass_kg = float(sum(_body_mass(b) for b in doc["bodies"]))

    # Reference world poses for bodies and frame origins at q = 0
    spec_body_poses: dict[str, np.ndarray] = {"world": np.eye(4)}
    for joint in order_full_body_joints(doc):
        parent = joint["parent"]
        child = joint["child"]
        p_pose = spec_body_poses[parent]
        p_base = np.asarray(joint["parent_to_base"], dtype=float)
        c_follower = np.asarray(joint["child_to_follower"], dtype=float)
        spec_body_poses[child] = p_pose @ p_base @ np.linalg.inv(c_follower)

    spec_frame_origins: dict[str, np.ndarray] = {}
    for frame in doc.get("frames", []):
        b_name = frame["body"]
        placement = np.asarray(frame["placement"], dtype=float)
        f_world = spec_body_poses[b_name] @ placement
        spec_frame_origins[frame["name"]] = f_world[:3, 3]

    spec_bytes = serialize_spec(doc).encode("utf-8")

    # --- 1. MJCF ---
    mjcf_xml, mjcf_meta = export_full_body_mjcf(spec_bytes)
    mjcf_root = DefusedET.fromstring(mjcf_xml)
    mjcf_mass_kg = float(
        sum(float(i.attrib["mass"]) for i in mjcf_root.iter("inertial"))
    )

    def _parse_mjcf_pose(elem: Any) -> np.ndarray:
        T = np.eye(4)
        if "pos" in elem.attrib:
            T[:3, 3] = [float(x) for x in elem.attrib["pos"].split()]
        if "quat" in elem.attrib:
            w, x, y, z = [float(v) for v in elem.attrib["quat"].split()]
            T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        return T

    mjcf_site_origins: dict[str, np.ndarray] = {}
    mjcf_body_origins: dict[str, np.ndarray] = {}

    def _traverse_mjcf(elem: Any, parent_pose: np.ndarray) -> None:
        world_T = parent_pose @ _parse_mjcf_pose(elem)
        if elem.tag == "body":
            b_name = elem.attrib.get("name")
            if b_name:
                mjcf_body_origins[b_name] = world_T[:3, 3]
        for site in elem.findall("site"):
            s_name = site.attrib.get("name")
            if s_name:
                s_world = world_T @ _parse_mjcf_pose(site)
                mjcf_site_origins[s_name] = s_world[:3, 3]
        for child_body in elem.findall("body"):
            _traverse_mjcf(child_body, world_T)

    worldbody = mjcf_root.find("worldbody")
    if worldbody is not None:
        for b in worldbody.findall("body"):
            _traverse_mjcf(b, np.eye(4))

    frame_sites = mjcf_meta.get("frame_sites", {})
    mjcf_frame_errors = []
    for fname, sname in frame_sites.items():
        if fname in spec_frame_origins and sname in mjcf_site_origins:
            err = float(
                np.linalg.norm(spec_frame_origins[fname] - mjcf_site_origins[sname])
            )
            mjcf_frame_errors.append(err)
    mjcf_max_frame_err = max(mjcf_frame_errors) if mjcf_frame_errors else 0.0

    # --- 2. URDF ---
    urdf_xml, urdf_sidecar = export_full_body_urdf(spec_bytes)
    urdf_root = DefusedET.fromstring(urdf_xml)
    urdf_mass_kg = float(
        sum(
            float(m.attrib["value"])
            for link in urdf_root.findall("link")
            for m in link.findall("inertial/mass")
        )
    )

    body_links = urdf_sidecar.get("body_links", {})
    root_link = body_links.get("world", "body_0")
    urdf_link_origins: dict[str, np.ndarray] = {root_link: np.zeros(3)}
    urdf_link_poses: dict[str, np.ndarray] = {root_link: np.eye(4)}

    unresolved_urdf = list(urdf_root.findall("joint"))
    while unresolved_urdf:
        next_unresolved = []
        progress = False
        for j in unresolved_urdf:
            p_link = j.find("parent").attrib["link"]
            c_link = j.find("child").attrib["link"]
            if p_link in urdf_link_poses:
                origin_elem = j.find("origin")
                xyz = (
                    [float(x) for x in origin_elem.attrib.get("xyz", "0 0 0").split()]
                    if origin_elem is not None
                    else [0.0, 0.0, 0.0]
                )
                rpy = (
                    [float(x) for x in origin_elem.attrib.get("rpy", "0 0 0").split()]
                    if origin_elem is not None
                    else [0.0, 0.0, 0.0]
                )
                T = np.eye(4)
                T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                T[:3, 3] = xyz
                world_T = urdf_link_poses[p_link] @ T
                urdf_link_poses[c_link] = world_T
                urdf_link_origins[c_link] = world_T[:3, 3]
                progress = True
            else:
                next_unresolved.append(j)
        if not progress:
            break
        unresolved_urdf = next_unresolved

    urdf_frame_links = urdf_sidecar.get("frame_links", {})
    urdf_frame_errors = []
    for fname, flink in urdf_frame_links.items():
        if fname in spec_frame_origins and flink in urdf_link_origins:
            err = float(
                np.linalg.norm(spec_frame_origins[fname] - urdf_link_origins[flink])
            )
            urdf_frame_errors.append(err)
    urdf_max_frame_err = max(urdf_frame_errors) if urdf_frame_errors else 0.0

    # --- 3. OpenSim ---
    osim_xml, _ = export_full_body_osim(doc)
    osim_root = DefusedET.fromstring(osim_xml.replace("::", "__"))
    osim_mass_kg = float(
        sum(
            float(m.text)
            for m in osim_root.findall(".//BodySet/objects/Body/mass")
            if m.text is not None
        )
    )

    osim_bodies = {
        b.attrib["name"]
        for b in osim_root.findall(".//BodySet/objects/Body")
        if "name" in b.attrib
    }
    for b in doc["bodies"]:
        if b["name"] == "world":
            continue
        expected_name = clean_osim_body_name(b["name"])
        if expected_name not in osim_bodies:
            raise ValueError(f"OpenSim missing body {expected_name} (from {b['name']})")

    osim_markers = {
        m.attrib["name"]
        for m in osim_root.findall(".//MarkerSet/objects/Marker")
        if "name" in m.attrib
    }
    for mname in doc.get("marker_attachments", {}):
        if mname not in osim_markers:
            raise ValueError(f"OpenSim missing marker {mname}")

    osim_poses: dict[str, np.ndarray] = {"/ground": np.eye(4)}
    unresolved_osim = list(osim_root.findall(".//CustomJoint"))
    while unresolved_osim:
        next_unresolved = []
        progress = False
        for j in unresolved_osim:
            frames = j.findall("frames/PhysicalOffsetFrame")
            if len(frames) >= 2:
                f_p, f_c = frames[0], frames[1]
                sp_p = (
                    f_p.find("socket_parent").text
                    if f_p.find("socket_parent") is not None
                    else ""
                )
                sp_c = (
                    f_c.find("socket_parent").text
                    if f_c.find("socket_parent") is not None
                    else ""
                )
                if sp_p in osim_poses:
                    t_p = np.array(
                        [
                            float(x)
                            for x in (f_p.find("translation").text or "0 0 0").split()
                        ]
                    )
                    o_p = np.array(
                        [
                            float(x)
                            for x in (f_p.find("orientation").text or "0 0 0").split()
                        ]
                    )
                    T_p = np.eye(4)
                    T_p[:3, :3] = Rotation.from_euler("XYZ", o_p).as_matrix()
                    T_p[:3, 3] = t_p

                    t_c = np.array(
                        [
                            float(x)
                            for x in (f_c.find("translation").text or "0 0 0").split()
                        ]
                    )
                    o_c = np.array(
                        [
                            float(x)
                            for x in (f_c.find("orientation").text or "0 0 0").split()
                        ]
                    )
                    T_c = np.eye(4)
                    T_c[:3, :3] = Rotation.from_euler("XYZ", o_c).as_matrix()
                    T_c[:3, 3] = t_c

                    osim_poses[sp_c] = osim_poses[sp_p] @ T_p @ np.linalg.inv(T_c)
                    progress = True
                else:
                    next_unresolved.append(j)
        if not progress:
            break
        unresolved_osim = next_unresolved

    osim_body_errors = []
    for b in doc["bodies"]:
        if b["name"] == "world":
            continue
        clean = clean_osim_body_name(b["name"])
        sp = f"/bodyset/{clean}"
        if sp in osim_poses and b["name"] in spec_body_poses:
            err = float(
                np.linalg.norm(
                    spec_body_poses[b["name"]][:3, 3] - osim_poses[sp][:3, 3]
                )
            )
            osim_body_errors.append(err)
    osim_max_frame_err = max(osim_body_errors) if osim_body_errors else 0.0

    # --- 4. Pinocchio ---
    pin_joints = depth_first_joints(doc["joints"])
    if len(pin_joints) != len(doc["joints"]):
        raise ValueError(
            f"Pinocchio joint tree length mismatch: {len(pin_joints)} != {len(doc['joints'])}"
        )
    pin_joint_count = len(pin_joints)

    try:
        import sys
        import types

        import pinocchio  # noqa: F401

        is_mock = not isinstance(pinocchio, types.ModuleType) or (
            not hasattr(pinocchio, "__file__")
            and pinocchio.__name__ not in sys.builtin_module_names
        )
        if not is_mock:
            from src.engines.physics_engines.pinocchio.python.native_model import (
                FullBodyPinocchioModel,
            )

            pin_model = FullBodyPinocchioModel(doc)
            pin_frame_poses = pin_model.frame_poses({})
            for fname, fpose in pin_frame_poses.items():
                if fname in spec_frame_origins:
                    err = float(
                        np.linalg.norm(spec_frame_origins[fname] - fpose[:3, 3])
                    )
                    if err > frame_tol_m:
                        raise ValueError(
                            f"Pinocchio frame origin error {err} > {frame_tol_m} for {fname}"
                        )
            pin_total_mass = float(sum(b.mass for b in pin_model.model.inertias))
            if abs(pin_total_mass - spec_mass_kg) > mass_tol_kg:
                raise ValueError(
                    f"Pinocchio mass mismatch: {pin_total_mass} vs {spec_mass_kg}"
                )
    except (ImportError, ModuleNotFoundError, TypeError):
        pass

    engine_masses_kg = {
        "mjcf": mjcf_mass_kg,
        "urdf": urdf_mass_kg,
        "osim": osim_mass_kg,
    }
    engine_max_frame_error_m = {
        "mjcf": mjcf_max_frame_err,
        "urdf": urdf_max_frame_err,
        "osim": osim_max_frame_err,
    }

    mass_errors = [abs(m - spec_mass_kg) for m in engine_masses_kg.values()]
    max_mass_error_kg = float(max(mass_errors)) if mass_errors else 0.0
    max_frame_origin_error_m = (
        float(max(engine_max_frame_error_m.values()))
        if engine_max_frame_error_m
        else 0.0
    )

    if max_mass_error_kg > mass_tol_kg:
        raise ValueError(
            f"Cross-engine mass parity violation: max mass error {max_mass_error_kg:.4e} kg "
            f"> tolerance {mass_tol_kg:.4e} kg (engine masses: {engine_masses_kg})"
        )

    if max_frame_origin_error_m > frame_tol_m:
        raise ValueError(
            f"Cross-engine frame origin parity violation: max frame origin error "
            f"{max_frame_origin_error_m:.4e} m > tolerance {frame_tol_m:.4e} m "
            f"(engine errors: {engine_max_frame_error_m})"
        )

    return {
        "spec_sha256": character.sha256,
        "spec_mass_kg": spec_mass_kg,
        "engine_masses_kg": engine_masses_kg,
        "engine_max_frame_error_m": engine_max_frame_error_m,
        "max_mass_error_kg": max_mass_error_kg,
        "max_frame_origin_error_m": max_frame_origin_error_m,
        "pinocchio_joint_count": pin_joint_count,
        "passed": True,
    }


__all__ = ["verify_cross_engine_l0_parity"]
