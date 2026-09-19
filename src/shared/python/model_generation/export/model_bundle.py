"""Engine-neutral model bundle generator and relocatable archive serializer.

Implements MV-01: Shared URDF bundles with deterministic float precision,
sidecar validation, and relocatable manifest-bound packaging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.model_generation.builders.urdf_writer import URDFWriter
from src.shared.python.model_generation.core.types import (
    Geometry,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)
from src.shared.python.model_generation.export.bundle_manifest import (
    IncompletePhysicsError,
    ModelBundleManifest,
)
from src.shared.python.motion_matching.full_body_spec import (
    order_directed_tree,
    upper_body_slice,
)

logger = logging.getLogger(__name__)


import warnings


def _compute_origin(value: Any) -> Origin:
    """Extract a rigid URDF Origin from a 4x4 homogeneous transformation matrix."""
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10, rtol=0
        )
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-10, rtol=0)
    ):
        raise ValueError("Invalid rigid transform for URDF origin")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Gimbal lock detected.*", category=UserWarning
        )
        rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
    xyz = (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))
    rpy_tuple = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return Origin(xyz=xyz, rpy=rpy_tuple)


def _order_spec_joints(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Sequence joints with tree structure dependency order."""
    if "upper_body_counts" in spec:
        upper_spec = upper_body_slice(spec)
        upper_joint_names = {j["name"] for j in upper_spec["joints"]}
        upper_ordered = order_directed_tree(upper_spec["joints"])
    else:
        upper_joint_names = {j["name"] for j in spec["joints"]}
        upper_ordered = order_directed_tree(spec["joints"])

    lower_ordered = [j for j in spec["joints"] if j["name"] not in upper_joint_names]
    return list(upper_ordered) + lower_ordered


def _build_primitives_for_joint(
    joint: Mapping[str, Any],
    parent_link: str,
    target_child_link: str,
    coordinates: list[str],
    links: list[Link],
    joints: list[Joint],
) -> None:
    """Decompose joint primitives into scalar 1-DOF joints and intermediate links."""
    parent = parent_link
    placement = joint["parent_to_base"]
    bound = sys.float_info.max

    for primitive in joint["primitives"]:
        kind, name = primitive["primitive"], primitive["coordinate"]
        if kind not in ("Px", "Py", "Pz", "Rx", "Ry", "Rz") or name in coordinates:
            raise ValueError(f"Duplicate or unsupported coordinate {name}")
        child = f"primitive_{len(coordinates)}"
        coordinates.append(name)
        links.append(Link(name=child, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))
        axis_index = "xyz".index(kind[1].lower())
        axis = (float(axis_index == 0), float(axis_index == 1), float(axis_index == 2))
        joints.append(
            Joint(
                name=name,
                parent=parent,
                child=child,
                joint_type=JointType.PRISMATIC
                if kind[0] == "P"
                else JointType.REVOLUTE,
                origin=_compute_origin(placement),
                axis=axis,
                limits=JointLimits(-bound, bound, bound, bound),
                dynamics=JointDynamics(damping=0.0, friction=0.0),
            )
        )
        parent, placement = child, np.eye(4)

    joints.append(
        Joint(
            name=f"fixed_{target_child_link}",
            joint_type=JointType.FIXED,
            parent=parent,
            child=target_child_link,
            origin=_compute_origin(np.linalg.inv(joint["child_to_follower"])),
            dynamics=JointDynamics(damping=0.0, friction=0.0),
        )
    )


@dataclass
class ModelBundle:
    """Relocatable model bundle containing canonical spec, URDF, sidecar, and manifest."""

    manifest: ModelBundleManifest
    urdf_xml: str
    sidecar: dict[str, Any] | None
    raw_spec: bytes
    mesh_assets: dict[str, bytes] = field(default_factory=dict)

    @property
    def is_visual_only(self) -> bool:
        """True if physics is incomplete or sidecar is absent."""
        return (
            self.sidecar is None or self.manifest.incomplete_physics_status is not None
        )

    def require_qualified_dynamics(self) -> None:
        """Enforce that dynamics qualification is allowed for this bundle.

        Raises:
            IncompletePhysicsError: When sidecar is missing or incomplete physics is flagged.
        """
        if self.manifest.requires_sidecar and self.sidecar is None:
            raise IncompletePhysicsError(
                f"Dynamics qualification blocked: sidecar is required but missing ({self.manifest.incomplete_physics_status})"
            )
        if self.manifest.incomplete_physics_status is not None:
            raise IncompletePhysicsError(
                f"Dynamics qualification blocked: {self.manifest.incomplete_physics_status}"
            )

    def validate(self) -> None:
        """Validate internal consistency and cryptographic hash bindings of the bundle."""
        self.manifest.validate()

        if self.raw_spec:
            computed_spec_sha = hashlib.sha256(self.raw_spec).hexdigest()
            if (
                self.manifest.model_sha256
                and computed_spec_sha != self.manifest.model_sha256
            ):
                raise ValueError(
                    "Manifest model_sha256 does not match canonical spec hash"
                )

        computed_urdf_sha = hashlib.sha256(self.urdf_xml.encode("utf-8")).hexdigest()
        if self.manifest.urdf_sha256 and computed_urdf_sha != self.manifest.urdf_sha256:
            raise ValueError("Manifest urdf_sha256 does not match URDF XML hash")

        if self.sidecar is not None:
            sidecar_coords = self.sidecar.get("coordinate_order", [])
            if sidecar_coords != self.manifest.coordinate_order:
                raise ValueError(
                    "Sidecar coordinate_order does not match manifest coordinate_order"
                )

    def save_archive(self, archive_path: Path | str) -> None:
        """Write the bundle to a relocatable zip archive."""
        path = Path(archive_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(self.manifest.to_dict(), indent=2))
            zf.writestr("model.urdf", self.urdf_xml)
            if self.sidecar is not None:
                zf.writestr("sidecar.json", json.dumps(self.sidecar, indent=2))
            if self.raw_spec:
                zf.writestr("spec.json", self.raw_spec)
            for mesh_rel_path, mesh_bytes in self.mesh_assets.items():
                zf.writestr(f"meshes/{mesh_rel_path}", mesh_bytes)

    def extract_to(self, target_dir: Path | str) -> Path:
        """Extract model.urdf and all mesh assets to target directory and return path to model.urdf."""
        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = out_dir / "model.urdf"
        urdf_path.write_text(self.urdf_xml, encoding="utf-8")
        if self.sidecar is not None:
            (out_dir / "sidecar.json").write_text(
                json.dumps(self.sidecar, indent=2), encoding="utf-8"
            )
        if self.raw_spec:
            (out_dir / "spec.json").write_bytes(self.raw_spec)
        (out_dir / "manifest.json").write_text(
            json.dumps(self.manifest.to_dict(), indent=2), encoding="utf-8"
        )
        for mesh_rel, mesh_bytes in self.mesh_assets.items():
            mesh_path = out_dir / "meshes" / mesh_rel
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            mesh_path.write_bytes(mesh_bytes)
        return urdf_path


def _build_body_and_joint_primitives(
    spec: dict[str, Any], body_links: dict[str, str]
) -> tuple[list[Link], list[Joint], list[str]]:
    links: list[Link] = [
        Link(name=link_name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0))
        for link_name in body_links.values()
    ]
    joints: list[Joint] = []
    coordinates: list[str] = []

    for joint in _order_spec_joints(spec):
        _build_primitives_for_joint(
            joint,
            body_links[joint["parent"]],
            body_links[joint["child"]],
            coordinates,
            links,
            joints,
        )

    if len(coordinates) != len(spec["coordinate_order"]) or set(coordinates) != set(
        spec["coordinate_order"]
    ):
        raise ValueError("Coordinate inventory order mismatch with specification")
    return links, joints, coordinates


def _build_solid_visual(
    body_name: str,
    solid_name: str,
    asset_root: Path | None,
    mesh_assets: dict[str, bytes] | None,
) -> tuple[Geometry | None, Material | None]:
    from src.shared.python.body_part_viz.anatomical_visuals import (
        resolve_anatomical_visual,
    )

    binding = resolve_anatomical_visual(body_name, solid_name)
    if binding is None or binding.is_fallback:
        return None, None
    mesh_name = Path(binding.mesh_relative_path).name
    visual_geom = Geometry.mesh(f"meshes/{mesh_name}", scale=binding.scale)
    visual_mat = Material(name=f"mat_{binding.semantic_body}", color=binding.color)
    if mesh_assets is not None:
        cand = (
            (asset_root / mesh_name) if asset_root else Path(binding.mesh_relative_path)
        )
        if cand.is_file():
            mesh_assets[mesh_name] = cand.read_bytes()
    return visual_geom, visual_mat


def _build_solid_links(
    spec: dict[str, Any],
    body_links: dict[str, str],
    links: list[Link],
    joints: list[Joint],
    include_visuals: bool = False,
    mesh_assets: dict[str, bytes] | None = None,
    asset_root: Path | None = None,
) -> dict[str, str]:
    solid_links: dict[str, str] = {}
    for body in spec.get("bodies", []):
        for solid in body.get("solids", []):
            name = f"solid_{len(solid_links)}"
            solid_links[solid["name"]] = name
            com = solid.get("com_m", [0.0, 0.0, 0.0])
            visual_geom, visual_mat = None, None
            if include_visuals:
                visual_geom, visual_mat = _build_solid_visual(
                    body["name"], solid["name"], asset_root, mesh_assets
                )
            links.append(
                Link(
                    name=name,
                    inertia=Inertia.from_matrix(
                        np.asarray(solid["inertia_com_kg_m2"], dtype=float),
                        mass=float(solid["mass_kg"]),
                        center_of_mass=(float(com[0]), float(com[1]), float(com[2])),
                    ),
                    visual_geometry=visual_geom,
                    visual_material=visual_mat,
                )
            )
            joints.append(
                Joint(
                    name=f"fixed_{name}",
                    joint_type=JointType.FIXED,
                    parent=body_links[body["name"]],
                    child=name,
                    origin=_compute_origin(solid["placement"]),
                    dynamics=JointDynamics(0.0, 0.0),
                )
            )
    return solid_links


def _build_frame_and_contact_links(
    spec: dict[str, Any],
    body_links: dict[str, str],
    links: list[Link],
    joints: list[Joint],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    frame_links: dict[str, str] = {}
    for frame in spec.get("frames", []):
        name = f"frame_{len(frame_links)}"
        frame_links[frame["name"]] = name
        links.append(Link(name=name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))
        joints.append(
            Joint(
                name=f"fixed_{name}",
                joint_type=JointType.FIXED,
                parent=body_links[frame["body"]],
                child=name,
                origin=_compute_origin(frame["placement"]),
                dynamics=JointDynamics(0.0, 0.0),
            )
        )

    contact_spheres: dict[str, dict[str, Any]] = {}
    if "contact" in spec and "spheres" in spec["contact"]:
        for sphere in spec["contact"]["spheres"]:
            s_name = sphere["name"]
            link_name = f"contact_{s_name}"
            b_name = sphere["body"]
            radius = float(sphere["radius_m"])
            pos = [float(x) for x in sphere["position_m"]]
            links.append(Link(name=link_name, inertia=Inertia(0.0, 0.0, 0.0, mass=0.0)))
            placement = np.eye(4)
            placement[:3, 3] = pos
            joints.append(
                Joint(
                    name=f"fixed_{link_name}",
                    joint_type=JointType.FIXED,
                    parent=body_links[b_name],
                    child=link_name,
                    origin=_compute_origin(placement),
                    dynamics=JointDynamics(0.0, 0.0),
                )
            )
            contact_spheres[s_name] = {
                "link": link_name,
                "body": b_name,
                "radius_m": radius,
                "position_m": pos,
            }
    return frame_links, contact_spheres


def _create_sidecar_and_manifest(
    spec: dict[str, Any],
    spec_bytes: bytes,
    urdf_xml: str,
    body_links: dict[str, str],
    solid_links: dict[str, str],
    frame_links: dict[str, str],
    contact_spheres: dict[str, dict[str, Any]],
    incomplete_physics_reason: str | None,
) -> tuple[dict[str, Any], ModelBundleManifest]:
    closure = spec.get("closure", {})
    model_sha = hashlib.sha256(spec_bytes).hexdigest()
    urdf_sha = hashlib.sha256(urdf_xml.encode("utf-8")).hexdigest()

    sidecar = {
        "schema_version": 1,
        "requires_sidecar": True,
        "representation": "native-full-body-urdf-v1",
        "qualification": "full-body URDF export; dynamics via explicit continuous KKT adapter",
        "model_sha256": model_sha,
        "urdf_sha256": urdf_sha,
        "coordinate_order": spec["coordinate_order"],
        "body_links": body_links,
        "solid_links": solid_links,
        "frame_links": frame_links,
        "contact_spheres": contact_spheres,
        "closure": closure,
        "gravity_m_s2": spec.get("gravity_m_s2", [0.0, 0.0, -9.80665]),
        "limit_semantics": "restore-unbounded-before-dynamics",
    }
    sidecar_sha = hashlib.sha256(
        json.dumps(sidecar, indent=2).encode("utf-8")
    ).hexdigest()

    manifest = ModelBundleManifest(
        schema_version=1,
        model_sha256=model_sha,
        urdf_sha256=urdf_sha,
        sidecar_sha256=sidecar_sha,
        nq=len(spec["coordinate_order"]),
        nv=len(spec["coordinate_order"]),
        coordinate_order=spec["coordinate_order"],
        body_links=body_links,
        solid_links=solid_links,
        frame_links=frame_links,
        requires_sidecar=True,
        incomplete_physics_status=incomplete_physics_reason,
        capability_losses=["bare_urdf_lacks_closure_contacts_muscles"],
    )
    return sidecar, manifest


def export_model_bundle(
    spec_bytes: bytes,
    incomplete_physics_reason: str | None = None,
    include_visuals: bool = False,
    asset_root: Path | None = None,
) -> ModelBundle:
    """Generate an engine-neutral ModelBundle from specification bytes.

    Preconditions:
        spec_bytes must be valid JSON matching full-body specification.
    """
    if not isinstance(spec_bytes, (bytes, bytearray)):
        raise TypeError("spec_bytes must be bytes")

    spec = json.loads(spec_bytes)
    if not isinstance(spec, dict) or "coordinate_order" not in spec:
        raise ValueError("Invalid full-body specification")

    body_links = {body["name"]: f"body_{i}" for i, body in enumerate(spec["bodies"])}
    if len(body_links) != len(spec["bodies"]) or "world" not in body_links:
        raise ValueError("Invalid specification body inventory")

    mesh_assets: dict[str, bytes] = {}
    links, joints, _ = _build_body_and_joint_primitives(spec, body_links)
    solid_links = _build_solid_links(
        spec, body_links, links, joints, include_visuals, mesh_assets, asset_root
    )
    frame_links, contact_spheres = _build_frame_and_contact_links(
        spec, body_links, links, joints
    )

    xml = URDFWriter(expand_composite_joints=False).write(
        "full_body_golf", links, joints
    )
    sidecar, manifest = _create_sidecar_and_manifest(
        spec,
        spec_bytes,
        xml,
        body_links,
        solid_links,
        frame_links,
        contact_spheres,
        incomplete_physics_reason,
    )

    bundle = ModelBundle(
        manifest=manifest,
        urdf_xml=xml,
        sidecar=sidecar,
        raw_spec=spec_bytes,
        mesh_assets=mesh_assets,
    )
    bundle.validate()
    return bundle


def _assemble_and_validate_bundle(
    manifest: ModelBundleManifest,
    urdf_xml: str,
    sidecar: dict[str, Any] | None,
    raw_spec: bytes,
    mesh_assets: dict[str, bytes],
) -> ModelBundle:
    bundle = ModelBundle(
        manifest=manifest,
        urdf_xml=urdf_xml,
        sidecar=sidecar,
        raw_spec=raw_spec,
        mesh_assets=mesh_assets,
    )
    bundle.validate()
    return bundle


def load_model_bundle(archive_path: Path | str) -> ModelBundle:
    """Load and validate a ModelBundle from a zip archive or directory.

    Preconditions:
        Archive members must not use path traversal (..).
    """
    path = Path(archive_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle path does not exist: {path}")

    if path.is_dir():
        manifest_path = path / "manifest.json"
        urdf_path = path / "model.urdf"
        if not manifest_path.exists() or not urdf_path.exists():
            raise FileNotFoundError(
                f"Model bundle directory missing manifest or urdf: {path}"
            )
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ModelBundleManifest.from_dict(manifest_data)
        urdf_xml = urdf_path.read_text(encoding="utf-8")
        sidecar_path = path / "sidecar.json"
        sidecar = (
            json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar_path.exists()
            else None
        )
        spec_path = path / "spec.json"
        raw_spec = spec_path.read_bytes() if spec_path.exists() else b""
        mesh_assets: dict[str, bytes] = {}
        meshes_dir = path / "meshes"
        if meshes_dir.is_dir():
            for f in meshes_dir.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(meshes_dir).as_posix()
                    mesh_assets[rel] = f.read_bytes()
        return _assemble_and_validate_bundle(
            manifest, urdf_xml, sidecar, raw_spec, mesh_assets
        )

    if path.is_file():
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                norm = PurePosixPath(name.replace("\\", "/"))
                if any(part == ".." for part in norm.parts) or norm.is_absolute():
                    raise ValueError(
                        f"Zip archive contains illegal path traversal: {name}"
                    )

            manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
            manifest = ModelBundleManifest.from_dict(manifest_data)
            urdf_xml = zf.read("model.urdf").decode("utf-8")
            sidecar = None
            if "sidecar.json" in zf.namelist():
                sidecar = json.loads(zf.read("sidecar.json").decode("utf-8"))
            raw_spec = zf.read("spec.json") if "spec.json" in zf.namelist() else b""

            mesh_assets = {}
            for name in zf.namelist():
                if name.startswith("meshes/") and not name.endswith("/"):
                    rel_name = name[len("meshes/") :]
                    mesh_assets[rel_name] = zf.read(name)

            return _assemble_and_validate_bundle(
                manifest, urdf_xml, sidecar, raw_spec, mesh_assets
            )

    raise ValueError(f"Unsupported model bundle path: {path}")
