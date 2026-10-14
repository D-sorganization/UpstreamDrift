"""Explicit generic shoulder lamina model, separate from its failed source.

The four replacement properties are a declared mechanical hypothesis. Mesh
placement is not asserted to recover unavailable original BUET geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, cast

from defusedxml import ElementTree
import numpy as np
from numpy.typing import NDArray
from src.shared.python.engine_core.opensim_runtime import import_opensim

RECIPE = "generic-shoulder-triangle-multiset-lamina/1.0.0"
REGISTRATION = "right-identity-left-local-z-reflection/1.0.0"
_PARTS = ("clavicle", "scapula")
_REFLECTION = np.diag([1.0, 1.0, -1.0])
REVIEWED_557_SOURCE_SHA256 = (
    "453e09c4e42dcc3f4b74a3b2efeff063e96c3820b3eee1efe95eef249579010d"
)
REVIEWED_RIGHT_MESH_SHA256 = {
    "clavicle": "2846d6daaa33b694d9eb59a02ea4716c43ac5d64c420a44909333e37d1223ff8",
    "scapula": "a62232c10aaa3b7c66a794ca947124fd59fffa3dea51e0c8457da50e16742d6b",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _provider_worker() -> tuple[Path, Path]:
    root = _repository_root()
    worker = root / "scripts/opensim/triangle_lamina_provider_worker.py"
    tools_src = root / "vendor/ud-tools/src"
    if not worker.is_file() or not (tools_src / "shared/python").is_dir():
        raise ValueError("pinned Tools triangle-lamina provider is unavailable")
    return worker, tools_src


def _call_provider_worker(input_path: Path | None = None) -> dict[str, Any]:
    worker, tools_src = _provider_worker()
    command = [sys.executable, "-I", str(worker), "--tools-src", str(tools_src)]
    if input_path is not None:
        command.extend(("--input", str(input_path)))
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            check=False,
            cwd=tools_src.parent,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("pinned Tools triangle-lamina worker failed") from exc
    if process.returncode != 0:
        raise ValueError(
            "pinned Tools triangle-lamina worker failed: "
            + process.stderr[-2000:].strip()
        )
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("pinned Tools worker returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("pinned Tools worker returned an invalid response")
    if result.get("worker_sha256") != _sha(worker):
        raise ValueError("pinned Tools worker source changed during execution")
    expected_paths = {
        "provider_path": tools_src
        / "shared/python/humanoid_character_builder/mesh/triangle_lamina.py",
        "representation_path": tools_src
        / "shared/python/humanoid_character_builder/mesh/inertia_calculator.py",
    }
    for path_field, expected_path in expected_paths.items():
        if result.get(path_field) != str(expected_path.resolve()):
            raise ValueError("pinned Tools provider resolved outside its source")
    for hash_field, path_field in (
        ("provider_sha256", "provider_path"),
        ("representation_sha256", "representation_path"),
    ):
        if result.get(hash_field) != _sha(Path(result[path_field])):
            raise ValueError("pinned Tools provider source changed during execution")
    return result


@dataclass(frozen=True, slots=True)
class ShoulderLaminaManifest:
    """Exact source and right-side mesh bytes for one prefit hypothesis."""

    source_path: Path
    source_sha256: str
    clavicle_mesh_path: Path
    clavicle_mesh_sha256: str
    scapula_mesh_path: Path
    scapula_mesh_sha256: str
    registration_id: str


@dataclass(frozen=True, slots=True)
class DerivedShoulderBody:
    """Body-frame COM/tensor and triangle-multiset diagnostics."""

    mass_kg: float
    center_m: NDArray[np.float64]
    inertia_kg_m2: NDArray[np.float64]
    mesh_sha256: str
    area_m2: float
    boundary_edges: int
    nonmanifold_edges: int
    duplicate_faces: int


@dataclass(frozen=True, slots=True)
class ShoulderLaminaCandidate:
    """Immutable candidate identity and the four derived property observations."""

    model_path: Path
    model_sha256: str
    source_sha256: str
    lamina_provider_sha256: str
    inertia_representation_sha256: str
    provider_worker_sha256: str
    recipe_id: str
    registration_id: str
    bodies: dict[str, DerivedShoulderBody]
    geometry_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class NativeShoulderCandidateAudit:
    """Necessary native physical and sampled kinematic invariance, not anatomy."""

    source_sha256: str
    model_sha256: str
    physical_audit_sha256: str
    pose_count: int
    body_count: int
    max_transform_error: float
    max_constraint_error_difference: float
    opensim_version: str


def _read_vtp(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Accept only explicit ASCII triangulated PolyData, without repair."""
    root = ElementTree.fromstring(path.read_bytes())
    piece = root.find("./PolyData/Piece")
    if root.get("type") != "PolyData" or piece is None:
        raise ValueError("mesh must be VTK PolyData")
    if any(
        int(piece.get(name, "0")) != 0
        for name in ("NumberOfVerts", "NumberOfLines", "NumberOfStrips")
    ):
        raise ValueError("mesh contains nontriangle cells")
    points = piece.find("./Points/DataArray")
    connectivity = piece.find("./Polys/DataArray[@Name='connectivity']")
    offsets = piece.find("./Polys/DataArray[@Name='offsets']")
    if points is None or connectivity is None or offsets is None:
        raise ValueError("mesh points or polygon indices are missing")
    if any(node.get("format") != "ascii" for node in (points, connectivity, offsets)):
        raise ValueError("mesh must use explicit ASCII arrays")
    vertices = np.fromstring(points.text or "", sep=" ", dtype=np.float64)
    indices = np.fromstring(connectivity.text or "", sep=" ", dtype=np.int64)
    ends = np.fromstring(offsets.text or "", sep=" ", dtype=np.int64)
    if (
        vertices.size != 3 * int(piece.get("NumberOfPoints", "-1"))
        or indices.size != 3 * int(piece.get("NumberOfPolys", "-1"))
        or not np.array_equal(ends, np.arange(3, indices.size + 1, 3))
    ):
        raise ValueError("mesh must contain only complete triangles")
    return vertices.reshape(-1, 3), indices.reshape(-1, 3)


def _write_reflected_vtp(
    source: Path,
    destination: Path,
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
) -> None:
    """Reflect vertices, reverse winding, and leave native frames untouched."""
    root = ElementTree.fromstring(source.read_bytes())
    piece = root.find("./PolyData/Piece")
    assert piece is not None
    points = piece.find("./Points/DataArray")
    connectivity = piece.find("./Polys/DataArray[@Name='connectivity']")
    assert points is not None and connectivity is not None
    reflected = vertices @ _REFLECTION
    points.text = " ".join(format(float(value), ".17g") for value in reflected.flat)
    connectivity.text = " ".join(str(int(value)) for value in faces[:, [0, 2, 1]].flat)
    normals = piece.find("./PointData/DataArray[@Name='Normals']")
    if normals is not None:
        values = np.fromstring(normals.text or "", sep=" ", dtype=np.float64)
        if normals.get("format") != "ascii" or values.size != vertices.size:
            raise ValueError("source mesh normals are malformed")
        reflected_normals = values.reshape(-1, 3) @ _REFLECTION
        normals.text = " ".join(
            format(float(value), ".17g") for value in reflected_normals.flat
        )
    destination.write_bytes(ElementTree.tostring(root, encoding="utf-8"))


def _source_bodies(root: ElementTree.Element) -> dict[str, ElementTree.Element]:
    model = root.find("./Model")
    if model is None or model.findtext("length_units") != "meters":
        raise ValueError("source length units must be meters")
    rows = {}
    for node in root.findall("./Model/BodySet/objects/Body"):
        name = node.get("name")
        if name in rows:
            raise ValueError("duplicate source body")
        rows[name] = node
    if {f"{part}_{side}" for part in _PARTS for side in "rl"} - rows.keys():
        raise ValueError("four shoulder bodies are required")
    return rows


def _shoulder_path_points(
    root: ElementTree.Element, body_name: str
) -> Counter[tuple[float, float, float]]:
    locations: Counter[tuple[float, float, float]] = Counter()
    for point in root.iter():
        if not point.tag.endswith("PathPoint"):
            continue
        if point.findtext("socket_parent_frame") != f"/bodyset/{body_name}":
            continue
        values = tuple(float(value) for value in point.findtext("location", "").split())
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError("source shoulder path point is malformed")
        locations[cast(tuple[float, float, float], values)] += 1
    return locations


def admit_reviewed_557_manifest(manifest: ShoulderLaminaManifest) -> None:
    """Bind this generic registration to one actual source and exact right assets.

    This is provenance admission, not recovered original BUET geometry.
    """
    if manifest.source_sha256 != REVIEWED_557_SOURCE_SHA256:
        raise ValueError("reviewed 557 source identity differs")
    for part in _PARTS:
        expected = REVIEWED_RIGHT_MESH_SHA256[part]
        declared = getattr(manifest, f"{part}_mesh_sha256")
        if declared != expected:
            raise ValueError("reviewed right mesh identity differs")
    if _sha(manifest.source_path) != manifest.source_sha256:
        raise ValueError("reviewed 557 source bytes differ")
    root = ElementTree.fromstring(manifest.source_path.read_bytes())
    for part, count in (("clavicle", 8), ("scapula", 81)):
        right = _shoulder_path_points(root, f"{part}_r")
        left = _shoulder_path_points(root, f"{part}_l")
        reflected = Counter((x, y, -z) for (x, y, z) in right.elements())
        if sum(right.values()) != count or left != reflected:
            raise ValueError("reviewed bilateral path-point mirror differs")


def _semantic(node: ElementTree.Element) -> tuple[object, ...]:
    return (
        node.tag,
        tuple(sorted(node.attrib.items())),
        (node.text or "").strip(),
        tuple(_semantic(child) for child in node),
    )


def _assert_allowed_xml_changes(source_raw: bytes, candidate_raw: bytes) -> None:
    """Compare the whole XML tree after undoing only the four declared edits."""
    source = ElementTree.fromstring(source_raw)
    candidate = ElementTree.fromstring(candidate_raw)
    originals, derived = _source_bodies(source), _source_bodies(candidate)
    for part in _PARTS:
        for side in "rl":
            name = f"{part}_{side}"
            original, body = originals[name], derived[name]
            for field in ("mass_center", "inertia", "attached_geometry/Mesh/mesh_file"):
                before, after = original.find(field), body.find(field)
                if before is None or after is None:
                    raise ValueError("declared shoulder property is missing")
                after.text = before.text
    if _semantic(source) != _semantic(candidate):
        raise ValueError("candidate changes an undeclared source property")


def _validate_geometry(body: ElementTree.Element, filename: str) -> None:
    meshes = body.findall("./attached_geometry/Mesh")
    if len(meshes) != 1:
        raise ValueError("shoulder body must have exactly one declared mesh")
    mesh = meshes[0]
    if (
        mesh.findtext("socket_frame") != ".."
        or mesh.findtext("scale_factors") != "1 1 1"
        or mesh.findtext("mesh_file") != filename
    ):
        raise ValueError("source shoulder mesh frame, scale or filename differs")


def _derive_body(
    mass: float,
    vertices: NDArray[np.float64],
    faces: NDArray[np.int64],
    mesh_sha: str,
    provider_identity: dict[str, Any],
) -> DerivedShoulderBody:
    with TemporaryDirectory(prefix="ud-tools-lamina-") as directory:
        input_path = Path(directory) / "triangles.npz"
        np.savez(input_path, vertices=vertices, faces=faces, mass=np.float64(mass))
        response = _call_provider_worker(input_path)
    if any(
        response.get(field) != provider_identity.get(field)
        for field in (
            "provider_sha256",
            "representation_sha256",
            "worker_sha256",
        )
    ):
        raise ValueError("pinned Tools provider changed during candidate derivation")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ValueError("pinned Tools worker omitted lamina moments")
    return DerivedShoulderBody(
        mass_kg=mass,
        center_m=np.asarray(result["center_m"], dtype=np.float64),
        inertia_kg_m2=np.asarray(result["inertia_kg_m2"], dtype=np.float64),
        mesh_sha256=mesh_sha,
        area_m2=float(result["area_m2"]),
        boundary_edges=int(result["boundary_edges"]),
        nonmanifold_edges=int(result["nonmanifold_edges"]),
        duplicate_faces=int(result["duplicate_faces"]),
    )


def _set_properties(body: ElementTree.Element, result: DerivedShoulderBody) -> None:
    center = body.find("mass_center")
    tensor = body.find("inertia")
    if center is None or tensor is None:
        raise ValueError("source shoulder mass properties are missing")
    values = result.inertia_kg_m2
    center.text = " ".join(format(float(v), ".17g") for v in result.center_m)
    tensor.text = " ".join(
        format(float(v), ".17g")
        for v in (
            values[0, 0],
            values[1, 1],
            values[2, 2],
            values[0, 1],
            values[0, 2],
            values[1, 2],
        )
    )


def build_generic_shoulder_lamina(
    manifest: ShoulderLaminaManifest, output_dir: Path
) -> ShoulderLaminaCandidate:
    """Produce a new model only after exact prefit source/mesh admission.

    Right geometry is placed with the declared identity body-frame hypothesis.
    Left geometry reflects local vertices, not any native rotation. The
    triangle-multiset law counts duplicate and partially overlapping faces.
    This function cannot qualify anatomy, muscle physiology, or captured motion.
    """
    if manifest.registration_id != REGISTRATION:
        raise ValueError("unreviewed shoulder registration policy")
    if _sha(manifest.source_path) != manifest.source_sha256:
        raise ValueError("source model hash mismatch")
    meshes = {
        "clavicle": (manifest.clavicle_mesh_path, manifest.clavicle_mesh_sha256),
        "scapula": (manifest.scapula_mesh_path, manifest.scapula_mesh_sha256),
    }
    for path, expected in meshes.values():
        if _sha(path) != expected:
            raise ValueError("mesh source hash mismatch")
    root = ElementTree.fromstring(manifest.source_path.read_bytes())
    bodies = _source_bodies(root)
    derived: dict[str, DerivedShoulderBody] = {}
    parsed: dict[str, tuple[NDArray[np.float64], NDArray[np.int64]]] = {}
    provider_identity = _call_provider_worker()
    for part, (path, expected) in meshes.items():
        vertices, faces = _read_vtp(path)
        parsed[part] = vertices, faces
        for side in "rl":
            name = f"{part}_{side}"
            _validate_geometry(bodies[name], f"{part}{'_l' if side == 'l' else ''}.vtp")
            mass = float(bodies[name].findtext("mass", "nan"))
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("shoulder body mass must be positive and finite")
            points = vertices if side == "r" else vertices @ _REFLECTION
            derived[name] = _derive_body(
                mass, points, faces, expected, provider_identity
            )
            _set_properties(bodies[name], derived[name])
            mesh_file = bodies[name].find("./attached_geometry/Mesh/mesh_file")
            assert mesh_file is not None
            mesh_file.text = f"generic_{name}.vtp"
    if _sha(manifest.source_path) != manifest.source_sha256:
        raise ValueError("source model changed during derivation")
    if any(_sha(path) != expected for path, expected in meshes.values()):
        raise ValueError("mesh source changed during derivation")
    if output_dir.exists():
        raise ValueError("candidate output directory must be new")
    geometry = output_dir / "Geometry"
    geometry.mkdir(parents=True)
    for part, (path, _) in meshes.items():
        (geometry / f"generic_{part}_r.vtp").write_bytes(path.read_bytes())
        vertices, faces = parsed[part]
        _write_reflected_vtp(path, geometry / f"generic_{part}_l.vtp", vertices, faces)
    target = output_dir / "generic-shoulder-lamina.osim"
    target.write_bytes(ElementTree.tostring(root, encoding="utf-8"))
    _assert_allowed_xml_changes(manifest.source_path.read_bytes(), target.read_bytes())
    if _call_provider_worker() != provider_identity:
        raise ValueError("pinned Tools provider changed during candidate derivation")
    geometry_hashes = {path.name: _sha(path) for path in sorted(geometry.iterdir())}
    return ShoulderLaminaCandidate(
        model_path=target,
        model_sha256=_sha(target),
        source_sha256=manifest.source_sha256,
        lamina_provider_sha256=provider_identity["provider_sha256"],
        inertia_representation_sha256=provider_identity["representation_sha256"],
        provider_worker_sha256=provider_identity["worker_sha256"],
        recipe_id=RECIPE,
        registration_id=REGISTRATION,
        bodies=derived,
        geometry_sha256=geometry_hashes,
    )


def build_reviewed_557_shoulder_lamina(
    manifest: ShoulderLaminaManifest, output_dir: Path
) -> ShoulderLaminaCandidate:
    """Apply the one reviewed final-source and author-right byte profile."""
    admit_reviewed_557_manifest(manifest)
    return build_generic_shoulder_lamina(manifest, output_dir)


def _native_pose_rows(model: Any, state: Any) -> tuple[np.ndarray, np.ndarray]:
    """Collect finite body transforms and native position/velocity residuals."""
    model.realizeVelocity(state)
    frames: list[float] = []
    for index in range(model.getBodySet().getSize()):
        body = model.getBodySet().get(index)
        transform = body.getTransformInGround(state)
        frames.extend(float(transform.p().get(i)) for i in range(3))
        frames.extend(
            float(transform.R().get(i, j)) for i in range(3) for j in range(3)
        )
    residuals = [float(state.getQErr().get(i)) for i in range(state.getQErr().size())]
    residuals.extend(
        float(state.getUErr().get(i)) for i in range(state.getUErr().size())
    )
    positions = np.asarray(frames, dtype=np.float64)
    errors = np.asarray(residuals, dtype=np.float64)
    if not np.isfinite(positions).all() or not np.isfinite(errors).all():
        raise ValueError("native sampled frame or constraint observation is nonfinite")
    return positions, errors


def _admit_candidate_artifacts(
    manifest: ShoulderLaminaManifest,
    candidate: ShoulderLaminaCandidate,
) -> None:
    """Recheck exact source, provider, visual geometry and allowed XML diff."""
    admit_reviewed_557_manifest(manifest)
    if _sha(candidate.model_path) != candidate.model_sha256:
        raise ValueError("candidate identity differs")
    if (
        candidate.source_sha256 != manifest.source_sha256
        or candidate.recipe_id != RECIPE
        or candidate.registration_id != REGISTRATION
    ):
        raise ValueError("candidate source or recipe identity differs")
    provider_identity = _call_provider_worker()
    if provider_identity["provider_sha256"] != candidate.lamina_provider_sha256:
        raise ValueError("triangle lamina provider changed")
    if (
        provider_identity["representation_sha256"]
        != candidate.inertia_representation_sha256
    ):
        raise ValueError("triangle lamina inertia representation changed")
    if provider_identity["worker_sha256"] != candidate.provider_worker_sha256:
        raise ValueError("triangle lamina provider worker changed")
    expected_geometry = {
        f"generic_{part}_{side}.vtp" for part in _PARTS for side in "rl"
    }
    if set(candidate.geometry_sha256) != expected_geometry:
        raise ValueError("derived geometry inventory differs")
    for name, expected in candidate.geometry_sha256.items():
        if _sha(candidate.model_path.parent / "Geometry" / name) != expected:
            raise ValueError("derived geometry hash mismatch")
    for part in _PARTS:
        original_path = getattr(manifest, f"{part}_mesh_path")
        source_vertices, source_faces = _read_vtp(original_path)
        right_path = candidate.model_path.parent / "Geometry" / f"generic_{part}_r.vtp"
        left_path = candidate.model_path.parent / "Geometry" / f"generic_{part}_l.vtp"
        if right_path.read_bytes() != original_path.read_bytes():
            raise ValueError("derived right mesh differs from declared source")
        left_vertices, left_faces = _read_vtp(left_path)
        if not np.array_equal(
            left_vertices, source_vertices @ _REFLECTION
        ) or not np.array_equal(left_faces, source_faces[:, [0, 2, 1]]):
            raise ValueError("derived left mesh is not the declared local reflection")
    _assert_allowed_xml_changes(
        manifest.source_path.read_bytes(), candidate.model_path.read_bytes()
    )


def _compare_native_kinematics(
    source: Any, derived: Any, pose_offsets: tuple[tuple[str, float], ...]
) -> tuple[float, float]:
    """Require sampled native FK and QErr/UErr equality at declared poses."""
    maximum_frame = maximum_constraint = 0.0
    for coordinate, offset in pose_offsets:
        if not np.isfinite(offset):
            raise ValueError("pose offset must be finite")
        original_state, derived_state = source.initSystem(), derived.initSystem()
        for model, state in ((source, original_state), (derived, derived_state)):
            joint = model.getCoordinateSet().get(coordinate)
            joint.setValue(state, joint.getValue(state) + offset, False)
        before_frames, before_errors = _native_pose_rows(source, original_state)
        after_frames, after_errors = _native_pose_rows(derived, derived_state)
        if (
            before_frames.shape != after_frames.shape
            or before_errors.shape != after_errors.shape
        ):
            raise ValueError("native frame or constraint dimensions changed")
        maximum_frame = max(
            maximum_frame, float(np.max(np.abs(before_frames - after_frames)))
        )
        maximum_constraint = max(
            maximum_constraint,
            float(np.max(np.abs(before_errors - after_errors)))
            if len(before_errors)
            else 0.0,
        )
    if maximum_frame > 1e-12 or maximum_constraint > 1e-12:
        raise ValueError("candidate native FK or constraints differ from source")
    return maximum_frame, maximum_constraint


def audit_reviewed_557_shoulder_candidate(
    manifest: ShoulderLaminaManifest,
    candidate: ShoulderLaminaCandidate,
    pose_offsets: tuple[tuple[str, float], ...],
) -> NativeShoulderCandidateAudit:
    """Observe necessary physical and sampled kinematic conditions.

    Pose offsets are relative to each native initialized coordinate. This
    cannot qualify anatomy, dynamics over a chart, or full-swing tracking.
    """
    osim = import_opensim()

    from .native_body_inertia import audit_native_body_inertia

    if not pose_offsets:
        raise ValueError("native pose inventory is empty")
    _admit_candidate_artifacts(manifest, candidate)
    physical = audit_native_body_inertia(candidate.model_path, candidate.model_sha256)
    physical.require_admitted()
    source, derived = (
        osim.Model(str(manifest.source_path)),
        osim.Model(str(candidate.model_path)),
    )
    source_names = tuple(
        source.getBodySet().get(i).getName()
        for i in range(source.getBodySet().getSize())
    )
    derived_names = tuple(
        derived.getBodySet().get(i).getName()
        for i in range(derived.getBodySet().getSize())
    )
    if source_names != derived_names:
        raise ValueError("native body order changed")
    maximum_frame, maximum_constraint = _compare_native_kinematics(
        source, derived, pose_offsets
    )
    if _sha(manifest.source_path) != manifest.source_sha256:
        raise ValueError("source model changed during native comparison")
    physical_sha = hashlib.sha256(
        json.dumps(asdict(physical), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return NativeShoulderCandidateAudit(
        manifest.source_sha256,
        candidate.model_sha256,
        physical_sha,
        len(pose_offsets),
        len(source_names),
        maximum_frame,
        maximum_constraint,
        osim.GetVersionAndDate(),
    )
