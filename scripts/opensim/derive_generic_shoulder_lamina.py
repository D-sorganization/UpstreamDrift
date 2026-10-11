"""Reproduce a separately identified 557-muscle generic shoulder candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--right-clavicle", required=True, type=Path)
    parser.add_argument("--right-scapula", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from src.engines.physics_engines.opensim.python.tour_matching import (
        generic_shoulder_lamina as shoulder,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_body_inertia import (
        audit_native_body_inertia,
    )

    manifest = shoulder.ShoulderLaminaManifest(
        args.source,
        shoulder.REVIEWED_557_SOURCE_SHA256,
        args.right_clavicle,
        shoulder.REVIEWED_RIGHT_MESH_SHA256["clavicle"],
        args.right_scapula,
        shoulder.REVIEWED_RIGHT_MESH_SHA256["scapula"],
        shoulder.REGISTRATION,
    )
    candidate = shoulder.build_reviewed_557_shoulder_lamina(manifest, args.output)
    poses = (("pelvis_tilt", 0.0), ("SC_x", 0.01), ("SC_x", -0.01))
    audit = shoulder.audit_reviewed_557_shoulder_candidate(manifest, candidate, poses)
    original = audit_native_body_inertia(args.source, manifest.source_sha256)
    receipt = {
        "schema_version": "generic-shoulder-lamina-candidate/1.0.0",
        "qualification": "sampled-generic-mechanical-hypothesis-only",
        "source_sha256": manifest.source_sha256,
        "model_sha256": candidate.model_sha256,
        "recipe_id": candidate.recipe_id,
        "registration_id": candidate.registration_id,
        "provider_sha256": candidate.lamina_provider_sha256,
        "representation_sha256": candidate.inertia_representation_sha256,
        "provider_worker_sha256": candidate.provider_worker_sha256,
        "adapter_sha256": _sha(Path(shoulder.__file__)),
        "script_sha256": _sha(Path(__file__)),
        "meshes_sha256": candidate.geometry_sha256,
        "bodies": {
            name: {
                "mass_kg": body.mass_kg,
                "center_m": body.center_m.tolist(),
                "inertia_kg_m2": body.inertia_kg_m2.tolist(),
                "area_m2": body.area_m2,
                "boundary_edges": body.boundary_edges,
                "nonmanifold_edges": body.nonmanifold_edges,
                "duplicate_faces": body.duplicate_faces,
            }
            for name, body in candidate.bodies.items()
        },
        "native_audit": asdict(audit),
        "original_source_physical_admitted": original.admitted,
        "original_source_failed_bodies": [
            body.name for body in original.bodies if body.failure_reason
        ],
        "sampled_pose_offsets": poses,
        "open_gates": [
            "anatomical mesh/segment ownership",
            "native acceleration and power over declared chart",
            "passive muscle and tendon laws",
            "native contact and grip",
            "independent head and marker registration",
            "full-horizon capture fit and excitation replay",
        ],
    }
    receipt_path = args.output / "candidate-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(receipt_path)


if __name__ == "__main__":
    main()
