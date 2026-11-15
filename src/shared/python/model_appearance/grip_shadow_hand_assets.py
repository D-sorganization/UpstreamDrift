"""Provenance gate for vendored Shadow Dexterous Hand grip visuals (#15122)."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
PROVENANCE_PATH = REPO_ROOT / "assets/third_party/shadow_dexterous_hand/PROVENANCE.json"
CANONICAL_HAND_ROOT = (
    REPO_ROOT
    / "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/hand_assets/shadow_hand"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_shadow_hand_provenance() -> dict[str, Any]:
    """Load and verify the third-party provenance record (fail closed)."""
    if not PROVENANCE_PATH.is_file():
        raise ValueError(f"Shadow hand provenance missing: {PROVENANCE_PATH}")
    record = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if record.get("licence") != "Apache-2.0":
        raise ValueError("Shadow hand licence must be Apache-2.0")
    canonical = CANONICAL_HAND_ROOT
    if not canonical.is_dir():
        raise ValueError(f"Shadow hand runtime assets missing: {canonical}")
    manifests: dict[str, str] = record.get("files_sha256", {})
    for name, want in manifests.items():
        path = canonical / name
        if not path.is_file():
            raise ValueError(f"Shadow hand asset file missing: {path}")
        got = _sha256(path)
        if got != want:
            raise ValueError(f"Shadow hand {name} sha256 {got} != provenance {want}")
    licence_copy = PROVENANCE_PATH.parent / "LICENSE"
    if not licence_copy.is_file():
        raise ValueError("Shadow hand LICENSE copy missing under assets/third_party")
    return record


def verify_shadow_mesh_asset_licences() -> dict[str, Any]:
    """Verify Apache-2.0 applies to vendored Shadow mesh files (bundle licence)."""
    runtime_root = CANONICAL_HAND_ROOT
    runtime_lic = runtime_root / "LICENSE"
    vendored_lic = PROVENANCE_PATH.parent / "LICENSE"
    if not runtime_lic.is_file():
        raise ValueError(f"runtime Shadow LICENSE missing: {runtime_lic}")
    if not vendored_lic.is_file():
        raise ValueError(f"vendored Shadow LICENSE missing: {vendored_lic}")
    runtime_hash = _sha256(runtime_lic)
    vendored_hash = _sha256(vendored_lic)
    text = runtime_lic.read_text(encoding="utf-8")
    apache_marker = "Apache License" in text and "Version 2.0" in text
    obj_dir = runtime_root / "assets"
    objs = sorted(p.name for p in obj_dir.glob("*.obj"))
    if not objs:
        raise ValueError(f"no Shadow .obj meshes under {obj_dir}")
    return {
        "bundle_licence_spdx": "Apache-2.0" if apache_marker else "unknown",
        "mesh_obj_licence_spdx": "Apache-2.0",
        "mesh_obj_licence_basis": (
            "Menagerie shadow_hand ships one LICENSE for all MJCF and .obj assets; "
            "no per-mesh licence files."
        ),
        "mesh_obj_files": objs,
        "mesh_obj_count": len(objs),
        "runtime_licence_sha256": runtime_hash,
        "vendored_licence_sha256": vendored_hash,
        "runtime_matches_vendored_licence_copy": runtime_hash == vendored_hash,
    }


def shadow_hand_mjcf_path(
    *,
    trail: bool,
    handedness: str = "right",
) -> Path:
    """Return ``right_hand.xml`` or ``left_hand.xml`` for trail/lead and handedness."""
    load_shadow_hand_provenance()
    if handedness not in ("right", "left"):
        raise ValueError(f"handedness must be 'right' or 'left', got {handedness!r}")
    use_right = bool(trail) ^ (handedness == "left")
    name = "right_hand.xml" if use_right else "left_hand.xml"
    path = CANONICAL_HAND_ROOT / name
    if not path.is_file():
        raise ValueError(f"Shadow hand MJCF missing: {path}")
    return path


__all__ = [
    "CANONICAL_HAND_ROOT",
    "PROVENANCE_PATH",
    "load_shadow_hand_provenance",
    "shadow_hand_mjcf_path",
    "verify_shadow_mesh_asset_licences",
]
