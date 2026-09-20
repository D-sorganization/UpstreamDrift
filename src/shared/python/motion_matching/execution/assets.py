"""Resource and asset provider for motion matching execution.

Epic #10508, child #10520.
Resolves reference assets (OpenSim model, native geometry spec, candidates,
C3D captures) without requiring hardcoded relative paths into repository docs.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_FULL_BODY = REPO_ROOT / "docs/development/full_body_models"


def get_native_geometry_spec(path: Path | str | None = None) -> Path:
    """Resolve native geometry specification file path.

    Precondition: if path is supplied, it must exist.
    Resolution order:
    1. Explicit path if provided.
    2. UD_NATIVE_GEOMETRY_SPEC environment variable.
    3. Repository checkout location.
    """
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit native geometry spec not found: {path}")

    env_path = os.environ.get("UD_NATIVE_GEOMETRY_SPEC")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"UD_NATIVE_GEOMETRY_SPEC points to non-existent file: {env_path}"
        )

    default_path = (
        REPO_ROOT
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    )
    if default_path.is_file():
        return default_path

    raise FileNotFoundError(
        "Native geometry spec (native_geometry_spec_9967.json) not found on disk. "
        "Set UD_NATIVE_GEOMETRY_SPEC environment variable or pass --native explicitly."
    )


def get_opensim_model(path: Path | str | None = None) -> Path:
    """Resolve OpenSim humanoid model file path.

    Precondition: if path is supplied, it must exist.
    Resolution order:
    1. Explicit path if provided.
    2. UD_OPENSIM_MODEL environment variable.
    3. Shipped engine package location (src/engines/physics_engines/opensim/models/golf_humanoid.osim).
    """
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit OpenSim model not found: {path}")

    env_path = os.environ.get("UD_OPENSIM_MODEL")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"UD_OPENSIM_MODEL points to non-existent file: {env_path}"
        )

    default_path = (
        REPO_ROOT / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
    )
    if default_path.is_file():
        return default_path

    raise FileNotFoundError(
        "OpenSim model (golf_humanoid.osim) not found on disk. "
        "Set UD_OPENSIM_MODEL environment variable or pass --osim explicitly."
    )


def get_candidate_geometry_spec(path: Path | str | None = None) -> Path:
    """Resolve native candidate geometry spec file path.

    Precondition: if path is supplied, it must exist.
    Resolution order:
    1. Explicit path if provided.
    2. UD_CANDIDATE_GEOMETRY_SPEC environment variable.
    3. Repository checkout location.
    """
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit candidate spec not found: {path}")

    env_path = os.environ.get("UD_CANDIDATE_GEOMETRY_SPEC")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"UD_CANDIDATE_GEOMETRY_SPEC points to non-existent file: {env_path}"
        )

    default_path = (
        DEFAULT_FULL_BODY / "evidence/native_candidates/returned81_candidate.json"
    )
    if default_path.is_file():
        return default_path

    raise FileNotFoundError(
        "Native candidate spec (returned81_candidate.json) not found on disk. "
        "Set UD_CANDIDATE_GEOMETRY_SPEC environment variable or pass --native-candidate explicitly."
    )


def get_capture_c3d(capture_name: str, path: Path | str | None = None) -> Path:
    """Resolve C3D capture file path for capture_name ('driver' or 'iron')."""
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit capture file not found: {path}")

    file_map = {
        "driver": "C3D_TA_Driver.c3d",
        "iron": "C3D_TA_Iron.c3d",
    }
    fname = file_map.get(capture_name, f"C3D_TA_{capture_name.capitalize()}.c3d")

    env_dir = os.environ.get("UD_MOTION_MATCHING_DATA_DIR")
    if env_dir:
        candidate = Path(env_dir) / fname
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Capture file {fname} not found in UD_MOTION_MATCHING_DATA_DIR: {env_dir}"
        )

    default_path = REPO_ROOT / "data" / fname
    if default_path.is_file():
        return default_path

    raise FileNotFoundError(
        f"Capture file {fname} for capture '{capture_name}' not found on disk. "
        "Set UD_MOTION_MATCHING_DATA_DIR environment variable."
    )


def resolve_output_root(explicit_root: Path | str | None = None) -> Path:
    """Resolve base directory for pipeline output artifacts outside docs."""
    if explicit_root is not None:
        return Path(explicit_root)

    env_out = os.environ.get("UD_MOTION_MATCHING_OUT_DIR")
    if env_out:
        return Path(env_out)

    if DEFAULT_FULL_BODY.is_dir():
        return DEFAULT_FULL_BODY

    return Path.cwd() / "artifacts" / "motion_matching"
