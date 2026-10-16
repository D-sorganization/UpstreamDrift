"""Resolve and audit OpenSim body-mesh geometry for the explorer (#12919).

OpenSim only draws a ``<mesh_file>`` it can find on its geometry search path.
Body meshes (``*.vtp``) come from the OpenSim Geometry distribution, which this
repository does not vendor, so a model such as ``golf_humanoid.osim`` loads with
nothing to draw.  This module (a) collects every legitimate geometry source,
(b) registers them through the single existing mechanism
:func:`club_visuals.register_geometry_path`, and (c) audits a model so the
caller can show a clear banner instead of a silent blank viewport.

Qt-free and OpenSim-import-free apart from the registration call.
"""

from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.engines.physics_engines.opensim.python import club_visuals

__all__ = [
    "DEFAULT_MODEL",
    "GEOMETRY_ENV_VAR",
    "GeometryAudit",
    "mesh_resolves",
    "audit_model_geometry",
    "candidate_geometry_dirs",
    "referenced_meshes",
    "register_geometry_dirs",
]

_MESH_FILE_RE = re.compile(r"<mesh_file>([^<]*)</mesh_file>")
GEOMETRY_ENV_VAR = "OPENSIM_GEOMETRY"
MODELS_DIR = club_visuals.OPENSIM_DIR / "models"
# The current anthropometric golfer the reference videos use.
DEFAULT_MODEL = MODELS_DIR / "generated" / "full_body_anthro_driver.osim"


def _package_geometry_dirs() -> list[Path]:
    """``Geometry`` dirs beside the installed ``opensim`` package, if any."""
    try:
        spec = importlib.util.find_spec("opensim")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.origin:
        return []
    package_dir = Path(spec.origin).resolve().parent
    return [package_dir / "Geometry", package_dir.parent / "Geometry"]


def _env_geometry_dirs(environ: Mapping[str, str]) -> list[Path]:
    raw = environ.get(GEOMETRY_ENV_VAR, "")
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()]


def candidate_geometry_dirs(
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Existing geometry dirs: repo, installed opensim package, ``OPENSIM_GEOMETRY``.

    Order is the search priority; duplicates are dropped.
    """
    env = os.environ if environ is None else environ
    sources: list[Path] = [
        MODELS_DIR / "geometry",
        club_visuals.GEOMETRY_DIR,
        *_package_geometry_dirs(),
        *_env_geometry_dirs(env),
    ]
    found: list[Path] = []
    for directory in sources:
        resolved = directory.resolve()
        if resolved.is_dir() and resolved not in found:
            found.append(resolved)
    return tuple(found)


def register_geometry_dirs(dirs: Iterable[Path] | None = None) -> tuple[Path, ...]:
    """Register ``dirs`` (default: all candidates); returns those registered."""
    chosen = candidate_geometry_dirs() if dirs is None else tuple(dirs)
    registered = []
    for directory in chosen:
        if club_visuals.register_geometry_path(directory):
            registered.append(Path(directory))
    return tuple(registered)


def referenced_meshes(model_path: Path | str) -> tuple[str, ...]:
    """Unique ``<mesh_file>`` names a model references, in file order."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"OpenSim model file not found: {path}")
    # Regex, not an XML parser: OpenSim writes ``Class::Name`` tags (for
    # example ``HuntCrossleyForce::ContactParametersSet``) that are not
    # well-formed XML, so ElementTree rejects real models.
    text = path.read_text(encoding="utf-8", errors="replace")
    names: list[str] = []
    for match in _MESH_FILE_RE.finditer(text):
        name = match.group(1).strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def mesh_resolves(name: str, model_dir: Path, search_dirs: Iterable[Path]) -> bool:
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate.is_file()
    roots = [model_dir, model_dir / "Geometry", *search_dirs]
    return any((root / candidate).is_file() for root in roots)


@dataclass(frozen=True)
class GeometryAudit:
    """Which of a model's referenced meshes could be found."""

    model_path: Path
    referenced: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def ok(self) -> bool:
        return not self.missing

    @property
    def banner(self) -> str | None:
        """User-facing message, or ``None`` when every mesh resolved."""
        if self.ok:
            return None
        noun = "mesh" if self.missing_count == 1 else "meshes"
        return f"{self.missing_count} {noun} missing; showing simplified geometry"


def audit_model_geometry(
    model_path: Path | str, search_dirs: Iterable[Path] | None = None
) -> GeometryAudit:
    """Check every mesh the model references against the geometry sources."""
    path = Path(model_path)
    dirs = candidate_geometry_dirs() if search_dirs is None else tuple(search_dirs)
    referenced = referenced_meshes(path)
    missing = tuple(
        name for name in referenced if not mesh_resolves(name, path.parent, dirs)
    )
    return GeometryAudit(path, referenced, missing)
