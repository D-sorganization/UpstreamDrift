"""Spine Viewer web service (SPN-15c, #12815).

Thin, framework-light layer between the API route and the Qt-free viewer core
(:mod:`src.tools.spine_viewer.core`). It adds only receipt discovery and safe
relative-path handling; verification (SHA-256 of the loads artifact), the
fail-closed scientific status and the view model all come from the core.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from src.engines.physics_engines.opensim.python.spine.assets import (
    DEFAULT_MANIFEST_PATH,
    SpineAssetStatus,
    admit_spine_asset,
    resolve_spine_asset,
)
from src.engines.physics_engines.opensim.python.spine.study import SpineStudyRequest

#: Status reported for the synthetic fixture model: always available, never
#: scientifically qualified (its values are invented, ADR-0055).
FIXTURE_ASSET_STATUS = "fixture"
from src.tools.spine_viewer import core

__all__ = [
    "STUDY_ROOT_ENV",
    "list_receipts",
    "list_spine_assets",
    "load_study_payload",
    "parse_study_request",
    "safe_relative_receipt",
    "study_root",
]

STUDY_ROOT_ENV = "UPSTREAMDRIFT_SPINE_STUDY_ROOT"
_RECEIPT_SUFFIX = "_receipt.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def study_root() -> Path:
    """Directory under which spine study receipts may be opened."""
    configured = os.environ.get(STUDY_ROOT_ENV)
    root = Path(configured) if configured else _PROJECT_ROOT / "data" / "spine_studies"
    return root.resolve()


def safe_relative_receipt(receipt: str) -> PurePosixPath:
    """Validate a client-supplied receipt path as a plain relative path.

    Raises:
        TypeError: ``receipt`` is not a string.
        ValueError: empty, absolute, drive-qualified, backslashed or with a
            ``..`` component.
    """
    if not isinstance(receipt, str):
        raise TypeError("receipt must be a string")
    if not receipt.strip():
        raise ValueError("receipt must be a non-empty relative path")
    if "\\" in receipt or "\x00" in receipt:
        raise ValueError("receipt must use '/' separators and no control bytes")
    path = PurePosixPath(receipt)
    if path.is_absolute() or ":" in path.parts[0]:
        raise ValueError("receipt must be relative to the study root")
    if ".." in path.parts:
        raise ValueError("receipt must not contain '..' components")
    return path


def list_receipts(root: Path) -> list[str]:
    """Sorted relative POSIX paths of ``*_receipt.json`` files under ``root``.

    Symlinked files and directories are skipped. A missing root lists nothing.
    """
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    if not root.is_dir():
        return []
    found: list[str] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
        for name in files:
            full = Path(current) / name
            if name.endswith(_RECEIPT_SUFFIX) and not full.is_symlink():
                found.append(full.relative_to(root).as_posix())
    return sorted(found)


def load_study_payload(receipt_path: Path) -> dict[str, object]:
    """Open a receipt through the core (hash-verified) and serialise it.

    Raises:
        OSError: the receipt cannot be read.
        AggregationIntegrityError: artifact missing, tampered or inconsistent.
        ValueError: the receipt is malformed.
    """
    if not isinstance(receipt_path, Path):
        raise TypeError("receipt_path must be a Path")
    return core.view_to_dict(core.open_study(receipt_path))


def list_spine_assets(env: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """List spine model asset entries from manifest with admission status, plus fixture.

    Precondition: ``env`` is None or a mapping of strings.
    Postcondition: Never returns licensed file contents or numerical values from them.
    """
    if env is not None and not isinstance(env, Mapping):
        raise TypeError("env must be a mapping or None")
    manifest_data = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw_assets = manifest_data.get("assets", {})
    results: list[dict[str, Any]] = []

    for asset_id, entry in raw_assets.items():
        if entry.get("model_sha256") is None:
            status = SpineAssetStatus.UNPINNED.value
        else:
            resolved = resolve_spine_asset(asset_id, env)
            if resolved is None:
                status = SpineAssetStatus.MISSING.value
            else:
                receipt = admit_spine_asset(resolved)
                status = receipt.status.value

        results.append(
            {
                "asset_id": asset_id,
                "status": status,
                "simtk_url": entry.get("simtk_url"),
                "env_var": entry.get("env_var"),
                "model_sha256": entry.get("model_sha256"),
                "resource_sha256": entry.get("resource_sha256", {}),
                "licence_id": entry.get("licence_id"),
                "citation": entry.get("citation"),
                "owner_accepted_terms_on": entry.get("owner_accepted_terms_on"),
                "note": entry.get("note"),
                "is_fixture": False,
            }
        )

    # Always-available synthetic fixture model
    results.append(
        {
            "asset_id": "fixture",
            "status": FIXTURE_ASSET_STATUS,
            "simtk_url": None,
            "env_var": None,
            "model_sha256": None,
            "resource_sha256": {},
            "licence_id": None,
            "citation": None,
            "owner_accepted_terms_on": None,
            "note": "Synthetic fixture model; always available, not scientifically qualified",
            "is_fixture": True,
        }
    )

    results.sort(key=lambda x: str(x["asset_id"]))
    return results


def _resolve_study_output_dir(payload: Mapping[str, Any], target_root: Path) -> Path:
    study_id = payload.get("study_id")
    if not isinstance(study_id, str) or not study_id.strip():
        raise ValueError(f"study_id must be a non-empty string: {study_id!r}")

    out_candidate = payload.get("output_dir", study_id)
    if not isinstance(out_candidate, (str, Path)):
        raise TypeError("output_dir must be a string or Path")
    out_path = Path(out_candidate)
    if "\\" in str(out_path) or "\x00" in str(out_path):
        raise ValueError("output_dir must use '/' separators and no control bytes")
    if ".." in out_path.parts:
        raise ValueError("output_dir must not contain '..' components")
    resolved_out = (target_root / out_path).resolve()
    try:
        resolved_out.relative_to(target_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"output_dir must be inside study root ({target_root}): {resolved_out}"
        ) from exc
    return resolved_out


def _resolve_study_model_paths(values: dict[str, Any], target_root: Path) -> None:
    # Normalize fixture asset_id if selected as asset
    if values.get("asset_id") == "fixture":
        values["asset_id"] = None
        values.setdefault("model_id", "fixture")
        values.setdefault("pelvis", "fixture")

    if values.get("model_id") == "fixture" and values.get("model_path") is None:
        fixture_path = target_root / ".fixtures" / "fixture.osim"
        if not fixture_path.is_file():
            from src.engines.physics_engines.opensim.python.spine.fixture_model import (
                FixtureSpec,
                write_fixture_osim,
            )

            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            write_fixture_osim(FixtureSpec(), fixture_path, muscles=True)
        values["model_path"] = fixture_path
        values.setdefault("pelvis", "fixture")

    if values.get("model_id") is None and values.get("asset_id"):
        values["model_id"] = values["asset_id"]

    if values.get("golf_model_path") is None:
        default_golf = (
            _PROJECT_ROOT
            / "src"
            / "engines"
            / "physics_engines"
            / "opensim"
            / "models"
            / "golf_humanoid.osim"
        )
        if default_golf.is_file():
            values["golf_model_path"] = default_golf

    # Resolve relative paths against _PROJECT_ROOT
    for name in ("motion_path", "golf_model_path", "model_path", "external_loads"):
        val = values.get(name)
        if val is not None:
            if not isinstance(val, (str, Path)):
                raise TypeError(f"{name} must be a string or Path")
            p = Path(val)
            values[name] = (
                (_PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
            )


def _validate_study_inputs(request: SpineStudyRequest) -> None:
    # Precondition: check required input files exist on disk
    for label, path in (
        ("motion", request.motion_path),
        ("golf model", request.golf_model_path),
        ("segmental model", request.model_path),
        ("external loads", request.external_loads),
    ):
        if path is not None and not path.is_file():
            raise ValueError(f"{label} file not found: {path.name}")

    if request.asset_id is not None:
        try:
            resolve_spine_asset(request.asset_id)
        except ValueError as exc:
            raise ValueError(f"unknown spine asset: {exc}") from exc


def parse_study_request(
    payload: Mapping[str, Any], root: Path | None = None
) -> SpineStudyRequest:
    """Validate and build a :class:`SpineStudyRequest` from a client payload.

    Args:
        payload: JSON-compatible mapping with study request attributes.
        root: Directory under which output must reside (default ``study_root()``).

    Returns:
        A validated :class:`SpineStudyRequest` targeting an output directory inside ``root``.

    Raises:
        TypeError: ``payload`` is not a mapping or ``root`` is not a Path.
        ValueError: missing required fields, bad identifiers, path escapes,
            missing inputs, or invalid request options.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    target_root = study_root() if root is None else root
    if not isinstance(target_root, Path):
        raise TypeError("root must be a Path")

    resolved_out = _resolve_study_output_dir(payload, target_root)

    values = dict(payload)
    values["output_dir"] = resolved_out

    _resolve_study_model_paths(values, target_root)

    try:
        request = SpineStudyRequest(**values)
    except TypeError as exc:
        raise ValueError(f"invalid request parameters: {exc}") from exc

    _validate_study_inputs(request)

    return request
