"""Spine model asset manifest and local-source SHA-256 admission (#12748).

SimTK downloads need a login and a click-through, so there is no downloader:
assets are owner-provided local files admitted by SHA-256 against the committed
manifest ``src/config/spine_model_assets.json``.  Status is fail-closed:

* ``MISSING``       env var unset, or the model/resource file is absent.
* ``UNPINNED``      the manifest ``model_sha256`` is null (hash observed only).
* ``HASH_MISMATCH`` the model or a pinned resource differs from the manifest.
* ``QUALIFIED``     every pinned hash matches.

``QUALIFIED`` is a byte-identity statement only; it does not assert anatomical
or scientific qualification.  Receipts carry basenames and hashes, never
absolute paths.  The OpenSim-dependent XML audit
(``tour_matching.model_admission.audit_muscle_model_asset``) is attached
best-effort when its runtime is importable; hashing never needs OpenSim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import enum
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Mapping

from src.shared.python.contracts import ensure, require
from src.shared.python.pose_estimation.model_files import sha256_of

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[6] / "src" / "config" / "spine_model_assets.json"
)
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class SpineAssetStatus(enum.Enum):
    """Fail-closed admission outcome."""

    QUALIFIED = "qualified"
    MISSING = "missing"
    HASH_MISMATCH = "hash_mismatch"
    UNPINNED = "unpinned"


@dataclass(frozen=True)
class SpineAssetRequest:
    """Owner-provided model file for a manifest asset id."""

    asset_id: str
    source_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, str) or not self.asset_id.strip():
            raise ValueError("asset_id must be a nonempty string")
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path")


@dataclass(frozen=True)
class SpineResourceReceipt:
    """One resource file: manifest-relative name, observed hash, outcome."""

    relative_name: str
    sha256: str | None
    status: SpineAssetStatus


@dataclass(frozen=True)
class SpineAssetReceipt:
    """Admission record; contains no absolute paths."""

    asset_id: str
    status: SpineAssetStatus
    model_basename: str
    model_sha256: str | None
    resources: tuple[SpineResourceReceipt, ...]
    licence_id: str
    message: str
    xml_audit: str = "not-attempted"
    scientific_status: str = "unqualified"

    def as_dict(self) -> dict[str, Any]:
        """JSON-compatible form (enums rendered by value)."""
        data = asdict(self)
        data["status"] = self.status.value
        for row, src in zip(data["resources"], self.resources, strict=True):
            row["status"] = src.status.value
        return data


def _load_entry(asset_id: str, manifest_path: Path) -> dict[str, Any]:
    require(isinstance(asset_id, str), "asset_id must be a string", asset_id)
    assets = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["assets"]
    if asset_id not in assets:
        raise ValueError(f"unknown spine asset {asset_id!r}; known: {sorted(assets)}")
    return assets[asset_id]


def resolve_spine_asset(
    asset_id: str,
    env: Mapping[str, str] | None = None,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> SpineAssetRequest | None:
    """Return the request named by the entry's env var, or None if unset.

    Postcondition: None means "not provided"; callers must treat it as
    MISSING, never as success.  Raises ValueError for unknown ``asset_id``.
    """
    entry = _load_entry(asset_id, manifest_path)
    environ = os.environ if env is None else env
    value = environ.get(entry["env_var"], "").strip()
    return SpineAssetRequest(asset_id, Path(value)) if value else None


def _check_resources(
    entry: Mapping[str, Any], directory: Path
) -> tuple[SpineResourceReceipt, ...]:
    rows = []
    for name, pinned in sorted(entry["resource_sha256"].items()):
        path = directory / name
        if not path.is_file():
            rows.append(SpineResourceReceipt(name, None, SpineAssetStatus.MISSING))
            continue
        observed = sha256_of(path)
        if pinned is None:
            status = SpineAssetStatus.UNPINNED
        else:
            status = (
                SpineAssetStatus.QUALIFIED
                if observed == pinned
                else SpineAssetStatus.HASH_MISMATCH
            )
        rows.append(SpineResourceReceipt(name, observed, status))
    return tuple(rows)


def _combine(
    model: SpineAssetStatus, resources: tuple[SpineResourceReceipt, ...]
) -> SpineAssetStatus:
    statuses = {model, *(r.status for r in resources)}
    for worst in (
        SpineAssetStatus.MISSING,
        SpineAssetStatus.HASH_MISMATCH,
        SpineAssetStatus.UNPINNED,
    ):
        if worst in statuses:
            return worst
    return SpineAssetStatus.QUALIFIED


def _xml_audit(path: Path, sha256: str) -> str:
    """Best-effort OpenSim-free XML facts via the existing admission audit."""
    try:
        from src.engines.physics_engines.opensim.python.tour_matching.model_admission import (  # noqa: E501
            audit_muscle_model_asset,
        )
    except (ImportError, OSError) as exc:
        return f"unavailable: {type(exc).__name__}"
    try:
        inventory = audit_muscle_model_asset(path, sha256, run_native=False)
    except (ValueError, OSError) as exc:
        return f"unavailable: {type(exc).__name__}"
    return inventory.status


def admit_spine_asset(
    request: SpineAssetRequest,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> SpineAssetReceipt:
    """Hash the local files and compare with the manifest.

    Postconditions: status is QUALIFIED only if the model hash and every
    resource hash are pinned and equal; the receipt holds basenames and
    hashes only.  Never raises for absent files (returns MISSING).
    """
    require(isinstance(request, SpineAssetRequest), "request must be SpineAssetRequest")
    entry = _load_entry(request.asset_id, manifest_path)
    licence = entry["licence_id"]["model"]
    path = request.source_path
    basename = path.name
    if not path.is_file():
        return SpineAssetReceipt(
            request.asset_id,
            SpineAssetStatus.MISSING,
            basename,
            None,
            (),
            licence,
            f"model file {basename!r} not found; set {entry['env_var']}",
        )
    observed = sha256_of(path)
    pinned = entry["model_sha256"]
    ensure(_SHA_RE.fullmatch(observed) is not None, "observed hash malformed")
    if pinned is None:
        model_status = SpineAssetStatus.UNPINNED
    elif observed == pinned:
        model_status = SpineAssetStatus.QUALIFIED
    else:
        model_status = SpineAssetStatus.HASH_MISMATCH
    resources = _check_resources(entry, path.parent)
    status = _combine(model_status, resources)
    audit = (
        _xml_audit(path, observed)
        if status is SpineAssetStatus.QUALIFIED
        else ("not-attempted")
    )
    return SpineAssetReceipt(
        request.asset_id,
        status,
        basename,
        observed,
        resources,
        licence,
        f"{status.value}: byte identity only; scientific status unqualified",
        audit,
    )
