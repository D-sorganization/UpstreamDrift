"""Pinned, hash-verified fetch of the MyoSuite MyoFullBody model assets.

Part of issue #11643 (epic #11642).  The 38 MB of ``myo_sim`` model XML and
meshes are never committed.  This module downloads the upstream repository
archive at a *pinned commit*, extracts only the files listed in the committed
manifest and verifies the sha256 of every one of them before the cache is
published.  A missing or altered file rejects the whole archive, so a tampered
download can never reach ``sys.path``.

Licence record (also in :data:`LICENSE_RECORD` and the manifest): the
``myo_sim`` repository is Apache-2.0; the arm muscle geometry descends from the
MoBL-ARMS model whose upstream terms are BSD-3 *non-commercial*.  The owner
ruled on 2026-10-07 that the application (a free website and free software) is
non-commercial, so those terms are acceptable.  A commercial deployment must
re-clear the arm muscles first.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any
import urllib.request

from src.shared.python.contracts import require

REPOSITORY = "MyoHub/myo_sim"
PINNED_COMMIT = "93b0ca8f4ec90c9899ee7f05fee561e9911da91b"
ARCHIVE_URL = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{PINNED_COMMIT}"
MANIFEST_PATH = Path(__file__).with_name("myofullbody_manifest.json")
CACHE_ENV_VAR = "UPSTREAMDRIFT_MYOFULLBODY_CACHE"
PACKAGE_PREFIX = "myo_sim/"
EXTRA_FILES = ("LICENSE",)
EXPECTED_MUSCLES = 416
DOWNLOAD_TIMEOUT_S = 120.0

LICENSE_RECORD: dict[str, str] = {
    "repository_license": "Apache-2.0 (myo_sim/LICENSE)",
    "arm_muscles": (
        "Derived from MoBL-ARMS (Saul et al. 2015); upstream terms are BSD-3 "
        "non-commercial with citation required."
    ),
    "owner_ruling": (
        "2026-10-07: the application is non-commercial (free website and free "
        "software), so the MoBL-derived arm terms are acceptable."
    ),
    "restriction": "Commercial use requires re-clearing the arm muscle geometry.",
}


class AssetIntegrityError(ValueError):
    """Raised when an archive or cache tree does not match the pinned manifest."""


@dataclass(frozen=True)
class AssetManifest:
    """Pinned source and the expected sha256 of every extracted file."""

    repository: str
    commit: str
    files: dict[str, str]
    license: dict[str, str]

    def __post_init__(self) -> None:
        require(bool(self.files), "manifest must list at least one file")
        for name, digest in self.files.items():
            require(
                not name.startswith("/") and ".." not in Path(name).parts,
                f"unsafe manifest path {name!r}",
            )
            require(
                len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                f"bad sha256 for {name!r}",
            )


def load_manifest(path: Path = MANIFEST_PATH) -> AssetManifest:
    """Read the committed manifest.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the document is malformed.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return AssetManifest(
        repository=str(document["repository"]),
        commit=str(document["commit"]),
        files={str(k): str(v) for k, v in document["files"].items()},
        license=dict(document.get("license", {})),
    )


def default_cache_root() -> Path:
    """Cache directory for the pinned commit (env override, else ``~/.cache``)."""
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "upstreamdrift" / "myofullbody"


def cache_dir(manifest: AssetManifest, root: Path | None = None) -> Path:
    """Directory holding the verified tree of ``manifest.commit``."""
    return (root if root is not None else default_cache_root()) / manifest.commit


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tree(directory: Path, manifest: AssetManifest) -> list[str]:
    """Problems found in ``directory`` versus the manifest (empty when clean).

    Unlisted ``.py`` files are reported because the tree is put on ``sys.path``.
    """
    problems: list[str] = []
    if not directory.is_dir():
        return [f"cache directory {directory} does not exist"]
    for name, expected in manifest.files.items():
        file = directory / name
        if not file.is_file():
            problems.append(f"missing {name}")
        elif sha256_file(file) != expected:
            problems.append(f"sha256 mismatch {name}")
    for file in directory.rglob("*.py"):
        rel = file.relative_to(directory).as_posix()
        if "__pycache__" not in file.parts and rel not in manifest.files:
            problems.append(f"unexpected python file {rel}")
    return problems


def _wanted(member_name: str, commit: str) -> str | None:
    """Manifest-relative path of an archive member, or ``None`` if unwanted."""
    prefix = f"myo_sim-{commit}/"
    if not member_name.startswith(prefix):
        return None
    rel = member_name[len(prefix) :]
    if rel.startswith(PACKAGE_PREFIX) or rel in EXTRA_FILES:
        return rel
    return None


def _read_archive(data: bytes, commit: str) -> dict[str, bytes]:
    """Regular files of the archive that belong to the package (path -> bytes)."""
    found: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive:
                rel = _wanted(member.name, commit)
                if rel is None or not member.isfile():
                    continue
                if ".." in Path(rel).parts or rel.startswith("/"):
                    raise AssetIntegrityError(f"unsafe archive path {member.name!r}")
                handle = archive.extractfile(member)
                if handle is not None:
                    found[rel] = handle.read()
    except tarfile.TarError as error:
        raise AssetIntegrityError(f"unreadable archive: {error}") from error
    return found


def extract_verified(data: bytes, manifest: AssetManifest, destination: Path) -> Path:
    """Verify ``data`` against the manifest and write it to ``destination``.

    Nothing is written unless every manifest file is present with the pinned
    sha256.  The tree appears atomically (temporary sibling, then rename).

    Raises:
        AssetIntegrityError: on a missing, altered or unsafe member.
    """
    found = _read_archive(data, manifest.commit)
    problems = [f"missing {n}" for n in manifest.files if n not in found]
    problems += [
        f"sha256 mismatch {n}"
        for n, expected in manifest.files.items()
        if n in found and sha256_bytes(found[n]) != expected
    ]
    if problems:
        raise AssetIntegrityError("; ".join(problems[:8]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fetch-", dir=destination.parent))
    try:
        for name in manifest.files:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(found[name])
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def download_archive(url: str = ARCHIVE_URL) -> bytes:
    """Download ``url`` (HTTPS only) with a timeout."""
    require(url.startswith("https://"), "archive url must be https")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310
        return bytes(response.read())


def fetch(
    manifest: AssetManifest | None = None,
    *,
    root: Path | None = None,
    download: Any = None,
) -> Path:
    """Ensure the verified MyoFullBody tree exists in the cache; return its path.

    Idempotent: a cache that already verifies is returned without any network
    access.  ``download`` is a zero-argument callable returning the archive
    bytes (default: the pinned GitHub archive).

    Raises:
        AssetIntegrityError: if the downloaded archive does not match the manifest.
    """
    manifest = manifest if manifest is not None else load_manifest()
    target = cache_dir(manifest, root)
    if not verify_tree(target, manifest):
        return target
    fetcher = download if download is not None else download_archive
    return extract_verified(fetcher(), manifest, target)


def cached_tree(
    manifest: AssetManifest | None = None, root: Path | None = None
) -> Path | None:
    """The verified cache directory, or ``None`` when absent or tampered.

    Never touches the network; tests use it to skip cleanly.
    """
    manifest = manifest if manifest is not None else load_manifest()
    target = cache_dir(manifest, root)
    return None if verify_tree(target, manifest) else target


def generate_manifest(data: bytes, commit: str = PINNED_COMMIT) -> dict[str, Any]:
    """Manifest document for an archive (maintainer use when bumping the pin)."""
    files = _read_archive(data, commit)
    require(bool(files), "archive contains no myo_sim files")
    keep = {
        n: sha256_bytes(b)
        for n, b in sorted(files.items())
        if "__pycache__" not in Path(n).parts
    }
    return {
        "schema": "myofullbody-assets/v1",
        "repository": REPOSITORY,
        "commit": commit,
        "archive_url": f"https://codeload.github.com/{REPOSITORY}/tar.gz/{commit}",
        "files": keep,
        "license": LICENSE_RECORD,
    }


def load_myofullbody(directory: Path) -> tuple[Any, Any]:
    """Compose MyoFullBody from a verified tree; returns ``(MjModel, MjData)``.

    Raises:
        ImportError: if ``mujoco`` is unavailable.
    """
    require(directory.is_dir(), f"asset tree {directory} not found")
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)
    loaded = sys.modules.get("myo_sim")
    if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(path):
        for name in [n for n in sys.modules if n.split(".")[0] == "myo_sim"]:
            del sys.modules[name]  # a myo_sim from a different tree
    import myo_sim

    return myo_sim.load("myofullbody")


def model_inventory(model: Any) -> dict[str, Any]:
    """Counts reported in the receipt (bodies, joints, muscles, mass)."""
    import mujoco

    muscles = int(
        sum(
            1
            for i in range(model.nu)
            if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE
        )
    )
    return {
        "bodies": int(model.nbody),
        "joints": int(model.njnt),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "actuators": int(model.nu),
        "muscles": muscles,
        "tendons": int(model.ntendon),
        "mass_kg": float(model.body_mass.sum()),
        "mujoco_version": str(mujoco.__version__),
    }


def asset_receipt(
    directory: Path, manifest: AssetManifest, model: Any | None = None
) -> dict[str, Any]:
    """Receipt: pin, licence, file count/hash digest and model inventory."""
    problems = verify_tree(directory, manifest)
    if problems:
        raise AssetIntegrityError("; ".join(problems[:8]))
    if model is None:
        model, _ = load_myofullbody(directory)
    inventory = model_inventory(model)
    digest = sha256_bytes(json.dumps(manifest.files, sort_keys=True).encode("utf-8"))
    return {
        "schema": "myofullbody-asset-receipt/v1",
        "repository": manifest.repository,
        "commit": manifest.commit,
        "files": len(manifest.files),
        "manifest_files_sha256": digest,
        "license": manifest.license,
        "inventory": inventory,
        "muscle_count_ok": inventory["muscles"] == EXPECTED_MUSCLES,
    }
