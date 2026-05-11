"""Snapshot a sibling biomech repo's models tree into ``vendor/biomech-models/``.

CLI:

    python scripts/update_biomech_vendor.py --repo MuJoCo_Models --ref v1.4.0

Implementation: shallow-clone the sibling repo into a temp directory, copy
the manifest plus the declared ``models_root`` (or a conventional models
tree if the manifest is missing) into
``vendor/biomech-models/<RepoName>/``, then delete the temp clone.

Stdlib + the system ``git`` executable only. No third-party deps.

See ``docs/adr/0014-shared-biomech-models.md`` (UpstreamDrift#5184).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("update_biomech_vendor")


KNOWN_SIBLINGS: dict[str, str] = {
    "MuJoCo_Models": "https://github.com/D-sorganization/MuJoCo_Models",
    "Drake_Models": "https://github.com/D-sorganization/Drake_Models",
    "Pinocchio_Models": "https://github.com/D-sorganization/Pinocchio_Models",
    "OpenSim_Models": "https://github.com/D-sorganization/OpenSim_Models",
    "Movement-Optimizer": "https://github.com/D-sorganization/Movement-Optimizer",
}


@dataclass(frozen=True)
class VendorOptions:
    """Resolved command-line options for one snapshot operation."""

    repo: str
    ref: str
    url: str
    vendor_root: Path


def _repo_root() -> Path:
    """Return the UpstreamDrift repo root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None) -> VendorOptions:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo",
        required=True,
        choices=sorted(KNOWN_SIBLINGS),
        help="Sibling repo name (e.g. MuJoCo_Models).",
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Git ref (tag or branch) to snapshot, e.g. v1.4.0.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override the upstream URL (defaults to the public GitHub URL).",
    )
    parser.add_argument(
        "--vendor-root",
        default=None,
        help="Override the destination root (defaults to vendor/biomech-models/).",
    )
    args = parser.parse_args(argv)

    vendor_root = (
        Path(args.vendor_root).resolve()
        if args.vendor_root is not None
        else _repo_root() / "vendor" / "biomech-models"
    )
    url = args.url or KNOWN_SIBLINGS[args.repo]
    return VendorOptions(
        repo=args.repo,
        ref=args.ref,
        url=url,
        vendor_root=vendor_root,
    )


def _shallow_clone(url: str, ref: str, target: Path) -> None:
    """Shallow-clone ``url`` at ``ref`` into ``target``."""
    logger.info("Cloning %s @ %s", url, ref)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            ref,
            url,
            str(target),
        ],
        check=True,
    )


def _models_root_from_manifest(checkout: Path) -> Path | None:
    """Best-effort: read ``model_pack.yaml`` / ``tool_pack.yaml`` for models_root.

    Returns ``None`` if PyYAML is missing or the manifest does not declare
    a ``models_root`` (tool packs typically don't).
    """
    for manifest_name in ("model_pack.yaml", "tool_pack.yaml"):
        manifest_path = checkout / manifest_name
        if not manifest_path.is_file():
            continue
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("PyYAML missing — skipping manifest parse")
            return None
        try:
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not parse %s: %s", manifest_path, exc)
            return None
        if isinstance(data, dict):
            declared = data.get("models_root")
            if isinstance(declared, str) and declared.strip():
                return (checkout / declared).resolve()
    return None


def _guess_models_root(checkout: Path, repo: str) -> Path | None:
    """Locate a sensible models tree when the manifest is absent."""
    candidates: list[Path] = [checkout / "models"]
    # Convention used by MuJoCo_Models / Drake_Models / etc.
    pkg_guess = repo.lower().replace("-", "_")
    candidates.extend(
        [
            checkout / "src" / pkg_guess / "exercises",
            checkout / "src" / pkg_guess / "models",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    logger.info("Copied %s -> %s", source, destination)


def _copy_manifest(checkout: Path, destination_root: Path) -> None:
    for manifest_name in ("model_pack.yaml", "tool_pack.yaml"):
        manifest_path = checkout / manifest_name
        if manifest_path.is_file():
            shutil.copy2(manifest_path, destination_root / manifest_name)
            logger.info("Copied %s", manifest_path.name)


def _write_provenance(destination_root: Path, options: VendorOptions) -> None:
    """Write a small marker file documenting how the snapshot was produced."""
    marker = destination_root / "VENDOR_PROVENANCE.txt"
    marker.write_text(
        "\n".join(
            [
                f"repo: {options.repo}",
                f"ref: {options.ref}",
                f"url: {options.url}",
                "produced by: scripts/update_biomech_vendor.py",
                "",
            ]
        ),
        encoding="utf-8",
    )


def snapshot(options: VendorOptions) -> Path:
    """Materialise a vendor snapshot for one sibling repo.

    Returns the destination directory containing the snapshot. The directory
    is overwritten on each invocation.
    """
    destination_root = options.vendor_root / options.repo
    with tempfile.TemporaryDirectory(prefix="biomech-vendor-") as tmpdir:
        checkout = Path(tmpdir) / "checkout"
        _shallow_clone(options.url, options.ref, checkout)

        models_root = _models_root_from_manifest(checkout)
        if models_root is None:
            models_root = _guess_models_root(checkout, options.repo)
        if models_root is None:
            raise FileNotFoundError(
                f"Could not locate a models tree in {options.repo}@{options.ref}; "
                "manifest missing and no conventional path resolved.",
            )

        destination_models = destination_root / "models"
        _copy_tree(models_root, destination_models)
        destination_root.mkdir(parents=True, exist_ok=True)
        _copy_manifest(checkout, destination_root)
        _write_provenance(destination_root, options)
    logger.info("Snapshot ready at %s", destination_root)
    return destination_root


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    options = _parse_args(argv)
    try:
        snapshot(options)
    except subprocess.CalledProcessError as exc:
        logger.error("git clone failed: %s", exc)
        return exc.returncode or 1
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
