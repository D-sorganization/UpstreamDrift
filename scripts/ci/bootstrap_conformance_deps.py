#!/usr/bin/env python3
"""Deterministic bootstrap of canonical conformance dependencies offline and online."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import shutil
import site
import subprocess
import sys

logger = logging.getLogger(__name__)

CANONICAL_PINS: dict[str, str] = {
    "numpy": "2.2.6",
    "scipy": "1.14.1",
    "pydantic": "2.12.5",
}


class MissingArtifactError(RuntimeError):
    """Raised when an approved wheel artifact is missing under offline operation."""


def clean_scipy_site_packages() -> None:
    """Clean lingering scipy and C-extension directories before reinstall."""
    try:
        site_packages = site.getsitepackages()
    except AttributeError:
        site_packages = []

    for root in site_packages:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for pattern in ("scipy", "scipy.libs", "scipy-*.dist-info", "~cipy*"):
            for path in root_path.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)


def verify_offline_artifacts(
    wheel_dir: Path,
    required_pins: dict[str, str],
) -> list[str]:
    """Verify that all required package pins have matching wheel files in wheel_dir."""
    if not wheel_dir.is_dir():
        raise MissingArtifactError(f"Wheel directory does not exist: {wheel_dir}")

    wheel_files = [f.name for f in wheel_dir.glob("*.whl")]
    missing: list[str] = []

    for pkg, pin in required_pins.items():
        # Look for pattern pkg-pin-...whl (case-insensitive for pkg)
        pattern = re.compile(
            rf"^{re.escape(pkg)}-{re.escape(pin)}[.-].*\.whl$", re.IGNORECASE
        )
        if not any(pattern.match(name) for name in wheel_files):
            missing.append(f"{pkg}=={pin}")

    if missing:
        raise MissingArtifactError(
            f"Missing approved wheel artifact for {', '.join(missing)} in {wheel_dir}"
        )

    return missing


def build_pip_install_args(
    *,
    no_index: bool = False,
    find_links: Path | None = None,
    canonical_pins: dict[str, str] | None = None,
    default_timeout: int = 100,
) -> list[str]:
    """Assemble the pip install argument list."""
    pins = canonical_pins or CANONICAL_PINS
    args = [sys.executable, "-m", "pip", "install", "--force-reinstall"]

    if no_index:
        args.append("--no-index")
    else:
        args.extend(["--default-timeout", str(default_timeout)])

    if find_links is not None:
        args.extend(["--find-links", str(find_links)])

    for pkg, pin in pins.items():
        args.append(f"{pkg}=={pin}")

    return args


def bootstrap_conformance(
    *,
    no_index: bool = False,
    find_links: Path | None = None,
    clean_scipy: bool = True,
    dry_run: bool = False,
) -> None:
    """Execute the deterministic conformance dependency bootstrap."""
    if clean_scipy and not dry_run:
        logger.info("Cleaning lingering scipy installation directories...")
        clean_scipy_site_packages()

    if no_index:
        if find_links is None:
            raise MissingArtifactError(
                "Offline --no-index mode requires --find-links wheel source directory."
            )
        verify_offline_artifacts(find_links, CANONICAL_PINS)

    cmd = build_pip_install_args(
        no_index=no_index,
        find_links=find_links,
    )

    logger.info("Running: %s", " ".join(cmd))
    if dry_run:
        return

    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for conformance dependency bootstrap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Operate in offline mode without network/PyPI index queries.",
    )
    parser.add_argument(
        "--find-links",
        type=Path,
        help="Path to local wheel cache directory.",
    )
    parser.add_argument(
        "--skip-clean-scipy",
        action="store_true",
        help="Skip cleaning lingering site-packages scipy artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print installation command without executing.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        bootstrap_conformance(
            no_index=args.no_index,
            find_links=args.find_links,
            clean_scipy=not args.skip_clean_scipy,
            dry_run=args.dry_run,
        )
    except MissingArtifactError as exc:
        logger.error("Conformance bootstrap artifact verification failed: %s", exc)
        return 1
    except subprocess.SubprocessError as exc:
        logger.error("Conformance bootstrap command execution failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
