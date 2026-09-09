"""OpenPose BODY_25 Caffe model files for the OpenCV-DNN estimator.

The official CMU host (``posefs1.perception.cs.cmu.edu``) is frequently
offline, so the weights are pinned to a public mirror by size and, once
verified, by SHA-256. Nothing here downloads implicitly: the estimator
resolves files on disk and names the exact command when they are missing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.core.error_utils import ModelError
from src.shared.python.logging_pkg.logging_config import get_logger

from .model_files import (
    Opener,
    default_cache_dir,
    fetch_verified,
    https_opener,
    verify_file,
)

logger = get_logger(__name__)

ALLOWED_ORIGINS = (
    "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/",
    "https://huggingface.co/camenduru/openpose/resolve/main/",
)


@dataclass(frozen=True)
class ModelFileSpec:
    """One file of the BODY_25 model."""

    filename: str
    url: str
    size_bytes: int
    sha256: str | None  # pinned 2026-09-07 from a verified download


BODY25_FILES: dict[str, ModelFileSpec] = {
    "prototxt": ModelFileSpec(
        filename="openpose_body25_pose_deploy.prototxt",
        url=ALLOWED_ORIGINS[0] + "master/models/pose/body_25/pose_deploy.prototxt",
        size_bytes=42_330,
        sha256="44d6ed3a5268d8d41ca59b3a040491277d876975c3234d82cf7ec0539b4b1f61",
    ),
    "weights": ModelFileSpec(
        filename="openpose_body25_pose_iter_584000.caffemodel",
        url=ALLOWED_ORIGINS[1] + "models/pose/body_25/pose_iter_584000.caffemodel",
        size_bytes=104_715_850,
        sha256="44e3d7ebd8c8b62d4366d67127f1b562611a9e8fd0f4f3cdeeb4bb4a6ed12be6",
    ),
}

DOWNLOAD_COMMAND = "python3 -m src.shared.python.pose_estimation.openpose_models"


def resolve_body25(cache_dir: Path | None = None) -> tuple[Path, Path]:
    """``(prototxt, weights)`` on disk, without touching the network.

    Raises :class:`ModelError` naming the download command when either file is
    missing or fails its size/digest check.
    """
    base = cache_dir or default_cache_dir()
    paths = []
    for key in ("prototxt", "weights"):
        spec = BODY25_FILES[key]
        path = base / spec.filename
        if not verify_file(path, sha256=spec.sha256, size_bytes=spec.size_bytes):
            raise ModelError(
                spec.filename,
                "resolve",
                details=f"not found or unverified at {path}; run `{DOWNLOAD_COMMAND}`",
            )
        paths.append(path)
    return paths[0], paths[1]


def download_body25(
    cache_dir: Path | None = None, *, opener: Opener | None = None
) -> tuple[Path, Path]:
    """Fetch both files (verified) into the cache; idempotent."""
    base = cache_dir or default_cache_dir()
    open_url = opener or https_opener(ALLOWED_ORIGINS)
    out = []
    for key in ("prototxt", "weights"):
        spec = BODY25_FILES[key]
        out.append(
            fetch_verified(
                spec.url,
                base / spec.filename,
                sha256=spec.sha256,
                size_bytes=spec.size_bytes,
                opener=open_url,
            )
        )
    return out[0], out[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the OpenPose BODY_25 model")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    prototxt, weights = download_body25(args.cache_dir)
    logger.info("BODY_25 ready: %s, %s", prototxt, weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
