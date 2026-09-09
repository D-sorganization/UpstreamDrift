"""RTMPose ONNX model pin (#9648) — **PENDING OWNER APPROVAL**.

The URLs are the official OpenMMLab ``rtmlib`` ONNX SDK archives (Apache-2.0),
also mirrored at ``huggingface.co/Tau-J/RTMPose``. Byte sizes were read from
the ``Content-Length`` of the official ``download.openmmlab.com`` responses.
SHA-256 digests are **not pinned**: verifying them requires downloading the
weights, which was deliberately NOT done for this change (no weights fetched,
committed, or verified — owner approval pending). The first real download logs
the actual digest so it can be pinned here.

Nothing here downloads implicitly: the estimator resolves files on disk and
names the exact command when they are missing.
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.core.error_utils import ModelError
from src.shared.python.logging_pkg.logging_config import get_logger

from .model_files import Opener, default_cache_dir, fetch_verified, https_opener

logger = get_logger(__name__)

ALLOWED_ORIGINS = ("https://download.openmmlab.com/",)

KEYPOINT_SETS = ("coco17", "halpe26")


@dataclass(frozen=True)
class RtmposeModelSpec:
    """One pinned RTMPose ONNX model (shipped inside a zip archive).

    Attributes:
        filename: Extracted ``.onnx`` file kept in the cache.
        archive_member: Name of the ``.onnx`` inside the pinned zip.
        url: Official OpenMMLab archive URL.
        size_bytes: Size of the zip archive (HTTP ``Content-Length``).
        sha256: Digest of the zip archive; ``None`` while the pin is
            PENDING OWNER APPROVAL.
    """

    filename: str
    archive_member: str
    url: str
    size_bytes: int
    sha256: str | None


RTMPOSE_MODELS: dict[str, RtmposeModelSpec] = {
    "coco17": RtmposeModelSpec(
        filename="rtmpose_t_simcc_coco17_256x192.onnx",
        archive_member="rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx",  # noqa: E501
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip"
        ),
        size_bytes=12_547_710,
        sha256=None,  # PENDING OWNER APPROVAL: pin after a verified download
    ),
    "halpe26": RtmposeModelSpec(
        filename="rtmpose_t_simcc_halpe26_256x192.onnx",
        archive_member=(
            "rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.onnx"
        ),
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.zip"
        ),
        size_bytes=13_179_169,
        sha256=None,  # PENDING OWNER APPROVAL: pin after a verified download
    ),
}

DOWNLOAD_COMMAND = "python3 -m src.shared.python.pose_estimation.rtmpose_models"


def resolve_rtmpose(keypoint_set: str, cache_dir: Path | None = None) -> Path:
    """The extracted ``.onnx`` on disk, without touching the network.

    Raises:
        ModelError: Naming the download command when the file is missing.
    """
    spec = RTMPOSE_MODELS[keypoint_set]
    path = (cache_dir or default_cache_dir()) / spec.filename
    if not path.is_file():
        raise ModelError(
            spec.filename,
            "resolve",
            details=f"not found at {path}; run `{DOWNLOAD_COMMAND}`",
        )
    return path


def download_rtmpose(
    keypoint_set: str, cache_dir: Path | None = None, *, opener: Opener | None = None
) -> Path:
    """Fetch the pinned archive (size-verified) and extract the ``.onnx``.

    The archive digest is not pinned yet (PENDING OWNER APPROVAL): the
    download enforces the pinned size and logs the actual SHA-256 so it can
    be recorded in :data:`RTMPOSE_MODELS` after verification.
    """
    spec = RTMPOSE_MODELS[keypoint_set]
    base = cache_dir or default_cache_dir()
    open_url = opener or https_opener(ALLOWED_ORIGINS)
    archive_path = base / (Path(spec.filename).stem + ".zip")
    fetch_verified(
        spec.url,
        archive_path,
        sha256=spec.sha256,
        size_bytes=spec.size_bytes,
        opener=open_url,
    )
    target = base / spec.filename
    with zipfile.ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".onnx")]
        if len(names) != 1:
            raise ModelError(
                spec.filename,
                "extract",
                details=f"expected one .onnx in the archive, found {len(names)}",
            )
        target.write_bytes(archive.read(names[0]))
    logger.warning(
        "%s extracted from %s; archive sha256 is NOT pinned yet "
        "(PENDING OWNER APPROVAL for #9648)",
        target,
        archive_path.name,
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the pinned RTMPose ONNX models (PENDING OWNER APPROVAL)"
    )
    parser.add_argument("--keypoint-set", choices=KEYPOINT_SETS, default="coco17")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    path = download_rtmpose(args.keypoint_set, args.cache_dir)
    logger.info("RTMPose %s ready: %s", args.keypoint_set, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
