"""MediaPipe pose-landmarker model files: where they live and how to trust them.

mediapipe >= 0.10 ships no built-in pose model; the Tasks API loads a
``.task`` bundle from disk. The library never fetches one implicitly — a pose
estimator that quietly reaches the network at ``load_model`` time is a surprise
in a lab and a liability in CI — so :func:`resolve_pose_model` only *finds* a
file, and :func:`download_pose_model` fetches one on explicit request and
verifies it against the SHA-256 pinned here (Apache-2.0 models published by
Google; hashes recorded 2026-09-06).

Configuration (``config.typed_settings.Settings``):

- ``MEDIAPIPE_POSE_MODEL_PATH`` — explicit file; wins over everything.
- ``MEDIAPIPE_POSE_MODEL_VARIANT`` — ``lite`` / ``full`` / ``heavy`` (default
  ``full``), looked up in the cache directory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.config.typed_settings import Settings
from src.shared.python.core.contracts import require
from src.shared.python.core.error_utils import ModelError
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_estimation.model_files import (
    default_cache_dir as default_cache_dir,
)
from src.shared.python.pose_estimation.model_files import (
    fetch_verified,
    https_opener,
)
from src.shared.python.pose_estimation.model_files import (
    sha256_of as sha256_of,
)

logger = get_logger(__name__)

_ALLOWED_ORIGIN = "https://storage.googleapis.com/"
_BASE_URL = (
    _ALLOWED_ORIGIN + "mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)
_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class PoseModelSpec:
    """One published pose-landmarker bundle."""

    variant: str
    sha256: str
    size_bytes: int

    @property
    def filename(self) -> str:
        return f"pose_landmarker_{self.variant}.task"

    @property
    def url(self) -> str:
        return _BASE_URL.format(variant=self.variant)


POSE_MODELS: dict[str, PoseModelSpec] = {
    "lite": PoseModelSpec(
        "lite",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
        5_777_746,
    ),
    "full": PoseModelSpec(
        "full",
        "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
        9_398_198,
    ),
    "heavy": PoseModelSpec(
        "heavy",
        "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
        30_664_242,
    ),
}
DEFAULT_VARIANT = "full"


def model_spec(variant: str) -> PoseModelSpec:
    """Look up a variant; raises ``ValueError`` listing the valid names."""
    try:
        return POSE_MODELS[variant]
    except KeyError:
        raise ValueError(
            f"unknown pose model variant {variant!r}; expected one of "
            f"{', '.join(sorted(POSE_MODELS))}"
        ) from None


def verify_pose_model(path: Path, variant: str) -> bool:
    """True when ``path`` matches the pinned digest for ``variant``."""
    return path.is_file() and sha256_of(path) == model_spec(variant).sha256


def resolve_pose_model(
    explicit: Path | None = None,
    variant: str | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    """Locate the model file without touching the network.

    Precedence: ``explicit`` argument, then ``MEDIAPIPE_POSE_MODEL_PATH``, then
    ``<cache>/pose_landmarker_<variant>.task``. Raises :class:`ModelError` with
    the exact download command when nothing exists.
    """
    cfg = settings or Settings()
    chosen_variant = variant or cfg.mediapipe_pose_model_variant or DEFAULT_VARIANT
    spec = model_spec(chosen_variant)
    candidate = explicit or (
        Path(cfg.mediapipe_pose_model_path) if cfg.mediapipe_pose_model_path else None
    )
    if candidate is None:
        candidate = default_cache_dir() / spec.filename
    if candidate.is_file():
        return candidate
    raise ModelError(
        spec.filename,
        "resolve",
        details=(
            f"not found at {candidate}. Download it with `python3 -m "
            f"src.shared.python.pose_estimation.mediapipe_models --variant "
            f"{chosen_variant}` or set MEDIAPIPE_POSE_MODEL_PATH."
        ),
    )


Opener = Callable[[str], Iterable[bytes]]

_default_opener = https_opener((_ALLOWED_ORIGIN,))


def download_pose_model(
    variant: str = DEFAULT_VARIANT,
    dest_dir: Path | None = None,
    *,
    opener: Opener = _default_opener,
) -> Path:
    """Fetch a variant into ``dest_dir`` and verify its SHA-256 before installing.

    The download lands in a temporary file and is moved into place only when
    the digest matches; a mismatch leaves no partial file behind and raises
    :class:`ModelError`. Postcondition: the returned path verifies.
    """
    spec = model_spec(variant)
    target_dir = dest_dir or default_cache_dir()
    return fetch_verified(
        spec.url,
        target_dir / spec.filename,
        sha256=spec.sha256,
        size_bytes=spec.size_bytes,
        opener=opener,
    )


def main(argv: list[str] | None = None) -> int:
    """``python3 -m src.shared.python.pose_estimation.mediapipe_models --variant full``."""
    import argparse

    parser = argparse.ArgumentParser(description="Download a MediaPipe pose model")
    parser.add_argument(
        "--variant", default=DEFAULT_VARIANT, choices=sorted(POSE_MODELS)
    )
    parser.add_argument("--dest-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    path = download_pose_model(args.variant, args.dest_dir)
    logger.info("pose model ready: %s", path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
