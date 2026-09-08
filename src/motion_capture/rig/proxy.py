"""H.264 proxies of session recordings for players that cannot decode MJPEG.

The recorder stream-copies the cameras' MJPEG into Matroska because that is
free on the capture host; browsers (the React Video Analyzer, the Tools web
Video Processor) do not decode MJPEG-in-Matroska. A *proxy* is a derived
H.264/yuv420p ``.mp4`` written beside each recording, indexed in
``proxies.json`` with the encoder used and the source it was made from. The
recording stays the evidence; a proxy is a viewing convenience and is never
read by ingest.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from .bundle import load_bundle

logger = get_logger(__name__)

PROXIES_SCHEMA_VERSION = "session-proxies/1.0.0"
PROXIES_FILE = "proxies.json"
ENCODERS = ("libx264", "h264_nvenc", "h264_mf")
DEFAULT_ENCODER = "libx264"
DEFAULT_CRF = 18

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class ProxyEntry(BaseModel):
    """One recording's proxy, or why it has none."""

    model_config = ConfigDict(frozen=True)

    view: str
    source: str
    file: str | None = None
    encoder: str | None = None
    bytes: int = 0
    returncode: int | None = None
    ok: bool = False
    reason: str | None = None


class ProxiesIndex(BaseModel):
    """``proxies.json``."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = PROXIES_SCHEMA_VERSION
    plan_name: str
    proxies: tuple[ProxyEntry, ...] = Field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return bool(self.proxies) and all(p.ok for p in self.proxies)


def proxy_args(
    ffmpeg_exe: str,
    source: Path,
    target: Path,
    *,
    encoder: str = DEFAULT_ENCODER,
    crf: int = DEFAULT_CRF,
) -> list[str]:
    """ffmpeg argv for a browser-playable H.264/yuv420p proxy.

    Precondition: ``encoder`` is one of :data:`ENCODERS`. ``crf`` only applies
    to ``libx264``; hardware encoders take their own rate control defaults.
    """
    require(encoder in ENCODERS, f"encoder must be one of {ENCODERS}", encoder)
    require(0 <= crf <= 51, "crf must be within 0..51", crf)
    quality = ["-crf", str(crf), "-preset", "fast"] if encoder == "libx264" else []
    return [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-c:v",
        encoder,
        *quality,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target),
    ]


def _entry(
    view: str, source: Path, target: Path, encoder: str, returncode: int
) -> ProxyEntry:
    size = target.stat().st_size if target.is_file() else 0
    ok = returncode == 0 and size > 0
    return ProxyEntry(
        view=view,
        source=source.name,
        file=target.name if ok else None,
        encoder=encoder,
        bytes=size,
        returncode=returncode,
        ok=ok,
        reason=None if ok else f"ffmpeg exited {returncode} with {size} bytes",
    )


def make_proxies(
    bundle_dir: Path,
    *,
    encoder: str = DEFAULT_ENCODER,
    crf: int = DEFAULT_CRF,
    ffmpeg_exe: str | None = None,
    runner: Runner = subprocess.run,
    timeout_s: float = 600.0,
) -> ProxiesIndex:
    """Transcode every usable recording of a bundle and write ``proxies.json``.

    Recordings the bundle marks unusable get an entry with the reason rather
    than being skipped silently. Postcondition: one entry per recording.
    """
    require(timeout_s > 0, "timeout_s must be positive", timeout_s)
    plan, index, _manifest = load_bundle(bundle_dir)
    exe = ffmpeg_exe or _bundled_ffmpeg()
    entries: list[ProxyEntry] = []
    for rec in index.recordings:
        source = bundle_dir / rec.file
        if not rec.ok:
            entries.append(
                ProxyEntry(
                    view=rec.view,
                    source=rec.file,
                    reason=f"recording not usable (returncode={rec.returncode})",
                )
            )
            continue
        target = source.with_suffix(".mp4")
        result = runner(
            proxy_args(exe, source, target, encoder=encoder, crf=crf),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        entries.append(_entry(rec.view, source, target, encoder, result.returncode))
    out = ProxiesIndex(plan_name=plan.name, proxies=tuple(entries))
    (bundle_dir / PROXIES_FILE).write_text(
        out.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info(
        "proxies for %s: %d/%d ok",
        bundle_dir,
        sum(p.ok for p in out.proxies),
        len(out.proxies),
    )
    return out


def _bundled_ffmpeg() -> str:
    import imageio_ffmpeg

    exe: str = imageio_ffmpeg.get_ffmpeg_exe()
    return exe
