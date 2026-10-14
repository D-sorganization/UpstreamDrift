"""Stamped figure exports, and the reader that refuses corrupt ones (#9545).

The in-frame stamp (:func:`~.report.validity_stamp`) keeps a panel's tier and
verdict in any screenshot, because it is drawn inside the axes. A file
written from the workbench needs the same guarantee in a form a program can
check, which pixels cannot give. This module writes a PNG that carries the
standing *as data* -- view, verdict status, fidelity tier, the in-frame
words, the sources of every number and, optionally, the digest of the v1
record it illustrates -- in an iTXt chunk, sealed by a SHA-256 over that
header and the decoded pixels together.

Contract
--------

* **Nothing unstamped leaves.** :func:`export_figure` refuses a figure that
  does not draw the exact stamp its :class:`ExportStamp` claims, so the data
  and the picture cannot disagree at the moment of writing.
* **No export is predictive.** The header says ``predictive: false``; no
  sand-to-ball transfer qualification ships (#9239), and the reader refuses
  any header that claims otherwise.
* **The write is atomic.** The PNG is staged beside its destination and
  promoted by the shared
  :func:`~src.shared.python.motion_matching.jobs.io_atomic.atomic_write_bytes`,
  so a reader never sees half a file.
* **A reloaded file is checked, never re-trusted.** :func:`read_export`
  fails closed with :class:`ExportIntegrityError` on a missing or garbled
  stamp chunk, an unknown schema, a header or pixel digest that no longer
  covers the file, or a stamp whose words contradict its status and tier.

Limitation: the digest is unkeyed. It detects corruption and edits, not a
deliberate forger who recomputes it; the reader's re-derivation of the stamp
words and the ``predictive`` refusal are what stop a re-sealed promotion.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo

from bunkershot3d.provenance.hashing import canonical_json, config_hash
from bunkershot3d.solvers import EnvelopeStatus, FidelityTier
from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_bytes

from .outcomes import DesignEvaluation
from .report import evaluation_sources, validity_stamp

__all__ = [
    "ANIMATION_METADATA_KEY",
    "ANIMATION_SCHEMA_VERSION",
    "EXPORT_METADATA_KEY",
    "EXPORT_SCHEMA_VERSION",
    "ExportIntegrityError",
    "ExportStamp",
    "ExportedAnimation",
    "ExportedFigure",
    "apply_loaded_evaluation_to_widget",
    "evaluation_stamp",
    "export_figure",
    "export_vtk_stamped_animation",
    "export_vtk_stamped_frame",
    "export_workbench_views",
    "header_digest",
    "load_workbench_evaluation",
    "prompt_export_views_dialog",
    "prompt_load_evaluation_dialog",
    "prompt_save_evaluation_dialog",
    "read_animation_export",
    "read_export",
    "save_workbench_evaluation",
    "update_action_tooltip_and_state",
]

logger = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = "bunker-workbench-figure/1"
"""Version of the export header; bumped on any breaking change."""

ANIMATION_SCHEMA_VERSION = "bunker-workbench-animation/1"
"""Version of the animation export header."""

ANIMATION_METADATA_KEY = "bunker-workbench-animation"
"""The GIF comment keyword the animation header is stored under."""

EXPORT_METADATA_KEY = "bunker-workbench-export"
"""The PNG iTXt keyword the header is stored under."""

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
"""A SHA-256 hex digest, as the v1 record and this header spell it."""

_PIXEL_MODE = "RGBA"
"""The only pixel layout written, so the reader hashes what was hashed."""


class ExportIntegrityError(ValueError):
    """A file is not a stamped workbench export this reader can trust."""


@dataclass(frozen=True)
class ExportStamp:
    """The standing an exported figure carries as data.

    Attributes:
        view: Which workbench view this is (``"sole_load"``, ``"slice"``...).
        status: The verdict status the figure is stamped with.
        tier: The ADR-0032 fidelity tier that produced it.
        sources: JSON-ready provenance; must name the ``model``.
        record_digest: The digest of the v1 record this figure illustrates,
            or ``None`` when it stands alone.

    Raises:
        TypeError: If ``status`` or ``tier`` is not the enum itself.
        ValueError: If ``view`` is blank, ``sources`` lacks a ``model`` or is
            not canonical JSON, or ``record_digest`` is malformed.
    """

    view: str
    status: EnvelopeStatus
    tier: FidelityTier
    sources: Mapping[str, Any] = field(hash=False)
    record_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, EnvelopeStatus):
            raise TypeError("status must be an EnvelopeStatus")
        if not isinstance(self.tier, FidelityTier):
            raise TypeError("tier must be a FidelityTier")
        if not isinstance(self.view, str) or not self.view.strip():
            raise ValueError("view must be a non-blank name")
        if not isinstance(self.sources, Mapping) or not self.sources:
            raise ValueError("sources must be a non-empty mapping")
        if "model" not in self.sources:
            raise ValueError("sources must name the model that produced the figure")
        canonical_json(self.sources)  # raises on anything not JSON-ready
        digest = self.record_digest
        if digest is not None and not _DIGEST_PATTERN.fullmatch(str(digest)):
            raise ValueError("record_digest must be 64 lowercase hex characters")

    @property
    def text(self) -> str:
        """The in-frame words this stamp requires the figure to draw."""
        return validity_stamp(self.status, self.tier)

    def as_json(self) -> dict[str, Any]:
        """The stamp as the JSON object stored in the header."""
        return {
            "view": self.view,
            "status": self.status.value,
            "tier": self.tier.value,
            "text": self.text,
            "sources": json.loads(canonical_json(self.sources)),
            "record_digest": self.record_digest,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ExportStamp:
        """Rebuild a stamp from its header object.

        Raises:
            ExportIntegrityError: If a field is missing or invalid, or the
                stored words are not the words its status and tier compose.
        """
        try:
            stamp = cls(
                view=data["view"],
                status=EnvelopeStatus(data["status"]),
                tier=FidelityTier(data["tier"]),
                sources=data["sources"],
                record_digest=data.get("record_digest"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExportIntegrityError(f"the export stamp is malformed: {exc}") from exc
        if data.get("text") != stamp.text:
            raise ExportIntegrityError(
                "the stamp text does not match its status and tier; the header "
                "was edited after it was drawn"
            )
        return stamp


@dataclass(frozen=True)
class ExportedFigure:
    """A stamped export, as written or as verified on reload.

    Attributes:
        path: Where the file is.
        stamp: The standing it carries.
        digest: SHA-256 over the header, pixels included by their digest.
        shape: ``(height, width, 4)`` of the RGBA pixels.
    """

    path: Path
    stamp: ExportStamp
    digest: str
    shape: tuple[int, int, int]


def evaluation_stamp(
    evaluation: DesignEvaluation, *, view: str, record_digest: str | None = None
) -> ExportStamp:
    """The stamp for a figure of one evaluation's nominal shot.

    Args:
        evaluation: The evaluation the figure draws.
        view: Which workbench view the figure is.
        record_digest: The digest of the evaluation's v1 record, if saved.

    Returns:
        A stamp with the shot's status and tier and the evaluation's
        sources, the same block the v1 API record carries.
    """
    shot = evaluation.shot
    return ExportStamp(
        view=view,
        status=shot.status,
        tier=shot.fidelity_tier,
        sources=evaluation_sources(evaluation),
        record_digest=record_digest,
    )


def header_digest(header: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a header without its ``digest``.

    Returns:
        64 lowercase hex characters.
    """
    return config_hash({key: value for key, value in header.items() if key != "digest"})


def _pixel_digest(pixels: NDArray[np.uint8]) -> str:
    """SHA-256 over the RGBA bytes, with the shape so a reshape is caught."""
    hasher = hashlib.sha256(canonical_json(list(pixels.shape)).encode("ascii"))
    hasher.update(np.ascontiguousarray(pixels).tobytes())
    return hasher.hexdigest()


def _draws_stamp(figure: Figure, text: str) -> bool:
    """Whether any text artist in ``figure`` draws ``text``."""
    return any(text in artist.get_text() for artist in figure.findobj(Text))


def _rasterise(figure: Figure) -> NDArray[np.uint8]:
    """Draw ``figure`` on an Agg canvas and return a copy of its RGBA pixels."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return np.array(canvas.buffer_rgba(), dtype=np.uint8, copy=True)


def _header(stamp: ExportStamp, pixels: NDArray[np.uint8]) -> dict[str, Any]:
    """The sealed header for ``pixels`` stamped with ``stamp``."""
    header: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "stamp": stamp.as_json(),
        "predictive": False,
        "pixels": {"shape": list(pixels.shape), "sha256": _pixel_digest(pixels)},
    }
    header["digest"] = header_digest(header)
    return header


def export_figure(
    figure: Figure, path: Path | str, stamp: ExportStamp
) -> ExportedFigure:
    """Write ``figure`` as a stamped PNG, atomically.

    Args:
        figure: The figure; it must already draw ``stamp.text`` in-frame.
        path: Destination, ending in ``.png``.
        stamp: The standing to carry as data.

    Returns:
        What was written; :func:`read_export` on ``path`` returns the same.

    Raises:
        ValueError: If ``path`` is not a ``.png`` or the figure does not draw
            the in-frame stamp ``stamp`` claims. Nothing is written then.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        raise ValueError(f"exports are written as .png, not {destination.suffix!r}")
    if not _draws_stamp(figure, stamp.text):
        raise ValueError(
            "the figure does not draw the in-frame stamp its export claims "
            f"({stamp.text.splitlines()[0]!r}); stamp the axes before exporting"
        )
    pixels = _rasterise(figure)
    header = _header(stamp, pixels)
    info = PngInfo()
    info.add_itxt(EXPORT_METADATA_KEY, json.dumps(header, sort_keys=True))
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode=_PIXEL_MODE).save(buffer, format="PNG", pnginfo=info)
    atomic_write_bytes(destination, buffer.getvalue())
    height, width, depth = pixels.shape
    return ExportedFigure(destination, stamp, header["digest"], (height, width, depth))


def _open(path: Path) -> tuple[NDArray[np.uint8], str | None]:
    """Decode the pixels and the raw header text of a PNG."""
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != _PIXEL_MODE:
                raise ExportIntegrityError(
                    f"{path.name} is not an {_PIXEL_MODE} PNG workbench export"
                )
            text = getattr(image, "text", {}).get(EXPORT_METADATA_KEY)
            return np.array(image, dtype=np.uint8), text
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ExportIntegrityError(f"{path.name} is not a readable PNG: {exc}") from exc


def _parse_header(text: str | None) -> dict[str, Any]:
    """The header object, refusing an absent, garbled or foreign one."""
    if text is None:
        raise ExportIntegrityError(
            "the file is unstamped: it carries no workbench export header"
        )
    try:
        header = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExportIntegrityError(f"the export header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ExportIntegrityError("the export header is not a JSON object")
    schema = header.get("schema_version")
    if schema != EXPORT_SCHEMA_VERSION:
        raise ExportIntegrityError(
            f"export schema {schema!r} is not {EXPORT_SCHEMA_VERSION!r}"
        )
    return header


def _check_seal(header: Mapping[str, Any], pixels: NDArray[np.uint8]) -> str:
    """Verify the header digest, then the pixel digest; return the digest."""
    digest = header.get("digest")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ExportIntegrityError("the export header carries no well-formed digest")
    try:
        expected = header_digest(header)
    except (TypeError, ValueError) as exc:
        raise ExportIntegrityError(f"the export header is malformed: {exc}") from exc
    if expected != digest:
        raise ExportIntegrityError(
            "the export header was changed after it was written; its digest "
            "no longer covers it"
        )
    recorded = header.get("pixels")
    if not isinstance(recorded, Mapping) or recorded.get("sha256") != _pixel_digest(
        pixels
    ):
        raise ExportIntegrityError(
            "the pixels were changed after the export was written"
        )
    return digest


def read_export(path: Path | str) -> ExportedFigure:
    """Load a stamped export and verify it before anything trusts it.

    Args:
        path: The PNG.

    Returns:
        The verified export.

    Raises:
        FileNotFoundError: If there is no file at ``path``.
        ExportIntegrityError: If the file is not a PNG, is unstamped, names
            an unknown schema, fails its header or pixel digest, carries a
            stamp whose words contradict it, or claims to be predictive.
    """
    source = Path(path)
    pixels, text = _open(source)
    header = _parse_header(text)
    digest = _check_seal(header, pixels)
    if header.get("predictive") is not False:
        raise ExportIntegrityError(
            "the export claims to be predictive; no workbench result is (#9239)"
        )
    stamp_data = header.get("stamp")
    if not isinstance(stamp_data, Mapping):
        raise ExportIntegrityError("the export header carries no stamp object")
    stamp = ExportStamp.from_json(stamp_data)
    height, width, depth = pixels.shape
    return ExportedFigure(source, stamp, digest, (height, width, depth))


@dataclass(frozen=True)
class ExportedAnimation:
    """A stamped animation export (GIF).

    Attributes:
        path: Where the file is.
        stamp: The standing it carries.
        digest: SHA-256 over the header without digest.
        frame_count: Number of frames in the animation.
        duration_s: Total duration in seconds.
    """

    path: Path
    stamp: ExportStamp
    digest: str
    frame_count: int
    duration_s: float


def save_workbench_evaluation(
    evaluation: DesignEvaluation | Mapping[str, Any],
    path: Path | str,
    *,
    swing: Any = None,
    handedness: str = "right",
    objective: Any = None,
) -> Path:
    """Save one evaluated design as a verifiable JSON record, atomically.

    Args:
        evaluation: The evaluated design or an existing evaluation record dict.
        path: Destination path ending in .json.
        swing: Delivery setup, or None to use default SwingSetup.
        handedness: Handedness, defaults to "right".
        objective: Exploratory objective, or None.

    Returns:
        The destination Path.

    Raises:
        ValueError: If path does not end in .json.
    """
    destination = Path(path)
    suffix = destination.suffix
    if suffix.lower() != ".json":
        raise ValueError(f"evaluations are saved as .json, not {suffix!r}")
    if isinstance(evaluation, Mapping):
        record = dict(evaluation)
        if "digest" not in record:
            body = {k: v for k, v in record.items() if k != "digest"}
            record["digest"] = hashlib.sha256(
                canonical_json(body).encode("utf-8")
            ).hexdigest()
    else:
        from src.api.routes.bunker_workbench import evaluation_record
        from src.tools.bunker_shot_gui.design import SwingSetup

        delivery = swing if swing is not None else SwingSetup()
        record = evaluation_record(
            evaluation,
            delivery,
            handedness=handedness,
            objective=objective,
        )
    encoded = json.dumps(record, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(destination, encoded)
    return destination


def load_workbench_evaluation(path: Path | str) -> dict[str, Any]:
    """Load and verify a saved workbench evaluation record.

    Fails closed on missing file, corrupted JSON, missing or unknown schema,
    tampered payload where the digest no longer covers the body, or an evaluation
    claiming to be predictive.

    Args:
        path: The JSON evaluation file.

    Returns:
        The verified evaluation dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ExportIntegrityError: If the record is invalid or tampered with.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"evaluation record not found: {source}")
    try:
        content = source.read_text(encoding="utf-8")
        record = json.loads(content)
    except (json.JSONDecodeError, OSError) as exc:
        raise ExportIntegrityError(
            f"{source.name} is not readable JSON: {exc}"
        ) from exc
    if not isinstance(record, dict):
        raise ExportIntegrityError("evaluation record must be a JSON object")

    schema = record.get("schema_version")
    if schema != "bunker-workbench-evaluation/1":
        raise ExportIntegrityError(
            f"evaluation schema {schema!r} is not 'bunker-workbench-evaluation/1'"
        )

    digest = record.get("digest")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ExportIntegrityError("evaluation record carries no well-formed digest")

    body = {k: v for k, v in record.items() if k != "digest"}
    expected = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    if not hmac.compare_digest(expected, digest):
        raise ExportIntegrityError(
            "the evaluation was changed after it was produced; its digest no longer covers its contents"
        )

    if record.get("predictive") is not False:
        raise ExportIntegrityError(
            "the evaluation claims to be predictive; no workbench result is (#9239)"
        )

    return record


def _require_vtk_stamp_match(artists: Any, stamp: ExportStamp) -> None:
    if hasattr(artists, "_stamp_text"):
        stamp_text = artists._stamp_text
        if stamp.text not in stamp_text:
            first_line = stamp.text.splitlines()[0]
            raise ValueError(
                f"the VTK scene does not draw the in-frame stamp its export claims ({first_line!r})"
            )


def export_vtk_stamped_frame(
    artists: Any,
    path: Path | str,
    stamp: ExportStamp,
) -> ExportedFigure:
    """Write one VTK frame as a stamped PNG, atomically.

    Args:
        artists: A VtkSceneArtists instance or an RGBA pixel array.
        path: Destination, ending in .png.
        stamp: The standing to carry as data.

    Returns:
        The verified ExportedFigure.

    Raises:
        ValueError: If path is not .png or the VTK frame does not draw the stamp.
        TypeError: If artists is not a supported type.
    """
    destination = Path(path)
    suffix = destination.suffix
    if suffix.lower() != ".png":
        raise ValueError(f"exports are written as .png, not {suffix!r}")

    if hasattr(artists, "image_array"):
        raw_pixels = artists.image_array()
    elif isinstance(artists, np.ndarray):
        raw_pixels = artists
    else:
        raise TypeError(
            f"artists must be a VtkSceneArtists or pixel array, got {type(artists)}"
        )

    _require_vtk_stamp_match(artists, stamp)

    ndim = raw_pixels.ndim
    shape = raw_pixels.shape
    if ndim != 3 or shape[2] not in (3, 4):
        raise ValueError(
            f"expected 3D image array with 3 or 4 channels, got shape {shape}"
        )

    if shape[2] == 3:
        h, w = shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        pixels = np.concatenate([raw_pixels, alpha], axis=-1)
    else:
        pixels = np.asarray(raw_pixels, dtype=np.uint8)

    header = _header(stamp, pixels)
    info = PngInfo()
    serialized = json.dumps(header, sort_keys=True)
    info.add_itxt(EXPORT_METADATA_KEY, serialized)

    buffer = io.BytesIO()
    img = Image.fromarray(pixels, mode=_PIXEL_MODE)
    img.save(buffer, format="PNG", pnginfo=info)

    atomic_write_bytes(destination, buffer.getvalue())
    height, width, depth = pixels.shape
    return ExportedFigure(destination, stamp, header["digest"], (height, width, depth))


def export_vtk_stamped_animation(
    artists: Any,
    path: Path | str,
    stamp: ExportStamp,
    *,
    fps: int = 30,
    stride: int = 1,
    max_frames: int | None = None,
) -> ExportedAnimation:
    """Write a VTK animation as a stamped animated GIF, atomically.

    Args:
        artists: A VtkSceneArtists instance or a sequence of frame arrays.
        path: Destination ending in .gif.
        stamp: The standing to carry as data.
        fps: Playback frames per second.
        stride: Stride between sampled frames.
        max_frames: Optional upper limit on exported frames.

    Returns:
        The ExportedAnimation metadata.

    Raises:
        ValueError: If path is not .gif or no frames are available.
        TypeError: If artists is not a supported type.
    """
    destination = Path(path)
    suffix = destination.suffix
    if suffix.lower() != ".gif":
        raise ValueError(f"animations are written as .gif, not {suffix!r}")

    pil_frames: list[Image.Image] = []
    if hasattr(artists, "update") and hasattr(artists, "_scene"):
        scene = artists._scene
        total_frames = scene.n_frames
        step = max(1, int(stride))
        indices = list(range(0, total_frames, step))
        if max_frames is not None:
            indices = indices[: max(1, int(max_frames))]
        for frame_idx in indices:
            artists.update(frame_idx)
            arr = artists.image_array()
            pil_frames.append(Image.fromarray(arr))
    elif isinstance(artists, (list, tuple)):
        items = artists[:max_frames] if max_frames is not None else artists
        for item in items:
            frame_img = item if isinstance(item, Image.Image) else Image.fromarray(item)
            pil_frames.append(frame_img)
    else:
        raise TypeError(
            f"artists must be a VtkSceneArtists or frame sequence, got {type(artists)}"
        )

    if not pil_frames:
        raise ValueError("no frames to export into animation")

    _require_vtk_stamp_match(artists, stamp)

    frame_count = len(pil_frames)
    frame_duration_ms = max(1, int(1000 / max(1, int(fps))))
    duration_s = (frame_count * frame_duration_ms) / 1000.0

    header: dict[str, Any] = {
        "schema_version": ANIMATION_SCHEMA_VERSION,
        "stamp": stamp.as_json(),
        "predictive": False,
        "fps": int(fps),
        "frame_count": frame_count,
        "duration_s": duration_s,
    }
    header["digest"] = header_digest(header)

    buffer = io.BytesIO()
    serialized = json.dumps(header, sort_keys=True)
    pil_frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        comment=serialized.encode("utf-8"),
    )
    atomic_write_bytes(destination, buffer.getvalue())
    return ExportedAnimation(
        destination, stamp, header["digest"], frame_count, duration_s
    )


def read_animation_export(path: Path | str) -> ExportedAnimation:
    """Load and verify a stamped animation export (GIF).

    Args:
        path: The animated GIF.

    Returns:
        The verified ExportedAnimation.

    Raises:
        FileNotFoundError: If the file does not exist.
        ExportIntegrityError: If the animation is unstamped or corrupted.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"animation export not found: {source}")
    try:
        with Image.open(source) as image:
            info = image.info
            comment_raw = info.get("comment", b"")
            if isinstance(comment_raw, bytes):
                comment_text = comment_raw.decode("utf-8")
            else:
                comment_text = str(comment_raw)
            header = json.loads(comment_text)
            n_frames = getattr(image, "n_frames", 1)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ExportIntegrityError(
            f"{source.name} is not a valid stamped GIF animation: {exc}"
        ) from exc

    if not isinstance(header, dict):
        raise ExportIntegrityError("animation export header is not a JSON object")
    schema = header.get("schema_version")
    if schema != ANIMATION_SCHEMA_VERSION:
        raise ExportIntegrityError(
            f"animation schema {schema!r} is not {ANIMATION_SCHEMA_VERSION!r}"
        )

    digest = header.get("digest")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise ExportIntegrityError("animation export carries no well-formed digest")

    expected = header_digest(header)
    if not hmac.compare_digest(expected, digest):
        raise ExportIntegrityError("animation export header was changed after writing")

    if header.get("predictive") is not False:
        raise ExportIntegrityError(
            "the animation export claims to be predictive; no workbench result is (#9239)"
        )

    stamp_data = header.get("stamp")
    if not isinstance(stamp_data, Mapping):
        raise ExportIntegrityError("animation export header carries no stamp object")

    stamp = ExportStamp.from_json(stamp_data)
    frame_count = int(header.get("frame_count", n_frames))
    duration_s = float(header.get("duration_s", 0.0))
    return ExportedAnimation(source, stamp, digest, frame_count, duration_s)


def export_workbench_views(
    evaluation: Any,
    destination_dir: Path | str,
    *,
    record_digest: str | None = None,
) -> list[ExportedFigure]:
    """Export stamped figures for all available evaluation views.

    Args:
        evaluation: The EvaluationResult / DesignEvaluation to render and export.
        destination_dir: Directory where figures will be saved.
        record_digest: Optional SHA-256 digest from a loaded record.

    Returns:
        List of verified ExportedFigure records.
    """
    from .render import sole_load_still

    out_dir = Path(destination_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: list[ExportedFigure] = []
    shot = getattr(evaluation, "shot", None)
    if shot is None:
        return exported

    sole_field = getattr(shot, "sole_field", None)
    if sole_field is not None:
        stamp = evaluation_stamp(
            evaluation, view="sole_load", record_digest=record_digest
        )
        fig = sole_load_still(sole_field)
        p = out_dir / "sole_load.png"
        exported.append(export_figure(fig, p, stamp))

    traces = getattr(shot, "traces", None)
    if traces is not None:
        from .render_traces import trace_panel_still

        stamp = evaluation_stamp(evaluation, view="traces", record_digest=record_digest)
        fig = trace_panel_still(traces)
        p = out_dir / "traces.png"
        exported.append(export_figure(fig, p, stamp))

    scene = getattr(shot, "scene", None)
    if scene is not None:
        from .render3d import shot_scene_still

        stamp = evaluation_stamp(
            evaluation, view="shot_scene", record_digest=record_digest
        )
        fig = shot_scene_still(scene)
        p = out_dir / "shot_scene.png"
        exported.append(export_figure(fig, p, stamp))

    return exported


def prompt_export_views_dialog(parent: Any, widget: Any) -> None:
    """Prompt user for an export directory and export views, handling errors."""
    from PyQt6.QtWidgets import QFileDialog

    dir_path = QFileDialog.getExistingDirectory(parent, "Export views to directory")
    if dir_path:
        try:
            exported = widget.export_views(dir_path)
            logger.info("exported %d views to %s", len(exported), dir_path)
        except (ValueError, OSError) as exc:
            logger.error("failed to export views: %s", exc)
            widget._banner.show_error(f"Export failed: {exc}")


def prompt_save_evaluation_dialog(parent: Any, widget: Any) -> None:
    """Prompt user for a destination file path and save evaluation, handling errors."""
    from PyQt6.QtWidgets import QFileDialog

    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save workbench evaluation",
        "",
        "JSON evaluation (*.json);;All files (*)",
    )
    if path:
        try:
            saved = widget.save_evaluation(path)
            logger.info("saved evaluation to %s", saved)
        except (ValueError, OSError) as exc:
            logger.error("failed to save evaluation: %s", exc)
            widget._banner.show_error(f"Save failed: {exc}")


def prompt_load_evaluation_dialog(parent: Any, widget: Any) -> None:
    """Prompt user for a file path and load verified evaluation, handling errors."""
    from PyQt6.QtWidgets import QFileDialog

    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Load workbench evaluation",
        "",
        "JSON evaluation (*.json);;All files (*)",
    )
    if path:
        try:
            widget.load_evaluation(path)
        except (ValueError, OSError) as exc:
            logger.error("failed to load evaluation: %s", exc)
            widget._banner.show_error(f"Load failed: {exc}")


def apply_loaded_evaluation_to_widget(
    widget: Any,
    record: dict[str, Any],
    path: Path | str,
) -> None:
    """Apply loaded evaluation inputs, verdict banner, and summary text to widget."""
    from bunkershot3d.solvers import EnvelopeStatus

    inputs = record.get("inputs", {})
    design_data = inputs.get("design", {})
    if design_data and hasattr(widget._design_a, "set_design"):
        widget._design_a.set_design(design_data)

    sand_data = inputs.get("sand", {})
    swing_data = inputs.get("swing", {})
    if hasattr(widget._conditions, "set_conditions"):
        widget._conditions.set_conditions(sand_data, swing_data)

    verdict_info = record.get("verdict", {})
    status_str = verdict_info.get("status", "beyond_validation")
    status = EnvelopeStatus(status_str)
    widget._banner.show_status(status)

    src_path = Path(path)
    digest = record.get("digest", "")
    stamp = record.get("stamp", "")
    schema = record.get("schema_version", "")
    headline = verdict_info.get("headline", "")
    summary = (
        f"Loaded evaluation: {src_path.name}\n"
        f"Schema: {schema}\n"
        f"Digest: {digest}\n"
        f"Stamp: {stamp}\n"
        f"Verdict: {headline}"
    )
    widget._results.setPlainText(summary)
    widget.update_control_states()
    widget.evaluation_changed.emit()


def update_action_tooltip_and_state(
    action: Any,
    reasons: dict[str, str | None],
    action_key: str,
    enabled_tooltip: str,
) -> None:
    """Helper to update a QAction enabled state and tooltip based on reasons dict."""
    if action is None:
        return
    reason = reasons.get(action_key)
    action.setEnabled(reason is None)
    if reason:
        action.setToolTip(f"Disabled: {reason}")
    else:
        action.setToolTip(enabled_tooltip)
