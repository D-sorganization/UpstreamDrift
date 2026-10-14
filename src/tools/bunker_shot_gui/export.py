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
import io
import json
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
    "EXPORT_METADATA_KEY",
    "EXPORT_SCHEMA_VERSION",
    "ExportIntegrityError",
    "ExportStamp",
    "ExportedFigure",
    "evaluation_stamp",
    "export_figure",
    "header_digest",
    "read_export",
]

EXPORT_SCHEMA_VERSION = "bunker-workbench-figure/1"
"""Version of the export header; bumped on any breaking change."""

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
