"""Bounded exact-byte review registration shared by native and HTTP consumers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from .artifact_handoff import ArtifactKind, ArtifactReference
from .necromatcher_source_scope import SOURCE_SCOPE_REVIEW_SCHEMA, SourceFitScope

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary

MAX_SOURCE_SCOPE_REVIEW_BYTES = 1024 * 1024


def import_fit_source_scope_review(
    library: NecromatcherLibrary, fit_id: str, raw: bytes
) -> SourceFitScope:
    """Register exact authored receipt bytes and return a portable checked DTO.

    Identical imports reuse the existing checked asset explicitly; they never
    overwrite it. The canonical library verifies capture, clock and boundaries.
    This operation registers evidence only and does not change any fit.
    """
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_SOURCE_SCOPE_REVIEW_BYTES:
        raise ValueError("Reviewed window requires nonempty bytes up to 1 MiB")
    if not isinstance(fit_id, str) or not fit_id or fit_id.strip() != fit_id:
        raise ValueError("Reviewed window requires a source fit identity")
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    review_id = "scope-review-" + digest.removeprefix("sha256:")
    with TemporaryDirectory(prefix="necromatcher-scope-review-") as directory:
        source = Path(directory) / "review.json"
        source.write_bytes(raw)
        reference = ArtifactReference(
            review_id,
            str(source),
            digest,
            SOURCE_SCOPE_REVIEW_SCHEMA,
            ArtifactKind.RECEIPT,
        )
        declared = SourceFitScope.from_review_artifact(reference, len(raw))
        fit = library.load_fit(fit_id)
        if (fit["capture_id"], fit["capture_hash"]) != (
            declared.capture_id,
            declared.capture_hash,
        ):
            raise ValueError("Reviewed window capture differs from selected fit")
        capture = library.load_asset(fit["capture_id"])
        if capture.kind != "image_capture":
            raise ValueError("Reviewed window requires an original image capture")
        try:
            saved = library.load_asset(review_id)
        except KeyError:
            library.add_source_scope_review(review_id, capture.session_id, source)
        else:
            if (saved.kind, saved.session_id, saved.metadata["hash"]) != (
                "scope_review",
                capture.session_id,
                digest,
            ):
                raise ValueError(
                    "Existing review identity differs; overwrite is forbidden"
                )
        return library.load_source_scope_review(review_id)
