"""Worker-only result history and the source bytes reviewed by the player."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from .operations import verify_samples
from .session import ReferenceSession

MAX_RESULT_BYTES = 4 * 1024 * 1024


def load_result(request: dict[str, Any]) -> dict[str, Any]:
    """Reopen a result only with its unchanged capture and observation revision."""
    root = Path(request["workspace"])
    session = ReferenceSession.model_validate(request["session"])
    result_id = UUID(request["parameters"]["result_id"])
    path = root / "reference_calibration" / "results" / f"{result_id}.json"
    with path.open("rb") as stream:
        data = stream.read(MAX_RESULT_BYTES + 1)
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError("Camera result exceeds its document limit")
    document = json.loads(data)
    if (
        document.get("schema_version") != "capture-reference-solve/1"
        or document.get("capture_id") != session.capture_id
        or document.get("reference_revision_id") != str(session.revision_id)
        or document.get("layout_id") != str(result_id)
        or document.get("session_sha256")
        != hashlib.sha256(session.model_dump_json().encode()).hexdigest()
    ):
        raise ValueError(
            "Open the matching capture and reference revision before reviewing this estimate"
        )
    revision = root / "reference_calibration" / f"{session.revision_id}.json"
    with revision.open("rb") as stream:
        revision_data = stream.read(MAX_RESULT_BYTES + 1)
    if len(revision_data) > MAX_RESULT_BYTES or hashlib.sha256(
        revision_data
    ).hexdigest() != document.get("revision_file_sha256"):
        raise ValueError("Saved reference observations changed; estimate again")
    verify_samples(root, session)
    return {
        "result_path": path.relative_to(root).as_posix(),
        "result_sha256": hashlib.sha256(data).hexdigest(),
        "result": document,
    }
