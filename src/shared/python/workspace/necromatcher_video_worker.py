"""Clean SDK worker for an owned source-overlay export request."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

import mujoco  # noqa: F401 -- native SDK loaded before workspace rendering helpers

from .artifact_handoff import compute_file_sha256
from .necromatcher import NecromatcherLibrary
from .necromatcher_fit_jobs import fit_execution_stamp
from .necromatcher_video import export_fit_video

logger = logging.getLogger(__name__)


def execute(request_path: Path) -> dict[str, Any]:
    """Validate a local request and export only under its owned run directory."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if (
        not isinstance(request, dict)
        or request.get("kind") != "necromatcher/video-job/1"
        or request_path.name != "request.json"
        or request_path.parent.name != request.get("run_id")
    ):
        raise ValueError("Invalid owned video worker request")
    library = NecromatcherLibrary(request["library_root"])
    if (
        request_path.resolve()
        != (library.root / "video-runs" / request["run_id"] / "request.json").resolve()
    ):
        raise ValueError("Video request lies outside its owned library run")
    stamp = fit_execution_stamp()
    expected = request["execution_stamp"]
    for key in ("source_sha256", "runtime_sha256"):
        if stamp[key] != expected[key]:
            raise ValueError("Video worker source or runtime differs from launch")
    if (
        library.load_asset(request["source_fit_id"]).metadata["hash"]
        != request["source_fit_hash"]
    ):
        raise ValueError("Video source fit changed before rendering")
    fit = library.load_fit(request["source_fit_id"])
    for kind in ("model", "capture"):
        if (
            fit[f"{kind}_id"] != request[f"{kind}_id"]
            or fit[f"{kind}_hash"] != request[f"{kind}_hash"]
        ):
            raise ValueError("Video parent differs from launch")
    destination = request_path.parent / "overlay"
    export_fit_video(
        library,
        request["source_fit_id"],
        destination,
        selected_frames=tuple(request["selected_frames"]),
    )
    current = fit_execution_stamp()
    if any(
        current[key] != expected[key] for key in ("source_sha256", "runtime_sha256")
    ):
        raise ValueError("Video worker source or runtime changed during export")
    return {
        "manifest_sha256": compute_file_sha256(
            destination / "manifest.json"
        ).removeprefix("sha256:")
    }


def main() -> None:
    try:
        result = execute(Path(sys.argv[1]))
        sys.stdout.write(json.dumps(result, allow_nan=False))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        ImportError,
    ):
        logger.exception("Owned native video export failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
