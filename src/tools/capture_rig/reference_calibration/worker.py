"""Isolated player-calibration IPC; launch with ``python -I worker.py``.

The application owns a separate Sidekick cluster. This process resolves the
selected Tools tree before loading any application modules, preserving both
owners' module identities. No imports or aliases are changed in the GUI process.
"""

from __future__ import annotations

import json
from importlib import metadata
import logging
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

MAX_REQUEST_BYTES = 4 * 1024 * 1024
SCHEMA = "capture-reference-worker/1"
LOGGER = logging.getLogger(__name__)


PROVIDER_FILE = Path("shared/python/sidekick/lab/mocap/reference_placements.py")
WORKER_FILE = Path("src/tools/capture_rig/reference_calibration/worker.py")


def _provider_root(root: Path) -> Path:
    vendor = root / "vendor" / "ud-tools" / "src"
    if vendor.exists():
        return vendor
    try:
        distribution = metadata.distribution("upstream-drift")
    except metadata.PackageNotFoundError as exc:
        raise ValueError("The selected Tools provider is unavailable") from exc
    files = {str(path).replace("\\", "/") for path in distribution.files or ()}
    if Path(distribution.locate_file("")).resolve() != root or not {
        PROVIDER_FILE.as_posix(),
        WORKER_FILE.as_posix(),
    }.issubset(files):
        raise ValueError("The installed provider does not belong to this application")
    return root


def _configure_provider() -> None:
    root = Path(__file__).resolve().parents[4]
    provider = _provider_root(root)
    if not (provider / PROVIDER_FILE).is_file():
        raise ValueError(
            "The selected Tools provider does not support reference placements"
        )
    paths = [
        provider,
        provider / "shared/python",
        provider / "python/src",
        root,
        root / "src",
    ]
    sys.path[:0] = list(dict.fromkeys(str(path) for path in paths))


def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    # Resolve the Tools family first in this fresh process, before UD's separate
    # Sidekick package can enter the module cache.
    from shared.python.sidekick.lab.mocap.reference_placements import common_reference

    action = request.get("action")
    if action == "reuse_choices":
        from src.tools.capture_rig.reference_calibration.reuse_catalog import (
            list_reviewed_layouts,
        )

        return list_reviewed_layouts(request)
    if action in {"inspect_reuse", "adopt_layout"}:
        from src.tools.capture_rig.reference_calibration.reuse import (
            inspect_reuse,
            adopt_layout,
        )

        return (
            inspect_reuse(request)
            if action == "inspect_reuse"
            else adopt_layout(request)
        )
    if action == "catalog":
        return {
            "targets": [
                asdict(common_reference(key))
                for key in ("us-letter", "a4", "yardstick", "meter-stick")
            ]
        }
    if action in {"frame", "mark", "target"}:
        from src.tools.capture_rig.reference_calibration.operations import (
            extract_frame,
            mark_sample,
            add_target,
        )

        if action == "target":
            return add_target(request)
        return extract_frame(request) if action == "frame" else mark_sample(request)
    if action in {"solve", "accept"}:
        from src.tools.capture_rig.reference_calibration.solver import (
            solve_reference,
            accept_result,
        )

        return solve_reference(request) if action == "solve" else accept_result(request)
    if action == "result":
        from src.tools.capture_rig.reference_calibration.results import load_result

        return load_result(request)
    if action not in {"load", "validate", "revise", "save"}:
        raise ValueError("Unsupported reference-workspace action")
    from src.tools.capture_rig.reference_calibration.session import (
        ReferenceSession,
        load_revision,
        save_revision,
    )

    if action == "load":
        from src.tools.capture_rig.reference_calibration.operations import (
            verify_samples,
        )

        revision = UUID(request["revision_id"])
        path = Path(request["workspace"]) / "reference_calibration" / f"{revision}.json"
        session = load_revision(path, capture_id=request["capture_id"])
        verify_samples(Path(request["workspace"]), session)
        return session.model_dump(mode="json")
    session = ReferenceSession.model_validate(request["session"])
    if action == "revise":
        session = session.revise(**request["changes"])
    elif action == "save":
        from src.tools.capture_rig.reference_calibration.operations import (
            verify_samples,
        )

        verify_samples(Path(request["workspace"]), session)
        save_revision(
            Path(request["workspace"]), session, capture_id=request["capture_id"]
        )
    elif action != "validate":
        raise ValueError("Unsupported reference-workspace action")
    return session.model_dump(mode="json")


def _read_request(path: Path | None) -> bytes:
    if path is not None:
        with path.open("rb") as stream:
            return stream.read(MAX_REQUEST_BYTES + 1)
    input_stream = sys.stdin.buffer
    return input_stream.read(MAX_REQUEST_BYTES + 1)


def main() -> int:
    """Accept one bounded JSON request and emit one JSON response."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    arguments = parser.parse_args()
    try:
        _configure_provider()
        raw = _read_request(arguments.request)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Reference request exceeds the document limit")
        request = json.loads(raw)
        allowed = {
            "schema_version",
            "action",
            "session",
            "changes",
            "workspace",
            "capture_id",
            "revision_id",
            "parameters",
        }
        if (
            not isinstance(request, dict)
            or request.get("schema_version") != SCHEMA
            or set(request) - allowed
        ):
            raise ValueError("Unsupported reference-workspace request schema")
        response = {"ok": True, "result": _dispatch(request)}
        code = 0
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        LOGGER.warning("Reference workspace request failed: %s", exc)
        response = {"ok": False, "error": str(exc)}
        code = 2
    rendered = json.dumps(response, allow_nan=False) + "\n"
    if arguments.response is not None:
        try:
            arguments.response.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Cannot publish reference response: %s", exc)
            return 2
    else:
        sys.stdout.write(rendered)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
