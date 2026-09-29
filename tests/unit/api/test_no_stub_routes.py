"""Architecture guard: API route modules must not ship canned/stub payloads.

Issue #7448 (parity epic #7462): the web app must never present
non-functional features as working. Endpoints either do the real work or
return an honest 501 ``{"detail": ..., "tracking_issue": ...}`` body.

This test greps every module under ``src/api/routes`` for sentinel patterns
of fabricated output that have been removed, preventing regression:

* ``suffix=f".{request.export_format}"`` — the pre-#7448 trajectory export
  wrote a JSON document into a file named with whatever extension the client
  requested, fabricating csv/mat/hdf5/c3d exports.
* Generic canned-data markers that should never appear in route modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROUTES_DIR = Path(__file__).resolve().parents[3] / "src" / "api" / "routes"

# Sentinel substrings of removed fake-output patterns. Keep these literal —
# the test is a plain substring scan so it stays robust to formatting.
CANNED_DATA_SENTINELS: tuple[str, ...] = (
    # JSON content written into a client-chosen file extension (issue #7448).
    'suffix=f".{request.export_format}"',
    # JSON payload stamped with a client-requested format it does not have.
    '{"frames": recorded, "format": request.export_format}',
    # Generic canned-result markers.
    "SAMPLE_METRICS",
    "SAMPLE_STATISTICS",
    "canned_response",
    "fake_data",
)


def _route_modules() -> list[Path]:
    """Return all Python route modules under src/api/routes."""
    assert ROUTES_DIR.is_dir(), f"routes dir not found: {ROUTES_DIR}"
    return sorted(ROUTES_DIR.glob("*.py"))


@pytest.mark.unit
def test_routes_dir_discovered() -> None:
    """The scan must actually cover the route modules (guard the guard)."""
    modules = _route_modules()
    names = {m.name for m in modules}
    assert "physics.py" in names
    assert "analysis_tools.py" in names


@pytest.mark.unit
@pytest.mark.parametrize("sentinel", CANNED_DATA_SENTINELS)
def test_no_canned_data_sentinels_in_routes(sentinel: str) -> None:
    """No route module may contain a known canned/fabricated-output pattern."""
    offenders = [
        module.name
        for module in _route_modules()
        if sentinel in module.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"Canned-data sentinel {sentinel!r} found in route modules "
        f"{offenders}; endpoints must do real work or return an honest 501 "
        "with a tracking_issue (issue #7448)."
    )
