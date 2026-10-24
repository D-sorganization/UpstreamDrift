"""Static contract for the daily host docker-cache-prune script (UD #12941).

ControlTower's /usr/local/sbin/docker-cache-prune was patched by hand during
#12878 to spare images labelled ``keep`` (explorer / ud-sim / profile images).
The canonical script lives in-repo so deploy jobs cannot drift from fleet policy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = _REPO_ROOT / "scripts" / "docker" / "docker-cache-prune.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    assert SCRIPT.is_file(), f"missing canonical script {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_image_prune_uses_age_window_and_spares_keep_label(script_text: str) -> None:
    """Slow WAN-bound images must survive the weekly all-image prune."""
    assert "docker image prune" in script_text
    assert "until=168h" in script_text
    assert "label!=keep" in script_text
    prune_lines = [
        line for line in script_text.splitlines() if "docker image prune" in line
    ]
    assert len(prune_lines) == 1
    line = prune_lines[0]
    assert "-af" in line or "--all" in line
    assert "label!=keep" in line


def test_builder_prune_bounds_cache_without_dropping_keep_images(
    script_text: str,
) -> None:
    assert "docker builder prune" in script_text
    assert "KEEP_BUILD" in script_text


def test_logs_disk_summary(script_text: str) -> None:
    assert "docker-cache-prune" in script_text
    assert "docker system df" in script_text
