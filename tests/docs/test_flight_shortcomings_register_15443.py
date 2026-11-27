"""The flight and impact shortcomings register exists, is linked, and cites real paths.

Epic #15443 child 10: a register that cites a moved or deleted file is stale, so
every repository path in backticks must exist, and a ``path::symbol`` citation
must find the symbol in that file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "physics" / "FLIGHT_AND_IMPACT_SHORTCOMINGS_REGISTER.md"
SOURCE_MAP = ROOT / "docs" / "physics" / "GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md"
_PATH = re.compile(r"`((?:src|tests|docs|rust_core|scripts)/[^`\s]+?)(?:::(\w+))?`")


def _citations() -> list[tuple[str, str | None]]:
    return _PATH.findall(REGISTER.read_text(encoding="utf-8"))


def test_register_exists_and_is_linked() -> None:
    assert REGISTER.is_file()
    assert REGISTER.name in SOURCE_MAP.read_text(encoding="utf-8")
    assert REGISTER.name in (ROOT / "docs" / "index.md").read_text(encoding="utf-8")


def test_register_covers_required_topics() -> None:
    text = REGISTER.read_text(encoding="utf-8").lower()
    for topic in (
        "gear effect",
        "bulge",
        "wind",
        "turf",
        "spin decay",
        "altitude",
        "shaft",
        "rollout",
        "iron cor",
        "launch-monitor",
    ):
        assert topic in text, topic


@pytest.mark.parametrize(("path", "symbol"), sorted(set(_citations())))
def test_cited_path_exists(path: str, symbol: str | None) -> None:
    target = ROOT / path
    assert target.exists(), f"register cites a missing path: {path}"
    if symbol and target.is_file():
        assert symbol in target.read_text(encoding="utf-8"), (path, symbol)
