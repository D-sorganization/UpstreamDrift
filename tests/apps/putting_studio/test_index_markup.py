"""The studio page loads offline and claims nothing it cannot support (#14141)."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_INDEX = Path(__file__).resolve().parents[3] / "src/apps/putting_studio/index.html"
# Result claims from the source handoff that no outside data supports yet.
_UNSUPPORTED = ("+2.42", "90.3%", "83.2%", "85.7%", "92.1%", "(1 &minus; (b/R")


class _Assets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.inline_scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = dict(attrs)
        url = found.get("src") or (found.get("href") if tag == "link" else None)
        if url:
            self.urls.append(url)
        elif tag == "script":
            self.inline_scripts += 1


def _page() -> str:
    return _INDEX.read_text(encoding="utf-8")


def test_page_loads_only_relative_assets_and_no_inline_script() -> None:
    assets = _Assets()
    assets.feed(_page())
    assert assets.urls, "the page must reference its vendored assets"
    for url in assets.urls:
        assert not re.match(r"^([a-z]+:)?//", url), f"remote load: {url}"
    assert assets.inline_scripts == 0


def test_page_states_that_values_are_unvalidated_model_output() -> None:
    notice = re.search(r'id="validationNotice"[^>]*>(.*?)</p>', _page(), re.DOTALL)
    assert notice is not None
    assert "not validated" in notice.group(1)


@pytest.mark.parametrize("claim", _UNSUPPORTED)
def test_page_carries_no_unsupported_result_claim(claim: str) -> None:
    assert claim not in _page()
