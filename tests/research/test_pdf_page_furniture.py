"""Research PDF page furniture retains geometry, labels and drawing state."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

pytestmark = pytest.mark.unit


def test_page_furniture_preserves_title_truncation_and_page_number():
    from scripts.research.shot_pattern_analysis.build_pdf import _page_furniture

    canvas = Mock()
    title = "Research " * 20
    _page_furniture(title, 600, 800, 54)(canvas, SimpleNamespace(page=3))
    assert canvas.method_calls[0] == call.saveState()
    assert canvas.method_calls[-1] == call.restoreState()
    assert canvas.line.call_args_list == [
        call(54, 756, 546, 756),
        call(54, 39, 546, 39),
    ]
    assert canvas.drawString.call_args_list == [
        call(54, 765, title[:85]),
        call(54, 27, "UpstreamDrift  |  Research Reference"),
    ]
    canvas.drawRightString.assert_called_once_with(546, 27, "Page 3")
