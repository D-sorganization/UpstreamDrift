"""Analysis reports must create a supported artifact or reject the request."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.shared.python.data_io._report_generators import export_analysis_report
from src.shared.python.data_io.output_manager import OutputManager
from src.shared.python.data_io.provenance import ProvenanceInfo

pytestmark = pytest.mark.unit


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Mock:
    capture = Mock(
        return_value=ProvenanceInfo(
            timestamp_utc="2026-11-16T00:00:00Z",
            timestamp_local="2026-11-16T00:00:00+00:00",
        )
    )
    monkeypatch.setattr(ProvenanceInfo, "capture", capture)
    return capture


@pytest.mark.parametrize("format_type", ["pdf", "", "JSON"])
@pytest.mark.parametrize("through_manager", [False, True])
def test_unsupported_report_format_has_no_export_side_effects(
    tmp_path: Path, capture: Mock, format_type: str, through_manager: bool
) -> None:
    report_dir = tmp_path / "reports"
    with pytest.raises(ValueError, match="(?i)unsupported.*format"):
        if through_manager:
            OutputManager(tmp_path).export_analysis_report(
                {"count": 7}, "report", format_type
            )
        else:
            export_analysis_report({"count": 7}, "report", report_dir, format_type)

    assert not report_dir.exists()
    capture.assert_not_called()


@pytest.mark.parametrize("format_type", ["json", "html"])
def test_supported_report_is_written(
    tmp_path: Path, capture: Mock, format_type: str
) -> None:
    report = OutputManager(tmp_path).export_analysis_report(
        {"count": 7}, "report", format_type
    )
    assert report.is_file()
    assert report.suffix == f".{format_type}"
    content = report.read_text(encoding="utf-8")
    if format_type == "json":
        assert json.loads(content)["count"] == 7
    else:
        assert "<td>7</td>" in content
    capture.assert_called_once_with()
