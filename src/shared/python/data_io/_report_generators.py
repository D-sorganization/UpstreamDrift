"""
Report Generators for Output Manager

Analysis report export (JSON, HTML) with provenance embedding.
Extracted from output_manager.py as part of monolith decomposition (#2486).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.contracts import precondition
from ..core.datetime_utils import timestamp_display, timestamp_filename
from ._format_handlers import _make_json_serializer
from .common_utils import get_logger
from .provenance import ProvenanceInfo

logger = get_logger(__name__)


@precondition(
    lambda analysis_data, report_name, report_dir, format_type="json": (
        analysis_data is not None
    ),
    "Analysis data must not be None",
)
@precondition(
    lambda analysis_data, report_name, report_dir, format_type="json": (
        report_name is not None and len(report_name) > 0
    ),
    "Report name must be a non-empty string",
)
def export_analysis_report(
    analysis_data: dict[str, Any],
    report_name: str,
    report_dir: Path,
    format_type: str = "json",
) -> Path:
    """
    Export an analysis report to the given directory.

    Args:
        analysis_data: Analysis results and metadata.
        report_name: Report filename (without extension).
        report_dir: Directory to write the report into.
        format_type: Report format — "json" or "html".

    Returns:
        Path to the exported report file.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp_filename(utc=False)
    filename = f"{report_name}_{timestamp}.{format_type}"
    file_path = report_dir / filename

    provenance = ProvenanceInfo.capture()

    try:
        if format_type == "json":
            report_data = {
                "provenance": {
                    "software": (
                        f"{provenance.software_name} v{provenance.software_version}"
                    ),
                    "timestamp_utc": provenance.timestamp_utc,
                    "git_commit": provenance.git_commit_sha,
                    "git_branch": provenance.git_branch,
                    "git_dirty": provenance.git_is_dirty,
                    "python_version": provenance.python_version,
                    "numpy_version": provenance.numpy_version,
                },
                **analysis_data,
            }

            with open(file_path, "w") as f:
                json.dump(report_data, f, indent=2, default=_make_json_serializer())

        elif format_type == "html":
            html_content = generate_html_report(analysis_data, report_name)
            with open(file_path, "w") as f:
                f.write(html_content)

        logger.info("analysis_report_exported file_path=%s", file_path)
        return file_path

    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error("report_export_failed report_name=%s error=%s", report_name, e)
        raise


def generate_html_report(data: dict[str, Any], title: str) -> str:
    """
    Generate a basic HTML report from a data dictionary.

    Args:
        data: Key/value data to render into the report.
        title: Report title string.

    Returns:
        HTML string.
    """
    timestamp_str = timestamp_display(utc=False)
    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title} - Golf Modeling Suite Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p class="timestamp">Generated: {timestamp_str}</p>

            <h2>Summary</h2>
            <table>
        """

    for key, value in data.items():
        if not isinstance(value, dict | list):
            html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"

    html += f"""
            </table>

            <h2>Detailed Data</h2>
            <pre>{json.dumps(data, indent=2, default=str)}</pre>
        </body>
        </html>
        """

    return html
