"""Request/response models for the swing report web routes (issue #12330).

The web dialog mirrors the PyQt ``SwingReportDialog`` (#12329): it receives the
Python ``ReportViewModel`` rows, which are text only, so no numeric value of a
withheld quantity can reach the browser (fail-closed policy of #10382).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SwingReportMode = Literal["full", "custom"]


class SwingReportViewRequest(BaseModel):
    """A ``swing-report-v1`` document to show; validated by the route."""

    report: dict[str, Any] = Field(..., description="swing-report-v1 document")


class SwingReportExportRequest(BaseModel):
    """Export a Full or Custom selection of a report as the S4 package."""

    report: dict[str, Any] = Field(..., description="swing-report-v1 document")
    mode: SwingReportMode = Field("full", description="Full report or Custom")
    plot_ids: list[str] = Field(
        default_factory=list, description="Custom selection (ignored for Full)"
    )

    @model_validator(mode="after")
    def _custom_needs_plots(self) -> SwingReportExportRequest:
        if self.mode == "custom" and not self.plot_ids:
            raise ValueError("custom mode needs at least one plot id")
        return self


class SwingReportPlotRow(BaseModel):
    """One plot as the dialog shows it (text only, never data)."""

    plot_id: str
    title: str
    kind: str
    enabled: bool = Field(..., description="False for a withheld plot")
    status_label: str = Field(..., description="'Not Validated' or advertised")
    detail: str = Field(..., description="Uncertainty text or withheld reason")
    unavailable_reason: str | None = None


class SwingReportViewResponse(BaseModel):
    """Rows of a validated report plus the default Custom selection."""

    report_id: str
    rows: list[SwingReportPlotRow]
    default_custom_ids: list[str] = Field(
        ..., description="Enabled plots selected by default in Custom mode"
    )
