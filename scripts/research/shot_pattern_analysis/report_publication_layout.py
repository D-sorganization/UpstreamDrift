"""Markdown layout helpers for the source-bound shot-pattern report."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .report_tables import resolved_selection_lines


def render_club_appendices(
    interpretation: dict, *, report_filename: str = "REPORT.md"
) -> dict[str, str]:
    """Render per-club figure inventories and keyed resolved comparison rows."""
    clubs = ("driver", "seven_iron", "pitching_wedge")
    figure_links = interpretation.get("figure_links", [])
    resolved_cells = interpretation.get("resolved_selected_cells", [])
    appendices: dict[str, str] = {}
    for club in clubs:
        lines = [
            f"# {club.replace('_', ' ').title()}: Resolved Results and Figure Inventory",
            "",
            f"[Return to the final conditional simulation report]({report_filename}).",
            "",
            "## Verified Academic and Social Figures",
            "",
            "Figure links retain their source protocol and population type. Individual cell densities use individually supported endpoints, while paired densities use the shared six-population supported-index intersection; displayed point overlays and overhead paths are sampled, as labeled in each image. A 95% highest-density region is not a confidence interval for the mean.",
            "",
        ]
        club_links = [figure for figure in figure_links if figure["club"] == club]
        for figure in club_links:
            path = Path(figure["relative_path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("figure link must remain relative to the report")
            label = (
                f"{figure['club'].replace('_', ' ').title()} "
                f"{figure['kind'].replace('_', ' ').title()} "
                f"({figure['presentation'].title()}, {figure['source_protocol']}, "
                f"{figure['source_kind'].replace('_', ' ')})"
            )
            if figure.get("physical_pattern"):
                label += f" · {figure['physical_pattern']}"
            if figure.get("comparison_kind"):
                label += " · " + str(figure["comparison_kind"]).replace(
                    "_", " "
                ).title().replace("Sd", "SD")
                label += " · " + " → ".join(figure["source_cells"])
            lines.append(f"- [{label}]({path.as_posix()})")
        if not club_links:
            lines.append("No qualified figure asset is available for this club.")
        lines.extend(("", "## Resolved Zero-Lean Curve and Variability Comparison", ""))
        lines.extend(resolved_selection_lines(resolved_cells, club_id=club))
        appendix = "\n".join(lines)
        if len(appendix.encode("utf-8")) >= 51_200:
            raise ValueError(f"{club} report appendix exceeds the 51,200-byte limit")
        appendices[club] = appendix
    return appendices


def assert_markdown_output_sizes(
    outputs: Mapping[str, str], *, max_bytes: int = 51_200
) -> None:
    """Keep each generated Markdown publication artifact within the docs budget."""
    oversized = {
        name: len(content.encode("utf-8"))
        for name, content in outputs.items()
        if len(content.encode("utf-8")) >= max_bytes
    }
    if oversized:
        details = ", ".join(f"{name}: {size} bytes" for name, size in oversized.items())
        raise ValueError(
            f"report Markdown output exceeds {max_bytes}-byte budget: {details}"
        )


def _read_snapshot_payload(repository: Path, snapshot: Mapping[str, object]) -> bytes:
    """Decode one archived payload and verify its storage and payload identities."""
    path = (repository / str(snapshot["archive_path"])).resolve()
    if not path.is_relative_to(repository.resolve()):
        raise ValueError("report archive path escapes the repository")
    stored = path.read_bytes()
    if (
        len(stored) != snapshot["storage_size_bytes"]
        or hashlib.sha256(stored).hexdigest() != snapshot["storage_sha256"]
    ):
        raise ValueError("report archive storage failed integrity check")
    encoding = snapshot["encoding"]
    if encoding == "identity":
        payload = stored
    elif encoding == "gzip":
        try:
            payload = gzip.decompress(stored)
        except OSError as exc:
            raise ValueError("report archive gzip payload is invalid") from exc
    else:
        raise ValueError(f"unsupported report archive encoding: {encoding}")
    if (
        len(payload) != snapshot["payload_size_bytes"]
        or hashlib.sha256(payload).hexdigest() != snapshot["payload_sha256"]
    ):
        raise ValueError("report archive payload failed integrity check")
    return payload


def split_archive_provenance(report_root: Path) -> tuple[str | None, str | None]:
    """Validate archived source and receipt payloads and return their identities."""
    manifest_path = Path(report_root) / "REPORT_PUBLICATION_SPLIT_ARCHIVE.json"
    if not manifest_path.is_file():
        return None, None
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "matched-report-split-archive/2":
        raise ValueError("report split archive manifest has an unsupported schema")
    snapshots = manifest.get("snapshots", [])
    for snapshot in snapshots:
        _read_snapshot_payload(Path(__file__).resolve().parents[3], snapshot)
    receipt = next(
        (
            item
            for item in snapshots
            if item.get("original_path") == "REPORT_PUBLICATION_RECEIPT.json"
        ),
        None,
    )
    if receipt is None:
        raise ValueError("report split archive manifest lacks its original receipt")
    return receipt["payload_sha256"], hashlib.sha256(manifest_bytes).hexdigest()
