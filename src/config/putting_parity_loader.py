"""Putting feature parity matrix loader (#14456, epic #14452).

Loads ``putting_parity_matrix.json``: every advanced putting feature mapped
to the shared core/API, PyQt6, React, Tauri and the offline studio. A cell
is a compact string, ``"<status> [#issue] | <path>, <path> | <note>"``, so
the JSON stays short under prettier.

Design by Contract:
    Preconditions: every feature has all of :data:`SURFACES`; every status
    is in :data:`VALID_STATUSES`; an issue, when given, is ``#<positive int>``.
    Postconditions: feature ids are unique; the matrix is immutable.
    Regenerate the doc: ``python -m src.config.putting_parity_loader --write``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).parent
MATRIX_PATH = CONFIG_DIR / "putting_parity_matrix.json"
DOC_PATH = CONFIG_DIR.parents[1] / "docs" / "development" / "putting_parity_matrix.md"

SURFACES = ("core_api", "pyqt6", "react", "tauri", "studio")
VALID_STATUSES = frozenset({"present", "partial", "missing"})
# One word list reads clearer than 13 one-per-line literals.
_FEATURE_WORDS = "units variance_channels dispersion impact_model strike_location"
_FEATURE_WORDS += " counter_torque balance_type green_types optimal_aim"
_FEATURE_WORDS += " strokes_gained scenarios json_io camera_playback"
REQUIRED_FEATURES = tuple(_FEATURE_WORDS.split())
_HEADERS = ("Core/API", "PyQt6", "React", "Tauri", "Studio")


@dataclass(frozen=True)
class Cell:
    """One feature on one surface."""

    status: str
    issue: int | None
    evidence: tuple[str, ...]
    note: str

    @classmethod
    def parse(cls, text: str, where: str) -> Cell:
        """Parse ``"<status> [#issue] | paths | note"``; raise on bad input."""
        parts = [part.strip() for part in text.split("|")]
        if len(parts) != 3:
            raise ValueError(f"{where}: expected 'status | paths | note': {text!r}")
        head, paths, note = parts
        status, _, issue_text = head.partition(" ")
        if status not in VALID_STATUSES:
            raise ValueError(f"{where}: unknown status {status!r}")
        issue = None
        if issue_text:
            if not (issue_text.startswith("#") and issue_text[1:].isdigit()):
                raise ValueError(f"{where}: issue must be '#<n>', got {issue_text!r}")
            issue = int(issue_text[1:])
        evidence = tuple(p.strip() for p in paths.split(",") if p.strip())
        return cls(status, issue, evidence, note)


@dataclass(frozen=True)
class Feature:
    """An advanced putting feature and its five surface cells."""

    feature_id: str
    title: str
    cluster: str
    cells: dict[str, Cell]


@dataclass(frozen=True)
class PuttingParityMatrix:
    """The whole matrix, in file order."""

    features: tuple[Feature, ...]

    @classmethod
    def load(cls, path: Path = MATRIX_PATH) -> PuttingParityMatrix:
        raw = json.loads(path.read_text(encoding="utf-8"))
        features = []
        for feature_id, entry in raw["features"].items():
            if tuple(entry["cells"]) != SURFACES:
                raise ValueError(f"{feature_id}: cells must be exactly {SURFACES}")
            cells = {
                surface: Cell.parse(text, f"{feature_id}/{surface}")
                for surface, text in entry["cells"].items()
            }
            features.append(
                Feature(feature_id, entry["title"], entry["cluster"], cells)
            )
        return cls(tuple(features))

    def feature_ids(self) -> list[str]:
        return [feature.feature_id for feature in self.features]

    def iter_cells(self) -> Iterator[tuple[str, str, Cell]]:
        for feature in self.features:
            for surface, cell in feature.cells.items():
                yield feature.feature_id, surface, cell

    def tauri_note(self) -> str:
        """The shared Tauri note (the shell wraps React for every feature)."""
        cell = self.features[0].cells["tauri"]
        return f"{', '.join(f'`{p}`' for p in cell.evidence)}. {cell.note}"

    def cell(self, feature_id: str, surface: str) -> Cell:
        for fid, sur, cell in self.iter_cells():
            if (fid, sur) == (feature_id, surface):
                return cell
        raise KeyError(f"{feature_id}/{surface}")


def _badge(cell: Cell) -> str:
    return cell.status + (f" (#{cell.issue})" if cell.issue else "")


def render_markdown(matrix: PuttingParityMatrix) -> str:
    """Render the human-readable matrix (deterministic)."""
    lines = [
        "# Putting Feature Parity Matrix",
        "",
        "Generated from `src/config/putting_parity_matrix.json` by",
        "`python -m src.config.putting_parity_loader --write`. Do not edit by hand.",
        "Epic #14452. Tauri wraps the React build, so it inherits React's status.",
        "",
        "| Feature | Cluster | " + " | ".join(_HEADERS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in _HEADERS) + " |",
    ]
    for f in matrix.features:
        badges = " | ".join(_badge(f.cells[s]) for s in SURFACES)
        lines.append(f"| {f.title} | ({f.cluster}) | {badges} |")
    lines += ["", "## Evidence", "", f"Tauri, every feature: {matrix.tauri_note()}", ""]
    for f in matrix.features:
        lines.append(f"**{f.title}**")
        for surface, header in zip(SURFACES, _HEADERS, strict=True):
            if surface == "tauri":
                continue
            cell = f.cells[surface]
            paths = ", ".join(f"`{p}`" for p in cell.evidence) or "none"
            lines.append(f"- {header}: {_badge(cell)}. {paths}. {cell.note}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="regenerate the doc")
    args = parser.parse_args(argv)
    rendered = render_markdown(PuttingParityMatrix.load())
    if args.write:
        DOC_PATH.write_text(rendered, encoding="utf-8")
        return 0
    return 0 if DOC_PATH.read_text(encoding="utf-8") == rendered else 1


if __name__ == "__main__":
    raise SystemExit(main())
