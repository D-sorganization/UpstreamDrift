"""Verify the supplied WSCG decks and extract cached chart observations."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

from defusedxml import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = (
    REPO_ROOT
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "sources"
    / "wscg_2024"
)
DATA_DIR = SOURCE_DIR.parent.parent / "data"
CHART_DECK = SOURCE_DIR / "Charts.pptx"
WSCG_DECK = SOURCE_DIR / (
    "WeT21_Evaluation_of_the_Effect_of_Momentum_on_Interaction_Forces_"
    "in_a_Linked_System.pptx"
)
EXPECTED_HASHES = {
    CHART_DECK.name: "17dff3b767ef432d76aacaa7be2ce24339d24755b00da293ced4f4ef62a307a5",
    WSCG_DECK.name: "b1d53150a9669744842ac5bb4200522fbd5aa4697092a0f6e88bcd4da687ea0a",
}
CHART_NS = {
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "c15": "http://schemas.microsoft.com/office/drawing/2012/chart",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> dict[str, str]:
    """Verify registered source hashes and return the observed mapping."""
    observed = {name: _sha256(SOURCE_DIR / name) for name in EXPECTED_HASHES}
    mismatches = {
        name: value
        for name, value in observed.items()
        if value != EXPECTED_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"WSCG source hash mismatch: {mismatches}")
    return observed


def _cache_values(node: ElementTree.Element, path: str) -> list[float]:
    points = node.findall(path, CHART_NS)
    indexed = sorted(
        (
            (
                int(point.attrib["idx"]),
                float(point.find("c:v", CHART_NS).text),
            )
            for point in points
        ),
        key=lambda item: item[0],
    )
    return [value for _, value in indexed]


def extract_series() -> list[dict[str, float | int | str]]:
    """Return long-form observations from every cached series in chart 1."""
    with zipfile.ZipFile(CHART_DECK) as archive:
        root = ElementTree.fromstring(archive.read("ppt/charts/chart1.xml"))
    rows: list[dict[str, float | int | str]] = []
    series_nodes = root.findall(".//c:ser", CHART_NS)
    series_nodes.extend(root.findall(".//c15:ser", CHART_NS))
    for series in series_nodes:
        name_node = series.find("c:tx/c:strRef/c:strCache/c:pt/c:v", CHART_NS)
        if name_node is None:
            name_node = series.find("c:tx/c:v", CHART_NS)
        if name_node is None or name_node.text is None:
            continue
        x_values = _cache_values(series, "c:xVal/c:numRef/c:numCache/c:pt")
        y_values = _cache_values(series, "c:yVal/c:numRef/c:numCache/c:pt")
        if not x_values:
            x_values = _cache_values(series, "c:cat/c:numRef/c:numCache/c:pt")
        if not y_values:
            y_values = _cache_values(series, "c:val/c:numRef/c:numCache/c:pt")
        if len(x_values) != len(y_values):
            raise ValueError(f"Unequal chart caches for {name_node.text}")
        rows.extend(
            {
                "series": name_node.text,
                "sample_index": index,
                "time_s": x_value,
                "value": y_value,
            }
            for index, (x_value, y_value) in enumerate(
                zip(x_values, y_values, strict=True)
            )
        )
    if not rows:
        raise ValueError("No cached series found in Charts.pptx")
    return rows


def main() -> None:
    """Verify the source package and write deterministic data artifacts."""
    hashes = verify_sources()
    rows = extract_series()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "wscg_2024_hand_force_series.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("series", "sample_index", "time_s", "value")
        )
        writer.writeheader()
        writer.writerows(rows)
    series_names = sorted({str(row["series"]) for row in rows})
    provenance = {
        "source_hashes_sha256": hashes,
        "source_chart": "ppt/charts/chart1.xml",
        "row_count": len(rows),
        "series_count": len(series_names),
        "series_names": series_names,
        "units": {
            "time_s": "s",
            "force_series": "N",
            "Wrist Torque": "N m",
        },
        "interpretation": "Cached source-chart values; not independently measured data.",
    }
    with (DATA_DIR / "wscg_2024_source_provenance.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
