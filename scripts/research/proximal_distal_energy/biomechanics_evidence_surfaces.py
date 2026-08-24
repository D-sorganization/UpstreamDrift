"""Generate reviewer and paper surfaces from biomechanics authorities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .biomechanics_evidence_bridge import BRIDGE_REL
from .biomechanics_source_register import SOURCE_REGISTER_REL

ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
REVIEWER_SURFACE_REL = ARTICLE_REL / "BIOMECHANICS_EVIDENCE_BRIDGE.md"
PAPER_FRAGMENT_REL = ARTICLE_REL / "chapters/_biomechanics_evidence_bridge.qmd"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cell(value: Any) -> str:
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _identifier(value: str) -> str:
    return f"`{_cell(value)}`"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    normalized_rows = [[_cell(value) for value in row] for row in rows]
    widths = [
        max(3, len(header), *(len(row[index]) for row in normalized_rows))
        for index, header in enumerate(headers)
    ]

    def formatted_row(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(values)
            )
            + " |"
        )

    lines = [
        formatted_row(headers),
        formatted_row(["-" * width for width in widths]),
    ]
    lines.extend(formatted_row(row) for row in normalized_rows)
    return "\n".join(lines)


def _measurement_table(bridge: dict[str, Any]) -> str:
    return _table(
        [
            "Modality ID",
            "Source Status",
            "Directly Observed",
            "Not Identifiable",
            "Processing Authority",
            "Data Gate",
        ],
        [
            [
                _identifier(record["modality_id"]),
                record["source_status"],
                record["directly_observed"],
                record["not_identifiable"],
                record["processing_method"],
                record["data_gate"],
            ]
            for record in bridge["modalities"]
        ],
    )


def _mechanism_table(bridge: dict[str, Any]) -> str:
    return _table(
        [
            "Mechanism ID",
            "Identifiability",
            "Required Measurements",
            "Human Evidence",
        ],
        [
            [
                _identifier(record["mechanism_id"]),
                record["identifiability"],
                [_identifier(item) for item in record["measurement_requirements"]],
                record["human_evidence_state"],
            ]
            for record in bridge["mechanisms"]
        ],
    )


def _mechanism_discrimination_table(bridge: dict[str, Any]) -> str:
    return _table(
        [
            "Mechanism ID",
            "Observable Discriminator",
            "Competing Explanations",
            "Falsifier",
            "Adverse Case",
        ],
        [
            [
                _identifier(record["mechanism_id"]),
                record["observable_discriminator"],
                record["competing_explanations"],
                record["falsifier"],
                record["adverse_case"],
            ]
            for record in bridge["mechanisms"]
        ],
    )


def _transport_table(bridge: dict[str, Any]) -> str:
    return _table(
        ["Dimension ID", "Current Coverage", "Limitation", "Data Gate"],
        [
            [
                _identifier(record["dimension_id"]),
                record["current_coverage"],
                record["limitation"],
                record["data_gate"],
            ]
            for record in bridge["transportability"]
        ],
    )


def _source_table(register: dict[str, Any]) -> str:
    return _table(
        [
            "Source ID",
            "Role",
            "Population",
            "Apparatus",
            "Estimand",
            "Principal Limitations",
        ],
        [
            [
                f"[{_identifier(record['source_id'])}]({record['stable_url']})",
                record["source_role"],
                record["population"],
                record["apparatus"],
                record["estimand"],
                record["limitations"],
            ]
            for record in register["sources"]
        ],
    )


def _coverage_table(register: dict[str, Any]) -> str:
    return _table(
        ["Domain ID", "Evidence Status", "Current Answer", "Sources", "Data Gate"],
        [
            [
                _identifier(record["domain_id"]),
                record["evidence_status"],
                record["current_answer"],
                [_identifier(item) for item in record["source_ids"]],
                record["data_gate"],
            ]
            for record in register["coverage"]
        ],
    )


def render_reviewer_surface(bridge: dict[str, Any], register: dict[str, Any]) -> str:
    """Render the complete reviewer-facing source and measurement surface."""
    sections = [
        "# Biomechanics Evidence Bridge",
        "",
        "This file is generated from the machine-readable source and measurement "
        "authorities. It is an index, not additional evidence. Human validation "
        f"status: **{bridge['human_validation_status']}**.",
        "",
        "## Interpretation Boundary",
        "",
        bridge["measurement_boundary"],
        "",
        "## Measurement Validity Register",
        "",
        _measurement_table(bridge),
        "",
        "## Mechanism Falsification Map",
        "",
        _mechanism_table(bridge),
        "",
        "## Observable Discriminators and Countermodels",
        "",
        _mechanism_discrimination_table(bridge),
        "",
        "## Transportability Register",
        "",
        _transport_table(bridge),
        "",
        "## Independently Authored Source Register",
        "",
        _source_table(register),
        "",
        "## Anatomical and Study-Domain Coverage",
        "",
        _coverage_table(register),
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 -m scripts.research.proximal_distal_energy.biomechanics_source_register validate",
        "python3 -m scripts.research.proximal_distal_energy.biomechanics_evidence_bridge validate",
        "python3 -m scripts.research.proximal_distal_energy.biomechanics_evidence_surfaces validate",
        "```",
        "",
    ]
    return "\n".join(sections)


def render_paper_fragment(bridge: dict[str, Any], register: dict[str, Any]) -> str:
    """Render a compact, neutral paper section from the same authorities."""
    sections = [
        "## Biomechanics Evidence and Measurement Bridge {#sec-biomechanics-evidence-bridge}",
        "",
        "A model quantity is not promoted to a biological observation merely "
        "because a sensor or inverse problem could be associated with it. The "
        "registered bridge separates calibrated primary observables, derived "
        "model-dependent quantities, structural non-identifiability, practical "
        "qualification gaps, and unavailable measurements. The current human "
        f"validation state is **{bridge['human_validation_status'].replace('_', ' ')}**.",
        "",
        "The complete generated reviewer surface is "
        "[`BIOMECHANICS_EVIDENCE_BRIDGE.md`](../BIOMECHANICS_EVIDENCE_BRIDGE.md); "
        "the underlying authorities are "
        "[`biomechanics_evidence_bridge.json`](../data/biomechanics_evidence_bridge.json) "
        "and [`biomechanics_source_register.json`](../data/biomechanics_source_register.json).",
        "",
        "### Measurement Validity Boundary",
        "",
        _measurement_table(bridge),
        "",
        "### Falsifiable Mechanism Map",
        "",
        _mechanism_table(bridge),
        "",
        "### Observable Discriminators and Countermodels",
        "",
        _mechanism_discrimination_table(bridge),
        "",
        "### Population, Equipment, and Task Transport",
        "",
        _transport_table(bridge),
        "",
        "The source register contains "
        f"{register['summary']['source_count']} independently authored works across "
        f"{register['summary']['coverage_domain_count']} domains. Source coverage "
        "does not remove the externally blocked data gates: no located governed "
        "dataset supplies the complete synchronized bilateral six-axis hand "
        "wrenches, motion, ground, activation, tissue, shaft, and launch signals "
        "required to identify the proposed human mechanisms.",
        "",
    ]
    return "\n".join(sections)


def validate_biomechanics_evidence_surfaces(root: str | Path) -> dict[str, Any]:
    """Fail closed when either generated reviewer surface is stale."""
    root_path = Path(root).resolve()
    bridge = json.loads((root_path / BRIDGE_REL).read_text(encoding="utf-8"))
    register = json.loads((root_path / SOURCE_REGISTER_REL).read_text(encoding="utf-8"))
    expected = {
        root_path / REVIEWER_SURFACE_REL: render_reviewer_surface(bridge, register),
        root_path / PAPER_FRAGMENT_REL: render_paper_fragment(bridge, register),
    }
    stale = [
        str(path)
        for path, content in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    if stale:
        raise ValueError(f"generated biomechanics evidence surfaces are stale: {stale}")
    return {"valid": True, "surface_count": len(expected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    args = parser.parse_args()
    root = _repository_root()
    bridge = json.loads((root / BRIDGE_REL).read_text(encoding="utf-8"))
    register = json.loads((root / SOURCE_REGISTER_REL).read_text(encoding="utf-8"))
    expected = {
        root / REVIEWER_SURFACE_REL: render_reviewer_surface(bridge, register),
        root / PAPER_FRAGMENT_REL: render_paper_fragment(bridge, register),
    }
    if args.action == "write":
        for path, content in expected.items():
            path.write_text(content, encoding="utf-8")
            print(path)
        return
    print(json.dumps(validate_biomechanics_evidence_surfaces(root), indent=2))


if __name__ == "__main__":
    main()
