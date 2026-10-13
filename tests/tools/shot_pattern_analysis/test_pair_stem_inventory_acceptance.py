"""Acceptance for paired-figure stems against the saved 384-comparison inventory (#12139).

The defect: paired stems built as ``_slug(f"{kind}_{first_cell}_vs_{second_cell}")``
omitted the ordered source protocols and the physical target, so the 384 saved
comparison records mapped to only 344 stems (40 groups, 80 records) and figures
could overwrite one another.

These tests use only tracked evidence, so they run portably in CI:

* ``protocol_identity_final_qualification_v1/inventory_384_planned_paths.json`` lists
  every saved comparison identity with the stem the qualified renderer planned.
* ``supplement_v5_protocol_collision_stop/collision_findings.json`` is the stop
  receipt (SHA-256 cited in the issue) holding every record of every legacy
  collision group.

They prove the legacy formula reproduces the reported collisions exactly, that the
production stem is unique over the full record set, that it is unchanged from the
qualified (already planned/published) names, and that the remaining acceptance-
review cases (close targets, delimiter ambiguity, fractional labels) cannot collide.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

from src.tools.shot_pattern_analysis import matched_figure_identity as identity
from src.tools.shot_pattern_analysis.matched_supplement_figures import (
    pair_comparison_identity,
    pair_comparison_slug,
    validate_planned_figure_identities,
)

pytestmark = pytest.mark.unit

_VALIDATION = (
    Path(__file__).resolve().parents[3]
    / "docs/research/shot_pattern_analysis/phase2/validation"
)
INVENTORY_PATH = (
    _VALIDATION
    / "protocol_identity_final_qualification_v1/inventory_384_planned_paths.json"
)
COLLISIONS_PATH = (
    _VALIDATION / "supplement_v5_protocol_collision_stop/collision_findings.json"
)
COLLISIONS_SHA256 = "381ef215d750e2c142a99b27d930771605bceb52fdf642cfaf3e09b2a2be1afa"

# The 16 paired exports per comparison (3 academic density, 9 per-pattern social
# density, 3 academic overhead, 1 social overhead).
PAIRED_FORMAT_SUFFIXES = (
    *(
        f"density_{pattern}_social.{ext}"
        for pattern in ("straight", "draw", "fade")
        for ext in ("png", "pdf", "svg")
    ),
    "overhead.png",
    "overhead.pdf",
    "overhead.svg",
    "overhead_social.png",
    "density.png",
    "density.pdf",
    "density.svg",
)


def _legacy_stem(kind: str, first_cell: str, second_cell: str) -> str:
    """The protocol- and target-omitting stem reported in #12139."""
    return identity._slug(f"{kind}_{first_cell}_vs_{second_cell}")


def _inventory_records() -> list[dict[str, Any]]:
    data = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    records = data["records"]
    assert data["summary"]["total_comparisons"] == len(records) == 384
    return records


def _collision_records() -> dict[str, list[dict[str, Any]]]:
    return json.loads(COLLISIONS_PATH.read_text(encoding="utf-8"))["collisions"]


def test_legacy_formula_reproduces_reported_384_to_344_collapse() -> None:
    records = _inventory_records()
    legacy = Counter(
        _legacy_stem(r["comparison_kind"], r["first_cell"], r["second_cell"])
        for r in records
    )
    groups = {stem: n for stem, n in legacy.items() if n > 1}
    assert len(legacy) == 344
    assert len(groups) == 40
    assert sum(groups.values()) == 80


def test_every_saved_record_maps_to_a_distinct_stable_stem() -> None:
    records = _inventory_records()
    identities = [pair_comparison_identity(r) for r in records]
    stems = [pair_comparison_slug(r) for r in records]
    assert len(set(identities)) == 384
    assert len(set(stems)) == 384
    # Stability: the production stem equals the qualified planned stem for every
    # record, so the fix renames no already-planned or published comparison file.
    mismatched = [
        r["index"] for r, s in zip(records, stems, strict=True) if s != r["stem"]
    ]
    assert mismatched == []


def test_all_16_paired_formats_yield_distinct_suffix_safe_paths() -> None:
    assert len(set(PAIRED_FORMAT_SUFFIXES)) == 16
    stems = [pair_comparison_slug(r) for r in _inventory_records()]
    # No dot in a stem, so suffix replacement can never truncate the identity.
    assert all("." not in stem for stem in stems)
    paths = {
        Path("comparisons") / f"{stem}_{suffix}"
        for stem in stems
        for suffix in PAIRED_FORMAT_SUFFIXES
    }
    assert len(paths) == 384 * 16


def test_planned_inventory_passes_fail_closed_preflight() -> None:
    validate_planned_figure_identities(_inventory_records(), [])


def test_collision_stop_receipt_is_the_issue_cited_evidence() -> None:
    digest = hashlib.sha256(COLLISIONS_PATH.read_bytes()).hexdigest()
    assert digest == COLLISIONS_SHA256


def test_each_legacy_collision_group_resolves_to_distinct_qualified_stems() -> None:
    collisions = _collision_records()
    inventory = {pair_comparison_identity(r): r["stem"] for r in _inventory_records()}
    resolved: list[str] = []
    kinds: Counter[str] = Counter()
    for legacy_key, members in collisions.items():
        assert len(members) >= 2
        for record in members:
            first, second = record["sources"]
            assert (
                _legacy_stem(record["comparison_kind"], first["cell"], second["cell"])
                == legacy_key
            )
            stem = pair_comparison_slug(record)
            assert stem == inventory[pair_comparison_identity(record)]
            resolved.append(stem)
        kinds[members[0]["comparison_kind"]] += 1
    assert len(resolved) == 80
    assert len(set(resolved)) == 80
    assert kinds == {"curve_strength": 20, "shaft_rotation_sd": 20}


def test_issue_example_records_carry_protocols_and_target() -> None:
    by_index = {r["index"]: r for r in _inventory_records()}
    primary, control = by_index[119], by_index[216]
    assert _legacy_stem(
        primary["comparison_kind"], primary["first_cell"], primary["second_cell"]
    ) == _legacy_stem(
        control["comparison_kind"], control["first_cell"], control["second_cell"]
    )
    primary_stem, control_stem = (
        pair_comparison_slug(primary),
        pair_comparison_slug(control),
    )
    assert primary_stem != control_stem
    assert "original_primary_1" in primary_stem and "108p027m" in primary_stem
    assert "original_control_1" in control_stem
    assert "driver-control-speed-bounds_1" in control_stem
    assert "178p949m" in control_stem


def _pair(target: float, **overrides: str) -> dict[str, Any]:
    fields = {"p0": "original_primary/1", "c0": "cell_a"}
    fields |= {"p1": "original_primary/1", "c1": "cell_b"}
    fields |= overrides
    return {
        "comparison_kind": "curve_strength",
        "first": {"source_protocol": fields["p0"], "cell": fields["c0"]},
        "second": {"source_protocol": fields["p1"], "cell": fields["c1"]},
        "target_x_m": target,
    }


@pytest.mark.parametrize(
    "target",
    [108.02738606179851, 178.94875782337786, 77.40247724866299, 100.0001],
)
def test_targets_one_ulp_apart_never_share_a_stem(target: float) -> None:
    neighbour = math.nextafter(target, math.inf)
    a, b = pair_comparison_slug(_pair(target)), pair_comparison_slug(_pair(neighbour))
    # The readable prefix rounds to millimetres, the digest must not.
    assert a.rsplit("_", 1)[0] == b.rsplit("_", 1)[0]
    assert a != b


def test_close_targets_rounding_together_stay_distinct_over_a_dense_grid() -> None:
    base = 108.0274
    stems = defaultdict(list)
    for step in range(200):
        target = base + step * 1e-7
        stems[pair_comparison_slug(_pair(target))].append(target)
    assert all(len(targets) == 1 for targets in stems.values())


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"p0": "a_b", "c0": "c"}, {"p0": "a", "c0": "b_c"}),
        ({"c0": "x_vs", "p1": "y"}, {"c0": "x", "p1": "vs_y"}),
        ({"p0": "proto/1", "c0": "cell"}, {"p0": "proto_1", "c0": "cell"}),
        ({"p0": "proto.1"}, {"p0": "protop1"}),
        (
            {"c1": "driver_rotation1.5_combined2"},
            {"c1": "driver_rotation1.6_combined2"},
        ),
    ],
)
def test_ambiguous_human_readable_prefixes_are_disambiguated_by_digest(
    left: dict[str, str], right: dict[str, str]
) -> None:
    a, b = (
        pair_comparison_slug(_pair(100.0, **left)),
        pair_comparison_slug(_pair(100.0, **right)),
    )
    assert a != b
    assert "." not in a and "." not in b


def test_fractional_cell_labels_keep_distinct_final_pair_paths() -> None:
    stems = [
        pair_comparison_slug(_pair(150.0, c1=label))
        for label in ("driver_rotation1.5_combined2", "driver_rotation1.6_combined2")
    ]
    finals = {
        (Path("comparisons") / f"{stem}_{kind}").with_suffix(".png")
        for stem in stems
        for kind in ("density", "overhead")
    }
    assert len(finals) == 4
    assert all(path.name.endswith(("_density.png", "_overhead.png")) for path in finals)


def test_preflight_raises_instead_of_overwriting_when_stems_still_collide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: distinct identities forced onto one stem are rejected."""
    monkeypatch.setattr(identity, "pair_comparison_slug", lambda record: "same")
    with pytest.raises(ValueError, match="collision in planned comparison stems"):
        validate_planned_figure_identities([_pair(100.0), _pair(100.0001)], [])


def test_stem_is_deterministic_and_independent_of_mapping_order() -> None:
    record = _pair(108.02738606179851, p1="driver-control-speed-bounds/1")
    reordered = dict(reversed(list(record.items())))
    assert pair_comparison_slug(record) == pair_comparison_slug(reordered)
    assert pair_comparison_slug(record) == pair_comparison_slug(
        json.loads(json.dumps(record))
    )
