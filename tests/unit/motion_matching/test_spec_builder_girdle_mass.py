"""#15150: the spec builder passes the shoulder-girdle mass rule through.

``anthropometric_geometry.build_upper_body`` already takes
``shoulder_girdle_mass`` (#15279); these tests cover its path through
``spec_builder`` (function, receipt and command line).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.anthropometric_geometry import (
    GIRDLE_MASS_ADDED,
    GIRDLE_MASS_FROM_UPPER_TRUNK,
    GIRDLE_MASS_RULES,
)
from src.shared.python.motion_matching.anthropometry import segment_parameters
from src.shared.python.motion_matching.execution import assets, spec_builder
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

pytestmark = pytest.mark.unit

STATURE_M = 1.80
MASS_KG = 84.0
SETTING = "shoulder_girdle_mass"


def _solid_mass(document: dict[str, Any], suffix: str) -> float:
    return sum(
        s["mass_kg"]
        for b in document["bodies"]
        for s in b["solids"]
        if s["name"].endswith("/" + suffix)
    )


def _total_mass(document: dict[str, Any]) -> float:
    return sum(s["mass_kg"] for b in document["bodies"] for s in b["solids"])


def _compose(**kwargs: Any) -> dict[str, Any]:
    built = spec_builder.compose_anthropometric_document(STATURE_M, MASS_KG, **kwargs)
    return built.document


def _run_cli(out: Path, *flags: str) -> tuple[dict[str, Any], dict[str, Any]]:
    code = spec_builder.main(
        [
            "--native",
            str(assets.get_native_geometry_spec(None)),
            "--osim",
            str(assets.get_opensim_model(None)),
            "--native-candidate",
            str(assets.get_candidate_geometry_spec(None)),
            "--stature",
            str(STATURE_M),
            "--mass",
            str(MASS_KG),
            "--output",
            str(out),
            "--name",
            "girdle_case",
            *flags,
        ]
    )
    assert code == 0
    spec = json.loads((out / "girdle_case.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (out / "build_receipt_girdle_case.json").read_text(encoding="utf-8")
    )
    return spec, receipt


def test_the_default_spec_is_the_added_rule_and_records_nothing() -> None:
    plain = _compose()
    explicit = _compose(**{SETTING: GIRDLE_MASS_ADDED})
    assert canonical_sha256(explicit) == canonical_sha256(plain)
    assert SETTING not in plain["subject"]


def test_the_trunk_gives_up_the_girdle_mass_through_the_builder() -> None:
    plain = _compose()
    shared = _compose(**{SETTING: GIRDLE_MASS_FROM_UPPER_TRUNK})
    table = segment_parameters(STATURE_M, MASS_KG, "upper_trunk").mass_kg
    girdle = _solid_mass(plain, "clavicle")
    assert girdle > 0.5
    assert _solid_mass(plain, "upper_trunk") == pytest.approx(table)
    assert _solid_mass(shared, "clavicle") == pytest.approx(girdle)
    assert _solid_mass(shared, "upper_trunk") + girdle == pytest.approx(table)
    # Unscaled legs: the total falls by exactly the double-counted girdle.
    assert _total_mass(plain) - _total_mass(shared) == pytest.approx(girdle)
    assert shared["subject"][SETTING] == GIRDLE_MASS_FROM_UPPER_TRUNK


def test_a_matched_subject_mass_is_conserved_under_either_rule() -> None:
    """With ``match_subject_mass`` the legs carry what the upper body does not,
    so the body without the club stays at the subject mass (#12199)."""
    for rule in GIRDLE_MASS_RULES:
        document = _compose(match_subject_mass=True, **{SETTING: rule})
        body = _total_mass(document) - document["club"]["total_mass_kg"]
        assert body == pytest.approx(MASS_KG, abs=1e-6)


def test_an_unknown_rule_is_refused_naming_the_allowed_rules() -> None:
    with pytest.raises(ValueError, match=SETTING) as raised:
        _compose(**{SETTING: "from_the_head"})
    for rule in GIRDLE_MASS_RULES:
        assert rule in str(raised.value)


def test_the_command_line_sets_the_rule_and_the_receipt_records_it(
    tmp_path: Path,
) -> None:
    plain, plain_receipt = _run_cli(tmp_path / "plain")
    shared, receipt = _run_cli(
        tmp_path / "shared", "--shoulder-girdle-mass", GIRDLE_MASS_FROM_UPPER_TRUNK
    )
    girdle = _solid_mass(plain, "clavicle")
    assert _solid_mass(plain, "upper_trunk") - _solid_mass(
        shared, "upper_trunk"
    ) == pytest.approx(girdle)
    assert shared["subject"][SETTING] == GIRDLE_MASS_FROM_UPPER_TRUNK
    assert receipt["subject"][SETTING] == GIRDLE_MASS_FROM_UPPER_TRUNK
    assert plain_receipt["total_mass_kg"] - receipt["total_mass_kg"] == pytest.approx(
        girdle
    )
    assert SETTING not in plain["subject"]
    assert SETTING not in plain_receipt["subject"]
    assert canonical_sha256(plain) == canonical_sha256(_compose())


def test_the_command_line_refuses_an_unknown_rule(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _run_cli(tmp_path / "bad", "--shoulder-girdle-mass", "from_the_head")
