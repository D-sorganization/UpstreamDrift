"""#15150 (Track C1): the shaft flex rating as a spec and bundle setting.

``with_shaft_model`` passes ``--shaft-flex``, ``--shaft-stiffness-scale`` and
``--shaft-links`` to ``build_shaft_chain``. With none of them the chain spec
is the one built before these options existed; with a rigid club they are
refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.same_input_bundle import build_parser as bundle_parser
from scripts.same_input_bundle import shaft_inputs
from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    DEFAULT_LINKS,
    apply_shaft_chain,
    build_shaft_chain,
    flex_ratings,
    shaft_options,
)
from src.shared.python.motion_matching.execution.spec_builder import (
    build_parser as spec_parser,
)
from src.shared.python.motion_matching.execution.spec_builder import (
    with_shaft_model,
)

pytestmark = pytest.mark.unit

SEED_SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)
RATINGS = ("Ladies", "Senior", "Regular", "Stiff", "X-Stiff")
SCALE = 2.0


def _document() -> dict[str, Any]:
    return json.loads(SEED_SPEC.read_bytes())


def _springs(document: dict[str, Any]) -> np.ndarray:
    """Every chain-joint spring constant, in joint and primitive order."""
    return np.array(
        [
            p["stiffness_n_m_rad"]
            for j in document["joints"]
            if "/Shaft Joint " in j["name"]
            for p in j["primitives"]
        ]
    )


def test_the_ratings_are_ordered_from_flexible_to_stiff() -> None:
    assert flex_ratings() == RATINGS


def test_default_options_reproduce_the_chain_byte_for_byte() -> None:
    document = _document()
    before = json.dumps(apply_shaft_chain(document, build_shaft_chain(DRIVER)))
    for options in (None, {}):
        after = with_shaft_model(document, DRIVER, "chain", options)
        assert json.dumps(after) == before
    spec = SEED_SPEC.read_bytes()
    track = np.ones((2, len(document["coordinate_order"])))
    plain, _ = shaft_inputs(spec, track, "chain")
    explicit, _ = shaft_inputs(spec, track, "chain", {})
    assert plain == explicit == before.encode()


def test_each_rating_stiffens_every_joint_in_order() -> None:
    springs = [
        _springs(with_shaft_model(_document(), DRIVER, "chain", {"flex": f}))
        for f in RATINGS
    ]
    for softer, stiffer in zip(springs, springs[1:], strict=False):
        assert np.all(stiffer > softer)


def test_the_rating_and_scale_are_recorded_in_the_shaft_block() -> None:
    options = {"flex": "Regular", "stiffness_scale": SCALE}
    shaft = with_shaft_model(_document(), DRIVER, "chain", options)["club"]["shaft"]
    assert shaft["flex"] == "Regular"
    assert shaft["stiffness_scale"] == SCALE


def test_the_stiffness_scale_multiplies_every_spring() -> None:
    plain = _springs(with_shaft_model(_document(), DRIVER, "chain"))
    scaled = with_shaft_model(_document(), DRIVER, "chain", {"stiffness_scale": SCALE})
    np.testing.assert_allclose(_springs(scaled), SCALE * plain, rtol=1e-12)


def test_the_link_count_sets_the_number_of_joints() -> None:
    links = DEFAULT_LINKS + 1
    document = with_shaft_model(_document(), DRIVER, "chain", {"n_links": links})
    assert document["club"]["shaft"]["n_links"] == links


@pytest.mark.parametrize(
    ("model", "options", "match"),
    [
        ("rigid", {"flex": "Regular"}, "chain"),
        ("rigid", {"stiffness_scale": SCALE}, "chain"),
        ("chain", {"flex": "Noodle"}, "flex"),
        ("chain", {"stiffness_scale": 0.0}, "stiffness_scale"),
        ("chain", {"stiffness_scale": float("nan")}, "stiffness_scale"),
        ("chain", {"n_links": 0}, "n_links"),
        ("chain", {"torsion": True}, "unknown"),
    ],
)
def test_invalid_options_are_refused(
    model: str, options: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        with_shaft_model(_document(), DRIVER, model, options)


def test_a_rigid_bundle_refuses_a_flex_rating() -> None:
    spec = SEED_SPEC.read_bytes()
    track = np.ones((2, len(_document()["coordinate_order"])))
    with pytest.raises(ValueError, match="chain"):
        shaft_inputs(spec, track, "rigid", {"flex": "Regular"})


def _spec_argv(*flags: str) -> list[str]:
    paths = ("--native", "n", "--osim", "o", "--native-candidate", "c")
    return [*paths, "--stature", "1.8", "--mass", "80", "--output", "out", *flags]


@pytest.mark.parametrize(
    "parse",
    [
        lambda flags: spec_parser().parse_args(_spec_argv(*flags)),
        lambda flags: bundle_parser().parse_args(
            ["export", "--run-dir", "r", "--out", "b.npz", *flags]
        ),
    ],
    ids=["spec_builder", "same_input_bundle"],
)
def test_both_command_lines_parse_the_shaft_options(parse: Any) -> None:
    assert shaft_options(parse([])) == {}
    flags = ["--shaft-flex", "Senior", "--shaft-stiffness-scale", "1.5"]
    args = parse([*flags, "--shaft-links", "6"])
    assert shaft_options(args) == {
        "flex": "Senior",
        "stiffness_scale": 1.5,
        "n_links": 6,
    }
    with pytest.raises(SystemExit):
        parse(["--shaft-flex", "Noodle"])
