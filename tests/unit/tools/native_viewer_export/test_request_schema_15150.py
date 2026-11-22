"""One request schema derived from the exporter parser (#15150 G7, C11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.capability_state import VALID_MATURITY_LEVELS
from src.tools.native_viewer_export import request_schema as rs
from src.tools.native_viewer_export.cli import build_parser

pytestmark = pytest.mark.unit


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "b.npz").write_bytes(b"")
    monkeypatch.setenv("UD_NATIVE_EXPORT_INPUT_ROOTS", str(tmp_path / "in"))
    monkeypatch.setenv("UD_NATIVE_EXPORT_OUTPUT_ROOT", str(tmp_path / "out"))
    return tmp_path


def _request(root: Path, **extra: object) -> dict[str, object]:
    options = {"bundle": str(root / "in" / "b.npz"), "out": "clips", "swing": "s1"}
    return {"options": {**options, **extra}}


def test_every_cli_option_is_in_the_schema_or_explicitly_excluded() -> None:
    names = {o["name"]: o for o in rs.describe({})["options"]}
    excluded = rs.load_schema_settings()["remote_exclusions"]
    for action in build_parser()._actions:
        if action.dest == "help":
            continue
        assert action.dest in names, f"{action.option_strings} missing from schema"
        remote = names[action.dest]["remote"]
        assert (remote == "excluded") == (action.dest in excluded)


def test_a_new_flag_appears_without_edits_and_shaft_flags_group() -> None:
    parser = build_parser()
    parser.add_argument("--shaft-flex", default=None, help="chain flex rating")
    parser.add_argument("--brand-new-flag", type=int, default=3)
    options = {o["name"]: o for o in rs.option_schema(parser)}
    assert options["shaft_flex"]["group"] == "Shaft"
    assert options["brand_new_flag"]["group"] == "Other"
    assert options["brand_new_flag"]["default"] == 3


def test_bundle_build_stage_lists_the_shaft_model_under_shaft() -> None:
    staged = [o for o in rs.describe({})["options"] if o["stage"] == "bundle_build"]
    shaft = {o["name"]: o for o in staged}
    assert shaft["shaft_model"]["group"] == "Shaft"
    assert "chain" in shaft["shaft_model"]["choices"]
    assert all(o["group"] == "Shaft" for o in staged)


def test_groups_carry_valid_maturity_and_sources() -> None:
    groups = {g["name"]: g for g in rs.describe({})["groups"]}
    assert groups["Ball Flight"]["maturity"] == "experimental"
    for group in groups.values():
        assert group["maturity"] in VALID_MATURITY_LEVELS
        assert group["sources"]


def test_min_maturity_hides_unready_groups() -> None:
    shown = rs.describe({"min_maturity": "beta"})
    assert {g["maturity"] for g in shown["groups"]} <= {"beta", "stable"}
    assert "ball_flight" not in {o["name"] for o in shown["options"]}
    with pytest.raises(ValueError, match="min_maturity"):
        rs.describe({"min_maturity": "ready"})


def test_validate_returns_a_command_line_the_parser_accepts(roots: Path) -> None:
    result = rs.validate(_request(roots, palm_thumb_force=False, fps=30))
    argv = result["argv"]
    args = build_parser().parse_args(argv)
    assert args.palm_thumb_force is False and args.fps == 30
    assert "--no-palm-thumb-force" in argv
    assert Path(args.out) == (roots / "out" / "clips").resolve()
    assert result["command_line"].startswith(
        "python3 -m src.tools.native_viewer_export"
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"bundle": "/etc/passwd"}, "outside the allowed input roots"),
        ({"out": "../escape"}, "outside the allowed output root"),
        ({"no_such_option": 1}, "unknown option"),
        ({"stride": 2}, "not accepted remotely"),
        ({"shaft_model": "chain"}, "set when the bundle is built"),
        ({"preset": "ultra"}, "invalid choice"),
    ],
)
def test_validate_rejects_unsafe_or_invalid_requests(
    roots: Path, extra: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rs.validate(_request(roots, **extra))
