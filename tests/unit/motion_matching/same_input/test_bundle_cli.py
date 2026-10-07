"""The same-input CLI accepts every registered engine (#11607)."""

from __future__ import annotations

import pytest

from scripts.same_input_bundle import build_parser
from src.shared.python.motion_matching.same_input import (
    ALL_ENGINES,
    OPTIONAL_ENGINES,
    PARITY_ENGINES,
)

pytestmark = pytest.mark.unit


def test_all_engines_lists_core_then_optional() -> None:
    assert (*PARITY_ENGINES, *OPTIONAL_ENGINES) == ALL_ENGINES


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("command", ["replay", "closed-loop"])
def test_parser_accepts_every_engine(command: str, engine: str) -> None:
    argv = [command, "--bundle", "b.npz", "--engine", engine, "--out", "r.json"]
    if command == "closed-loop":
        argv += ["--run-dir", "run"]
    assert build_parser().parse_args(argv).engine == engine


def test_parser_rejects_unknown_engine() -> None:
    argv = ["replay", "--bundle", "b.npz", "--engine", "bullet", "--out", "r.json"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)
