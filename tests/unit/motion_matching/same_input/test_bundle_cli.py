"""The same-input CLI accepts every registered engine (#11607) and chain bundles (#12691)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.same_input_bundle import build_parser, padded_track, shaft_inputs
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


# --- chain-shaft bundles (R11, #12691) -------------------------------------------

SEED_SPEC = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
    / "anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)


def test_export_shaft_model_defaults_to_rigid() -> None:
    argv = ["export", "--run-dir", "run", "--out", "b.npz"]
    assert build_parser().parse_args(argv).shaft_model == "rigid"
    chain = build_parser().parse_args([*argv, "--shaft-model", "chain"])
    assert chain.shaft_model == "chain"
    with pytest.raises(SystemExit):
        build_parser().parse_args([*argv, "--shaft-model", "noodle"])


def test_rigid_shaft_inputs_are_returned_unchanged() -> None:
    spec = SEED_SPEC.read_bytes()
    track = np.ones((3, len(json.loads(spec)["coordinate_order"])))
    out_spec, out_track = shaft_inputs(spec, track, "rigid")
    assert out_spec is spec and out_track is track


def test_chain_shaft_inputs_append_an_undeflected_shaft() -> None:
    spec = SEED_SPEC.read_bytes()
    nv = len(json.loads(spec)["coordinate_order"])
    track = np.ones((3, nv))
    out_spec, out_track = shaft_inputs(spec, track, "chain")
    document = json.loads(out_spec)
    shaft = document["club"]["shaft"]["coordinates"]
    assert document["club"]["shaft"]["model"] == "chain"
    assert document["coordinate_order"][-len(shaft) :] == shaft
    assert out_track.shape == (3, nv + len(shaft))
    np.testing.assert_array_equal(out_track[:, :nv], track)
    assert not out_track[:, nv:].any()
    # A chain bundle's own spec pads the run's rigid-width track the same way.
    np.testing.assert_array_equal(padded_track(out_spec, track), out_track)
    np.testing.assert_array_equal(padded_track(out_spec, out_track), out_track)


def test_padded_track_rejects_a_mismatched_width() -> None:
    spec = SEED_SPEC.read_bytes()
    with pytest.raises(ValueError, match="q_track"):
        padded_track(spec, np.ones((3, 5)))
