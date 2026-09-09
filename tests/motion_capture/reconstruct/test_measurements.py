"""Many tape measurements, both sides at once, and their effect on the fit (#9707)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.bundle import BundleOptions
from src.motion_capture.reconstruct.fit import fit_bundle
from src.motion_capture.reconstruct.measurements import (
    RECOMMENDED,
    SEGMENT_ALIASES,
    TAPE_GUIDE,
    Measurement,
    describe,
    expand_measurements,
    gauge,
    parse_measurement,
)
from src.motion_capture.reconstruct.skeleton import PARENTS, SYMMETRIC_PAIRS
from src.motion_capture.reconstruct.synthetic import load_truth

pytestmark = pytest.mark.unit


def test_aliases_name_real_segments_and_recommendations_have_tape_guides() -> None:
    for name, (joints, factor) in SEGMENT_ALIASES.items():
        assert all(j in PARENTS and PARENTS[j] is not None for j in joints), name
        assert factor in (1.0, 0.5)
    for name in RECOMMENDED:
        assert name in TAPE_GUIDE and name in SEGMENT_ALIASES
    pairs = {frozenset(p) for p in SYMMETRIC_PAIRS}
    for name in ("upper_arm", "forearm", "thigh", "shank"):
        assert frozenset(SEGMENT_ALIASES[name][0]) in pairs  # bilateral by construction


def test_parse_expand_and_gauge() -> None:
    m = parse_measurement("Shank=0.42")
    assert m == Measurement("shank", 0.42)
    assert m.segments() == {"left_ankle": 0.42, "right_ankle": 0.42}
    lengths = expand_measurements(["shank=0.42", "forearm=0.26", "left_forearm=0.28"])
    assert lengths["left_ankle"] == 0.42 and lengths["right_wrist"] == 0.26
    assert lengths["left_wrist"] == pytest.approx(0.27)  # two readings averaged
    assert gauge(lengths) == ("left_ankle", 0.42)  # first mention sets the scale
    assert expand_measurements(["shoulder_width=0.40"]) == {
        "left_shoulder": 0.2,
        "right_shoulder": 0.2,
    }
    assert expand_measurements(["left_wrist=0.25"]) == {"left_wrist": 0.25}
    assert describe({"neck": 0.5}) == "neck=0.500"
    with pytest.raises(Exception, match="unknown segment"):
        parse_measurement("femur=0.4")
    with pytest.raises(Exception, match="NAME=METRES"):
        parse_measurement("shank")
    with pytest.raises(ValueError, match="not a number"):
        parse_measurement("shank=long")
    with pytest.raises(Exception, match="metres"):
        parse_measurement("shank=42")
    with pytest.raises(Exception, match="at least one"):
        gauge({})


def test_bundle_options_validate_measured_lengths() -> None:
    BundleOptions(measured_lengths_m={"left_ankle": 0.42})
    with pytest.raises(Exception, match="positive"):
        BundleOptions(measured_lengths_m={"left_ankle": -1.0})


@pytest.mark.slow
def test_more_measurements_pull_the_fit_towards_true_lengths(tmp_path: Path) -> None:
    from src.motion_capture.reconstruct.__main__ import main as reconstruct_main

    bundle = tmp_path / "synth"
    assert (
        reconstruct_main(
            [
                "synth",
                "--out",
                str(bundle),
                "--frames",
                "60",
                "--fps",
                "60",
                "--seed",
                "3",
            ]
        )
        == 0
    )
    truth = load_truth(bundle / "truth.json")
    true_len = truth.bone_lengths_m
    # A deliberately wrong anthropometric prior: only measurements can fix it.
    prior = {k: v * 1.15 for k, v in true_len.items()}
    anchor = ("neck", true_len["neck"])
    one = fit_bundle(bundle, scale_anchor=anchor, length_prior_m=prior)
    many = fit_bundle(
        bundle,
        scale_anchor=anchor,
        length_prior_m=prior,
        measured_lengths_m={
            "left_ankle": true_len["left_ankle"],
            "right_ankle": true_len["right_ankle"],
            "left_wrist": true_len["left_wrist"],
            "right_wrist": true_len["right_wrist"],
        },
    )
    assert set(many.measured_lengths_m) == {
        "left_ankle",
        "right_ankle",
        "left_wrist",
        "right_wrist",
    }

    def err(record, joint: str) -> float:
        return abs(record.bone_lengths_m[joint] - true_len[joint]) / true_len[joint]

    for joint in ("left_ankle", "right_wrist"):
        assert err(many, joint) < 0.02
        assert err(many, joint) <= err(one, joint) + 1e-9
    assert many.rms_px <= one.rms_px * 1.2  # measurements do not fight the images
    assert np.isfinite(many.rms_px)
