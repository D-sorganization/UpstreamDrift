import copy
import numpy as np
import pytest

from src.shared.python.motion_matching.same_input.stage_archive import (
    encode_stage_archive,
    decode_stage_archive,
    compare_stage_archives,
)
from src.shared.python.motion_matching.same_input.integrator import (
    StepPolicy,
    integrate,
)
from src.shared.python.motion_matching.same_input.stage_trace import StageRecorder
from tests.unit.motion_matching.same_input.test_lattice_ideal_replay import Plant


def trace():
    r = StageRecorder(capture_ideal_drive=True)
    integrate(
        Plant(),
        np.zeros(7),
        np.zeros(7),
        lambda *_: np.ones(7),
        steps=1,
        dt_s=0.01,
        policy=StepPolicy(substeps=2, project=False),
        recorder=r,
    )
    return r


def test_roundtrip_and_comparison():
    archive = encode_stage_archive(trace())
    restored = decode_stage_archive(copy.deepcopy(archive))
    assert compare_stage_archives(archive, encode_stage_archive(restored))
    archive["steps"][0]["stages"][0]["applied_external"][-1] = 1
    assert not compare_stage_archives(archive, encode_stage_archive(restored))


@pytest.mark.parametrize("mutation", ["old", "partial", "nan", "clock", "missing"])
def test_reject_invalid(mutation):
    archive = encode_stage_archive(trace())
    if mutation == "old":
        archive["schema"] = "research-stage-evidence/v0"
    if mutation == "partial":
        archive["steps"][0]["stages"].pop()
    if mutation == "nan":
        archive["steps"][0]["stages"][0]["q"][0] = float("nan")
    if mutation == "clock":
        archive["steps"][0]["stages"][0]["time_s"] += 0.001
    if mutation == "missing":
        del archive["steps"][0]["stages"][0]["applied_drive"]
    with pytest.raises((ValueError, KeyError)):
        decode_stage_archive(archive)
