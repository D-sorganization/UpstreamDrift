"""Typed NPZ stage evidence preserves exact numerical observations."""

import io
import numpy as np
import pytest

from src.shared.python.motion_matching.same_input.stage_numeric_archive import (
    encode_numeric,
    decode_numeric,
)
from src.shared.python.motion_matching.same_input.stage_archive import (
    encode_stage_archive,
)
from tests.unit.motion_matching.same_input.test_stage_archive import trace

pytestmark = pytest.mark.unit


def test_numeric_roundtrip_and_arrays_are_typed():
    original = trace()
    raw = encode_numeric(original)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        assert data["q"].dtype == np.float64
        assert data["stage_ids"].dtype == np.int64
        assert data["q"].shape == (8, 7)
    assert encode_stage_archive(decode_numeric(raw)) == encode_stage_archive(original)


def test_numeric_rejects_nonfinite_and_missing_array():
    raw = encode_numeric(trace())
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["q"][0, 0] = np.nan
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    with pytest.raises(ValueError):
        decode_numeric(out.getvalue())
    del arrays["v"]
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    with pytest.raises(ValueError):
        decode_numeric(out.getvalue())
