"""Tests for the MachineLearning <-> canonical ClubTarget adapter."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.loaders._machinelearning_compat import (
    CLUBFACE_POSITION,
    CLUBLOGS_POSITION,
    DEFAULT_SHAFT_LENGTH_M,
    to_canonical_target_from_clubface,
    to_canonical_target_from_clublogs,
    to_machinelearning_clubface,
    to_machinelearning_clublogs,
)

from ._fixtures import make_target, repo_root


def _silence_lossy() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module=".*ml.*")


def test_clublogs_round_trip() -> None:
    """``target -> clublogs DF -> target`` is bit-equal on numerical fields."""
    target = make_target(n=301)
    df = to_machinelearning_clublogs(target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        recovered = to_canonical_target_from_clublogs(df)

    assert isinstance(recovered, ClubTarget)
    assert recovered.time.shape == target.time.shape
    np.testing.assert_allclose(recovered.time, target.time, atol=1e-12)
    np.testing.assert_allclose(recovered.clubhead, target.clubhead, atol=1e-12)
    assert recovered.impact_idx == target.impact_idx


def test_clubface_round_trip_lossy() -> None:
    """``target -> clubface DF -> target`` preserves clubhead positions; butt /
    quat are reconstructed and not necessarily equal to the original."""
    target = make_target(n=301)
    df = to_machinelearning_clubface(target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        recovered = to_canonical_target_from_clubface(df)

    np.testing.assert_allclose(recovered.clubhead, target.clubhead, atol=1e-12)
    np.testing.assert_allclose(recovered.time, target.time, atol=1e-12)
    # Documented loss: butt and quat are reconstructions, not the originals.
    assert not np.allclose(recovered.butt, target.butt, atol=1e-3)
    # Quaternions remain unit-norm regardless.
    np.testing.assert_allclose(
        np.linalg.norm(recovered.club_quat, axis=1),
        np.ones(recovered.club_quat.shape[0]),
        atol=1e-9,
    )


def test_to_machinelearning_emits_correct_columns() -> None:
    target = make_target(n=51)
    cf = to_machinelearning_clubface(target)
    cl = to_machinelearning_clublogs(target)
    for c in CLUBFACE_POSITION:
        assert c in cf.columns
    for c in CLUBLOGS_POSITION:
        assert c in cl.columns
    assert "time" in cf.columns and "time" in cl.columns


def test_invalid_columns_raise_clear_error() -> None:
    bad = pd.DataFrame({"foo": [1.0, 2.0], "bar": [3.0, 4.0]})
    with (
        pytest.raises(ValueError, match="missing required columns"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        to_canonical_target_from_clubface(bad)
    with (
        pytest.raises(ValueError, match="missing required columns"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        to_canonical_target_from_clublogs(bad)


def test_lossy_warning_emitted() -> None:
    target = make_target(n=51)
    df = to_machinelearning_clublogs(target)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        to_canonical_target_from_clublogs(df)
    assert any(
        issubclass(w.category, UserWarning) and "lossy" in str(w.message)
        for w in caught
    )


def test_too_short_dataframe_rejected() -> None:
    cols = {c: [1.0] for c in CLUBLOGS_POSITION}
    df = pd.DataFrame(cols)
    with pytest.raises(ValueError, match=">=2 rows"), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        to_canonical_target_from_clublogs(df)


def test_butt_uses_default_shaft_length() -> None:
    """When velocity is finite, butt = clubhead - shaft * v_hat."""
    target = make_target(n=101)
    df = to_machinelearning_clublogs(target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        recovered = to_canonical_target_from_clublogs(df)
    # Centre sample has well-defined velocity; check geometry there.
    i = 50
    delta = recovered.butt[i] - recovered.clubhead[i]
    assert abs(np.linalg.norm(delta) - DEFAULT_SHAFT_LENGTH_M) < 1e-9
