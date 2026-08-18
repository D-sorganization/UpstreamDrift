"""RNG discipline for reproducible BunkerShot3D runs (issue #8617, B18).

Contract (research digest section 7): ``secrets.randbits(128)`` ->
``SeedSequence`` -> ``Generator(PCG64DXSM(ss))``; record ``ss.entropy`` *and*
``numpy.__version__`` because NEP 19 permits stream changes on X.Y releases.
Never ``np.random.seed``; never ``root_seed + worker_id`` -- spawn children
from the parent :class:`~numpy.random.SeedSequence`.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import PCG64DXSM, Generator, SeedSequence

from bunkershot3d.provenance import (
    ENTROPY_BITS,
    GENERATOR_NAME,
    SeedRecord,
    make_generator,
    new_entropy,
    root_seed_sequence,
    seed_record,
    spawn_generators,
    spawn_sequences,
)

pytestmark = pytest.mark.unit


def test_new_entropy_is_128_bit() -> None:
    values = {new_entropy() for _ in range(8)}
    assert len(values) == 8, "entropy must not repeat"
    for value in values:
        assert 0 <= value < 2**ENTROPY_BITS
        assert value.bit_length() > 64, "entropy should exercise the full width"


def test_root_seed_sequence_records_requested_entropy() -> None:
    ss = root_seed_sequence(entropy=12345)
    assert isinstance(ss, SeedSequence)
    assert ss.entropy == 12345


def test_make_generator_uses_pcg64dxsm() -> None:
    gen = make_generator(root_seed_sequence(entropy=7))
    assert isinstance(gen, Generator)
    assert isinstance(gen.bit_generator, PCG64DXSM)


def test_make_generator_rejects_raw_int_seed() -> None:
    """A bare int seed bypasses the recorded SeedSequence: reject it loudly."""
    with pytest.raises(TypeError, match="SeedSequence"):
        make_generator(7)  # type: ignore[arg-type]


def test_same_entropy_gives_identical_streams() -> None:
    left = make_generator(root_seed_sequence(entropy=99)).standard_normal(16)
    right = make_generator(root_seed_sequence(entropy=99)).standard_normal(16)
    np.testing.assert_array_equal(left, right)


def test_different_entropy_gives_different_streams() -> None:
    left = make_generator(root_seed_sequence(entropy=1)).standard_normal(16)
    right = make_generator(root_seed_sequence(entropy=2)).standard_normal(16)
    assert not np.array_equal(left, right)


def test_spawned_children_are_independent_and_not_offset_seeds() -> None:
    parent = root_seed_sequence(entropy=4242)
    children = spawn_sequences(parent, 4)
    assert len(children) == 4
    assert [child.spawn_key for child in children] == [(0,), (1,), (2,), (3,)]

    streams = [make_generator(child).standard_normal(8) for child in children]
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            assert not np.array_equal(streams[i], streams[j])

    # The failure mode being guarded against: root_seed + worker_id.
    offset = [
        make_generator(root_seed_sequence(entropy=4242 + k)).standard_normal(8)
        for k in range(4)
    ]
    assert not any(
        np.array_equal(child, naive)
        for child, naive in zip(streams, offset, strict=True)
    )


def test_spawn_generators_is_reproducible_from_the_record() -> None:
    parent = root_seed_sequence(entropy=555)
    first = [gen.standard_normal(4) for gen in spawn_generators(parent, 3)]
    replay_parent = seed_record(root_seed_sequence(entropy=555), "root").to_sequence()
    second = [gen.standard_normal(4) for gen in spawn_generators(replay_parent, 3)]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_spawn_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        spawn_sequences(root_seed_sequence(entropy=1), 0)


# ---------------------------------------------------------------------------
# SeedRecord
# ---------------------------------------------------------------------------


def test_seed_record_captures_entropy_generator_and_numpy_version() -> None:
    record = seed_record(root_seed_sequence(entropy=31337), "grains")
    assert record.name == "grains"
    assert record.entropy == 31337
    assert record.generator == GENERATOR_NAME
    assert record.numpy_version == np.__version__


def test_seed_record_round_trips_through_dict() -> None:
    parent = root_seed_sequence(entropy=8)
    child = spawn_sequences(parent, 2)[1]
    record = seed_record(child, "worker-1")
    restored = SeedRecord.from_dict(record.to_dict())
    assert restored == record
    np.testing.assert_array_equal(
        make_generator(restored.to_sequence()).standard_normal(4),
        make_generator(child).standard_normal(4),
    )


def test_seed_record_tracks_children_already_spawned() -> None:
    parent = root_seed_sequence(entropy=17)
    spawn_sequences(parent, 3)
    record = seed_record(parent, "root")
    assert record.n_children_spawned == 3
