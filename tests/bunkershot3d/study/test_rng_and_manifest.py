"""RNG discipline and run manifests (#8615).

These tests pin the rules from the research digest (section 7): 128-bit
entropy from :mod:`secrets`, ``PCG64DXSM`` generators, ``spawn`` for parallel
streams (never ``root_seed + worker_id``), and a manifest that records the
NumPy version because NEP 19 lets bit streams change across ``X.Y`` releases.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy
from bunkershot3d.study import SeedRecord, StudyManifest, as_generator, new_seed_record
from numpy.random import PCG64DXSM

pytestmark = pytest.mark.unit


class TestSeedRecord:
    """The replayable seed."""

    def test_fresh_records_have_128_bits_of_entropy(self) -> None:
        record = new_seed_record()
        assert 0 <= record.entropy < 2**128
        assert record.numpy_version == np.__version__

    def test_fresh_records_differ(self) -> None:
        entropies = {new_seed_record().entropy for _ in range(16)}
        assert len(entropies) == 16

    def test_uses_pcg64dxsm(self) -> None:
        generator = new_seed_record(42).generator()
        assert isinstance(generator.bit_generator, PCG64DXSM)

    def test_same_entropy_gives_the_same_stream(self) -> None:
        first = new_seed_record(20260813).generator().random(64)
        second = new_seed_record(20260813).generator().random(64)
        np.testing.assert_array_equal(first, second)

    def test_different_entropy_gives_a_different_stream(self) -> None:
        first = new_seed_record(1).generator().random(64)
        second = new_seed_record(2).generator().random(64)
        assert not np.allclose(first, second)

    def test_passing_a_record_through_is_idempotent(self) -> None:
        record = new_seed_record(7)
        assert new_seed_record(record) is record

    def test_spawned_streams_are_distinct(self) -> None:
        children = new_seed_record(99).spawn(8)
        draws = [child.random(32) for child in children]
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                assert not np.allclose(draws[i], draws[j])

    def test_spawning_is_reproducible(self) -> None:
        first = [child.random(16) for child in new_seed_record(5).spawn(4)]
        second = [child.random(16) for child in new_seed_record(5).spawn(4)]
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_spawning_differs_from_seed_plus_worker_id(self) -> None:
        # `root + worker_id` is the anti-pattern the digest calls out: it
        # produces streams that can overlap. Assert we are not doing it.
        root = 1000
        spawned = new_seed_record(root).spawn(3)[1].random(16)
        naive = new_seed_record(root + 1).generator().random(16)
        assert not np.allclose(spawned, naive)

    def test_rejects_non_positive_spawn_count(self) -> None:
        with pytest.raises(ValueError, match="count"):
            new_seed_record(1).spawn(0)

    @pytest.mark.parametrize("entropy", [-1, 2**128])
    def test_rejects_out_of_range_entropy(self, entropy: int) -> None:
        with pytest.raises(ValueError, match="128 bits"):
            SeedRecord(entropy=entropy, numpy_version=np.__version__)

    def test_rejects_non_integer_entropy(self) -> None:
        with pytest.raises(TypeError, match="int"):
            SeedRecord(entropy=1.5, numpy_version=np.__version__)  # type: ignore[arg-type]

    def test_rejects_empty_numpy_version(self) -> None:
        with pytest.raises(ValueError, match="numpy_version"):
            SeedRecord(entropy=1, numpy_version="")

    def test_round_trips_through_a_dict(self) -> None:
        record = new_seed_record()
        restored = SeedRecord.from_dict(record.to_dict())
        assert restored == record
        np.testing.assert_array_equal(
            record.generator().random(8), restored.generator().random(8)
        )


class TestAsGenerator:
    """The generator coercion helper."""

    def test_passes_generators_through_unchanged(self) -> None:
        generator = new_seed_record(3).generator()
        assert as_generator(generator) is generator

    def test_builds_a_generator_from_an_integer(self) -> None:
        np.testing.assert_array_equal(
            as_generator(11).random(8), new_seed_record(11).generator().random(8)
        )

    def test_builds_a_generator_from_a_record(self) -> None:
        record = new_seed_record(12)
        np.testing.assert_array_equal(
            as_generator(record).random(8), record.generator().random(8)
        )

    def test_none_draws_fresh_entropy(self) -> None:
        assert not np.allclose(
            as_generator(None).random(8), as_generator(None).random(8)
        )


class TestStudyManifest:
    """Run provenance."""

    def test_records_library_versions_by_default(self) -> None:
        manifest = StudyManifest.create(
            method="sobol", parameter_names=("a", "b"), n_samples=64
        )
        assert manifest.numpy_version == np.__version__
        assert manifest.scipy_version == scipy.__version__
        assert manifest.dimension == 2

    def test_round_trips_through_json(self) -> None:
        manifest = StudyManifest.create(
            method="morris",
            parameter_names=("bounce_deg", "sole_width_mm"),
            n_samples=42,
            seed=123456789,
            extra={"n_levels": 4, "delta": 0.6667},
        )
        payload = json.dumps(manifest.to_dict(), allow_nan=False)
        restored = StudyManifest.from_dict(json.loads(payload))

        assert restored.seed == manifest.seed
        assert restored.method == manifest.method
        assert restored.parameter_names == manifest.parameter_names
        assert restored.extra == manifest.extra

    def test_rejects_empty_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            StudyManifest.create(method="", parameter_names=("a",), n_samples=1)

    def test_rejects_negative_sample_count(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            StudyManifest.create(method="sobol", parameter_names=("a",), n_samples=-1)

    def test_create_accepts_an_explicit_seed_record(self) -> None:
        record = new_seed_record(2468)
        manifest = StudyManifest.create(
            method="sobol", parameter_names=("a",), n_samples=8, seed=record
        )
        assert manifest.seed is record
