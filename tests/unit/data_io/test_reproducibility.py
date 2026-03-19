"""Tests for src.shared.python.data_io.reproducibility (Issues #1949, #1744)."""

from __future__ import annotations

import logging
import random
import time

import numpy as np
import pytest

from src.shared.python.data_io.reproducibility import (
    DEFAULT_SEED,
    MAX_SEED,
    ensure_reproducibility,
    get_rng,
    log_execution_time,
    set_seeds,
)

# ---------------------------------------------------------------------------
# set_seeds
# ---------------------------------------------------------------------------


class TestSetSeeds:
    def test_default_seed_constant(self) -> None:
        assert DEFAULT_SEED == 42

    def test_max_seed_constant(self) -> None:
        assert np.iinfo(np.uint32).max == MAX_SEED

    def test_set_seeds_makes_numpy_deterministic(self) -> None:
        set_seeds(0)
        a = np.random.rand(5)
        set_seeds(0)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_set_seeds_makes_random_deterministic(self) -> None:
        set_seeds(123)
        a = random.random()
        set_seeds(123)
        b = random.random()
        assert a == b

    def test_different_seeds_give_different_results(self) -> None:
        set_seeds(1)
        a = np.random.rand()
        set_seeds(2)
        b = np.random.rand()
        assert a != b

    def test_invalid_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="Seed must"):
            set_seeds(-1)

    def test_seed_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="Seed must"):
            set_seeds(MAX_SEED + 1)

    def test_validate_false_skips_our_check(self) -> None:
        # With validate=False, our range check is skipped (numpy may still raise)
        # Test with a seed at the boundary that our check would reject but is valid
        set_seeds(MAX_SEED, validate=False)  # should not raise

    def test_zero_seed_valid(self) -> None:
        set_seeds(0)  # should not raise

    def test_max_seed_valid(self) -> None:
        set_seeds(MAX_SEED)  # should not raise


# ---------------------------------------------------------------------------
# get_rng
# ---------------------------------------------------------------------------


class TestGetRng:
    def test_returns_numpy_generator(self) -> None:
        rng = get_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_seeded_rng_is_deterministic(self) -> None:
        a = get_rng(7).random()
        b = get_rng(7).random()
        assert a == b

    def test_unseeded_rng_returns_float(self) -> None:
        # Just verify no exception is raised and result is a float
        value = get_rng(None).random()
        assert isinstance(value, float)

    def test_rng_does_not_affect_global_state(self) -> None:
        np.random.seed(99)
        _ = get_rng(42).random(10)
        first = np.random.rand()

        np.random.seed(99)
        _ = get_rng(0).random(10)
        second = np.random.rand()

        # Global np.random state only depends on np.random.seed(99), not get_rng
        assert first == second


# ---------------------------------------------------------------------------
# log_execution_time
# ---------------------------------------------------------------------------


class TestLogExecutionTime:
    def test_yields_without_error(self) -> None:
        with log_execution_time("test_op"):
            pass  # no exception

    def test_code_inside_runs(self) -> None:
        ran = []
        with log_execution_time("check"):
            ran.append(1)
        assert ran == [1]

    def test_duration_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO), log_execution_time("my_op"):
            time.sleep(0.01)
        assert any("my_op" in r.message for r in caplog.records)

    def test_custom_logger_used(self) -> None:
        custom = logging.getLogger("custom_test_logger")
        records: list[logging.LogRecord] = []
        custom.addHandler(
            type(
                "_H",
                (logging.Handler,),
                {"emit": lambda self, r: records.append(r)},
            )()
        )
        custom.setLevel(logging.DEBUG)

        with log_execution_time("custom_op", logger_obj=custom):
            pass

        assert any("custom_op" in r.getMessage() for r in records)

    def test_exception_propagates(self) -> None:
        with pytest.raises(ZeroDivisionError), log_execution_time("failing_op"):
            _ = 1 / 0


# ---------------------------------------------------------------------------
# ensure_reproducibility
# ---------------------------------------------------------------------------


class TestEnsureReproducibility:
    def test_sets_numpy_seed(self) -> None:
        ensure_reproducibility(10)
        a = np.random.rand(3)
        ensure_reproducibility(10)
        b = np.random.rand(3)
        np.testing.assert_array_equal(a, b)

    def test_default_seed_used(self) -> None:
        ensure_reproducibility()  # uses DEFAULT_SEED
        a = np.random.rand()
        ensure_reproducibility(DEFAULT_SEED)
        b = np.random.rand()
        assert a == b
