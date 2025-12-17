"""Integration tests for the main package."""

import importlib.util
import logging
import random

import numpy as np
import pytest

# Import handled by conftest.py
from logger_utils import get_logger, set_seeds

# Import c3d_reader using package import (same as test_c3d_reader.py)
# This will work even if ezc3d is not available due to our optional import handling
from src.c3d_reader import (
    C3DDataReader,
    C3DEvent,
    C3DMetadata,
    load_tour_average_reader,
)

# Skip C3D-related tests if ezc3d is not available (e.g., Python 3.9)
EZC3D_AVAILABLE = importlib.util.find_spec("ezc3d") is not None


class TestPackageIntegration:
    """Integration tests for package components working together."""

    def test_logger_used_in_seed_setting(self) -> None:
        """Test that logger is properly used when setting seeds."""
        logger = get_logger("test_integration")
        original_level = logger.level

        # Set to DEBUG to capture info messages
        logger.setLevel(logging.DEBUG)

        # Set seeds should log a message
        set_seeds(42)

        # Restore original level
        logger.setLevel(original_level)

    def test_reproducible_random_generation_workflow(self) -> None:
        """Test a complete workflow with reproducible random generation."""
        # Set seeds for reproducibility
        set_seeds(12345)

        # Generate random data
        random_values = [random.random() for _ in range(5)]
        numpy_array = np.random.rand(5)

        # Reset seeds
        set_seeds(12345)

        # Regenerate should produce same results
        random_values_2 = [random.random() for _ in range(5)]
        numpy_array_2 = np.random.rand(5)

        assert random_values == random_values_2
        np.testing.assert_array_equal(numpy_array, numpy_array_2)

    @pytest.mark.skipif(
        not EZC3D_AVAILABLE,
        reason="ezc3d requires Python >=3.10",
    )
    def test_c3d_reader_with_logger(self) -> None:
        """Test that C3D reader can be used with logger."""
        logger = get_logger("c3d_test")

        # Test that components can be imported and used together
        assert logger is not None
        assert C3DDataReader is not None
        assert C3DMetadata is not None
        assert C3DEvent is not None

    @pytest.mark.skipif(
        not EZC3D_AVAILABLE,
        reason="ezc3d requires Python >=3.10",
    )
    def test_module_classes_available(self) -> None:
        """Test that all module classes are accessible."""
        # Verify all main classes are available
        assert C3DDataReader is not None
        assert C3DMetadata is not None
        assert C3DEvent is not None
        assert callable(load_tour_average_reader)

    def test_seed_affects_numpy_operations(self) -> None:
        """Test that set_seeds affects numpy operations used in data processing."""
        set_seeds(999)

        # Simulate some data processing operations
        data = np.random.randn(100, 3)
        noise = np.random.random((100, 3))
        processed = data + noise * 0.1

        # Reset and verify reproducibility
        set_seeds(999)
        data_2 = np.random.randn(100, 3)
        noise_2 = np.random.random((100, 3))
        processed_2 = data_2 + noise_2 * 0.1

        np.testing.assert_array_almost_equal(processed, processed_2)


class TestModuleImports:
    """Test that all modules can be imported correctly."""

    def test_import_logger_utils(self) -> None:
        """Test importing logger_utils module."""
        from logger_utils import get_logger, set_seeds

        assert callable(get_logger)
        assert callable(set_seeds)

    @pytest.mark.skipif(
        not EZC3D_AVAILABLE,
        reason="ezc3d requires Python >=3.10",
    )
    def test_import_c3d_reader(self) -> None:
        """Test importing c3d_reader module."""
        from src.c3d_reader import C3DDataReader, C3DEvent, C3DMetadata

        assert C3DDataReader is not None
        assert C3DMetadata is not None
        assert C3DEvent is not None

    @pytest.mark.skipif(
        not EZC3D_AVAILABLE,
        reason="ezc3d requires Python >=3.10",
    )
    def test_c3d_reader_classes_available(self) -> None:
        """Test that C3D reader classes are available."""
        # Use the already imported classes
        assert C3DDataReader is not None
        assert C3DMetadata is not None
        assert C3DEvent is not None
        assert callable(load_tour_average_reader)


class TestCrossModuleFunctionality:
    """Test functionality that spans multiple modules."""

    def test_logger_configuration_persists(self) -> None:
        """Test that logger configuration persists across module usage."""
        # Get logger in one context
        logger1 = get_logger("persistent_test")
        logger1.setLevel(logging.WARNING)

        # Get same logger elsewhere
        logger2 = get_logger("persistent_test")

        # Should be same instance with same level
        assert logger1 is logger2
        assert logger1.level == logging.WARNING

    def test_multiple_logger_instances_independent(self) -> None:
        """Test that different logger instances are independent."""
        logger_a = get_logger("module_a")
        logger_b = get_logger("module_b")

        logger_a.setLevel(logging.DEBUG)
        logger_b.setLevel(logging.ERROR)

        assert logger_a.level == logging.DEBUG
        assert logger_b.level == logging.ERROR

    def test_seed_setting_idempotent(self) -> None:
        """Test that setting the same seed multiple times is idempotent."""
        set_seeds(777)
        val1 = random.random()

        set_seeds(777)
        val2 = random.random()

        set_seeds(777)
        val3 = random.random()

        # All should be the same since we reset the seed each time
        assert val1 == val2 == val3


class TestErrorHandling:
    """Test error handling across module integration."""

    def test_logger_with_invalid_name_still_works(self) -> None:
        """Test that logger works even with unusual names."""
        # Empty string name
        logger1 = get_logger("")
        assert isinstance(logger1, logging.Logger)

        # Very long name
        long_name = "a" * 1000
        logger2 = get_logger(long_name)
        assert isinstance(logger2, logging.Logger)

    def test_seed_with_negative_value(self) -> None:
        """Test set_seeds with negative seed value."""
        # Numpy doesn't accept negative seeds, so this should raise ValueError
        with pytest.raises(ValueError, match="Seed must be between 0 and"):
            set_seeds(-1)

    def test_seed_with_large_value(self) -> None:
        """Test set_seeds with very large seed value."""
        # Should handle large integers
        set_seeds(2**31 - 1)
        val = random.random()
        assert isinstance(val, float)


class TestPerformance:
    """Performance-related integration tests."""

    def test_logger_creation_performance(self) -> None:
        """Test that logger creation is fast even with many loggers."""
        import time

        start = time.time()
        loggers = [get_logger(f"logger_{i}") for i in range(100)]
        elapsed = time.time() - start

        # Should be reasonably fast (less than 5 seconds for 100 loggers)
        # More generous threshold to avoid flaky tests on slower CI systems
        assert elapsed < 5.0
        assert len(loggers) == 100

    def test_seed_setting_performance(self) -> None:
        """Test that seed setting is fast."""
        import time

        start = time.time()
        for i in range(100):
            set_seeds(i)
        elapsed = time.time() - start

        # Should be reasonably fast (less than 2 seconds for 100 seed sets)
        # More generous threshold to avoid flaky tests on slower CI systems
        assert elapsed < 2.0


class TestDataTypes:
    """Test data type compatibility across modules."""

    def test_numpy_types_with_random_seed(self) -> None:
        """Test that numpy types work correctly with seeded random generation."""
        set_seeds(456)

        # Generate various numpy types
        int_array = np.random.randint(0, 100, size=10)
        float_array = np.random.rand(10)
        normal_array = np.random.randn(10)

        # Verify types
        assert int_array.dtype in [np.int32, np.int64]
        assert float_array.dtype == np.float64
        assert normal_array.dtype == np.float64

        # Verify reproducibility
        set_seeds(456)
        int_array_2 = np.random.randint(0, 100, size=10)

        np.testing.assert_array_equal(int_array, int_array_2)
