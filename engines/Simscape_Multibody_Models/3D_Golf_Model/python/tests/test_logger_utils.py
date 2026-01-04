"""Tests for logger utilities module."""

import logging
import random

# Import handled by conftest.py
import logger_utils
import numpy as np
from logger_utils import get_logger, set_seeds


def test_get_logger_returns_logger() -> None:
    """Test that get_logger returns a logging.Logger instance."""
    logger = get_logger(__name__)
    assert isinstance(logger, logging.Logger)


def test_get_logger_same_name_returns_same_instance() -> None:
    """Test that get_logger returns the same logger instance for the same name."""
    logger1 = get_logger("test_module")
    logger2 = get_logger("test_module")
    assert logger1 is logger2


def test_get_logger_different_names_return_different_instances() -> None:
    """Test that get_logger returns different logger instances for different names."""
    logger1 = get_logger("test_module_1")
    logger2 = get_logger("test_module_2")
    assert logger1 is not logger2


def test_logger_has_handler() -> None:
    """Test that the logger has at least one handler."""
    logger = get_logger("test_handler")
    assert len(logger.handlers) > 0


def test_logger_level_setting() -> None:
    """Test that logger level can be set and retrieved."""
    logger = get_logger("test_level")
    original_level = logger.level

    # Set to DEBUG level
    logger.setLevel(logging.DEBUG)
    assert logger.level == logging.DEBUG

    # Restore original level
    logger.setLevel(original_level)


def test_set_seeds_sets_random_state() -> None:
    """Test that set_seeds properly sets both random and numpy seeds."""
    # Set seed and capture initial values
    set_seeds(42)
    random_val1 = random.random()
    numpy_val1 = np.random.random()

    # Generate more random values to change state
    random.random()
    np.random.random()

    # Reset to same seed should produce same values
    set_seeds(42)
    random_val2 = random.random()
    numpy_val2 = np.random.random()

    assert random_val1 == random_val2
    assert numpy_val1 == numpy_val2


def test_set_seeds_different_seeds_produce_different_values() -> None:
    """Test that different seeds produce different random values."""
    set_seeds(42)
    random_val_42 = random.random()
    numpy_val_42 = np.random.random()

    set_seeds(123)
    random_val_123 = random.random()
    numpy_val_123 = np.random.random()

    assert random_val_42 != random_val_123
    assert numpy_val_42 != numpy_val_123


def test_set_seeds_with_zero() -> None:
    """Test that set_seeds works with zero seed value."""
    set_seeds(0)
    val1 = random.random()

    set_seeds(0)
    val2 = random.random()

    assert val1 == val2


def test_set_seeds_reproducibility_for_arrays() -> None:
    """Test that set_seeds ensures reproducibility for numpy array generation."""
    set_seeds(999)
    array1 = np.random.rand(5)

    set_seeds(999)
    array2 = np.random.rand(5)

    np.testing.assert_array_equal(array1, array2)


def test_module_logger_has_handlers() -> None:
    """Test that the module-level logger in logger_utils has handlers configured."""
    assert len(logger_utils.logger.handlers) > 0


def test_set_seeds_invalid_input() -> None:
    """Test that set_seeds raises ValueError for invalid input."""
    import pytest

    with pytest.raises(ValueError, match="Seed must be between"):
        set_seeds(-1)

    with pytest.raises(ValueError, match="Seed must be between"):
        set_seeds(np.iinfo(np.uint32).max + 1)
