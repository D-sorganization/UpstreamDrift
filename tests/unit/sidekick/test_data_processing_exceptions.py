"""Tests for sidekick.data_processing.exceptions (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.data_processing.exceptions import (
    ColumnNotFoundError,
    DataNotLoadedError,
    DataProcessingError,
    FileIOError,
    FilterError,
    FitError,
    TransformationError,
    UnsupportedOperationError,
)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_data_not_loaded_is_data_processing_error(self) -> None:
        assert issubclass(DataNotLoadedError, DataProcessingError)

    def test_column_not_found_is_data_processing_error(self) -> None:
        assert issubclass(ColumnNotFoundError, DataProcessingError)

    def test_file_io_error_is_data_processing_error(self) -> None:
        assert issubclass(FileIOError, DataProcessingError)

    def test_transformation_error_is_data_processing_error(self) -> None:
        assert issubclass(TransformationError, DataProcessingError)

    def test_filter_error_is_data_processing_error(self) -> None:
        assert issubclass(FilterError, DataProcessingError)

    def test_fit_error_is_data_processing_error(self) -> None:
        assert issubclass(FitError, DataProcessingError)

    def test_unsupported_operation_is_data_processing_error(self) -> None:
        assert issubclass(UnsupportedOperationError, DataProcessingError)

    def test_data_processing_error_is_exception(self) -> None:
        assert issubclass(DataProcessingError, Exception)


# ---------------------------------------------------------------------------
# ColumnNotFoundError message
# ---------------------------------------------------------------------------


class TestColumnNotFoundError:
    def test_message_includes_column_name(self) -> None:
        err = ColumnNotFoundError("price")
        assert "price" in str(err)

    def test_stores_column_attribute(self) -> None:
        err = ColumnNotFoundError("date")
        assert err.column == "date"

    def test_available_columns_in_message(self) -> None:
        err = ColumnNotFoundError("missing", available=["a", "b", "c"])
        msg = str(err)
        assert "a" in msg or "b" in msg

    def test_empty_available_columns(self) -> None:
        err = ColumnNotFoundError("x", available=[])
        assert err.available == []

    def test_raises_correctly(self) -> None:
        with pytest.raises(ColumnNotFoundError):
            raise ColumnNotFoundError("col")


# ---------------------------------------------------------------------------
# Other exceptions raise correctly
# ---------------------------------------------------------------------------


class TestOtherExceptions:
    def test_data_not_loaded_raises(self) -> None:
        with pytest.raises(DataProcessingError):
            raise DataNotLoadedError("no data")

    def test_file_io_error_raises(self) -> None:
        with pytest.raises(DataProcessingError):
            raise FileIOError("file not found")

    def test_transformation_error_raises(self) -> None:
        with pytest.raises(DataProcessingError):
            raise TransformationError("bad transform")

    def test_filter_error_raises(self) -> None:
        with pytest.raises(DataProcessingError):
            raise FilterError("bad filter")

    def test_fit_error_raises(self) -> None:
        with pytest.raises(DataProcessingError):
            raise FitError("fitting failed")

    def test_unsupported_operation_raises(self) -> None:
        with pytest.raises(DataProcessingError):
            raise UnsupportedOperationError("unknown op")
