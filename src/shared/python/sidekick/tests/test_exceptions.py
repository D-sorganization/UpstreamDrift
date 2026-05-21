"""Comprehensive tests for upstream_drift_tools.data_processing.exceptions module.

Covers all custom exception classes, their hierarchy, and ColumnNotFoundError
custom formatting.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.data_processing.exceptions import (
    ColumnNotFoundError,
    DataNotLoadedError,
    DataProcessingError,
    FileIOError,
    FilterError,
    FitError,
    TransformationError,
    UnsupportedOperationError,
)


class TestExceptionHierarchy:
    def test_data_processing_error_is_exception(self):
        err = DataProcessingError("base error")
        assert isinstance(err, Exception)

    def test_data_not_loaded_is_data_processing_error(self):
        err = DataNotLoadedError("no data")
        assert isinstance(err, DataProcessingError)

    def test_column_not_found_is_data_processing_error(self):
        err = ColumnNotFoundError("col")
        assert isinstance(err, DataProcessingError)

    def test_file_io_error_is_data_processing_error(self):
        err = FileIOError("bad file")
        assert isinstance(err, DataProcessingError)

    def test_transformation_error_is_data_processing_error(self):
        err = TransformationError("bad transform")
        assert isinstance(err, DataProcessingError)

    def test_filter_error_is_data_processing_error(self):
        err = FilterError("bad filter")
        assert isinstance(err, DataProcessingError)

    def test_fit_error_is_data_processing_error(self):
        err = FitError("fit failed")
        assert isinstance(err, DataProcessingError)

    def test_unsupported_operation_is_data_processing_error(self):
        err = UnsupportedOperationError("not supported")
        assert isinstance(err, DataProcessingError)


class TestColumnNotFoundError:
    def test_message_without_available(self):
        err = ColumnNotFoundError("my_col")
        assert "my_col" in str(err)
        assert err.column == "my_col"
        assert err.available == []

    def test_message_with_available_columns(self):
        err = ColumnNotFoundError("bad_col", available=["a", "b", "c"])
        assert "bad_col" in str(err)
        assert "Available" in str(err)
        assert err.available == ["a", "b", "c"]

    def test_message_with_empty_available(self):
        """When available is empty list, the detail should not include 'Available'."""
        err = ColumnNotFoundError("missing", available=[])
        assert "Available" not in str(err)

    def test_message_with_none_available(self):
        """When available is None, defaults to empty list."""
        err = ColumnNotFoundError("missing", available=None)
        assert err.available == []

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ColumnNotFoundError, match="'pressure'"):
            raise ColumnNotFoundError("pressure", ["temp", "flow"])
