"""TDD tests for Simscape dataset pipeline contracts (issue #2491).

Two bugs:
1. extractLogsoutDataFixed.m only accepts signals where size(data,1)==expected_length.
   Real 3D tensor signals ([3 1 N] or [3 3 N]) where time is the last dimension are
   silently skipped, producing incomplete datasets.
2. resampleDataToFrequency.m rebuilds time as 0:dt:sim_time (always starts at 0),
   then uses interp1(..., 'extrap'), fabricating values outside the recorded data span.
"""

from __future__ import annotations

from pathlib import Path

_EXTRACT_FILE = Path(
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/"
    "functions/dataset_generator/extractLogsoutDataFixed.m"
)
_RESAMPLE_FILE = Path(
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/"
    "functions/dataset_generator/resampleDataToFrequency.m"
)


class TestExtractLogsoutTensorSupport:
    """extractLogsoutDataFixed.m must handle 3D tensor signals."""

    def _source(self) -> str:
        return _EXTRACT_FILE.read_text(encoding="utf-8")

    def test_handles_tensor_signals_via_last_dimension(self) -> None:
        """Script must check the last dimension (not the first) for time-length match."""
        source = self._source()
        # The fix must inspect the last/third dimension for tensor signals
        # This can be done via ndims or size(data, ndims(data)) or size(data, 3)
        has_last_dim_check = (
            "ndims(data)" in source
            or "size(data, ndims" in source
            or "size(data, 3)" in source
        )
        assert has_last_dim_check, (
            "extractLogsoutDataFixed.m does not check the last dimension for 3D tensor "
            "signals. 3D tensors [3 1 N] and [3 3 N] have time as the Nth (last) "
            "dimension. Fix: check size(data, ndims(data)) == expected_length."
        )

    def test_no_silent_skip_without_unsupported_message(self) -> None:
        """3D tensor signals must not be silently skipped as 'not supported'."""
        source = self._source()
        lines = source.splitlines()
        # Before fix: skips non-matching sizes with 'not supported' message
        # After fix: the 'not supported' message should be gone (tensors are handled)
        skip_unsupported_lines = [
            line
            for line in lines
            if "not supported" in line.lower() and "3 1 N" in line
        ]
        assert not skip_unsupported_lines, (
            "extractLogsoutDataFixed.m still has a 'not supported' skip path for "
            "[3 1 N] tensors. These are valid 3D signals that must be handled.\n"
            "Offending lines:\n" + "\n".join(skip_unsupported_lines)
        )


class TestResampleNoExtrapolation:
    """resampleDataToFrequency.m must not extrapolate beyond the recorded data span."""

    def _source(self) -> str:
        return _RESAMPLE_FILE.read_text(encoding="utf-8")

    def test_time_vector_uses_actual_data_span(self) -> None:
        """New time vector must start from original_time(1), not hard-coded 0."""
        source = self._source()
        lines = source.splitlines()
        zero_start_lines = [
            line
            for line in lines
            if (
                ("0:target_dt:" in line or "0:dt:" in line or "new_time = 0:" in line)
                and not line.strip().startswith("%")
            )
        ]
        assert not zero_start_lines, (
            "resampleDataToFrequency.m builds time vector starting at 0. "
            "This extrapolates before the actual recording start. "
            "Fix: use original_time(1):target_dt:original_time(end).\n"
            "Offending lines:\n" + "\n".join(zero_start_lines)
        )

    def test_no_extrapolation_flag(self) -> None:
        """interp1 must not use 'extrap' flag — clamp to actual data range instead."""
        source = self._source()
        lines = source.splitlines()
        extrap_lines = [
            line
            for line in lines
            if "'extrap'" in line and not line.strip().startswith("%")
        ]
        assert not extrap_lines, (
            "resampleDataToFrequency.m uses interp1(..., 'extrap'), fabricating "
            "values outside the recorded data span. "
            "Fix: remove 'extrap' and use NaN as fill value, or clip the time "
            "vector to the actual data range.\n"
            "Offending lines:\n" + "\n".join(extrap_lines)
        )
