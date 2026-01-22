"""Basic statistical analysis components."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from shared.python.analysis.dataclasses import PeakInfo, SummaryStatistics


class BasicStatsMixin:
    """Mixin for basic statistical operations and peak detection.

    Expects the following attributes to be available on the instance:
    - times: np.ndarray
    - club_head_speed: np.ndarray | None
    """

    # Type hints for anticipated attributes (Protocol pattern preferred but simple typing used here)
    times: np.ndarray
    club_head_speed: np.ndarray | None

    def compute_summary_stats(self, data: np.ndarray) -> SummaryStatistics:
        """Compute summary statistics for a 1D array.

        Args:
            data: 1D numpy array

        Returns:
            SummaryStatistics object
        """
        min_idx = np.argmin(data)
        max_idx = np.argmax(data)
        min_val = float(data[min_idx])
        max_val = float(data[max_idx])

        return SummaryStatistics(
            mean=float(np.mean(data)),
            median=float(np.median(data)),
            std=float(np.std(data)),
            min=min_val,
            max=max_val,
            range=max_val - min_val,
            min_time=float(self.times[min_idx]),
            max_time=float(self.times[max_idx]),
            rms=float(np.sqrt(np.mean(data**2))),
        )

    def find_peaks_in_data(
        self,
        data: np.ndarray,
        height: float | None = None,
        prominence: float | None = None,
        distance: int | None = None,
    ) -> list[PeakInfo]:
        """Find peaks in time series data.

        Args:
            data: 1D array
            height: Minimum peak height
            prominence: Minimum peak prominence
            distance: Minimum samples between peaks

        Returns:
            List of PeakInfo objects
        """
        peaks, properties = find_peaks(
            data,
            height=height,
            prominence=prominence,
            distance=distance,
        )

        peak_list = []
        for i, peak_idx in enumerate(peaks):
            peak_info = PeakInfo(
                value=float(data[peak_idx]),
                time=float(self.times[peak_idx]),
                index=int(peak_idx),
                prominence=(
                    float(properties["prominences"][i])
                    if "prominences" in properties
                    else None
                ),
                width=(
                    float(properties["widths"][i]) if "widths" in properties else None
                ),
            )
            peak_list.append(peak_info)

        return peak_list

    def find_club_head_speed_peak(self) -> PeakInfo | None:
        """Find peak club head speed.

        Returns:
            PeakInfo for maximum club head speed
        """
        if self.club_head_speed is None or len(self.club_head_speed) == 0:
            return None

        max_idx = np.argmax(self.club_head_speed)
        return PeakInfo(
            value=float(self.club_head_speed[max_idx]),
            time=float(self.times[max_idx]),
            index=int(max_idx),
        )
