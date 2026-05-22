# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""Signal filtering utilities.

This module provides a comprehensive set of digital filters for signal
processing including IIR and FIR filters, smoothing, and specialized filters.
"""

from __future__ import annotations  # noqa: E402, F404

from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from enum import Enum  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal as scipy_signal  # noqa: E402
from scipy.signal import (  # noqa: E402
    bessel as _scipy_bessel,
)
from scipy.signal import (  # noqa: E402
    butter,
    cheby1,
    cheby2,
    ellip,
    filtfilt,
    lfilter,
    medfilt,
    savgol_filter,
)

from .adaptive_filter import AdaptiveFilter  # noqa: E402,F401
from .core import Signal  # noqa: E402


class FilterType(Enum):
    """Types of frequency-domain filters."""

    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    NOTCH = "notch"


class FilterDesign(Enum):
    """Filter design methods."""

    BUTTERWORTH = "butterworth"  # Maximally flat passband
    CHEBYSHEV1 = "chebyshev1"  # Ripple in passband
    CHEBYSHEV2 = "chebyshev2"  # Ripple in stopband
    ELLIPTIC = "elliptic"  # Ripple in both (sharpest cutoff)
    BESSEL = "bessel"  # Maximally flat group delay


@dataclass
class FilterSpec:
    """Specification for a digital filter.

    Attributes:
        b: Numerator (FIR) coefficients.
        a: Denominator (IIR) coefficients.
        filter_type: Type of filter (lowpass, highpass, etc.).
        design: Filter design method.
        order: Filter order.
        cutoff: Cutoff frequency/frequencies.
        fs: Sampling frequency.
    """

    b: np.ndarray
    a: np.ndarray
    filter_type: FilterType
    design: FilterDesign
    order: int
    cutoff: float | tuple[float, float]
    fs: float

    def get_frequency_response(
        self,
        num_points: int = 512,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the frequency response of the filter.

        Args:
            num_points: Number of frequency points.

        Returns:
            Tuple of (frequencies, magnitude, phase).
        """
        w, h = scipy_signal.freqz(self.b, self.a, worN=num_points, fs=self.fs)
        magnitude = np.abs(h)
        phase = np.angle(h)
        return w, magnitude, phase

    def get_impulse_response(
        self,
        num_samples: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the impulse response of the filter.

        Args:
            num_samples: Number of samples.

        Returns:
            Tuple of (time, impulse_response).
        """
        impulse = np.zeros(num_samples)
        impulse[0] = 1.0

        response = lfilter(self.b, self.a, impulse)
        t = np.arange(num_samples) / self.fs

        return t, response


def _normalize_cutoff(
    filter_type: FilterType,
    cutoff: float | tuple[float, float],
    fs: float,
) -> tuple[float | tuple[float, float], str]:
    """Normalize cutoff to Nyquist-relative value and resolve btype.

    Preconditions:
        - *fs* must be > 0.
        - Band filters require a (low, high) *cutoff* tuple.

    Returns:
        (wn, btype) ready for scipy filter design functions.

    Raises:
        ValueError: If *fs* is not positive or *cutoff* is out of range.
    """
    if fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}")
    nyquist = fs / 2
    btype = filter_type.value

    if filter_type in (FilterType.BANDPASS, FilterType.BANDSTOP, FilterType.NOTCH):
        if not isinstance(cutoff, tuple):
            msg = "Bandpass/bandstop/notch filters require (low, high) cutoff tuple"
            raise ValueError(msg)
        wn: float | tuple[float, float] = (
            cutoff[0] / nyquist,
            cutoff[1] / nyquist,
        )
        if filter_type == FilterType.NOTCH:
            btype = "bandstop"
    else:
        wn = (
            cutoff / nyquist if isinstance(cutoff, int | float) else cutoff[0] / nyquist
        )

    return wn, btype


class FilterDesigner:
    """Factory class for creating various digital filters."""

    @staticmethod
    def butterworth(
        filter_type: FilterType,
        cutoff: float | tuple[float, float],
        fs: float,
        order: int = 4,
    ) -> FilterSpec:
        """Design a Butterworth filter.

        Args:
            filter_type: Type of filter.
            cutoff: Cutoff frequency (Hz) or (low, high) for bandpass/bandstop.
            fs: Sampling frequency in Hz.
            order: Filter order.

        Returns:
            FilterSpec with filter coefficients.
        """
        if order < 1:
            raise ValueError(f"Filter order must be >= 1, got {order}")
        wn, btype = _normalize_cutoff(filter_type, cutoff, fs)
        b, a = butter(order, wn, btype=btype)
        return FilterSpec(
            b=b,
            a=a,
            filter_type=filter_type,
            design=FilterDesign.BUTTERWORTH,
            order=order,
            cutoff=cutoff,
            fs=fs,
        )

    @staticmethod
    def chebyshev1(
        filter_type: FilterType,
        cutoff: float | tuple[float, float],
        fs: float,
        order: int = 4,
        ripple_db: float = 1.0,
    ) -> FilterSpec:
        """Design a Chebyshev Type I filter (ripple in passband).

        Args:
            filter_type: Type of filter.
            cutoff: Cutoff frequency (Hz).
            fs: Sampling frequency in Hz.
            order: Filter order.
            ripple_db: Maximum ripple in passband (dB).

        Returns:
            FilterSpec with filter coefficients.
        """
        if order < 1:
            raise ValueError(f"Filter order must be >= 1, got {order}")
        if ripple_db <= 0:
            raise ValueError(f"ripple_db must be > 0, got {ripple_db}")
        wn, btype = _normalize_cutoff(filter_type, cutoff, fs)
        b, a = cheby1(order, ripple_db, wn, btype=btype)
        return FilterSpec(
            b=b,
            a=a,
            filter_type=filter_type,
            design=FilterDesign.CHEBYSHEV1,
            order=order,
            cutoff=cutoff,
            fs=fs,
        )

    @staticmethod
    def chebyshev2(
        filter_type: FilterType,
        cutoff: float | tuple[float, float],
        fs: float,
        order: int = 4,
        attenuation_db: float = 40.0,
    ) -> FilterSpec:
        """Design a Chebyshev Type II filter (ripple in stopband).

        Args:
            filter_type: Type of filter.
            cutoff: Cutoff frequency (Hz).
            fs: Sampling frequency in Hz.
            order: Filter order.
            attenuation_db: Minimum attenuation in stopband (dB).

        Returns:
            FilterSpec with filter coefficients.
        """
        if order < 1:
            raise ValueError(f"Filter order must be >= 1, got {order}")
        if attenuation_db <= 0:
            raise ValueError(f"attenuation_db must be > 0, got {attenuation_db}")
        wn, btype = _normalize_cutoff(filter_type, cutoff, fs)
        b, a = cheby2(order, attenuation_db, wn, btype=btype)
        return FilterSpec(
            b=b,
            a=a,
            filter_type=filter_type,
            design=FilterDesign.CHEBYSHEV2,
            order=order,
            cutoff=cutoff,
            fs=fs,
        )

    @staticmethod
    def elliptic(
        filter_type: FilterType,
        cutoff: float | tuple[float, float],
        fs: float,
        order: int = 4,
        ripple_db: float = 1.0,
        attenuation_db: float = 40.0,
    ) -> FilterSpec:
        """Design an elliptic (Cauer) filter.

        Args:
            filter_type: Type of filter.
            cutoff: Cutoff frequency (Hz).
            fs: Sampling frequency in Hz.
            order: Filter order.
            ripple_db: Maximum ripple in passband (dB).
            attenuation_db: Minimum attenuation in stopband (dB).

        Returns:
            FilterSpec with filter coefficients.
        """
        if order < 1:
            raise ValueError(f"Filter order must be >= 1, got {order}")
        if ripple_db <= 0:
            raise ValueError(f"ripple_db must be > 0, got {ripple_db}")
        if attenuation_db <= 0:
            raise ValueError(f"attenuation_db must be > 0, got {attenuation_db}")
        wn, btype = _normalize_cutoff(filter_type, cutoff, fs)
        b, a = ellip(order, ripple_db, attenuation_db, wn, btype=btype)
        return FilterSpec(
            b=b,
            a=a,
            filter_type=filter_type,
            design=FilterDesign.ELLIPTIC,
            order=order,
            cutoff=cutoff,
            fs=fs,
        )

    @staticmethod
    def bessel(
        filter_type: FilterType,
        cutoff: float | tuple[float, float],
        fs: float,
        order: int = 4,
    ) -> FilterSpec:
        """Design a Bessel filter (maximally flat group delay).

        Args:
            filter_type: Type of filter.
            cutoff: Cutoff frequency (Hz).
            fs: Sampling frequency in Hz.
            order: Filter order.

        Returns:
            FilterSpec with filter coefficients.
        """
        if order < 1:
            raise ValueError(f"Filter order must be >= 1, got {order}")
        wn, btype = _normalize_cutoff(filter_type, cutoff, fs)
        b, a = _scipy_bessel(order, wn, btype=btype, norm="phase")
        return FilterSpec(
            b=b,
            a=a,
            filter_type=filter_type,
            design=FilterDesign.BESSEL,
            order=order,
            cutoff=cutoff,
            fs=fs,
        )


def apply_filter(
    signal: Signal,
    filter_spec: FilterSpec,
    zero_phase: bool = True,
) -> Signal:
    """Apply a filter to a signal.

    Args:
        signal: Input signal.
        filter_spec: Filter specification.
        zero_phase: If True, use zero-phase filtering (filtfilt).

    Returns:
        Filtered signal.
    """
    if zero_phase:
        # Zero-phase filtering (no phase distortion)
        filtered_values = filtfilt(filter_spec.b, filter_spec.a, signal.values)
    else:
        # Causal filtering (introduces phase shift)
        filtered_values = lfilter(filter_spec.b, filter_spec.a, signal.values)

    # Postcondition: filter output should be finite
    if not np.all(np.isfinite(filtered_values)):
        import logging

        logging.getLogger(__name__).warning(
            "Filter output contains non-finite values; clamping to input range."
        )
        filtered_values = np.nan_to_num(
            filtered_values,
            nan=0.0,
            posinf=np.max(signal.values),
            neginf=np.min(signal.values),
        )

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_filtered",
        units=signal.units,
        metadata={
            **signal.metadata,
            "filter_type": filter_spec.filter_type.value,
            "filter_design": filter_spec.design.value,
            "cutoff": filter_spec.cutoff,
        },
    )


# Convenience functions for common filter types


def create_butterworth_filter(
    filter_type: str,
    cutoff: float | tuple[float, float],
    fs: float,
    order: int = 4,
) -> FilterSpec:
    """Create a Butterworth filter (convenience wrapper).

    Args:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'notch'.
        cutoff: Cutoff frequency or (low, high) tuple.
        fs: Sampling frequency.
        order: Filter order.

    Returns:
        FilterSpec.
    """
    ft = FilterType(filter_type)
    return FilterDesigner.butterworth(ft, cutoff, fs, order)


def create_chebyshev_filter(
    filter_type: str,
    cutoff: float | tuple[float, float],
    fs: float,
    order: int = 4,
    ripple_db: float = 1.0,
) -> FilterSpec:
    """Create a Chebyshev Type I filter (convenience wrapper).

    Args:
        filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop', 'notch'.
        cutoff: Cutoff frequency or (low, high) tuple.
        fs: Sampling frequency.
        order: Filter order.
        ripple_db: Passband ripple in dB.

    Returns:
        FilterSpec.
    """
    ft = FilterType(filter_type)
    return FilterDesigner.chebyshev1(ft, cutoff, fs, order, ripple_db)


def create_moving_average_filter(
    window_size: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a moving average filter function.

    Args:
        window_size: Size of the moving average window.

    Returns:
        Function that applies moving average to values.
    """
    kernel = np.ones(window_size) / window_size

    def apply(values: np.ndarray) -> np.ndarray:
        return np.convolve(values, kernel, mode="same")

    return apply


def create_savgol_filter(
    window_length: int = 11,
    polyorder: int = 3,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a Savitzky-Golay filter function.

    Args:
        window_length: Window length (must be odd).
        polyorder: Polynomial order.

    Returns:
        Function that applies Savitzky-Golay filter to values.
    """
    if window_length % 2 == 0:
        window_length += 1

    def apply(values: np.ndarray) -> np.ndarray:
        if len(values) < window_length:
            return values
        return savgol_filter(values, window_length, polyorder)

    return apply


def apply_moving_average(
    signal: Signal,
    window_size: int,
) -> Signal:
    """Apply moving average filter to a signal.

    Args:
        signal: Input signal.
        window_size: Size of moving average window.

    Returns:
        Filtered signal.
    """
    filter_func = create_moving_average_filter(window_size)
    filtered_values = filter_func(signal.values)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_ma{window_size}",
        units=signal.units,
        metadata={**signal.metadata, "filter": "moving_average", "window": window_size},
    )


def apply_savgol(
    signal: Signal,
    window_length: int = 11,
    polyorder: int = 3,
) -> Signal:
    """Apply Savitzky-Golay filter to a signal.

    Args:
        signal: Input signal.
        window_length: Window length (must be odd).
        polyorder: Polynomial order.

    Returns:
        Filtered signal.
    """
    if window_length % 2 == 0:
        window_length += 1

    if len(signal.values) < window_length:
        return signal.copy()

    filtered_values = savgol_filter(signal.values, window_length, polyorder)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_savgol",
        units=signal.units,
        metadata={
            **signal.metadata,
            "filter": "savgol",
            "window": window_length,
            "order": polyorder,
        },
    )


def apply_median_filter(
    signal: Signal,
    kernel_size: int = 5,
) -> Signal:
    """Apply median filter to a signal.

    Useful for removing impulse noise.

    Args:
        signal: Input signal.
        kernel_size: Size of median filter kernel (must be odd).

    Returns:
        Filtered signal.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    filtered_values = medfilt(signal.values, kernel_size)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_median",
        units=signal.units,
        metadata={**signal.metadata, "filter": "median", "kernel": kernel_size},
    )


def apply_exponential_smoothing(
    signal: Signal,
    alpha: float = 0.3,
) -> Signal:
    """Apply exponential smoothing to a signal.

    Args:
        signal: Input signal.
        alpha: Smoothing factor (0 < alpha <= 1). Higher = less smoothing.

    Returns:
        Smoothed signal.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in the range (0, 1]")
    values = signal.values
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]

    return Signal(
        time=signal.time,
        values=smoothed,
        name=f"{signal.name}_ema",
        units=signal.units,
        metadata={**signal.metadata, "filter": "exponential", "alpha": alpha},
    )


def apply_gaussian_smoothing(
    signal: Signal,
    sigma: float = 1.0,
) -> Signal:
    """Apply Gaussian smoothing to a signal.

    Args:
        signal: Input signal.
        sigma: Standard deviation of Gaussian kernel.

    Returns:
        Smoothed signal.
    """
    from scipy.ndimage import gaussian_filter1d

    filtered_values = gaussian_filter1d(signal.values, sigma)

    return Signal(
        time=signal.time,
        values=filtered_values,
        name=f"{signal.name}_gaussian",
        units=signal.units,
        metadata={**signal.metadata, "filter": "gaussian", "sigma": sigma},
    )


def apply_bilateral_filter(
    signal: Signal,
    window_size: int = 5,
    sigma_space: float = 1.0,
    sigma_intensity: float = 0.1,
) -> Signal:
    """Apply bilateral filter to a signal.

    Edge-preserving smoothing filter.

    Args:
        signal: Input signal.
        window_size: Size of the filter window.
        sigma_space: Spatial sigma (controls distance weighting).
        sigma_intensity: Intensity sigma (controls value similarity weighting).

    Returns:
        Filtered signal.
    """
    values = signal.values
    n = len(values)
    filtered = np.zeros(n)

    half_window = window_size // 2

    for i in range(n):
        # Get window
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # Spatial weights (distance from center)
        positions = np.arange(start, end)
        spatial_weights = np.exp(-((positions - i) ** 2) / (2 * sigma_space**2))

        # Intensity weights (value similarity)
        intensity_weights = np.exp(
            -((values[start:end] - values[i]) ** 2) / (2 * sigma_intensity**2)
        )

        # Combined weights
        weights = spatial_weights * intensity_weights
        weights /= np.sum(weights) + 1e-10

        filtered[i] = np.sum(weights * values[start:end])

    return Signal(
        time=signal.time,
        values=filtered,
        name=f"{signal.name}_bilateral",
        units=signal.units,
        metadata={
            **signal.metadata,
            "filter": "bilateral",
            "window": window_size,
            "sigma_space": sigma_space,
            "sigma_intensity": sigma_intensity,
        },
    )
