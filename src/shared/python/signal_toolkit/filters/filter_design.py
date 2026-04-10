"""Filter design primitives: enums, FilterSpec dataclass, and FilterDesigner factory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import (
    bessel as _scipy_bessel,
)
from scipy.signal import (
    butter,
    cheby1,
    cheby2,
    ellip,
    lfilter,
)

from src.shared.python.core.contracts import require  # type: ignore[import-untyped]


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
        if not (num_points is not None):
            raise ValueError("num_points must be provided")
        if not (num_points is not None):
            raise ValueError("num_points must be provided")
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
        if not (num_samples is not None):
            raise ValueError("num_samples must be provided")
        if not (num_samples is not None):
            raise ValueError("num_samples must be provided")
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
            cutoff / nyquist
            if isinstance(cutoff, (int, float))
            else cutoff[0] / nyquist
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
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
        require(order > 0, f"Filter order must be positive, got {order}")
        require(fs > 0, f"Sampling frequency must be positive, got {fs}")
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
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
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
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
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
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
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
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
        if not (filter_type is not None):
            raise ValueError("filter_type must be provided")
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
