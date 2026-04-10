"""Signal processing utilities for biomechanical data analysis.

This module provides common signal processing functions used across
different physics engines for vibration analysis, frequency domain
analysis, and signal quality assessment.

Performance optimizations:
- Numba JIT compilation for DTW and other tight loops
- LRU caching for wavelet generation in CWT
- Parallelization hooks for lag matrix computation
"""

from src.shared.python.signal_toolkit._dtw import (
    _dtw_core,
    _dtw_path_core,
    compute_dtw_distance,
    compute_dtw_path,
)
from src.shared.python.signal_toolkit._filters import KalmanFilter
from src.shared.python.signal_toolkit._spectral_analysis import (
    compute_coherence,
    compute_psd,
    compute_spectral_arc_length,
    compute_spectrogram,
)
from src.shared.python.signal_toolkit._time_domain import (
    compute_jerk,
    compute_time_shift,
)
from src.shared.python.signal_toolkit._wavelet import (
    _convolve_wavelet_at_scale,
    _get_cached_wavelet,
    _morlet2_impl,
    _prepare_cwt_fft,
    _validate_cwt_inputs,
    compute_cwt,
    compute_xwt,
)

__all__ = [
    "KalmanFilter",
    "_convolve_wavelet_at_scale",
    "_dtw_core",
    "_dtw_path_core",
    "_get_cached_wavelet",
    "_morlet2_impl",
    "_prepare_cwt_fft",
    "_validate_cwt_inputs",
    "compute_coherence",
    "compute_cwt",
    "compute_dtw_distance",
    "compute_dtw_path",
    "compute_jerk",
    "compute_psd",
    "compute_spectral_arc_length",
    "compute_spectrogram",
    "compute_time_shift",
    "compute_xwt",
]
