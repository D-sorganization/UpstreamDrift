"""Backward compatibility shim - module moved to signal_toolkit.signal_processing."""

import sys as _sys

from .signal_toolkit import signal_processing as _real_module  # noqa: E402
from .signal_toolkit.signal_processing import (  # noqa: F401
    KalmanFilter,
    compute_coherence,
    compute_cwt,
    compute_dtw_distance,
    compute_dtw_path,
    compute_jerk,
    compute_psd,
    compute_spectral_arc_length,
    compute_spectrogram,
    compute_time_shift,
    compute_xwt,
)

_sys.modules[__name__] = _real_module
