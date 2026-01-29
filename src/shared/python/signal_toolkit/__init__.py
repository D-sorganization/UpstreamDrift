"""Signal Processing Toolkit.

A comprehensive, reusable signal processing library for generating,
fitting, filtering, and analyzing signals. Designed for use in control
systems, simulation, and data analysis.

Features:
- Function fitting (sin, cos, exponential, linear, polynomial, custom)
- Limit/saturation with smoothing (soft clipping, tanh, sigmoid)
- Differentiation and integration with visualization support
- Noise generation (white, pink, brown, Gaussian)
- Comprehensive filter library (Butterworth, Chebyshev, etc.)
- Signal import/export (CSV, numpy arrays)

Usage:
    from src.shared.python.signal_toolkit import Signal, FunctionFitter, Filters

    # Create a signal
    t = np.linspace(0, 10, 1000)
    signal = Signal(t, np.sin(2 * np.pi * t))

    # Apply a filter
    filtered = signal.apply_filter(Filters.butterworth_lowpass(cutoff=5, fs=100))

    # Fit a function
    fitter = FunctionFitter()
    params = fitter.fit_sinusoid(signal)
"""

from __future__ import annotations

from src.shared.python.signal_toolkit.calculus import (
    Differentiator,
    Integrator,
    compute_derivative,
    compute_integral,
    compute_tangent_line,
)
from src.shared.python.signal_toolkit.core import Signal, SignalGenerator
from src.shared.python.signal_toolkit.filters import (
    FilterDesigner,
    FilterType,
    apply_filter,
    create_butterworth_filter,
    create_chebyshev_filter,
    create_moving_average_filter,
    create_savgol_filter,
)
from src.shared.python.signal_toolkit.fitting import (
    CustomFunctionFitter,
    ExponentialFitter,
    FunctionFitter,
    LinearFitter,
    PolynomialFitter,
    SinusoidFitter,
)
from src.shared.python.signal_toolkit.io import (
    SignalExporter,
    SignalImporter,
    export_to_csv,
    import_from_csv,
)
from src.shared.python.signal_toolkit.limits import (
    SaturationMode,
    apply_deadband,
    apply_hysteresis,
    apply_rate_limiter,
    apply_saturation,
)
from src.shared.python.signal_toolkit.noise import (
    NoiseGenerator,
    NoiseType,
    add_noise_to_signal,
)

__all__ = [
    # Core
    "Signal",
    "SignalGenerator",
    # Fitting
    "FunctionFitter",
    "SinusoidFitter",
    "ExponentialFitter",
    "LinearFitter",
    "PolynomialFitter",
    "CustomFunctionFitter",
    # Limits
    "apply_saturation",
    "apply_rate_limiter",
    "apply_deadband",
    "apply_hysteresis",
    "SaturationMode",
    # Calculus
    "Differentiator",
    "Integrator",
    "compute_derivative",
    "compute_integral",
    "compute_tangent_line",
    # Noise
    "NoiseGenerator",
    "NoiseType",
    "add_noise_to_signal",
    # Filters
    "FilterDesigner",
    "FilterType",
    "apply_filter",
    "create_butterworth_filter",
    "create_chebyshev_filter",
    "create_moving_average_filter",
    "create_savgol_filter",
    # IO
    "SignalImporter",
    "SignalExporter",
    "import_from_csv",
    "export_to_csv",
]

__version__ = "1.0.0"
