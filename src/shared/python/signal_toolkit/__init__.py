"""Signal Processing Toolkit - Comprehensive Signal Analysis Library.

A production-ready signal processing library for generating, fitting,
filtering, and analyzing signals. Designed for use in control systems,
simulation, robotics, and data analysis applications.

This package is part of the shared Tools repository and can be used
across multiple projects including UpstreamDrift and Gasification_Model.

Features:
    - Signal Generation: 13 signal types (sine, cosine, chirp, etc.)
    - Function Fitting: Sinusoid, exponential, linear, polynomial, custom
    - Digital Filters: Butterworth, Chebyshev, Bessel, adaptive (LMS/RLS)
    - Calculus: Differentiation, integration, tangent lines, curvature
    - Series: Taylor/Maclaurin series expansions with convergence analysis
    - Noise Generation: White, pink, brown, blue, violet, impulse
    - Limits: Saturation, rate limiting, deadband, hysteresis, backlash
    - I/O: CSV, JSON, NPZ, MAT, numpy array support
    - Visualization: PyQt6 widget for interactive signal analysis

Usage:
    from signal_toolkit import Signal, SignalGenerator, FunctionFitter

    # Create a signal
    import numpy as np
    t = np.linspace(0, 10, 1000)
    signal = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=2.0)

    # Fit a function
    fitter = FunctionFitter()
    result = fitter.fit_sinusoid(signal)
    logger.info(f"R-squared: {result.r_squared:.4f}")

    # Apply a filter
    from signal_toolkit import create_butterworth_filter, apply_filter
    filter_spec = create_butterworth_filter('lowpass', cutoff=5, fs=100, order=4)
    filtered = apply_filter(signal, filter_spec)

Dependencies:
    Required: numpy, scipy
    Optional: matplotlib (visualization), PyQt6 (GUI widget)

Version: 2.1.0
"""

from __future__ import annotations

import logging

from .calculus import (
    DifferentiationMethod,
    Differentiator,
    IntegralResult,
    IntegrationMethod,
    Integrator,
    TangentLine,
    compute_arc_length,
    compute_curvature,
    compute_derivative,
    compute_integral,
    compute_tangent_line,
    find_extrema,
    find_inflection_points,
)
from .core import Signal, SignalGenerator
from .filters import (
    AdaptiveFilter,
    FilterDesign,
    FilterDesigner,
    FilterSpec,
    FilterType,
    apply_bilateral_filter,
    apply_exponential_smoothing,
    apply_filter,
    apply_gaussian_smoothing,
    apply_median_filter,
    apply_moving_average,
    apply_savgol,
    create_butterworth_filter,
    create_chebyshev_filter,
    create_moving_average_filter,
    create_savgol_filter,
)
from .fitting import (
    CosineFitter,
    CustomFunctionFitter,
    ExponentialFitter,
    FitResult,
    FunctionFitter,
    LinearFitter,
    PolynomialFitter,
    SinusoidFitter,
)
from .io import (
    BatchProcessor,
    SignalExporter,
    SignalImporter,
    SignalLoader,
    export_to_csv,
    import_from_csv,
)
from .limits import (
    SaturationMode,
    apply_backlash,
    apply_deadband,
    apply_hysteresis,
    apply_rate_limiter,
    apply_saturation,
    create_saturation_function,
    visualize_saturation_curves,
)
from .noise import (
    DisturbanceSimulator,
    NoiseGenerator,
    NoiseType,
    add_noise_to_signal,
    generate_disturbance_profile,
)
from .series import (
    SeriesExpansion,
    SeriesResult,
    arctan_series,
    cos_series,
    cosh_series,
    exp_series,
    geometric_series,
    ln_series,
    sin_series,
    sinh_series,
)

logger = logging.getLogger(__name__)

# Optional: Polynomial generator (PyQt6 required)
try:
    from .polynomial_generator import PolynomialGeneratorWidget

    HAS_POLYNOMIAL_GENERATOR = True
except ImportError:
    PolynomialGeneratorWidget = None  # type: ignore[misc, assignment]
    HAS_POLYNOMIAL_GENERATOR = False
    logger.debug("PolynomialGeneratorWidget not available (requires PyQt6)")

# Optional: Interactive widget (PyQt6 + matplotlib required)
try:
    from .widget import SignalToolkitWidget

    HAS_WIDGET = True
except ImportError:
    SignalToolkitWidget = None  # type: ignore[misc, assignment]
    HAS_WIDGET = False
    logger.debug("SignalToolkitWidget not available (requires PyQt6 + matplotlib)")


__all__ = [
    # Core
    "Signal",
    "SignalGenerator",
    # Fitting
    "FitResult",
    "FunctionFitter",
    "SinusoidFitter",
    "CosineFitter",
    "ExponentialFitter",
    "LinearFitter",
    "PolynomialFitter",
    "CustomFunctionFitter",
    # Filters
    "FilterType",
    "FilterDesign",
    "FilterSpec",
    "FilterDesigner",
    "AdaptiveFilter",
    "apply_filter",
    "apply_moving_average",
    "apply_savgol",
    "apply_median_filter",
    "apply_exponential_smoothing",
    "apply_gaussian_smoothing",
    "apply_bilateral_filter",
    "create_butterworth_filter",
    "create_chebyshev_filter",
    "create_moving_average_filter",
    "create_savgol_filter",
    # Calculus
    "DifferentiationMethod",
    "IntegrationMethod",
    "TangentLine",
    "IntegralResult",
    "Differentiator",
    "Integrator",
    "compute_derivative",
    "compute_integral",
    "compute_tangent_line",
    "compute_curvature",
    "compute_arc_length",
    "find_extrema",
    "find_inflection_points",
    # Limits
    "SaturationMode",
    "apply_saturation",
    "apply_rate_limiter",
    "apply_deadband",
    "apply_hysteresis",
    "apply_backlash",
    "create_saturation_function",
    "visualize_saturation_curves",
    # Noise
    "NoiseType",
    "NoiseGenerator",
    "DisturbanceSimulator",
    "add_noise_to_signal",
    "generate_disturbance_profile",
    # I/O
    "SignalImporter",
    "SignalExporter",
    "SignalLoader",
    "BatchProcessor",
    "import_from_csv",
    "export_to_csv",
    # Series (Taylor/Maclaurin)
    "SeriesExpansion",
    "SeriesResult",
    "exp_series",
    "sin_series",
    "cos_series",
    "ln_series",
    "geometric_series",
    "arctan_series",
    "sinh_series",
    "cosh_series",
    # Optional (GUI)
    "PolynomialGeneratorWidget",
    "SignalToolkitWidget",
    "HAS_POLYNOMIAL_GENERATOR",
    "HAS_WIDGET",
]

__version__ = "2.1.0"
