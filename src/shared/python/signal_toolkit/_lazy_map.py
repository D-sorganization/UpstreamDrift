"""Lazy import dispatch table for ``signal_toolkit``.

Extracted from ``__init__.py`` (issue #1696) to keep the package entry-point
below 120 lines.  Each entry maps a public name to the (relative_module,
attribute_name) pair that provides it.
"""

from __future__ import annotations

# Values are (relative_module_path, attribute_name).
# Use relative paths (dot-prefixed) so they work with importlib.import_module
# when package=__name__ is supplied.
LAZY: dict[str, tuple[str, str]] = {
    # Core
    "Signal": (".core", "Signal"),
    "SignalGenerator": (".core", "SignalGenerator"),
    # Fitting
    "FitResult": (".fitting", "FitResult"),
    "FunctionFitter": (".fitting", "FunctionFitter"),
    "SinusoidFitter": (".fitting", "SinusoidFitter"),
    "CosineFitter": (".fitting", "CosineFitter"),
    "ExponentialFitter": (".fitting", "ExponentialFitter"),
    "LinearFitter": (".fitting", "LinearFitter"),
    "PolynomialFitter": (".fitting", "PolynomialFitter"),
    "CustomFunctionFitter": (".fitting", "CustomFunctionFitter"),
    # Filters
    "FilterType": (".filters", "FilterType"),
    "FilterDesign": (".filters", "FilterDesign"),
    "FilterSpec": (".filters", "FilterSpec"),
    "FilterDesigner": (".filters", "FilterDesigner"),
    "AdaptiveFilter": (".filters", "AdaptiveFilter"),
    "apply_filter": (".filters", "apply_filter"),
    "apply_moving_average": (".filters", "apply_moving_average"),
    "apply_savgol": (".filters", "apply_savgol"),
    "apply_median_filter": (".filters", "apply_median_filter"),
    "apply_exponential_smoothing": (".filters", "apply_exponential_smoothing"),
    "apply_gaussian_smoothing": (".filters", "apply_gaussian_smoothing"),
    "apply_bilateral_filter": (".filters", "apply_bilateral_filter"),
    "create_butterworth_filter": (".filters", "create_butterworth_filter"),
    "create_chebyshev_filter": (".filters", "create_chebyshev_filter"),
    "create_moving_average_filter": (".filters", "create_moving_average_filter"),
    "create_savgol_filter": (".filters", "create_savgol_filter"),
    # Calculus
    "DifferentiationMethod": (".calculus", "DifferentiationMethod"),
    "IntegrationMethod": (".calculus", "IntegrationMethod"),
    "TangentLine": (".calculus", "TangentLine"),
    "IntegralResult": (".calculus", "IntegralResult"),
    "Differentiator": (".calculus", "Differentiator"),
    "Integrator": (".calculus", "Integrator"),
    "compute_derivative": (".calculus", "compute_derivative"),
    "compute_integral": (".calculus", "compute_integral"),
    "compute_tangent_line": (".calculus", "compute_tangent_line"),
    "compute_curvature": (".calculus", "compute_curvature"),
    "compute_arc_length": (".calculus", "compute_arc_length"),
    "find_extrema": (".calculus", "find_extrema"),
    "find_inflection_points": (".calculus", "find_inflection_points"),
    # Limits
    "SaturationMode": (".limits", "SaturationMode"),
    "apply_saturation": (".limits", "apply_saturation"),
    "apply_rate_limiter": (".limits", "apply_rate_limiter"),
    "apply_deadband": (".limits", "apply_deadband"),
    "apply_hysteresis": (".limits", "apply_hysteresis"),
    "apply_backlash": (".limits", "apply_backlash"),
    "create_saturation_function": (".limits", "create_saturation_function"),
    "visualize_saturation_curves": (".limits", "visualize_saturation_curves"),
    # Noise
    "NoiseType": (".noise", "NoiseType"),
    "NoiseGenerator": (".noise", "NoiseGenerator"),
    "DisturbanceSimulator": (".noise", "DisturbanceSimulator"),
    "add_noise_to_signal": (".noise", "add_noise_to_signal"),
    "generate_disturbance_profile": (".noise", "generate_disturbance_profile"),
    # I/O
    "SignalImporter": (".io", "SignalImporter"),
    "SignalExporter": (".io", "SignalExporter"),
    "SignalLoader": (".io", "SignalLoader"),
    "BatchProcessor": (".io", "BatchProcessor"),
    "import_from_csv": (".io", "import_from_csv"),
    "export_to_csv": (".io", "export_to_csv"),
    # Series (Taylor/Maclaurin)
    "SeriesExpansion": (".series", "SeriesExpansion"),
    "SeriesResult": (".series", "SeriesResult"),
    "exp_series": (".series", "exp_series"),
    "sin_series": (".series", "sin_series"),
    "cos_series": (".series", "cos_series"),
    "ln_series": (".series", "ln_series"),
    "geometric_series": (".series", "geometric_series"),
    "arctan_series": (".series", "arctan_series"),
    "sinh_series": (".series", "sinh_series"),
    "cosh_series": (".series", "cosh_series"),
}
