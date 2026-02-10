"""Backward compatibility shim - module moved to gui_pkg.plotting_utils."""

import sys as _sys

from .gui_pkg import plotting_utils as _real_module  # noqa: E402
from .gui_pkg.plotting_utils import (  # noqa: F401
    DEFAULT_FIGSIZE,
    LARGE_FIGSIZE,
    SQUARE_FIGSIZE,
    WIDE_FIGSIZE,
    add_horizontal_line,
    add_vertical_line,
    annotate_point,
    close_all_figures,
    create_comparison_plot,
    create_error_plot,
    create_figure,
    create_subplot_grid,
    format_axis,
    logger,
    plot_multiple_time_series,
    plot_time_series,
    save_figure,
    setup_plot_style,
)

_sys.modules[__name__] = _real_module
