function opts = default_viz_options()
%DEFAULT_VIZ_OPTIONS  Default options struct for motion_matching viz entry points.
%
%   See VISUALIZATION_SPEC.md for the styling and behaviour these options
%   control.  Every viz function (plot_trajectory_overlay,
%   plot_error_timecourse, plot_fit_quality_card, ...) accepts a struct of
%   this shape as its third argument.
%
%   Fields:
%     .figure_visible       - 'on' | 'off'.  Use 'off' for headless CI.
%     .sample_rate_noise_m  - baseline 1-sigma position noise (metres).
%                             Used to draw the +/-1 sigma band on Panel 1
%                             of plot_error_timecourse.  Default: 5e-4 m.
%     .dpi                  - export DPI for exportgraphics PNG output.
%     .tight_inset          - whether to apply tightInset on save.
    opts = struct();
    opts.figure_visible      = 'off';
    opts.sample_rate_noise_m = 5e-4;
    opts.dpi                 = 200;
    opts.tight_inset         = true;
end
