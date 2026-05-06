function opts = default_viz_options()
%DEFAULT_VIZ_OPTIONS  Canonical visualization options struct.
%
%   opts = DEFAULT_VIZ_OPTIONS() returns the default options struct used
%   by every visualization entry point in the motion_matching shared
%   package. The fields and types here are the documented contract — see
%   VISUALIZATION_SPEC.md for usage and rationale.
%
%   Styling fields (apply to all views):
%     .color_measured     (string) hex colour for measured trace      ("#1f77b4")
%     .color_simulated    (string) hex colour for simulated trace     ("#d62728")
%     .color_error        (string) hex colour for error vector        ("#7f7f7f")
%     .dpi                (double) export DPI for exportgraphics      (200)
%     .fontsize_axes      (double) axes font size                     (11)
%     .fontsize_title     (double) title font size                    (13)
%     .tight_inset        (logical) apply tightInset on save          (true)
%
%   Trajectory overlay (View 1) fields:
%     .visible            (string) figure 'Visible' property          ("on")
%     .figure_visible     (string) alias of .visible for headless CI  ("on")
%     .show_impact_marker (logical) draw impact-frame inset           (true)
%     .trace_alpha        (double) clubhead path trace alpha          (0.3)
%     .video_fps          (double) target VideoWriter frame rate      (30)
%     .video_format       (string) VideoWriter profile                ("MPEG-4")
%     .save_path          (string) optional file path for animate     ("")
%     .live_refresh_hz    (double) interactive redraw cap (Hz)        (5)
%
%   Error timecourse (View 2) fields:
%     .sample_rate_noise_m (double) baseline 1-sigma position noise   (5e-4)
%
%   GitHub issues: #3989 (View 1), #3990 (View 2). Combined here per the
%   shared/VISUALIZATION_SPEC.md contract that every viz function takes
%   the same options struct shape.

    opts = struct();
    % styling
    opts.color_measured      = "#1f77b4";
    opts.color_simulated     = "#d62728";
    opts.color_error         = "#7f7f7f";
    opts.dpi                 = 200;
    opts.fontsize_axes       = 11;
    opts.fontsize_title      = 13;
    opts.tight_inset         = true;
    % trajectory overlay (View 1)
    opts.visible             = "on";
    opts.figure_visible      = "on";   % alias for headless CI; kept for back-compat with test_plot_error_timecourse
    opts.show_impact_marker  = true;
    opts.trace_alpha         = 0.3;
    opts.video_fps           = 30;
    opts.video_format        = "MPEG-4";
    opts.save_path           = "";
    opts.live_refresh_hz     = 5;
    % error timecourse (View 2)
    opts.sample_rate_noise_m = 5e-4;
end
