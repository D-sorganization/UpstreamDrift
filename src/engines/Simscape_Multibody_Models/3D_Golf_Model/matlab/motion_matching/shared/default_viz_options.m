function opts = default_viz_options()
%DEFAULT_VIZ_OPTIONS  Canonical visualization options struct.
%
%   opts = DEFAULT_VIZ_OPTIONS() returns the default options struct used
%   by every visualization entry point in the motion_matching shared
%   package. The fields and types here are the documented contract — see
%   VISUALIZATION_SPEC.md for usage and rationale.
%
%   Fields:
%     .color_measured     (string) hex colour for measured trace (#1f77b4)
%     .color_simulated    (string) hex colour for simulated trace (#d62728)
%     .color_error        (string) hex colour for error vector     (#7f7f7f)
%     .dpi                (double) export DPI for exportgraphics    (200)
%     .fontsize_axes      (double) axes font size                   (11)
%     .fontsize_title     (double) title font size                  (13)
%     .video_fps          (double) target VideoWriter frame rate    (30)
%     .video_format       (string) VideoWriter profile              ("MPEG-4")
%     .show_impact_marker (logical) draw impact-frame inset         (true)
%     .tight_inset        (logical) apply tightInset on save        (true)
%     .visible            (string) figure 'Visible' property        ("on")
%     .save_path          (string) optional file path for animate   ("")
%     .live_refresh_hz    (double) interactive redraw cap (Hz)      (5)
%     .trace_alpha        (double) clubhead path trace alpha        (0.3)
%
%   GitHub issue: #3989.

    opts = struct();
    opts.color_measured     = "#1f77b4";
    opts.color_simulated    = "#d62728";
    opts.color_error        = "#7f7f7f";
    opts.dpi                = 200;
    opts.fontsize_axes      = 11;
    opts.fontsize_title     = 13;
    opts.video_fps          = 30;
    opts.video_format       = "MPEG-4";
    opts.show_impact_marker = true;
    opts.tight_inset        = true;
    opts.visible            = "on";
    opts.save_path          = "";
    opts.live_refresh_hz    = 5;
    opts.trace_alpha        = 0.3;
end
